"""Budget-aware inference passes and answer aggregation."""

from __future__ import annotations

import ast
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from . import textproc, validators
from .prompts import REFINE_MSG, finalize_msg, followup_messages, solve_messages

_LIVE_GOLD: dict[str, list] | None = None  # lazily loaded from LIVE_ANSWERS (dev only)


def _em_norm(s: object) -> str:
    """Scorer-faithful normalize for live EM — MIRROR of dev/score_local.normalize.

    Deliberately NOT textproc.normalize (which also collapses internal whitespace and folds
    curly apostrophes, so it would read EM *higher* than the platform — a falsely-optimistic
    dev signal). This tracks the real scorer so live EM is an honest preview. KEEP IN SYNC
    with dev/score_local.normalize and solver/script.py _em_norm.
    """
    s = str(s).strip().strip("\"'‘’“”").removesuffix(".")  # noqa: RUF001 — curly-quote variants
    return s.strip().lower()


def _live_gold() -> dict[str, list]:
    """Load {id: [per-item alternatives]} from LIVE_ANSWERS once (empty if unset/missing, dev).

    Fail-closed: gold items are str()-coerced (to match the scorer and avoid normalize()
    crashing on numeric answers), and any malformed row/file is skipped rather than raised —
    qlog is optional observability and must never defeat the incremental-flush crash safety.
    """
    global _LIVE_GOLD  # noqa: PLW0603 — intentional one-shot module cache
    if _LIVE_GOLD is None:
        _LIVE_GOLD = {}
        path = os.environ.get("LIVE_ANSWERS", "")
        if path and Path(path).is_file():
            import csv  # noqa: PLC0415 — dev-only, lazy

            with Path(path).open(encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    try:
                        v = ast.literal_eval(r["answer"])
                        _LIVE_GOLD[str(r["id"])] = [
                            [str(a) for a in it] if isinstance(it, (list, tuple)) else [str(it)]
                            for it in v
                        ]
                    except Exception:  # noqa: BLE001, S112, PERF203 — dev gold; skip bad rows
                        continue
    return _LIVE_GOLD


def qlog(  # noqa: PLR0913 — live-signal knobs; a param object would just add indirection
    pid: str, kind: str, text: str, k: int, context: str, labels: list[str] | None = None
) -> None:
    """Per-question live signal printed the instant a generation finishes.

    So a failing model shows at problem 1, not after the whole run. Structural (marker +
    answers-delivered vs k) => safe in the sandbox; adds per-problem EM only when LIVE_ANSWERS
    gold is set (dev). Pure-Python — no sacrebleu, no test-content echo. Wrapped fail-closed:
    observability must never sink the run.
    """
    try:
        text = text or ""
        parsed = textproc.parse_answers(text, k, context, labels=labels)
        delivered = sum(1 for a in parsed if a.strip())
        marker = int(textproc.has_final_marker(text))
        line = f"[q] {pid} {kind} marker={marker} ans={delivered}/{k}"
        gold = _live_gold().get(str(pid))
        if gold is not None:
            hits = sum(
                any(_em_norm(p) == _em_norm(x) for x in alts)
                for p, alts in zip(parsed, gold, strict=False)
            )
            line += f" EM={hits}/{len(gold)}"
        print(line, flush=True)
    except Exception as e:  # noqa: BLE001 — a logging failure must never sink the run
        print(f"[q] {pid} {kind} log-failed: {type(e).__name__}", flush=True)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .backends import Backend
    from .budget import Budget
    from .config import Config
    from .prompts import Messages
    from .transcript import Transcript

FINALIZE_MAX_NEW = 400


def _ceil_div(n: int, d: int) -> int:
    """Ceiling division with a floor of one batch."""
    return max(1, -(-n // max(1, d)))


@dataclass
class ProblemState:
    """Everything the pipeline knows about one problem."""

    row: dict[str, str]
    k: int
    samples: list[dict[str, str]] = field(default_factory=list)  # {kind, text[, finalized]}
    verdicts: list[str] = field(default_factory=list)
    retrieved: str = ""

    @classmethod
    def from_row(cls, row: dict[str, str]) -> ProblemState:
        """Build the initial state for one test.csv row."""
        k = textproc.n_items(row.get("query", ""), row.get("context", ""), row.get("task_type", ""))
        return cls(row=row, k=k)

    @property
    def id(self) -> str:
        """The problem id (submission key)."""
        return self.row["id"]

    @property
    def context(self) -> str:
        """The puzzle data the answers must be derived from."""
        return self.row.get("context", "")

    @property
    def query(self) -> str:
        """The asked items."""
        return self.row.get("query", "")

    @property
    def labels(self) -> list[str]:
        """Slot-addressable item labels from the query ([] when order-only)."""
        return textproc.query_labels(self.query, self.k)

    @property
    def multi_slot(self) -> bool:
        """Whether the query demands multiple forms per item (enables " / " expansion)."""
        return bool(textproc.MULTI_SLOT_RE.search(self.query))

    def solve_count(self) -> int:
        """Return the number of plain solve samples (refine samples don't shift vote indices)."""
        return sum(1 for s in self.samples if s["kind"] == "solve")

    def refine_count(self) -> int:
        """Return the number of refine samples (drives the round cap; resume-safe by counting)."""
        return sum(1 for s in self.samples if s["kind"] == "refine")

    def base_messages(self, *, fewshot: bool) -> Messages:
        """Build the initial solve conversation for this problem."""
        return solve_messages(
            self.context,
            self.query,
            fewshot=fewshot,
            task_type=self.row.get("task_type", ""),
            extra=self.retrieved,
        )


@dataclass
class Context:
    """Shared services handed to every pass."""

    cfg: Config
    backend: Backend
    budget: Budget
    transcript: Transcript
    flush: Callable[[], None]


class Pass(Protocol):
    """One pipeline step, run over all problems."""

    name: str

    def run(self, states: list[ProblemState], ctx: Context) -> None:
        """Do this step's work; respect ctx.budget; log via ctx.transcript."""
        ...


def aggregate(state: ProblemState) -> tuple[list[str], str]:
    """Parse samples, aggregate answers, and fill missing items. Refinements precede solve samples so they win first-occurrence ties; a majority can override them."""
    ordered = sorted(state.samples, key=lambda s: s["kind"] != "refine")
    texts = [s["text"] for s in ordered]
    parsed = [
        textproc.parse_answers(
            t, state.k, state.context, labels=state.labels, multi_slot=state.multi_slot
        )
        for t in texts
    ]
    final = textproc.vote(parsed, state.k)
    fb = textproc.fallback_items(state.context, state.k)
    final = [v if v.strip() else fb[i] for i, v in enumerate(final)]
    expl = next((r for t in texts if (r := textproc.parse_rules(t))), "")
    return final, expl or "Rules deduced from the given examples (see prediction)."


def _generate_batched(  # noqa: PLR0913 — keyword-only knobs; a spec object would just add indirection
    states: Sequence[ProblemState],
    conversations: Sequence[Messages],
    ctx: Context,
    *,
    greedy: bool,
    max_new: int,
    kind: str,
    pass_name: str,
) -> None:
    """Shared batching loop: budget-checked chunks, transcript logging."""
    batch_size = ctx.cfg.batch
    todo = list(zip(states, conversations, strict=True))
    i = 0
    while i < len(todo):
        if ctx.budget.exhausted():
            print(f"[solver] deadline guard tripped in {pass_name}", flush=True)
            return
        chunk = todo[i : i + batch_size]
        batches_left = max(1, (len(todo) - i + batch_size - 1) // batch_size)
        cap = (
            ctx.budget.max_new_adaptive(max_new)
            if ctx.budget._total_rows
            else ctx.budget.max_new_for(batches_left, max_new)
        )
        t_batch = time.time()
        try:
            texts = ctx.backend.chat([c for _, c in chunk], greedy=greedy, max_new=cap)
        except Exception as e:  # noqa: BLE001 — one bad batch (OOM, template) must not sink the run
            print(f"[solver] batch failed in {pass_name}: {e!r} — skipping chunk", flush=True)
            i += len(chunk)
            continue
        ctx.budget.record_batch(time.time() - t_batch)
        for (state, _), text in zip(chunk, texts, strict=True):
            state.samples.append({"kind": kind, "text": text})
            qlog(
                state.id, kind, text, state.k, state.context, state.labels
            )  # per-question live signal
            ctx.transcript.log(
                {"event": "sample", "id": state.id, "kind": kind, "pass": pass_name, "text": text}
            )
            ctx.budget.finish_row()
        i += len(chunk)
        ctx.flush()  # bank partial credit per batch — the kill can land mid-pass


class SolvePass:
    """Generate one solve sample per problem (greedy first, sampled votes later)."""

    delivery = False  # generates new analysis — must stop at the soft deadline

    def __init__(self, sample_index: int, *, greedy: bool, optional: bool = False) -> None:
        self.sample_index = sample_index
        self.greedy = greedy
        self.optional = optional
        self.name = f"solve[{sample_index}]"

    def run(self, states: list[ProblemState], ctx: Context) -> None:
        """Skip problems that already have this sample (resume); batch the rest."""
        todo = [s for s in states if s.solve_count() <= self.sample_index]
        if not todo:
            return
        if self.optional and not ctx.budget.affords(_ceil_div(len(todo), ctx.cfg.batch)):
            print(f"[solver] skipping optional {self.name} (behind schedule)", flush=True)
            return
        convs = [s.base_messages(fewshot=ctx.cfg.fewshot) for s in todo]
        _generate_batched(
            todo,
            convs,
            ctx,
            greedy=self.greedy,
            max_new=ctx.cfg.max_new,
            kind="solve",
            pass_name=self.name,
        )


class FinalizePass:
    """Rescue truncated analyses: ask for the two output blocks, greedy and short.

    v1 parity: EVERY generated sample lacking a FINAL ANSWERS marker gets one
    finalize follow-up (not just the latest). The continuation is merged into
    its sample in place — it extends that vote, it never becomes a new one —
    and the transcript records the target slot so resume replays the merge.
    """

    name = "finalize"
    # Delivery pass: allowed to run past the soft deadline into the hard-kill
    # headroom (the Pipeline honours this), because emitting answers is mandatory
    # and cheap. Without it a solve-pass overrun leaves marker-less rows blank.
    delivery = True

    def _build_jobs(
        self, states: list[ProblemState], ctx: Context
    ) -> tuple[list[tuple[ProblemState, int]], list[Messages]]:
        """Return (jobs, conversations) for every sample still lacking its answers."""
        jobs: list[tuple[ProblemState, int]] = []
        convs: list[Messages] = []
        if not any(not smp.get("finalized") for s in states for smp in s.samples):
            return jobs, convs  # every sample had its one attempt: skip the scan entirely
        for s in states:
            for idx, smp in enumerate(s.samples):
                if smp.get("finalized"):  # one attempt per sample — the loop guard
                    continue
                txt = smp["text"] or ""
                nonblank = sum(1 for a in textproc.parse_answers(txt, s.k) if a.strip())
                if textproc.has_final_marker(txt) and nonblank >= s.k:
                    continue
                prior = textproc.strip_thinking(smp["text"])
                jobs.append((s, idx))
                convs.append(
                    followup_messages(
                        s.base_messages(fewshot=ctx.cfg.fewshot), prior, finalize_msg(s.k)
                    )
                )
        return jobs, convs

    def run(self, states: list[ProblemState], ctx: Context) -> None:
        """Follow up once on every sample that never reached FINAL ANSWERS."""
        if ctx.budget.delivery_exhausted():
            print(
                f"[solver] delivery deadline already passed — {self.name} does nothing", flush=True
            )
            return
        jobs, convs = self._build_jobs(states, ctx)
        batch_size = ctx.cfg.batch
        i = 0
        while i < len(jobs):
            if ctx.budget.delivery_exhausted():
                print(f"[solver] delivery deadline guard tripped in {self.name}", flush=True)
                return
            chunk = range(i, min(i + batch_size, len(jobs)))
            t_batch = time.time()
            try:
                texts = ctx.backend.chat(
                    [convs[j] for j in chunk], greedy=True, max_new=FINALIZE_MAX_NEW
                )
            except Exception as e:  # noqa: BLE001 — one bad batch must not sink the run
                print(f"[solver] batch failed in {self.name}: {e!r} — skipping chunk", flush=True)
                i += len(chunk)
                continue
            ctx.budget.record_batch(time.time() - t_batch)
            for j, text in zip(chunk, texts, strict=True):
                state, idx = jobs[j]
                state.samples[idx]["text"] += "\n" + text
                if text.strip():
                    state.samples[idx]["finalized"] = "1"
                # live: show whether finalize rescued delivery for this question
                qlog(
                    state.id,
                    "finalize",
                    state.samples[idx]["text"],
                    state.k,
                    state.context,
                    state.labels,
                )
                ctx.transcript.log(
                    {
                        "event": "sample",
                        "id": state.id,
                        "kind": "finalize",
                        "slot": idx,
                        "pass": self.name,
                        "text": text,
                    }
                )
            i += len(chunk)
            ctx.flush()  # bank partial credit per batch


class ValidatePass:
    """Run the deterministic checks; verdicts drive (and gate) the refine pass."""

    delivery = False  # cheap, but produces no answers on its own

    name = "validate"

    def run(self, states: list[ProblemState], ctx: Context) -> None:
        """Store verdicts on each state and log them."""
        for s in states:
            s.verdicts = validators.check(s)
            if s.verdicts:
                ctx.transcript.log({"event": "verdicts", "id": s.id, "verdicts": s.verdicts})


class RefinePass:
    """Refine previous answers using validator feedback."""

    delivery = False  # generates new analysis — must stop at the soft deadline

    def __init__(self, max_rounds: int) -> None:
        self.max_rounds = max_rounds
        self.name = "refine"

    def run(self, states: list[ProblemState], ctx: Context) -> None:
        """One revision per flagged problem, budget and round-cap permitting."""
        todo, convs = [], []
        for s in states:
            if not s.verdicts or s.refine_count() >= self.max_rounds or not s.samples:
                continue
            prior = textproc.strip_thinking(s.samples[-1]["text"])
            msg = REFINE_MSG.format(verdicts="\n".join(f"- {v}" for v in s.verdicts), k=s.k)
            todo.append(s)
            convs.append(followup_messages(s.base_messages(fewshot=ctx.cfg.fewshot), prior, msg))
        if not todo:
            return
        if not ctx.budget.affords(_ceil_div(len(todo), ctx.cfg.batch)):
            print("[solver] skipping refine (behind schedule)", flush=True)
            return
        _generate_batched(
            todo,
            convs,
            ctx,
            greedy=True,
            max_new=ctx.cfg.max_new,
            kind="refine",
            pass_name=self.name,
        )


class Pipeline:
    """Runs passes in order; re-aggregates and flushes after every pass."""

    def __init__(self, passes: Sequence[Pass]) -> None:
        self.passes = list(passes)
        undeclared = [
            p.name for p in self.passes if not isinstance(getattr(p, "delivery", None), bool)
        ]
        if undeclared:
            msg = (
                f"passes without a `delivery` declaration: {undeclared}. Set delivery=True "
                "for a pass that emits answers (it may run in the hard-kill headroom), "
                "False for one that generates new analysis."
            )
            raise RuntimeError(msg)

    def run(self, states: list[ProblemState], ctx: Context) -> None:
        """Execute the passes, honouring the soft deadline and the later delivery one.

        Past the soft deadline the scheduler stops starting work that GENERATES
        analysis but keeps running DELIVERY passes, so rows that reasoned without
        emitting FINAL ANSWERS still get their answer lines.

        Non-delivery passes are SKIPPED individually rather than breaking the loop.
        Breaking is safe only while exactly one finalize follows the first solve; a
        voting pipeline is [solve, finalize, vote, finalize] and a break at `vote`
        would discard the finalize that rescues every sampled vote. Kept identical to
        solver/deploy/v2 and pinned by dev/test_delivery_deadline.py, which runs
        against BOTH trees, so the two cannot drift apart again.
        """
        for p in self.passes:
            delivery = p.delivery
            if delivery:
                if ctx.budget.delivery_exhausted():
                    print(f"[solver] delivery budget spent — skipping {p.name}", flush=True)
                    continue
                if ctx.budget.exhausted():
                    print(f"[solver] past soft deadline — delivery pass {p.name}", flush=True)
            elif ctx.budget.exhausted():
                print(f"[solver] out of budget — skipping {p.name}", flush=True)
                continue
            p.run(states, ctx)
            ctx.flush()
            print(f"[solver] pass {p.name} done at t+{ctx.budget.elapsed():.0f}s", flush=True)


def default_passes(cfg: Config) -> list[Pass]:
    """Build the v2 pipeline: solve -> finalize -> validate -> refine -> extra votes.

    Every solve pass is followed by a finalize pass (v1 finalized every
    generation, not just the first) so truncated votes never enter the vote raw.
    """
    passes: list[Pass] = [
        SolvePass(0, greedy=True),
        FinalizePass(),
        ValidatePass(),
        RefinePass(cfg.max_refine_rounds),
    ]
    for i in range(1, cfg.n_samples):
        passes += [SolvePass(i, greedy=False, optional=True), FinalizePass()]
    return passes
