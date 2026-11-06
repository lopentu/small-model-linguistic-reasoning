"""Budget-aware inference passes and answer aggregation."""

from __future__ import annotations

import ast
import os
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from . import assignment, confidence, detlanes, ensemble, textproc, validators
from .prompts import FINALIZE_MSG, REFINE_MSG, followup_messages, solve_messages

_LIVE_GOLD: dict[str, list] | None = None  # lazily loaded from LIVE_ANSWERS (dev only)


def _em_norm(s: object) -> str:
    """Scorer-faithful normalize for live EM — MIRROR of dev/score_local.normalize.

    Deliberately NOT textproc.normalize (which also collapses internal whitespace and folds
    curly apostrophes, so it would read EM *higher* than the platform — a falsely-optimistic
    dev signal). This tracks the real scorer so live EM is an honest preview. KEEP IN SYNC
    with dev/score_local.normalize and solver/script.py _em_norm.
    """
    s = str(s).strip().strip("\"'‘’“”").removesuffix(".")  # the curly-quote variants are deliberate
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
            import csv  # lazy: dev-only path

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


def qlog(pid: str, kind: str, text: str, k: int, context: str) -> None:
    """Per-question live signal printed the instant a generation finishes.

    So a failing model shows at problem 1, not after the whole run. Structural (marker +
    answers-delivered vs k) => safe in the sandbox; adds per-problem EM only when LIVE_ANSWERS
    gold is set (dev). Pure-Python — no sacrebleu, no test-content echo. Wrapped fail-closed:
    observability must never sink the run.
    """
    try:
        text = text or ""
        parsed = textproc.parse_answers(text, k, context)
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
    samples: list[dict[str, Any]] = field(default_factory=list)
    verdicts: list[str] = field(default_factory=list)

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

    def solve_count(self) -> int:
        """Return the number of plain solve samples (refine samples don't shift vote indices)."""
        return sum(1 for s in self.samples if s["kind"] == "solve")

    def refine_count(self) -> int:
        """Return the number of refine samples (drives the round cap; resume-safe by counting)."""
        return sum(1 for s in self.samples if s["kind"] == "refine")

    def base_messages(self, *, fewshot: bool) -> Messages:
        """Build the initial solve conversation for this problem."""
        return solve_messages(self.context, self.query, fewshot=fewshot)


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
    # A DELIVERY pass turns work already done into answer lines. It is mandatory and
    # cheap, so the scheduler lets it run past the soft deadline into the hard-kill
    # headroom (see Budget.delivery_exhausted). Everything else stops at 85%.
    delivery: bool

    def run(self, states: list[ProblemState], ctx: Context) -> None:
        """Do this step's work; respect ctx.budget; log via ctx.transcript."""
        ...


def sample_messages(state: ProblemState, smp: dict, *, fewshot: bool) -> Messages:
    """Return the conversation a sample came from, honouring its item order.

    A follow-up (finalize/refine) built from the ORIGINAL query while the sample
    answered a PERMUTED one gets answers back in original order, which aggregate
    would then un-permute into nonsense. Rebuilding the exact
    permuted query keeps the sample's stored `perm` valid for its whole life."""
    perm = smp.get("perm")
    if perm and perm != list(range(state.k)):
        q = ensemble.permuted_query(state.query, state.k, perm)
        if q is not None:
            return solve_messages(state.context, q, fewshot=fewshot)
    return state.base_messages(fewshot=fewshot)


def _perm_seed(problem_id: str) -> int:
    """Stable seed per problem: same item orders on a resumed run.

    Deliberately NOT hash() — PYTHONHASHSEED randomizes it per process, so a
    resumed run would rebuild different orders and un-permute stored answers
    against the wrong mapping.
    """
    return zlib.crc32(problem_id.encode("utf-8"))


def _last_metas(backend: object, n: int) -> list[dict | None]:
    """Per-sample logprob metadata from the backend's last chat() call.

    Only LlamaParallelBackend publishes it (and only with VOTE_WEIGHT set); a
    length mismatch means we're looking at some other call's leftovers, so it is
    discarded rather than mis-attributed.
    """
    metas = getattr(backend, "last_meta", None)
    if isinstance(metas, list) and len(metas) == n:
        return list(metas)
    return [None] * n


def _is_matching(state: ProblemState) -> bool:
    """Return True when this row is a letter-matching task with enough items to assign."""
    tt = str(state.row.get("task_type", "")).strip().lower()
    return tt == "match_letters" and state.k >= assignment.MIN_ROWS_FOR_ASSIGNMENT


def _assign_letters(
    state: ProblemState, parsed: list[list[str]], majority: list[str], *, lp_t: float
) -> list[str]:
    """Re-pick a match_letters row as a one-to-one assignment; majority stays the fallback."""
    cands: list[list[str]] = [[] for _ in range(state.k)]
    wts: list[list[float]] = [[] for _ in range(state.k)]
    ordered = sorted(state.samples, key=lambda s: s["kind"] != "refine")
    for smp, ans in zip(ordered, parsed, strict=False):
        w = confidence.item_weights(smp["text"], ans, smp.get("meta"), lp_t=lp_t)
        for i in range(min(state.k, len(ans))):
            v = (ans[i] or "").strip()
            if not v:
                continue
            wi = w[i] if i < len(w) and w[i] else None
            if wi is None:  # no logprobs for this item: counts alone measured WORSE than
                continue  # majority, so contribute nothing rather than flatten the matrix
            cands[i].append(v)
            wts[i].append(wi)
    flat = [v for row in cands for v in row]
    if not assignment.looks_like_letters(flat):
        return majority  # metadata said match_letters but the output is prose — leave it
    return assignment.assign(cands, wts, fallback=majority)


def aggregate(
    state: ProblemState,
    *,
    mode: str | None = None,
    lp_t: float | None = None,
    floor: float | None = None,
) -> tuple[list[str], str]:
    """Parse samples, aggregate answers, and fill missing items. Refinements precede solve samples so they win first-occurrence ties; a majority can override them."""
    mode = (mode if mode is not None else os.environ.get("VOTE_WEIGHT", "none")).strip()
    lp_t = lp_t if lp_t is not None else float(os.environ.get("VOTE_LP_T", "1.0"))
    floor = floor if floor is not None else float(os.environ.get("VOTE_FLOOR", "0"))
    ordered = sorted(state.samples, key=lambda s: s["kind"] != "refine")
    texts = [s["text"] for s in ordered]
    parsed, presented = [], []
    for s, txt in zip(ordered, texts, strict=True):
        ans = textproc.parse_answers(txt, state.k, state.context)
        presented.append(ans)  # order the TEXT used — what confidence spans align to
        perm = s.get("perm")
        parsed.append(ensemble.unpermute(ans, perm, state.k) if perm else ans)
    if mode in ("", "none") or not any(s.get("meta") for s in ordered):
        final = textproc.vote(parsed, state.k)
    else:
        metas = [s.get("meta") for s in ordered]
        # weights are computed against the presented answers (spans live in that text)
        # and then mapped home by the same permutation as the answers themselves
        weights = []
        for s, txt, pres, meta in zip(ordered, texts, presented, metas, strict=True):
            w = confidence.item_weights(txt, pres, meta, lp_t=lp_t)
            perm = s.get("perm")
            weights.append(ensemble.unpermute(w, perm, state.k, fill=None) if perm else w)
        if mode == "bestof":
            scores = [
                confidence.sample_score(t, p, m, lp_t=lp_t) or 0.0
                for t, p, m in zip(texts, presented, metas, strict=True)
            ]
            best = parsed[scores.index(max(scores))] if scores else []
            # majority still fills anything the chosen sample left blank, so
            # best-of-N can only differ from the vote where it has an answer
            majority = textproc.vote(parsed, state.k)
            final = [
                (best[i] if i < len(best) and best[i].strip() else majority[i])
                for i in range(state.k)
            ]
        else:
            final = textproc.vote_weighted(parsed, state.k, weights, mode=mode, floor=floor)
    if os.environ.get("ASSIGNMENT", "0") == "1" and _is_matching(state):
        final = _assign_letters(state, parsed, final, lp_t=lp_t)
    final = detlanes.apply(state.row, final, state.k)
    fb = textproc.fallback_items(state.context, state.k)
    final = [v if v.strip() else fb[i] for i, v in enumerate(final)]
    expl = next((r for t in texts if (r := textproc.parse_rules(t))), "")
    # Only a REAL rules block earns the explanation column. The direct-JSON recipe
    # never writes one, and filling it with boilerplate ("Rules deduced from the
    # given examples") would submit a sentence that explains nothing to the jury
    # track and inflate explanation_rate to a meaningless 100%. The field is
    # optional in the submission contract, so silence is the honest answer.
    return final, expl


# many keyword-only knobs by design; a spec object would just add indirection
def _generate_batched(
    states: Sequence[ProblemState],
    conversations: Sequence[Messages],
    ctx: Context,
    *,
    greedy: bool,
    max_new: int,
    kind: str,
    pass_name: str,
    perms: Sequence[list[int] | None] | None = None,
) -> None:
    """Shared batching loop: budget-checked chunks, transcript logging."""
    batch_size = ctx.cfg.batch
    # a refine of a shuffled sample answers in THAT sample's item order, so the new
    # sample must inherit its perm or aggregate would un-permute it wrongly
    order = list(perms) if perms is not None else [None] * len(states)
    todo = list(zip(states, conversations, order, strict=True))
    i = 0
    while i < len(todo):
        if ctx.budget.exhausted():
            print(f"[solver] deadline guard tripped in {pass_name}", flush=True)
            return
        chunk = todo[i : i + batch_size]
        batches_left = max(1, (len(todo) - i + batch_size - 1) // batch_size)
        cap = ctx.budget.max_new_for(batches_left, max_new)
        t_batch = time.time()
        try:
            texts = ctx.backend.chat([c for _, c, _ in chunk], greedy=greedy, max_new=cap)
        except Exception as e:  # noqa: BLE001 — one bad batch (OOM, template) must not sink the run
            print(f"[solver] batch failed in {pass_name}: {e!r} — skipping chunk", flush=True)
            i += len(chunk)
            continue
        ctx.budget.record_batch(time.time() - t_batch)
        metas = _last_metas(ctx.backend, len(texts))
        for (state, _, perm), text, meta in zip(chunk, texts, metas, strict=True):
            sample = {"kind": kind, "text": text, "meta": meta}
            if perm and perm != list(range(state.k)):
                sample["perm"] = perm
            state.samples.append(sample)
            qlog(state.id, kind, text, state.k, state.context)  # per-question live signal
            ctx.transcript.log(
                {
                    "event": "sample",
                    "id": state.id,
                    "kind": kind,
                    "pass": pass_name,
                    "text": text,
                    "meta": meta,
                    "perm": sample.get("perm"),
                }
            )
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
    # Mandatory DELIVERY pass: allowed past the soft deadline (Budget.delivery_deadline).
    # Cheap by construction — a short greedy continuation of a conversation already
    # prefilled — and the only thing that turns paid-for analysis into scoreable lines.
    delivery = True

    def run(self, states: list[ProblemState], ctx: Context) -> None:
        """Follow up once on every sample that never reached FINAL ANSWERS."""
        if ctx.budget.delivery_exhausted():
            print(
                f"[solver] delivery deadline already passed — {self.name} does nothing", flush=True
            )
            return
        jobs: list[tuple[ProblemState, int]] = []
        convs: list[Messages] = []
        if not any(not smp.get("finalized") for s in states for smp in s.samples):
            return  # every sample already had its one attempt: skip the scan entirely
        for s in states:
            for idx, smp in enumerate(s.samples):
                if smp.get("finalized"):  # one attempt per sample — the loop guard
                    continue
                txt = smp["text"] or ""
                nonblank = sum(1 for a in textproc.parse_answers(txt, s.k) if a.strip())
                if textproc.has_delivery(txt, s.k) and nonblank >= s.k:
                    continue
                prior = textproc.strip_thinking(smp["text"])
                jobs.append((s, idx))
                convs.append(
                    followup_messages(
                        sample_messages(s, smp, fewshot=ctx.cfg.fewshot), prior, FINALIZE_MSG
                    )
                )
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
                qlog(state.id, "finalize", state.samples[idx]["text"], state.k, state.context)
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

    name = "validate"
    delivery = False  # cheap, but produces no answers on its own

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
        refine_perms: list[list[int] | None] = []  # the item order each refine inherits
        for s in states:
            if not s.verdicts or s.refine_count() >= self.max_rounds or not s.samples:
                continue
            last = s.samples[-1]
            prior = textproc.strip_thinking(last["text"])
            msg = REFINE_MSG.format(verdicts="\n".join(f"- {v}" for v in s.verdicts), k=s.k)
            todo.append(s)
            # follow up in the order that sample answered in (see sample_messages)
            convs.append(
                followup_messages(sample_messages(s, last, fewshot=ctx.cfg.fewshot), prior, msg)
            )
            refine_perms.append(last.get("perm"))
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
            perms=refine_perms,
        )


class VotePass:
    """E32: all of a problem's remaining sampled votes in ONE backend call.

    Built for LlamaParallelBackend: [conv]*need collapses to a single prompt
    prefill shared across `need` concurrent decode streams (llama_memory_seq_cp
    in the driver). One problem per call, sequential over problems — vs the
    per-vote SolvePass ladder this cuts prompt prefill from n_samples x 90 to
    1 x 90, which is the dominant cost of self-consistency voting on the T4.

    Degradation shape: budget pressure first shrinks max_new (max_new_for),
    then the deadline guard stops voting entirely — later problems keep their
    greedy solve[0] sample plus fallback, so every row still delivers.
    """

    delivery = False  # optional extra votes — the first thing to drop when behind

    def __init__(self, n_target: int) -> None:
        self.n_target = n_target
        self.name = "votes"

    def _conversations(self, s: ProblemState, need: int, ctx: Context) -> tuple[list, list]:
        """Build `need` vote conversations and the item permutation each one used.

        Without VOTE_SHUFFLE every vote gets the identical prompt (one prefill,
        cloned KV). With it, votes are spread over vote_shuffle_groups item
        ORDERS: each order is one prefill shared by its samples, so diversity
        costs G prefills instead of `need` of them. A row whose query cannot be
        split into exactly k labelled items keeps the identity order — the
        deployed behavior — rather than risking a mis-mapped answer.
        """
        base = s.base_messages(fewshot=ctx.cfg.fewshot)
        ident = list(range(s.k))
        if not ctx.cfg.vote_shuffle or s.k < 2:
            return [base] * need, [ident] * need
        groups = max(1, min(ctx.cfg.vote_shuffle_groups, need))
        # identity is already covered by the greedy solve sample, so votes take the
        # NON-identity orders (seeded by problem id: a resumed run must rebuild the
        # same orders or its stored answers would un-permute wrongly)
        perms = ensemble.vote_permutations(s.k, groups + 1, seed=_perm_seed(s.id))[1:]
        convs, used = [], []
        for j in range(need):
            perm = perms[j % len(perms)]
            q = ensemble.permuted_query(s.query, s.k, perm)
            if q is None:  # unsplittable query: fall back to the validated prompt
                convs.append(base)
                used.append(ident)
                continue
            convs.append(solve_messages(s.context, q, fewshot=ctx.cfg.fewshot))
            used.append(perm)
        return convs, used

    def run(self, states: list[ProblemState], ctx: Context) -> None:
        """Vote every problem up to n_target solve samples (resume-safe by count)."""
        todo = [s for s in states if s.solve_count() < self.n_target]
        for i, s in enumerate(todo):
            if ctx.budget.exhausted() or not ctx.budget.affords(1):
                print(f"[solver] deadline guard tripped in {self.name}", flush=True)
                return
            need = self.n_target - s.solve_count()
            cap = ctx.budget.max_new_for(len(todo) - i, ctx.cfg.max_new)
            convs, perms = self._conversations(s, need, ctx)
            t_batch = time.time()
            try:
                texts = ctx.backend.chat(convs, greedy=False, max_new=cap)
            except Exception as e:  # noqa: BLE001 — one bad problem must not sink the run
                print(f"[solver] batch failed in {self.name}: {e!r} — skipping problem", flush=True)
                continue
            ctx.budget.record_batch(time.time() - t_batch)
            metas = _last_metas(ctx.backend, len(texts))
            # error/timeout salvage returns "" — a blank must not become a vote nor
            # advance solve_count; its meta and permutation are dropped with it
            trio = zip(texts, metas, perms, strict=True)
            kept = [(t, m, p) for t, m, p in trio if t.strip()]
            for text, meta, perm in kept:
                sample = {"kind": "solve", "text": text, "meta": meta}
                if perm != list(range(s.k)):  # only record a real reordering
                    sample["perm"] = perm
                s.samples.append(sample)
                ctx.transcript.log(
                    {
                        "event": "sample",
                        "id": s.id,
                        "kind": "solve",
                        "pass": self.name,
                        "text": text,
                        "meta": meta,
                        "perm": sample.get("perm"),
                    }
                )
            qlog(s.id, f"votes[{len(kept)}]", kept[0][0] if kept else "", s.k, s.context)
            ctx.flush()  # bank partial credit per problem


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
        emitting FINAL ANSWERS still get their answer lines. Non-delivery passes are
        skipped individually rather than breaking the loop, because a finalize pass
        sits after each of them and must still be reached.
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

    REFINE_ROUNDS is the number of ROUNDS, not a cap that runs once: each round is a
    [validate -> refine -> finalize] cycle. A single RefinePass call only appends ONE
    refine per flagged problem, so N rounds means the cycle is scheduled N times — every
    round re-validates the CURRENT aggregate (including the previous round's refine),
    refines only problems still flagged, and finalizes so the new answers are delivered.
    RefinePass(max_rounds) additionally caps total refines per problem, so a problem that
    stops being flagged drops out. All of it is budget-gated and stops at the soft
    deadline (LOBSTER's iterative-refinement lever, E20). REFINE_ROUNDS=1 reproduces the
    prior single-round pipeline exactly; REFINE_ROUNDS=0 skips refinement entirely.
    """
    passes: list[Pass] = [
        SolvePass(0, greedy=True),
        FinalizePass(),
    ]
    for _ in range(cfg.max_refine_rounds):
        passes += [ValidatePass(), RefinePass(cfg.max_refine_rounds), FinalizePass()]
    if cfg.backend == "parallel":
        if cfg.n_samples > 1:
            passes += [VotePass(cfg.n_samples), FinalizePass()]
        return passes
    for i in range(1, cfg.n_samples):
        passes += [SolvePass(i, greedy=False, optional=True), FinalizePass()]
    return passes
