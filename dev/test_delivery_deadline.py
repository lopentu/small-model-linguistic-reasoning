"""Test generation and delivery deadlines in both development solver trees."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TREES = ("solver", "solver/deploy")

FAILS: list[str] = []


def check(cond: bool, detail: str) -> None:  # noqa: FBT001
    """Record a failure without aborting, so one run reports every problem."""
    if not cond:
        FAILS.append(detail)


TRUNCATED = "STAGE 1 — I compare the pairs and see that manz"  # no marker, no answers
RESCUED = "RULES:\n- rescued rule\nFINAL ANSWERS:\nrescued one\nrescued two"


def run_tree(tree: str) -> None:  # noqa: C901, PLR0915 — one function per clock state
    # C901/PLR0915: this is a linear sequence of eight independent scenarios sharing
    # one import block. Splitting it would mean re-importing the tree per case (the
    # `v2` package name is claimed by both trees) or threading eight fixtures through
    # helpers — more moving parts in the thing that guards the delivery contract.
    """Assert the two-deadline contract inside one solver tree."""
    sys.path.insert(0, str(REPO / tree))
    from v2.backends import MockBackend  # noqa: PLC0415 — import must follow the path insert
    from v2.budget import Budget  # noqa: PLC0415
    from v2.config import Config  # noqa: PLC0415
    from v2.pipeline import (  # noqa: PLC0415
        Context,
        FinalizePass,
        Pipeline,
        ProblemState,
        SolvePass,
    )
    from v2.script import make_flusher  # noqa: PLC0415
    from v2.transcript import Transcript  # noqa: PLC0415

    # ---- 1. the two deadlines exist and are ordered -------------------------------
    b = Budget(1000.0)
    check(
        b.delivery_deadline > b.deadline,
        f"[{tree}] delivery deadline is not later than the soft one",
    )
    slack = b.delivery_deadline - b.deadline
    check(
        abs(slack - 120.0) < 1.0,
        f"[{tree}] delivery window is {slack:.0f}s, expected ~120s of 1000",
    )
    reserve = (b.t0 + 1000.0) - b.delivery_deadline
    check(reserve > 25.0, f"[{tree}] only {reserve:.0f}s reserved for the final flush")
    # the backend's respawn/group-timeout logic reads this and must see the LAST clock
    # it may not outlive; publishing the soft deadline made every delivery-window
    # driver hiccup fatal to delivery.
    # Only required where a backend actually reads it (the deploy tree's parallel
    # driver). Deciding that from the SOURCE keeps the check honest in both trees
    # instead of skipping silently if the publish line is ever dropped.
    if "SOLVER_DEADLINE_TS" in (REPO / tree / "v2/backends.py").read_text(encoding="utf-8"):
        published = float(os.environ.get("SOLVER_DEADLINE_TS", "0"))
        check(
            abs(published - b.delivery_deadline) < 1.0,
            f"[{tree}] SOLVER_DEADLINE_TS={published} != delivery deadline {b.delivery_deadline}",
        )

    # ---- 2. soft deadline passed: solving stops, DELIVERY still happens ------------
    calls: list[str] = []

    def handler(msgs: list[dict[str, str]], _greedy: bool, _max_new: int) -> str:  # noqa: FBT001
        # Key on the shared opening, not on a symbol: solver/v2 exposes `finalize_msg(k)`
        # (k-aware count enforcement) while solver/deploy/v2 still has the FINALIZE_MSG
        # constant. Both texts start the same way, and this test is about scheduling.
        last = msgs[-1]["content"]
        is_finalize = last.startswith("Stop. Do NOT analyze")
        calls.append("finalize" if is_finalize else "solve")
        return RESCUED if is_finalize else "fresh solve output"

    os.environ.update({"N_SAMPLES": "1", "TIME_LIMIT_S": "600", "BATCH": "2"})
    cfg = Config.from_env()
    tmp = Path(tempfile.mkdtemp(prefix=f"deadline-{tree.replace('/', '-')}-"))
    row = {"id": "p1", "context": "data", "query": "Translate:\n(1) first\n(2) second"}
    state = ProblemState.from_row(row)
    # a row mid-run: it reasoned (costly, already paid for) but never delivered
    state.samples = [{"kind": "solve", "text": TRUNCATED}]
    states = [state]
    flush = make_flusher(states, str(tmp / "submission.csv"))
    budget = Budget(cfg.time_limit_s)
    budget.deadline = time.time() - 1.0  # soft deadline JUST passed; delivery has not
    ctx = Context(
        cfg=cfg,
        backend=MockBackend(handler),
        budget=budget,
        transcript=Transcript(str(tmp / "t.jsonl")),
        flush=flush,
    )
    Pipeline([SolvePass(1, greedy=False), FinalizePass()]).run(states, ctx)
    flush()
    check("solve" not in calls, f"[{tree}] a solve pass ran past the soft deadline: {calls}")
    check("finalize" in calls, f"[{tree}] DELIVERY was skipped past the soft deadline: {calls}")
    check(
        "rescued one" in state.samples[0]["text"],
        f"[{tree}] truncated sample never got answers: {state.samples[0]['text'][:60]!r}",
    )
    sub = (tmp / "submission.csv").read_text(encoding="utf-8")
    check("rescued one" in sub, f"[{tree}] rescued answers never reached submission.csv")

    # ---- 3. delivery deadline ALSO passed: nothing runs, nothing crashes -----------
    calls.clear()
    state2 = ProblemState.from_row(row)
    state2.samples = [{"kind": "solve", "text": TRUNCATED}]
    states2 = [state2]
    flush2 = make_flusher(states2, str(tmp / "sub2.csv"))
    b3 = Budget(cfg.time_limit_s)
    b3.deadline = b3.delivery_deadline = time.time() - 1.0
    ctx2 = Context(
        cfg=cfg,
        backend=MockBackend(handler),
        budget=b3,
        transcript=Transcript(str(tmp / "t2.jsonl")),
        flush=flush2,
    )
    Pipeline([SolvePass(1, greedy=False), FinalizePass()]).run(states2, ctx2)
    flush2()
    check(not calls, f"[{tree}] work ran past the DELIVERY deadline (hard-kill risk): {calls}")
    check((tmp / "sub2.csv").is_file(), f"[{tree}] no submission.csv written on the exhausted path")

    order: list[str] = []

    class Recorder:
        """A pass that only records that it ran.

        Parameter NAMES match the Pass protocol deliberately: it is structural, so a
        rename here would stop this fake from satisfying it and the test would be
        checking something other than a real pass.
        """

        def __init__(self, name: str, *, delivery: bool) -> None:
            self.name = name
            self.delivery = delivery

        def run(self, states: list[ProblemState], ctx: Context) -> None:
            """Record the call; the scheduler is what is under test."""
            del states, ctx
            order.append(self.name)

    b5 = Budget(cfg.time_limit_s)
    ctx5 = Context(
        cfg=cfg,
        backend=MockBackend(handler),
        budget=b5,
        transcript=Transcript(str(tmp / "t5.jsonl")),
        flush=lambda: None,
    )

    class Overrunner(Recorder):
        """A generator that exhausts the soft budget while running, as solve does."""

        def run(self, states: list[ProblemState], ctx: Context) -> None:
            """Record, then burn the rest of the soft budget."""
            super().run(states, ctx)
            b5.deadline = time.time() - 1.0

    # The clock must trip DURING a pass, not before the pipeline: a pre-tripped clock
    # cannot tell `continue` from `break`, because both stop at the very first
    # (non-delivery) pass. This is the real timeline — solve overruns, then everything
    # after it has to cope.
    Pipeline(
        [
            Overrunner("solve", delivery=False),
            Recorder("fin1", delivery=True),
            Recorder("vote", delivery=False),
            Recorder("fin2", delivery=True),
        ]
    ).run([], ctx5)
    check(
        order == ["solve", "fin1", "fin2"],
        f"[{tree}] pass order after an overrun was {order}, expected "
        "['solve', 'fin1', 'fin2'] — a `break` regression drops fin2, the pass that "
        "rescues sampled votes, and yields ['solve', 'fin1']",
    )

    # ---- 6. the clock can also run out DURING a pass, not only before the pipeline ----
    order.clear()

    class DeadlineTripper:
        """A generator whose own work consumes the rest of the soft budget."""

        name = "solve-slow"
        delivery = False

        def __init__(self, budget: Budget) -> None:
            self.budget = budget

        def run(self, states: list[ProblemState], ctx: Context) -> None:
            """Record, then overrun — exactly what a slow solve pass does."""
            del states, ctx
            order.append(self.name)
            self.budget.deadline = time.time() - 1.0

    b6 = Budget(cfg.time_limit_s)
    ctx6 = Context(
        cfg=cfg,
        backend=MockBackend(handler),
        budget=b6,
        transcript=Transcript(str(tmp / "t6.jsonl")),
        flush=lambda: None,
    )
    Pipeline([DeadlineTripper(b6), Recorder("fin", delivery=True)]).run([], ctx6)
    check(
        order == ["solve-slow", "fin"],
        f"[{tree}] a pass that overran its budget left delivery unreached: {order}",
    )

    # ---- 7. a pass that never declares its side must be REFUSED at construction ------
    class Undeclared:
        """A pass that forgot to say which side of the deadline it belongs on."""

        name = "mystery"

        def run(self, states: list[ProblemState], ctx: Context) -> None:
            """Never reached: construction must refuse this pass."""
            del states, ctx
            order.append("mystery")

    try:
        # ty flags this line too, which is the good outcome — the flag is enforced both
        # statically in-repo and at runtime. The runtime check is what protects a
        # submission package, where the tree is copied and ty never runs again.
        Pipeline([Undeclared()])  # ty: ignore[invalid-argument-type]
    except RuntimeError:
        pass
    else:
        check(
            False,  # noqa: FBT003 — unconditional failure marker
            f"[{tree}] Pipeline accepted a pass with no `delivery` declaration, which "
            "would silently be treated as non-delivery and skipped past the deadline",
        )

    # ---- 8. the published clocks stay consistent with the Budget ----------------------
    if "SOLVER_DEADLINE_TS" in (REPO / tree / "v2/backends.py").read_text(encoding="utf-8"):
        b8 = Budget(1800.0)  # the production limit, not just the 1000 used above
        reserve_1800 = (b8.t0 + 1800.0) - b8.delivery_deadline
        check(
            reserve_1800 > 45.0,
            f"[{tree}] the 1800s eval leaves only {reserve_1800:.0f}s of flush reserve; "
            "the backend's teardown margin alone is 15s",
        )
        check(
            float(os.environ["SOLVER_DEADLINE_TS"]) == b8.delivery_deadline
            and float(os.environ["SOLVER_SOFT_DEADLINE_TS"]) == b8.deadline,
            f"[{tree}] the published clocks do not match the newest Budget — a stale "
            "SOLVER_DEADLINE_TS makes every backend timeout clamp wrong",
        )
        b9 = Budget(1800.0, headroom=0.99)
        check(
            b9.delivery_deadline >= b9.deadline,
            f"[{tree}] headroom=0.99 inverted the clocks: soft {b9.deadline} > "
            f"delivery {b9.delivery_deadline}, which would let solving outlive delivery",
        )

    bk = REPO / tree / "v2/backends.py"
    if "def group_timeout" in bk.read_text(encoding="utf-8"):
        from v2.backends import LlamaParallelBackend  # noqa: PLC0415

        probe = object.__new__(LlamaParallelBackend)  # no binary, no model, no GPU
        probe._warned_no_deadline = True  # noqa: SLF001
        probe._t_init = time.time()  # noqa: SLF001
        limit, t0 = 1800.0, time.time()
        soft, deliver = t0 + limit * 0.85, t0 + limit * 0.97
        margin = probe._TEARDOWN_MARGIN_S  # noqa: SLF001

        def at(elapsed: float) -> float:
            """group_timeout as if `elapsed` seconds of the 1800s limit had passed."""
            os.environ["SOLVER_SOFT_DEADLINE_TS"] = str(soft - elapsed)
            os.environ["SOLVER_DEADLINE_TS"] = str(deliver - elapsed)
            return probe.group_timeout(2000.0, 12, 700)

        # early: the estimate binds, not the clock
        check(at(60.0) > 300.0, f"[{tree}] an early group was clamped to {at(60.0):.0f}s")
        # a SOLVE-phase group just before the soft deadline must not reach into the
        # delivery window: 1520s elapsed leaves 10s of soft, below the teardown margin
        check(
            at(1520.0) <= 0.0,
            f"[{tree}] a group started 10s before the soft deadline got {at(1520.0):.0f}s — "
            "a wedge would spend the window reserved for delivery",
        )
        # a DELIVERY-phase group is bounded by the delivery clock, never past it
        t_deliver = 1600.0  # past soft, 146s of delivery left
        allowed = at(t_deliver)
        check(
            0 < allowed <= (limit * 0.97 - t_deliver) - margin + 1,
            f"[{tree}] delivery-phase group allowance {allowed:.0f}s exceeds the "
            f"{limit * 0.97 - t_deliver:.0f}s left minus the {margin:.0f}s margin",
        )
        # and at the very end it refuses rather than burning the reserve
        check(
            at(1740.0) <= 0.0,
            f"[{tree}] a group 6s before the delivery deadline got {at(1740.0):.0f}s, "
            "which would run into the reserve that pays for the final flush",
        )

    # ---- 4. every pass declares which side it is on --------------------------------
    import v2.pipeline as pl  # noqa: PLC0415

    check(FinalizePass.delivery is True, f"[{tree}] FinalizePass is not marked as delivery")
    # VotePass exists only in the deploy tree; absent classes are a tree difference,
    # a present class with an unstated side is the bug this catches.
    for name in ("SolvePass", "ValidatePass", "RefinePass", "VotePass"):
        cls = getattr(pl, name, None)
        if cls is None:
            continue
        check(
            getattr(cls, "delivery", None) is False,
            f"[{tree}] {name} does not explicitly declare delivery=False",
        )


def main() -> int:
    """Run one tree in-process, or fan out over both in subprocesses."""
    tree = os.environ.get("TREE")
    if tree:
        run_tree(tree)
        for f in FAILS:
            print("FAIL " + f)
        if not FAILS:
            print(f"PASS [{tree}] two-deadline contract: solve stops at 85%, delivery runs to 97%")
        return 1 if FAILS else 0
    # `v2` is a package name both trees claim, so each needs a clean interpreter
    rc = 0
    for t in TREES:
        env = {**os.environ, "TREE": t}
        rc |= subprocess.run([sys.executable, __file__], env=env, check=False).returncode
    return rc


if __name__ == "__main__":
    sys.exit(main())
