"""Test the numeral pre-pass routing contract (no GPU, no model).

    uv run python dev/test_numbrute.py

Covers: a provable number task is solved and excluded (answer injected as a
solve sample so aggregate writes it); a non-number task is never touched; a
number task the solver cannot prove falls through untouched; the env kill-switch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solver"))

from v2.numbrute import presolve_numerals  # noqa: E402
from v2.pipeline import ProblemState, aggregate  # noqa: E402


def mk(ctx: str, q: str, task_type: str = "text_to_num", pid: str = "t1") -> ProblemState:
    return ProblemState.from_row({"id": pid, "context": ctx, "query": q, "task_type": task_type})


def main() -> None:
    os.environ["NUMBRUTE"] = "1"

    # 1. a fully-determined number task: every asked word is in the training table,
    #    so the solver proves both -> solved, excluded, answers injected for the flush.
    table = "one = 1\ntwo = 2\nthree = 3\nfour = 4\nfive = 5"
    solved = mk(table, "Write in numerals:\n(1) three\n(2) five")
    got = presolve_numerals([solved])
    assert got == {"t1"}, f"provable number task not solved: {got}"
    assert any(s["kind"] == "solve" for s in solved.samples), "no solve sample injected"
    pred, _ = aggregate(solved)
    assert pred == ["3", "5"], f"wrong proven answers: {pred}"

    # 2. a non-number task is never touched (no sample, not in the solved set)
    other = mk("gato = cat\nperro = dog", "Translate:\n(1) gato", task_type="translate", pid="t2")
    got2 = presolve_numerals([other])
    assert got2 == set(), f"non-number task was routed to brute force: {got2}"
    assert other.samples == [], "non-number task got a sample injected"

    # 3. a number task with no usable training pairs cannot be proven -> falls through
    unprov = mk("(some prose, no numeral table)", "Write in numerals:\n(1) mystery", pid="t3")
    got3 = presolve_numerals([unprov])
    assert "t3" not in got3, "an unprovable number task was wrongly committed"

    # 4. kill-switch: NUMBRUTE=0 disables the pre-pass entirely
    os.environ["NUMBRUTE"] = "0"
    off = mk(table, "Write in numerals:\n(1) three", pid="t4")
    assert presolve_numerals([off]) == set(), "pre-pass ran despite NUMBRUTE=0"
    assert off.samples == [], "pre-pass mutated state despite NUMBRUTE=0"
    os.environ["NUMBRUTE"] = "1"

    print("OK: numbrute routing — solved+excluded, non-number untouched, unprovable falls through, kill-switch")


if __name__ == "__main__":
    main()
