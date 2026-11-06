"""E43 assignment solver: unit guards + a replay of the real 12-sample dump.

The headline number (+10/223 items on match_letters) was measured with a standalone script. This
suite proves the SHIPPED code path reproduces it, that the flag is inert when off, and that the
guards refuse the cases where forcing a bijection would be wrong.

    uv run python dev/test_assignment.py
"""

from __future__ import annotations

import ast
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "solver" / "deploy"))
sys.path.insert(0, str(REPO / "dev"))

from v2 import assignment  # noqa: E402
from v2.pipeline import ProblemState, aggregate  # noqa: E402

FAILS: list[str] = []
DUMP = REPO / "dev/local/e33-lp12-run1/transcript.jsonl"
FIX = REPO / "dev/mock-eval/fixtures"


def check(cond: bool, detail: str) -> None:  # noqa: FBT001
    """Record a failure without aborting the suite."""
    if not cond:
        FAILS.append(detail)


# ---- unit: the guard that keeps this away from prose rows ------------------------------
check(assignment.looks_like_letters(["A", "B", "c", "(d)"]), "rejected plain option labels")
check(assignment.looks_like_letters(["1", "12", "3"]), "rejected numeric labels")
check(
    not assignment.looks_like_letters(["el libro verde", "la manzana roja"]),
    "accepted a TRANSLATION row — would overwrite prose answers with letters",
)
check(not assignment.looks_like_letters(["A"]), "accepted a single answer (nothing to assign)")

# ---- unit: assign() prefers confidence and enforces one-label-each ---------------------
# item 0 weakly wants B, item 1 strongly wants B; the optimum gives B to item 1
out = assignment.assign([["B", "A"], ["B"]], [[0.2, 0.1], [0.9]], fallback=["?", "?"])
check(out == ["A", "B"], f"assignment ignored confidence: {out}")
# degenerate inputs must fall back rather than raise or invent labels
check(
    assignment.assign([["A"]], [[1.0]], fallback=["z"]) == ["z"], "single item not passed through"
)
check(
    assignment.assign([["A"], ["A"]], [[1.0], [1.0]], fallback=["x", "y"]) == ["x", "y"],
    "invented a label when only ONE distinct option was proposed for two items",
)
check(
    assignment.assign([[], []], [[], []], fallback=["p", "q"]) == ["p", "q"],
    "empty candidates did not fall back",
)

# ---- integration: replay the real dump through aggregate() -----------------------------
if not DUMP.is_file() or not (FIX / "answers.csv").is_file():
    print(f"FAIL replay assets missing ({DUMP.name} / fixtures) — cannot verify the 26 -> 36 claim")
    sys.exit(1)

if True:  # the replay always runs; the guard above exits on missing assets
    sys.path.insert(0, str(REPO / "dev"))
    from score_local import normalize

    test = {r["id"]: r for r in csv.DictReader((FIX / "test.csv").open(encoding="utf-8"))}
    gold = {
        r["id"]: ast.literal_eval(r["answer"])
        for r in csv.DictReader((FIX / "answers.csv").open(encoding="utf-8"))
    }
    evs: dict[str, list[dict]] = defaultdict(list)
    with DUMP.open(encoding="utf-8") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") == "sample" and ev.get("text"):
                evs[str(ev["id"])].append(ev)

    def score_rows(*, flag: str) -> tuple[int, int]:
        """Total correct items over match_letters rows with ASSIGNMENT set to `flag`."""
        os.environ["ASSIGNMENT"] = flag
        os.environ["VOTE_WEIGHT"] = "none"  # the vote itself stays plain majority
        hit = tot = 0
        for pid, row in test.items():
            if row.get("task_type") != "match_letters":
                continue
            g = gold[pid]
            st = ProblemState.from_row(row)
            st.samples = [
                {"kind": "solve", "text": e["text"], "meta": e.get("meta")}
                for e in evs.get(pid, [])
            ]
            if not st.samples:
                continue
            final, _ = aggregate(st)
            want = [normalize(str(a[0] if isinstance(a, (list, tuple)) else a)) for a in g]
            for i, w in enumerate(want):
                tot += 1
                hit += i < len(final) and normalize(str(final[i])) == w
        return hit, tot

    off_hit, tot = score_rows(flag="0")
    on_hit, _ = score_rows(flag="1")
    print(f"replay over {tot} match_letters items: ASSIGNMENT=0 -> {off_hit}, =1 -> {on_hit}")
    check(off_hit == 26, f"flag OFF changed the baseline: expected 26, got {off_hit}")
    check(on_hit == 36, f"flag ON did not reproduce the measured gain: expected 36, got {on_hit}")
    check(on_hit > off_hit, f"assignment did not improve on majority ({on_hit} vs {off_hit})")

    # a non-match_letters row must be untouched by the flag
    # EVERY non-match_letters type, so the mutation `task_type != "translation"` cannot survive
    others = sorted({r.get("task_type", "") for r in test.values()} - {"match_letters"})
    for tt in others:
        rid = next((i for i, r in test.items() if r.get("task_type") == tt and evs.get(i)), None)
        if rid is None:
            continue
        outs = []
        for flag in ("0", "1"):
            os.environ["ASSIGNMENT"] = flag
            st = ProblemState.from_row(test[rid])
            st.samples = [
                {"kind": "solve", "text": e["text"], "meta": e.get("meta")} for e in evs[rid]
            ]
            outs.append(aggregate(st)[0])
        check(outs[0] == outs[1], f"a {tt} row changed when ASSIGNMENT was enabled")
    print(f"cross-type check covered: {others}")

for f in FAILS:
    print("FAIL " + f)
print(
    "OK: assignment guards, confidence preference, fallbacks, dump replay, no cross-type effect"
    if not FAILS
    else f"{len(FAILS)} FAILURE(S)"
)
sys.exit(1 if FAILS else 0)
