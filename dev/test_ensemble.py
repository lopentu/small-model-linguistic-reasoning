"""E34 tests: query-item permutation, answer/weight round-tripping, aggregate wiring.

The failure this suite exists to prevent: a shuffled vote's answers being counted
against the WRONG items. That bug is silent — every row still delivers k answers —
and it previously cost 0.067 vs 0.141 when a group desync shifted predictions by
one problem. So the round trip is asserted directly, and end to end through
aggregate(), not just per helper.

    uv run python dev/test_ensemble.py [v2_parent_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

V2_PARENT = Path(
    sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "solver/deploy"
)
sys.path.insert(0, str(V2_PARENT))

from v2 import ensemble  # noqa: E402
from v2.pipeline import ProblemState, aggregate  # noqa: E402

FAILS: list[str] = []


def check(tag: str, cond: bool, detail: str = "") -> None:
    """Record a failure without aborting the suite."""
    if not cond:
        FAILS.append(f"[{tag}] {detail}")


QUERY = "Translate into English:\n\n1. alpha one\n2. beta two\n3. gamma three\n4. delta four"

# ---- [a] splitting ----------------------------------------------------------
split = ensemble.split_query_items(QUERY, 4)
check("a", split is not None, "clean 4-item query did not split")
if split:
    header, items = split
    check("a", items == ["alpha one", "beta two", "gamma three", "delta four"], f"items {items}")
    check("a", header == ["Translate into English:", ""], f"header {header}")
# a k mismatch must refuse rather than drop/duplicate a slot
check("a", ensemble.split_query_items(QUERY, 5) is None, "k=5 mismatch accepted")
check("a", ensemble.split_query_items(QUERY, 3) is None, "k=3 mismatch accepted")
# trailing prose is structure we don't model
check(
    "a",
    ensemble.split_query_items(QUERY + "\nUse the table above.", 4) is None,
    "trailing prose accepted",
)
# paren labels are the other IOL convention
check("a", ensemble.split_query_items("Q:\n(1) a\n(2) b", 2) is not None, "paren labels rejected")

# ---- [b] permuted rendering -------------------------------------------------
perm = [2, 0, 3, 1]
q = ensemble.permuted_query(QUERY, 4, perm)
check("b", q is not None, "permuted_query returned None")
if q:
    body = [ln for ln in q.splitlines() if ln.strip() and not ln.endswith(":")]
    check("b", body[0] == "1. gamma three", f"first line {body[0]!r}")
    check("b", body[1] == "2. alpha one", f"second line {body[1]!r}")
    check("b", body[3] == "4. beta two", f"fourth line {body[3]!r}")
    # renumbered 1..k, and every original item still present exactly once
    check("b", [ln.split(". ", 1)[0] for ln in body] == ["1", "2", "3", "4"], "not renumbered 1..k")
    check(
        "b",
        sorted(ln.split(". ", 1)[1] for ln in body)
        == sorted(["alpha one", "beta two", "gamma three", "delta four"]),
        "items lost or duplicated",
    )
    check("b", q.startswith("Translate into English:"), "header not preserved")
# an unsplittable query must not be silently shuffled
check("b", ensemble.permuted_query("no items here", 4, perm) is None, "unsplittable query permuted")

# ---- [c] the round trip -----------------------------------------------------
# the model answers the items IN PRESENTED ORDER; unpermute must map them home
presented_answers = ["ANS-gamma", "ANS-alpha", "ANS-delta", "ANS-beta"]
home = ensemble.unpermute(presented_answers, perm, 4)
check("c", home == ["ANS-alpha", "ANS-beta", "ANS-gamma", "ANS-delta"], f"round trip {home}")
# exhaustive: for every permutation of 5 items, answering "in order" must come home
import itertools  # noqa: E402

for p in itertools.permutations(range(5)):
    pl = list(p)
    ans = [f"a{orig}" for orig in pl]  # answer for the item shown at position j
    got = ensemble.unpermute(ans, pl, 5)
    if got != [f"a{i}" for i in range(5)]:
        check("c", False, f"perm {pl} round trip {got}")
        break
# weights travel by the same mapping, with None (not "") as the empty slot
w_home = ensemble.unpermute([0.9, 0.1, None, 0.5], perm, 4, fill=None)
check("c", w_home == [0.1, 0.5, 0.9, None], f"weight round trip {w_home}")
# a short answer list leaves the missing slots empty rather than shifting
check("c", ensemble.unpermute(["x"], [1, 0], 2) == ["", "x"], "short list shifted")

# ---- [d] permutation set ----------------------------------------------------
perms = ensemble.vote_permutations(4, 4, seed=7)
check("d", perms[0] == [0, 1, 2, 3], f"identity not first: {perms[0]}")
check("d", len(perms) == 4, f"wanted 4 orders, got {len(perms)}")
check("d", all(sorted(p) == [0, 1, 2, 3] for p in perms), "not all valid permutations")
check("d", len({tuple(p) for p in perms}) == 4, "duplicate orders when space allows")
check("d", perms == ensemble.vote_permutations(4, 4, seed=7), "not deterministic for a seed")
check("d", perms != ensemble.vote_permutations(4, 4, seed=8), "seed ignored")
# k=1 and k=2 must not hang or produce invalid orders
check("d", ensemble.vote_permutations(1, 5, seed=1) == [[0]] * 5, "k=1 degenerate case")
p2 = ensemble.vote_permutations(2, 5, seed=1)
check("d", len(p2) == 5 and all(sorted(x) == [0, 1] for x in p2), f"k=2 orders {p2}")

# ---- [e] end to end through aggregate --------------------------------------
row = {"id": "q1", "context": "data", "query": QUERY, "task_type": ""}


def st_with(samples: list[dict]) -> ProblemState:
    """Build a ProblemState with the given samples and k=4."""
    st = ProblemState.from_row(row)
    st.k = 4
    st.samples = samples
    return st


# two votes, DIFFERENT orders, both "correct": the vote must agree on every item
p1, p2 = [2, 0, 3, 1], [1, 3, 0, 2]
truth = ["A", "B", "C", "D"]
text1 = '["' + '", "'.join(truth[o] for o in p1) + '"]'
text2 = '["' + '", "'.join(truth[o] for o in p2) + '"]'
st = st_with(
    [
        {"kind": "solve", "text": text1, "meta": None, "perm": p1},
        {"kind": "solve", "text": text2, "meta": None, "perm": p2},
    ]
)
pred, _ = aggregate(st, mode="none")
check("e", pred == truth, f"differently-ordered agreeing votes disagreed: {pred}")

# a sample WITHOUT perm is treated as identity (mixed ensemble)
st_mixed = st_with(
    [
        {"kind": "solve", "text": '["A", "B", "C", "D"]', "meta": None},
        {"kind": "solve", "text": text1, "meta": None, "perm": p1},
    ]
)
check("e", aggregate(st_mixed, mode="none")[0] == truth, "mixed identity/shuffled vote broke")


# and the confidence path: low confidence on the item shown at position 0 (= original
# item 2) must land on original item 2, not item 0
def char_meta(text: str, lp_of: dict[str, float]) -> dict:
    """One token per character, so byte offsets are exercised."""
    return {
        "lps": [lp_of.get(ch, -0.05) for ch in text],
        "tlens": [len(ch.encode("utf-8")) for ch in text],
        "lp_sum": sum(lp_of.get(ch, -0.05) for ch in text),
        "n_decoded": len(text),
        "stop": "eos",
    }


# presented order p1 = [2,0,3,1]; make the FIRST presented answer ("C") hesitant
hesitant = '["C", "A", "D", "B"]'
meta = char_meta(hesitant, {"C": -5.0})
st_w = st_with([{"kind": "solve", "text": hesitant, "meta": meta, "perm": p1}])
from v2 import confidence  # noqa: E402

w = confidence.item_weights(hesitant, ["C", "A", "D", "B"], meta)
w_home = ensemble.unpermute(w, p1, 4, fill=None)
check("e", w_home[2] is not None and w_home[0] is not None, f"weights missing {w_home}")
if w_home[2] is not None and w_home[0] is not None:
    check("e", w_home[2] < w_home[0], f"hesitant confidence landed on the wrong item: {w_home}")
check("e", aggregate(st_w, mode="tiebreak")[0] == truth, "single shuffled sample mis-mapped")

from v2.backends import MockBackend  # noqa: E402
from v2.budget import Budget  # noqa: E402
from v2.config import Config  # noqa: E402
from v2.pipeline import Context, FinalizePass, sample_messages  # noqa: E402
from v2.transcript import Transcript  # noqa: E402

seen_prompts: list[str] = []


def spy(msgs, greedy, max_new):  # noqa: ANN001, ANN201, ARG001 — mock protocol
    """Record the user turn the pass sent, then answer in the trained format."""
    seen_prompts.append("\n".join(m["content"] for m in msgs))
    return '["X", "Y", "Z", "W"]'


perm_f = [2, 0, 3, 1]
st_trunc = st_with([{"kind": "solve", "text": "cut off mid-analys", "meta": None, "perm": perm_f}])
# the conversation helper itself must render the permuted order
conv = sample_messages(st_trunc, st_trunc.samples[0], fewshot=False)
user_turn = "\n".join(m["content"] for m in conv)
check("f", "1. gamma three" in user_turn, "sample_messages did not rebuild the permuted query")
check("f", "1. alpha one" not in user_turn, "sample_messages used the original order")

os_env = {"GGUF_PATH": "", "N_SAMPLES": "1", "FEWSHOT": "0", "TRANSCRIPT": ""}
import os  # noqa: E402

os.environ.update(os_env)
cfg = Config.from_env()
ctx = Context(
    cfg=cfg,
    backend=MockBackend(spy),
    budget=Budget(600.0),
    transcript=Transcript(""),
    flush=lambda: None,
)
FinalizePass().run([st_trunc], ctx)
check("f", bool(seen_prompts), "finalize never called the backend")
if seen_prompts:
    check("f", "1. gamma three" in seen_prompts[0], "finalize follow-up used the original order")
# the merged sample keeps its perm, so its answers still map home
check("f", st_trunc.samples[0].get("perm") == perm_f, "finalize dropped the sample's perm")
pred_f, _ = aggregate(st_trunc, mode="none")
check("f", pred_f == ["Y", "W", "X", "Z"], f"merged finalize answer mis-mapped: {pred_f}")

print("PASS [a] query splitting" if not any(f.startswith("[a]") for f in FAILS) else "FAIL [a]")
print("PASS [b] permuted rendering" if not any(f.startswith("[b]") for f in FAILS) else "FAIL [b]")
print("PASS [c] round trip" if not any(f.startswith("[c]") for f in FAILS) else "FAIL [c]")
print("PASS [d] permutation set" if not any(f.startswith("[d]") for f in FAILS) else "FAIL [d]")
print(
    "PASS [e] aggregate end to end" if not any(f.startswith("[e]") for f in FAILS) else "FAIL [e]"
)
print(
    "PASS [f] follow-up item order" if not any(f.startswith("[f]") for f in FAILS) else "FAIL [f]"
)
for f in FAILS:
    print("  " + f)
sys.exit(1 if FAILS else 0)
