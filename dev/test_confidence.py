"""E33 tests: logprob->item alignment, weighted voting, and safe degradation.

Runs against the DEPLOYMENT tree solver/deploy/v2 (what ships), not the repo's
experimental solver/v2. Pass another root as argv[1] to test a build area copy:

    uv run python dev/test_confidence.py [v2_parent_dir]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

V2_PARENT = Path(
    sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "solver/deploy"
)
sys.path.insert(0, str(V2_PARENT))

from v2 import confidence, textproc  # noqa: E402
from v2.pipeline import ProblemState, aggregate  # noqa: E402

FAILS: list[str] = []


def check(tag: str, cond: bool, detail: str = "") -> None:
    """Record a failure without aborting the rest of the suite."""
    if not cond:
        FAILS.append(f"[{tag}] {detail}")


def char_meta(text: str, lp_of: dict[str, float] | None = None, default: float = -0.1) -> dict:
    """Meta for a hypothetical tokenizer that emits ONE token per character.

    Byte lengths come from the characters themselves, so multibyte text
    exercises the char->byte mapping rather than assuming 1 byte per token.
    """
    lp_of = lp_of or {}
    return {
        "lps": [lp_of.get(ch, default) for ch in text],
        "tlens": [len(ch.encode("utf-8")) for ch in text],
        "lp_sum": sum(lp_of.get(ch, default) for ch in text),
        "n_decoded": len(text),
        "stop": "eos",
    }


# ---- [a] JSON array spans ---------------------------------------------------
text = '["alpha", "b\\"eta", "gam ma"]'
got = confidence.json_array_spans(text)
check("a", got is not None, "spans not found")
if got:
    values, spans = got
    check("a", values == ["alpha", 'b"eta', "gam ma"], f"values {values}")
    # each span must delimit the literal's inner content in the RAW text
    check(
        "a",
        text[spans[0][0] : spans[0][1]] == "alpha",
        f"span0 {text[spans[0][0] : spans[0][1]]!r}",
    )
    check(
        "a",
        text[spans[1][0] : spans[1][1]] == 'b\\"eta',
        f"span1 {text[spans[1][0] : spans[1][1]]!r}",
    )
    check(
        "a",
        text[spans[2][0] : spans[2][1]] == "gam ma",
        f"span2 {text[spans[2][0] : spans[2][1]]!r}",
    )

# the LAST array wins, matching textproc._json_preds' choice
two = 'junk ["x"] then ["y", "z"]'
got2 = confidence.json_array_spans(two)
check("a", got2 is not None and got2[0] == ["y", "z"], f"last-array choice {got2}")

# THE invariant: whatever array _json_preds fed to parse_answers is the array we
# span-scan. (On '[["x"], ["y"]]' both pick the inner ["y"] — the last-first scan
# reaches it first — so agreement, not rejection, is what makes the weights land
# on the right items.)
for shape in ('[["x"], ["y"]]', "[1, 2]", '["a", "b"]', 'text ["only"]', "[]", "no array here"):
    preds = textproc._json_preds(shape, 2)  # noqa: SLF001 — asserting the coupling
    sp = confidence.json_array_spans(shape)
    if preds is None:
        check("a", sp is None, f"{shape!r}: spans without preds")
    else:
        vals = ((sp[0] if sp else []) + ["", ""])[:2]
        check("a", sp is not None and vals == preds, f"{shape!r}: {vals} != {preds}")

# ---- [b] item weights use each item's OWN tokens ----------------------------
# 'zzz' is written with very low confidence, 'aaa' with high
body = '["aaa", "zzz"]'
meta = char_meta(body, lp_of={"z": -3.0, "a": -0.01})
answers = textproc.parse_answers(body, 2)
check("b", answers == ["aaa", "zzz"], f"parse_answers {answers}")
w = confidence.item_weights(body, answers, meta)
check("b", w[0] is not None and w[1] is not None, f"weights {w}")
if w[0] is not None and w[1] is not None:
    check("b", math.isclose(w[0], math.exp(-0.01), rel_tol=1e-6), f"w0 {w[0]}")
    check("b", math.isclose(w[1], math.exp(-3.0), rel_tol=1e-6), f"w1 {w[1]}")
    check("b", w[0] > w[1], "confident item did not outweigh the hesitant one")

# multibyte content must not shift the alignment (2-byte chars in item 0)
uni = '["ñé", "ok"]'
umeta = char_meta(uni, lp_of={"ñ": -2.0, "é": -2.0, "o": -0.02, "k": -0.02})
uans = textproc.parse_answers(uni, 2)
uw = confidence.item_weights(uni, uans, umeta)
check("b", uw[0] is not None and uw[1] is not None, f"unicode weights {uw}")
if uw[0] is not None and uw[1] is not None:
    check("b", math.isclose(uw[0], math.exp(-2.0), rel_tol=1e-6), f"unicode w0 {uw[0]}")
    check("b", math.isclose(uw[1], math.exp(-0.02), rel_tol=1e-6), f"unicode w1 {uw[1]}")

# ---- [c] meta_ok is the guard against mis-attribution ------------------------
bad = char_meta(body)
bad["tlens"] = bad["tlens"][:-2]  # byte total no longer describes the text
check("c", not confidence.meta_ok(body, bad), "truncated tlens accepted")
check("c", not confidence.meta_ok(body, None), "None meta accepted")
check("c", confidence.item_weights(body, answers, bad) == [None, None], "bad meta produced weights")
check("c", confidence.meta_ok(body, meta), "valid meta rejected")

# ---- [d] weighted voting semantics ------------------------------------------
# strict majority is untouched by confidence, even a very confident dissenter
samples = [["x"], ["x"], ["y"]]
weights = [[0.1], [0.1], [0.99]]
check(
    "d",
    textproc.vote_weighted(samples, 1, weights, mode="tiebreak") == ["x"],
    "tiebreak broke a majority",
)
# a 1-1 tie is decided by confidence instead of sample order
tie = [["late"], ["early"]]
tie_w = [[0.9], [0.2]]
check(
    "d",
    textproc.vote_weighted(tie, 1, tie_w, mode="tiebreak") == ["late"],
    "tie not broken by confidence",
)
check(
    "d",
    textproc.vote([["early"], ["late"]], 1) == ["early"],
    "plain vote no longer first-occurrence",
)
# full mode lets 2 confident samples outvote 3 hesitant ones
full = [["a"], ["a"], ["b"], ["b"], ["b"]]
full_w = [[0.95], [0.95], [0.2], [0.2], [0.2]]
check(
    "d",
    textproc.vote_weighted(full, 1, full_w, mode="full") == ["a"],
    "full mode ignored confidence",
)
check(
    "d",
    textproc.vote_weighted(full, 1, full_w, mode="tiebreak") == ["b"],
    "tiebreak overrode a majority",
)
# missing weights => weight 1.0 => same answer as plain counting
none_w: list[list[float | None]] = [[None], [None], [None], [None], [None]]
check(
    "d",
    textproc.vote_weighted(full, 1, none_w, mode="full") == textproc.vote(full, 1),
    "unweighted samples diverged from plain vote",
)
# surface form: tiebreak keeps first occurrence, full takes the confident one
surf = [["Roma"], ["roma"]]
check(
    "d",
    textproc.vote_weighted(surf, 1, [[0.1], [0.9]], mode="tiebreak") == ["Roma"],
    "tiebreak surface",
)
check("d", textproc.vote_weighted(surf, 1, [[0.1], [0.9]], mode="full") == ["roma"], "full surface")

# ---- [e] aggregate() end to end ---------------------------------------------
row = {"id": "q1", "context": "some data table", "query": "1. one\n2. two", "task_type": ""}


def state_with(*texts_metas: tuple[str, dict | None]) -> ProblemState:
    """Build a ProblemState carrying the given (text, meta) solve samples."""
    st = ProblemState.from_row(row)
    st.k = 2
    st.samples = [{"kind": "solve", "text": t, "meta": m} for t, m in texts_metas]
    return st


hi = '["aaa", "bbb"]'
lo = '["aaa", "zzz"]'
hi_meta = char_meta(hi, lp_of={"b": -0.01, "a": -0.01})
lo_meta = char_meta(lo, lp_of={"z": -4.0, "a": -0.01})
# 1-1 tie on item 2: confidence must pick 'bbb'
st = state_with((lo, lo_meta), (hi, hi_meta))
pred, _ = aggregate(st, mode="tiebreak")
check("e", pred == ["aaa", "bbb"], f"tiebreak aggregate {pred}")
# mode none reproduces plain majority (first occurrence wins the tie)
pred_none, _ = aggregate(st, mode="none")
check("e", pred_none == ["aaa", "zzz"], f"none-mode aggregate {pred_none}")
# metadata absent => weighted modes fall back to plain majority, never crash
st_nometa = state_with((lo, None), (hi, None))
check(
    "e",
    aggregate(st_nometa, mode="full")[0] == aggregate(st_nometa, mode="none")[0],
    "no-meta divergence",
)
# best-of-N picks the more confident sample's whole answer set
st_best = state_with((lo, lo_meta), (hi, hi_meta))
pred_b, _ = aggregate(st_best, mode="bestof")
check("e", pred_b == ["aaa", "bbb"], f"bestof aggregate {pred_b}")
# a blank in the chosen sample is still filled from the majority vote
half = '["aaa", ""]'
half_meta = char_meta(half, lp_of={"a": -0.001})
st_half = state_with((half, half_meta), (lo, lo_meta))
pred_h, _ = aggregate(st_half, mode="bestof")
check("e", pred_h == ["aaa", "zzz"], f"bestof blank fill {pred_h}")

# ---- [f] line-heuristic answers still get weights (rfind fallback) ----------
lines = "FINAL ANSWERS:\nalpha\nbeta"
lmeta = char_meta(lines, lp_of={"b": -2.5, "e": -2.5, "t": -2.5})
lans = textproc.parse_answers(lines, 2)
check("f", lans == ["alpha", "beta"], f"line parse {lans}")
lw = confidence.item_weights(lines, lans, lmeta)
check("f", lw[0] is not None and lw[1] is not None and lw[0] > lw[1], f"line weights {lw}")

# Regression cases for confidence parsing.


pre = 'analysis ["a", "b"]\nFINAL ANSWERS:\na\nb'
pans = textproc.parse_answers(pre, 2)
check("g", pans == ["a", "b"], f"setup: parse_answers {pans}")
spans_pre = confidence._item_spans(pre, pans)  # noqa: SLF001 — asserting internal alignment
marker_end = pre.index("FINAL ANSWERS:")
check(
    "g",
    all(s is None or s[0] > marker_end for s in spans_pre),
    f"weights taken from the pre-marker array: {spans_pre}",
)

sub = "FINAL ANSWERS:\na\nba"
sans = textproc.parse_answers(sub, 2)
check("g", sans == ["a", "ba"], f"setup: {sans}")
ss = confidence._item_spans(sub, sans)  # noqa: SLF001
check("g", ss[0] is not None and ss[1] is not None, f"spans missing {ss}")
if ss[0] and ss[1]:
    check("g", ss[0][1] <= ss[1][0], f"spans overlap: {ss}")
    check("g", sub[ss[0][0] : ss[0][1]] == "a", f"item0 span {sub[ss[0][0] : ss[0][1]]!r}")
    check("g", sub[ss[1][0] : ss[1][1]] == "ba", f"item1 span {sub[ss[1][0] : ss[1][1]]!r}")

maj = textproc.vote([["x"], ["y"], ["y"]], 1)
short = textproc.vote_weighted([["x"], ["y"], ["y"]], 1, [[0.5]], mode="full")
check("g", short == maj, f"short weight list changed the winner: {short} vs majority {maj}")

for bad_meta in (
    {"lps": ["oops"], "tlens": [1]},
    {"lps": [float("nan")], "tlens": [1]},
    {"lps": [float("inf")], "tlens": [1]},
    {"lps": [-0.1], "tlens": ["x"]},
    {"lps": [True], "tlens": [1]},
    "not a dict",
):
    check("g", not confidence.meta_ok("a", bad_meta), f"meta_ok accepted {bad_meta!r}")
    try:
        got = confidence.item_weights("a", ["a"], bad_meta)
        check("g", got == [None], f"weights from malformed meta: {got}")
    except Exception as e:  # noqa: BLE001 — the point is that it must NOT raise
        check("g", False, f"item_weights raised on {bad_meta!r}: {type(e).__name__}")

# a text containing a thinking block cannot have its offsets mapped back, so it must
# degrade to no weights rather than align against shifted bytes
think = '<think>\nreasoning\n</think>\n["a", "b"]'
tmeta = char_meta(think, lp_of={"a": -0.01})
check(
    "g",
    confidence.item_weights(think, ["a", "b"], tmeta) == [None, None],
    "thinking-block text aligned anyway",
)

print("PASS [a] json array spans" if not any(f.startswith("[a]") for f in FAILS) else "FAIL [a]")
print("PASS [b] item weights" if not any(f.startswith("[b]") for f in FAILS) else "FAIL [b]")
print("PASS [c] meta guard" if not any(f.startswith("[c]") for f in FAILS) else "FAIL [c]")
print("PASS [d] weighted vote" if not any(f.startswith("[d]") for f in FAILS) else "FAIL [d]")
print("PASS [e] aggregate modes" if not any(f.startswith("[e]") for f in FAILS) else "FAIL [e]")
print("PASS [f] line-parse fallback" if not any(f.startswith("[f]") for f in FAILS) else "FAIL [f]")
print("PASS [g] review regressions" if not any(f.startswith("[g]") for f in FAILS) else "FAIL [g]")
for f in FAILS:
    print("  " + f)
sys.exit(1 if FAILS else 0)
