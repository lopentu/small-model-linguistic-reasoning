"""Local EM + chrF scorer replicating the platform's documented behavior.

Scoring per the challenge (`/competition_info` + the organizers' Colab cell 11):
  - preds in submission.csv `pred` are JSON lists, aligned to gold items BY POSITION
    (padded with "" / truncated to len(gold))
  - exact match: normalized equality against ANY accepted alternative
  - chrF: sacrebleu CHRF sentence score, max over alternatives
  - normalization: case, surrounding whitespace and quotes, trailing period
  - score = sqrt(EM * chrF)  (platform weights by official IOL points — unknown to us,
    so this replica weights items uniformly; treat results as relative, not absolute)

Usage:
  uv run python dev/score_local.py <submission.csv> <answers.csv>

answers.csv: columns id,answer where answer is a Python-literal list; items may
themselves be lists of accepted alternatives (facebook/linguini format).
"""

import argparse
import os
import ast
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sacrebleu.metrics import CHRF

chrf = CHRF()
SPLIT_JSON = Path(os.environ.get("SPLIT_JSON", str(Path(__file__).parent / "mock-eval" / "split.json")))
Raw = list[tuple[str, float, float, int]]  # per-problem-raw rows: (id, em_sum, cf_sum, gold_len)


def normalize(s: str) -> str:
    s = str(s).strip()
    s = s.strip("\"'‘’“”")  # noqa: RUF001 — curly-quote variants are deliberately normalized
    s = s.removesuffix(".")
    return s.strip().lower()


def gold_alternatives(reference: str) -> list[list[str]]:
    v = ast.literal_eval(str(reference))
    return [
        [str(a) for a in item] if isinstance(item, (list, tuple)) else [str(item)] for item in v
    ]


def score(  # noqa: PLR0915 — one cohesive scorer
    submission_csv: str, answers_csv: str, subset_ids: set[str] | None = None
) -> dict:
    sub = pd.read_csv(submission_csv, dtype=str).fillna("")
    gold = pd.read_csv(answers_csv, dtype=str).fillna("")
    if subset_ids is not None:  # score only a fold (Search/Holdout) — anti-overfit discipline
        gold = gold[gold["id"].isin(subset_ids)]
    preds_by_id = dict(zip(sub["id"], sub["pred"], strict=True))

    em_total = cf_total = n = 0
    per_problem = []
    per_problem_raw: list[tuple[str, float, float, int]] = []  # (id, em_sum, cf_sum, gold_len)
    # structural diagnostics — the class of failure aggregate scores HIDE (e.g. the
    # n_items under-count: a length-1 pred vs a 26-item gold scores ~1/26 and looks
    # like weak reasoning, not a forfeit). Computed from submission.csv alone.
    forfeited = rows_under = rows_over = blank_items = rows_blank = 0
    order_gap_rows = order_gap_items = 0
    worst: list[tuple[str, int, int]] = []  # (id, pred_len, gold_len) where pred_len < gold_len
    for _, row in gold.iterrows():
        alts = gold_alternatives(row["answer"])
        gold_len = len(alts)
        try:
            preds = json.loads(preds_by_id.get(row["id"], "[]"))
            if not isinstance(preds, list):
                preds = [str(preds)]
        except Exception:  # noqa: BLE001 — any malformed pred scores as empty, like the platform
            preds = []
        pred_len = len(preds)  # BEFORE padding — the honest count the solver emitted
        if pred_len < gold_len:
            rows_under += 1
            forfeited += gold_len - pred_len
            worst.append((row["id"], pred_len, gold_len))
        elif pred_len > gold_len:
            rows_over += 1
        preds = ([str(p) for p in preds] + [""] * gold_len)[:gold_len]
        row_blanks = sum(1 for p in preds if not p.strip())
        blank_items += row_blanks
        rows_blank += 1 if row_blanks else 0
        p_em = p_cf = pos_hits = 0
        for p, a in zip(preds, alts, strict=True):  # padded/truncated above
            hit = any(normalize(p) == normalize(x) for x in a)
            # default=0.0 guards an empty alternatives list (a malformed/empty gold item):
            # without it max() raises ValueError and crashes the ENTIRE score for the model
            cf = (
                max((chrf.sentence_score(p, [x]).score for x in a), default=0.0)
                if p.strip()
                else 0.0
            )
            p_em += hit
            pos_hits += hit
            p_cf += cf
        # order-error probe: how many items would match if order didn't matter? a big gap
        # means the model HAS the answers but in the wrong positions — a distinct failure
        # (prompt/anchoring), invisible to positional EM and to the length diagnostics.
        #
        # DEGENERATE CASE (12 Jul, Qwen 35B): on a bijective matching task the gold is a
        # PERMUTATION of a fixed option set (e.g. 18 distinct letters A-R), so ANY complete
        # enumeration multiset-matches 100% - a model that just dumps the sorted option list
        # (a genuine non-answer) would score multiset=k, pos≈0 and get falsely flagged as
        # "right answer, wrong position." Skip the probe when gold is all-distinct single-char
        # tokens (the letter-matching signature): the multiset metric carries no signal there.
        gold_single_char = all(len(a) == 1 and len(a[0]) == 1 for a in alts)
        # `if a` guards an empty accepted-alternatives item ([]): a[0] would IndexError,
        # and the scorer tolerates empty alts elsewhere (chrF max(..., default=0.0)).
        all_distinct = len({a[0] for a in alts if a}) == len(alts)
        if not (gold_single_char and all_distinct):
            used: set[int] = set()
            multiset_hits = 0
            for p in preds:
                for j, a in enumerate(alts):
                    if j not in used and any(normalize(p) == normalize(x) for x in a):
                        multiset_hits += 1
                        used.add(j)
                        break
            if multiset_hits > pos_hits:
                order_gap_rows += 1
                order_gap_items += multiset_hits - pos_hits
        per_problem.append((row["id"], p_em / gold_len, p_cf / gold_len / 100))
        per_problem_raw.append((row["id"], p_em, p_cf, gold_len))  # for problem-level bootstrap
        em_total += p_em
        cf_total += p_cf
        n += gold_len

    em = em_total / n if n else 0.0
    cf = cf_total / n / 100 if n else 0.0
    return {
        "items": n,
        "exact_match": em,
        "chrf": cf,
        "geometric_mean": (em * cf) ** 0.5,
        "per_problem": per_problem,
        "per_problem_raw": per_problem_raw,
        "diagnostics": {
            "rows_undercounted": rows_under,  # pred shorter than gold -> guaranteed zeros
            "items_forfeited": forfeited,  # items that CANNOT score, whatever the model said
            "rows_overcounted": rows_over,  # harmless (scorer truncates) but worth noticing
            "blank_items": blank_items,  # emitted blank -> chrF 0 (never-blank rule violated)
            "rows_with_blanks": rows_blank,
            "worst_undercounts": sorted(worst, key=lambda w: w[2] - w[1], reverse=True)[:5],
            "order_gap_rows": order_gap_rows,
            "order_gap_items": order_gap_items,
        },
    }


def _pooled_geo(rows: Raw) -> float:
    """Item-weighted sqrt(EM*chrF) over a set of per-problem-raw rows."""
    em_s = sum(r[1] for r in rows)
    cf_s = sum(r[2] for r in rows)
    n_s = sum(r[3] for r in rows)
    return ((em_s / n_s) * (cf_s / n_s / 100)) ** 0.5 if n_s else 0.0


def _group_by_problem(raw: Raw, problem_of: dict[str, str] | None) -> list[Raw]:
    """Bucket per-QUERY rows into PROBLEM groups (via problem_of map; id-per-group if absent).

    The resample unit must be the problem, not the query — queries of one puzzle are
    correlated, so resampling them independently under-estimates variance."""
    groups: dict[str, Raw] = defaultdict(list)
    for row in raw:
        groups[problem_of.get(row[0], row[0]) if problem_of else row[0]].append(row)
    return list(groups.values())


def _percentile_ci(vals: list[float]) -> tuple[float, float, float]:
    """(median, 2.5%, 97.5%) of a bootstrap distribution."""
    vals.sort()
    return (vals[len(vals) // 2], vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))])


def bootstrap_ci(
    raw: Raw, problem_of: dict[str, str] | None = None, n_boot: int = 10000, seed: int = 0
) -> tuple[float, float, float]:
    """Problem-level grouped bootstrap of the pooled geo => (median, lo2.5%, hi97.5%).

    Resamples whole PROBLEMS with replacement (see _group_by_problem). Descriptive only — to
    RANK two recipes use bootstrap_paired_delta (an absolute-CI overlap is not a valid test).
    """
    groups = _group_by_problem(raw, problem_of)
    if not groups:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)  # noqa: S311 — statistical resample, not cryptographic
    k = len(groups)
    geos = [
        _pooled_geo([r for _ in range(k) for r in groups[rng.randrange(k)]]) for _ in range(n_boot)
    ]
    return _percentile_ci(geos)


def bootstrap_paired_delta(
    raw_a: Raw,
    raw_b: Raw,
    problem_of: dict[str, str] | None = None,
    n_boot: int = 10000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired problem-level bootstrap of geo(A)-geo(B) => (median Δ, lo, hi).

    The GATING metric: align A and B by problem, resample the SAME problem
    indices for both each draw (noise is problem-shared, so paired Δ CI << two absolute CIs).
    The interval describes this paired comparison under the chosen resampling units."""

    def _key(g: Raw) -> str:
        return problem_of.get(g[0][0], g[0][0]) if problem_of else g[0][0]

    b_by_key = {_key(g): g for g in _group_by_problem(raw_b, problem_of)}
    paired = [
        (g, b_by_key[_key(g)]) for g in _group_by_problem(raw_a, problem_of) if _key(g) in b_by_key
    ]
    if not paired:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)  # noqa: S311 — statistical resample, not cryptographic
    k = len(paired)
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(k) for _ in range(k)]
        a_rows = [r for i in idx for r in paired[i][0]]
        b_rows = [r for i in idx for r in paired[i][1]]
        deltas.append(_pooled_geo(a_rows) - _pooled_geo(b_rows))
    return _percentile_ci(deltas)


def _problem_of() -> dict[str, str] | None:
    """Id -> problem key from split.json (for problem-level resampling); None if absent."""
    try:
        return json.loads(SPLIT_JSON.read_text(encoding="utf-8")).get("problem_of")
    except (OSError, json.JSONDecodeError):
        return None


def _fold_ids(fold: str) -> set[str]:
    """Load the frozen Search/Holdout id set from split.json."""
    data = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    return set(data[fold])


def main() -> None:
    """CLI: score a submission (optionally on a fold), with a problem-level bootstrap CI."""
    ap = argparse.ArgumentParser(description="Local EM+chrF scorer with fold + bootstrap CI.")
    ap.add_argument("submission")
    ap.add_argument("answers")
    ap.add_argument("--split", choices=["search", "holdout"], help="score only this frozen fold")
    ap.add_argument("--vs", metavar="SUBMISSION_B", help="paired-delta CI: this vs SUBMISSION_B")
    args = ap.parse_args()

    subset = _fold_ids(args.split) if args.split else None
    fold_label = args.split or "all"
    pmap = _problem_of()
    r = score(args.submission, args.answers, subset_ids=subset)
    for pid, em, cf in r["per_problem"]:
        print(f"  {pid}  EM={em:.2f}  chrF={cf:.3f}")
    unmapped = [x[0] for x in r["per_problem_raw"] if x[0] not in (pmap or {})]
    unit = "problem-level" if not unmapped else "row fallback for unmapped IDs"
    if unmapped:
        print(f"WARNING: {len(unmapped)} row IDs lack problem mappings; intervals may ignore shared-problem dependence")
    mid, lo, hi = bootstrap_ci(r["per_problem_raw"], pmap)
    n_prob = len({(pmap or {}).get(x[0], x[0]) for x in r["per_problem_raw"]})
    print(
        f"TOTAL [{fold_label}, {n_prob} problems, {r['items']} items]: "
        f"EM={r['exact_match']:.4f}  chrF={r['chrf']:.4f}  sqrt(EM*chrF)={r['geometric_mean']:.4f}"
    )
    print(f"  bootstrap 95% CI ({unit}): [{lo:.4f}, {hi:.4f}]  (median {mid:.4f})")
    if args.vs:
        rb = score(args.vs, args.answers, subset_ids=subset)
        dmid, dlo, dhi = bootstrap_paired_delta(r["per_problem_raw"], rb["per_problem_raw"], pmap)
        verdict = "0 outside CI" if (dlo > 0 or dhi < 0) else "0 inside CI"
        print(f"  PAIRED Δgeo(this-vs)={dmid:+.4f}  95% CI [{dlo:+.4f},{dhi:+.4f}]  => {verdict}")
    d = r["diagnostics"]
    print("\nDIAGNOSTICS (issues aggregate scores hide):")
    print(
        f"  items forfeited by under-count: {d['items_forfeited']}/{r['items']} "
        f"({d['items_forfeited'] / max(r['items'], 1):.0%}) across {d['rows_undercounted']} rows"
    )
    print(f"  blank items emitted: {d['blank_items']} in {d['rows_with_blanks']} rows")
    print(f"  over-counted rows (harmless): {d['rows_overcounted']}")
    print(
        f"  order-error items (right answer, wrong position): {d['order_gap_items']} "
        f"in {d['order_gap_rows']} rows"
    )
    if d["worst_undercounts"]:
        shown = ", ".join(f"{pid}:{pl}/{gl}" for pid, pl, gl in d["worst_undercounts"])
        print(f"  worst under-counts (id:pred/gold): {shown}")


if __name__ == "__main__":
    main()
