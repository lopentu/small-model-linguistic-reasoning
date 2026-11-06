"""Match public and private leaderboard rows by submission timestamp. Public-best selection and split differences both affect observed score changes; these are not component-effect intervals."""

from __future__ import annotations

import csv
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "refs/leaderboard_final"
BASELINES = ("Baseline-", "BaselinesTinyAya", "Tiny-Aya-R1", "iolai-test")
US = "LOPE-NTU"
Pair = tuple[dict, dict]


def load(name: str) -> list[dict]:
    """Read one leaderboard TSV with numeric columns coerced."""
    rows = list(csv.DictReader((DATA / name).open(encoding="utf-8"), delimiter="\t"))
    for r in rows:
        for k in ("score", "chrf", "exact_match", "explanation_rate"):
            r[k] = float(r[k])
        r["rank"] = int(r["rank"])
    return rows


def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson r, stdlib only."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else 0.0


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rho via average ranks."""

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for t in range(i, j + 1):
                out[order[t]] = avg
            i = j + 1
        return out

    return pearson(ranks(xs), ranks(ys))


def sd(v: list[float]) -> float:
    """Sample standard deviation."""
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def report_pairs(real: list[Pair]) -> tuple[list[float], list[float]]:
    """Print every matched pair and return (public scores, deltas)."""
    print("MATCHED PAIRS (matched submission timestamp), sorted by public score")
    print(f"{'team':22s} {'pub':>7s} {'priv':>7s} {'d':>7s} {'dEM%':>7s} {'dchrF%':>7s} rank")
    deltas: list[float] = []
    pubs: list[float] = []
    for a, b in sorted(real, key=lambda t: -t[0]["score"]):
        d = b["score"] - a["score"]
        dem = (b["exact_match"] / a["exact_match"] - 1) * 100 if a["exact_match"] else float("nan")
        dch = (b["chrf"] / a["chrf"] - 1) * 100 if a["chrf"] else float("nan")
        deltas.append(d)
        pubs.append(a["score"])
        print(
            f"{a['id']:22s} {a['score']:7.4f} {b['score']:7.4f} {d:+7.4f} "
            f"{dem:+7.1f} {dch:+7.1f} {a['rank']:>3d}->{b['rank']:<3d}"
        )
    print(f"\nmean delta {sum(deltas) / len(deltas):+.4f} | sd {sd(deltas):.4f}")
    print(f"  range {min(deltas):+.4f} .. {max(deltas):+.4f}")
    return pubs, deltas


def report_shrinkage(pubs: list[float], deltas: list[float]) -> None:
    """Regression toward the mean: association between public scores and subsequent changes."""
    n = len(pubs)
    mx, my = sum(pubs) / n, sum(deltas) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(pubs, deltas, strict=True))
    slope = cov / sum((x - mx) ** 2 for x in pubs)
    print(f"\ncorr(public score, delta) = {pearson(pubs, deltas):+.3f}  slope {slope:+.3f}")
    print("  negative => the higher a system scored publicly, the more it lost privately:")
    ter = sorted(zip(pubs, deltas, strict=True), key=lambda t: -t[0])
    third = max(1, len(ter) // 3)
    chunks = (
        ("top third", ter[:third]),
        ("middle", ter[third : 2 * third]),
        ("bottom third", ter[2 * third :]),
    )
    for label, chunk in chunks:
        pm = sum(x for x, _ in chunk) / len(chunk)
        dm = sum(y for _, y in chunk) / len(chunk)
        print(f"    {label:13s} public mean {pm:.4f} -> mean delta {dm:+.4f}")


def report_metric_split(real: list[Pair]) -> None:
    """Report changes in exact match and chrF separately."""
    dem = [b["exact_match"] - a["exact_match"] for a, b in real if a["exact_match"]]
    dch = [b["chrf"] - a["chrf"] for a, b in real if a["chrf"]]
    rem = [b["exact_match"] / a["exact_match"] - 1 for a, b in real if a["exact_match"]]
    rch = [b["chrf"] / a["chrf"] - 1 for a, b in real if a["chrf"]]
    em_rel = sum(rem) / len(rem) * 100
    ch_rel = sum(rch) / len(rch) * 100
    print(f"\nEM   : mean delta {sum(dem) / len(dem):+.4f} ({em_rel:+.1f}% rel)")
    print(f"chrF : mean delta {sum(dch) / len(dch):+.4f} ({ch_rel:+.1f}% rel)")


def report_ranks(real: list[Pair], pub: list[dict], priv: list[dict]) -> list[str]:
    """Rank instability, and whether the public top-10 survives. Returns private team order."""
    rho = spearman([float(a["rank"]) for a, _ in real], [float(b["rank"]) for _, b in real])
    print(f"\nspearman(public rank, private rank) over matched pairs = {rho:+.3f}")
    for a, b in sorted(real, key=lambda t: -abs(t[0]["rank"] - t[1]["rank"]))[:5]:
        d = b["score"] - a["score"]
        print(f"  {a['id']:22s} public #{a['rank']:<3d} -> private #{b['rank']:<3d} ({d:+.4f})")

    best: dict[str, float] = {}
    for row in priv:
        best[row["id"]] = max(best.get(row["id"], 0.0), row["score"])
    priv_order = [t for t, _ in sorted(best.items(), key=lambda kv: -kv[1])]
    pub_order = [r["id"] for r in sorted(pub, key=lambda r: -r["score"])]
    pub10, priv10 = set(pub_order[:10]), set(priv_order[:10])
    print(f"\npublic top-10 teams still in the private top-10: {len(pub10 & priv10)}/10")
    print(f"  dropped out : {sorted(pub10 - priv10)}")
    print(f"  moved in    : {sorted(priv10 - pub10)}")
    return priv_order


def report_delivery(priv: list[dict]) -> None:
    """Arms that produced relevant characters but zero exact matches on the run that counted."""
    zero = [r for r in priv if r["exact_match"] == 0 and r["chrf"] > 0.05]
    print(f"\nprivate arms with EM exactly 0 but chrF > 0.05: {len(zero)}")
    for r in sorted(zero, key=lambda r: -r["chrf"]):
        expl = r["explanation_rate"]
        print(f"  {r['id']:22s} sub{r['submission']} chrF {r['chrf']:.4f} expl {expl:.0f}%")


def report_determinism(pub: list[dict]) -> None:
    """List equal score triples; these do not establish artifact identity."""
    dupes: dict[tuple, list[str]] = {}
    for row in pub:
        dupes.setdefault((row["score"], row["chrf"], row["exact_match"]), []).append(row["id"])
    print("\nequal public triples (score, chrF, EM) shared by >1 account:")
    for k, v in dupes.items():
        if len(v) > 1 and k[0] > 0:
            print(f"  {k} <- {v}")


def report_selection(pairs: list[Pair], pubs: list[float], deltas: list[float]) -> None:
    """Compare changes against one baseline descriptively; no causal attribution."""
    base = next(((p, b) for p, b in pairs if p["id"] == "Baseline-Qwen2.5-14B-AWQ"), None)
    if base is None:
        print("Selection comparison skipped: reference baseline is absent")
        return
    shift = base[1]["score"] - base[0]["score"]
    top = [d for x, d in zip(pubs, deltas, strict=True) if x >= 0.18]
    if not top:
        print("Selection comparison skipped: no public scores >= 0.18")
        return
    top_mean = sum(top) / len(top)
    print(f"\nunselected control (organiser 14B-AWQ, one submission): delta {shift:+.4f}")
    print(f"publicly top arms (public >= 0.18, n={len(top)}): mean {top_mean:+.4f}")
    print(f"  => descriptive difference from baseline (not a causal estimate): {top_mean - shift:+.4f}")


def report_us(priv: list[dict], pub: list[dict], priv_order: list[str]) -> None:
    """Our two finals, and the team standing on each board."""
    if US not in priv_order or not any(r["id"] == US for r in pub):
        return
    print("\nOURS")
    for row in [r for r in priv if r["id"] == US]:
        print(
            f"  private #{row['rank']} sub{row['submission']} {row['score']:.4f} "
            f"(EM {row['exact_match']:.4f} chrF {row['chrf']:.4f}) "
            f"expl {row['explanation_rate']:.0f}%  [{row['submission_datetime']}]"
        )
    pub_order = [r["id"] for r in sorted(pub, key=lambda r: -r["score"])]
    print(
        f"  team standing: public #{pub_order.index(US) + 1}, "
        f"private #{priv_order.index(US) + 1} of {len(priv_order)} teams"
    )
    print(f"  teams ahead of us privately: {priv_order[: priv_order.index(US)]}")


def main() -> None:
    """Print the matched-pair analysis."""
    pub, priv = load("public.tsv"), load("private.tsv")
    by_ts = {r["submission_datetime"]: r for r in priv}
    pairs = [(p, by_ts[p["submission_datetime"]]) for p in pub if p["submission_datetime"] in by_ts]
    real = [(a, b) for a, b in pairs if not a["id"].startswith(BASELINES)]
    print(f"public rows {len(pub)} | private rows {len(priv)} | matched by timestamp {len(pairs)}")
    print(f"matched excluding organiser baselines/test rows: {len(real)}\n")

    pubs, deltas = report_pairs(real)
    report_shrinkage(pubs, deltas)
    report_metric_split(real)
    priv_order = report_ranks(real, pub, priv)
    report_delivery(priv)
    report_determinism(pub)
    report_selection(pairs, pubs, deltas)

    clean = [
        b["score"] - a["score"] for a, b in real if b["exact_match"] > 0 or a["exact_match"] == 0
    ]
    print(f"\nsd of delta, all {len(deltas)} pairs           : {sd(deltas):.4f}")
    print(f"sd of delta, excluding EM->0 rows ({len(clean)}) : {sd(clean):.4f}")
    report_us(priv, pub, priv_order)


if __name__ == "__main__":
    main()
