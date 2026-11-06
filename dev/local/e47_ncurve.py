"""Reaggregate saved generations into greedy-led and sampled-only voting curves. See docs/analysis.md."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import zlib
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "solver/deploy"))
sys.path.insert(0, str(REPO / "dev"))

import score_local
from v2.pipeline import ProblemState, aggregate

N_VALUES = (1, 2, 3, 5, 7, 9, 11)
ANSWERS = REPO / "dev/mock-eval/fixtures/answers.csv"
TEST = REPO / "dev/mock-eval/fixtures/test.csv"


def load_rows() -> dict[str, dict[str, str]]:
    """Fixture rows by id — context/query/task_type drive k and the fallbacks."""
    with TEST.open(encoding="utf-8") as f:
        return {r["id"]: dict(r) for r in csv.DictReader(f)}


def load_pool(transcript: Path) -> dict[str, list[dict]]:
    """Rebuild {id: [solve sample, ...]} in generation order, keeping logprob metadata."""
    pool: dict[str, list[dict]] = defaultdict(list)
    slots: dict[str, list[dict]] = defaultdict(list)
    for line in transcript.read_text(encoding="utf-8").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("event") != "sample" or "text" not in ev:
            continue
        rid = str(ev["id"])
        if ev.get("kind") == "finalize":
            slot = ev.get("slot")
            if not isinstance(slot, int) or not 0 <= slot < len(slots[rid]):
                raise ValueError(f"Finalize event has no preceding sample: {rid}, {slot}")
            slots[rid][slot]["text"] += "\n" + ev["text"]
        else:
            sample = {"kind": ev.get("kind"), "text": ev["text"], "meta": ev.get("meta"), "perm": ev.get("perm")}
            slots[rid].append(sample)
            if ev.get("kind") == "solve":
                pool[rid].append(sample)

    return pool


def write_vote(
    path: Path,
    rows: dict[str, dict[str, str]],
    pool: dict[str, list[dict]],
    pick: dict[str, list[dict]],
) -> None:
    """Write one submission.csv from a pre-selected sample subset per row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "pred", "explanation"])
        w.writeheader()
        for rid, row in rows.items():
            st = ProblemState.from_row(row)
            st.samples = pick.get(rid, pool.get(rid, [])[:1])
            pred, expl = aggregate(st, mode="none", lp_t=1.0, floor=0.0)
            w.writerow(
                {"id": rid, "pred": json.dumps(pred, ensure_ascii=False), "explanation": expl}
            )


def subsets(
    pool: dict[str, list[dict]], n: int, rng: random.Random | None
) -> dict[str, list[dict]]:
    """Per row, the first n samples (rng=None) or a uniform random n-subset."""
    out: dict[str, list[dict]] = {}
    for rid, samples in pool.items():
        k = min(n, len(samples))
        out[rid] = samples[:k] if rng is None else rng.sample(samples, k)
    return out


def geo_of(path: Path, ids: set[str] | None) -> dict:
    """Score one submission on the fold; returns the scorer's dict of record."""
    return score_local.score(str(path), str(ANSWERS), ids)


def mean_sd(v: list[float]) -> tuple[float, float]:
    """Mean and sample sd of a list (sd=0 for a single element)."""
    m = sum(v) / len(v)
    if len(v) < 2:
        return m, 0.0
    return m, (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def main() -> int:
    """Run the N sweep and print the curve with paired CIs against N=1."""
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--split", default="search", choices=["search", "holdout", "all"])
    a = ap.parse_args()

    # plain majority is the whole point; a stray env knob would silently change the rule
    os.environ["ASSIGNMENT"] = "0"
    os.environ["DETLANES"] = "0"
    os.environ["VOTE_WEIGHT"] = "none"

    dump = Path(a.dump_dir)
    pool = load_pool(dump / "transcript.jsonl")
    rows = load_rows()
    ids = None if a.split == "all" else score_local._fold_ids(a.split)  # noqa: SLF001
    problem_of = score_local._problem_of()  # noqa: SLF001
    missing = set(pool) - set(problem_of or {})
    if missing:
        print(f"WARNING: {len(missing)} IDs lack problem mappings; bootstrap falls back to rows")
    depth = sorted(len(v) for v in pool.values())
    if not depth:
        raise SystemExit("No solve samples found")
    print(f"pool: {len(pool)} rows | samples/row min {depth[0]} median {depth[len(depth) // 2]}")
    print(f"split {a.split} ({'all' if ids is None else len(ids)} ids) | draws per N {a.draws}\n")

    out = dump / "ncurve"
    base_raw: list | None = None
    print(
        f"{'N':>3s}  {'first-N geo':>11s} {'EM':>7s} {'chrF':>7s}  "
        f"{'mean-of-R geo':>13s} {'sd':>6s}  {'paired d vs N=1':>26s}"
    )
    for n in N_VALUES:
        p = out / f"first{n}.csv"
        write_vote(p, rows, pool, subsets(pool, n, None))
        first = geo_of(p, ids)
        if base_raw is None:
            base_raw = first["per_problem_raw"]

        geos = []
        for d in range(a.draws):
            pd_ = out / f"rand{n}_{d}.csv"
            write_vote(pd_, rows, pool, subsets(pool, n, random.Random(1000 * n + d)))
            geos.append(geo_of(pd_, ids)["geometric_mean"])
        m, s = mean_sd(geos)

        if n == 1:
            ci = "baseline"
        else:
            med, lo, hi = score_local.bootstrap_paired_delta(
                first["per_problem_raw"], base_raw, problem_of, n_boot=4000, seed=7
            )
            real = "0 outside CI" if (lo > 0 or hi < 0) else "0 inside CI"
            ci = f"{med:+.4f} [{lo:+.4f},{hi:+.4f}] {real}"
        print(
            f"{n:>3d}  {first['geometric_mean']:11.4f} {first['exact_match']:7.4f} "
            f"{first['chrf']:7.4f}  {m:13.4f} {s:6.4f}  {ci:>26s}"
        )

    print(f"\nsubmissions written to {out}")
    print("first-N uses saved sample order; mean-of-R averages sampled subsets")
    stage2(rows, pool, ids, problem_of, out, a.draws)
    return 0


def stage2(
    rows: dict[str, dict[str, str]],
    pool: dict[str, list[dict]],
    ids: set[str] | None,
    problem_of: dict[str, str] | None,
    out: Path,
    draws: int,
) -> None:
    """Separate 'voting helps' from 'greedy helps'.

    `pipeline.build_passes` runs `SolvePass(0, greedy=True)` and every later sample at
    the configured sampling temperature. This analysis requires greedy-first transcripts.
    A vote pool therefore mixes two populations, and "majority@N vs N=1" silently compares
    greedy-plus-sampled against greedy. These three families separate them.
    """
    print("\n\nSTAGE 2 — greedy vs sampled (requires sample 0 greedy and later samples stochastic)")

    def measure(label: str, pick: object, n_draws: int) -> tuple[str, float, list]:
        geos: list[float] = []
        raw: list | None = None
        for d in range(n_draws):
            # zlib.crc32, NOT hash(): PYTHONHASHSEED randomises str hashing per process, so
            # hash(label) gave a different draw on every run and the table moved in the third
            # decimal between two runs of the SAME committed script.
            rng = random.Random(zlib.crc32(label.encode()) % 99991 + d)
            p = out / "stage2.csv"
            write_vote(p, rows, pool, {rid: pick(s, rng) for rid, s in pool.items()})  # type: ignore[operator]
            r = geo_of(p, ids)
            geos.append(r["geometric_mean"])
            if raw is None:
                raw = r["per_problem_raw"]
        return label, sum(geos) / len(geos), raw or []

    fams: list[tuple[str, float, list]] = [
        measure("greedy only (sample 0)", lambda s, _r: s[:1], 1)
    ]
    fams.extend(
        measure(
            f"sampled-only majority@{n}",
            lambda s, r, n=n: r.sample(s[1:], min(n, len(s) - 1)) if len(s) > 1 else s[:1],
            draws,
        )
        for n in (1, 3, 5, 7, 11)
    )
    fams.extend(
        measure(
            f"greedy + sampled@{n}",
            lambda s, r, n=n: ([s[0], *r.sample(s[1:], min(n - 1, len(s) - 1))] if s else []),
            draws,
        )
        for n in (3, 5, 11)
    )

    base = fams[0]
    print(f"{'variant':28s} {'mean geo':>9s}  {'draw-0 paired d vs greedy-only':>34s}")
    for label, m, raw in fams:
        if label == base[0]:
            print(f"{label:28s} {m:9.4f}  {'baseline':>34s}")
            continue
        med, lo, hi = score_local.bootstrap_paired_delta(raw, base[2], problem_of, 4000, 7)
        tag = "0 outside CI" if (lo > 0 or hi < 0) else "0 inside CI"
        print(f"{label:28s} {m:9.4f}  {med:+.4f} [{lo:+.4f},{hi:+.4f}] {tag:>5s}")



if __name__ == "__main__":
    sys.exit(main())
