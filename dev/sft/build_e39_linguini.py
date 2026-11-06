"""Build an experimental training mixture from user-supplied inputs. See docs/training.md."""

from __future__ import annotations

import argparse
import os
import ast
import csv
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "dev" / "mock-eval" / "fixtures"
SPLIT = REPO / "dev" / "mock-eval" / "split.json"
DATASET = REPO / "dev" / "sft" / "dataset"
HF_DATA = os.environ.get("DATASET_REPO", "")

SYS_A = (
    "You are an expert linguist solving a linguistics olympiad puzzle. Deduce the "
    "rules from the data given, then output ONLY a JSON list of the answers to the "
    "asked items, in the order asked — no explanation, no markdown fences, just the "
    "JSON list of strings."
)


def linguini_rows() -> tuple[list[dict], int]:
    """Direct-JSON rows for the SEARCH fold only; returns (rows, holdout_skipped)."""
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    search, holdout = set(split["search"]), set(split["holdout"])
    rows = {r["id"]: r for r in csv.DictReader((FIXTURES / "test.csv").open(encoding="utf-8"))}
    gold = {
        r["id"]: ast.literal_eval(r["answer"])
        for r in csv.DictReader((FIXTURES / "answers.csv").open(encoding="utf-8"))
    }
    out, skipped = [], 0
    for rid, r in rows.items():
        if rid in holdout:
            skipped += 1
            continue
        if rid not in search or rid not in gold:
            continue
        answers = gold[rid]
        # an item may carry accepted ALTERNATIVES; train on the first (canonical) one,
        # VERBATIM — inclusion/exclusion is the only quality lever we allow ourselves
        flat = [str(a[0] if isinstance(a, list) and a else a) for a in answers]
        if not flat or not all(f.strip() for f in flat):
            continue
        user = f"{r['context']}\n\n{r['query']}"
        out.append(
            {
                "messages": [
                    {"role": "system", "content": SYS_A},
                    {"role": "user", "content": user},
                ],
                "completion": json.dumps(flat, ensure_ascii=False),
                "problem_id": f"linguini-{rid}",
                "tier": "direct",
                "exemplar": "none",
                # marks these rows as coming from the eval fixture, so any future
                # scoring on Linguini knows it is scoring on trained data
                "in_linguini": "True",
            }
        )
    return out, skipped


def main() -> int:
    """Build train_e39_linguini.jsonl = supplied direct rows + the SEARCH fold."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    if not os.environ.get("IOL_TRAIN_PATH"):
        raise SystemExit("Set IOL_TRAIN_PATH to your direct-answer training JSONL")


    base = [
        json.loads(line)
        for line in Path(
            os.environ["IOL_TRAIN_PATH"]
        ).open(encoding="utf-8")
    ]
    ling, skipped = linguini_rows()
    print(f"[e39] existing direct rows : {len(base)}")
    print(f"[e39] Linguini SEARCH rows  : {len(ling)}  (holdout rows skipped: {skipped})")
    items = sum(len(json.loads(r["completion"])) for r in ling)
    print(f"[e39] added answer items    : {items}")

    rows = base + ling
    random.Random(20260726).shuffle(rows)  # noqa: S311 — data shuffling
    DATASET.mkdir(parents=True, exist_ok=True)
    out = DATASET / "train_e39_linguini.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[e39] wrote {out} — {len(rows)} rows ({100 * len(ling) / len(rows):.0f}% new)")

    if args.upload:
        if not HF_DATA:
            raise SystemExit("Set DATASET_REPO to an explicit upload destination")
        from huggingface_hub import HfApi

        from release_io import require_private_repo

        api = HfApi()
        require_private_repo(api, HF_DATA, repo_type="dataset")
        api.upload_file(
            path_or_fileobj=str(out),
            path_in_repo=out.name,
            repo_id=HF_DATA,
            repo_type="dataset",
            commit_message=f"E39: {len(base)} direct + {len(ling)} Linguini SEARCH rows",
        )
        print(f"[e39] uploaded to {HF_DATA}/{out.name} (PRIVATE repo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
