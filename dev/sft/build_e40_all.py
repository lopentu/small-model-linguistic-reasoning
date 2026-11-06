"""Build an experimental training mixture from user-supplied inputs. See docs/training.md."""

from __future__ import annotations

import argparse
import os
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dev" / "sft"))
DATASET = REPO / "dev" / "sft" / "dataset"
HF_DATA = os.environ.get("DATASET_REPO", "")


def main() -> int:
    """Build the union mixture and report its composition."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--mist-ratio", type=int, default=2, help="MIST rows per ON-TASK row")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    if not os.environ.get("IOL_TRAIN_PATH"):
        raise SystemExit("Set IOL_TRAIN_PATH to your direct-answer training JSONL")

    from build_e36_mist import mist_rows
    from build_e39_linguini import linguini_rows

    ours = [
        json.loads(line)
        for line in Path(
            os.environ["IOL_TRAIN_PATH"]
        ).open(encoding="utf-8")
    ]
    ling, skipped = linguini_rows()
    on_task = len(ours) + len(ling)
    mist = mist_rows(args.mist_ratio, on_task)

    rows = ours + ling + mist
    random.Random(20260726).shuffle(rows)  # noqa: S311 — data shuffling
    print(f"[e40] supplied direct rows : {len(ours)}")
    print(f"[e40] Linguini SEARCH  : {len(ling)}  (holdout skipped: {skipped})")
    print(f"[e40] MIST (permissive): {len(mist)}")
    print(f"[e40] TOTAL            : {len(rows)}  on-task share {100 * on_task / len(rows):.0f}%")

    DATASET.mkdir(parents=True, exist_ok=True)
    out = DATASET / "train_e40_all.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[e40] wrote {out}")

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
            commit_message=f"E40: {len(ours)} ours + {len(ling)} Linguini + {len(mist)} MIST",
        )
        print(f"[e40] uploaded to {HF_DATA}/{out.name} (PRIVATE)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
