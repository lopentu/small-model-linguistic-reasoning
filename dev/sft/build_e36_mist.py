"""Format source-filtered multilingual instruction data for the solver answer format. See docs/training.md."""

from __future__ import annotations

import argparse
import os
import json
import random
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "dev" / "sft" / "dataset"
HF_DATA = os.environ.get("DATASET_REPO", "")
MIST_REPO = "alyxhou00/wmt26-mist-v2"
MIST_REVISION = os.environ.get("MIST_REVISION", "")

# Historical source allowlist; check upstream terms before obtaining data.
ALLOWED_SOURCES = {
    "facebook/belebele": "cc-by-sa-4.0",
    "CohereLabs/aya_dataset": "apache-2.0",
    "copenlu/answerable_tydiqa": "apache-2.0",
}

SYS_A = (
    "You are an expert linguist solving a linguistics olympiad puzzle. Deduce the "
    "rules from the data given, then output ONLY a JSON list of the answers to the "
    "asked items, in the order asked — no explanation, no markdown fences, just the "
    "JSON list of strings."
)


def mist_rows(ratio_cap: int, n_iol: int, seed: int = 20260726) -> list[dict]:
    """Load, license-filter, and format MIST rows into our direct-JSON schema."""
    from huggingface_hub import hf_hub_download

    if not MIST_REVISION:
        raise SystemExit("Set MIST_REVISION to the dataset commit you intend to use; the historical revision is not recorded")
    path = hf_hub_download(MIST_REPO, "train_v2.jsonl", repo_type="dataset", revision=MIST_REVISION)
    raw = [json.loads(line) for line in Path(path).open(encoding="utf-8")]
    kept = [r for r in raw if r.get("source") in ALLOWED_SOURCES]
    dropped = Counter(r.get("source") for r in raw if r.get("source") not in ALLOWED_SOURCES)
    print(f"[e36] MIST rows: {len(raw)} -> {len(kept)} after license filter")
    for src, n in dropped.most_common():
        print(f"[e36]   dropped {n:5d} from {src} (license not cleared)")

    out = []
    for i, r in enumerate(kept):
        ans = str(r.get("output", "")).strip()
        user = str(r.get("input", "")).strip()
        # 2500 chars, not 6000: Bengali/Japanese/Korean passages tokenize 3-4x longer
        # than Latin text, and one 8k-token row aborts the whole run (the never-truncate
        # gate is correct to refuse). Long passages teach format no better than short ones.
        if not ans or not user or len(user) > 2500:
            continue
        out.append(
            {
                "messages": [
                    {"role": "system", "content": SYS_A},
                    {"role": "user", "content": user},
                ],
                "completion": json.dumps([ans], ensure_ascii=False),
                "problem_id": f"mist-{r.get('source', '?').split('/')[-1]}-{i}",
                "tier": "direct",
                "exemplar": "none",
                "in_linguini": "False",
            }
        )
    rng = random.Random(seed)  # noqa: S311 — data shuffling, not cryptography
    rng.shuffle(out)
    cap = ratio_cap * n_iol
    if len(out) > cap:
        print(f"[e36] capping MIST at {cap} rows ({ratio_cap}:1 vs {n_iol} IOL rows)")
        out = out[:cap]
    return out


def main() -> int:
    """Build the mixture and report its composition."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=int, default=4, help="MIST rows per IOL row")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    iol_path = os.environ.get("IOL_TRAIN_PATH")
    if not iol_path:
        raise SystemExit("Set IOL_TRAIN_PATH to your direct-answer training JSONL")
    iol = [json.loads(line) for line in Path(iol_path).open(encoding="utf-8")]
    print(f"[e36] IOL direct rows: {len(iol)}")

    mist = mist_rows(args.ratio, len(iol))
    rows = iol + mist
    random.Random(7).shuffle(rows)  # noqa: S311 — data shuffling

    DATASET.mkdir(parents=True, exist_ok=True)
    out = DATASET / "train_e36_mist.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[e36] wrote {out} — {len(rows)} rows ({len(iol)} IOL + {len(mist)} MIST)")
    print(f"[e36] IOL share: {100 * len(iol) / len(rows):.1f}%")

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
            commit_message=f"E36: MIST({len(mist)} permissive) + IOL direct({len(iol)})",
        )
        print(f"[e36] uploaded to {HF_DATA}/{out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
