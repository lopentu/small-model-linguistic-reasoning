"""Embed user-supplied grammar chunks for local retrieval."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-layers", type=int, default=-1)
    ap.add_argument("--n-ctx", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    from llama_cpp import Llama

    texts = [json.loads(line)["text"] for line in Path(args.chunks).open(encoding="utf-8")]
    print(f"loaded {len(texts):,} chunks", flush=True)

    llm = Llama(
        args.model,
        embedding=True,
        n_gpu_layers=args.gpu_layers,
        n_ctx=args.n_ctx,
        n_batch=args.n_ctx,
        verbose=False,
    )

    def norm(mat: np.ndarray) -> np.ndarray:
        return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)

    vecs: list[np.ndarray] = []
    t0 = time.time()
    for i in range(0, len(texts), args.batch):
        out = llm.embed(texts[i : i + args.batch])  # list[list[float]] (one vec per doc)
        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim == 3:  # some builds return per-token; mean-pool
            arr = arr.mean(axis=1)
        vecs.append(norm(arr).astype(np.float16))
        if (i // args.batch) % 20 == 0:
            done = i + args.batch
            rate = done / max(1e-9, time.time() - t0)
            print(f"  {min(done, len(texts)):>8,}/{len(texts):,}  {rate:.0f} chunks/s", flush=True)

    mat = np.concatenate(vecs, axis=0)
    assert mat.shape[0] == len(texts), f"row mismatch {mat.shape[0]} vs {len(texts)}"
    np.save(args.out, mat)
    print(f"DONE: {mat.shape} {mat.dtype} -> {args.out} ({mat.nbytes/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
