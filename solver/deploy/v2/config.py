"""Environment-variable configuration (same contract as v1, plus v2 knobs)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """All solver knobs; construct via Config.from_env()."""

    model_dir: str
    gguf_path: str
    test_csv: str
    n_samples: int
    max_new: int
    batch: int
    n_gpu_layers: int
    think: bool
    fewshot: bool
    time_limit_s: float
    transcript_path: str  # "" disables transcript/resume
    max_refine_rounds: int
    # CLI-driver backend (LlamaCliBackend) — all empty/0 => bindings backend is used instead.
    llama_cli: str  # path to the llama-cli binary; "" => use llama-cpp-python bindings
    model_draft: str  # MTP drafter GGUF; "" => no speculative decoding
    spec_draft_n: int  # --spec-draft-n-max (MTP lookahead); ignored without model_draft
    reasoning_budget: int
    llama_ctx: int  # -c for the CLI backend; 4096 default (26B+KV must fit the T4's 16 GB)
    # Parallel-vote driver backend (LlamaParallelBackend) — BACKEND=parallel activates it.
    backend: str  # "" = auto (cli > bindings > transformers); "parallel" = JSONL batch driver
    llama_parallel: str  # path to the llama-iolbatch binary
    slots: int  # driver -np: max concurrent decode streams (KV is a shared unified pool)
    vote_temp: float  # sampling temperature for non-greedy votes (parallel backend)
    vote_top_p: float  # top-p for non-greedy votes (parallel backend)
    vote_weight: str
    vote_lp_t: float  # temperature on the weights: exp(mean_logprob / T)
    vote_floor: float  # ignore votes below this confidence (0 = keep every vote)
    vote_shuffle: bool
    vote_shuffle_groups: int
    repeat_penalty: float = 1.0

    @classmethod
    def from_env(cls) -> Config:
        """Read configuration from the process environment."""
        gguf_path = os.environ.get("GGUF_PATH", "")
        backend = os.environ.get("BACKEND", "")
        slots = int(os.environ.get("SLOTS", "16"))
        if backend == "parallel":
            # chat() chunks feed the driver, which decodes up to `slots` streams at once
            batch = int(os.environ.get("BATCH", str(slots)))
        else:
            batch = 1 if gguf_path else int(os.environ.get("BATCH", "8"))
        return cls(
            model_dir=os.environ.get("MODEL_DIR", "."),
            gguf_path=gguf_path,
            test_csv=os.environ.get("TEST_CSV", "/tmp/data/test.csv"),
            n_samples=int(os.environ.get("N_SAMPLES", "4")),
            max_new=int(os.environ.get("MAX_NEW", "1024")),
            batch=batch,
            n_gpu_layers=int(os.environ.get("N_GPU_LAYERS", "-1")),
            think=os.environ.get("THINK", "0") == "1",
            fewshot=os.environ.get("FEWSHOT", "1") == "1",
            time_limit_s=float(os.environ.get("TIME_LIMIT_S", "1800")),
            transcript_path=os.environ.get("TRANSCRIPT", "transcript.jsonl"),
            max_refine_rounds=int(os.environ.get("REFINE_ROUNDS", "1")),
            llama_cli=os.environ.get("LLAMA_CLI", ""),
            model_draft=os.environ.get("MODEL_DRAFT", ""),
            spec_draft_n=int(os.environ.get("SPEC_DRAFT_N", "1")),
            reasoning_budget=int(os.environ.get("REASONING_BUDGET", "-1")),
            llama_ctx=int(os.environ.get("LLAMA_CTX", "4096")),
            backend=backend,
            llama_parallel=os.environ.get("LLAMA_PARALLEL", ""),
            slots=slots,
            vote_temp=float(os.environ.get("VOTE_TEMP", "1.0")),
            vote_top_p=float(os.environ.get("VOTE_TOP_P", "0.95")),
            vote_weight=os.environ.get("VOTE_WEIGHT", "none"),
            vote_lp_t=float(os.environ.get("VOTE_LP_T", "1.0")),
            vote_floor=float(os.environ.get("VOTE_FLOOR", "0")),
            vote_shuffle=os.environ.get("VOTE_SHUFFLE", "0") == "1",
            vote_shuffle_groups=int(os.environ.get("VOTE_SHUFFLE_GROUPS", "3")),
            repeat_penalty=float(os.environ.get("REPEAT_PENALTY", "1.0")),
        )
