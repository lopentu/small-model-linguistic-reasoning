# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "unsloth",
#   "trl>=0.12",
#   "datasets",
#   "huggingface_hub[hf_transfer]",
#   "wandb",
# ]
# ///
"""Supervised fine-tuning with completion-only loss and LoRA. See docs/training.md."""

from __future__ import annotations

import json
import os
import re

import torch

MODEL_ID = os.environ.get("MODEL_ID", "unsloth/gemma-4-26B-A4B-it")
DATASET_REPO = os.environ.get("DATASET_REPO", "")  # private HF dataset with train jsonl
TRAIN_FILE = os.environ.get("TRAIN_FILE", "train_a_verbatim.jsonl")
OUT_REPO = os.environ.get("OUT_REPO", "")  # private model repo for adapters ("" = no push)
R = int(os.environ.get("LORA_R", os.environ.get("R", "16")))  # HF Jobs rejects 1-char env keys
LR = float(os.environ.get("LR", "1e-4"))
EPOCHS = int(os.environ.get("EPOCHS", "6"))
MAX_LEN = int(os.environ.get("MAX_LEN", "16384"))
SEED = int(os.environ.get("SEED", "3407"))
SMOKE = os.environ.get("SMOKE", "0") == "1"
# Regularization and evaluation settings.
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.0"))  # MUST stay 0.0 for gemma fused experts
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
NEFTUNE = float(os.environ.get("NEFTUNE", "0"))  # 0 = off; 5 = unsloth-recommended noise
EVAL_FRAC = float(os.environ.get("EVAL_FRAC", "0"))  # >0: problem-level held-out eval split
EARLY_STOP = int(
    os.environ.get("EARLY_STOP", "0")
)  # >0: patience in epochs on eval_loss (needs EVAL_FRAC)
EARLY_STOP_DELTA = float(
    os.environ.get("EARLY_STOP_DELTA", "0")
)  # min improvement to reset patience
LABEL_SMOOTH = float(
    os.environ.get("LABEL_SMOOTH", "0")
)  # Disabled by default; support depends on the model and training backend.
MIN_LR_RATE = float(os.environ.get("MIN_LR_RATE", "0"))  # >0: cosine floor = rate * LR
RUN = os.environ.get("RUN_NAME", f"e23-r{R}-lr{LR:g}-{TRAIN_FILE.split('.')[0]}")


def load_rows() -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = os.environ.get("TRAIN_PATH") or hf_hub_download(DATASET_REPO, TRAIN_FILE, repo_type="dataset")
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    if SMOKE:
        rows = rows[:8]
    print(f"loaded {len(rows)} rows from {path}")
    return rows


def main() -> None:
    from unsloth import FastModel  # MUST import before trl/datasets: patches transformers

    # isort: split
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    if OUT_REPO:
        from huggingface_hub import HfApi
        from release_io import require_private_repo
        api = HfApi()
        api.create_repo(OUT_REPO, private=True, exist_ok=True)
        require_private_repo(api, OUT_REPO, repo_type="model")

    rows = load_rows()
    if not rows:
        raise SystemExit("dataset empty")
    for idx, row in enumerate(rows):  # every row, not rows[0]
        for k in ("messages", "completion", "problem_id"):
            if k not in row or not row[k]:
                raise SystemExit(f"dataset row {idx} missing/empty field {k!r}")

    model, tokenizer = FastModel.from_pretrained(
        MODEL_ID,
        max_seq_length=MAX_LEN,
        load_in_4bit=False,  # MoE + bnb-4bit unsupported (fused 3D experts); 16-bit LoRA
        dtype=None,  # auto: bf16 on A100
    )
    model = FastModel.get_peft_model(
        model,
        r=R,
        lora_alpha=2 * R,  # rsLoRA convention alpha=2r
        # 0.0 REQUIRED: peft's ParamWrapper (gemma-4 fused 3D expert params) rejects
        # dropout != 0. Epoch
        # selection by generated score is the regularizer here.
        lora_dropout=LORA_DROPOUT,
        use_rslora=True,
        # unsloth's gemma-4 defaults target attention + expert FFN and FREEZE the
        # router; do not widen to "all-linear" (would include the router)
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    # -- render must equal v2 deployment's render_chat byte-for-byte.
    # render_chat_min replicates solver/v2/chat.py:render_chat (minimal jinja env);
    # the GGUF's embedded template is this same tokenizer.chat_template string.
    def render_chat_min(template: str, messages: list, *, think: bool) -> str:
        import jinja2

        env = jinja2.Environment(loader=jinja2.BaseLoader(), autoescape=False)  # noqa: S701
        env.globals["raise_exception"] = lambda m="": (_ for _ in ()).throw(RuntimeError(m))
        env.filters.setdefault("tojson", lambda v, **_k: json.dumps(v))
        return env.from_string(template).render(
            messages=messages, add_generation_prompt=True, enable_thinking=think
        )

    def render(row: dict) -> dict:
        # BOS prepended as a string: the TRL/unsloth prompt-completion tokenizer adds
        # no special tokens while llama.cpp adds BOS
        # at deployment — the exactly-one-BOS batch assert below guards both ways
        prompt = (txt_tok.bos_token or "") + render_chat_min(
            tokenizer.chat_template, row["messages"], think=False
        )
        return {"prompt": prompt, "completion": row["completion"] + tokenizer.eos_token}

    # gemma-4 ships a multimodal Processor: positional `tokenizer(text)` lands in
    # `images` and crashes — direct tokenization must use the inner one
    txt_tok = getattr(tokenizer, "tokenizer", tokenizer)

    # eos must be the template's turn-ender or the model never learns to stop the
    # turn the way the GGUF stop-set expects (gemma-4 eos = "<turn|>")
    assert tokenizer.eos_token, "tokenizer has no eos_token"
    print(f"eos_token={tokenizer.eos_token!r} id={tokenizer.eos_token_id}")
    turn_end = "<turn|>"
    if os.environ.get("EOS_OK", "0") != "1" and tokenizer.eos_token != turn_end:
        raise SystemExit(
            f"eos_token {tokenizer.eos_token!r} != {turn_end!r} (gemma-4 turn end). "
            "Verify the template's turn-ender and set EOS_OK=1 to proceed."
        )

    eval_ds = None
    if EVAL_FRAC > 0:
        # held-out EVAL split BY PROBLEM (not by row: tier/exemplar variants of one
        # problem would leak across the split) — eval_loss curve is the overfit gauge
        import random

        # alt-trace ids ("gemini-IOL_2007_P2") name the SAME problem as "IOL_2007_P2" —
        # group by the underlying problem or the split leaks across trace sources
        def base_pid(r: dict) -> str:
            return r.get("problem_id", "").removeprefix("gemini-")

        pids = sorted({base_pid(r) for r in rows})
        rng = random.Random(SEED)  # noqa: S311 — split sampling, not crypto
        held = set(rng.sample(pids, max(1, int(len(pids) * EVAL_FRAC))))
        ev_rows = [r for r in rows if base_pid(r) in held]
        rows = [r for r in rows if base_pid(r) not in held]
        # rotation bakes ANOTHER problem's trace into `messages` as a typed exemplar
        # (mode recorded, exemplar pid not) — a held-out problem hiding inside a train
        # row's exemplar would leak its solution into training and flatter eval_loss.
        # Fingerprint = head of each held row's own problem turn; drop matching rows.
        prints = {r["messages"][-1]["content"][:200] for r in ev_rows if r.get("messages")}
        n0 = len(rows)
        rows = [
            r
            for r in rows
            if not any(
                fp in m.get("content", "")
                for m in r["messages"][:-1]
                if m.get("role") == "user"
                for fp in prints
            )
        ]
        if n0 - len(rows):
            print(f"dropped {n0 - len(rows)} train rows whose exemplar embeds a held-out problem")
        eval_ds = Dataset.from_list([render(r) for r in ev_rows])
        print(f"eval split: {len(held)} problems / {len(ev_rows)} rows held out; {len(rows)} train")
    ds = Dataset.from_list([render(r) for r in rows]).shuffle(seed=SEED)

    # cross-check our minimal render against transformers' apply_chat_template.
    # Known benign delta: HF interpolates the literal bos string where the
    # GGUF path renders bos_token empty and llama.cpp adds BOS at tokenization —
    # so equality must hold modulo one leading bos. Anything else is a real drift.
    hf_render = tokenizer.apply_chat_template(
        rows[0]["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ours = render_chat_min(tokenizer.chat_template, rows[0]["messages"], think=False)
    bos = txt_tok.bos_token or ""
    if hf_render not in (ours, bos + ours):
        stripped = hf_render.removeprefix(bos)
        i = next((k for k, (a, b) in enumerate(zip(stripped, ours)) if a != b), -1)
        raise SystemExit(f"render drift beyond BOS at char {i} — refusing to train on bad prompts")
    fp = os.environ.get("THINK_OFF_FINGERPRINT", "<|channel>thought\n<channel|>")
    if not fp.strip():
        raise SystemExit("THINK_OFF_FINGERPRINT must be non-empty")
    if fp not in ours:
        raise SystemExit(
            f"think-off fingerprint {fp!r} missing from rendered prompt — enable_thinking="
            "False did not yield the empty-thought suppressor (THINK_OFF_FINGERPRINT overrides)"
        )

    # Check token lengths and loss masking before training.
    probe = ds[0]
    p_ids = txt_tok(probe["prompt"], add_special_tokens=False)["input_ids"]
    c_ids = txt_tok(probe["completion"], add_special_tokens=False)["input_ids"]
    print(f"example 0: prompt {len(p_ids)} tok + completion {len(c_ids)} tok")
    print("prompt tail:", repr(probe["prompt"][-160:]))
    print("completion tail:", repr(probe["completion"][-160:]))
    over = sum(
        1
        for r in ds
        if len(txt_tok(r["prompt"], add_special_tokens=False)["input_ids"])
        + len(txt_tok(r["completion"], add_special_tokens=False)["input_ids"])
        > MAX_LEN - 8
    )
    if over:
        # Refuse examples that would be truncated before their final answer.
        raise SystemExit(f"{over} examples exceed MAX_LEN={MAX_LEN} — raise it, never truncate")

    args = SFTConfig(
        output_dir=f"out/{RUN}",
        run_name=RUN,
        num_train_epochs=1 if SMOKE else EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # eff. batch 8 (TM: LoRA dislikes large batches)
        learning_rate=LR,
        lr_scheduler_type="cosine_with_min_lr" if MIN_LR_RATE > 0 else "cosine",
        lr_scheduler_kwargs={"min_lr_rate": MIN_LR_RATE} if MIN_LR_RATE > 0 else None,
        warmup_ratio=0.04,
        max_grad_norm=1.0,
        weight_decay=WEIGHT_DECAY,
        label_smoothing_factor=LABEL_SMOOTH,
        logging_steps=5,
        seed=SEED,
        max_length=MAX_LEN,
        packing=False,
        completion_only_loss=True,
        save_strategy="epoch",
        bf16=True,
        push_to_hub=bool(OUT_REPO),
        hub_model_id=OUT_REPO or None,
        hub_strategy="all_checkpoints",
        hub_always_push=True,
        hub_private_repo=True,
        # External logging requires an explicit REPORT_TO setting.
        report_to=os.environ.get("REPORT_TO", "none"),
        neftune_noise_alpha=NEFTUNE or None,
        eval_strategy="epoch" if EVAL_FRAC > 0 else "no",
        eval_on_start=EVAL_FRAC > 0,  # baseline eval_loss before step 0 anchors the curve
        per_device_eval_batch_size=1,  # bs4 @ 28k ctx OOMed an A100-80G on the logits tensor
        prediction_loss_only=True,
        bf16_full_eval=EVAL_FRAC > 0,
        logging_first_step=True,
        # early stopping on eval_loss: stops after EARLY_STOP epochs without improvement;
        # all epoch checkpoints still push to hub, so the probe can scan them regardless
        load_best_model_at_end=EARLY_STOP > 0 and EVAL_FRAC > 0,
        metric_for_best_model="eval_loss" if EARLY_STOP > 0 and EVAL_FRAC > 0 else None,
        greater_is_better=False if EARLY_STOP > 0 and EVAL_FRAC > 0 else None,
    )
    assert getattr(args, "max_length", None) == MAX_LEN, "SFTConfig.max_length ignored (renamed?)"

    # Verify that router parameters remain frozen.
    pcfg = model.peft_config["default"]
    tmods = pcfg.target_modules
    print(f"LoRA target_modules: {sorted(tmods) if not isinstance(tmods, str) else tmods}")
    routerish = [
        n
        for n, p in model.named_parameters()
        if p.requires_grad and ("router" in n.lower() or re.search(r"\bgate\b(?!_proj)", n.lower()))
    ]
    if routerish:
        raise SystemExit(f"MoE router params trainable — recipe forbids this: {routerish[:5]}")

    callbacks = []
    if EARLY_STOP > 0 and EVAL_FRAC > 0:
        from transformers import EarlyStoppingCallback

        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOP, early_stopping_threshold=EARLY_STOP_DELTA
            )
        )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=callbacks or None,
    )

    # decoded-label check on one collated batch: prompt tokens must be -100
    batch = trainer.data_collator([trainer.train_dataset[0]])
    labels = batch["labels"][0]
    if txt_tok.bos_token_id is not None:
        n_bos = int((batch["input_ids"][0] == txt_tok.bos_token_id).sum())
        if n_bos != 1:
            raise SystemExit(f"expected exactly 1 BOS in the collated batch, got {n_bos}")
    n_masked = int((labels == -100).sum())
    n_train = int((labels != -100).sum())
    print(f"label check: {n_masked} masked / {n_train} trained tokens")
    first_trained = tokenizer.decode([t for t in labels.tolist() if t != -100][:30])
    print("first trained tokens:", repr(first_trained))
    if n_train == 0 or n_masked < len(p_ids) // 2:
        raise SystemExit("label masking looks wrong — refusing to burn GPU time")

    trainer.train()
    trainer.save_model()
    # NO final push_to_hub(): checkpoints are already pushed per epoch
    # (hub_strategy="all_checkpoints"); the SELECTED checkpoint is chosen locally by
    # generated CLEAN-11 score (dataset/clean11_ids.txt) — never CE, never Search.
    print(f"DONE {RUN}: final loss {trainer.state.log_history[-1].get('train_loss', 'NA')}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("WANDB_PROJECT", "small-model-linguistic-reasoning")
    torch.manual_seed(SEED)
    main()
