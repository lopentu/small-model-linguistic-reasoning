# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "unsloth",
#   "cmake",
#   "ninja",
#   "torch",
#   "transformers",
#   "peft",
#   "safetensors",
#   "sentencepiece",
#   "protobuf",
#   "gguf",
#   "numpy",
#   "huggingface_hub[hf_transfer]",
# ]
# ///
"""Merge a LoRA adapter and verify changes in its target tensors before export. The Unsloth loader is the default to match the training model structure."""

from __future__ import annotations

import os
import subprocess
import sys

BASE_ID = os.environ.get("BASE_ID", "unsloth/Qwen3.5-4B")
ADAPTER_REPO = os.environ["ADAPTER_REPO"]
CKPT = os.environ.get("CKPT", "")  # "" = repo-root adapter
OUT_REPO = os.environ.get("OUT_REPO", "")
UPLOAD = os.environ.get("UPLOAD", "0") == "1"
QUANT = os.environ.get("QUANT", "Q8_0")  # Q8_0 for the 4B: no memory pressure on a T4
LLAMA_TAG = os.environ.get("LLAMA_TAG", "b9964")
# how many of the adapter's target tensors must visibly move for the merge to count
MIN_CHANGED_FRACTION = float(os.environ.get("MIN_CHANGED_FRACTION", "0.9"))
# "unsloth" merges through the SAME loader that trained the adapter, which is the
# only path that handles Qwen3.5-4B's hybrid linear_attn modules — a vanilla
# AutoModelForCausalLM + key remap merged only 51.6% of targets and left every
# linear_attn.{in_proj_qkv,in_proj_z,out_proj} untouched.
MERGE_VIA = os.environ.get("MERGE_VIA", "unsloth")



def prepare_upload(api):
    if not UPLOAD:
        return
    from release_io import require_private_repo
    if not OUT_REPO:
        raise SystemExit("UPLOAD=1 requires OUT_REPO")
    api.create_repo(OUT_REPO, private=True, exist_ok=True)
    require_private_repo(api, OUT_REPO, repo_type="model")


def upload(api, kind, **kwargs):
    if UPLOAD:
        prepare_upload(api)
        getattr(api, "upload_" + kind)(**kwargs)
        print(f"Uploaded to {OUT_REPO}", flush=True)


def sh(cmd: str) -> None:
    """Run a shell step, failing loudly."""
    print(f"+ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def load_adapter_state(path: str) -> dict:
    """Read the adapter's safetensors (or .bin) state dict."""
    from safetensors.torch import load_file

    st_path = os.path.join(path, "adapter_model.safetensors")
    if os.path.isfile(st_path):
        return load_file(st_path)
    import torch

    return torch.load(os.path.join(path, "adapter_model.bin"), map_location="cpu")


def is_vision(name: str) -> bool:
    """True for vision-tower modules, which a text-only run never trains.

    unsloth's default target regex matches `visual|vision|image|patch`, so a
    text-only fine-tune still carries LoRA on the vision tower — with lora_B
    LITERALLY ALL ZERO (verified: 96/96 zero on E39, while all 248 text modules were
    nonzero). Those cannot change any weight, so counting them in the
    weights-changed denominator made a PERFECT text merge look like 72.1%.
    """
    return ".visual." in name or "vision_tower" in name or ".visual_" in name


def base_module_names(state: dict) -> set[str]:
    """Base-model module names the adapter TRAINED, from its LoRA key names.

    Vision-tower targets are excluded — see is_vision().
    """
    names = set()
    for k in state:
        if ".lora_A" not in k and ".lora_B" not in k:
            continue
        if is_vision(k):
            continue
        # base_model.model.<path>.lora_A.weight -> <path>
        core = k.split(".lora_")[0]
        for prefix in ("base_model.model.", "base_model."):
            if core.startswith(prefix):
                core = core[len(prefix) :]
                break
        names.add(core)
    return names


def merge_with_unsloth(adapter_dir: str, targets: set[str]) -> None:
    """Merge through unsloth's own API, then verify and quantize.

    unsloth trained the adapter, so its loader is the one whose module naming and
    module TYPES line up — including Qwen3.5-4B's hybrid `linear_attn` blocks, which a
    vanilla transformers load silently declines to merge (51.6% of targets moved).
    """
    import torch
    from huggingface_hub import HfApi
    from unsloth import FastModel

    api = HfApi()
    model, tokenizer = FastModel.from_pretrained(
        BASE_ID, max_seq_length=4096, load_in_4bit=False, dtype=torch.bfloat16
    )
    before = {}
    for name, mod in model.named_modules():
        short = name.replace("base_model.model.", "")
        if short in targets and hasattr(mod, "weight"):
            before[short] = mod.weight.detach().float().clone()
    print(f"snapshotted {len(before)} target tensors before the merge", flush=True)

    from peft import PeftModel

    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()

    changed, unchanged, worst = 0, [], 0.0
    for name, mod in model.named_modules():
        short = name.replace("base_model.model.", "")
        if short not in before or not hasattr(mod, "weight"):
            continue
        after = mod.weight.detach().float()
        if torch.allclose(before[short], after, atol=1e-6, rtol=0):
            unchanged.append(short)
        else:
            changed += 1
            worst = max(worst, (before[short] - after).abs().max().item())
    total = changed + len(unchanged)
    frac = changed / max(1, total)
    print(f"MERGE CHECK: {changed}/{total} target tensors changed ({frac:.1%})", flush=True)
    print(f"MERGE CHECK: largest single-weight delta {worst:.6f}", flush=True)
    if frac < MIN_CHANGED_FRACTION:
        print(f"FATAL: only {frac:.1%} of targets moved (need {MIN_CHANGED_FRACTION:.0%})")
        print(f"       unchanged examples: {unchanged[:3]}")
        sys.exit(3)

    model.save_pretrained("/data/merged", safe_serialization=True)
    tokenizer.save_pretrained("/data/merged")

    # UPLOAD_MERGED: hand the verified fp16 weights back and convert to GGUF elsewhere.
    # llama.cpp's gguf writer crawled at 145-363 kB/s on the job volume (9-13 h for 8.4 GB)
    # while save_pretrained had just written the same volume fine — so ship the safetensors
    # and let a machine with a working llama.cpp do the conversion in minutes.
    if os.environ.get("UPLOAD_MERGED") == "1":
        prepare_upload(api)
        upload(api, "folder", folder_path="/data/merged", repo_id=OUT_REPO)
        print("Merged weights saved to /data/merged", flush=True)
        print("MERGE_VERIFIED_DONE_WEIGHTS_ONLY", flush=True)
        return

    gh = "https://github.com/ggml-org/llama.cpp"
    sh(
        f"git clone --depth 1 --branch {LLAMA_TAG} {gh} /data/llama.cpp"
    )
    # convert FIRST: pure python (gguf), so it cannot fail for want of a compiler
    sh(
        "python /data/llama.cpp/convert_hf_to_gguf.py /data/merged --outfile /data/model-f16.gguf --outtype f16"
    )
    # quantizing needs llama-quantize BUILT, and this image ships no cmake/g++ by
    # default (cmake now comes from PyPI; a missing compiler is still possible). On a
    # build failure, upload the f16 GGUF rather than throw away a 40-minute merge: f16
    # is a size/speed problem, never a correctness one, and packaging can quantize it.
    prepare_upload(api)
    quantized = False
    try:
        sh(
            "cmake -B /data/llama.cpp/build -S /data/llama.cpp -DLLAMA_CURL=OFF"
            " && cmake --build /data/llama.cpp/build --target llama-quantize -j"
        )
        sh(
            f"/data/llama.cpp/build/bin/llama-quantize /data/model-f16.gguf /data/model-{QUANT}.gguf {QUANT}"
        )
        quantized = True
    except subprocess.CalledProcessError as exc:
        print(f"WARNING: llama-quantize unavailable ({exc}) — keeping f16 instead", flush=True)

    src = f"/data/model-{QUANT}.gguf" if quantized else "/data/model-f16.gguf"
    dest = f"model-{QUANT}.gguf" if quantized else "model-f16.gguf"
    upload(api, "file", path_or_fileobj=src, path_in_repo=dest, repo_id=OUT_REPO)
    print(f"Saved {src}", flush=True)
    print("MERGE_VERIFIED_DONE" if quantized else "MERGE_VERIFIED_DONE_F16_ONLY", flush=True)


def main() -> None:
    """Merge, verify, save, quantize, upload."""
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    api = HfApi()

    adapter_dir = snapshot_download(ADAPTER_REPO, allow_patterns=[f"{CKPT}/*"] if CKPT else None)
    if CKPT:
        adapter_dir = os.path.join(adapter_dir, CKPT)
    print(f"adapter: {adapter_dir}", flush=True)

    state = load_adapter_state(adapter_dir)
    targets = base_module_names(state)
    print(f"adapter targets {len(targets)} modules; e.g. {sorted(targets)[:3]}", flush=True)

    if MERGE_VIA == "unsloth":
        merge_with_unsloth(adapter_dir, targets)
        return

    print("loading base in fp32 on CPU (keeps LoRA deltas out of rounding range)...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(BASE_ID, dtype=torch.float32, device_map="cpu")
    base_names = {n for n, _ in base.named_modules()}

    # Match adapter module names to the base model loaded here.
    # unsloth trains through a vision wrapper, so its keys may carry (or lack) a
    # `language_model.` segment relative to the class we just loaded.
    missing = {t for t in targets if t not in base_names}
    if missing:
        print(f"{len(missing)} adapter targets absent from the base's module names", flush=True)
        remapped = 0
        fixed: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            nk = k
            if "model.language_model." in k and "model.language_model." not in str(base_names):
                cand = k.replace("model.language_model.", "model.")
                if base_module_names({cand: v}) <= base_names:
                    nk, remapped = cand, remapped + 1
            elif ".visual." in k or "vision_tower" in k:
                continue  # vision-tower LoRA has no counterpart in a text-only base
            fixed[nk] = v
        if remapped:
            print(f"remapped {remapped} adapter keys to the base's naming", flush=True)
            from safetensors.torch import save_file

            save_file(fixed, os.path.join(adapter_dir, "adapter_model.safetensors"))
            state = fixed
            targets = base_module_names(state)
            missing = {t for t in targets if t not in base_names}
        if missing:
            print(f"FATAL: {len(missing)} targets still unmatched, e.g. {sorted(missing)[:3]}")
            print("       merging would be a silent no-op — refusing")
            sys.exit(2)

    # snapshot BEFORE
    before = {}
    for name, mod in base.named_modules():
        if name in targets and hasattr(mod, "weight"):
            before[name] = mod.weight.detach().clone()
    print(f"snapshotted {len(before)} target tensors before the merge", flush=True)
    if not before:
        print("FATAL: nothing to compare — the adapter targets no weighted module")
        sys.exit(2)

    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()

    # VERIFY: the whole point of this script
    changed, unchanged, worst = 0, [], 0.0
    for name, mod in model.named_modules():
        if name not in before or not hasattr(mod, "weight"):
            continue
        after = mod.weight.detach()
        if torch.allclose(before[name], after, atol=1e-6, rtol=0):
            unchanged.append(name)
        else:
            changed += 1
            worst = max(worst, (before[name] - after).abs().max().item())
    total = changed + len(unchanged)
    frac = changed / max(1, total)
    print(f"MERGE CHECK: {changed}/{total} target tensors changed ({frac:.1%})", flush=True)
    print(f"MERGE CHECK: largest single-weight delta {worst:.6f}", flush=True)
    if frac < MIN_CHANGED_FRACTION:
        print(f"FATAL: only {frac:.1%} of targets moved (need {MIN_CHANGED_FRACTION:.0%})")
        print(f"       unchanged examples: {unchanged[:3]}")
        print("       this is the silent-no-op merge that shipped base models three times")
        sys.exit(3)

    model = model.to(torch.float16)
    model.save_pretrained("/data/merged", safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE_ID).save_pretrained("/data/merged")
    del model, base

    gh = "https://github.com/ggml-org/llama.cpp"
    sh(
        f"git clone --depth 1 --branch {LLAMA_TAG} {gh} /data/llama.cpp"
    )
    sh(
        "cmake -B /data/llama.cpp/build -S /data/llama.cpp -DLLAMA_CURL=OFF && cmake --build /data/llama.cpp/build --target llama-quantize -j"
    )
    sh(
        "python /data/llama.cpp/convert_hf_to_gguf.py /data/merged --outfile /data/model-f16.gguf --outtype f16"
    )
    out_gguf = f"/data/model-{QUANT}.gguf"
    sh(f"/data/llama.cpp/build/bin/llama-quantize /data/model-f16.gguf {out_gguf} {QUANT}")

    prepare_upload(api)
    upload(api, "file", path_or_fileobj=out_gguf, path_in_repo=f"model-{QUANT}.gguf", repo_id=OUT_REPO)
    print(f"Saved {out_gguf}", flush=True)
    print("MERGE_VERIFIED_DONE", flush=True)


if __name__ == "__main__":
    main()
