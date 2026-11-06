"""Model backends behind one protocol: chat(batch of message lists) -> texts.

Ports of the two replica-validated v1 drivers, a mock for tests, and the
llama-cli subprocess driver (LlamaCliBackend) — the only path to MTP +
--reasoning-budget — all behind the one Backend protocol.

Heavy imports (torch, llama_cpp) happen inside constructors so the entry
point can write a valid fallback submission before anything can crash.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

if TYPE_CHECKING:
    from .config import Config
    from .prompts import Messages


class Backend(Protocol):
    """A chat model that turns message lists into completion texts."""

    def chat(self, batch: list[Messages], *, greedy: bool, max_new: int) -> list[str]:
        """Generate one completion per conversation in the batch."""
        ...


class MockBackend:
    """Deterministic backend for tests: delegates to a caller-supplied handler."""

    def __init__(self, handler: Callable[[Messages, bool, int], str]) -> None:
        self.handler = handler
        self.calls = 0

    def chat(self, batch: list[Messages], *, greedy: bool, max_new: int) -> list[str]:
        """Run the handler once per conversation."""
        self.calls += len(batch)
        return [self.handler(msgs, greedy, max_new) for msgs in batch]


class LlamaCppBackend:
    """llama-cpp-python bindings driver (single-stream).

    Thinking control is model-agnostic: we render the model's OWN embedded chat
    template with an explicit `enable_thinking` (see chat.render_chat) and feed the
    string to create_completion — the templates default to thinking-ON when the
    flag is absent, so relying on create_chat_completion would silently enable it
    (the exact trap that hid on Qwen). Falls back to create_chat_completion if the
    GGUF carries no template.

    GPU-VALIDATED 12 Jul on Gemma-4-26B: create_completion tokenizes the rendered
    string with special=True, so Gemma-4's channel tokens parse correctly
    (<|turn> = id 105, one token — not shredded); think=False is respected (output
    goes straight to the final channel, no leaked thought block); generation is
    coherent. (Qwen already ran a full scored v2 sweep.) So render->create_completion
    is sound for both families — the CLI backend's --jinja path is not required here.
    """

    def __init__(self, cfg: Config) -> None:
        from llama_cpp import Llama  # vendored; import deferred by design

        self.think = cfg.think
        self.llm = Llama(
            model_path=cfg.gguf_path,
            n_gpu_layers=cfg.n_gpu_layers,
            n_ctx=cfg.llama_ctx,  # 4096 default — 26B+KV must fit the T4's 16 GB (8192 -> ~15.6 GB, OOM risk)
            verbose=False,
        )
        self.template = self.llm.metadata.get("tokenizer.chat_template")

    def chat(self, batch: list[Messages], *, greedy: bool, max_new: int) -> list[str]:
        """Generate sequentially (the bindings are single-stream)."""
        from .chat import render_chat

        temp = 0.0 if greedy else 1.0
        top_p = 1.0 if greedy else 0.95
        outs = []
        for msgs in batch:
            if self.template:
                prompt = render_chat(self.template, msgs, think=self.think)
                r = self.llm.create_completion(
                    prompt=prompt, max_tokens=max_new, temperature=temp, top_p=top_p
                )
                outs.append(r["choices"][0]["text"] or "")
            else:  # no embedded template: fall back (no thinking control)
                r = self.llm.create_chat_completion(
                    messages=msgs, max_tokens=max_new, temperature=temp, top_p=top_p
                )
                outs.append(r["choices"][0]["message"]["content"] or "")
        return outs


class LlamaCliBackend:
    """llama-cli subprocess driver — the ONLY path to MTP + --reasoning-budget.

    llama-cli does the work the bindings can't: `--jinja` applies the model's OWN chat
    template, `--reasoning`/`--reasoning-budget` manage the thinking block, and special tokens
    tokenize internally — so we hand it PLAIN system + user text (-sysf/-f) and let it template.
    No manual render_chat, no raw special-token passing. `-st` (single-turn) + stdin=/dev/null
    runs one predefined turn and exits (guards the interactive '> ' hang — the "MTP times out"
    root cause). Flags verified in-image on the 26B (cli_prompt_probe.sh + native run, 12 Jul).

    INTENDED RECIPE — SOLVE-ONLY with thinking budget (E8/E16), opt-in via LLAMA_CLI:
    THINK=1 + REASONING_BUDGET=900 (+ the wrap-up message below forces delivery) + MTP, and
    FEWSHOT=0 REFINE_ROUNDS=0. In that mode every call is one system+user turn and the solve
    pass itself emits RULES/FINAL ANSWERS — finalize/refine are unnecessary.

    KNOWN LIMITS (previously observed, gated before any scored run depends on this):
    - Multi-turn (fewshot exemplar, finalize/refine follow-ups) is FLATTENED into one user
      turn — the prior assistant analysis becomes user text, so finalize/refine pragmatics are
      NOT preserved. __init__ warns when fewshot/refine are on. Not a validated production path;
      a dual/composite backend (CLI solve + bindings multi-turn) is the real fix if needed.
    - One model load PER call (~5 s A5000 / ~12-15 s T4 measured). For ~50 solve-only calls
      that is tolerable; adding finalize/refine (~2-3x calls) or n>1 likely blows the 30-min
      T4 budget — a persistent-pipe process is the future optimization. Gate n>1/refine on
      measured wall.
    """

    WRAP_UP = (
        "Time is up. Output RULES: (2-4 bullets) then FINAL ANSWERS: with one bare "
        "answer per line for every QUERY item now."
    )
    DELIVERY_HEADROOM = 600  # tokens reserved for RULES + answers after the thinking budget
    _TIMEOUT_FLOOR_S = 60.0

    def __init__(self, cfg: Config) -> None:
        import subprocess  # llama-cli templates itself; no model load, just a startup smoke

        self.cfg = cfg
        try:
            v = subprocess.run(
                [cfg.llama_cli, "--version"],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
                stdin=subprocess.DEVNULL,
            )
            tag = (v.stderr or v.stdout or "").strip().splitlines()[:1]
            print(f"[solver] llama-cli smoke rc={v.returncode} {tag}", flush=True)
        except (OSError, subprocess.SubprocessError) as e:
            print(f"[solver] WARN llama-cli --version failed ({e!r}) — check LLAMA_CLI", flush=True)
        if cfg.reasoning_budget >= 0 and not cfg.think:
            print("[solver] WARN REASONING_BUDGET set but THINK=0 — ignored", flush=True)
        if cfg.fewshot or cfg.max_refine_rounds > 0:
            print(
                "[solver] WARN CLI backend is solve-only; fewshot/refine turns get FLATTENED "
                "into one user turn (unvalidated) — set FEWSHOT=0 REFINE_ROUNDS=0 for the "
                "intended thinking-budget recipe",
                flush=True,
            )

    def _base_cmd(self, *, greedy: bool, max_new: int) -> list[str]:
        """Build the fixed CLI args for one generation (prompt files added per-call)."""
        n_predict = max_new
        cmd = [
            self.cfg.llama_cli,
            "-m",
            self.cfg.gguf_path,
            "-ngl",
            str(self.cfg.n_gpu_layers),
            "-c",
            str(self.cfg.llama_ctx),
            "--jinja",  # apply the model's own chat template (handles thinking + special tokens)
            "-st",  # single-turn: template one predefined turn, generate, exit
            "--offline",  # sealed sandbox: never touch the network
            "--no-mmap",
            "--fit",
            "off",  # fail loud if ctx won't fit, don't silently shrink it
            "-ctk",
            "f16",
            "-ctv",
            "f16",  # T4 sm_75: fp16 KV only, never bf16
            "--temp",
            "0" if greedy else "1",
            "--top-p",
            "1" if greedy else "0.95",  # parity with the bindings backend
            # cleaner subprocess output (docs read + in-image verified 12 Jul): --log-disable
            # empties STDERR (CUDA-init/perf/spinner logs — the decode-crash source); --simple-io
            # kills the \x08 backspace spinner; --no-display-prompt: no prompt echo; -co off: no
            # color escapes. NOTE: llama-cli still prints its startup logo/model-card to STDOUT
            # regardless — harmless: no FINAL ANSWERS marker so parse_answers skips it, and it's
            # valid UTF-8. errors="replace" (below) stays as belt-and-suspenders.
            "--log-disable",
            "--simple-io",
            "--no-display-prompt",
            "-co",
            "off",
        ]
        # thinking is gated on cfg.think; the budget applies ONLY when thinking is on
        if self.cfg.think:
            cmd += ["--reasoning", "on"]
            if self.cfg.reasoning_budget >= 0:
                cmd += [
                    "--reasoning-budget",
                    str(self.cfg.reasoning_budget),
                    "--reasoning-budget-message",
                    self.WRAP_UP,
                ]
                n_predict = max(n_predict, self.cfg.reasoning_budget + self.DELIVERY_HEADROOM)
        else:
            cmd += ["--reasoning", "off"]
        cmd += ["-n", str(n_predict)]
        if self.cfg.model_draft:  # MTP speculative decoding (1.49x in-image, 12 Jul)
            cmd += [
                "--model-draft",
                self.cfg.model_draft,
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                str(self.cfg.spec_draft_n),
            ]
        return cmd

    def chat(self, batch: list[Messages], *, greedy: bool, max_new: int) -> list[str]:
        """One llama-cli subprocess per conversation; native templating via -sysf/-f + --jinja."""
        import os
        import subprocess
        import tempfile

        base = self._base_cmd(greedy=greedy, max_new=max_new)
        timeout_s = max(self._TIMEOUT_FLOOR_S, max_new * 0.5 + 120)
        env = {**os.environ, "GGML_CUDA_NO_PINNED": "1"}
        outs: list[str] = []
        for msgs in batch:
            system = "\n".join(m["content"] for m in msgs if m["role"] == "system")
            user = "\n\n".join(m["content"] for m in msgs if m["role"] != "system")
            with (
                tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8") as uf,
                tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8") as sf,
            ):
                uf.write(user)
                uf.flush()
                cmd = [*base, "-f", uf.name]
                if system:
                    sf.write(system)
                    sf.flush()
                    cmd += ["-sysf", sf.name]
                try:
                    p = subprocess.run(
                        cmd,
                        capture_output=True,
                        # errors="replace" is MANDATORY: llama-cli interleaves its ASCII-art
                        # banner + backspace spinner with the model's IPA/LaTeX output, and a
                        # split multibyte sequence makes strict decode raise UnicodeDecodeError
                        # — which under text=True killed EVERY batch -> fallback (silent score
                        # collapse; matches dev/local/think_ab.py). Replace keeps the answer text.
                        encoding="utf-8",
                        errors="replace",
                        timeout=timeout_s,
                        check=False,
                        stdin=subprocess.DEVNULL,  # one-shot; guards the interactive '> ' hang
                        env=env,
                    )
                except subprocess.TimeoutExpired as e:
                    # salvage partial output — it may already hold a FINAL ANSWERS block
                    print("[solver] llama-cli timed out — using partial output", flush=True)
                    outs.append(e.stdout or "")
                    continue
                except Exception as exc:  # noqa: BLE001 — one bad call must never sink the run
                    print(f"[solver] llama-cli call failed: {exc!r} — empty completion", flush=True)
                    outs.append("")
                    continue
            if p.returncode != 0:  # OOM / bad flag / missing draft — don't fail silent
                tail = "|".join((p.stderr or "").splitlines()[-3:])
                print(f"[solver] llama-cli rc={p.returncode}: {tail}", flush=True)
            outs.append(p.stdout or "")
        return outs


class TransformersBackend:
    """HF transformers driver with left-padded batch generation (v1 port)."""

    def __init__(self, cfg: Config) -> None:
        import torch  # available in the sandbox image / jobs image only
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.think = cfg.think
        self.tok = AutoTokenizer.from_pretrained(cfg.model_dir)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model_dir, dtype=dtype)
        except TypeError:  # transformers<5 spells it torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model_dir, torch_dtype=dtype)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model = self.model.eval()
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

    def _render(self, msgs: Messages) -> str:
        """Apply the chat template, tolerating templates without enable_thinking."""
        try:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=self.think
            )
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def chat(self, batch: list[Messages], *, greedy: bool, max_new: int) -> list[str]:
        """Generate the whole batch in one padded forward pass."""
        prompts = [self._render(m) for m in batch]
        with self.torch.no_grad():
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=not greedy,
                temperature=1.0 if not greedy else None,
                top_p=0.95 if not greedy else None,
                repetition_penalty=1.05,
                pad_token_id=self.tok.pad_token_id,
            )
        cut = enc["input_ids"].shape[1]
        return [self.tok.decode(o[cut:], skip_special_tokens=True) for o in out]


def make_backend(cfg: Config) -> Backend:
    """Pick the driver from config: llama-cli (MTP/budget) > bindings (GGUF) > transformers."""
    if cfg.gguf_path and cfg.llama_cli:
        return LlamaCliBackend(cfg)
    if cfg.gguf_path:
        return LlamaCppBackend(cfg)
    return TransformersBackend(cfg)
