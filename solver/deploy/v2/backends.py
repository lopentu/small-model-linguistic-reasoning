"""Model backends behind one protocol: chat(batch of message lists) -> texts.

Ports of the two replica-validated v1 drivers, a mock for tests, and the
llama-cli subprocess driver (LlamaCliBackend) — the only path to MTP +
--reasoning-budget — all behind the one Backend protocol.

Heavy imports (torch, llama_cpp) happen inside constructors so the entry
point can write a valid fallback submission before anything can crash.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
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
        self.vote_temp = cfg.vote_temp
        self.vote_top_p = cfg.vote_top_p
        self.repeat_penalty = cfg.repeat_penalty
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

        temp = 0.0 if greedy else self.vote_temp
        top_p = 1.0 if greedy else self.vote_top_p
        outs = []
        for msgs in batch:
            if self.template:
                prompt = render_chat(self.template, msgs, think=self.think)
                r = self.llm.create_completion(
                    prompt=prompt,
                    max_tokens=max_new,
                    temperature=temp,
                    top_p=top_p,
                    repeat_penalty=self.repeat_penalty,
                )
                outs.append(r["choices"][0]["text"] or "")
            else:  # no embedded template: fall back (no thinking control)
                r = self.llm.create_chat_completion(
                    messages=msgs,
                    max_tokens=max_new,
                    temperature=temp,
                    top_p=top_p,
                    repeat_penalty=self.repeat_penalty,
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


def gguf_chat_template(path: str) -> str:
    """Read tokenizer.chat_template straight from a GGUF file's KV header.

    Pure-python (no llama_cpp import, no model load): the parallel backend
    renders prompts itself and only the DRIVER touches the weights. Walks the
    GGUF v2/v3 KV section and returns "" when the key is absent or on any
    parse problem (fail-soft; callers decide whether that is fatal).
    """
    import struct

    def read(f: object, fmt: str) -> int:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, f.read(size))[0]  # type: ignore[attr-defined]

    # GGUF value type -> struct format for fixed-size scalars
    scalar = {
        0: "<B",
        1: "<b",
        2: "<H",
        3: "<h",
        4: "<I",
        5: "<i",
        6: "<f",
        7: "<B",
        10: "<Q",
        11: "<q",
        12: "<d",
    }
    t_string, t_array = 8, 9
    try:
        with open(path, "rb") as f:  # deployment code: os.path style, pathlib not needed
            if f.read(4) != b"GGUF":
                return ""
            version = read(f, "<I")
            if version < 2:  # v1 used 32-bit counts; our models are v3
                return ""
            f.read(8)  # tensor count
            n_kv = read(f, "<Q")

            def skip_value(vtype: int) -> bytes | None:
                """Skip one value; return the bytes only for a string."""
                if vtype in scalar:
                    f.read(struct.calcsize(scalar[vtype]))
                    return None
                if vtype == t_string:
                    return f.read(read(f, "<Q"))
                if vtype == t_array:
                    etype = read(f, "<I")
                    count = read(f, "<Q")
                    if etype in scalar:
                        f.read(struct.calcsize(scalar[etype]) * count)
                    elif etype == t_string:
                        for _ in range(count):
                            f.read(read(f, "<Q"))
                    else:  # nested arrays don't occur in practice
                        for _ in range(count):
                            skip_value(etype)
                    return None
                msg = f"unknown gguf value type {vtype}"
                raise ValueError(msg)

            for _ in range(n_kv):
                key = f.read(read(f, "<Q"))
                vtype = read(f, "<I")
                val = skip_value(vtype)
                if key == b"tokenizer.chat_template" and val is not None:
                    return val.decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001 — malformed file must not crash the solver
        print(f"[solver] gguf template read failed: {e!r}", flush=True)
    return ""


class LlamaParallelBackend:
    """JSONL pipe driver (llama-iolbatch): parallel vote decoding, model resident.

    Motivation (E32, 25 Jul): the eval seccomp kills every socket syscall —
    TCP loopback, AF_UNIX, socketpair all EPERM (dev/mock-eval/socket-probe,
    re-verified in-replica 25 Jul) — so llama-server is unusable, but pipes
    survive. The driver (dev/iolbatch/iolbatch.cpp, adapted from llama.cpp
    examples/parallel @ b9964) loads the GGUF ONCE and continuous-batch
    decodes up to cfg.slots streams; all n_samples of one prompt share a
    single prefill via llama_memory_seq_cp on a unified KV pool.

    chat() groups identical rendered prompts into one request (n_samples =
    multiplicity), so VotePass's [conv]*n batch costs ONE prompt prefill.
    Greedy batches collapse duplicates to a single decoded sample. Sampling
    parity with LlamaCppBackend: driver-side common defaults (top_k 40,
    min_p 0.05, repeat 1.0) — the repeat 1.0 is llama.cpp's common_params_sampling
    default (penalty_repeat), NOT the bindings' create_completion default, which is
    1.1; LlamaCppBackend now passes cfg.repeat_penalty (1.0) explicitly to match this
    path. greedy = temp 0. Vote temp/top_p come from cfg (VOTE_TEMP/VOTE_TOP_P).

    Robustness: a reader thread feeds a queue (select-free timeouts); on
    driver death or group timeout the call salvages what arrived, respawns
    the driver (bounded), and returns "" for the missing samples — one sick
    group must never sink the run.
    """

    _MAX_RESPAWNS = 2

    # a delivery-window reload only pays for itself if a cold load (<=9s) plus one

    # short finalize batch still fit before the delivery clock

    _DELIVERY_RELOAD_MIN_S = 45.0
    _TEARDOWN_MARGIN_S = 15.0  # must cover _kill()'s collapsed waits plus a flush
    _COLD_LOAD_S = 12.0  # measured 5.5s (4B Q8) / 9s (26B IQ4) + slack

    _DELIVERY_READY_WAIT_S = 20.0  # generous vs the 5.5-9s measured cold load

    _READY_TIMEOUT_S = 600.0  # T4 cold load ≈ 10-60 s; first CUDA init can add more

    def __init__(self, cfg: Config) -> None:
        import atexit

        self.cfg = cfg
        if not cfg.llama_parallel or not os.path.isfile(cfg.llama_parallel):
            msg = f"LLAMA_PARALLEL binary not found: {cfg.llama_parallel!r}"
            raise RuntimeError(msg)
        self.template = gguf_chat_template(cfg.gguf_path)
        if not self.template:
            msg = f"no tokenizer.chat_template in {cfg.gguf_path!r} — parallel backend needs it"
            raise RuntimeError(msg)
        if cfg.n_samples - 1 > cfg.slots:  # votes/problem must fit one driver group
            msg = f"N_SAMPLES-1={cfg.n_samples - 1} votes exceed SLOTS={cfg.slots}"
            raise RuntimeError(msg)
        self.think = cfg.think
        self._seed = int(os.environ.get("SEED_BASE", "12345"))
        self._warned_no_deadline = False
        self._t_init = time.time()
        self._group = 1  # per-call frame tag (0 = the driver's pre-first-group default)
        self._respawns = 0
        self.proc = None
        self.last_meta: list[dict | None] = []
        self.log_path = os.path.join(tempfile.gettempdir(), "iolbatch-driver.log")
        self._spawn()
        atexit.register(self.close)

    # ---- process management -------------------------------------------------

    @staticmethod
    def _cuda_lib_path() -> str:
        """LD_LIBRARY_PATH for the driver: the eval image ships CUDA libs (incl.
        libnccl.so.2, linked-but-unused multi-GPU support) only inside torch's
        bundled nvidia-* wheels — mirror of the T4-validated e15 loader setup."""
        import glob
        import sysconfig

        sp = sysconfig.get_paths()["purelib"]
        libdirs = glob.glob(f"{sp}/nvidia/*/lib") + [f"{sp}/torch/lib"]
        return ":".join([*libdirs, os.environ.get("LD_LIBRARY_PATH", "")])

    def _spawn(self) -> None:
        """Start the driver and block until its ready line (fail loud if not)."""
        with contextlib.suppress(OSError):  # hf upload strips the exec bit — restore it
            os.chmod(self.cfg.llama_parallel, 0o755)  # noqa: S103 — it IS an executable
        cmd = [
            self.cfg.llama_parallel,
            "-m",
            self.cfg.gguf_path,
            "-ngl",
            str(self.cfg.n_gpu_layers if self.cfg.n_gpu_layers >= 0 else 999),
            "-c",
            str(self.cfg.llama_ctx),
            "-np",
            str(self.cfg.slots),
            "--no-mmap",
            "-ctk",
            "f16",
            "-ctv",
            "f16",  # sm_75: fp16 KV only, never bf16
            "--fit",
            "off",  # fail loud if ctx won't fit — a silent shrink would starve
            #                  admission (Python sized LLAMA_CTX for slots x max_new)
            "--offline",
        ]
        env = {
            **os.environ,
            "GGML_CUDA_NO_PINNED": "1",  # pinned buffers vs the T4 RAM cgroup
            "LD_LIBRARY_PATH": self._cuda_lib_path(),
        }
        self._log = open(self.log_path, "ab")  # noqa: SIM115 — lifetime == process lifetime
        self.proc = subprocess.Popen(  # fixed argv, no shell
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            env=env,
        )
        self._q: queue.Queue[dict | None] = queue.Queue()
        self._reader = threading.Thread(
            target=self._read_loop, args=(self.proc, self._q), daemon=True
        )
        self._reader.start()
        ready_wait = min(self._READY_TIMEOUT_S, self._deadline_remaining() - 30.0)
        if self._deadline_remaining(soft=True) <= 0:
            ready_wait = min(ready_wait, self._DELIVERY_READY_WAIT_S)
        if ready_wait < self._COLD_LOAD_S:
            self._kill()
            msg = f"not enough time to load the driver ({ready_wait:.0f}s < {self._COLD_LOAD_S}s)"
            raise RuntimeError(msg)
        ready = self._next_msg(ready_wait)
        if not ready or not ready.get("ready"):
            self._kill()
            msg = f"iolbatch driver failed to start (see {self.log_path})"
            raise RuntimeError(msg)
        print(
            f"[solver] iolbatch ready: n_ctx={ready.get('n_ctx')} slots={ready.get('slots')}",
            flush=True,
        )

    def _read_loop(self, proc: subprocess.Popen, q: queue.Queue) -> None:
        """Reader thread: push parsed stdout JSON lines into ITS OWN queue."""
        try:
            for raw in proc.stdout:  # type: ignore[union-attr]
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("{"):
                    continue  # stray library output — protocol lines only
                try:
                    q.put(json.loads(line))
                except ValueError:
                    continue
        except Exception:  # noqa: BLE001, S110 — pipe torn down mid-read
            pass
        q.put({"__eof__": True})  # distinct from a get() timeout (None)

    def _next_msg(self, timeout: float) -> dict | None:
        """Next protocol message, or None on timeout/driver death."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def _kill(self) -> None:
        """Terminate the driver. Reduce each wait from ten seconds to one when less than a minute remains, preserving time to write the submission."""
        w = 10.0 if self._deadline_remaining() > 60.0 else 1.0
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=w)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=w)
        if getattr(self, "_reader", None):
            self._reader.join(timeout=w)  # its EOF lands in ITS queue, not a successor's
        if self.proc:
            for pipe in (self.proc.stdin, self.proc.stdout):
                if pipe and not pipe.closed:
                    with contextlib.suppress(OSError):
                        pipe.close()
        if getattr(self, "_log", None) and not self._log.closed:
            self._log.close()

    def close(self) -> None:
        """Clean shutdown: ask the driver to quit, then make sure it is gone."""
        try:
            if self.proc and self.proc.poll() is None and self.proc.stdin:
                self.proc.stdin.write(b'{"cmd": "quit"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass
        self._kill()

    def _deadline_remaining(self, *, soft: bool = False) -> float:
        """Seconds until a published solver deadline; `soft` selects the 85% one.

        Budget.__init__ publishes both. If NEITHER is set the process was wired without
        a Budget (a dev harness, or a future entry point that builds the backend first),
        and returning inf would silently turn every clamp below into a no-op — the
        failure mode is a group that runs straight through the hard kill. So fall back
        to a finite bound derived from TIME_LIMIT_S and SAY SO once.
        """
        key = "SOLVER_SOFT_DEADLINE_TS" if soft else "SOLVER_DEADLINE_TS"
        ts = float(os.environ.get(key, "0") or 0)
        if ts > 0:
            return ts - time.time()
        if not self._warned_no_deadline:
            self._warned_no_deadline = True
            print(
                f"[solver] WARNING {key} unset — no Budget published a deadline; "
                "falling back to TIME_LIMIT_S so the timeout clamps stay finite",
                flush=True,
            )
        limit = float(os.environ.get("TIME_LIMIT_S", "1800") or 1800)
        return (self._t_init + limit * 0.97) - time.time()

    def _clock_left(self) -> float:
        """Seconds under the clock that owns the CURRENT phase.

        While the soft deadline is ahead, the caller can only be solve/vote/refine, and
        bounding its group by the DELIVERY clock would let one wedge eat the window
        reserved for finalize — the exact blank-answer failure the two-deadline split
        exists to remove. Once soft has
        passed, only delivery calls remain, so the delivery clock is the right bound."""
        soft_left = self._deadline_remaining(soft=True)
        return soft_left if soft_left > 0 else self._deadline_remaining()

    def group_timeout(self, est_prompt_tok: float, n_expected: int, max_new: int) -> float:
        """Wall-clock allowance for one driver group; <=0 means "do not start it".

        Worst-case estimate — prefill at >=300 tok/s plus decode at >=20 tok/s aggregate,
        generous so only a real wedge trips it — then clamped to the phase clock minus a
        teardown margin. There is deliberately NO positive floor: a 30s floor against a
        54s flush reserve let one wedged chunk spend most of it, with _kill()'s own waits
        on top. Extracted so the reserve arithmetic is testable without
        a GPU (dev/test_delivery_deadline.py case 9)."""
        est = min(900.0, 120.0 + est_prompt_tok / 300.0 + (n_expected * max_new) / 20.0)
        return min(est, self._clock_left() - self._TEARDOWN_MARGIN_S)

    def _recover(self, why: str) -> None:
        """Driver died or wedged: log, kill, respawn (bounded by count AND clock)."""
        print(f"[solver] iolbatch {why} — recovering", flush=True)
        self._kill()
        if self._respawns >= self._MAX_RESPAWNS:
            msg = f"iolbatch driver failed {self._respawns + 1}x ({why}); giving up on backend"
            raise RuntimeError(msg)
        soft_left, deliver_left = self._deadline_remaining(soft=True), self._deadline_remaining()
        if soft_left > 0 and soft_left < 120:
            msg = f"iolbatch {why} with <120s to the soft deadline — not reloading"
            raise RuntimeError(msg)
        if soft_left <= 0 and deliver_left < self._DELIVERY_RELOAD_MIN_S:
            msg = (
                f"iolbatch {why} with {deliver_left:.0f}s of the delivery window left "
                f"(<{self._DELIVERY_RELOAD_MIN_S}s) — not reloading"
            )
            raise RuntimeError(msg)
        self._respawns += 1
        self._spawn()

    # ---- generation ---------------------------------------------------------

    def chat(self, batch: list[Messages], *, greedy: bool, max_new: int) -> list[str]:
        """Render, group identical prompts (shared prefill), decode in parallel."""
        from .chat import render_chat

        prompts = [render_chat(self.template, msgs, think=self.think) for msgs in batch]

        # group duplicates: one request per unique prompt, in first-seen order
        order: dict[str, list[int]] = {}
        for i, p in enumerate(prompts):
            order.setdefault(p, []).append(i)

        reqs = []
        for rid, (prompt, idxs) in enumerate(order.items()):
            n = 1 if greedy else len(idxs)  # greedy duplicates decode once, fan out below
            reqs.append(
                {
                    "id": f"r{rid}",
                    "prompt": prompt,
                    "n_samples": n,
                    "max_new": max_new,
                    "temp": 0.0 if greedy else self.cfg.vote_temp,
                    "top_p": 1.0 if greedy else self.cfg.vote_top_p,
                    "seed_base": self._seed,
                    "logprobs": self.cfg.vote_weight != "none"
                    or os.environ.get("ASSIGNMENT", "0") == "1",
                }
            )
            self._seed += n
        n_expected = sum(r["n_samples"] for r in reqs)

        est_prompt_tok = sum(len(r["prompt"]) for r in reqs) / 2
        timeout_s = self.group_timeout(est_prompt_tok, n_expected, max_new)
        if timeout_s <= 0:
            msg = (
                f"refusing a {n_expected}-stream group with "
                f"{self._clock_left():.0f}s left (< {self._TEARDOWN_MARGIN_S}s teardown "
                "margin) — the reserve pays for the flush"
            )
            raise RuntimeError(msg)

        # every group is tagged and the loop below reads THROUGH the group's own
        # done line, discarding frames from other groups. Without the tag, one
        # early exit desyncs the stream and every later call reads the PREVIOUS
        # group's results under colliding r0../r1.. ids — answers land on the
        # wrong problems (exactly what the first 27-row A/B caught: chunk-2
        # predictions shifted by one problem, score 0.067 vs 0.141).
        group = self._group
        self._group += 1
        while not self._q.empty():  # drain leftovers from any abandoned group
            with contextlib.suppress(queue.Empty):
                self._q.get_nowait()

        results: dict[tuple[str, int], dict] = {}
        try:
            payload = json.dumps({"group": group, "reqs": reqs}, ensure_ascii=False)
            assert self.proc and self.proc.stdin  # noqa: S101 — invariant, spawn guarantees it
            self.proc.stdin.write(payload.encode("utf-8") + b"\n")
            self.proc.stdin.flush()

            t0 = time.time()
            deadline = t0 + timeout_s
            while True:
                msg = self._next_msg(max(0.1, deadline - time.time()))
                if msg is None or msg.get("__eof__"):
                    why = (
                        f"died mid-group after {time.time() - t0:.0f}s"
                        if msg is not None
                        else f"produced nothing for {timeout_s:.0f}s (wedged)"
                    )
                    self._recover(why)  # raises after _MAX_RESPAWNS
                    break
                if msg.get("group") != group:
                    continue  # stale frame from an abandoned group — discard
                if msg.get("done"):
                    if len(results) < n_expected:
                        print(
                            f"[solver] iolbatch group short: {len(results)}/{n_expected}",
                            flush=True,
                        )
                    break
                if msg.get("stop") == "error":
                    print(f"[solver] iolbatch error: {msg.get('error', '?')}", flush=True)
                if "id" in msg and "sample_idx" in msg:
                    results[(msg["id"], int(msg["sample_idx"]))] = msg
        except OSError as e:  # broken pipe on write — driver already gone
            self._recover(f"pipe error ({e!r})")

        # map results back to input positions (greedy fan-out included)
        outs = [""] * len(batch)
        metas: list[dict | None] = [None] * len(batch)
        for rid, (_prompt, idxs) in enumerate(order.items()):
            for j, i in enumerate(idxs):
                msg = results.get((f"r{rid}", 0 if greedy else j))
                if msg is None:
                    continue
                outs[i] = msg.get("text", "") or ""
                if "lps" in msg:
                    metas[i] = {
                        "lps": msg.get("lps") or [],
                        "tlens": msg.get("tlens") or [],
                        "lp_sum": msg.get("lp_sum", 0.0),
                        "lp_eos": msg.get("lp_eos", 0.0),
                        "n_decoded": msg.get("n_decoded", 0),
                        "stop": msg.get("stop", ""),
                    }
        # published for the caller to pick up immediately after chat() (the Backend
        # protocol stays list[str]; backends without logprobs simply never set it)
        self.last_meta = metas
        return outs


class TransformersBackend:
    """HF transformers driver with left-padded batch generation (v1 port)."""

    def __init__(self, cfg: Config) -> None:
        import torch  # available in the sandbox image / jobs image only
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.think = cfg.think
        self.repeat_penalty = cfg.repeat_penalty  # 1.0 default: see Config.repeat_penalty
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
                repetition_penalty=self.repeat_penalty,
                pad_token_id=self.tok.pad_token_id,
            )
        cut = enc["input_ids"].shape[1]
        return [self.tok.decode(o[cut:], skip_special_tokens=True) for o in out]


def make_backend(cfg: Config) -> Backend:
    """Pick the driver from config: parallel (voting) > llama-cli > bindings > transformers."""
    if cfg.backend == "parallel":
        if not cfg.gguf_path:
            msg = "BACKEND=parallel requires GGUF_PATH"
            raise RuntimeError(msg)
        return LlamaParallelBackend(cfg)
    if cfg.gguf_path and cfg.llama_cli:
        return LlamaCliBackend(cfg)
    if cfg.gguf_path:
        return LlamaCppBackend(cfg)
    return TransformersBackend(cfg)
