"""Entry point: thin wiring of config, backend, pipeline, and crash-safe output.

The submission repo root runs a two-line script.py that imports main() from
here (see the package docstring). Order of operations is load-bearing:
a valid fallback submission.csv exists BEFORE any heavy import can crash.
"""

from __future__ import annotations

import atexit
import csv
import importlib.machinery
import json
import os
import signal
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

from .budget import Budget
from .config import Config
from .pipeline import Context, Pipeline, ProblemState, aggregate, default_passes
from .transcript import Transcript

if TYPE_CHECKING:
    from collections.abc import Callable


def bootstrap_vendor() -> None:
    """Prepend vendor/ to sys.path, shim sklearn, and pin offline mode.

    Must run before any import that touches huggingface_hub. The sklearn shim
    covers image builds that pair a broken scipy with transformers>=5 imports.
    """
    here = Path(__file__).resolve().parent
    for root in (here.parent, here):  # repo root first, then package
        vendor = root / "vendor"
        if vendor.is_dir() and str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))

    if "sklearn" not in sys.modules:
        sk = types.ModuleType("sklearn")
        sk.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
        skm = types.ModuleType("sklearn.metrics")
        skm.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None)

        def _shimmed(*_a: object, **_k: object) -> object:
            msg = "sklearn shimmed"
            raise RuntimeError(msg)

        skm.roc_curve = _shimmed  # type: ignore[attr-defined]
        sk.metrics = skm  # type: ignore[attr-defined]
        sys.modules["sklearn"] = sk
        sys.modules["sklearn.metrics"] = skm

    offline = "1" if os.environ.get("MODEL_DIR", ".") == "." else "0"
    os.environ.setdefault("TRANSFORMERS_OFFLINE", offline)
    os.environ.setdefault("HF_HUB_OFFLINE", os.environ["TRANSFORMERS_OFFLINE"])


def load_rows(path: str) -> list[dict[str, str]]:
    """Read test.csv into dicts."""
    with Path(path).open(encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def make_flusher(
    states: list[ProblemState], out_path: str = "submission.csv"
) -> Callable[[], None]:
    """Build the incremental writer: every call rewrites a complete submission."""

    def flush() -> None:
        # write-then-rename: a hard kill mid-flush must never tear the CSV the
        # scorer reads — the previous complete submission stays in place
        tmp = Path(out_path + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "pred", "explanation"])
            w.writeheader()
            for s in states:
                pred, expl = aggregate(s)
                w.writerow(
                    {"id": s.id, "pred": json.dumps(pred, ensure_ascii=False), "explanation": expl}
                )
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(out_path)

    return flush


def main() -> None:
    """Run the solver end to end."""
    budget = Budget(float(os.environ.get("TIME_LIMIT_S", "1800")))  # clock starts first
    bootstrap_vendor()
    cfg = Config.from_env()

    rows = load_rows(cfg.test_csv)
    states = [ProblemState.from_row(r) for r in rows]
    budget.init_rows(len(states))
    print(f"[solver] {len(rows)} problems | n={cfg.n_samples} max_new={cfg.max_new}", flush=True)

    flush = make_flusher(states)
    flush()  # a valid (fallback) submission exists before resume or heavy imports
    atexit.register(flush)
    if hasattr(signal, "SIGALRM"):  # Linux sandbox only; guard for Windows dev box
        signal.signal(signal.SIGALRM, lambda _s, _f: flush())
        signal.alarm(max(1, int(budget.remaining()) - 60))

    # resume: reload any generations a killed prior run already paid for
    prior_samples = Transcript.load_samples(cfg.transcript_path)
    for s in states:
        s.samples = prior_samples.get(s.id, [])
    prior = sum(len(s.samples) for s in states)
    if prior:
        print(f"[solver] resumed {prior} generations from transcript", flush=True)
        flush()

    try:
        from .retrieval import retrieve_all

        addons = retrieve_all(rows)
        for s in states:
            s.retrieved = addons.get(s.id, "")
        if addons:
            print(
                f"[solver] retrieval: grounded {len(addons)}/{len(states)} "
                f"problems at t+{budget.elapsed():.0f}s",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001 — retrieval must never take the solver down
        print(f"[solver] retrieval skipped: {exc}", flush=True)

    # Numeral pre-pass: a deterministic, model-free solver for number tasks runs
    # on CPU before the GPU model loads. Rows it PROVES entirely are committed and
    # skip the model; the rest fall through unchanged. Fully fail-closed.
    transcript = Transcript(cfg.transcript_path)
    solved_ids: set[str] = set()
    try:
        from .numbrute import presolve_numerals

        solved_ids = presolve_numerals(states, transcript=transcript)
        if solved_ids:
            print(
                f"[solver] numbrute: proved {len(solved_ids)}/{len(states)} number "
                f"tasks at t+{budget.elapsed():.0f}s (skipping the model for them)",
                flush=True,
            )
            flush()  # bank the proven answers immediately
    except Exception as exc:  # noqa: BLE001 — the number pre-pass must never sink the run
        print(f"[solver] numbrute skipped: {exc}", flush=True)

    from .backends import make_backend  # heavy imports live behind the flush

    backend = make_backend(cfg)
    print(f"[solver] backend ready at t+{budget.elapsed():.0f}s", flush=True)

    ctx = Context(
        cfg=cfg,
        backend=backend,
        budget=budget,
        transcript=transcript,
        flush=flush,
    )
    # Only the rows the number pre-pass did not fully prove go to the model.
    pipeline_states = [s for s in states if s.id not in solved_ids]
    budget.init_rows(len(pipeline_states))  # adaptive token budget excludes pre-solved rows
    Pipeline(default_passes(cfg)).run(pipeline_states, ctx)

    flush()
    print(f"[solver] done: {len(states)} problems at t+{budget.elapsed():.0f}s", flush=True)


if __name__ == "__main__":
    main()
