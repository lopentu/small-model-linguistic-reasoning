"""Append-only JSONL transcript: one event per generation.

Provenance: tau's durable sessions — "history is append-only JSONL; active
context can be compacted without rewriting the record" (github.com/huggingface/tau).
Serves two purposes here:
  - debugging: the "read raw model output first" working rule needs raw text;
  - resume: a restart inside the 30-min window reloads completed generations
    and skips straight to unfinished work.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class Transcript:
    """Append-only JSONL writer; path "" disables logging entirely."""

    def __init__(self, path: str) -> None:
        self.path = Path(path) if path else None

    def log(self, event: dict[str, object]) -> None:
        """Append one event (adds a timestamp); never raises."""
        if self.path is None:
            return
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"t": round(time.time(), 2), **event}, ensure_ascii=False))
                f.write("\n")
        except OSError:
            self.path = None  # a broken transcript must never kill the solver

    @staticmethod
    def load_samples(path: str) -> dict[str, list[dict]]:
        """Rebuild {problem id: [{kind, text}, ...]} from a prior run's transcript.

        Replays FinalizePass's merge: a "finalize" event is a continuation of the
        sample at its recorded slot, not a vote of its own — merging here makes a
        resumed run reconstruct exactly the state an uninterrupted run would hold.
        A kill can tear the file mid-character, so decoding must never raise:
        resume runs BEFORE the first fallback flush.
        """
        samples: dict[str, list[dict]] = {}
        p = Path(path) if path else None
        if p is None or not p.exists():
            return samples
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return samples  # an unreadable transcript must never kill the solver
        for line in lines:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn tail write from a killed run
            if ev.get("event") != "sample" or "id" not in ev or "text" not in ev:
                continue
            kind, text = str(ev.get("kind", "solve")), str(ev["text"])
            have = samples.setdefault(str(ev["id"]), [])
            if kind == "finalize" and have:
                slot = ev.get("slot")
                idx = slot if isinstance(slot, int) and 0 <= slot < len(have) else len(have) - 1
                have[idx]["text"] += "\n" + text
                have[idx]["finalized"] = "1"
                have[idx]["meta"] = None
            else:
                # perm travels with the sample: without it a resumed shuffled vote
                # would be read as identity order and its answers scrambled
                have.append(
                    {"kind": kind, "text": text, "meta": ev.get("meta"), "perm": ev.get("perm")}
                )
        return samples
