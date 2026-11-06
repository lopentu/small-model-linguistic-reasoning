"""Optional numeral pre-pass. Requires the separately supplied iol package; missing dependencies leave the row unchanged."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import ProblemState
    from .transcript import Transcript

# Mirrors dev/classifier classify(): the task_type override that routes to
# 'exhaustive_numeral'. task_type is normalised the same way (lower, - -> _).
_NUMERAL_TASK_TYPES = frozenset({"text_to_num", "num_to_text"})


def _is_numeral(row: dict[str, str]) -> bool:
    """Return True when the row's task_type marks it a number task."""
    tt = str(row.get("task_type", "")).strip().lower().replace("-", "_")
    return tt in _NUMERAL_TASK_TYPES


def presolve_numerals(
    states: list[ProblemState], *, transcript: Transcript | None = None
) -> set[str]:
    """Solve provable number tasks up front; return the set of solved ids.

    For each solved row a synthetic ``solve`` sample carrying the proven answers
    is appended (so the normal flush/aggregate writes them), and the row is
    returned in the solved set so the caller can exclude it from the pipeline.
    Rows that are not fully proven are left untouched.
    """
    if os.environ.get("NUMBRUTE", "1") != "1":
        return set()
    try:
        pkg_parent = str(Path(__file__).resolve().parent.parent)  # .../solver
        if pkg_parent not in sys.path:
            sys.path.insert(0, pkg_parent)
        from iol.numbers_adapter import answer  # noqa: PLC0415
    except Exception:  # noqa: BLE001 — no brute-force package = no pre-pass, never fatal
        return set()

    seconds = float(os.environ.get("NUMBRUTE_SECONDS", "8"))
    cap = int(os.environ.get("NUMBRUTE_CAP", "60000"))
    solved: set[str] = set()
    for s in states:
        if not _is_numeral(s.row):
            continue
        try:
            res = answer(s.context, s.query, cap=cap, seconds=seconds, proven=True)
        except Exception:  # noqa: BLE001 — a bad row must fall through to the model
            res = None
        if res is None or not res.all_committed:
            continue  # not fully proven -> let the model solve it
        solved.add(s.id)
        # Inject the proven answers as a solve sample so the existing
        # flush/aggregate machinery writes them. Guard against duplicating on
        # resume (the sample was transcript-logged and reloaded already).
        if not any(x.get("kind") == "solve" for x in s.samples):
            text = "FINAL ANSWERS:\n" + "\n".join(res.answers)
            s.samples.append({"kind": "solve", "text": text})
            if transcript is not None:
                transcript.log(
                    {
                        "event": "sample",
                        "id": s.id,
                        "kind": "solve",
                        "pass": "numbrute",
                        "text": text,
                    }
                )
    return solved
