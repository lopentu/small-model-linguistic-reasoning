"""Deterministic answer checks — the grounded half of the refine loop (E20).

Each check inspects a ProblemState's current aggregate and returns human-readable
verdict strings; the refine pass injects them into the follow-up prompt so the
model revises against real failures instead of re-reading its own text.

Provenance: tau's "tools are ordinary typed functions returning structured
results" (github.com/huggingface/tau), adapted for the offline sandbox — the
orchestrator runs the checks and pastes verdicts into the prompt, so no
model-native tool calling is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import textproc

if TYPE_CHECKING:
    from .pipeline import ProblemState

MAX_ANSWER_LEN = 60


def check(state: ProblemState) -> list[str]:
    """Run all validators; empty list means nothing to complain about."""
    verdicts: list[str] = []
    texts = [s["text"] for s in state.samples]
    if not texts:
        return ["no solution attempt was produced"]

    if not any(textproc.has_final_marker(t) for t in texts):
        verdicts.append("the output never reached a 'FINAL ANSWERS:' block")

    parsed = [
        textproc.parse_answers(
            t, state.k, state.context, labels=state.labels, multi_slot=state.multi_slot
        )
        for t in texts
    ]
    voted = textproc.vote(parsed, state.k)

    blanks = [i + 1 for i, v in enumerate(voted) if not v.strip()]
    if blanks:
        verdicts.append(
            f"items {blanks} have no answer — {state.k} answer lines are required, "
            f"one per asked item"
        )

    # echo check runs on the UNFILTERED parse: parse_answers(context=...) already
    # drops echoed lines (they become blanks above), so detecting the copying —
    # and telling the model about it — requires parsing without the filter
    raw_voted = textproc.vote(
        [textproc.parse_answers(t, state.k, labels=state.labels) for t in texts], state.k
    )
    ctx_lines = {textproc.normalize(cl) for cl in state.context.splitlines() if cl.strip()}
    echoed = [
        i + 1 for i, v in enumerate(raw_voted) if v.strip() and textproc.normalize(v) in ctx_lines
    ]
    if echoed:
        verdicts.append(
            f"items {echoed} merely copy lines from the problem data — answers must be "
            f"newly derived forms"
        )

    overlong = [i + 1 for i, v in enumerate(voted) if len(v) > MAX_ANSWER_LEN]
    if overlong:
        verdicts.append(
            f"items {overlong} look like prose, not answers — give the bare answer only"
        )

    task_type = state.row.get("task_type", "")

    if task_type == "match_letters":
        non_blank = [textproc.normalize(v) for v in voted if v.strip()]
        if len(non_blank) != len(set(non_blank)):
            verdicts.append(
                "matching is a bijection — each option letter must appear exactly once; "
                "no letter may be repeated across your answers"
            )

    elif task_type in {"translate", "fill_blanks"}:
        non_blank = [textproc.normalize(v) for v in voted if v.strip()]
        if len(non_blank) > 1 and len(non_blank) != len(set(non_blank)):
            verdicts.append(
                "two or more answers are identical — the items are distinct, so check "
                "whether you missed a morphological or semantic difference"
            )

    return verdicts
