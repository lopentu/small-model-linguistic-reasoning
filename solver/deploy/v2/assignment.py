"""Optional one-to-one assignment of answer labels using vote weights. Disabled by default: the task metadata does not guarantee a bijection, and some tasks legitimately reuse labels."""

from __future__ import annotations

import re
from collections import Counter

# a "letter answer" is one or two alphanumerics, optionally parenthesised: A, (b), 12, iv
LETTERISH = re.compile(r"^\(?\s*([A-Za-z]|[0-9]{1,2})\s*\)?[.,]?$")
MIN_ROWS_FOR_ASSIGNMENT = 2


def canonical(label: str) -> str:
    """Return the option label in one canonical form, or "" if it is not a label.

    "A", "(A)", "a." and " a " are the same option; keeping their raw spellings as separate
    matrix columns let the solver satisfy the bijection while reusing one option.
    """
    m = LETTERISH.match((label or "").strip())
    return m.group(1).lower() if m else ""


def looks_like_letters(answers: list[str], *, min_frac: float = 0.8) -> bool:
    """Return True when most answers are single option labels rather than prose.

    A guard, not a classifier: this module must never touch a translation row whose task_type
    metadata happens to be wrong. Requiring 80% short labels keeps it to genuine matching output.
    """
    vals = [a for a in answers if a and a.strip()]
    if len(vals) < MIN_ROWS_FOR_ASSIGNMENT:
        return False
    hits = sum(1 for a in vals if LETTERISH.match(a.strip()))
    return hits >= min_frac * len(vals)


def assign(  # noqa: PLR0911 — every early return is a distinct refusal-to-guess condition,
    # and collapsing them would hide WHICH guard fired when this falls back to majority
    candidates: list[list[str]],
    weights: list[list[float]],
    *,
    fallback: list[str],
) -> list[str]:
    """Return one label per item, maximising total confidence with each label used once.

    `candidates[i]` / `weights[i]` are the labels proposed for item i across samples and their
    confidences (same length, aligned). `fallback` is returned unchanged if anything is off —
    scipy missing, degenerate matrix, fewer labels than items — because a matching row that keeps
    its majority answer is merely unimproved, while a crash loses the whole run.
    """
    k = len(candidates)
    if k < MIN_ROWS_FOR_ASSIGNMENT or len(weights) != k:
        return fallback
    canon = [[canonical(c) for c in row if canonical(c)] for row in candidates]
    labels = sorted({c for row in canon for c in row})
    # canonical -> the spelling the model actually used most often. We match on the canonical
    # form but EMIT the model's spelling: EM is case-insensitive (the scorer lowercases) but
    # chrF is not, so rewriting "A" as "a" could give away partial credit for nothing.
    spelling: dict[str, Counter] = {lab: Counter() for lab in labels}
    for row in candidates:
        for raw in row:
            c = canonical(raw)
            if c in spelling:
                spelling[c][(raw or "").strip()] += 1
    display = {
        lab: (spelling[lab].most_common(1)[0][0] if spelling[lab] else lab) for lab in labels
    }
    if len(labels) < k:
        # fewer distinct options than items: a bijection is impossible, so leave it alone rather
        # than inventing labels the model never proposed
        return fallback
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError:  # pragma: no cover - scipy IS in the eval image (verified)
        return fallback
    index = {lab: j for j, lab in enumerate(labels)}
    mat = np.zeros((k, len(labels)), dtype=float)
    for i, (row, wrow) in enumerate(zip(candidates, weights, strict=True)):
        for lab, w in zip(row, wrow, strict=True):
            lab = canonical(lab)
            if lab and w > 0:
                mat[i, index[lab]] += w
    if not mat.any() or any(not mat[i].any() for i in range(k)):
        return fallback
    try:
        _, cols = linear_sum_assignment(-mat)
    except ValueError:  # pragma: no cover - shape guards above make this unreachable
        return fallback
    if any(mat[i, j] <= 0 for i, j in enumerate(cols) if i < k):
        return fallback
    out = list(fallback)
    for i, j in enumerate(cols):
        if i < len(out):
            out[i] = display[labels[j]]
    return out
