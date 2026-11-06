"""Align token log probabilities to answer items for weighted voting. Missing or ambiguous alignments fall back to unweighted voting."""

from __future__ import annotations

import json
import math

# A token span counts toward an item if it overlaps the item's bytes at all:
# tokenizers routinely merge a closing quote or comma into the last content
# token, so demanding containment would drop the final token of most items.


def token_byte_bounds(tlens: list[int]) -> list[tuple[int, int]]:
    """Byte [start, end) of each token within the concatenated text."""
    bounds = []
    pos = 0
    for n in tlens:
        bounds.append((pos, pos + n))
        pos += n
    return bounds


def _char_to_byte_prefix(text: str) -> list[int]:
    """Byte offset of every char index (len+1 entries; UTF-8 is variable width)."""
    pref = [0] * (len(text) + 1)
    for i, ch in enumerate(text):
        pref[i + 1] = pref[i] + len(ch.encode("utf-8"))
    return pref


def meta_ok(text: str, meta: object) -> bool:
    """Return True when meta's token arrays really describe `text`.

    The driver JSON-dumps text with error_handler=replace, so a generation cut
    mid-multibyte comes back with U+FFFD substituted and the byte lengths no
    longer add up. That is exactly when alignment would silently mis-attribute
    logprobs, so it is a hard reject rather than a best effort.
    """
    if not isinstance(meta, dict) or not meta:
        return False
    lps, tlens = meta.get("lps"), meta.get("tlens")
    if not isinstance(lps, list) or not isinstance(tlens, list) or len(lps) != len(tlens):
        return False
    if not lps:
        return False
    if not all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in lps
    ):
        return False
    if not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in tlens):
        return False
    return sum(tlens) == len(text.encode("utf-8"))


def seq_weight(meta: object, *, lp_t: float = 1.0) -> float | None:
    """Length-normalized sequence confidence: exp(mean per-token logprob / T)."""
    if not isinstance(meta, dict) or not meta:
        return None
    lps = meta.get("lps")
    if isinstance(lps, list) and lps:
        return math.exp(sum(lps) / len(lps) / max(1e-6, lp_t))
    # driver may have emitted only the aggregate (error salvage paths)
    lp_sum, n = meta.get("lp_sum"), meta.get("n_decoded")
    if isinstance(lp_sum, (int, float)) and isinstance(n, int) and n > 0:
        return math.exp(float(lp_sum) / n / max(1e-6, lp_t))
    return None


def json_array_spans(text: str) -> tuple[list[str], list[tuple[int, int]]] | None:
    """Elements of the LAST parsable all-scalar JSON array, with char spans.

    Mirrors textproc._json_preds' choice of array (raw_decode from each "[",
    last first) so the spans describe the SAME elements that became answers.
    Returns the decoded values and the span of each element's literal
    (quotes excluded for strings — the content bytes are what carry the answer).
    """
    dec = json.JSONDecoder()
    for i in range(len(text) - 1, -1, -1):
        if text[i] != "[":
            continue
        try:
            value, end = dec.raw_decode(text, i)
        except ValueError:
            continue
        scalars = (
            isinstance(value, list)
            and value
            and all(isinstance(x, (str, int, float)) for x in value)
        )
        if not scalars:
            continue
        spans = _scan_scalar_spans(text, i, end)
        if spans is None or len(spans) != len(value):
            return None  # scanner and decoder disagree -> refuse to guess
        return [str(x) for x in value], spans
    return None


def _scan_scalar_spans(text: str, start: int, end: int) -> list[tuple[int, int]] | None:
    """Char spans of the depth-1 scalar literals in text[start:end] ('[' at start).

    Returns None on anything the caller shouldn't trust: nesting, a bad literal,
    or an unterminated string. (An all-scalar array is the only shape
    _json_preds accepts, so nesting here means we found a different array.)
    """
    spans: list[tuple[int, int]] = []
    i = start + 1
    while i < end:
        ch = text[i]
        if ch in " \t\r\n,":
            i += 1
            continue
        if ch == "]":
            return spans
        if ch == '"':
            j = i + 1
            while j < end:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == '"':
                    break
                j += 1
            if j >= end:
                return None  # unterminated
            spans.append((i + 1, j))  # inner content, quotes excluded
            i = j + 1
            continue
        if ch in "-+.0123456789":  # bare number
            j = i
            while j < end and text[j] not in ",]":
                j += 1
            spans.append((i, len(text[i:j].rstrip()) + i))
            i = j
            continue
        return None  # object/array/literal -> not the all-scalar array we parsed
    return None  # ran off the end without a closing bracket


def _delivery_region(text: str) -> tuple[str, int] | None:
    """Return the slice parse_answers actually read, plus its offset in `text`.

    parse_answers strips thinking blocks and then searches only AFTER the last
    FINAL ANSWERS marker. Weights must be taken from the SAME region or a
    reasoning-embedded array wins the span scan while the real delivery wins the
    parse. Stripping would shift every offset, so a text that
    actually contains a thinking block is refused rather than mis-mapped — the
    trained direct-JSON format has none, and refusing means weight 1.0."""
    from . import textproc

    if textproc.strip_thinking(text) != text:
        return None
    marks = list(textproc.FINAL_RE.finditer(text))
    base = marks[-1].end() if marks else 0
    return text[base:], base


def _sequential_spans(region: str, answers: list[str]) -> list[tuple[int, int] | None]:
    """Locate answers left to right, never reusing bytes.

    A per-answer rfind() puts item i's span inside item j's text whenever one
    answer is a substring of another ("a" and "ba" both resolve into "ba"), which
    silently swaps their confidences. Scanning forward from the previous match
    keeps the spans ordered and disjoint; anything not found from there is None."""
    out: list[tuple[int, int] | None] = []
    cursor = 0
    for a in answers:
        a = a.strip()
        pos = region.find(a, cursor) if a else -1
        if pos < 0:
            out.append(None)
            continue
        out.append((pos, pos + len(a)))
        cursor = pos + len(a)
    return out


def _item_spans(text: str, answers: list[str]) -> list[tuple[int, int] | None]:
    """Char span of each answer inside `text`, or None where it can't be located.

    Fast path: the answers came from the trained JSON tail, so the array scan
    lines up 1:1. Fallback: locate them in order within the delivery region.
    """
    k = len(answers)
    reg = _delivery_region(text)
    if reg is None:
        return [None] * k
    region, base = reg
    js = json_array_spans(region)
    if js is not None:
        values, spans = js
        # compare against the same padded/truncated projection parse_answers used
        proj = (values + [""] * k)[:k]
        if [p.strip() for p in proj] == [a.strip() for a in answers]:
            return [
                (spans[i][0] + base, spans[i][1] + base) if i < len(spans) else None
                for i in range(k)
            ]
    return [(s[0] + base, s[1] + base) if s else None for s in _sequential_spans(region, answers)]


def item_weights(
    text: str, answers: list[str], meta: object, *, lp_t: float = 1.0
) -> list[float | None]:
    """Per-item confidence: exp(mean logprob of the item's own tokens / T).

    None for an item whose tokens can't be identified — the caller must treat
    that as "no opinion" (fall back to counting), never as zero confidence.
    """
    k = len(answers)
    if not isinstance(meta, dict) or not meta_ok(text, meta):
        return [None] * k
    lps: list[float] = meta["lps"]
    bounds = token_byte_bounds(meta["tlens"])
    bpref = _char_to_byte_prefix(text)
    spans = _item_spans(text, answers)
    t = max(1e-6, lp_t)
    out: list[float | None] = []
    for span in spans:
        if span is None or span[0] >= span[1]:
            out.append(None)
            continue
        b0, b1 = bpref[span[0]], bpref[min(span[1], len(text))]
        picked = [lps[i] for i, (tb0, tb1) in enumerate(bounds) if tb0 < b1 and tb1 > b0]
        out.append(math.exp(sum(picked) / len(picked) / t) if picked else None)
    return out


def sample_score(text: str, answers: list[str], meta: object, *, lp_t: float = 1.0) -> float | None:
    """Whole-sample confidence for best-of-N: mean of its per-item confidences.

    Item-mean, not sequence-mean, so a sample isn't rewarded for a fluent RULES
    block it wrote before guessing the answers.
    """
    ws = [w for w in item_weights(text, answers, meta, lp_t=lp_t) if w is not None]
    if ws:
        return sum(ws) / len(ws)
    return seq_weight(meta, lp_t=lp_t)
