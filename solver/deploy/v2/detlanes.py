"""Deterministic answer lanes, routed on the free task_type column.

Two lanes, both measured on the 894-item Linguini dev set (26 Jul):

- fill_blanks OVERRIDE: fb_solver.RowSolver (paradigm-table transducer cascade,
  route picked by leave-one-out on the row's OWN attested pairs) replaces the
  model's answer wherever it produces a candidate. Measured EM 0.2500 /
  chrF 0.5474 on all 132 dev fill_blanks items, vs a model lane that lives
  near its 0.10 overall EM.
- translation FLOOR: never-blank pair retrieval (emit the other side of the
  context pair whose side best char-3-gram-matches the query item) fills ONLY
  item slots the model left blank. Measured chrF 0.287 / EM 0.002 standalone
  on all 474 dev translation items — above the whole solver's 0.266 chrF, so
  it strictly beats the generic fallback for this task type.

Design rules: fail-closed (any exception returns the input answers unchanged),
cached per row id (aggregate() runs on every flush), wall-capped (a per-row
thread guard plus a global spend cap — a pathological hidden row must never
eat the T4 budget), and DETLANES=0 disables the whole module.
"""

from __future__ import annotations

import os
import re
import threading
import unicodedata
from collections import Counter

from . import fb_solver

# ---------- caps ----------

ROW_TIMEOUT_S = 20.0  # per fill_blanks row (dev worst case ~5s; hung thread is daemon)
TOTAL_BUDGET_S = 240.0  # cumulative cap across all fill_blanks rows
_spent = [0.0]  # mutable cell: cumulative fb-solver wall
_fb_cache: dict[str, list[str] | None] = {}
_tr_cache: dict[str, list[str] | None] = {}


def _enabled() -> bool:
    return os.environ.get("DETLANES", "1").strip() != "0"


# ---------- fill_blanks lane ----------


def _fb_solve_row(context: str, query: str, k: int) -> list[str] | None:
    """Run RowSolver in a daemon thread with a wall cap; None on any failure."""
    out: list[list[str] | None] = [None]

    def work() -> None:
        try:
            rs = fb_solver.RowSolver(context, query)
            keys = sorted(rs.blanks)
            # Blank markers usually run 1..k; when numbering continues across
            # rows (dataset note: absolute item numbers) map markers to slots
            # positionally instead of by value.
            if len(keys) == k and keys != list(range(1, k + 1)):
                slot_key = dict(enumerate(keys))
            else:
                slot_key = {i: i + 1 for i in range(k)}
            preds = []
            for i in range(k):
                n = slot_key.get(i)
                if n in rs.blanks:
                    preds.append(fb_solver.nfc(rs.predict_blank(n)[0] or ""))
                else:
                    preds.append("")
            out[0] = preds
        except Exception:  # noqa: BLE001, S110 — lane is optional; model path remains
            pass

    t = threading.Thread(target=work, daemon=True)
    import time

    t0 = time.time()
    t.start()
    t.join(ROW_TIMEOUT_S)
    _spent[0] += time.time() - t0
    if t.is_alive():
        return None  # abandoned; daemon thread may finish but result is unused
    return out[0]


def _fb_answers(row: dict, k: int) -> list[str] | None:
    rid = str(row.get("id", ""))
    if rid not in _fb_cache:
        if _spent[0] >= TOTAL_BUDGET_S:
            return None  # over global budget: stop starting new rows, keep cache misses uncached
        _fb_cache[rid] = _fb_solve_row(row.get("context", ""), row.get("query", ""), k)
    return _fb_cache[rid]


# ---------- translation floor lane ----------

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_ENUM_RE = re.compile(r"^\s*(?:\(\d{1,3}\)|\d{1,3}[.)])\s*")
_LEAD_RE = re.compile(r"(?m)^\s*\d{1,3}[.)]\s")
_PAREN_RE = re.compile(r"\(\d{1,3}\)")
_SEPS = ["|", "\t", " — ", " – ", " = ", " ‘", ": ", " - "]


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", str(s))


def _collapse(s: str) -> str:
    return re.sub(r"\s+", " ", _nfc(s)).strip()


def _grams(s: str, n: int = 3) -> Counter:
    s = _collapse(s).casefold()
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def _dice(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    ov = sum(min(c, b[g]) for g, c in a.items())
    return 2 * ov / (sum(a.values()) + sum(b.values()))


def _split_items(query: str, k: int) -> list[str]:
    """Per-item source text; layered markers -> lines -> whole query (never fails)."""
    lead = list(_LEAD_RE.finditer(query))
    if lead:
        segs = []
        for i, m in enumerate(lead):
            end = lead[i + 1].start() if i + 1 < len(lead) else len(query)
            segs.append(_collapse(query[m.end() : end]))
        while len(segs) < k:  # one marker sharing two comma-joined items
            for i, s in enumerate(segs):
                if "," in s:
                    a, b = s.split(",", 1)
                    segs[i : i + 1] = [a.strip(), b.strip()]
                    break
            else:
                segs.append("")
        return segs[:k]
    paren = list(_PAREN_RE.finditer(query))
    if len(paren) == k:
        segs = []
        for i, m in enumerate(paren):
            end = paren[i + 1].start() if i + 1 < len(paren) else len(query)
            segs.append(_collapse(query[m.end() : end]))
        return segs
    lines = [ln for ln in query.splitlines() if ln.strip()]
    if k == 1:
        return [_collapse(" ".join(lines[1:]) if len(lines) >= 2 else query)]
    body = [_collapse(ln) for ln in lines[1:]]
    return (body + [""] * k)[:k]


def _extract_pairs(ctx_text: str) -> list[tuple[str, str]]:
    pairs = []
    for ln in ctx_text.splitlines():
        ln = _ENUM_RE.sub("", ln).strip()
        if not ln:
            continue
        for sep in _SEPS:
            if sep in ln:
                a, b = ln.split(sep, 1)
                a = a.strip().strip("‘’“”\"'")
                b = b.strip().strip("‘’“”\"'")
                if a and b:
                    pairs.append((a, b))
                break
    return pairs


def _tr_floor(row: dict, k: int) -> list[str] | None:
    rid = str(row.get("id", ""))
    if rid not in _tr_cache:
        try:
            ctx_text = row.get("context", "")
            items = _split_items(row.get("query", ""), k)
            pairs = _extract_pairs(ctx_text)
            toks = _TOKEN_RE.findall(ctx_text)
            last_resort = max(toks, key=len) if toks else ""
            preds = []
            for it in items:
                if not pairs or not it:
                    preds.append(last_resort)
                    continue
                g = _grams(it)
                best, bs = last_resort, -1.0
                for a, b in pairs:
                    for side, other in ((a, b), (b, a)):
                        s = _dice(g, _grams(side))
                        if s > bs:
                            bs, best = s, other
                preds.append(_collapse(best))
            _tr_cache[rid] = preds
        except Exception:  # noqa: BLE001 — optional lane
            _tr_cache[rid] = None
    return _tr_cache[rid]


# ---------- entry point ----------


def apply(row: dict, final: list[str], k: int) -> list[str]:
    """Route on task_type; return possibly-improved answers. Never raises."""
    try:
        if not _enabled() or k <= 0:
            return final
        tt = str(row.get("task_type", "")).strip()
        if tt == "fill_blanks":
            det = _fb_answers(row, k)
            if det:
                return [d if d.strip() else final[i] for i, d in enumerate(det)]
        elif tt == "translation" and any(not v.strip() for v in final):
            floor = _tr_floor(row, k)
            if floor:
                return [v if v.strip() else floor[i] for i, v in enumerate(final)]
    except Exception:  # noqa: BLE001 — a lane bug must never sink aggregation
        return final
    return final
