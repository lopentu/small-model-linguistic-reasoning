"""Offline dense grammar-example retrieval (E31 — semantic gramquery pass).

Given a puzzle whose target language is ``task_lang`` (a FLORES / ISO-639-3 code
such as ``['apu_Latn']``), retrieve the reference-grammar chunks of that language
(or its closest same-family relative) that are most SIMILAR to the puzzle's own
data, and inject the top few as grounding.

This is the E31 successor to E29's static per-language dump. The selection is now
query-dependent: a small GGUF embedder (Qwen3-Embedding-0.6B, CPU) embeds each
puzzle, and we cosine-search that puzzle's vector against the matched language's
precomputed chunk vectors. Everything is offline — the embedder ships in the repo,
the corpus vectors are precomputed by dev/gramquery_embed.py with the SAME
embedder (one shared vector space), and the search is a numpy dot product. No
network, no LanceDB, no runtime embedding of the corpus.

VRAM: the embedder runs on CPU (n_gpu_layers=0), so the retrieval pre-pass never
touches the GPU and never competes with the resident 26B solver. It is meant to
run ONCE over all rows before the solver model loads (see script.main).

DEFAULT-OFF and fully FAIL-CLOSED: with GRAMQUERY unset, or any missing data file
/ embedder / import error, retrieve_all() returns {} and the solver runs exactly
as if this module did not exist.

Env:
  GRAMQUERY=1                  -> enable (default DISABLED)
  GRAMQUERY_DIR=<path>         -> data dir (default <repo>/data/gramquery) holding
                                  chunks.jsonl, vectors.f16.npy, lang_index.json,
                                  iso_to_glottocode.json, iso_to_family.json
  GRAMQUERY_EMBED_MODEL=<path> -> embedder GGUF (default <data_dir>/embedder.gguf)
"""

from __future__ import annotations

import ast
import gc
import json
import os
from pathlib import Path

_QUERY_TASK = "Retrieve reference-grammar passages relevant to this language sample"


def _data_dir() -> Path:
    env = os.environ.get("GRAMQUERY_DIR", "")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent.parent / "data" / "gramquery"


def _parse_iso(task_lang: str) -> str:
    """Pull the ISO-639-3 code out of a task_lang cell (``['apu_Latn']`` -> apu)."""
    raw = (task_lang or "").strip()
    if not raw:
        return ""
    val = raw
    if raw.startswith("["):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)) and parsed:
                val = str(parsed[0])
            else:
                return ""
        except (ValueError, SyntaxError):
            val = raw.strip("[]'\" ")
    return val.split("_", 1)[0].strip().lower()


def _iso_candidates(iso: str) -> list[str]:
    if not iso:
        return []
    cands = [iso]
    if "-" in iso:  # FLORES compound code e.g. qda-gua -> also try base
        cands.append(iso.split("-", 1)[0])
    return cands


class _Corpus:
    """Loaded-once retrieval store; construction is guarded/fail-closed by load()."""

    def __init__(self, d: Path) -> None:
        import numpy as np  # noqa: PLC0415

        with (d / "chunks.jsonl").open(encoding="utf-8") as fh:
            self.texts = [json.loads(line)["text"] for line in fh]
        self.vectors = np.load(d / "vectors.f16.npy", mmap_mode="r")
        self.index = json.loads((d / "lang_index.json").read_text(encoding="utf-8"))
        self.iso2glotto = json.loads((d / "iso_to_glottocode.json").read_text(encoding="utf-8"))
        self.iso2family = json.loads((d / "iso_to_family.json").read_text(encoding="utf-8"))
        if self.vectors.shape[0] != len(self.texts):
            msg = "vectors/chunks row mismatch"
            raise ValueError(msg)
        # richest-first language per family, for the fallback path
        self.by_family: dict[str, list[str]] = {}
        for code, rec in self.index.items():
            self.by_family.setdefault(rec.get("family", ""), []).append(code)
        for langs in self.by_family.values():
            langs.sort(key=lambda c: self.index[c]["row1"] - self.index[c]["row0"], reverse=True)

    def resolve(self, iso: str) -> tuple[str, str]:
        """ISO -> (glottocode, relation) using exact match then family fallback."""
        for c in _iso_candidates(iso):
            code = self.iso2glotto.get(c)
            if code and code in self.index:
                return code, "exact"
        for c in _iso_candidates(iso):
            fam = self.iso2family.get(c, "")
            langs = self.by_family.get(fam, []) if fam else []
            if langs:
                return langs[0], "family"
        return "", ""


_CORPUS: _Corpus | None = None
_TRIED = False


def _load() -> _Corpus | None:
    """Load the corpus once; fail-closed to None (disabled)."""
    global _CORPUS, _TRIED  # noqa: PLW0603 — one-shot module cache
    if _TRIED:
        return _CORPUS
    _TRIED = True
    if os.environ.get("GRAMQUERY", "0") != "1":  # DEFAULT-OFF
        return None
    try:
        _CORPUS = _Corpus(_data_dir())
    except (OSError, ValueError, KeyError, ImportError):
        _CORPUS = None
    return _CORPUS


def _embed_queries(queries: list[str]) -> object | None:
    """Embed all query strings on CPU with the shipped GGUF embedder; None on failure."""
    model = os.environ.get("GRAMQUERY_EMBED_MODEL", "") or str(_data_dir() / "embedder.gguf")
    if not Path(model).exists():
        return None
    try:
        import numpy as np  # noqa: PLC0415
        from llama_cpp import Llama  # noqa: PLC0415

        llm = Llama(model, embedding=True, n_gpu_layers=0, n_ctx=1024, n_batch=1024, verbose=False)
        try:
            out = np.asarray(llm.embed(queries), dtype=np.float32)
        finally:
            del llm
            gc.collect()
        if out.ndim == 3:  # per-token pooling fallback
            out = out.mean(axis=1)
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)
    except Exception:  # noqa: BLE001 — fail-closed: no retrieval beats a crash
        return None


def _format(texts: list[str], name: str, relation: str, max_chars: int) -> str:
    if relation == "family":
        header = (
            f"Background — reference-grammar examples from {name}, a relative in the same "
            f"language family (the puzzle language may differ; the data below is self-contained, "
            f"use these only as a hint about likely structure):"
        )
    else:
        header = (
            f"Background — reference-grammar examples for {name} (the puzzle is still "
            f"self-contained; use these only as a hint about the language's structure):"
        )
    lines = [header]
    used = len(header)
    for t in texts:
        snippet = " ".join((t or "").split())
        if not snippet:
            continue
        piece = f"- {snippet}"
        if used + len(piece) > max_chars:
            piece = piece[: max(0, max_chars - used)]
            if len(piece) < 10:  # truncation left nothing useful — stop here
                break
        lines.append(piece)
        used += len(piece)
    return "\n".join(lines) if len(lines) > 1 else ""


def retrieve_all(
    rows: list[dict[str, str]], *, k: int = 2, max_chars: int = 1000
) -> dict[str, str]:
    """Return {row_id: grounding_addon} for every row that resolves; else omitted.

    Fail-closed: returns {} on any disable flag, missing data, or embedder failure.
    Intended to run ONCE before the solver model loads (embedder is CPU-only).
    """
    corpus = _load()
    if corpus is None or not rows:
        return {}

    # resolve each row's language once; only embed rows that hit a corpus language
    resolved: dict[str, tuple[str, str]] = {}
    for r in rows:
        code, relation = corpus.resolve(_parse_iso(r.get("task_lang", "")))
        if code:
            resolved[r["id"]] = (code, relation)
    if not resolved:
        return {}

    by_id = {r["id"]: r for r in rows}
    ids = list(resolved)

    def _query(r: dict[str, str]) -> str:
        sample = (r.get("context", "") + " " + r.get("query", "")).strip()[:1500]
        return f"Instruct: {_QUERY_TASK}\nQuery:{sample}"

    qvecs = _embed_queries([_query(by_id[i]) for i in ids])
    if qvecs is None:
        return {}

    import numpy as np  # noqa: PLC0415

    out: dict[str, str] = {}
    for pos, rid in enumerate(ids):
        code, relation = resolved[rid]
        rec = corpus.index[code]
        r0, r1 = rec["row0"], rec["row1"]
        if r1 <= r0:
            continue
        sims = np.asarray(corpus.vectors[r0:r1], dtype=np.float32) @ qvecs[pos]
        top = np.argsort(-sims)[:k]
        texts = [corpus.texts[r0 + int(j)] for j in top]
        name = (rec.get("names") or [code])[0]
        addon = _format(texts, name, relation, max_chars)
        if addon:
            out[rid] = addon
    return out
