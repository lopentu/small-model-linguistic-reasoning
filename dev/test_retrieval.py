"""Unit tests for solver/v2/retrieval.py (the E31 dense gramquery pass).

Covers the pure, shippable logic — ISO parsing, exact/family resolution, addon
formatting, and the FAIL-CLOSED contract (default-off, missing data -> {}) — using
a tiny synthetic corpus so no embedder or GPU is needed. The full query-embedding
search path is integration-tested on the A5000 (see dev/local), not here.

Run:  uv run python dev/test_retrieval.py
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solver"))

import numpy as np

import v2.retrieval as R


def _reset() -> None:
    R._CORPUS = None
    R._TRIED = False


@contextmanager
def _tmpdir() -> Iterator[Path]:
    """Temp dir whose cleanup tolerates a still-mmapped vectors file (Windows locks it)."""
    d = Path(tempfile.mkdtemp())
    try:
        yield d
    finally:
        _reset()
        gc.collect()  # release any numpy memmap holding the vectors file open
        shutil.rmtree(d, ignore_errors=True)


def _fake_corpus(d: Path) -> None:
    """Write a minimal corpus: 2 langs, 4 chunks, 8-dim unit vectors."""
    chunks = [
        {"glottocode": "dido1241", "text": "ergative -k marks the agent NOM ABS"},
        {"glottocode": "dido1241", "text": "vowel harmony spreads backness"},
        {"glottocode": "araw1276", "text": "possessive prefix nu- 1SG on nouns"},
        {"glottocode": "araw1276", "text": "verb takes subject agreement -ka"},
    ]
    (d / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in chunks), encoding="utf-8")
    vecs = np.eye(4, 8, dtype=np.float16)  # 4 distinct unit rows
    np.save(d / "vectors.f16.npy", vecs)
    index = {
        "dido1241": {"row0": 0, "row1": 2, "iso": "ddo", "family": "Nakh-Daghestanian", "names": ["Tsez"]},
        "araw1276": {"row0": 2, "row1": 4, "iso": "lok", "family": "Arawakan", "names": ["Lokono"]},
    }
    (d / "lang_index.json").write_text(json.dumps(index), encoding="utf-8")
    (d / "iso_to_glottocode.json").write_text(json.dumps({"ddo": "dido1241", "lok": "araw1276"}), encoding="utf-8")
    # apu is NOT a corpus language but shares Arawakan -> should family-fallback
    (d / "iso_to_family.json").write_text(json.dumps({"apu": "Arawakan"}), encoding="utf-8")


def test_parse_iso() -> None:
    assert R._parse_iso("['apu_Latn']") == "apu"
    assert R._parse_iso("['qda-gua_Latn']") == "qda-gua"
    assert R._parse_iso("crk_Latn") == "crk"
    assert R._parse_iso("CRK") == "crk"
    assert R._parse_iso("") == ""
    assert R._parse_iso("[]") == ""


def test_iso_candidates() -> None:
    assert R._iso_candidates("apu") == ["apu"]
    assert R._iso_candidates("qda-gua") == ["qda-gua", "qda"]
    assert R._iso_candidates("") == []


def test_resolve_exact_and_family() -> None:
    with _tmpdir() as d:
        _fake_corpus(d)
        c = R._Corpus(d)
        assert c.resolve("ddo") == ("dido1241", "exact")
        assert c.resolve("apu") == ("araw1276", "family")  # via shared Arawakan family
        assert c.resolve("zzz") == ("", "")


def test_format_labels() -> None:
    exact = R._format(["a b c"], "Tsez", "exact", 1000)
    fam = R._format(["a b c"], "Lokono", "family", 1000)
    assert "Tsez" in exact and "same language family" not in exact
    assert "Lokono" in fam and "same language family" in fam
    # token cap respected
    capped = R._format(["x " * 500], "Tsez", "exact", 200)
    assert len(capped) <= 220


def test_default_off() -> None:
    # GRAMQUERY unset -> retrieve_all is a hard no-op even with data present
    os.environ.pop("GRAMQUERY", None)
    _reset()
    with _tmpdir() as d:
        _fake_corpus(d)
        os.environ["GRAMQUERY_DIR"] = str(d)
        try:
            assert R.retrieve_all([{"id": "p1", "task_lang": "['ddo_Cyrl']"}]) == {}
        finally:
            os.environ.pop("GRAMQUERY_DIR", None)
            _reset()


def test_fail_closed_missing_dir() -> None:
    os.environ["GRAMQUERY"] = "1"
    os.environ["GRAMQUERY_DIR"] = "/nonexistent/path/xyz"
    _reset()
    try:
        assert R.retrieve_all([{"id": "p1", "task_lang": "['ddo_Cyrl']"}]) == {}
    finally:
        os.environ.pop("GRAMQUERY", None)
        os.environ.pop("GRAMQUERY_DIR", None)
        _reset()


def test_fail_closed_no_embedder() -> None:
    # enabled + valid corpus but NO embedder.gguf -> still "" (never raises)
    os.environ["GRAMQUERY"] = "1"
    _reset()
    with _tmpdir() as d:
        _fake_corpus(d)  # no embedder.gguf written
        os.environ["GRAMQUERY_DIR"] = str(d)
        try:
            assert R.retrieve_all([{"id": "p1", "task_lang": "['ddo_Cyrl']"}]) == {}
        finally:
            os.environ.pop("GRAMQUERY", None)
            os.environ.pop("GRAMQUERY_DIR", None)
            _reset()


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"OK: {len(tests)} retrieval tests passed")


if __name__ == "__main__":
    main()
