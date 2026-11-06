"""Extract grammar chunks from a user-supplied LanceDB corpus and Glottolog metadata."""

from __future__ import annotations

import argparse
import heapq
import json
import os
import re
import time
from pathlib import Path

# --- interlinear-gloss / example scoring (quality filter, not the ranker) -----

_GLOSS_TAGS = (
    "SG PL DU NOM ACC GEN DAT ERG ABS LOC INS COM ABL ALL ILL ESS PST PRS FUT "
    "PFV IPFV PROG PERF AOR NEG DEF INDF SBJ OBJ POSS CAUS PASS REFL RECP REL "
    "COP AUX TOP FOC IMP CVB PTCP INF GER PRF HAB IRR SBJV COND VOC CLF HON "
    "DIM AUG DIST PROX ANIM INAN MASC FEM NEUT DECL INTR TR DETR APPL ANTIP"
).split()
_GLOSS_RE = re.compile(r"\b(?:[123])?(?:" + "|".join(_GLOSS_TAGS) + r")\b")
_MORPH_RE = re.compile(r"\w[-=]\w")


def example_score(text: str) -> float:
    """Higher = more like glossed example sentences / paradigms, less like prose."""
    return len(_GLOSS_RE.findall(text)) + 0.4 * len(_MORPH_RE.findall(text))


# --- glottolog iso639-3 -> glottocode ----------------------------------------

_ISO_RE = re.compile(r"^\s*iso639-3\s*=\s*([a-z]{3})\s*$", re.MULTILINE)


def build_iso_map(glottolog_root: str) -> dict[str, str]:
    """Walk the raw Glottolog tree; map iso639-3 -> glottocode (leaf dir name)."""
    tree = Path(glottolog_root) / "languoids" / "tree"
    iso_to_glotto: dict[str, str] = {}
    for dirpath, _dirs, files in os.walk(tree):
        if "md.ini" not in files:
            continue
        try:
            txt = Path(dirpath, "md.ini").read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        m = _ISO_RE.search(txt)
        if m:
            iso_to_glotto[m.group(1)] = os.path.basename(dirpath)
    return iso_to_glotto


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lancedb", required=True)
    ap.add_argument("--table", default="reference_grammar_chunks")
    ap.add_argument("--glottolog", required=True)
    ap.add_argument("--meta", required=True, help="glottocode_to_glottolog_metadata.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-lang", type=int, default=200, help="max chunks kept per glottocode")
    ap.add_argument("--min-score", type=float, default=3.0, help="drop chunks below this example-score")
    ap.add_argument("--max-chars", type=int, default=900, help="truncate each stored chunk")
    ap.add_argument("--batch", type=int, default=50000)
    args = ap.parse_args()

    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] building iso639-3 -> glottocode map ...", flush=True)
    iso_map = build_iso_map(args.glottolog)
    print(f"      {len(iso_map)} iso codes mapped", flush=True)
    (out / "iso_to_glottocode.json").write_text(json.dumps(iso_map, ensure_ascii=False), encoding="utf-8")

    meta = json.loads(Path(os.path.expanduser(args.meta)).read_text(encoding="utf-8"))
    iso2fam = {iso: (meta.get(glc, {}).get("family") or "") for iso, glc in iso_map.items()}
    (out / "iso_to_family.json").write_text(
        json.dumps({k: v for k, v in iso2fam.items() if v}, ensure_ascii=False), encoding="utf-8"
    )

    print("[2/4] streaming LanceDB (text columns only, no vectors) ...", flush=True)
    import lancedb

    db = lancedb.connect(os.path.expanduser(args.lancedb))
    ds = db.open_table(args.table).to_lance()
    cols = ["glottocodes", "family", "original_text", "headings"]

    heaps: dict[str, list] = {}
    fam_of: dict[str, str] = {}
    seen = tie = 0
    t0 = time.time()
    for batch in ds.scanner(columns=cols, batch_size=args.batch).to_batches():
        d = batch.to_pydict()
        glc, fam, txts = d["glottocodes"], d["family"], d["original_text"]
        for i in range(len(txts)):
            seen += 1
            codes = glc[i]
            text = txts[i] or ""
            if not codes:
                continue
            sc = example_score(text)
            if sc < args.min_score:
                continue
            code = codes[0]
            fam_of[code] = fam[i] or ""
            tie += 1
            h = heaps.setdefault(code, [])
            item = (sc, tie, text[: args.max_chars])
            if len(h) < args.per_lang:
                heapq.heappush(h, item)
            elif sc > h[0][0]:
                heapq.heapreplace(h, item)
        if seen % 200000 == 0:
            print(f"      {seen:>9,} rows | {len(heaps)} langs | {time.time()-t0:.0f}s", flush=True)
    print(f"      scanned {seen:,} rows in {time.time()-t0:.0f}s; {len(heaps)} languages", flush=True)

    print("[3/4] writing flat chunks.jsonl + lang_index.json ...", flush=True)
    glotto_to_iso = {v: k for k, v in iso_map.items()}
    index: dict[str, dict] = {}
    row = 0
    n_chunks = 0
    with (out / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for code in sorted(heaps):
            chunks = [t for _s, _t, t in sorted(heaps[code], key=lambda x: -x[0])]
            row0 = row
            for text in chunks:
                f.write(json.dumps({"glottocode": code, "text": text}, ensure_ascii=False) + "\n")
                row += 1
            m = meta.get(code, {})
            index[code] = {
                "row0": row0,
                "row1": row,
                "iso": glotto_to_iso.get(code, ""),
                "family": m.get("family", fam_of.get(code, "")) or fam_of.get(code, ""),
                "names": (m.get("languoid_names") or [])[:6],
            }
            n_chunks += len(chunks)
    (out / "lang_index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    size_mb = (out / "chunks.jsonl").stat().st_size / 1e6
    print(
        f"[4/4] DONE: {len(index)} languages, {n_chunks:,} chunks, {size_mb:.1f} MB\n"
        f"      next: dev/gramquery_embed.py --chunks {out/'chunks.jsonl'} --model <embedder.gguf>",
        flush=True,
    )


if __name__ == "__main__":
    main()
