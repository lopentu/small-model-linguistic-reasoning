"""Build training examples and validate delete-only compressed traces from user-supplied review data. See docs/training.md."""

from __future__ import annotations

import argparse
import os
import ast
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "solver"))

from v2 import prompts, textproc  # noqa: E402

SFT_DIR = REPO / "dev" / "sft"
WORK = SFT_DIR / "work"
DATASET = SFT_DIR / "dataset"
CORPUS = Path(os.environ.get("REVIEW_CORPUS", "data/reviewed_only.jsonl"))

FIXTURE = REPO / "dev" / "mock-eval" / "fixtures" / "test.csv"

# ---------------------------------------------------------------- prompt tiers

_BRIEF_CLAUSE = "Be brief: keep all reasoning under 300 words, then IMMEDIATELY output, "
_LONG_CLAUSE = (
    "Reason as thoroughly as the problem demands - depth is welcome. When your "
    "analysis is complete, output, "
)


def system_long() -> str:
    """Tier-A system prompt: v2 SYSTEM with the brevity clause swapped for an
    unlimited-reasoning one. Hard-fails if prompts.py drifts (sync guard)."""
    if _BRIEF_CLAUSE not in prompts.SYSTEM:
        raise SystemExit("prompts.SYSTEM drifted: brevity clause not found - update this file")
    return prompts.SYSTEM.replace(_BRIEF_CLAUSE, _LONG_CLAUSE)


_COMPACT_CLAUSE = (
    "Be efficient: compress your reasoning to the essential observations and "
    "derivations, then IMMEDIATELY output, "
)


def system_compact() -> str:
    """Tier-B system prompt: mid-budget variant ."""
    if _BRIEF_CLAUSE not in prompts.SYSTEM:
        raise SystemExit("prompts.SYSTEM drifted: brevity clause not found - update this file")
    return prompts.SYSTEM.replace(_BRIEF_CLAUSE, _COMPACT_CLAUSE)


def messages_for(problem_text: str, *, tier: str) -> list[dict[str, str]]:
    """The training prompt for one problem: exactly v2's solve_messages, with
    the system swapped per tier (A=unlimited, B=compact, C=deployment).
    context+query arrive concatenated in eval rows too, so the single
    problem_text block matches deployment shape."""
    msgs = prompts.solve_messages(problem_text, "", fewshot=True)
    # deployment joins context+query with "\n\n"; our query is empty, so strip the
    # dangling joiner or the trained user turn ends with whitespace deployment lacks
    msgs[-1]["content"] = msgs[-1]["content"].rstrip()
    if tier == "A":
        msgs[0] = {"role": "system", "content": system_long()}
    elif tier == "B":
        msgs[0] = {"role": "system", "content": system_compact()}
    return msgs


# ------------------------------------------------------------- trace cleaning

# meta-sections stripped from Tier-A completions (about the DOCUMENT, not the reasoning)
_DROP_SECTIONS = re.compile(
    r"^#{1,3}\s*\d*\.?\s*(problem info|problem restatement|overview|alternative approaches|"
    r"common pitfalls|extended knowledge|answers?|summary of answers?|final answers?)\b",
    re.I,
)
_META_LINE = re.compile(r"^_?(source|imported|tier \d)\b.*_?$", re.I)


def clean_problem_md(md: str) -> str:
    """Strip corpus artifacts (H1 title, _Source:..._ lines, leading rules) so the
    user message looks like an eval row, not a markdown document."""
    lines = md.splitlines()
    out: list[str] = []
    for i, ln in enumerate(ln.rstrip() for ln in lines):
        if i < 6 and (ln.startswith("# ") or _META_LINE.match(ln.strip()) or ln.strip() == "---"):
            continue
        out.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def clean_trace_for_tier_a(solution_md: str) -> str:
    """Keep the reasoning body of the trace; drop document-meta sections AND the
    trace's own answer sections (the canonical blocks are appended instead)."""
    parts = re.split(r"(?m)^(#{1,3} .*)$", solution_md)
    kept: list[str] = []
    keep = True
    for part in parts:
        if re.match(r"^#{1,3} ", part):
            keep = not _DROP_SECTIONS.match(part)
            if keep:
                kept.append(part)
        elif keep:
            kept.append(part)
    body = "\n".join(kept)
    # the H1 document title is never reasoning
    body = re.sub(r"(?m)^# .*$", "", body, count=1)
    return re.sub(r"\n{3,}", "\n\n", body).strip()


# ----------------------------------------------------------------- validation

MAX_WORDS = {"standard": 300, "short": 150}
ANSWER_LINE_BAD = re.compile(r"(->|→|\$\\rightarrow\$)")
LIST_MARKER = re.compile(r"^\s*(?:\d{1,2}[.)]\s*|[-*•]\s*)")


def _overrides() -> dict:
    p = WORK / "gold_overrides.json"
    if not p.exists():
        return {"accept": {}, "quarantine": []}
    return json.loads(p.read_text(encoding="utf-8"))


_SELECT_TAG = re.compile(r"<select(\d+),(\d+|inf),(\d+|inf)>")


def check_gold_content(pid: str, answers: list[str], lob: dict | None) -> list[str]:
    """Answers must match LOBSTER gold content (marker/NFC-aware).

    Select semantics: <select{n_options},{min},{max}>[...] — the answer must
    cover between min and max of the listed options (decoded 24 Jul from the
    corpus's own usage: min=1 cases are alternatives where any one option
    suffices; min=n cases require the full set on one line).

    Adjudicated overrides (work/gold_overrides.json) map index -> exact expected
    value (reviewed trace corrects LOBSTER; recorded in the supplied override file). An
    override only passes if the answer equals its recorded value — index-only
    skips would let later edits through unchecked."""
    if not lob:
        return []
    accepted = _overrides()["accept"].get(pid, {})
    if isinstance(accepted, list):  # legacy index-list form — refuse silently passing
        return [f"gold_overrides for {pid} uses the legacy index-list form; use index->value"]
    errs = []
    if len(answers) != len(lob):  # zip would silently truncate
        errs.append(f"answer count {len(answers)} != LOBSTER count {len(lob)}")
    for i, (gv, a) in enumerate(zip(lob.values(), answers)):
        na = unicodedata.normalize("NFC", str(a).strip())
        if str(i) in accepted:
            if na != unicodedata.normalize("NFC", str(accepted[str(i)]).strip()):
                errs.append(f"answer {i} != its adjudicated override value: {a!r}")
            continue
        opts, fuzzy = gold_options(gv)
        if fuzzy:
            continue
        nopts = [unicodedata.normalize("NFC", o.strip()) for o in opts]
        m = _SELECT_TAG.match(str(gv).strip())
        need_min = int(m.group(2)) if m and m.group(2) != "inf" else 1
        need_max = int(m.group(3)) if m and m.group(3) != "inf" else len(nopts)
        covered, leftover = _tile_options(na, nopts)
        # complete tiling: the answer must be fully consumed by
        # matched options + separators — substring coverage alone let '16' cover '6'
        ok = (na in nopts and need_min <= 1) or (
            not leftover and need_min <= covered <= max(need_max, need_min)
        )
        if not ok:
            errs.append(
                f"answer {i} tiles {covered}/{len(nopts)} options "
                f"(need {need_min}..{need_max}, leftover {leftover!r}): {a!r} vs {opts}"
            )
    return errs


_SEPARATORS = re.compile(r"(?:\s+|/|;|,|\||\bor\b|\band\b|[()\[\]{}.…])+", re.I)


def _tile_options(answer: str, options: list[str]) -> tuple[int, str]:
    """Greedy longest-first BOUNDARY-ANCHORED tiling: consume option occurrences
    (separator/edge on both sides — raw substring let single-letter options tile
    'cat' as a+c+t), then separators; return (#consumed,
    non-separator leftover)."""
    rest = answer
    covered = 0
    for o in sorted(set(options), key=len, reverse=True):
        if not o:
            continue
        pat = re.compile(
            r"(?:(?<=^)|(?<=[\s/;,|()\[\]{}.]))" + re.escape(o) + r"(?:(?=$)|(?=[\s/;,|()\[\]{}.]))"
        )
        rest, n = pat.subn(" ", rest, count=1)
        covered += n
    return covered, _SEPARATORS.sub("", rest).strip()


def _parser_postprocessed(answer: str) -> str:
    """What the deployment parser makes of a lone verbatim answer line.

    Gold answers legitimately carry forms the parser's noise-cleanup mangles
    (trailing '(sg)'/'(on the head)' disambiguators, leading '*' proto-forms,
    trailing glottal-stop apostrophes). A round-trip mismatch that equals this
    postprocessing is a PARSER defect (tracked as E25), not a data defect."""
    out = textproc.parse_answers("FINAL ANSWERS:\n" + answer, 1)
    return out[0] if out else ""


def _only_parser_mangling(answer: str, parsed: str) -> bool:
    return parsed == _parser_postprocessed(answer)


def canonical_blocks(rules: list[str], final_answers: list[str]) -> str:
    bullets = "\n".join(f"- {r}" for r in rules)
    answers = "\n".join(final_answers)
    return f"RULES:\n{bullets}\nFINAL ANSWERS:\n{answers}"


def check_compressed(obj: dict, job: dict) -> list[str]:
    """Return a list of human-readable failures (empty = pass)."""
    errs: list[str] = []
    for key in ("analysis_standard", "analysis_short", "rules", "final_answers"):
        if key not in obj:
            errs.append(f"missing key: {key}")
    if errs:
        return errs

    rules, answers = obj["rules"], obj["final_answers"]
    if not (2 <= len(rules) <= 4):
        errs.append(f"rules must be 2-4 bullets, got {len(rules)}")
    if any(len(r.split()) > 30 for r in rules):
        errs.append("a rules bullet exceeds 30 words")
    if not answers or any(not str(a).strip() for a in answers):
        errs.append("final_answers empty or contains blank line")

    lob = job.get("lobster_answers")
    if lob and len(answers) != len(lob):
        errs.append(f"final_answers count {len(answers)} != lobster count {len(lob)}")

    for i, a in enumerate(answers):
        a = str(a)
        if "\n" in a:
            errs.append(f"answer {i} contains newline")
        if ANSWER_LINE_BAD.search(a):
            errs.append(f"answer {i} contains an arrow (deployment parser splits on it): {a!r}")
        # trailing-parenthetical damage is caught authoritatively by the round-trip
        # below
        if LIST_MARKER.match(a):
            errs.append(f"answer {i} starts with a list marker the parser strips: {a!r}")
        if unicodedata.normalize("NFC", a) != a:
            errs.append(f"answer {i} is not NFC-normalized")

    errs += check_gold_content(
        job["problem_id"], [str(a) for a in answers], job.get("lobster_answers")
    )

    for tier in ("standard", "short"):
        text = obj[f"analysis_{tier}"]
        words = len(text.split())
        if words > MAX_WORDS[tier]:
            errs.append(f"analysis_{tier} too long: {words} > {MAX_WORDS[tier]} words")
        if textproc.FINAL_RE.search(text) or re.search(r"(?im)^rules\s*:", text):
            errs.append(f"analysis_{tier} must not contain its own RULES/FINAL ANSWERS block")
        if re.search(r"(?i)\bas an ai\b|^#{1,3} ", text, re.M):
            errs.append(f"analysis_{tier} contains meta-text or markdown headers")

    # the deployment parser must round-trip the completion exactly
    k = len(answers)
    for tier in ("standard", "short"):
        completion = obj[f"analysis_{tier}"].strip() + "\n" + canonical_blocks(rules, answers)
        parsed = textproc.parse_answers(completion, k)
        if parsed != [str(a).strip() for a in answers]:
            bad = [i for i, (p, a) in enumerate(zip(parsed, answers)) if p != str(a).strip()]
            errs.append(f"parse round-trip failed ({tier}) at items {bad}: {parsed} != {answers}")
    return errs


# -------------------------------------------------------------------- extract


def linguini_problem_keys() -> set[tuple[str, int]]:
    import csv

    with open(FIXTURE, encoding="utf-8") as f:
        return {(r["id"][2:6], int(r["id"][6:8])) for r in csv.DictReader(f)}


def cmd_extract() -> None:
    ling = linguini_problem_keys()
    (WORK / "jobs").mkdir(parents=True, exist_ok=True)
    DATASET.mkdir(parents=True, exist_ok=True)

    corpus = [json.loads(line) for line in open(CORPUS, encoding="utf-8")]
    cmap: dict[str, dict] = {}
    for c in corpus:
        pid = c["problem_id"]
        key = (str(c["year"]), int(c["problem_no"])) if c["contest"] == "IOL" else None
        in_ling = key in ling if key else False
        lob = (c.get("lobster_reference") or {}).get("answer") or None
        job = {
            "problem_id": pid,
            "contest": c["contest"],
            "year": c["year"],
            "problem_no": c["problem_no"],
            "topic": c["topic"],
            "in_linguini": in_ling,
            "problem_text": clean_problem_md(c["problem_md"]),
            "trace_md": c["solution_en_md"],
            "official_solution_md": c.get("official_solution_md"),
            "lobster_answers": lob,
        }
        (WORK / "jobs" / f"{pid}.json").write_text(
            json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        cmap[pid] = {"in_linguini": in_ling, "contest": c["contest"], "year": c["year"]}

    trained_keys = {(str(c["year"]), int(c["problem_no"])) for c in corpus if c["contest"] == "IOL"}
    clean_probe = sorted(f"{y}-{p}" for (y, p) in ling - trained_keys)
    (DATASET / "contamination_map.json").write_text(
        json.dumps(
            {
                "policy": "train on all supplied reviewed traces; "
                "measure on clean_probe_linguini_problems + leaderboard A/B only",
                "n_traces": len(corpus),
                "n_linguini_source_problems": len(ling),
                "n_linguini_problems_trained_on": len(ling & trained_keys),
                "clean_probe_linguini_problems": clean_probe,
                "per_problem": cmap,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    n_ling = sum(1 for v in cmap.values() if v["in_linguini"])
    print(f"jobs: {len(cmap)}  (in-linguini: {n_ling}; clean probe: {len(clean_probe)} problems)")
    print(f"wrote {WORK / 'jobs'} and {DATASET / 'contamination_map.json'}")


# ---------------------------------------------------------------------- check


def cmd_check(paths: list[str]) -> None:
    n_bad = 0
    for p in paths:
        path = Path(p)
        pid = path.stem
        job = json.loads((WORK / "jobs" / f"{pid}.json").read_text(encoding="utf-8"))
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 — report, don't crash the batch
            print(f"FAIL {pid}: unreadable JSON ({e})")
            n_bad += 1
            continue
        errs = check_compressed(obj, job)
        if errs:
            n_bad += 1
            print(f"FAIL {pid}:")
            for e in errs:
                print(f"  - {e}")
        else:
            print(f"OK   {pid}")
    if n_bad:
        raise SystemExit(1)


# ----------------------------------------------------- programmatic answers

_SELECT_RE = re.compile(r"<select[\d,a-z]*>\s*(\[.*\])\s*$", re.S)


def gold_options(value: str) -> tuple[list[str], bool]:
    """Decode one LOBSTER gold value -> (accepted options, is_fuzzy).

    LOBSTER wraps grading specs around answers: <fuzzy> marks free-form prose,
    <selectN,..>[...] lists interchangeable accepted strings. Plain values are
    literal. (See work/audit_report.json for the classes found in practice.)
    """
    v = str(value).strip()
    fuzzy = v.startswith("<fuzzy>")
    if fuzzy:
        v = v[len("<fuzzy>") :].strip()
    m = _SELECT_RE.match(v)
    if m:
        try:
            opts = [str(x) for x in ast.literal_eval(m.group(1))]
        except (ValueError, SyntaxError):
            opts = [v]  # unparseable select-list: keep raw (content check will flag)
        return opts, fuzzy
    return [v], fuzzy


def cmd_extract_answers() -> None:
    """Write work/answers/{pid}.json for problems whose FINAL ANSWERS can be
    built without judgment: every LOBSTER value is a literal or a select-list
    (first option taken, NFC'd). Fuzzy values need a human/LLM phrasing pass —
    those problems are skipped and reported."""
    (WORK / "answers").mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in (WORK / "compressed").glob("*.json")}
    n_new, skipped = 0, []
    for jp in sorted((WORK / "jobs").glob("*.json")):
        pid = jp.stem
        if pid in done or (WORK / "answers" / f"{pid}.json").exists():
            continue
        job = json.loads(jp.read_text(encoding="utf-8"))
        lob = job.get("lobster_answers")
        if not lob:
            skipped.append((pid, "no-lobster"))
            continue
        answers, ok = [], True
        for v in lob.values():
            opts, fuzzy = gold_options(v)
            # NFC PIN (documented decision): all trained answer text is NFC; LOBSTER
            # mixes NFC/NFD. NFC-once-at-source is the single canonicalization point.
            cand = unicodedata.normalize("NFC", opts[0].strip())
            if (
                fuzzy
                or "\n" in cand
                or cand.startswith("<")
                or ANSWER_LINE_BAD.search(cand)
                or LIST_MARKER.match(cand)
            ):
                ok = False
                break
            answers.append(cand)
        if not ok:
            skipped.append((pid, "needs-judgment"))
            continue
        (WORK / "answers" / f"{pid}.json").write_text(
            json.dumps(
                {"final_answers": answers, "source": "lobster-programmatic"},
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        n_new += 1
    print(f"programmatic answers written: {n_new}")
    from collections import Counter

    print("skipped:", Counter(r for _, r in skipped))
    print("  pids:", [p for p, _ in skipped])


# ---------------------------------------------------------------------------
# Delete-only compression validation

_EDGE_PUNCT = '.,;:!?"\u201c\u201d()[]{}|*_`\u2014\u2013-'


def _cave_words(text: str) -> list[str]:
    """Word tokens for the subsequence check: NFC, edge-punctuation-stripped,
    punctuation-only tokens dropped (deleting a comma is a deletion)."""
    out = []
    for w in unicodedata.normalize("NFC", text).split():
        w = w.strip(_EDGE_PUNCT)
        if w:
            out.append(w)
    return out


def _tok_eq(a: str, b: str) -> bool:
    # only sentence-start recasing is tolerated (first char case), ASCII-alpha only —
    # blanket lower() let ALLCAPS glosses (SG, PL) be freely recased
    if a == b:
        return True
    return (
        len(a) == len(b) > 0 and a[1:] == b[1:] and a[0].isascii() and a[0].lower() == b[0].lower()
    )


def is_subsequence(small: list[str], big: list[str]) -> tuple[bool, str]:
    """Ordered containment; on failure, return the first offending token."""
    it = iter(big)
    for x in small:
        if not any(_tok_eq(x, y) for y in it):
            return False, x
    return True, ""


CAVE_RATIO = {"tier_b": 0.55, "tier_c": 0.20}
CAVE_ABS_MAX = {"tier_b": None, "tier_c": 900}


def check_caveman(obj: dict, job: dict) -> list[str]:
    """Validate one work/caveman/{pid}.json: delete-only tiers B/C.

    tier_b/tier_c must be word-level SUBSEQUENCES of the trace (zero invented
    content by construction), within their compression ratios; rules
    bullets (only needed when work/compressed lacks them) are delete-only too."""
    errs: list[str] = []
    src = _cave_words(job["trace_md"])
    n_src = len(src)
    for tier in ("tier_b", "tier_c"):
        text = obj.get(tier)
        if not isinstance(text, str) or not text.strip():
            errs.append(f"{tier} missing/empty")
            continue
        if textproc.FINAL_RE.search(text) or re.search(r"(?im)^rules\s*:", text):
            errs.append(f"{tier} must not contain RULES:/FINAL ANSWERS: blocks")
        words = _cave_words(text)
        ok, bad = is_subsequence(words, src)
        if not ok:
            errs.append(f"{tier} is NOT a delete-only subsequence of the trace (token {bad!r})")
        ratio = len(words) / max(n_src, 1)
        if ratio > CAVE_RATIO[tier]:
            cap_pct = f"{CAVE_RATIO[tier]:.0%}"
            errs.append(f"{tier} too long: {len(words)} words = {ratio:.0%} (cap {cap_pct})")
        cap = CAVE_ABS_MAX[tier]
        if cap and len(words) > cap:
            errs.append(f"{tier} exceeds absolute cap {cap} words ({len(words)})")
    b, c = obj.get("tier_b", ""), obj.get("tier_c", "")
    if isinstance(b, str) and isinstance(c, str) and len(_cave_words(c)) > len(_cave_words(b)):
        errs.append("tier_c longer than tier_b")

    has_rules_already = (WORK / "compressed" / f"{job['problem_id']}.json").exists()
    rules = obj.get("rules")
    if not has_rules_already:
        if not (isinstance(rules, list) and 2 <= len(rules) <= 4):
            errs.append("rules: 2-4 bullets required (no compressed/ file for this problem)")
        else:
            for i, r in enumerate(rules):
                if len(str(r).split()) > 30:
                    errs.append(f"rules[{i}] exceeds 30 words")
                ok, bad = is_subsequence(_cave_words(str(r)), src)
                if not ok:
                    errs.append(f"rules[{i}] not delete-only from the trace (token {bad!r})")
    return errs


def cmd_check_caveman(paths: list[str]) -> None:
    n_bad = 0
    for p in paths:
        path = Path(p)
        pid = path.stem
        job = json.loads((WORK / "jobs" / f"{pid}.json").read_text(encoding="utf-8"))
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {pid}: unreadable ({e})")
            n_bad += 1
            continue
        errs = check_caveman(obj, job)
        if errs:
            n_bad += 1
            print(f"FAIL {pid}:")
            for e in errs:
                print(f"  - {e}")
        else:
            wb, wc = len(_cave_words(obj["tier_b"])), len(_cave_words(obj["tier_c"]))
            print(f"OK   {pid}  (B {wb}w, C {wc}w)")
    if n_bad:
        raise SystemExit(1)


def cmd_check_neacsu(paths: list[str]) -> None:
    """Validate Neacșu extraction records (E23c).

    Shape + answer-line hygiene + deployment-parser round-trip. (No LOBSTER gold
    exists for these; provenance rests on the extraction spec + spot audits.)
    """
    n_bad = 0
    for p in paths:
        path = Path(p)
        errs: list[str] = []
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {path.stem}: unreadable ({e})")
            n_bad += 1
            continue
        for k in ("source", "problem_text", "solution_text", "rules", "final_answers"):
            if not obj.get(k):
                errs.append(f"missing/empty {k}")
        if not errs:
            if len(obj["problem_text"]) < 200:
                errs.append("problem_text suspiciously short")
            if not (2 <= len(obj["rules"]) <= 4):
                errs.append("rules must be 2-4 bullets")
            answers = [str(a) for a in obj["final_answers"]]
            for i, a in enumerate(answers):
                na = unicodedata.normalize("NFC", a.strip())
                if "\n" in a or ANSWER_LINE_BAD.search(a) or LIST_MARKER.match(a):
                    errs.append(f"answer {i} newline/arrow/list-marker: {a!r}")
                if na != a:
                    errs.append(f"answer {i} not NFC/stripped")
                if "\\" in a:
                    errs.append(f"answer {i} contains raw LaTeX: {a!r}")
            comp = "FINAL ANSWERS:\n" + "\n".join(a.strip() for a in answers)
            parsed = textproc.parse_answers(comp, len(answers))
            hard = [
                i
                for i, (q, a) in enumerate(zip(parsed, [a.strip() for a in answers]))
                if q != a and not _only_parser_mangling(a, q)
            ]
            if hard:
                errs.append(f"parser round-trip failed at {hard}")
        if errs:
            n_bad += 1
            print(f"FAIL {path.stem}:")
            for e in errs:
                print(f"  - {e}")
        else:
            print(f"OK   {path.stem}")
    if n_bad:
        raise SystemExit(1)


def cmd_check_answers(paths: list[str]) -> None:
    """Validate work/answers/{pid}.json files provided for answer extraction.

    Guarantee: extraction cannot INVENT text — every answer must occur verbatim
    (NFC, whitespace-collapsed) inside the trace or the official solution, plus
    the usual line-shape checks and the parser round-trip."""
    n_bad = 0
    for p in paths:
        path = Path(p)
        pid = path.stem
        job = json.loads((WORK / "jobs" / f"{pid}.json").read_text(encoding="utf-8"))
        sources_ws = [
            re.sub(r"\s+", " ", unicodedata.normalize("NFC", t))
            for t in (job["trace_md"], job.get("official_solution_md") or "")
            if t
        ]
        errs: list[str] = []
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            answers = [str(a) for a in obj["final_answers"]]
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {pid}: unreadable ({e})")
            n_bad += 1
            continue
        if not answers:
            errs.append("final_answers empty")
        for i, a in enumerate(answers):
            na = unicodedata.normalize("NFC", a.strip())
            if na != a:
                errs.append(f"answer {i} not NFC/stripped")
            if "\n" in a:
                errs.append(f"answer {i} contains newline")
            if ANSWER_LINE_BAD.search(a) or LIST_MARKER.match(a):
                errs.append(f"answer {i} has arrow/list-marker the parser mangles: {a!r}")
            if not any(re.sub(r"\s+", " ", na) in src for src in sources_ws):
                errs.append(f"answer {i} is NOT a verbatim substring of trace/official: {a!r}")
        comp = "FINAL ANSWERS:\n" + "\n".join(a.strip() for a in answers)
        parsed = textproc.parse_answers(comp, len(answers))
        hard = [
            i
            for i, (q, a) in enumerate(zip(parsed, [a.strip() for a in answers]))
            if q != a and not _only_parser_mangling(a, q)
        ]
        if hard:
            errs.append(f"parser round-trip failed at items {hard}")
        errs += check_gold_content(pid, [a.strip() for a in answers], job.get("lobster_answers"))
        if errs:
            n_bad += 1
            print(f"FAIL {pid}:")
            for e in errs:
                print(f"  - {e}")
        else:
            print(f"OK   {pid}")
    if n_bad:
        raise SystemExit(1)


def problem_text_for(pid: str, job: dict) -> str:
    """Problem text, preferring the booklet-sliced scope when one exists
    (TOL 2025/2026 jobs share one concatenated exam booklet — corpus artifact;
    work/scoped/{pid}.json holds the verbatim contiguous per-problem slice)."""
    sp = WORK / "scoped" / f"{pid}.json"
    if sp.exists():
        scoped = json.loads(sp.read_text(encoding="utf-8"))["problem_text"]
        if len(scoped) < 400 or scoped not in job["problem_text"]:
            raise SystemExit(f"{pid}: scoped problem_text empty/short or not a verbatim slice")
        return scoped
    return job["problem_text"]


def answers_for(pid: str) -> list[str] | None:
    """Canonical FINAL ANSWERS lines for a problem, from whichever source exists."""
    cp = WORK / "compressed" / f"{pid}.json"
    if cp.exists():
        return [str(a) for a in json.loads(cp.read_text(encoding="utf-8"))["final_answers"]]
    ap = WORK / "answers" / f"{pid}.json"
    if ap.exists():
        return [str(a) for a in json.loads(ap.read_text(encoding="utf-8"))["final_answers"]]
    return None


# --------------------------------------------------- assemble rung A (E23a)


def cmd_assemble_verbatim() -> None:
    """One example per problem with a verbatim trace (no
    section cleanup; deliberately NO RULES block — capability rung, documented
    in the trace data) + an appended FINAL ANSWERS block. Unlimited-reasoning system
    prompt (verbatim traces are ~6k tok; the deployment brevity clause would
    be a lie against these targets).

    Hard gates per row: quarantine list, LOBSTER gold-content
    check (with adjudicated overrides), and the deployment-parser round-trip —
    a row that fails any gate is EXCLUDED and reported, never silently trained."""
    rows, missing, failed, parser_conflicts = [], [], [], []
    quarantine = set(_overrides()["quarantine"])
    for jp in sorted((WORK / "jobs").glob("*.json")):
        pid = jp.stem
        job = json.loads(jp.read_text(encoding="utf-8"))
        if pid in quarantine:
            failed.append((pid, "quarantined (gold_overrides.json)"))
            continue
        answers = answers_for(pid)
        if answers is None:
            missing.append(pid)
            continue
        answers = [str(a).strip() for a in answers]
        gold_errs = check_gold_content(pid, answers, job.get("lobster_answers"))
        if gold_errs:
            failed.append((pid, gold_errs[0]))
            continue
        completion = job["trace_md"].strip() + "\n\nFINAL ANSWERS:\n" + "\n".join(answers)
        parsed = textproc.parse_answers(completion, len(answers))
        if parsed != answers:
            hard = [
                i
                for i, (p, a) in enumerate(zip(parsed, answers))
                if p != a and not _only_parser_mangling(a, p)
            ]
            if hard:
                failed.append((pid, f"parser round-trip failed at items {hard}"))
                continue
            parser_conflicts.append(pid)
        rows.append(
            {
                "problem_id": pid,
                "rung": "a-verbatim",
                "in_linguini": job["in_linguini"],
                "messages": messages_for(problem_text_for(pid, job), tier="A"),
                "completion": completion,
            }
        )
    DATASET.mkdir(parents=True, exist_ok=True)
    out = DATASET / "train_a_verbatim.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_clean11_ids()
    toks = sorted(len(r["completion"]) // 4 for r in rows)
    print(
        f"wrote {out}: {len(rows)} examples; completion ~tok p50={toks[len(toks) // 2]} "
        f"p99={toks[int(len(toks) * 0.99)]} max={toks[-1]}"
    )
    for pid, why in failed:
        print(f"  EXCLUDED {pid}: {why}")
    if parser_conflicts:
        print(f"kept with parser-conflict (trailing marker paren, see E25): {parser_conflicts}")
    if missing:
        print(f"missing answers ({len(missing)}): {missing}")


def write_clean11_ids() -> None:
    """Emit the CLEAN-11 fixture row ids (checkpoint selection may ONLY use these;
    Search/Holdout are contaminated once SFT training starts)."""
    import csv

    corpus_keys = set()
    for jp in (WORK / "jobs").glob("*.json"):
        job = json.loads(jp.read_text(encoding="utf-8"))
        if job["contest"] == "IOL":
            pno = job.get("problem_no") or job["problem_id"].rsplit("P", 1)[1]
            corpus_keys.add((str(job["year"]), int(pno)))
    with open(FIXTURE, encoding="utf-8") as f:
        rows = [
            r["id"]
            for r in csv.DictReader(f)
            if (r["id"][2:6], int(r["id"][6:8])) not in corpus_keys
        ]
    out = DATASET / "clean11_ids.txt"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(rows)} fixture rows (selection-only fold)")


# ------------------------------------------------------------------- assemble


def rules_for(pid: str) -> list[str] | None:
    """Canonical RULES bullets: legacy compressed file first, else the caveman file."""
    cp = WORK / "compressed" / f"{pid}.json"
    if cp.exists():
        return [str(r) for r in json.loads(cp.read_text(encoding="utf-8"))["rules"]]
    cv = WORK / "caveman" / f"{pid}.json"
    if cv.exists():
        rules = json.loads(cv.read_text(encoding="utf-8")).get("rules")
        if isinstance(rules, list) and rules:
            return [str(r) for r in rules]
    return None


_TYPE_POOL_MAP = {  # heuristic sub-labels pool with the fixture's plain label
    "translation_to_english": "translation",
    "translation_from_english": "translation",
}


def _exemplar_messages(pid: str, ttypes: dict) -> list[dict[str, str]] | None:
    """Same-task-type cross-problem exemplar :
    another problem's tier-C caveman text + canonical blocks as the assistant turn."""
    my_type = ttypes.get(pid, {}).get("task_type", "")
    pool_of = lambda t: _TYPE_POOL_MAP.get(t, t)  # noqa: E731
    pool = sorted(
        p
        for p, v in ttypes.items()
        if p != pid
        and pool_of(v["task_type"]) == pool_of(my_type)
        and (WORK / "caveman" / f"{p}.json").exists()
    )
    if not pool:
        return None
    ex_pid = pool[sum(ord(c) for c in pid) % len(pool)]  # deterministic, seedless
    ex_job = json.loads((WORK / "jobs" / f"{ex_pid}.json").read_text(encoding="utf-8"))
    ex_cave = json.loads((WORK / "caveman" / f"{ex_pid}.json").read_text(encoding="utf-8"))
    ex_rules, ex_answers = rules_for(ex_pid), answers_for(ex_pid)
    if not ex_rules or not ex_answers:
        return None
    ex_completion = ex_cave["tier_c"].strip() + "\n" + canonical_blocks(ex_rules, ex_answers)
    return [
        {"role": "user", "content": problem_text_for(ex_pid, ex_job)},
        {"role": "assistant", "content": ex_completion},
    ]


def cmd_assemble_caveman() -> None:
    """E23b: 3 tiers x 148 problems, delete-only completions + canonical blocks,
    with exemplar ROTATION (fixed / none / same-type cross-problem).
    Same hard gates as assemble-verbatim per row."""
    ttypes_p = WORK / "task_types.json"
    ttypes = json.loads(ttypes_p.read_text(encoding="utf-8")) if ttypes_p.exists() else {}
    quarantine = set(_overrides()["quarantine"])
    rows, failed, ex_counts = [], [], Counter()

    for jp in sorted((WORK / "jobs").glob("*.json")):
        pid = jp.stem
        if pid in quarantine:
            continue
        job = json.loads(jp.read_text(encoding="utf-8"))
        cvp = WORK / "caveman" / f"{pid}.json"
        answers, rules = answers_for(pid), rules_for(pid)
        if not cvp.exists() or answers is None or rules is None:
            failed.append((pid, "missing caveman/answers/rules"))
            continue
        cave = json.loads(cvp.read_text(encoding="utf-8"))
        errs = check_caveman(cave, job) + check_gold_content(
            pid, [a.strip() for a in answers], job.get("lobster_answers")
        )
        if errs:
            failed.append((pid, errs[0]))
            continue
        answers = [str(a).strip() for a in answers]
        blocks = canonical_blocks(rules, answers)
        tier_texts = {
            "A": clean_trace_for_tier_a(job["trace_md"]) + "\n\n" + blocks,
            "B": cave["tier_b"].strip() + "\n" + blocks,
            "C": cave["tier_c"].strip() + "\n" + blocks,
        }
        ptext = problem_text_for(pid, job)
        for tier, completion in tier_texts.items():
            parsed = textproc.parse_answers(completion, len(answers))
            if parsed != answers:
                hard = [
                    i
                    for i, (p, a) in enumerate(zip(parsed, answers))
                    if p != a and not _only_parser_mangling(a, p)
                ]
                if hard:
                    failed.append((pid, f"tier {tier} round-trip failed at {hard}"))
                    continue
            variant = ["fixed", "none", "typed"][(sum(ord(c) for c in pid) + ord(tier)) % 3]
            msgs = messages_for(ptext, tier=tier)
            if variant == "none":
                msgs = [msgs[0], msgs[-1]]  # drop the fewshot pair
            elif variant == "typed":
                ex = _exemplar_messages(pid, ttypes)
                if ex is None:
                    variant = "fixed"  # thin type pool — keep deployment default
                else:
                    msgs = [msgs[0], *ex, msgs[-1]]
            ex_counts[variant] += 1
            rows.append(
                {
                    "problem_id": pid,
                    "tier": tier,
                    "exemplar": variant,
                    "in_linguini": job["in_linguini"],
                    "messages": msgs,
                    "completion": completion,
                }
            )

    DATASET.mkdir(parents=True, exist_ok=True)
    out = DATASET / "train_e23b.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    toks = sorted(len(r["completion"]) // 4 for r in rows)
    print(
        f"wrote {out}: {len(rows)} examples "
        f"({len({r['problem_id'] for r in rows})} problems); exemplar mix {dict(ex_counts)}"
    )
    print(f"completion ~tok p50={toks[len(toks) // 2]} p99={toks[int(len(toks) * 0.99)]}")
    for pid, why in failed:
        print(f"  EXCLUDED {pid}: {why}")


def cmd_assemble() -> None:
    jobs = {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((WORK / "jobs").glob("*.json"))
    }
    comp_dir = WORK / "compressed"
    rows: list[dict] = []
    missing, failed = [], []
    lens: Counter[str] = Counter()

    quarantine = set(_overrides()["quarantine"])
    quarantined = []
    for pid, job in jobs.items():
        if pid in quarantine:  # excluded, NOT a validation failure
            quarantined.append(pid)
            continue
        cpath = comp_dir / f"{pid}.json"
        if not cpath.exists():
            missing.append(pid)
            continue
        obj = json.loads(cpath.read_text(encoding="utf-8"))
        errs = check_compressed(obj, job)
        if errs:
            failed.append((pid, errs))
            continue

        blocks = canonical_blocks(obj["rules"], obj["final_answers"])
        tier_texts = {
            "A": clean_trace_for_tier_a(job["trace_md"]) + "\n\n" + blocks,
            "B": obj["analysis_standard"].strip() + "\n" + blocks,
            "C": obj["analysis_short"].strip() + "\n" + blocks,
        }
        for tier, completion in tier_texts.items():
            rows.append(
                {
                    "problem_id": pid,
                    "tier": tier,
                    "in_linguini": job["in_linguini"],
                    "messages": messages_for(problem_text_for(pid, job), tier=tier),
                    "completion": completion,
                }
            )
            lens[tier] += len(completion) // 4  # rough tokens, for the stats line

    if quarantined:
        print(f"quarantined (excluded): {quarantined}")
    if missing or failed:
        print(f"NOT ASSEMBLED - missing: {len(missing)} {missing[:8]}")
        for pid, errs in failed[:10]:
            print(f"  FAIL {pid}: {errs[0]}")
        raise SystemExit(f"{len(failed)} failed validation, {len(missing)} missing")

    DATASET.mkdir(parents=True, exist_ok=True)
    out = DATASET / "train.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_prob = len({r["problem_id"] for r in rows})
    print(f"wrote {out}: {len(rows)} examples / {n_prob} problems")
    for tier in "ABC":
        n = sum(1 for r in rows if r["tier"] == tier)
        print(f"  tier {tier}: {n} examples, ~{lens[tier] // max(n, 1)} tok/completion avg")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract")
    pc = sub.add_parser("check")
    pc.add_argument("paths", nargs="+")
    sub.add_parser("assemble")
    sub.add_parser("extract-answers")
    sub.add_parser("assemble-verbatim")
    sub.add_parser("assemble-caveman")
    pca = sub.add_parser("check-answers")
    pca.add_argument("paths", nargs="+")
    pcv = sub.add_parser("check-caveman")
    pcv.add_argument("paths", nargs="+")
    pcn = sub.add_parser("check-neacsu")
    pcn.add_argument("paths", nargs="+")
    args = ap.parse_args()
    if args.cmd == "extract":
        cmd_extract()
    elif args.cmd == "check":
        cmd_check(args.paths)
    elif args.cmd == "extract-answers":
        cmd_extract_answers()
    elif args.cmd == "assemble-verbatim":
        cmd_assemble_verbatim()
    elif args.cmd == "assemble-caveman":
        cmd_assemble_caveman()
    elif args.cmd == "check-answers":
        cmd_check_answers(args.paths)
    elif args.cmd == "check-caveman":
        cmd_check_caveman(args.paths)
    elif args.cmd == "check-neacsu":
        cmd_check_neacsu(args.paths)
    else:
        cmd_assemble()


if __name__ == "__main__":
    main()
