"""Prompt text and message builders for solving, finalizing, and refining answers."""

from __future__ import annotations

import os

from .textproc import is_phonetic_task

Messages = list[dict[str, str]]

SYSTEM = (
    "You solve International Linguistics Olympiad puzzles by pure deduction from "
    "the data given. You may meet a task type you have never seen: read the "
    "instruction and the examples, and answer in the same form they use.\n"
    "Work in two stages.\n"
    "STAGE 1 — INDUCTION: systematically determine as much as possible about the "
    "language: the meaning of each word or morpheme, the word order, the "
    "morphology (affixes, conjugations, sound changes), or the number system. "
    "Test every rule you propose against EVERY example given — a correct rule "
    "set explains all of the data, not most of it.\n"
    "STAGE 2 — ANSWER: apply your rules to the asked items.\n"
    "Be brief: keep all reasoning under 300 words, then IMMEDIATELY output, in "
    "this exact order:\n"
    "RULES:\n"
    "- 2 to 4 short bullet points stating the key rules you deduced and the "
    "evidence for them.\n"
    "FINAL ANSWERS:\n"
    "one answer per line, in the order the items are asked — the bare answer "
    "only: no numbering, no quotes, no commentary. For matching tasks give only "
    "the option letter; for number tasks give exactly what is asked (digits or "
    "words). Never leave a line empty — if unsure, give your best guess."
)

# worked example (known language) demonstrating the induction method AND the exact
# output format — per Ramji & Ramji / UNVEILING, the single best prompt addition
# for small models. No pipe/table notation (E5 iter 5: models echoed data tables).
FEWSHOT_USER = (
    "Here are phrases in Spanish with their English translations:\n"
    "'la manzana roja' means 'the red apple'\n"
    "'el libro rojo' means 'the red book'\n"
    "'la manzana verde' means 'the green apple'\n\n"
    "Translate into Spanish:\n(1) the green book\n(2) the red apple"
)
FEWSHOT_ASSISTANT = (
    "Comparing pairs: 'manzana' = apple (appears with both colors), 'libro' = book, "
    "'roja/rojo' = red (agrees with noun gender), 'verde' = green. Adjectives follow "
    "the noun. 'la' goes with manzana, 'el' with libro.\n"
    "Check against all three examples: consistent.\n"
    "RULES:\n"
    "- noun precedes adjective; article agrees with noun (la manzana / el libro)\n"
    "- red = rojo/roja by gender; green = verde (invariant)\n"
    "FINAL ANSWERS:\n"
    "el libro verde\n"
    "la manzana roja"
)

FEWSHOT_MATCH_USER = (
    "In Miraki, each bird name matches exactly one call type. Known pairs: kiru→(1) whistle, "
    "felo→(3) trill. Remaining calls: (2) chirp, (4) hoot.\n\n"
    "Match each name to a call type:\n(a) banu\n(b) tiwa"
)
FEWSHOT_MATCH_ASSISTANT = (
    "kiru→(1) and felo→(3) are given. Remaining: banu and tiwa must map to (2) chirp and (4) hoot.\n"
    "tiwa ends in -wa (resonant vowel) — consistent with sustained hoot. banu → chirp by elimination.\n"
    "RULES:\n"
    "- bijection: each name maps to exactly one call; given pairs eliminate options\n"
    "- remaining pairs resolved by elimination and phonological cues\n"
    "FINAL ANSWERS:\n"
    "2\n"
    "4"
)

FEWSHOT_NUM_USER = (
    "In Velani, numbers are expressed as follows:\n"
    "'oma'=1, 'dua'=2, 'tara'=3, 'kela'=5, 'kela-oma'=6, 'kela-dua'=7, 'kela-tara'=8\n"
    "'dua-kela'=10, 'dua-kela-oma'=11, 'dua-kela-tara'=13\n\n"
    "Write in Velani:\n(1) 8\n(2) 15"
)
FEWSHOT_NUM_ASSISTANT = (
    "Pattern: kela=5; X-kela = X×5; kela-Y = 5+Y; X-kela-Y = X×5+Y.\n"
    "Verified: dua-kela=2×5=10 ✓; dua-kela-tara=2×5+3=13 ✓.\n"
    "8 = 5+3 = kela-tara. 15 = 3×5 = tara-kela.\n"
    "RULES:\n"
    "- base 5: kela=5; prefix multiplies, suffix adds\n"
    "- X-kela-Y = X×5+Y; kela-Y = 5+Y; X-kela = X×5\n"
    "FINAL ANSWERS:\n"
    "kela-tara\n"
    "tara-kela"
)

FEWSHOT_PHON_USER = (
    "In Moran, stops and fricatives voice before voiced-initial words:\n"
    "[kata] alone but [gada-ba]; [pitu] alone but [bidu-ga]; [seku] alone but [zegu-da]\n\n"
    "Apply the rule:\n(1) [tapi] before [go]\n(2) [seku] alone"
)
FEWSHOT_PHON_ASSISTANT = (
    "Rule: before a voiced-initial word, all stops and fricatives voice (k→g, t→d, p→b, s→z).\n"
    "Check: kata→gada (k→g,t→d) ✓; pitu→bidu (p→b,t→d) ✓; seku→zegu (s→z,k→g) ✓.\n"
    "1. [tapi]+[go]: t→d, p→b → [dabi-go].\n"
    "2. [seku] alone: no voicing context → [seku].\n"
    "RULES:\n"
    "- all obstruents voice before voiced-initial words; unchanged in isolation\n"
    "FINAL ANSWERS:\n"
    "dabi-go\n"
    "seku"
)

_BUILTIN_FEWSHOTS: dict[str, tuple[str, str]] = {
    "match_letters": (FEWSHOT_MATCH_USER, FEWSHOT_MATCH_ASSISTANT),
    "num_to_text": (FEWSHOT_NUM_USER, FEWSHOT_NUM_ASSISTANT),
    "text_to_num": (FEWSHOT_NUM_USER, FEWSHOT_NUM_ASSISTANT),
}


# FINALIZE pass: truncated analysis never reaches FINAL ANSWERS — the derivations
# are already committed in the text, so one short follow-up extracts them
# Delivery-FIRST: demand the answer block ONLY (no rules, no more analysis). Verbose models
# (Qwen3.6-35B) burned the whole finalize budget re-deriving RULES and never reached the
# answers → 88% no-marker. Rules/explanation still come from the solve pass; finalize's one
# job is to get the answers out. (v1 solver/script.py keeps an identical copy.)
def finalize_msg(k: int) -> str:
    """Finalize-pass instruction: delivery-first + count enforcement + distinctness check.

    k=0 means n_items failed; omit the hard count in that case.
    """
    count = f"exactly {k} bare answers" if k > 0 else "one bare answer per asked item"
    return (
        "Stop. Do NOT analyze or write rules. Output ONLY this, nothing before it: a line reading "
        f"exactly\nFINAL ANSWERS:\nthen {count}, one per line in the order asked — "
        "best guess if unsure. "
        "If two answers are identical, double-check you haven't missed a distinguishing feature. "
        "Nothing after the answers."
    )


REFINE_MSG = (
    "Review your previous solution above. Automated checks flagged these issues:\n"
    "{verdicts}\n"
    "Re-derive the rules, test them against EVERY example given, and output the "
    "two blocks again — RULES: with 2-4 bullets, then FINAL ANSWERS: with exactly "
    "{k} lines, one bare answer per line, best guess if unsure. Write only the "
    "unknown part without copying the question; for matching tasks, labels only."
)


_EXEMPLAR_BANK: dict | None = None


def _typed_exemplar(task_type: str) -> tuple[str, str] | None:
    """Same-task-type exemplar from the verified-trace bank, or None.

    E26: EXEMPLAR_BANK names a JSON {task_type: {user, assistant}} built from the
    148 expert-verified traces (dev/sft exemplar machinery). Fail-closed: any
    problem loading the bank leaves the fixed exemplar in place.
    """
    global _EXEMPLAR_BANK  # noqa: PLW0603 — one-shot module cache
    if _EXEMPLAR_BANK is None:
        _EXEMPLAR_BANK = {}
        path = os.environ.get("EXEMPLAR_BANK", "")
        if path:
            try:
                import json  # noqa: PLC0415
                from pathlib import Path  # noqa: PLC0415

                _EXEMPLAR_BANK = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 — fail-closed to the fixed exemplar
                _EXEMPLAR_BANK = {}
    ex = _EXEMPLAR_BANK.get(task_type)
    if isinstance(ex, dict) and ex.get("user") and ex.get("assistant"):
        return ex["user"], ex["assistant"]
    return None


def solve_messages(
    context: str, query: str, *, fewshot: bool, task_type: str = "", extra: str = ""
) -> Messages:
    """Build the initial solve conversation for one problem.

    `extra` is an optional grounding preamble (E28 gramquery grammar examples);
    it is prepended to the user turn so the puzzle data stays adjacent to the
    query. Empty string = no-op (the retrieval pass is fully fail-closed).

    No `k` here on purpose: item-count enforcement moved to the finalize follow-up
    (`finalize_msg(k)`), which uses a separate deployment prompt.
    Stating the count twice made the solve prompt longer without making the solve
    output more compliant. Keeping a `k` parameter that nothing reads would be worse
    than not having one — the caller would believe it was doing something."""
    msgs: Messages = [{"role": "system", "content": SYSTEM}]
    if fewshot:
        typed = _typed_exemplar(task_type) if task_type else None
        builtin = _BUILTIN_FEWSHOTS.get(task_type) if task_type else None
        if not typed and not builtin and is_phonetic_task(context):
            builtin = (FEWSHOT_PHON_USER, FEWSHOT_PHON_ASSISTANT)
        ex_user, ex_assistant = typed or builtin or (FEWSHOT_USER, FEWSHOT_ASSISTANT)
        msgs += [
            {"role": "user", "content": ex_user},
            {"role": "assistant", "content": ex_assistant},
        ]
    user = context.strip() + "\n\n" + query.strip()
    if extra.strip():
        user = extra.strip() + "\n\n" + user
    msgs.append({"role": "user", "content": user})
    return msgs


def followup_messages(base: Messages, prior_text: str, instruction: str) -> Messages:
    """Continue a conversation: prior model output, then a follow-up instruction.

    Thinking blocks must be stripped from prior_text by the caller (Gemma-4
    multi-turn rule: never re-feed thought blocks).
    """
    return [
        *base,
        {"role": "assistant", "content": prior_text},
        {"role": "user", "content": instruction},
    ]
