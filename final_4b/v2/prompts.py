"""Prompt text and message builders for solving, finalizing, and refining answers."""

from __future__ import annotations

Messages = list[dict[str, str]]

import os as _os

SYSTEM_DIRECT = (
    "You are an expert linguist solving a linguistics olympiad puzzle. Deduce the "
    "rules from the data given, then output ONLY a JSON list of the answers to the "
    "asked items, in the order asked — no explanation, no markdown fences, just the "
    "JSON list of strings."
)

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

# FINALIZE pass: truncated analysis never reaches FINAL ANSWERS — the derivations
# are already committed in the text, so one short follow-up extracts them
# Delivery-FIRST: demand the answer block ONLY (no rules, no more analysis). Verbose models
# (Qwen3.6-35B) burned the whole finalize budget re-deriving RULES and never reached the
# answers → 88% no-marker. Rules/explanation still come from the solve pass; finalize's one
# job is to get the answers out. (v1 solver/script.py keeps an identical copy.)
FINALIZE_MSG = (
    "Stop. Do NOT analyze or write rules. Output ONLY this, nothing before it: a line reading "
    "exactly\nFINAL ANSWERS:\nthen one bare answer per asked item, in order, one per line — "
    "best guess if unsure. Nothing after the answers."
)

REFINE_MSG = (
    "Review your previous solution above. Automated checks flagged these issues:\n"
    "{verdicts}\n"
    "Re-derive the rules, test them against EVERY example given, and output the "
    "two blocks again — RULES: with 2-4 bullets, then FINAL ANSWERS: with exactly "
    "{k} lines, one bare answer per line, best guess if unsure. Write only the "
    "unknown part without copying the question; for matching tasks, labels only."
)


def solve_messages(context: str, query: str, *, fewshot: bool) -> Messages:
    """Build the initial solve conversation for one problem."""
    mode = _os.environ.get("PROMPT_MODE", "")
    if mode == "direct":
        # E30a: trained direct-JSON delivery — no exemplar, no CoT scaffold
        return [
            {"role": "system", "content": SYSTEM_DIRECT},
            {"role": "user", "content": context.strip() + "\n\n" + query.strip()},
        ]
    sys_txt = SYSTEM
    ex_assistant = FEWSHOT_ASSISTANT
    if mode == "reason_json":
        # E30b: same reasoning scaffold, delivery = one JSON list after the marker;
        # the exemplar must DEMONSTRATE the JSON tail, not the old line format
        import json as _json

        sys_txt = SYSTEM.replace(
            "one answer per line, in the order the items are asked",
            "a single JSON list of the answers, in the order the items are asked",
        ).replace(
            "no numbering, no quotes, no commentary",
            "each answer a JSON string, no numbering, no commentary",
        )
        head, _, tail = FEWSHOT_ASSISTANT.partition("FINAL ANSWERS:")
        if tail:
            lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
            ex_assistant = head + "FINAL ANSWERS:\n" + _json.dumps(lines, ensure_ascii=False)
    msgs: Messages = [{"role": "system", "content": sys_txt}]
    if fewshot:
        msgs += [
            {"role": "user", "content": FEWSHOT_USER},
            {"role": "assistant", "content": ex_assistant},
        ]
    msgs.append({"role": "user", "content": context.strip() + "\n\n" + query.strip()})
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
