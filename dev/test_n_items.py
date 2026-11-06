"""Regression test for n_items() — the answer-count bug fixed in E21."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _load_v1_n_items():
    """Import n_items from the self-contained v1 script without running main()."""
    spec = importlib.util.spec_from_file_location("_v1solver", ROOT / "solver" / "script.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # module guards main() under __main__, safe to import
    return mod.n_items


def main() -> None:
    """Run all n_items guards; loud failure, quiet success."""
    v1 = _load_v1_n_items()
    sys.path.insert(0, str(ROOT / "solver"))
    from v2.textproc import n_items as v2

    fx = ROOT / "dev" / "mock-eval" / "fixtures"
    df = pd.read_csv(fx / "test.csv", dtype=str).fillna("")
    gold = {
        r["id"]: len(ast.literal_eval(r["answer"]))
        for _, r in pd.read_csv(fx / "answers.csv", dtype=str).fillna("").iterrows()
    }

    known_hard = {"012021020400", "012017010200"}
    unexpected = []
    for _, r in df.iterrows():
        k1 = v1(r["query"], r["context"], r["task_type"])
        k2 = v2(r["query"], r["context"], r["task_type"])
        assert k1 == k2, f"v1/v2 drift on {r['id']}: {k1} != {k2}"
        if k1 < gold[r["id"]] and r["id"] not in known_hard:
            unexpected.append(f"{r['id']} {r['task_type']} got={k1} gold={gold[r['id']]}")
    assert not unexpected, "NEW under-count rows (regression):\n" + "\n".join(unexpected)

    letters10 = "\n".join(f"{c}. y" for c in "ABCDEFGHIJ")
    cases = [
        # name, query, context, task_type, expected
        ("absolute labels count, not max", "Translate:\n17. a\n18. b\n19. c", "", "translation", 3),
        ("absolute labels with a gap", "Translate:\n17. a\n19. c", "", "translation", 2),
        (
            "incidental paren can't pre-empt",
            "See (1). Translate:\n17. a\n18. b\n19. c",
            "",
            "translation",
            3,
        ),
        ("range-only header (en dash)", "Fill in the blanks (1–18).", "", "fill_blanks", 18),
        ("range-only header (hyphen)", "Answer questions 1-5:", "", "translation", 5),
        ("num_to_text inline digits", "Write out in X:\n1 18 27 105 290", "", "num_to_text", 5),
        (
            "num_to_text stray paren (over-count OK)",
            "Write out (problem 5):\n14\n108\n151\n402\n1533",
            "",
            "num_to_text",
            5,
            False,
        ),
        ("num_to_text range", "Write out: 1-5", "", "num_to_text", 5),
        (
            "text_to_num no blank sep",
            "Write in digits:\nword one\nword two\nword three",
            "x | 1",
            "text_to_num",
            3,
        ),
        (
            "match_letters lowercase keys",
            "Determine.",
            "\n".join(f"{c}. y" for c in "abcdefghij"),
            "match_letters",
            10,
        ),
        (
            "match_letters unequal sides",
            "Determine the correspondences.",
            "\n".join(f"{i}. x" for i in range(1, 6)) + "\n" + letters10,
            "match_letters",
            10,
        ),
        ("no signal at all", "Solve the problem.", "", "", 1),
    ]
    for case in cases:
        name, q, c, tt, exp = case[:5]
        exact = case[5] if len(case) > 5 else True
        g1, g2 = v1(q, c, tt), v2(q, c, tt)
        assert g1 == g2, f"v1/v2 drift on case '{name}': {g1} != {g2}"
        if exact:
            assert g1 == exp, f"case '{name}': expected {exp}, got {g1}"
        else:
            assert g1 >= exp, f"case '{name}': expected >= {exp} (over-count OK), got {g1}"

    print(f"OK: v1==v2 on {len(df)} fixture rows + {len(cases)} cases; 0 undercounts")


if __name__ == "__main__":
    main()
