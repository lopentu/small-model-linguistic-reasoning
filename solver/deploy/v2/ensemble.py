"""Query-item permutations and restoration of answer order for voting."""

from __future__ import annotations

import random
import re

# an item line starts with its label: "1.", "12)", "(3)" — the same markers
# textproc.n_items counts, so a row it can count is a row we can usually split
ITEM_RE = re.compile(r"^\s*(?:\((\d{1,3})\)|(\d{1,3})[.)])\s+(.*\S)\s*$")


def split_query_items(query: str, k: int) -> tuple[list[str], list[str]] | None:
    """Split a query into (header lines, k labelled item bodies).

    Returns None unless exactly k labelled items are found — a partial split
    would silently drop or duplicate an answer slot, which is far worse than
    not shuffling this row.
    """
    header: list[str] = []
    items: list[str] = []
    seen_item = False
    for line in query.splitlines():
        m = ITEM_RE.match(line)
        if m:
            seen_item = True
            items.append(m.group(3))
        elif not seen_item:
            header.append(line)
        elif line.strip():
            return None  # prose after the list: structure we don't model
    if len(items) != k:
        return None
    return header, items


def permuted_query(query: str, k: int, perm: list[int]) -> str | None:
    """Rebuild the query with items in `perm` order, renumbered 1..k.

    perm[j] is the ORIGINAL index of the item shown in position j.
    """
    split = split_query_items(query, k)
    if split is None:
        return None
    header, items = split
    lines = [*header]
    lines += [f"{j + 1}. {items[orig]}" for j, orig in enumerate(perm)]
    return "\n".join(lines)


def unpermute(values: list, perm: list[int], k: int, fill: object = "") -> list:
    """Map a per-item list from presented order back to original item order.

    Used for both the answers and their confidence weights: the weights are
    computed against the TEXT (presented order), so they must travel home by the
    same mapping or item 3's confidence would be applied to item 7's answer.
    """
    out = [fill] * k
    for j, orig in enumerate(perm):
        if j < len(values) and 0 <= orig < k:
            out[orig] = values[j]
    return out


def vote_permutations(k: int, n: int, *, seed: int) -> list[list[int]]:
    """N item orders for one problem: identity first, then derangement-ish shuffles.

    Identity stays first so the ensemble always contains the exact input the
    deployed recipe uses today — the shuffles can only add perspectives, never
    replace the validated one. Seeded per problem so a resumed run rebuilds the
    same orders (a vote's answers are meaningless under a different permutation).
    """
    ident = list(range(k))
    if k < 2:
        return [ident] * n
    rng = random.Random(seed)  # noqa: S311 — sampling diversity, not cryptography
    out = [ident]
    seen = {tuple(ident)}
    guard = 0
    while len(out) < n and guard < 50 * n:
        guard += 1
        cand = ident[:]
        rng.shuffle(cand)
        # for small k the permutation space is tiny; allow repeats once exhausted
        if tuple(cand) in seen and len(seen) < _factorial_capped(k):
            continue
        seen.add(tuple(cand))
        out.append(cand)
    while len(out) < n:  # k=2 with n>2: cycle what we have
        out.append(out[len(out) % max(1, len(seen))])
    return out[:n]


def _factorial_capped(k: int, cap: int = 5000) -> int:
    """k!, capped (we only need to know whether the space is exhausted)."""
    n = 1
    for i in range(2, k + 1):
        n *= i
        if n >= cap:
            return cap
    return n
