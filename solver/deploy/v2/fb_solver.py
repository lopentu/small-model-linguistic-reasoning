"""Deterministic fill_blanks solver for linguini (v2).

Pipeline: parse pipe-tables from context+query -> locate (n) blanks ->
learn per-column-pair string transducers from the row's OWN complete
records -> pick source column / learner / route by leave-one-out (LOO)
accuracy on the row's own attested pairs -> predict every blank.

Learners: copy, affix (Cotterell-style prefix/suffix rules), rewrite
(eager contextual char rules), rewrite2 (conservative variant), infix
(single-site edit with context anchors).  Routes: direct col i->j,
composite i->k->j.  Overlay: proportional analogy over record triples
(A:B :: C:X ported with single-pair affix diffs), used when the direct
route's LOO exact-match is weak.

SECRECY: nothing from the puzzle text is printed except through mask()
(letters -> 'a').  Default run prints aggregate numbers only.
"""
import re
import sys
import unicodedata
from collections import Counter

# Vendored chrF/EM helpers (mirror of the dev scorer's functions), used ONLY
# for row-internal leave-one-out routing against the row's own attested
# records — this module never sees gold answers.

def _nfc(s):
    return unicodedata.normalize("NFC", str(s))


def chrf(pred, gold, max_n=6, beta=2.0):
    p = "".join(_nfc(pred).split())
    g = "".join(_nfc(gold).split())
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    precs, recs = [], []
    for n in range(1, max_n + 1):
        pn, gn = {}, {}
        for i in range(len(p) - n + 1):
            pn[p[i:i + n]] = pn.get(p[i:i + n], 0) + 1
        for i in range(len(g) - n + 1):
            gn[g[i:i + n]] = gn.get(g[i:i + n], 0) + 1
        if not pn and not gn:
            continue
        ov = sum(min(c, gn.get(k, 0)) for k, c in pn.items())
        tp, tg = sum(pn.values()), sum(gn.values())
        precs.append(ov / tp if tp else 0.0)
        recs.append(ov / tg if tg else 0.0)
    if not precs:
        return 0.0
    P, R = sum(precs) / len(precs), sum(recs) / len(recs)
    if P == 0.0 and R == 0.0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * P * R / (b2 * P + R)


def score_item(pred, alts):
    return {
        "em": max(1.0 if _nfc(pred) == _nfc(a) else 0.0 for a in alts),
        "chrf": max(chrf(pred, a) for a in alts),
    }

BLANK_RE = re.compile(r"^\((\d{1,2})\)\s*(.*)$")
DASHES = {"-", "–", "—", "−"}
ENGLISH = {"to", "the", "a", "an", "of", "in", "on", "it", "is", "be", "he", "she",
           "my", "your", "his", "her", "one", "not", "with", "at", "for", "or",
           "and", "you", "i", "we", "they", "this", "that", "sg", "pl", "smth",
           "smb", "oneself", "something", "somebody", "e.g.", "i.e."}
HEADER_WORDS = {"translation", "translations", "meaning", "meanings", "gloss",
                "glosses", "singular", "plural", "word", "words", "form", "forms",
                "base form", "reduplicated form", "pronunciation", "spelling",
                "stem", "noun", "verb"}


def nfc(s):
    return unicodedata.normalize("NFC", s)


def mask(s):
    return re.sub(r"[^\W\d_]", "a", str(s))


def base_char(c):
    d = unicodedata.normalize("NFD", c)
    return d[0].lower() if d else c


def sub_cost(a, b):
    if a == b:
        return 0.0
    ba, bb = base_char(a), base_char(b)
    if ba == bb:
        return 0.25
    va, vb = ba in "aeiouy", bb in "aeiouy"
    na, nb = ba in "mn", bb in "mn"
    if na and nb:
        return 0.4
    if va and vb:
        return 0.5
    if a.isalpha() and b.isalpha() and not va and not vb:
        return 0.7 if (na or nb) else 0.6
    if va != vb and a.isalpha() and b.isalpha():
        return 1.5           # vowel<->consonant: prefer del+ins paths
    return 1.0


def indel_cost(c):
    b = base_char(c)
    if b in "mn":
        return 0.5           # (pre)nasal insertion/deletion is common
    if not c.isalpha():
        return 0.6
    return 0.85


# ---------------------------------------------------------------- parsing

class Blank:
    __slots__ = ("n",)

    def __init__(self, n):
        self.n = n

    def __repr__(self):
        return "(%d)" % self.n


def parse_cells(line):
    out = []
    for raw in line.split("|"):
        c = nfc(raw.strip())
        m = BLANK_RE.match(c)
        if m:
            out.append(Blank(int(m.group(1))))
            rest = m.group(2).strip()
            if rest:
                out.append(rest if rest not in DASHES else None)
        elif c == "" or c in DASHES:
            out.append(None)
        else:
            out.append(c)
    return out


def englishness(cell):
    if not isinstance(cell, str):
        return 0.0
    toks = [t.strip(".,!?:;()'’").lower() for t in cell.split()]
    toks = [t for t in toks if t]
    if not toks:
        return 0.0
    hit = sum(1 for t in toks if t in ENGLISH)
    return hit / len(toks) + (0.3 if len(toks) > 1 else 0.0)


def looks_like_header(cells, data_records):
    strs = [c for c in cells if isinstance(c, str)]
    if any(isinstance(c, Blank) for c in cells):
        return False
    if len(strs) < len(cells):
        return True
    if any(re.search(r"\d", c) for c in strs):
        return True
    if any(c.lower() in HEADER_WORDS for c in strs):
        return True
    caps = sum(1 for c in strs if c[:1].isupper())
    if caps >= 2 and data_records:
        dcaps = dtot = 0
        for r in data_records:
            for c in r:
                if isinstance(c, str):
                    dtot += 1
                    dcaps += c[:1].isupper()
        if dtot and dcaps / dtot < 0.2:
            return True
    return False


def parse_tables(text):
    recs = [parse_cells(ln) for ln in text.splitlines() if "|" in ln]
    if recs and looks_like_header(recs[0], recs[1:]):
        recs = recs[1:]
    return recs


# --------------------------------------------------------------- helpers

def lc_prefix(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def lc_suffix(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[len(a) - 1 - i] == b[len(b) - 1 - i]:
        i += 1
    return i


def longest_common_substring(a, b):
    best = (0, 0, 0)
    bestkey = (0, 1, 1)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                L = dp[i][j]
                key = (L, -(i - L) - (j - L), -(i - L))
                if key > bestkey:
                    bestkey = key
                    best = (L, i - L, j - L)
    return best


def pick(counter):
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


# --------------------------------------------------------------- learners

class IdentityLearner:
    name = "copy"

    def fit(self, pairs):
        return self

    def predict(self, x):
        return x


class AffixLearner:
    name = "affix"

    def fit(self, pairs):
        self.srules = {}
        self.prules = {}
        self.whole = {}
        for s, t in pairs:
            L, pa, pb = longest_common_substring(s, t)
            if L == 0:
                self.whole.setdefault(s, Counter())[t] += 1
                continue
            root = s[pa:pa + L]
            ipref, isuf = s[:pa], s[pa + L:]
            opref, osuf = t[:pb], t[pb + L:]
            for k in range(0, L + 1):
                self.srules.setdefault(root[L - k:] + isuf, Counter())[root[L - k:] + osuf] += 1
                self.prules.setdefault(ipref + root[:k], Counter())[opref + root[:k]] += 1
        return self

    def predict(self, x):
        if x in self.whole:
            return pick(self.whole[x])
        y = x
        cands = [l for l in self.srules if y.endswith(l)]
        if cands:
            lhs = sorted(cands, key=lambda l: (-len(l), -max(self.srules[l].values()), l))[0]
            y = y[:len(y) - len(lhs)] + pick(self.srules[lhs])
            consumed = len(x) - len(lhs)
        else:
            consumed = len(x)
        pc = [l for l in self.prules if x.startswith(l) and len(l) <= consumed]
        if pc:
            lhp = sorted(pc, key=lambda l: (-len(l), -max(self.prules[l].values()), l))[0]
            y = pick(self.prules[lhp]) + y[len(lhp):]
        return y


class RewriteLearner:
    """Contextual character-level rewrite rules from edit alignments.

    out_left=True keys the left context on the OUTPUT built so far, which
    captures left-to-right spreading processes (nasal/vowel harmony)."""

    def __init__(self, conservative=False, out_left=False):
        self.conservative = conservative
        self.out_left = out_left
        base = "rewol" if out_left else "rewrite"
        self.name = base + ("2" if conservative else "")

    def _align(self, s, t):
        n, m = len(s), len(t)
        D = [[0.0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            D[i][0] = D[i - 1][0] + indel_cost(s[i - 1])
        for j in range(1, m + 1):
            D[0][j] = D[0][j - 1] + indel_cost(t[j - 1])
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                sc = sub_cost(s[i - 1], t[j - 1])
                D[i][j] = min(D[i - 1][j - 1] + sc,
                              D[i - 1][j] + indel_cost(s[i - 1]),
                              D[i][j - 1] + indel_cost(t[j - 1]))
        ops = []
        i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                sc = sub_cost(s[i - 1], t[j - 1])
                if abs(D[i][j] - (D[i - 1][j - 1] + sc)) < 1e-9:
                    ops.append(("sub", s[i - 1], t[j - 1]))
                    i, j = i - 1, j - 1
                    continue
            if i > 0 and abs(D[i][j] - (D[i - 1][j] + indel_cost(s[i - 1]))) < 1e-9:
                ops.append(("del", s[i - 1], ""))
                i -= 1
                continue
            ops.append(("ins", "", t[j - 1]))
            j -= 1
        ops.reverse()
        return ops

    def fit(self, pairs):
        self.rules = {}

        def add(key, out):
            self.rules.setdefault(key, Counter())[out] += 1

        for s, t in pairs:
            ops = self._align(s, t)
            chunks = [""]
            src_idx = []
            for op, a, b in ops:
                if op == "ins":
                    chunks[-1] += b
                else:
                    chunks.append(b if op == "sub" else "")
                    src_idx.append(len(chunks) - 1)
            ext = "^" + s + "$"
            add(("^", s[0] if s else "$"), chunks[0])
            add(("^",), chunks[0])
            acc = chunks[0]
            for pos, ci in enumerate(src_idx):
                c, r = s[pos], ext[pos + 2]
                if self.out_left:
                    l = acc[-1] if acc else "^"
                else:
                    l = ext[pos]
                out = chunks[ci]
                add((l, c, r), out)
                add((c, r), out)
                add((l, c), out)
                add((c,), out)
                acc += out
        return self

    def _choose(self, keys, default):
        for lvl, k in enumerate(keys):
            if k not in self.rules:
                continue
            cnt = self.rules[k]
            top, tn = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            tot = sum(cnt.values())
            if not self.conservative:
                return top
            if lvl == 0 and tn / tot > 0.5:
                return top
            if lvl > 0 and tn >= 2 and tn / tot >= 0.7:
                return top
        return default

    def predict(self, x):
        if not x:
            return x
        ext = "^" + x + "$"
        out = self._choose([("^", x[0]), ("^",)], "")
        for pos, c in enumerate(x):
            r = ext[pos + 2]
            if self.out_left:
                l = out[-1] if out else "^"
            else:
                l = ext[pos]
            out += self._choose([(l, c, r), (c, r), (l, c), (c,)], c)
        return out


class InfixLearner:
    """Single contiguous edit site, anchored by surrounding characters."""
    name = "infix"

    def fit(self, pairs):
        self.rules = {}
        self.mids = set()
        for s, t in pairs:
            k1 = lc_prefix(s, t)
            k2 = min(lc_suffix(s, t), min(len(s), len(t)) - k1)
            k2 = max(k2, 0)
            mid_s = s[k1:len(s) - k2]
            mid_t = t[k1:len(t) - k2]
            if mid_s == mid_t:
                continue
            ext = "^" + s + "$"
            l1 = ext[k1]
            l2 = ext[k1 - 1] if k1 >= 1 else ""
            r1 = ext[k1 + len(mid_s) + 1]
            self.mids.add(mid_s)
            for lvl, key in enumerate([(l2, l1, mid_s, r1), (l1, mid_s, r1),
                                       (mid_s, r1), (l1, mid_s), (mid_s,)]):
                self.rules.setdefault((lvl,) + key, Counter())[mid_t] += 1
        return self

    def predict(self, x):
        ext = "^" + x + "$"
        best = None
        for mid in sorted(self.mids, key=lambda m: (-len(m), m)):
            start = 0
            while start <= len(x) - len(mid):
                if x[start:start + len(mid)] == mid:
                    l1 = ext[start]
                    l2 = ext[start - 1] if start >= 1 else ""
                    r1 = ext[start + len(mid) + 1]
                    for lvl, key in enumerate([(l2, l1, mid, r1), (l1, mid, r1),
                                               (mid, r1), (l1, mid), (mid,)]):
                        if mid == "" and lvl > 1:
                            break
                        rk = (lvl,) + key
                        if rk in self.rules:
                            cnt = self.rules[rk]
                            top, tn = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0]
                            cand = (lvl, -len(mid), -tn, start, top, mid)
                            if best is None or cand < best:
                                best = cand
                            break
                start += 1
            # also try empty-mid insertion sites when mid == "" handled above
        if best is None:
            return x
        lvl, _, _, start, mid_t, mid = best
        return x[:start] + mid_t + x[start + len(mid):]


def char_class(c):
    b = base_char(c)
    if b in "mn":
        return "N"
    if b in "aeiouy":
        return "V"
    if c.isalpha():
        return "C"
    return c


class ClusterLearner:
    """Single-site edit generalized by character class: learns a trigger
    class-pattern (e.g. N C V) plus per-character substitution maps, so one
    cluster mutation (delete nasal, devoice stop, front vowel) transfers
    across stems."""
    name = "cluster"

    def fit(self, pairs):
        self.patterns = Counter()
        self.charmap = {}
        aligner = RewriteLearner()
        for s, t in pairs:
            k1 = lc_prefix(s, t)
            k2 = min(lc_suffix(s, t), min(len(s), len(t)) - k1)
            k2 = max(k2, 0)
            mid_s = s[k1:len(s) - k2]
            mid_t = t[k1:len(t) - k2]
            if not mid_s or mid_s == mid_t or len(mid_s) > 4:
                continue
            ops = aligner._align(mid_s, mid_t)
            if any(op == "ins" for op, _a, _b in ops):
                continue
            pat = tuple(char_class(c) for c in mid_s)
            self.patterns[pat] += 1
            for op, a, b in ops:
                self.charmap.setdefault(a, Counter())[b] += 1
        return self

    def predict(self, x):
        for pat, _cnt in sorted(self.patterns.items(),
                                key=lambda kv: (-kv[1], kv[0])):
            L = len(pat)
            for start in range(len(x) - L + 1):
                seg = x[start:start + L]
                if tuple(char_class(c) for c in seg) != pat:
                    continue
                out = []
                known = 0
                for c in seg:
                    if c in self.charmap:
                        out.append(pick(self.charmap[c]))
                        known += 1
                    else:
                        out.append(c)
                if known == 0:
                    continue
                rep = "".join(out)
                if rep == seg:
                    continue
                return x[:start] + rep + x[start + L:]
        return x


class SpreadLearner:
    """Left-anchored bounded rewrite: char maps applied from the word start
    until a terminating character (captures Terena-style nasal spread,
    initial-syllable mutations, etc.)."""
    name = "spread"

    def fit(self, pairs):
        self.cont = {}
        self.term = {}
        self.head = Counter()
        aligner = RewriteLearner()
        for s, t in pairs:
            ops = aligner._align(s, t)
            chunks = [""]
            for op, a, b in ops:
                if op == "ins":
                    chunks[-1] += b
                else:
                    chunks.append(b if op == "sub" else "")
            outs = chunks[1:]
            self.head[chunks[0]] += 1
            T = -1
            for pos, c in enumerate(s):
                if outs[pos] != c:
                    T = pos
            if T < 0:
                continue
            for pos in range(T + 1):
                c = s[pos]
                d = self.cont if pos < T else self.term
                d.setdefault(c, Counter())[outs[pos]] += 1
        return self

    def predict(self, x):
        out = pick(self.head) if self.head else ""
        for pos, c in enumerate(x):
            nc = sum(self.cont.get(c, Counter()).values())
            nt = sum(self.term.get(c, Counter()).values())
            if nc == 0 and nt == 0:
                return out + x[pos:]
            if nt > nc:
                return out + pick(self.term[c]) + x[pos + 1:]
            out += pick(self.cont[c])
        return out


class RedupLearner:
    """Pure-insertion edits where the inserted material may be a copy of
    adjacent material (reduplication), keyed by position predicates."""
    name = "redup"

    @staticmethod
    def _first_vowel(s):
        for idx, c in enumerate(s):
            if base_char(c) in "aeiou":
                return idx
        return -1

    @staticmethod
    def _vowel_positions(s):
        return [idx for idx, c in enumerate(s) if base_char(c) in "aeiou"]

    def fit(self, pairs):
        self.votes = Counter()
        for s, t in pairs:
            if len(t) <= len(s):
                continue
            k1 = lc_prefix(s, t)
            k2 = min(lc_suffix(s, t), len(s) - k1)
            k2 = max(k2, 0)
            if s[k1:len(s) - k2]:
                continue                      # not a pure insertion
            ins = t[k1:len(t) - k2]
            ext = "^" + s + "$"
            l1, r1 = ext[k1], ext[k1 + 1]
            pos_specs = [("idx", k1), ("ridx", len(s) - k1)]
            vps = self._vowel_positions(s)
            if vps:
                pos_specs.append(("v1", k1 - (vps[0] + 1)))
                pos_specs.append(("vL", k1 - (vps[-1] + 1)))
                if len(vps) >= 2:
                    pos_specs.append(("v2", k1 - (vps[1] + 1)))
            copy_specs = [("lit", ins, 0)]
            for cut in range(len(ins) + 1):
                lit, cp = ins[:cut], ins[cut:]
                if cp and k1 - len(cp) >= 0 and cp == s[k1 - len(cp):k1]:
                    copy_specs.append(("Lprev", lit, len(cp)))
                if cp and cp == s[k1:k1 + len(cp)]:
                    copy_specs.append(("Lnext", lit, len(cp)))
                cp2, lit2 = ins[:cut], ins[cut:]
                if cp2 and k1 - len(cp2) >= 0 and cp2 == s[k1 - len(cp2):k1]:
                    copy_specs.append(("prevL", lit2, len(cp2)))
                if cp2 and cp2 == s[k1:k1 + len(cp2)]:
                    copy_specs.append(("nextL", lit2, len(cp2)))
            for p in pos_specs:
                for c in copy_specs:
                    for ctx in [(l1, r1), ("", r1), (l1, ""), ("", "")]:
                        self.votes[(p, c, ctx)] += 1
        return self

    def predict(self, x):
        ext = "^" + x + "$"
        ranked = sorted(self.votes.items(),
                        key=lambda kv: (-(len(kv[0][2][0]) + len(kv[0][2][1])),
                                        -kv[1], str(kv[0])))
        for (p, cs, ctx), _cnt in ranked:
            ptype, pval = p
            vps = self._vowel_positions(x)
            if ptype == "idx":
                k = pval
            elif ptype == "ridx":
                k = len(x) - pval
            elif ptype == "v1":
                if not vps:
                    continue
                k = vps[0] + 1 + pval
            elif ptype == "vL":
                if not vps:
                    continue
                k = vps[-1] + 1 + pval
            else:  # v2
                if len(vps) < 2:
                    continue
                k = vps[1] + 1 + pval
            if not 0 <= k <= len(x):
                continue
            cl, cr = ctx
            if cl and ext[k] != cl:
                continue
            if cr and ext[k + 1] != cr:
                continue
            kind, lit, L = cs
            if kind == "lit":
                ins = lit
            elif kind == "Lprev":
                if k - L < 0:
                    continue
                ins = lit + x[k - L:k]
            elif kind == "Lnext":
                if k + L > len(x):
                    continue
                ins = lit + x[k:k + L]
            elif kind == "prevL":
                if k - L < 0:
                    continue
                ins = x[k - L:k] + lit
            else:  # nextL
                if k + L > len(x):
                    continue
                ins = x[k:k + L] + lit
            return x[:k] + ins + x[k:]
        return x


class EnvelopeWrapper:
    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name

    @staticmethod
    def _envelope(strs):
        p = strs[0]
        for s in strs[1:]:
            p = p[:lc_prefix(p, s)]
        q = strs[0]
        for s in strs[1:]:
            q = q[len(q) - lc_suffix(q, s):]
        mn = min(len(s) for s in strs)
        while len(p) + len(q) > mn - 1 and (p or q):
            if len(p) >= len(q):
                p = p[:-1]
            else:
                q = q[1:]
        return p, q

    def fit(self, pairs):
        srcs = [s for s, _ in pairs]
        tgts = [t for _, t in pairs]
        self.sp, self.ss = self._envelope(srcs) if len(srcs) > 1 else ("", "")
        self.tp, self.ts = self._envelope(tgts) if len(tgts) > 1 else ("", "")
        inner_pairs = []
        for s, t in pairs:
            s2 = s[len(self.sp):len(s) - len(self.ss) or None] if (self.sp or self.ss) else s
            t2 = t[len(self.tp):len(t) - len(self.ts) or None] if (self.tp or self.ts) else t
            inner_pairs.append((s2, t2))
        self.inner.fit(inner_pairs)
        return self

    def predict(self, x):
        if self.sp and x.startswith(self.sp):
            x = x[len(self.sp):]
        if self.ss and x.endswith(self.ss) and len(x) > len(self.ss):
            x = x[:len(x) - len(self.ss)]
        return self.tp + self.inner.predict(x) + self.ts


LEARNERS = ["copy", "affix", "rewrite", "rewrite2", "rewol", "rewol2",
            "infix", "redup", "spread", "cluster"]


def make_learner(name):
    if name == "copy":
        return EnvelopeWrapper(IdentityLearner())
    if name == "affix":
        return EnvelopeWrapper(AffixLearner())
    if name == "rewrite":
        return EnvelopeWrapper(RewriteLearner(False))
    if name == "rewrite2":
        return EnvelopeWrapper(RewriteLearner(True))
    if name == "rewol":
        return EnvelopeWrapper(RewriteLearner(False, out_left=True))
    if name == "rewol2":
        return EnvelopeWrapper(RewriteLearner(True, out_left=True))
    if name == "redup":
        return EnvelopeWrapper(RedupLearner())
    if name == "spread":
        return EnvelopeWrapper(SpreadLearner())
    if name == "cluster":
        return EnvelopeWrapper(ClusterLearner())
    return EnvelopeWrapper(InfixLearner())


_PORT_CACHE = {}


def port(a1, a2, c):
    """Apply the a1->a2 string diff to c (single-pair affix port)."""
    key = (a1, a2, c)
    if key not in _PORT_CACHE:
        lr = AffixLearner().fit([(a1, a2)])
        _PORT_CACHE[key] = lr.predict(c)
    return _PORT_CACHE[key]


# ------------------------------------------------------------- row solver

ANALOGY_GATE = 0.4      # run analogy overlay when direct LOO EM below this


class RowSolver:
    def __init__(self, context, query):
        recs = parse_tables(context) + parse_tables(query)
        widths = Counter(len(r) for r in recs)
        self.W = sorted(widths.items(), key=lambda kv: (-kv[1], -kv[0]))[0][0] if recs else 0
        self.records = []
        self.amb = {}          # blank n -> candidate target cols
        for r in recs:
            self.records.append(self._map_cols(r))
        self.blanks = {}
        for ri, r in enumerate(self.records):
            for col, cell in r.items():
                if isinstance(cell, Blank):
                    self.blanks[cell.n] = (ri, col)
        self._c_pairs = {}
        self._c_loo = {}
        self._c_fit = {}
        self._c_hold = {}

    # ---- column profiles / partial-record mapping
    def _profiles(self):
        full = [r for r in self.records if len(r) == self.W]
        att = {i: set() for i in range(self.W)}
        eng = {i: [] for i in range(self.W)}
        for r in full:
            for i in range(self.W):
                c = r.get(i)
                if isinstance(c, str):
                    att[i].add(c)
                    eng[i].append(englishness(c))
        engcol = {i: (sum(v) / len(v) if v else 0.0) for i, v in eng.items()}
        return att, engcol

    def _map_cols(self, cells):
        if len(cells) == self.W:
            return {i: c for i, c in enumerate(cells)}
        att, engcol = self._profiles()
        gloss_col = max(range(self.W), key=lambda i: engcol[i]) if self.W else 0
        gloss_ok = self.W and engcol[gloss_col] > 0.15
        mapping = {}
        used = set()
        for pos, c in enumerate(cells):
            if isinstance(c, str):
                for i in range(self.W):
                    if i not in used and c in att[i]:
                        mapping[pos] = i
                        used.add(i)
                        break
        for pos, c in enumerate(cells):
            if pos in mapping or not isinstance(c, str):
                continue
            if gloss_ok and englishness(c) >= 0.3 and gloss_col not in used:
                mapping[pos] = gloss_col
                used.add(gloss_col)
        rem = [i for i in range(self.W) if i not in used]
        for pos, c in enumerate(cells):
            if pos in mapping:
                continue
            if isinstance(c, Blank):
                if pos == len(cells) - 1 and gloss_ok and gloss_col in rem:
                    # trailing blank after a form: it is the gloss/translation
                    mapping[pos] = gloss_col
                    rem.remove(gloss_col)
                    continue
                cand = [i for i in rem if not (gloss_ok and i == gloss_col)]
                if len(cand) > 1:
                    self.amb[c.n] = list(cand)
                mapping[pos] = cand[0] if cand else (rem[0] if rem else None)
                if mapping[pos] in rem:
                    rem.remove(mapping[pos])
            elif rem:
                mapping[pos] = rem.pop(0)
        return {mapping[pos]: c for pos, c in enumerate(cells)
                if pos in mapping and mapping[pos] is not None}

    # ---- training data access
    def rec_ids(self, i, j):
        return [ri for ri, r in enumerate(self.records)
                if isinstance(r.get(i), str) and isinstance(r.get(j), str)]

    def pairs(self, i, j, exclude=None):
        key = (i, j, exclude)
        if key not in self._c_pairs:
            self._c_pairs[key] = [(self.records[ri][i], self.records[ri][j])
                                  for ri in self.rec_ids(i, j) if ri != exclude]
        return self._c_pairs[key]

    def fitted(self, i, j, ln, exclude=None):
        key = (i, j, ln, exclude)
        if key not in self._c_fit:
            self._c_fit[key] = make_learner(ln).fit(self.pairs(i, j, exclude))
        return self._c_fit[key]

    # ---- routes: ('d', i, j, ln)  or  ('c', i, k, j, ln)
    def route_predict(self, route, x, exclude=None):
        if route[0] == "d":
            _, i, j, ln = route
            return self.fitted(i, j, ln, exclude).predict(x)
        if route[0] == "v":
            _, i, j = route
            tally = {}
            for ln in LEARNERS:
                em, ch, _n = self.route_loo(("d", i, j, ln))
                w = em + 0.2 * ch
                if w <= 0:
                    continue
                p = self.fitted(i, j, ln, exclude).predict(x)
                tally[p] = tally.get(p, 0.0) + w
            if not tally:
                return x
            return sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        if route[0] == "s":
            _, i, j, lnA, lnB = route
            A, B = self.stacked(i, j, lnA, lnB, exclude)
            return B.predict(A.predict(x))
        if route[0] == "p":
            _, i, j, ln, k = route
            key = ("pf", i, j, ln, k, exclude)
            if key not in self._c_fit:
                ps = self.pairs(i, j, exclude) + self.pairs(i, k, exclude)
                self._c_fit[key] = make_learner(ln).fit(ps)
            return self._c_fit[key].predict(x)
        _, i, k, j, ln = route
        mid = self.fitted(i, k, ln, exclude).predict(x)
        return self.fitted(k, j, ln, exclude).predict(mid)

    def stacked(self, i, j, lnA, lnB, exclude=None):
        key = ("sf", i, j, lnA, lnB, exclude)
        if key not in self._c_fit:
            ps = self.pairs(i, j, exclude)
            A = make_learner(lnA).fit(ps)
            resid = [(A.predict(s), t) for s, t in ps]
            B = make_learner(lnB).fit(resid)
            self._c_fit[key] = (A, B)
        return self._c_fit[key]

    def route_loo(self, route):
        key = route
        if key not in self._c_loo:
            if route[0] == "c":
                i, j = route[1], route[3]
            else:
                i, j = route[1], route[2]
            ids = self.rec_ids(i, j)
            if len(ids) < 2:
                self._c_loo[key] = (0.0, 0.0, len(ids))
            else:
                em = ch = 0.0
                for ri in ids:
                    pred = self.route_predict(route, self.records[ri][i], exclude=ri)
                    sc = score_item(pred, [self.records[ri][j]])
                    em += sc["em"]
                    ch += sc["chrf"]
                self._c_loo[key] = (em / len(ids), ch / len(ids), len(ids))
        return self._c_loo[key]

    STACKS = []          # measured: stacked residual routes overfit row LOO

    def sister_cols(self, j):
        """Columns whose values are near-identical to column j."""
        key = ("sis", j)
        if key not in self._c_hold:
            _chrf = chrf
            out = []
            for k in range(self.W):
                if k == j:
                    continue
                shared = [(r[j], r[k]) for r in self.records
                          if isinstance(r.get(j), str) and isinstance(r.get(k), str)]
                if len(shared) >= 3:
                    sim = sum(_chrf(a, b) for a, b in shared) / len(shared)
                    if sim >= 0.75:
                        out.append(k)
            self._c_hold[key] = out
        return self._c_hold[key]

    def candidate_routes(self, i, j):
        routes = [("d", i, j, ln) for ln in LEARNERS]
        routes.append(("v", i, j))
        for lnA, lnB in self.STACKS:
            routes.append(("s", i, j, lnA, lnB))
        for k in self.sister_cols(j):
            if k != i:
                for ln in ("affix", "rewrite2", "infix"):
                    routes.append(("p", i, j, ln, k))
        for k in range(self.W):
            if k in (i, j):
                continue
            if len(self.rec_ids(i, k)) >= 3 and len(self.rec_ids(k, j)) >= 3:
                for ln in ("affix", "rewrite2"):
                    routes.append(("c", i, k, j, ln))
        return routes

    def best_route(self, i, j):
        key = ("best", i, j)
        if key not in self._c_hold:
            best = None
            for route in self.candidate_routes(i, j):
                em, ch, n = self.route_loo(route)
                cand = (em, ch, n, -LEARNERS.index(route[-1]) if route[-1] in LEARNERS else 0,
                        route[0] == "d", route)
                if best is None or cand > best:
                    best = cand
            self._c_hold[key] = (best[5], best[0], best[1], best[2]) if best else (None, 0, 0, 0)
        return self._c_hold[key]

    # ---- analogy overlay
    def analogy_votes(self, i, j, x, exclude=None):
        ids = [ri for ri in self.rec_ids(i, j) if ri != exclude]
        votes = Counter()
        vals = [(self.records[ri][i], self.records[ri][j]) for ri in ids]
        for a1, b1 in vals:
            for a2, b2 in vals:
                if a1 == a2:
                    continue
                for a3, b3 in vals:
                    if port(a1, a2, a3) == x:
                        d = port(b1, b2, b3)
                        if d:
                            votes[d] += 1
        return votes

    def analogy_loo(self, i, j, fallback_route):
        key = ("aloo", i, j)
        if key not in self._c_hold:
            ids = self.rec_ids(i, j)
            if len(ids) < 4:
                self._c_hold[key] = (0.0, 0.0, 0)
            else:
                em = ch = 0.0
                fired = 0
                for ri in ids:
                    x = self.records[ri][i]
                    votes = self.analogy_votes(i, j, x, exclude=ri)
                    if votes:
                        pred = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                        fired += 1
                    elif fallback_route:
                        pred = self.route_predict(fallback_route, x, exclude=ri)
                    else:
                        pred = x
                    sc = score_item(pred, [self.records[ri][j]])
                    em += sc["em"]
                    ch += sc["chrf"]
                self._c_hold[key] = (em / len(ids), ch / len(ids), fired)
        return self._c_hold[key]

    def medoid(self, j):
        """Column value with highest mean chrF to the rest: the best blind
        guess for an unknown value drawn from this column."""
        key = ("med", j)
        if key not in self._c_hold:
            vals = [r[j] for r in self.records if isinstance(r.get(j), str)]
            if not vals:
                self._c_hold[key] = ""
            else:
                _chrf = chrf
                best = max(vals, key=lambda v: (sum(_chrf(v, w) for w in vals), -len(v), v))
                self._c_hold[key] = best
        return self._c_hold[key]

    def exact_lookup(self, i, j, x):
        cnt = Counter(b for a, b in self.pairs(i, j) if a == x)
        if not cnt:
            return None
        top = cnt.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            return None
        return top[0][0]

    # ---- per-blank prediction
    def predict_blank(self, n):
        ri, j0 = self.blanks[n]
        rec = self.records[ri]
        j_cands = self.amb.get(n, [j0])
        sources = [(i, c) for i, c in sorted(rec.items())
                   if isinstance(c, str) and i not in j_cands]
        if not sources:
            col_vals = [r.get(j0) for r in self.records if isinstance(r.get(j0), str)]
            if col_vals:
                return pick(Counter(col_vals)), "majority"
            return "", "empty"
        # exact lookup: if any (source, target col) has an attested mapping
        exact_hits = []
        for j in j_cands:
            for i, c in sources:
                hit = self.exact_lookup(i, j, c)
                if hit is not None:
                    exact_hits.append((j, i, hit))
        if len({h[0] for h in exact_hits}) == 1:
            j, i, hit = exact_hits[0]
            return hit, f"exact:{j}<-{i}"
        # score routes for every (source, target) combo
        scored = []
        for j in j_cands:
            for i, c in sources:
                route, em, ch, ntr = self.best_route(i, j)
                if route is not None and ntr > 0:
                    scored.append((em, ch, ntr, -i, j, i, c, route))
        if not scored:
            c = sources[0][1]
            return c.lstrip("*"), "copysrc"
        scored.sort(reverse=True)
        em, ch, ntr, _, j, i, c, route = scored[0]
        # analogy overlay (also used to resolve ambiguous target col)
        use_analogy = em < ANALOGY_GATE
        if use_analogy:
            best_an = None
            for jj in j_cands:
                for ii, cc in sources:
                    r2, em2, ch2, n2 = self.best_route(ii, jj)
                    aem, ach, fired = self.analogy_loo(ii, jj, r2)
                    votes = self.analogy_votes(ii, jj, cc)
                    if votes:
                        top, tv = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0]
                        cand = (aem, ach, tv, jj, ii, top)
                        if best_an is None or cand > best_an:
                            best_an = cand
            if best_an is not None and (best_an[0] > em or em == 0.0):
                aem, ach, tv, jj, ii, top = best_an
                return top, f"analogy:{jj}<-{ii}"
        if em == 0.0 and ch < 0.2:
            med = self.medoid(j)
            if med:
                return med, f"medoid:{j}"
        # backcheck: if the reverse mapping is reliably learnable, prefer the
        # highest-LOO forward candidate that round-trips to the source
        if em < 0.9 and len(self.rec_ids(j, i)) >= 4:
            rroute, rem, rch, rn = self.best_route(j, i)
            if rroute is not None and rem >= 0.4 and rem > em:
                cands = [(em, self.route_predict(route, c).strip())]
                for ln in LEARNERS:
                    lem, lch, _ = self.route_loo(("d", i, j, ln))
                    cands.append((lem, self.fitted(i, j, ln).predict(c).strip()))
                cands.sort(key=lambda t: -t[0])
                seen = set()
                for _lem, cand in cands:
                    if not cand or cand in seen:
                        continue
                    seen.add(cand)
                    if self.route_predict(rroute, cand) == c:
                        return cand, f"bck:{j}<-{i}"
        pred = self.route_predict(route, c).strip()
        if pred:
            if route[0] == "d":
                tag = f"{route[-1]}:{j}<-{i}"
            elif route[0] == "v":
                tag = f"vote:{j}<-{i}"
            elif route[0] == "s":
                tag = f"stk.{route[3]}+{route[4]}:{j}<-{i}"
            elif route[0] == "p":
                tag = f"pool{route[4]}.{route[3]}:{j}<-{i}"
            else:
                tag = f"via{route[2]}.{route[-1]}:{j}<-{i}"
            return pred, tag
        return c.lstrip("*"), "copysrc"


# ------------------------------------------------------------------ main
