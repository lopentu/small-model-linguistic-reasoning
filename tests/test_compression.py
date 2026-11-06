"""Validate deletion-only constraints using invented text."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dev/sft'))
import build_dataset as b
assert b.is_subsequence(b._cave_words('Birds fly.'), b._cave_words('Birds can fly.'))[0]
assert not b.is_subsequence(b._cave_words('Birds swim.'), b._cave_words('Birds can fly.'))[0]
assert not b.is_subsequence(b._cave_words('fly Birds'), b._cave_words('Birds can fly'))[0]
assert not b.is_subsequence(b._cave_words('SG'), b._cave_words('sg'))[0]
assert b.is_subsequence(b._cave_words('Birds'), b._cave_words('birds'))[0]
print('PASS: deletion, ordering, insertion rejection, and restricted recasing')
