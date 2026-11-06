"""Check archived entry-point defaults without importing a GPU backend."""
from pathlib import Path
import ast
import os
import sys
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'final_4b'))
from v2.config import Config
from v2.pipeline import default_passes, SolvePass
# Read literal setdefault calls without executing the inference entry point.
for node in ast.walk(ast.parse((root / 'final_4b/script.py').read_text())):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'setdefault':
        if all(isinstance(x, ast.Constant) for x in node.args):
            key, value = [x.value for x in node.args]
            os.environ[key] = value
cfg = Config.from_env()
solves = [p for p in default_passes(cfg) if isinstance(p, SolvePass)]
assert cfg.n_samples == 5 and len(solves) == 5
assert [p.greedy for p in solves] == [True, False, False, False, False]
assert cfg.max_refine_rounds == 0 and not cfg.fewshot and not cfg.think
print('PASS: final entry-point defaults and greedy-first pass schedule')
