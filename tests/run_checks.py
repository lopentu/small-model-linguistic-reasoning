"""Run CPU-only release checks in isolated processes, without research data."""
from pathlib import Path
import ast
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT / 'provenance.json').read_text())
    for path, record in manifest['files'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == record['release_sha256'], path
    for path in ROOT.rglob('*.py'):
        if '.venv' not in path.parts:
            ast.parse(path.read_text(), filename=str(path))
    print('PASS: release file hashes and Python syntax', flush=True)
    for name in ['test_ensemble', 'test_confidence', 'test_delivery_deadline', 'test_markerless_parse', 'test_retrieval']:
        subprocess.run([sys.executable, str(ROOT / 'dev' / (name + '.py'))], cwd=ROOT, check=True)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for name, header, rows in [
            ('gold.csv', ['id', 'answer'], [('synthetic-a', "['red', ['blue', 'azure']]")]),
            ('pred.csv', ['id', 'pred'], [('synthetic-a', '["RED", "azure"]')]),
            ('base.csv', ['id', 'pred'], [('synthetic-a', '["red", "wrong"]')])]:
            with (d / name).open('w', newline='') as f:
                w = csv.writer(f); w.writerow(header); w.writerows(rows)
        split = d / 'split.json'
        split.write_text(json.dumps({'search': ['synthetic-a'], 'holdout': [], 'problem_of': {'synthetic-a': 'synthetic-problem'}}))
        result = subprocess.run([sys.executable, str(ROOT / 'dev/score_local.py'), str(d / 'pred.csv'), str(d / 'gold.csv'), '--split', 'search', '--vs', str(d / 'base.csv')], env={**os.environ, 'SPLIT_JSON': str(split)}, capture_output=True, text=True, check=True)
        assert 'EM=1.0000' in result.stdout and 'PAIRED' in result.stdout, result.stdout
        split.write_text(json.dumps({'search': ['synthetic-a'], 'holdout': [], 'problem_of': {}}))
        fallback = subprocess.run([sys.executable, str(ROOT / 'dev/score_local.py'), str(d / 'pred.csv'), str(d / 'gold.csv')], env={**os.environ, 'SPLIT_JSON': str(split)}, capture_output=True, text=True, check=True)
        assert 'WARNING:' in fallback.stdout and 'row fallback for unmapped IDs' in fallback.stdout, fallback.stdout

    subprocess.run([sys.executable, str(ROOT / 'tests/test_compression.py')], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'tests/test_final_config.py')], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'tests/test_final_pipeline.py')], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'tests/test_release_guards.py')], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'tests/test_saved_votes.py')], cwd=ROOT, check=True)
    print('PASS: synthetic scorer CLI, compression, final pipeline, release guards, and saved votes', flush=True)

if __name__ == '__main__':
    main()
