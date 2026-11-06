"""Replay shuffled solve and completion events before voting on synthetic answers."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('curve', ROOT / 'dev/local/e47_ncurve.py')
curve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curve)
os.environ.update(ASSIGNMENT='0', DETLANES='0', LIVE_ANSWERS='0')
events = [
    dict(event='sample', id='synthetic', kind='solve', text='["blue", "red"]', perm=[1, 0]),
    dict(event='sample', id='synthetic', kind='refine', text='not a solve sample'),
    dict(event='sample', id='synthetic', kind='solve', text='Thinking.', perm=[1, 0]),
    dict(event='sample', id='synthetic', kind='finalize', slot=2, text='FINAL ANSWERS:\nblue\nred'),
]
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    transcript = d / 'transcript.jsonl'
    transcript.write_text(''.join(json.dumps(e) + '\n' for e in events), encoding='utf-8')
    pool = curve.load_pool(transcript)
    assert len(pool['synthetic']) == 2
    assert pool['synthetic'][1]['text'] == 'Thinking.\nFINAL ANSWERS:\nblue\nred'
    row = dict(id='synthetic', context='Synthetic colors.', query='1. red\n2. blue', task_type='translation')
    target = d / 'votes.csv'
    curve.write_vote(target, {'synthetic': row}, pool, pool)
    output = list(csv.DictReader(target.open()))
    assert json.loads(output[0]['pred']) == ['red', 'blue'], output
    transcript.write_text(json.dumps(events[-1]), encoding='utf-8')
    try:
        curve.load_pool(transcript)
    except ValueError:
        pass
    else:
        raise AssertionError('Finalize without its preceding sample was accepted')
print('PASS: sample permutations, completion replay, slot validation, and voted item order')
