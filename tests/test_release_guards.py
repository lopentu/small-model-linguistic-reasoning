"""Check upload opt-in and visibility gates without network or model access."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dev/sft'))
from release_io import require_private_repo
os.environ['ADAPTER_REPO'] = 'synthetic/adapter'
import merge_verified as merge
class API:
    def __init__(self, private):
        self.private = private
        self.uploads = []
    def repo_info(self, *args, **kwargs):
        return SimpleNamespace(private=self.private)
    def create_repo(self, *args, **kwargs):
        pass
    def upload_file(self, **kwargs):
        self.uploads.append(kwargs)
for private in (True, False):
    api = API(private)
    merge.UPLOAD = False
    merge.upload(api, 'file', path_or_fileobj='synthetic.bin', repo_id='synthetic/model')
    assert not api.uploads
    merge.UPLOAD = True
    merge.OUT_REPO = 'synthetic/model'
    try:
        merge.upload(api, 'file', path_or_fileobj='synthetic.bin', repo_id=merge.OUT_REPO)
    except SystemExit:
        assert not private and not api.uploads
    else:
        assert private and len(api.uploads) == 1
    try:
        require_private_repo(api, 'synthetic/data', repo_type='dataset')
    except SystemExit:
        assert not private
    else:
        assert private
merge.OUT_REPO = ''
try:
    merge.prepare_upload(API(True))
except SystemExit:
    pass
else:
    raise AssertionError('Empty upload destination accepted')
spec = importlib.util.spec_from_file_location('leaderboard', ROOT / 'dev/leaderboard_delta.py')
board = importlib.util.module_from_spec(spec)
spec.loader.exec_module(board)
stream = io.StringIO()
with contextlib.redirect_stdout(stream):
    board.report_selection([], [.1], [.02])
    board.report_determinism([dict(id='synthetic', score=.2, chrf=.4, exact_match=.1)])
text = stream.getvalue()
assert 'skipped' in text and 'scorer is deterministic' not in text
print('PASS: local-only default, private upload gates, missing baseline, descriptive output')
