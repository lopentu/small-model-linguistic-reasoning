"""Exercise the final CSV pipeline with synthetic model responses."""
import csv,json,os,runpy,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'final_4b'))
from v2 import backends, prompts
calls=[]
def answer(messages, greedy, max_new):
 assert messages[0]['content'] == prompts.SYSTEM_DIRECT
 calls.append(greedy)
 return '["red", "blue"]'
backends.make_backend=lambda cfg:backends.MockBackend(answer)
with tempfile.TemporaryDirectory() as d:
 os.chdir(d)
 with open('test.csv','w') as f:
  w=csv.DictWriter(f,fieldnames=['id','context','query','work_lang','task_lang','task_type','eval_type'])
  w.writeheader();w.writerow(dict(id='synthetic',context='Synthetic color mapping.',query='1. red\n2. blue',work_lang='English',task_lang='Synthetic',task_type='translation',eval_type='translation'))
 for k in list(os.environ):
  if k in {'N_SAMPLES','FEWSHOT','THINK','REFINE_ROUNDS','TRANSCRIPT','TIME_LIMIT_S','PROMPT_MODE','ASSIGNMENT','LIVE_ANSWERS','SEED_BASE'} or k.startswith('VOTE_'):
   os.environ.pop(k,None)
 os.environ['TEST_CSV']=str(Path('test.csv').resolve())
 runpy.run_path(str(root/'final_4b/script.py'),run_name='__main__')
 rows=list(csv.DictReader(open('submission.csv')))
 assert len(rows)==1 and json.loads(rows[0]['pred'])==['red','blue'],rows
 assert calls==[True,False,False,False,False],calls
 assert not Path('submission.csv.tmp').exists()
 print('PASS: final entry point writes scored-format answers with five mock passes')
