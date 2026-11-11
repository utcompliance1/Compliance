"""Frozen baseline control on the revised-prompt trial's same 600 cases."""
import json,os,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'jev'))
from baseline_eval_jev import load_local_env,predict,MODEL,wilson_interval
from prompts import FOCUSED
from typesafe_sdk import TypeSafeClient,RetryPolicy
folder=HERE/'pass3'
plan=json.loads((folder/'plan.json').read_text());ids=set(plan['evaluation_ids'])
rows=[r for r in json.loads((ROOT/'jev/full-test-focused/jev_sample.json').read_text())['sample'] if r['row_id'] in ids]
assert len(rows)==600
(folder/'control_plan.json').write_text(json.dumps({'scope':'Matched reference, not another tuned method. Original focused prompt; original pass1 threshold 0.94. No selection on this batch.','threshold':.94,'prompt':FOCUSED,'row_ids':sorted(ids)},indent=2)+'\n')
load_local_env()
with TypeSafeClient(api_key=os.environ['TYPESAFE_API_KEY'],base_url='https://api.typesafe.ai',timeout=30,retry=RetryPolicy(max_retries=2)) as client,(folder/'control_predictions.jsonl').open('x') as f:
 def query(r):return {'row_id':r['row_id'],'true':r['decision'],**predict(client,r['text'],MODEL,FOCUSED)}
 results=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for r in pool.map(query,rows):
   results.append(r);f.write(json.dumps(r)+'\n');f.flush()
   if len(results)%100==0:print('Matched control:',len(results),'/600',flush=True)
accepted=[r for r in results if r['confidence']>=.94];k=sum(r['pred']==r['true'] for r in accepted);n=len(accepted)
new={r['row_id']:r for r in map(json.loads,(folder/'evaluation_predictions.jsonl').read_text().splitlines())}
rule=json.loads((folder/'rule.json').read_text());t=rule['minimum_confidence']
isaccepted=lambda r:r.get('choice','HumanReview')!='HumanReview' and r.get('confidence',-1)>=t
new_only=[new[r['row_id']] for r in results if r['confidence']<.94 and isaccepted(new[r['row_id']])]
summary={'total':600,'accepted':n,'acceptance':n/600,'correct':k,'accuracy':k/n if n else None,'wilson95':wilson_interval(k,n),'new_prompt_only_accepted':len(new_only),'new_prompt_only_correct':sum(r['pred']==r['true'] for r in new_only),'new_api_calls':600}
(folder/'control_metrics.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
