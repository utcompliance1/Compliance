"""Live JEV experiment: explicitly ask for evidence-based automation or deferral."""
import csv, hashlib, importlib.util, json, math, random, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'jev'))
from baseline_eval_jev import load_local_env, MODEL, wilson_interval
from typesafe_sdk import TypeSafeClient, RetryPolicy
import os

QUESTIONS={'routing':{'type':'choice','instructions':{
 'task':'Predict the final independent medical review outcome ONLY when the supplied clinical facts make it sufficiently determinate for automation. Otherwise defer to a human reviewer. Choose AutomateUpheld, AutomateOverturned, or HumanReview.',
 'evidence_test':'Identify the requested service and the clinical fact that directly establishes or refutes its need for THIS patient. Distinguish explicit negative findings from missing information. Evaluate whether the same facts could reasonably support the opposite review outcome. If they could, use HumanReview.',
 'patterns':[
 'For inpatient, residential, detoxification or intensive outpatient care: explicit stability, absence of acute danger or withdrawal, preserved functioning and feasible lower-intensity care can support Upheld. Persistent dangerous symptoms or functional impairment despite adequate lower-intensity care can support Overturned. Connect the findings to the requested dates and intensity.',
 'For drugs or treatments: documented failure, intolerance or contraindication of relevant alternatives, persistent disease and an established indication for the specific requested treatment can support Overturned. Failure alone does not establish benefit of an investigational treatment.',
 'Explicit findings that the specific service is unnecessary, inappropriate for the indication, or unsupported by evidence can support Upheld when grounded in clinical facts rather than merely repeating the insurer allegation.',
 'Generic diagnosis, request and insurer denial without discriminating clinical details must go to HumanReview. Mixed evidence, unclear indication, missing critical severity or treatment history, or truncated facts that change the interpretation also require HumanReview.'],
 'boundaries':'The original denial is not the review outcome. Do not infer a negative clinical finding from an omitted fact. Do not invent guidelines, policy exclusions, treatment failures or indications. Words such as documented or did not are not sufficient by themselves. No preference for Upheld over Overturned; automate either when concrete facts resolve the case.'},
 'criteria':{
 'AutomateUpheld':'Concrete patient-specific findings establish that the requested care, intensity or indication is not warranted or supported. The facts favor sustaining the denial and no material contrary evidence or critical unresolved issue remains.',
 'AutomateOverturned':'Concrete patient-specific findings establish the need for the requested care and undermine the denial rationale, such as persistent disease plus documented failure/intolerance of relevant alternatives or dangerous impairment requiring escalation. No material contrary evidence or critical unresolved issue remains.',
 'HumanReview':'The description contains only a diagnosis/request/denial, lacks decisive clinical evidence, has material conflicting evidence, or needs medical/policy assumptions to choose the outcome reliably.'}}}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,o):p.write_text(json.dumps(o,indent=2)+'\n')
def metrics(rows,t):
 a=[r for r in rows if r.get('choice','HumanReview')!='HumanReview' and r.get('confidence',-1)>=t]
 k=sum(r['pred']==r['true'] for r in a);n=len(a)
 return {'total':len(rows),'accepted':n,'human_review':len(rows)-n,'acceptance':n/len(rows),'correct':k,'errors':n-k,'accuracy':k/n if n else None,'wilson95':wilson_interval(k,n),'api_failures':sum('error' in r for r in rows)}

def main():
 folder=HERE/'pass3'
 if folder.exists():
  # Only an authentication-blocked attempt with zero successful cases may restart.
  assert not (folder/'rule.json').exists(), 'A frozen/evaluated trial cannot be overwritten'
  previous=[json.loads(line) for line in (folder/'calibration_predictions.jsonl').read_text().splitlines()]
  assert previous and all(r.get('http_status') in [401,403] for r in previous), 'Only authentication failures may restart'
  assert json.loads((folder/'prompt.json').read_text())==QUESTIONS, 'Cannot change frozen prompt'
  failures=folder/'authentication_failures.jsonl'
  with failures.open('a') as f:
   for r in previous:f.write(json.dumps(r)+'\n')
  (folder/'calibration_predictions.jsonl').unlink()
 else:folder.mkdir()
 old=[json.loads(s) for s in (ROOT/'jev/full-test-focused/jev_predictions.jsonl').read_text().splitlines()]
 discovery=set(json.loads((HERE/'splits.json').read_text())['discovery'])
 calibration=[r for r in old if r['row_id'] in discovery];random.Random(33).shuffle(calibration);calibration=calibration[:600]
 normalize=lambda t:' '.join(t.lower().split())
 seen={normalize(r['text']) for r in old}
 manifest=json.loads((ROOT/'jev/full-test-focused/jev_sample.json').read_text())
 fresh=[]
 for r in manifest['sample']:
  key=normalize(r['text'])
  if key not in seen:fresh.append(r);seen.add(key)
 random.Random(34).shuffle(fresh);evaluation=fresh[:600]
 assert len(evaluation)==600 and len(calibration)==600
 dump(folder/'prompt.json',QUESTIONS)
 dump(folder/'plan.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'model':MODEL,'calibration_ids':[r['row_id'] for r in calibration],'evaluation_ids':[r['row_id'] for r in evaluation],'prompt_sha256':sha(folder/'prompt.json'),'source_manifest_sha256':sha(ROOT/'jev/full-test-focused/jev_sample.json'),'threshold_candidates':[0,.2,.4,.5,.6,.7,.8,.9,.95],'selection':'Maximize acceptance with calibration accuracy >=92% and at least 60 accepted; if none, maximize Wilson lower bound among thresholds with >=60 accepted. Evaluation never selects threshold.','hypothesis':'Ask JEV to distinguish decisive explicit negative clinical findings or documented need after failed alternatives from missing/ambiguous evidence.','input':'text only; no labels, metadata or prior predictions','new_calls_planned':1200})
 load_local_env()
 with TypeSafeClient(api_key=os.environ['TYPESAFE_API_KEY'],base_url='https://api.typesafe.ai',timeout=30,retry=RetryPolicy(max_retries=2)) as client:
  def query(row):
   start=time.monotonic()
   try:
    result=client.system_one(state=row['text'],questions=QUESTIONS,model=MODEL)
    a=result.choices['routing'];assert a.choice in QUESTIONS['routing']['criteria']
    assert math.isfinite(a.confidence) and 0<=a.confidence<=1
    return {'row_id':row['row_id'],'true':row['decision'],'choice':a.choice,'pred':{'AutomateUpheld':'Upheld','AutomateOverturned':'Overturned','HumanReview':None}[a.choice],'confidence':a.confidence,'probabilities':dict(a.probabilities),'model':result.model,'input_tokens':result.usage.input_tokens,'latency_seconds':time.monotonic()-start}
   except Exception as e:
    return {'row_id':row['row_id'],'true':row['decision'],'error':type(e).__name__,'http_status':getattr(e,'status',None)}
  def batch(rows,name):
   results=[]
   with (folder/f'{name}_predictions.jsonl').open('x') as f:
    first=query(rows[0]);results.append(first);f.write(json.dumps(first)+'\n');f.flush()
    if 'error' in first:raise RuntimeError(f"Integration failure: {first['error']}; HTTP {first['http_status']}")
    with ThreadPoolExecutor(max_workers=4) as pool:
     for r in pool.map(query,rows[1:]):
      results.append(r);f.write(json.dumps(r)+'\n');f.flush()
      if len(results)%100==0:print(f'{name}: {len(results)}/600',flush=True)
   return results
  cal=batch(calibration,'calibration')
  sweep=[{'threshold':t,**metrics(cal,t)} for t in [0,.2,.4,.5,.6,.7,.8,.9,.95]]
  candidates=[m for m in sweep if m['accepted']>=60]
  if not candidates: chosen={'threshold':1.01}
  else:
   good=[m for m in candidates if m['accuracy']>=.92]
   chosen=max(good,key=lambda m:m['accepted']) if good else max(candidates,key=lambda m:m['wilson95'][0])
  t=chosen['threshold'];dump(folder/'calibration_sweep.json',sweep)
  dump(folder/'rule.json',{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'accept_choices':['AutomateUpheld','AutomateOverturned'],'minimum_confidence':t,'fallback':'HumanReview','calibration_target_met':bool(candidates and any(m['accuracy']>=.92 for m in candidates))})
  print('Frozen threshold:',t,'calibration:',metrics(cal,t),flush=True)
  test=batch(evaluation,'evaluation')
 m=metrics(test,t);c=metrics(cal,t);dump(folder/'metrics.json',{'calibration':c,'evaluation':m,'new_api_calls':1200,'prompt_sha256':sha(folder/'prompt.json'),'rule_sha256':sha(folder/'rule.json')})
 with (folder/'decisions.csv').open('w') as f:
  w=csv.writer(f,lineterminator="\n");w.writerow(['row_id','jev_choice','jev_confidence','route','prediction','true_label','correct'])
  for r in test:
   accepted=r.get('choice','HumanReview')!='HumanReview' and r.get('confidence',-1)>=t
   w.writerow([r['row_id'],r.get('choice'),r.get('confidence'),'automate' if accepted else 'human',r.get('pred'),r['true'],r.get('pred')==r['true']])
 fmt=lambda v: 'N/A' if v is None else f'{v:.2%}'
 table='\n'.join(f"| {name} | {v['total']} | {v['accepted']} | {fmt(v['acceptance'])} | {v['correct']}/{v['accepted']} | {fmt(v['accuracy'])} | {v['wilson95']} |" for name,v in [('Calibration (threshold selection)',c),('Held-out evaluation',m)])
 (folder/'README.md').write_text(f'''# Pass 3: ask JEV for evidence-based automation eligibility

This trial changes the actual JEV question and criteria. JEV chooses AutomateUpheld, AutomateOverturned, or HumanReview from case text. The prompt asks for decisive patient-specific evidence, distinguishes explicit negative findings from missing information, and defers mixed or generic cases. See prompt.json for exact instructions.

Pattern hypothesis: explicit stability / no acute need can resolve intensity-of-care appeals; persistent disease with documented failed alternatives can resolve supported-treatment appeals. The discovery association for “did not” + Upheld + confidence >=0.80 was 84/85 correct. This motivated a semantic question, not a keyword filter or a causal claim.

| Split | Tasks | Accepted | Acceptance | Correct/accepted | Accuracy | 95% Wilson interval |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{table}

Frozen rule: JEV must choose an automation outcome with native three-choice confidence >= {t}. Everything else, including API failure, goes to a human. Human-review cases in evaluation: {m['human_review']}/600. Observed 90% target: {'MET' if m['accuracy'] is not None and m['accuracy']>=.90 else 'NOT MET'}.

There are 600 live calibration cases and 600 separate live evaluation cases. Evaluation cases are outside all 6,915 cached baseline predictions, with normalized exact text overlap excluded. Calibration selects the highest-coverage threshold with >=92% observed accuracy and >=60 acceptances; fallback maximizes the lower Wilson bound with >=60 acceptances and is explicitly exploratory. No evaluation result changes the prompt or threshold. Calibration intervals are selection-biased; evaluation intervals are approximate and not adjusted for repeated trials. A point estimate above 90% is not a guarantee. Near-duplicate group separation is not verified.

Model: {MODEL}. Input: text only, never outcome labels or full_text. Source remains the historical cleaned IMR dataset, not prospective production data. Some text retains clinical-review-style narrative; performance may not transfer to incoming appeals without that narrative. New and old trials use different evaluation batches, so their difference is not a controlled causal estimate of prompt improvement. No human decisions were measured or executed.

Artifacts: plan.json records split IDs and selection procedure; prompt.json and rule.json freeze instructions and threshold; predictions JSONL and decisions.csv permit independent scoring; metrics.json stores results. Reproduce with `.venv/bin/python jev/selective-automation/prompt_trial.py`; completed results are protected against overwrite. Only a zero-success authentication failure may restart in place.
''')
 print(json.dumps({'calibration':c,'evaluation':m},indent=2),flush=True)
if __name__=='__main__':main()
