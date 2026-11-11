"""Nine frozen prompt methods plus one untouched confirmation; live JEV calls."""
import argparse,copy,csv,hashlib,json,math,os,random,sys,time
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'jev'))
from baseline_eval_jev import load_local_env,wilson_interval
from prompts import FOCUSED
from typesafe_sdk import TypeSafeClient,RetryPolicy

def choice(instructions,criteria):return {'type':'choice','instructions':instructions,'criteria':criteria}
BOUNDARY='Use supplied patient facts only. The insurer denial is the starting allegation, not the independent review outcome. Missing information is unknown, not a negative finding. Do not invent policy terms or clinical facts.'
def binary(question,yes,no):return choice(question+' '+BOUNDARY,{'Yes':yes,'No':no})
DECISION=FOCUSED['decision']
SUFFICIENT=binary('Are there concrete patient-specific findings that discriminate whether the SPECIFIC requested service is medically necessary or supported?', 'The description provides clinical findings, treatment response, risk, or indication evidence that resolves the disputed need.','Only a diagnosis/request/denial, generic claims, or materially incomplete/conflicting information is supplied.')
GROUND=binary('Does the clinical evidence go beyond a repeated insurer assertion or a provider request?', 'Actual patient findings, documented treatment outcomes or specific indication evidence are supplied independently of what either party merely alleges.','The claimed necessity or lack of necessity is just an assertion, request, or denial; there are no discriminating patient findings.')
TRIAGE={'Upheld':'Explicit clinical facts support sustaining the denial of this particular requested service.','Overturned':'Explicit clinical facts support reversing the denial for this particular requested service.','Unresolved':'The clinical facts do not resolve the dispute without important assumptions.'}
CONFIGS={
4:{'name':'Separate outcome prediction from evidence sufficiency','hypothesis':'A separate factual sufficiency question can reject underdetermined cases while preserving confident outcomes in either direction.','questions':{'decision':DECISION,'sufficient':SUFFICIENT},'mode':'gated','gates':['sufficient']},
5:{'name':'Intensity-of-care specialist','hypothesis':'Explicit stability versus acute danger or failure of lower-intensity care may form a more reliable subgroup.','questions':{'assessment':choice('Assess ONLY appeals for inpatient, hospital, psychiatric residential, detoxification, partial hospital, or intensive outpatient care. Match the patient condition to the requested dates and intensity. Explicit clinical stability, no acute danger/withdrawal, preserved functioning and a feasible lower setting support sustaining denial. Persistent acute danger, severe impairment or documented failure of lower settings can support overturning. A diagnosis alone is insufficient. '+BOUNDARY,{'Upheld':'Requested intensive setting is unnecessary: explicit findings show stability or needs manageable at lower intensity.','Overturned':'Requested intensive setting is necessary: explicit findings show acute danger, severe impairment or failed lower care requiring this intensity.','Unresolved':'Not an intensity-of-care appeal, or clinical findings are insufficient, conflicting, or do not establish the required setting.'})},'mode':'ternary'},
6:{'name':'Treatment-alternatives specialist','hypothesis':'Documented treatment failure or intolerance plus a supported indication may identify reliable medication and therapy outcomes.','questions':{'assessment':choice('Assess appeals for a drug, procedure, device or therapy. Determine whether documented treatment response and indication evidence resolve the request. Failed alternatives alone do not justify investigational care; patient preference alone is not necessity. Exclude level-of-care disputes. '+BOUNDARY,{'Upheld':'Explicit findings show the requested intervention is ineffective, inappropriate for this indication, or unnecessary given an effective suitable alternative.','Overturned':'Persistent disease plus documented failure, intolerance or contraindication of relevant alternatives supports this specifically indicated intervention, or explicit indication evidence directly rebuts denial.','Unresolved':'Insufficient indication or treatment history, conflicting evidence, investigational uncertainty, or an out-of-scope level-of-care appeal.'})},'mode':'ternary'},
7:{'name':'Opposing evidence tests','hypothesis':'Require strong evidence for one outcome and weak evidence for the opposite, rather than generic self-confidence.','questions':{'upheld':binary('Do concrete patient facts affirmatively show that the requested care is NOT warranted at the requested intensity or indication?','Explicit clinical findings establish lack of need, inappropriate setting, effective alternative, or lack of demonstrated benefit.','Lack of need is not established by patient facts; denial language or omitted facts alone do not count.'),'overturned':binary('Do concrete patient facts affirmatively establish that the particular requested care IS warranted despite the denial?','Explicit patient need, failed relevant alternatives, severity or supported indication establishes this requested care.','Necessity is not established; a diagnosis, request, preference or unsubstantiated provider claim is insufficient.')},'mode':'opposing'},
8:{'name':'Concise evidence adjudication with unresolved option','hypothesis':'A short question centered on discriminating findings may outperform a lengthy automation rubric.','questions':{'assessment':choice('Which final independent review outcome is supported by the explicit clinical evidence? Stable patients without the need for requested intensity favor Upheld; documented unmet need and failure of suitable alternatives favor Overturned. Select Unresolved when the text does not distinguish the two. '+BOUNDARY,TRIAGE)},'mode':'ternary'},
9:{'name':'Outcome plus sufficiency and evidence provenance','hypothesis':'Separating concrete clinical evidence from insurer/provider assertions may remove confidently wrong decisions.','questions':{'decision':DECISION,'sufficient':SUFFICIENT,'grounded':GROUND},'mode':'gated','gates':['sufficient','grounded']},
10:{'name':'Agreement of outcome and clinical-necessity questions','hypothesis':'Agreement across differently framed questions may identify less ambiguous decisions. Errors may remain correlated.','questions':{'decision':DECISION,'necessity':choice('Does the patient-specific clinical evidence justify the particular requested care, including its indication, intensity and timing? '+BOUNDARY,{'Supported':'Patient-specific facts support this requested care as necessary and appropriate.','Unsupported':'Patient-specific facts show this requested care is not necessary or not supported.','Unresolved':'The supplied facts do not establish either conclusion.'})},'mode':'agreement'},
11:{'name':'Counterevidence-aware outcome prediction','hypothesis':'Explicitly distinguishing decisive clinical negatives from mere omission, and considering contrary evidence, may improve confidence ranking.','questions':{'decision':choice('Predict the independent review outcome. First distinguish documented negative findings from missing information. Then consider the strongest supplied clinical evidence against each outcome. Favor Upheld only when facts actually undermine the specific requested service, not merely because it was denied. Favor Overturned when facts establish its need or supported indication, not merely because disease is serious. If evidence conflicts, spread probabilities instead of guessing confidently. '+BOUNDARY,{'Upheld':'Specific requested care is not warranted or supported by the supplied clinical facts.','Overturned':'Specific requested care is warranted and supported by the supplied clinical facts.'})},'mode':'binary'},
12:{'name':'Newer JEV model with atomic evidence questions','hypothesis':'JEV 1.14 may improve the same outcome/sufficiency/provenance method used in pass9. This changes model version, not just criteria.','questions':{'decision':DECISION,'sufficient':SUFFICIENT,'grounded':GROUND},'mode':'gated','gates':['sufficient','grounded'],'model':'jev-1.14.0'}
}

def dump(p,o):p.write_text(json.dumps(o,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def norm(t):return ' '.join(t.lower().split())
def setup():
 path=HERE/'additional_trials_plan.json'
 if path.exists():return json.loads(path.read_text())
 source=ROOT/'jev/full-test-focused/jev_sample.json';allrows=json.loads(source.read_text())['sample'];old=[json.loads(s) for s in (ROOT/'jev/full-test-focused/jev_predictions.jsonl').read_text().splitlines()]
 prior=json.loads((HERE/'pass3/plan.json').read_text());seen={norm(r['text']) for r in old};seen|={norm(r['text']) for r in allrows if r['row_id'] in set(prior['evaluation_ids'])}
 fresh=[]
 for r in allrows:
  k=norm(r['text'])
  if k not in seen:fresh.append(r['row_id']);seen.add(k)
 random.Random(2026111201).shuffle(fresh);assert len(fresh)>=1800
 disc=set(json.loads((HERE/'splits.json').read_text())['discovery']);cal=[r['row_id'] for r in old if r['row_id'] in disc];random.Random(2026111202).shuffle(cal);cal=cal[:1000]
 plan={'created_at_utc':now(),'source_sha256':sha(source),'calibration_ids':cal,'development_evaluation_ids':fresh[:600],'confirmation_ids':fresh[600:1800],'unused_fresh_ids':fresh[1800:],'methods':CONFIGS,'target_accuracy':.90,'target_acceptance':.10,'calibration_selection':'Highest acceptance with >=93% point accuracy, >=60 accepted, and Wilson95 lower bound >=85%; fallback highest Wilson lower bound with >=60 accepted. Fallback is exploratory.','thresholds':[0,.5,.6,.65,.7,.75,.8,.825,.85,.875,.9,.925,.95,.975,.99],'confirmation_selection':'Among passes4-12 with >=90% development accuracy and >=60 accepted choose highest acceptance, tie by Wilson lower bound. If none meet target choose highest Wilson lower bound among >=60 accepted. Freeze selected prompt and calibration rule, evaluate on all 1200 reserved cases without retuning.','limits':'Nine methods share development evaluation cases: results are selection-biased when used to choose winner. Only pass13 confirmation is untouched by selection. Normalized exact-text separation verified; near-duplicate group independence not verified.'}
 dump(path,plan);return json.loads(path.read_text())

def score(answers,cfg):
 mode=cfg['mode'];pred=None
 if mode in ['binary','gated']:
  a=answers['decision'];pred=a['choice'];s=a['probabilities'][pred]
  for g in cfg.get('gates',[]):s=min(s,answers[g]['probabilities']['Yes'])
 elif mode=='ternary':
  a=answers['assessment'];pred=a['choice'] if a['choice']!='Unresolved' else None;s=a['probabilities'].get(pred,0)
 elif mode=='opposing':
  u=answers['upheld']['probabilities']['Yes'];o=answers['overturned']['probabilities']['Yes'];pred='Upheld' if u>o else 'Overturned';s=min(u,1-o) if u>o else min(o,1-u)
 elif mode=='agreement':
  a=answers['decision'];b=answers['necessity'];mapped={'Supported':'Overturned','Unsupported':'Upheld','Unresolved':None}[b['choice']]
  pred=a['choice'] if a['choice']==mapped else None;s=min(a['probabilities'][a['choice']],b['probabilities'][b['choice']]) if pred else 0
 return pred,float(s)

def metrics(rows,t):
 a=[r for r in rows if r.get('pred') in ['Upheld','Overturned'] and r.get('score',-1)>=t];k=sum(r['pred']==r['true'] for r in a);n=len(a)
 return {'total':len(rows),'accepted':n,'human_review':len(rows)-n,'acceptance':n/len(rows),'correct':k,'errors':n-k,'accuracy':k/n if n else None,'wilson95':wilson_interval(k,n),'api_failures':sum('error' in r for r in rows),'per_label':{label:{'accepted':sum(r['pred']==label for r in a),'correct':sum(r['pred']==label and r['true']==label for r in a)} for label in ['Upheld','Overturned']}}
def choose(sweep):
 viable=[m for m in sweep if m['accepted']>=60]
 good=[m for m in viable if m['accuracy']>=.93 and m['wilson95'][0]>=.85]
 if good:return max(good,key=lambda m:(m['accepted'],m['wilson95'][0])),True
 if viable:return max(viable,key=lambda m:(m['wilson95'][0],m['accepted'])),False
 return {'threshold':1.01},False

def batch(client,rows,cfg,path):
 # Checkpoint resume: successful and nonfatal failed rows are retained; labels never enter requests.
 records=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
 done={r['row_id'] for r in records};todo=[r for r in rows if r['row_id'] not in done]
 def query(r):
  start=time.monotonic()
  try:
   response=client.system_one(state=r['text'],questions=cfg['questions'],model=cfg.get('model','jev-1.13.0'))
   assert response.model==cfg.get('model','jev-1.13.0')
   answers={}
   for key,q in cfg['questions'].items():
    a=response.choices[key];probs=dict(a.probabilities)
    assert set(probs)==set(q['criteria']) and a.choice in probs
    assert all(math.isfinite(v) and 0<=v<=1 for v in probs.values()) and abs(sum(probs.values())-1)<.02
    answers[key]={'choice':a.choice,'probabilities':probs,'confidence':a.confidence}
   pred,s=score(answers,cfg)
   return {'row_id':r['row_id'],'true':r['decision'],'pred':pred,'score':s,'answers':answers,'model':response.model,'input_tokens':response.usage.input_tokens,'output_tokens':response.usage.output_tokens,'latency_seconds':time.monotonic()-start}
  except Exception as e:
   status=getattr(e,'status',None)
   if status in [400,401,402,403,404,422] or isinstance(e,(AssertionError,KeyError)):raise RuntimeError(f'Fatal {type(e).__name__}, HTTP {status}') from None
   return {'row_id':r['row_id'],'true':r['decision'],'error':type(e).__name__,'http_status':status}
 with path.open('a') as f:
  def save(r):records.append(r);f.write(json.dumps(r)+'\n');f.flush()
  if todo:save(query(todo.pop(0)))
  with ThreadPoolExecutor(max_workers=8) as pool:
   it=iter(todo);pending=set()
   def fill():
    while len(pending)<8:
     r=next(it,None)
     if r is None:break
     pending.add(pool.submit(query,r))
   fill()
   while pending:
    completed,pending=wait(pending,return_when=FIRST_COMPLETED)
    for future in completed:save(future.result())
    if len(records)%200<len(completed):print(path.parent.name,path.stem,len(records),'/',len(rows),flush=True)
    fill()
 assert len(records)==len(rows) and {r['row_id'] for r in records}=={r['row_id'] for r in rows}
 return records

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pass-number',type=int,choices=range(4,14),required=True);args=ap.parse_args();n=args.pass_number
 plan=setup();folder=HERE/f'pass{n}';folder.mkdir(exist_ok=True)
 assert not (folder/'metrics.json').exists(),'Completed trial cannot be overwritten'
 allrows={r['row_id']:r for r in json.loads((ROOT/'jev/full-test-focused/jev_sample.json').read_text())['sample']}
 selected=None
 if n==13:
  results=[(i,json.loads((HERE/f'pass{i}/metrics.json').read_text())['evaluation']) for i in range(4,13)]
  viable=[(i,m) for i,m in results if m['accepted']>=60];good=[(i,m) for i,m in viable if m['accuracy']>=.9]
  selected=max(good,key=lambda x:(x[1]['acceptance'],x[1]['wilson95'][0]))[0] if good else max(viable,key=lambda x:x[1]['wilson95'][0])[0] if viable else max(results,key=lambda x:x[1]['wilson95'][0] if x[1]['wilson95'] else 0)[0]
  cfg=plan['methods'][str(selected)];rule=json.loads((HERE/f'pass{selected}/rule.json').read_text());rule={**rule,'confirmation_frozen_at_utc':now(),'selected_pass':selected};eval_ids=plan['confirmation_ids']
 else:cfg=plan['methods'][str(n)];eval_ids=plan['development_evaluation_ids']
 frozen={'frozen_at_utc':now(),'configuration':cfg,'calibration_ids':plan['calibration_ids'],'evaluation_ids':eval_ids,'evaluation_role':'untouched confirmation' if n==13 else 'shared development evaluation','selected_pass':selected,'input':'text only','overall_plan_sha256':sha(HERE/'additional_trials_plan.json')}
 if (folder/'plan.json').exists():
  old=json.loads((folder/'plan.json').read_text());assert old['configuration']==cfg and old['evaluation_ids']==eval_ids
 else:dump(folder/'plan.json',frozen)
 dump(folder/'prompt.json',cfg['questions']);load_local_env()
 with TypeSafeClient(api_key=os.environ['TYPESAFE_API_KEY'],base_url='https://api.typesafe.ai',timeout=30,retry=RetryPolicy(max_retries=2)) as client:
  if n==13:
   cal=[json.loads(s) for s in (HERE/f'pass{selected}/calibration_predictions.jsonl').read_text().splitlines()];dump(folder/'rule.json',rule);t=rule['threshold'];cal_met=rule['calibration_target_met']
  else:
   cal=batch(client,[allrows[i] for i in plan['calibration_ids']],cfg,folder/'calibration_predictions.jsonl')
   sweep=[{'threshold':t,**metrics(cal,t)} for t in plan['thresholds']];dump(folder/'calibration_sweep.json',sweep);chosen,cal_met=choose(sweep);t=chosen['threshold']
   rule={'frozen_at_utc':now(),'threshold':t,'calibration_target_met':cal_met,'score_mode':cfg['mode'],'gates':cfg.get('gates',[]),'fallback':'HumanReview'};dump(folder/'rule.json',rule)
  print('Frozen',n,'threshold',t,'calibration',json.dumps(metrics(cal,t)),flush=True)
  test=batch(client,[allrows[i] for i in eval_ids],cfg,folder/'evaluation_predictions.jsonl')
 c=metrics(cal,t);m=metrics(test,t)
 result={'calibration':c,'evaluation':m,'configuration':cfg['name'],'selected_pass':selected,'calibration_target_met':cal_met,'new_case_requests':len(test)+(0 if n==13 else len(cal)),'successful_responses':sum('error' not in r for r in test)+(0 if n==13 else sum('error' not in r for r in cal)),'prompt_sha256':sha(folder/'prompt.json'),'rule_sha256':sha(folder/'rule.json'),'evaluation_role':frozen['evaluation_role']}
 dump(folder/'metrics.json',result)
 with (folder/'decisions.csv').open('w') as f:
  w=csv.writer(f,lineterminator='\n');w.writerow(['row_id','prediction','selection_score','route','true_label','correct'])
  for r in test:w.writerow([r['row_id'],r.get('pred'),r.get('score'),'automate' if r.get('pred') and r.get('score',-1)>=t else 'human',r['true'],r.get('pred')==r['true'] if r.get('pred') else ''])
 fmt=lambda v:'N/A' if v is None else f'{v:.2%}'
 table='\n'.join(f"| {label} | {v['total']} | {v['accepted']} | {fmt(v['acceptance'])} | {v['correct']}/{v['accepted']} | {fmt(v['accuracy'])} | {fmt(v['wilson95'][0])+'–'+fmt(v['wilson95'][1]) if v['wilson95'] else 'N/A'} |" for label,v in [('Calibration (selected threshold)',c),(frozen['evaluation_role'],m)])
 (folder/'README.md').write_text(f'''# Pass {n}: {cfg['name']}

## Target pattern and actual JEV question

{cfg['hypothesis']}

Exact JEV questions/criteria: [prompt.json](prompt.json). Model: `{cfg.get('model','jev-1.13.0')}`. Selection mode: `{cfg['mode']}`; gates: `{cfg.get('gates',[])}`. Actual calls send text only. No labels, original model predictions, row IDs or full_text are sent.

For gated methods, score is the minimum of selected outcome probability and each gate's Yes probability. Binary uses selected outcome probability. Ternary accepts only Upheld/Overturned and uses that probability. Opposing tests choose the stronger direction and score the minimum of its Yes probability and the other direction's No probability. Agreement requires mapped clinical-necessity and outcome choices to match, then takes the minimum of their selected probabilities. These combined scores are ranking statistics, not calibrated probabilities or statistical confidence intervals. Questions in one call are evaluated independently, not chained.

Frozen acceptance: valid predicted outcome and score >= **{t}**; otherwise human review. Calibration target met: **{cal_met}**. The selection rule prefers >=93% calibration accuracy with >=60 accepted and a Wilson lower bound >=85%; if impossible it uses the best lower bound with >=60 accepted, explicitly exploratory. Target: >=90% evaluation accuracy, ideally >=10% acceptance.

## Results

| Split | Cases | Accepted | Acceptance | Correct/accepted | Accepted accuracy | 95% Wilson interval |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{table}

Human review: **{m['human_review']}/{m['total']}**. Accepted errors: **{m['errors']}**. API failures: **{m['api_failures']}** (always human review).
Observed >=90% accuracy and >=10% acceptance: **{bool(m['accuracy'] is not None and m['accuracy']>=.9 and m['acceptance']>=.1)}**.
Evaluation 95% lower bound >=90%: **{bool(m['wilson95'] and m['wilson95'][0]>=.9)}**.
New case requests: {result['new_case_requests']}. Confirmation source method: {selected}.

## Evidence limits and reproduction

Passes4–12 reuse the same 600-case development evaluation batch. Their prompt configurations were frozen together before any new calls, but choosing a winner from their results creates selection bias. Pass13 alone uses 1,200 reserved cases excluded from all prior baseline predictions, pass3 evaluation and this development batch, and copies the chosen calibration rule unchanged. Calibration and evaluation have no normalized exact-text overlaps. Near-duplicate groups were not available for blocking. Intervals are approximate binomial Wilson intervals, not adjusted for multiple comparisons, and assume independent observations. The historical dataset contains clinical-review-style narrative and may not represent incoming appeals. No result establishes clinical causality or future production accuracy.

`plan.json` records IDs, prompt configuration and experiment role. `rule.json` is saved before evaluation. Predictions and `decisions.csv` retain every case for audit. `metrics.json` contains unrounded results. Run `.venv/bin/python jev/selective-automation/advanced_trials.py --pass-number {n}` from repository root; completed trials are protected, partial predictions checkpointed. A valid ignored `jev/.env` is required. For pass13, calibration responses are reused from the selected method, not newly called.
''')
 print('RESULT',n,json.dumps(m),flush=True)
if __name__=='__main__':main()
