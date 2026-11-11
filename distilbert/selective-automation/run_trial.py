"""Final authorized trial: calibrate cached DistilBERT / JEV specialist hybrid, then confirm."""
import csv,hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];JEV=ROOT/'jev/selective-automation'
sys.path.insert(0,str(JEV))
from advanced_trials import batch,load_local_env,TypeSafeClient,RetryPolicy,wilson_interval
import os

def dump(p,o):p.write_text(json.dumps(o,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def route(bert,specialist,rule):
 bpred=bert['pred'];bp=bert['probability'];jpred=specialist.get('pred') if specialist else None;jp=specialist.get('score',0) if specialist else 0
 ba=bp>=rule['distilbert_threshold'];ja=jpred in ['Upheld','Overturned'] and jp>=rule['jev_threshold']
 if rule['mode']=='distilbert':return (bpred,'DistilBERT') if ba else (None,'Human')
 if rule['mode']=='union_defer_conflicts' and ba and ja and bpred!=jpred:return None,'Human'
 if ja:return jpred,'JEV specialist'
 return (bpred,'DistilBERT') if ba else (None,'Human')
def summarize(rows):
 a=[r for r in rows if r['route']=='automate'];k=sum(r['prediction']==r['true'] for r in a);n=len(a)
 return {'total':len(rows),'accepted':n,'acceptance':n/len(rows),'human_review':len(rows)-n,'correct':k,'errors':n-k,'accuracy':k/n if n else None,'wilson95':wilson_interval(k,n),'per_source':{s:{'accepted':sum(r['prediction_source']==s for r in a),'correct':sum(r['prediction_source']==s and r['prediction']==r['true'] for r in a)} for s in ['DistilBERT','JEV specialist']}}
def apply(ids,bert,jev,rule):
 rows=[]
 for i in ids:
  pred,source=route(bert[i],jev.get(i),rule)
  rows.append({'row_id':i,'distilbert_prediction':bert[i]['pred'],'distilbert_probability':bert[i]['probability'],'jev_prediction':jev.get(i,{}).get('pred'),'jev_score':jev.get(i,{}).get('score'),'prediction':pred,'prediction_source':source,'route':'automate' if pred else 'human','true':bert[i]['true'],'has_shortcut':bert[i]['has_shortcut']})
 return rows

def main():
 folder=HERE/'pass13';folder.mkdir(exist_ok=False)
 original=json.loads((JEV/'additional_trials_plan.json').read_text());manifest=json.loads((ROOT/'jev/full-test-focused/jev_sample.json').read_text())['sample'];cases={r['row_id']:r for r in manifest}
 source=ROOT/'week3-results/test_predictions.csv';raw=list(csv.DictReader(source.open()));assert len(raw)==len(manifest)
 bert={}
 for r in raw:
  i=int(r['row_id']);idx=int(r['orig_idx']);assert manifest[idx]['row_id']==i and cases[i]['decision']==r['decision']
  pu=float(r['val_selected_p_upheld']);po=float(r['val_selected_p_overturned']);pred=r['val_selected_pred'];assert pred in ['Upheld','Overturned'] and abs(pu+po-1)<.001
  assert pred==('Upheld' if pu>po else 'Overturned')
  bert[i]={'pred':pred,'probability':max(pu,po),'true':r['decision'],'has_shortcut':r['has_shortcut']=='True'}
 cal_ids=original['calibration_ids'];test_ids=original['confirmation_ids'];assert len(cal_ids)==1000 and len(test_ids)==1200 and set(cal_ids).isdisjoint(test_ids)
 cfg=original['methods']['5'];jt=json.loads((JEV/'pass5/rule.json').read_text())['threshold'];caljev={r['row_id']:r for r in map(json.loads,(JEV/'pass5/calibration_predictions.jsonl').read_text().splitlines())};assert set(caljev)==set(cal_ids)
 thresholds=[.5,.6,.7,.75,.8,.825,.85,.875,.9,.925,.95,.96,.97,.975,.98,.99]
 plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'user_amendment':'Target raised to >=25% acceptance and >=90% accuracy; allow DistilBERT as predictor. Final trial moved to separate folder; remains the tenth additional trial.','calibration_ids':cal_ids,'evaluation_ids':test_ids,'source_predictions_sha256':sha(source),'dataset_manifest_sha256':sha(ROOT/'jev/full-test-focused/jev_sample.json'),'distilbert_model':'utcompliance1/distilbert-imr-val-selected','distilbert_provenance':'Saved week3 predictions. Checkpoint selected by validation loss using only source train/validation. Test was not used for checkpoint selection. Historical aggregate test statistics are already public.','jev_configuration':cfg,'jev_threshold':jt,'candidate_modes':['distilbert','specialist_union','union_defer_conflicts'],'thresholds':thresholds,'selection':'Highest calibration acceptance with >=92% accuracy, >=250 accepted, Wilson95 lower >=88%. If none, highest Wilson lower among >=60 accepted. Calibration chooses method and threshold; test never changes either. Ties favor first enumerated.','evaluation_role':'1200 reserved cases, untouched by current method/threshold selection; cached DistilBERT predictions preexist. Retrospective confirmation, not a new prospective trial.','ground_truth':'Recorded dataset decision, never a model prediction.'}
 dump(folder/'plan.json',plan);dump(folder/'jev_prompt.json',cfg['questions'])
 sweep=[]
 for mode in plan['candidate_modes']:
  for t in thresholds:
   rule={'mode':mode,'distilbert_threshold':t,'jev_threshold':jt};m=summarize(apply(cal_ids,bert,caljev,rule));sweep.append({'rule':rule,**m})
 dump(folder/'calibration_sweep.json',sweep)
 good=[m for m in sweep if m['accepted']>=250 and m['accuracy']>=.92 and m['wilson95'][0]>=.88]
 chosen=max(good,key=lambda m:m['accepted']) if good else max([m for m in sweep if m['accepted']>=60],key=lambda m:m['wilson95'][0])
 rule={**chosen['rule'],'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'calibration_target_met':bool(good)};dump(folder/'rule.json',rule)
 print('FROZEN DISTILBERT/HYBRID',json.dumps(chosen),flush=True)
 testjev={};newcalls=0
 if rule['mode']!='distilbert':
  load_local_env()
  with TypeSafeClient(api_key=os.environ['TYPESAFE_API_KEY'],base_url='https://api.typesafe.ai',timeout=30,retry=RetryPolicy(max_retries=2)) as client:
   predicted=batch(client,[cases[i] for i in test_ids],cfg,folder/'jev_evaluation_predictions.jsonl')
  testjev={r['row_id']:r for r in predicted};newcalls=len(test_ids)
 cal=apply(cal_ids,bert,caljev,rule);test=apply(test_ids,bert,testjev,rule);cm=summarize(cal);tm=summarize(test)
 # Literal outcome shortcut exclusion is a predeclared sensitivity metric, never used for routing or selection.
 clean=summarize([r for r in test if not r['has_shortcut']])
 dump(folder/'metrics.json',{'calibration':cm,'evaluation':tm,'evaluation_without_literal_outcome_shortcuts':clean,'new_jev_calls':newcalls,'new_distilbert_calls':0,'cached_distilbert_rows_scored':2200,'rule':rule,'rule_sha256':sha(folder/'rule.json')})
 for name,rows in [('calibration',cal),('evaluation',test)]:
  with (folder/f'{name}_decisions.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 fmt=lambda x:'N/A' if x is None else f'{x:.2%}'
 table='\n'.join(f"| {name} | {m['total']} | {m['accepted']} | {fmt(m['acceptance'])} | {m['correct']}/{m['accepted']} | {fmt(m['accuracy'])} | {fmt(m['wilson95'][0])+'–'+fmt(m['wilson95'][1]) if m['wilson95'] else 'N/A'} |" for name,m in [('Calibration (selected)',cm),('Reserved confirmation',tm),('Confirmation excluding outcome shortcuts',clean)])
 (folder/'README.md').write_text(f'''# Pass 13: DistilBERT / JEV specialist hybrid

This is the tenth additional authorized trial. The updated goal is **>=25% acceptance and >=90% accuracy**. DistilBERT predicts outcomes; ground truth remains the recorded dataset decision. The validation-selected DistilBERT was trained/selected on source train/validation, not source test. Cached test predictions are reused; no retraining or fresh DistilBERT inference occurred.

## Pattern and frozen acceptance criteria

Calibration compares three methods: DistilBERT confidence alone; a union with JEV's intensity-of-care specialist; and that union with disagreement between two accepted predictions sent to human review. The JEV question distinguishes documented stability/lower-intensity suitability from acute danger or failed lower care, and marks out-of-scope or unresolved cases Unresolved. Its exact wording is in jev_prompt.json, unchanged from JEV pass5.

Selected mode: **{rule['mode']}**. DistilBERT selected-class probability threshold: **{rule['distilbert_threshold']}**. JEV specialist selected-class probability threshold: **{jt}**. In specialist_union, an eligible JEV specialist takes precedence; otherwise accept eligible DistilBERT. In union_defer_conflicts, two eligible but disagreeing predictions go to review. All remaining cases go to review. DistilBERT-only mode ignores JEV. Scores are not statistical confidence intervals.

A 1,000-case calibration set selects the method and threshold by maximum acceptance subject to >=92% accuracy, >=250 accepted, and 95% Wilson lower bound >=88%. If no method qualifies, fallback is exploratory. Calibration target met: **{bool(good)}**. Calibration method selection is optimistic; the 1,200-case reserved confirmation never changes the rule.

## Results

| Split | Tasks | Accepted | Acceptance | Correct/accepted | Accuracy | 95% Wilson interval |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{table}

Human review on confirmation: **{tm['human_review']}/1200**. Accepted errors: **{tm['errors']}**. Both updated targets met: **{tm['acceptance']>=.25 and tm['accuracy'] is not None and tm['accuracy']>=.90}**. Accuracy 95% lower bound >=90%: **{bool(tm['wilson95'] and tm['wilson95'][0]>=.9)}**.

## Provenance and limits

All 2,200 DistilBERT source rows are joined by row_id; dataset labels and original row order are checked against the JEV dataset manifest. Source and policy hashes are preserved. DistilBERT source: week3-results/test_predictions.csv, model utcompliance1/distilbert-imr-val-selected (epoch2 checkpoint selected by validation loss). New JEV calls: {newcalls}. New DistilBERT calls: 0.

Confirmation IDs are the 1,200 cases reserved before the new JEV comparison trials. These outcomes were not used for current prompt/threshold selection, but historical DistilBERT aggregate test results existed; this is retrospective confirmation. Normalized exact texts are separated from calibration and development. Near-duplicate group independence and historical checkpoint training exclusion are inherited from repository documentation, not freshly audited by retraining. Wilson intervals assume independent cases. Clinical-review-style narrative and dataset shift limit transfer to incoming appeals. Literal outcome shortcut exclusion is reported separately and does not affect selection. No clinical causal claim or future accuracy guarantee is made.

`plan.json`, `rule.json`, `calibration_sweep.json`, per-case CSVs and `metrics.json` preserve the experiment. Reproduce with `.venv/bin/python distilbert/selective-automation/run_trial.py` in a fresh output directory. Completed outputs are protected. The candidate search is one calibration procedure, not 48 additional held-out tests.
''')
 print('FINAL CONFIRMATION',json.dumps(tm),flush=True)
if __name__=='__main__':main()
