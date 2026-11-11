"""Independent reconstruction of the hybrid's held-out decisions."""
import csv,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1];F=P/'pass13';J=R/'jev/selective-automation'
plan=json.loads((F/'plan.json').read_text());rule=json.loads((F/'rule.json').read_text());m=json.loads((F/'metrics.json').read_text())
assert hashlib.sha256((R/'week3-results/test_predictions.csv').read_bytes()).hexdigest()==plan['source_predictions_sha256']
assert hashlib.sha256((F/'rule.json').read_bytes()).hexdigest()==m['rule_sha256']
bert={int(r['row_id']):r for r in csv.DictReader((R/'week3-results/test_predictions.csv').open())}
manifest={r['row_id']:r for r in json.loads((R/'jev/full-test-focused/jev_sample.json').read_text())['sample']}
base=json.loads((J/'additional_trials_plan.json').read_text());assert plan['evaluation_ids']==base['confirmation_ids']
cal=set(plan['calibration_ids']);test=set(plan['evaluation_ids']);dev=set(base['development_evaluation_ids']);assert cal.isdisjoint(test) and dev.isdisjoint(test)
norm=lambda s:' '.join(s.lower().split())
assert {norm(manifest[i]['text']) for i in test}.isdisjoint({norm(manifest[i]['text']) for i in cal|dev})
actual={int(r['row_id']):r for r in csv.DictReader((F/'evaluation_decisions.csv').open())}
assert len(actual)==1200 and set(actual)==test
jev={r['row_id']:r for r in map(json.loads,(F/'jev_evaluation_predictions.jsonl').read_text().splitlines())};assert set(jev)==test
accepted=correct=clean_n=clean_accepted=clean_correct=0
for i in test:
 b=bert[i];j=jev[i];assert b['decision']==j['true']==manifest[i]['decision']==actual[i]['true']
 a=j['answers']['assessment'];jp=a['probabilities'].get(a['choice'],0);jpred=a['choice'];assert j['model']=='jev-1.13.0'
 bp=max(float(b['val_selected_p_upheld']),float(b['val_selected_p_overturned']));bpred=b['val_selected_pred']
 # Independently implement the actual frozen union policy, without importing its router.
 assert rule['mode']=='specialist_union'
 pred=jpred if jpred in ['Upheld','Overturned'] and jp>=rule['jev_threshold'] else bpred if bp>=rule['distilbert_threshold'] else None
 assert actual[i]['prediction']==(pred or '') and actual[i]['route']==('automate' if pred else 'human')
 accepted+=pred is not None;correct+=pred is not None and pred==b['decision']
 if b['has_shortcut']=='False':
  clean_n+=1;clean_accepted+=pred is not None;clean_correct+=pred is not None and pred==b['decision']
assert accepted==m['evaluation']['accepted'] and correct==m['evaluation']['correct']
assert clean_n==m['evaluation_without_literal_outcome_shortcuts']['total'] and clean_accepted==m['evaluation_without_literal_outcome_shortcuts']['accepted'] and clean_correct==m['evaluation_without_literal_outcome_shortcuts']['correct']
sweep=json.loads((F/'calibration_sweep.json').read_text());eligible=[x for x in sweep if x['accepted']>=250 and x['accuracy']>=.92 and x['wilson95'][0]>=.88];winner=max(eligible,key=lambda x:x['accepted'])
assert all(rule[k]==v for k,v in winner['rule'].items())
print(f'PASS: all 1200 hybrid routes independently reconstructed; {accepted} accepted, {correct} correct; source/hash/split/label joins and calibration-only policy selection verified.')
