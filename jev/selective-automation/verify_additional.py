"""Independent audit of new trials; checks saved artifacts without API calls."""
import csv,hashlib,json,math
from pathlib import Path
P=Path(__file__).resolve().parent
plan=json.loads((P/'additional_trials_plan.json').read_text())
source=json.loads((P.parents[1]/'jev/full-test-focused/jev_sample.json').read_text())
source={r['row_id']:r for r in source['sample']}
norm=lambda t:' '.join(t.lower().split())
sets=[set(plan[k]) for k in ['calibration_ids','development_evaluation_ids','confirmation_ids']]
texts=[{norm(source[i]['text']) for i in ids} for ids in sets]
for i,a in enumerate(sets):
 assert len(a)==len(texts[i])
 for j,b in enumerate(sets[i+1:],i+1):assert a.isdisjoint(b) and texts[i].isdisjoint(texts[j])
old_ids={r['row_id'] for r in map(json.loads,(P.parents[1]/'jev/full-test-focused/jev_predictions.jsonl').read_text().splitlines())}
prior=set(json.loads((P/'pass3/plan.json').read_text())['evaluation_ids'])
assert (sets[1]|sets[2]).isdisjoint(old_ids|prior)
count=0
for n in range(4,14):
 folder=P/f'pass{n}'
 if not (folder/'metrics.json').exists():continue
 m=json.loads((folder/'metrics.json').read_text());rule=json.loads((folder/'rule.json').read_text());saved=json.loads((folder/'plan.json').read_text())
 assert saved['configuration']==plan['methods'][str(m['selected_pass'] if n==13 else n)]
 assert hashlib.sha256((folder/'prompt.json').read_bytes()).hexdigest()==m['prompt_sha256']
 assert hashlib.sha256((folder/'rule.json').read_bytes()).hexdigest()==m['rule_sha256']
 rows=list(map(json.loads,(folder/'evaluation_predictions.jsonl').read_text().splitlines()))
 expected=sets[2] if n==13 else sets[1]
 assert len(rows)==len(expected) and {r['row_id'] for r in rows}==expected
 for r in rows:assert r['true']==source[r['row_id']]['decision']
 a=[r for r in rows if r.get('pred') in ['Upheld','Overturned'] and r['score']>=rule['threshold']]
 correct=sum(r['pred']==r['true'] for r in a);v=m['evaluation']
 assert len(a)==v['accepted'] and correct==v['correct'] and v['human_review']==len(rows)-len(a)
 assert v['accuracy']==(correct/len(a) if a else None)
 assert v['acceptance']==len(a)/len(rows)
 decisions=list(csv.DictReader((folder/'decisions.csv').open()))
 assert {int(r['row_id']) for r in decisions if r['route']=='automate'}=={r['row_id'] for r in a}
 if n==13:
  original=json.loads((P/f"pass{m['selected_pass']}/rule.json").read_text())
  assert all(rule[k]==v for k,v in original.items())
 count+=1
print(f'PASS: {count} completed trials audited; exact-text/ID separation, truth joins, frozen configurations and hashes, all acceptance/accuracy counts, CSV decisions, and confirmation rule integrity.')
