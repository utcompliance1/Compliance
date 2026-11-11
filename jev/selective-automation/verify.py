"""Audit saved results independently; makes no API calls."""
import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent
split=json.loads((P/'splits.json').read_text())
parts=[set(split['discovery']),set(split['calibration'])]+[set(x) for x in split['evaluation']]
for i,a in enumerate(parts):
 for b in parts[i+1:]:assert a.isdisjoint(b)
for number in [1,2,3]:
 p=P/f'pass{number}';rows=list(csv.DictReader((p/'decisions.csv').open()))
 m=json.loads((p/'metrics.json').read_text())['evaluation']
 accepted=[r for r in rows if r['route']=='automate']
 correct=sum(r['correct']=='True' for r in accepted)
 assert len(rows)==600 and len({r['row_id'] for r in rows})==600
 assert len(accepted)==m['accepted'] and correct==m['correct']
 assert m['human_review']==600-len(accepted) and m['acceptance']==len(accepted)/600
 assert m['accuracy']==correct/len(accepted)
plan=json.loads((P/'pass3/plan.json').read_text())
assert set(plan['calibration_ids']).isdisjoint(plan['evaluation_ids'])
assert set(plan['evaluation_ids']).isdisjoint(set.union(*parts))
for name in ['calibration','evaluation']:
 rows=[json.loads(s) for s in (P/f'pass3/{name}_predictions.jsonl').read_text().splitlines()]
 assert len(rows)==600 and {r['row_id'] for r in rows}==set(plan[f'{name}_ids'])
 assert all(r['model']=='jev-1.13.0' for r in rows)
 for r in rows:
  assert 0<=r['confidence']<=1 and abs(sum(r['probabilities'].values())-1)<.011
print('Verified: 3 x 600 unique evaluation cases; disjoint splits; per-case counts, accuracy, routing, model and probabilities.')
