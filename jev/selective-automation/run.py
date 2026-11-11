"""Reproducible cached-JEV selective routing; no labels enter route()."""
import argparse, csv, hashlib, json, math, random, re
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'jev/full-test-focused/jev_predictions.jsonl'

def sha(b): return hashlib.sha256(b).hexdigest()
def dump(p, obj): p.write_text(json.dumps(obj, indent=2)+'\n')
def interval(k,n,z=1.959963984540054):
    if not n: return [0.,1.]
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h,c+h]
@lru_cache(maxsize=10000)
def normalized(text): return " ".join(re.findall(r"[a-z]+", text.lower()))

def route(r, rule):
    if r.get('pred') not in ['Upheld','Overturned'] or not isinstance(r.get('confidence'),(float,int)): return False
    return any(r['confidence'] >= t['threshold'] and (t['label']=='any' or r['pred']==t['label']) and (not t.get('phrase') or t['phrase'] in normalized(r['text'])) for t in rule['terms'])
def stats(rows,rule):
    a=[r for r in rows if route(r,rule)]; k=sum(r['pred']==r['true'] for r in a);n=len(a)
    return dict(total=len(rows),accepted=n,human_review=len(rows)-n,acceptance=n/len(rows),correct=k,errors=n-k,accuracy=k/n if n else None,wilson95=interval(k,n),wilson_one_sided95_lower=interval(k,n,1.6448536269514722)[0],per_predicted_label={label:{'accepted':sum(r['pred']==label for r in a),'correct':sum(r['pred']==label and r['pred']==r['true'] for r in a)} for label in ['Upheld','Overturned']})
def load():
    rows=[json.loads(s) for s in SOURCE.read_text().splitlines()]
    unique={}
    for r in rows:
        key=sha(' '.join(r['text'].lower().split()).encode())
        if key not in unique: unique[key]={**r,'text_sha256':key}
    rows=list(unique.values());random.Random(20261111).shuffle(rows)
    assert len(rows)>=3800
    return rows[2800:],rows[1800:2800],[rows[i*600:(i+1)*600] for i in range(3)],len(rows)
def best(candidates, rows, minimum=50):
    scored=[(stats(rows,c),c) for c in candidates]
    valid=[x for x in scored if x[0]['accepted']>=minimum]
    good=[x for x in valid if x[0]['wilson95'][0]>=.90]
    pool=good or valid
    chosen=max(pool,key=lambda x:(x[0]['accepted'],x[0]['wilson95'][0])) if good else max(pool,key=lambda x:(x[0]['wilson95'][0],x[0]['accepted']))
    return chosen[1],scored

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pass-number',type=int,choices=[1,2,3],required=True);args=ap.parse_args();p=args.pass_number
    folder=OUT/f'pass{p}';folder.mkdir(exist_ok=False)
    discovery,cal,batches,n=load()
    split={'seed':20261111,'source_sha256':sha(SOURCE.read_bytes()),'source_records':6915,'unique_normalized_texts':n,'discovery':[r['row_id'] for r in discovery],'calibration':[r['row_id'] for r in cal],'evaluation':[[r['row_id'] for r in b] for b in batches]}
    if not (OUT/'splits.json').exists(): dump(OUT/'splits.json',split)
    else: assert json.loads((OUT/'splits.json').read_text())==split
    thresholds=[.80,.85,.90,.92,.94,.95,.96,.97,.98,.99,1.]
    def candidate(t,label='any',phrase=''):return {'terms':[dict(threshold=t,label=label,phrase=phrase)]}
    if p==1:
        title='Global JEV confidence threshold';rule,scores=best([candidate(t) for t in thresholds],discovery)
    elif p==2:
        title='Separate confidence thresholds for predicted outcomes';rule={'terms':[]};scores=[]
        for label in ['Upheld','Overturned']:
            chosen,s=best([candidate(t,label) for t in thresholds],discovery);scores+=s
            # Do not include a class whose discovery precision fails even the point target.
            if stats(discovery,chosen)['accuracy']>=.90:rule['terms']+=chosen['terms']
    else:
        title='Text-pattern groups combined with JEV confidence'
        from collections import Counter
        counts=Counter()
        for r in discovery:
            words=re.findall(r'[a-z]+',r['text'].lower());counts.update(set(words)|set(' '.join(words[i:i+2]) for i in range(len(words)-1)))
        phrases=[w for w,c in counts.items() if c>=80 and len(w)>=5]
        candidates=[candidate(t,label,w) for w in sorted(phrases) for label in ['Upheld','Overturned'] for t in [.8,.9,.94,.96,.98]]
        rule,scores=best(candidates,discovery,minimum=70)
    dump(folder/'rule.json',{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'title':title,**rule})
    # Only discovery chooses the policy; calibration and test never alter it.
    with (folder/'discovery_candidates.csv').open('w') as f:
        w=csv.writer(f,lineterminator="\n");w.writerow(['rule','accepted','accuracy','wilson95_lower'])
        for m,c in scores:w.writerow([json.dumps(c),m['accepted'],m['accuracy'],m['wilson95'][0]])
    d=stats(discovery,rule);c=stats(cal,rule);test=batches[p-1];m=stats(test,rule)
    dump(folder/'metrics.json',{'discovery':d,'calibration':c,'evaluation':m,'rule_sha256':sha((folder/'rule.json').read_bytes()),'new_api_calls':0})
    with (folder/'decisions.csv').open('w') as f:
        w=csv.writer(f,lineterminator="\n");w.writerow(['row_id','text_sha256','jev_prediction','jev_confidence','selected_probability','route','true_label','correct'])
        for r in test:w.writerow([r['row_id'],r['text_sha256'],r['pred'],r['confidence'],r['probabilities'][r['pred']],'automate' if route(r,rule) else 'human',r['true'],r['pred']==r['true']])
    fmt=lambda x:f'{100*x:.2f}%'
    ruletext=' OR '.join(f"(confidence >= {t['threshold']}; predicted label = {t['label']}; text contains {t['phrase']!r})" for t in rule['terms']) or 'Accept none'
    table='\n'.join(f"| {name} | {v['total']} | {v['accepted']} | {fmt(v['acceptance'])} | {v['correct']}/{v['accepted']} | {fmt(v['accuracy']) if v['accuracy'] is not None else 'N/A'} | {fmt(v['wilson95'][0])}–{fmt(v['wilson95'][1])} |" for name,v in [('Discovery (selection-biased)',d),('Calibration',c),('Evaluation',m)])
    (folder/'README.md').write_text(f'''# Pass {p}: {title}

Target: at least 90% accuracy among accepted tasks; all other tasks route to a human.

## Frozen acceptance pattern

{ruletext}

Thresholds use the native JEV `confidence`, not an accuracy confidence interval. Text is lowercased, tokenized to letters, and joined by spaces before phrase matching. Only text, predicted label, and confidence enter routing; true outcomes are used solely for scoring.

## Results

| Split | Tasks | Accepted | Acceptance | Correct/accepted | Accepted accuracy | 95% Wilson interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

Evaluation sends **{m['human_review']}/{m['total']}** tasks to human review. Accepted errors: **{m['errors']}**.
Observed 90% target: **{'MET' if m['accuracy'] is not None and m['accuracy']>=.9 else 'NOT MET'}**.
Evaluation lower 95% two-sided accuracy bound >=90%: **{'YES' if m['wilson95'][0]>=.9 else 'NO'}**.
Calibration lower 95% two-sided accuracy bound >=90%: **{'YES' if c['wilson95'][0]>=.9 else 'NO'}**.

## Method and limits

This is a new routing evaluation of **600 existing JEV predictions**, not 600 fresh API calls. Model `jev-1.13.0`, focused prompt, cleaned IMR test data. Predictions are reused unchanged. No outcome is flipped. The source is an interrupted 6,915-case prefix, not a representative new production sample. Original model-development exclusions are inherited. Normalized exact duplicates are removed before a deterministic random split; near-duplicate group metadata is unavailable here, so residual dependence remains possible.

Discovery selects the largest subset whose 95% Wilson lower bound reaches 90%; if none qualifies, it selects the highest lower bound subject to minimum support (50 for passes 1/2; 70 for pass 3). Pass 2 applies this separately per predicted label and omits classes below 90% discovery point accuracy. Pass 3 searches frequent words and word pairs plus confidence and predicted label. Discovery intervals are selection-biased and are not certification. Calibration never changes the selected rule. The rule file is written before scoring calibration/evaluation. Each pass has its own disjoint 600-case evaluation batch; discovery and calibration are shared. Previously published aggregate statistics existed, so this is retrospective validation, not a pristine prospective trial.

Intervals are approximate binomial Wilson intervals, assume independent cases, and are not multiplicity-adjusted across passes. These results do not certify future accuracy or measure human-review accuracy. Treat any rule failing calibration as exploratory. A deployment decision needs a separate fresh, group-disjoint validation batch.

## Reproduce / audit

From repository root: `.venv/bin/python jev/selective-automation/run.py --pass-number {p}` (requires the local source cache and a nonexistent pass folder). `../splits.json` records every split ID and source hash. `rule.json`, `metrics.json`, `decisions.csv`, and `discovery_candidates.csv` preserve the full selection and evaluation trail. The script refuses to overwrite a pass.
''')
    print(json.dumps({'pass':p,'rule':rule,'calibration':c,'evaluation':m},indent=2))
if __name__=='__main__': main()
