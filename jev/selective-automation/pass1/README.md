# Pass 1: Global JEV confidence threshold

Target: at least 90% accuracy among accepted tasks; all other tasks route to a human.

## Frozen acceptance pattern

(confidence >= 0.94; predicted label = any; text contains '')

Thresholds use the native JEV `confidence`, not an accuracy confidence interval. Text is lowercased, tokenized to letters, and joined by spaces before phrase matching. Only text, predicted label, and confidence enter routing; true outcomes are used solely for scoring.

## Results

| Split | Tasks | Accepted | Acceptance | Correct/accepted | Accepted accuracy | 95% Wilson interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Discovery (selection-biased) | 4115 | 502 | 12.20% | 448/502 | 89.24% | 86.23%–91.66% |
| Calibration | 1000 | 128 | 12.80% | 116/128 | 90.62% | 84.33%–94.56% |
| Evaluation | 600 | 60 | 10.00% | 53/60 | 88.33% | 77.82%–94.23% |

Evaluation sends **540/600** tasks to human review. Accepted errors: **7**.
Observed 90% target: **NOT MET**.
Evaluation lower 95% two-sided accuracy bound >=90%: **NO**.
Calibration lower 95% two-sided accuracy bound >=90%: **NO**.

## Method and limits

This is a new routing evaluation of **600 existing JEV predictions**, not 600 fresh API calls. Model `jev-1.13.0`, focused prompt, cleaned IMR test data. Predictions are reused unchanged. No outcome is flipped. The source is an interrupted 6,915-case prefix, not a representative new production sample. Original model-development exclusions are inherited. Normalized exact duplicates are removed before a deterministic random split; near-duplicate group metadata is unavailable here, so residual dependence remains possible.

Discovery selects the largest subset whose 95% Wilson lower bound reaches 90%; if none qualifies, it selects the highest lower bound subject to minimum support (50 for passes 1/2; 70 for pass 3). Pass 2 applies this separately per predicted label and omits classes below 90% discovery point accuracy. Pass 3 searches frequent words and word pairs plus confidence and predicted label. Discovery intervals are selection-biased and are not certification. Calibration never changes the selected rule. The rule file is written before scoring calibration/evaluation. Each pass has its own disjoint 600-case evaluation batch; discovery and calibration are shared. Previously published aggregate statistics existed, so this is retrospective validation, not a pristine prospective trial.

Intervals are approximate binomial Wilson intervals, assume independent cases, and are not multiplicity-adjusted across passes. These results do not certify future accuracy or measure human-review accuracy. Treat any rule failing calibration as exploratory. A deployment decision needs a separate fresh, group-disjoint validation batch.

## Reproduce / audit

From repository root: `.venv/bin/python jev/selective-automation/run.py --pass-number 1` (requires the local source cache and a nonexistent pass folder). `../splits.json` records every split ID and source hash. `rule.json`, `metrics.json`, `decisions.csv`, and `discovery_candidates.csv` preserve the full selection and evaluation trail. The script refuses to overwrite a pass.
