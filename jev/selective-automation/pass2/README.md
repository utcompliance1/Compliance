# Pass 2: Separate confidence thresholds for predicted outcomes

Target: at least 90% accuracy among accepted tasks; all other tasks route to a human.

## Frozen acceptance pattern

(confidence >= 0.94; predicted label = Overturned; text contains '')

Thresholds use the native JEV `confidence`, not an accuracy confidence interval. Text is lowercased, tokenized to letters, and joined by spaces before phrase matching. Only text, predicted label, and confidence enter routing; true outcomes are used solely for scoring.

## Results

| Split | Tasks | Accepted | Acceptance | Correct/accepted | Accepted accuracy | 95% Wilson interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Discovery (selection-biased) | 4115 | 146 | 3.55% | 136/146 | 93.15% | 87.85%–96.24% |
| Calibration | 1000 | 35 | 3.50% | 32/35 | 91.43% | 77.62%–97.04% |
| Evaluation | 600 | 23 | 3.83% | 19/23 | 82.61% | 62.86%–93.02% |

Evaluation sends **577/600** tasks to human review. Accepted errors: **4**.
Observed 90% target: **NOT MET**.
Evaluation lower 95% two-sided accuracy bound >=90%: **NO**.
Calibration lower 95% two-sided accuracy bound >=90%: **NO**.

## Method and limits

This is a new routing evaluation of **600 existing JEV predictions**, not 600 fresh API calls. Model `jev-1.13.0`, focused prompt, cleaned IMR test data. Predictions are reused unchanged. No outcome is flipped. The source is an interrupted 6,915-case prefix, not a representative new production sample. Original model-development exclusions are inherited. Normalized exact duplicates are removed before a deterministic random split; near-duplicate group metadata is unavailable here, so residual dependence remains possible.

Discovery selects the largest subset whose 95% Wilson lower bound reaches 90%; if none qualifies, it selects the highest lower bound subject to minimum support (50 for passes 1/2; 70 for pass 3). Pass 2 applies this separately per predicted label and omits classes below 90% discovery point accuracy. Pass 3 searches frequent words and word pairs plus confidence and predicted label. Discovery intervals are selection-biased and are not certification. Calibration never changes the selected rule. The rule file is written before scoring calibration/evaluation. Each pass has its own disjoint 600-case evaluation batch; discovery and calibration are shared. Previously published aggregate statistics existed, so this is retrospective validation, not a pristine prospective trial.

Intervals are approximate binomial Wilson intervals, assume independent cases, and are not multiplicity-adjusted across passes. These results do not certify future accuracy or measure human-review accuracy. Treat any rule failing calibration as exploratory. A deployment decision needs a separate fresh, group-disjoint validation batch.

## Reproduce / audit

From repository root: `.venv/bin/python jev/selective-automation/run.py --pass-number 2` (requires the local source cache and a nonexistent pass folder). `../splits.json` records every split ID and source hash. `rule.json`, `metrics.json`, `decisions.csv`, and `discovery_candidates.csv` preserve the full selection and evaluation trail. The script refuses to overwrite a pass.
