# Pass 9: Outcome plus sufficiency and evidence provenance

## Target pattern and actual JEV question

Separating concrete clinical evidence from insurer/provider assertions may remove confidently wrong decisions.

Exact JEV questions/criteria: [prompt.json](prompt.json). Model: `jev-1.13.0`. Selection mode: `gated`; gates: `['sufficient', 'grounded']`. Actual calls send text only. No labels, original model predictions, row IDs or full_text are sent.

For gated methods, score is the minimum of selected outcome probability and each gate's Yes probability. Binary uses selected outcome probability. Ternary accepts only Upheld/Overturned and uses that probability. Opposing tests choose the stronger direction and score the minimum of its Yes probability and the other direction's No probability. Agreement requires mapped clinical-necessity and outcome choices to match, then takes the minimum of their selected probabilities. These combined scores are ranking statistics, not calibrated probabilities or statistical confidence intervals. Questions in one call are evaluated independently, not chained.

Frozen acceptance: valid predicted outcome and score >= **0.8**; otherwise human review. Calibration target met: **False**. The selection rule prefers >=93% calibration accuracy with >=60 accepted and a Wilson lower bound >=85%; if impossible it uses the best lower bound with >=60 accepted, explicitly exploratory. Target: >=90% evaluation accuracy, ideally >=10% acceptance.

## Results

| Split | Cases | Accepted | Acceptance | Correct/accepted | Accepted accuracy | 95% Wilson interval |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Calibration (selected threshold) | 1000 | 93 | 9.30% | 81/93 | 87.10% | 78.79%–92.46% |
| shared development evaluation | 600 | 100 | 16.67% | 73/100 | 73.00% | 63.57%–80.73% |

Human review: **500/600**. Accepted errors: **27**. API failures: **0** (always human review).
Observed >=90% accuracy and >=10% acceptance: **False**.
Evaluation 95% lower bound >=90%: **False**.
New case requests: 1600. Confirmation source method: None.

## Evidence limits and reproduction

Passes4–12 reuse the same 600-case development evaluation batch. Their prompt configurations were frozen together before any new calls, but choosing a winner from their results creates selection bias. Pass13 alone uses 1,200 reserved cases excluded from all prior baseline predictions, pass3 evaluation and this development batch, and copies the chosen calibration rule unchanged. Calibration and evaluation have no normalized exact-text overlaps. Near-duplicate groups were not available for blocking. Intervals are approximate binomial Wilson intervals, not adjusted for multiple comparisons, and assume independent observations. The historical dataset contains clinical-review-style narrative and may not represent incoming appeals. No result establishes clinical causality or future production accuracy.

`plan.json` records IDs, prompt configuration and experiment role. `rule.json` is saved before evaluation. Predictions and `decisions.csv` retain every case for audit. `metrics.json` contains unrounded results. Run `.venv/bin/python jev/selective-automation/advanced_trials.py --pass-number 9` from repository root; completed trials are protected, partial predictions checkpointed. A valid ignored `jev/.env` is required. For pass13, calibration responses are reused from the selected method, not newly called.
