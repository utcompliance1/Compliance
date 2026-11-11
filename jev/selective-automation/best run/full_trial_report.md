# Pass 13: DistilBERT / JEV specialist hybrid

This is the tenth additional authorized trial. The updated goal is **>=25% acceptance and >=90% accuracy**. DistilBERT predicts outcomes; ground truth remains the recorded dataset decision. The validation-selected DistilBERT was trained/selected on source train/validation, not source test. Cached test predictions are reused; no retraining or fresh DistilBERT inference occurred.

## Pattern and frozen acceptance criteria

Calibration compares three methods: DistilBERT confidence alone; a union with JEV's intensity-of-care specialist; and that union with disagreement between two accepted predictions sent to human review. The JEV question distinguishes documented stability/lower-intensity suitability from acute danger or failed lower care, and marks out-of-scope or unresolved cases Unresolved. Its exact wording is in jev_prompt.json, unchanged from JEV pass5.

Selected mode: **specialist_union**. DistilBERT selected-class probability threshold: **0.9**. JEV specialist selected-class probability threshold: **0.875**. In specialist_union, an eligible JEV specialist takes precedence; otherwise accept eligible DistilBERT. In union_defer_conflicts, two eligible but disagreeing predictions go to review. All remaining cases go to review. DistilBERT-only mode ignores JEV. Scores are not statistical confidence intervals.

A 1,000-case calibration set selects the method and threshold by maximum acceptance subject to >=92% accuracy, >=250 accepted, and 95% Wilson lower bound >=88%. If no method qualifies, fallback is exploratory. Calibration target met: **True**. Calibration method selection is optimistic; the 1,200-case reserved confirmation never changes the rule.

## Results

| Split | Tasks | Accepted | Acceptance | Correct/accepted | Accuracy | 95% Wilson interval |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Calibration (selected) | 1000 | 355 | 35.50% | 327/355 | 92.11% | 88.84%–94.49% |
| Reserved confirmation | 1200 | 541 | 45.08% | 502/541 | 92.79% | 90.30%–94.68% |
| Confirmation excluding outcome shortcuts | 1181 | 522 | 44.20% | 483/522 | 92.53% | 89.95%–94.49% |

Human review on confirmation: **659/1200**. Accepted errors: **39**. Both updated targets met: **True**. Accuracy 95% lower bound >=90%: **True**.

## Provenance and limits

All 2,200 DistilBERT source rows are joined by row_id; dataset labels and original row order are checked against the JEV dataset manifest. Source and policy hashes are preserved. DistilBERT source: week3-results/test_predictions.csv, model utcompliance1/distilbert-imr-val-selected (epoch2 checkpoint selected by validation loss). New JEV calls: 1200. New DistilBERT calls: 0.

Confirmation IDs are the 1,200 cases reserved before the new JEV comparison trials. These outcomes were not used for current prompt/threshold selection, but historical DistilBERT aggregate test results existed; this is retrospective confirmation. Normalized exact texts are separated from calibration and development. Near-duplicate group independence and historical checkpoint training exclusion are inherited from repository documentation, not freshly audited by retraining. Wilson intervals assume independent cases. Clinical-review-style narrative and dataset shift limit transfer to incoming appeals. Literal outcome shortcut exclusion is reported separately and does not affect selection. No clinical causal claim or future accuracy guarantee is made.

`plan.json`, `rule.json`, `calibration_sweep.json`, per-case CSVs and `metrics.json` preserve the experiment. Reproduce with `.venv/bin/python distilbert/selective-automation/run_trial.py` in a fresh output directory. Completed outputs are protected. The candidate search is one calibration procedure, not 48 additional held-out tests.
