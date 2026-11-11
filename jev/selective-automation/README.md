# JEV selective automation experiments

Goal: use JEV to identify tasks suitable for automation at >=90% accepted-task accuracy, sending the remaining tasks to human review. Acceptance is the accepted count divided by every task in the evaluation batch; accuracy is correct JEV outcomes divided by accepted tasks. Rejections are not counted as errors or assumed correct human decisions.

## Trials

- **pass1**: existing JEV answers, global confidence threshold; 600 tasks, 60 accepted (10%), 53 correct (88.33%). Committed and pushed as `f686ec3`.
- **pass2**: existing JEV answers, outcome-specific confidence criterion; 600 tasks, 23 accepted (3.83%), 19 correct (82.61%). Committed and pushed as `e3a0c88`.
- **pass3**: actual new JEV question and criteria: identify decisive clinical evidence, choose automated Upheld / automated Overturned / human review. Uses 600 live calibration cases and 600 separate live evaluation cases. Evaluation: 147/600 accepted (24.5%), 122 correct (82.99%); target not met. The matched original-prompt control accepted 66/600 (11%) at 89.39% accuracy. See its README for the frozen prompt and details.
- **exploratory-keyword-filter**: preserved offline search that finished before the user clarified that trials should change what JEV is asked. It is not the revised pass3 and is not a recommended policy. Its tiny evaluation subset (12/600 accepted, 11 correct) illustrates why high point accuracy alone is inadequate.

Passes 1/2 are preserved as requested. They tested post-processing, not prompt changes. Their strict confidence filtering reduced coverage without reliably hitting the target. The revised pass3 asks JEV to interpret the characteristics behind promising discovery associations instead of requiring literal keywords.

## Evidence motivating the revised question

On the original discovery split, `did not` + predicted Upheld + confidence >=0.80 yielded 84/85 correct. Inspection showed concrete negative clinical findings such as no withdrawal, no acute danger, and medical/psychiatric stability in appeals for intensive care. Other promising examples contained documented failed treatments, intolerance, persistent disease, and ongoing dangerous impairment supporting escalation. These are hypotheses about useful discriminating evidence, not proven causes. The broad keyword search is selection-biased; its high training precision is not independent validation.

The new prompt asks whether facts directly support/refute the requested care, whether material contrary evidence remains, and whether missing information would require assumptions. It includes both outcomes to avoid the severe coverage loss from accepting only one class. Generic diagnosis/request/denial descriptions and unresolved cases go to review. Exact wording is preserved in `pass3/prompt.json`.

Some source descriptions retain clinical-review-style narratives even though only cleaned `text` is used. This can make historical outcomes easier to infer than incoming claims. The experiment therefore establishes, at most, retrospective performance on this dataset. It cannot establish causal effects of clinical characteristics or certify production safety.

## Research and statistical interpretation

[Selective Classification for Deep Neural Networks (Geifman and El-Yaniv, 2017)](https://papers.neurips.cc/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf) studies the risk–coverage tradeoff: abstain on uncertain cases to reduce error among accepted cases. This motivates reporting both coverage and conditional accuracy rather than accuracy alone.

[SelectiveNet (Geifman and El-Yaniv, 2019)](https://proceedings.mlr.press/v97/geifman19a.html) learns a selection mechanism jointly with prediction. Here the practical analogue is a separate automation/deferral judgment within JEV's question; we do not train SelectiveNet or claim its guarantees.

[TypeSafe confidence documentation](https://docs.typesafe.ai/confidence) distinguishes confidence from outcome probabilities and describes routing based on uncertainty. Native confidence is a concentration statistic, not an empirical accuracy confidence interval. Adding more answer options changes its interpretation; the new three-choice prompt is calibrated separately.

[TypeSafe customization documentation](https://docs.typesafe.ai/models) explicitly supports adapting instructions/criteria and decomposing judgments into atomic questions. A future method could ask separate evidence-sufficiency, contradictory-evidence, and clinical-indication questions, then calibrate their combined probabilities. This would require its own >=500-case trial rather than claiming success from the present data.

Use untouched evaluation cases after prompt and threshold selection. Report binomial intervals and adequate accepted counts. If searching many rules, discovery intervals are optimistic; subsequent calibration selection also invalidates treating its nominal interval as certification. The final evaluation interval remains only approximate, subject to residual dependence and dataset shift. A fresh, group-disjoint confirmation with a lower bound above 90% would provide stronger support than a 90% point estimate alone.

## Reproduction

`run.py` reproduces the cached routing baselines with `--pass-number 1` or `2`; output folders must not already exist. Its former keyword-search mode is archived separately. `splits.json` stores source hashes and row IDs. The large original source cache is local and may need to be supplied separately.

`prompt_trial.py` performs the live revised prompt experiment and writes per-case predictions, calibration sweep, frozen rule, metrics and README. It requires a valid `TYPESAFE_API_KEY` in ignored `jev/.env`. Completed experiments cannot be overwritten; an authentication-only failure with zero successful calls can restart. No credentials or full patient narratives are included in the new committed prediction artifacts.

## Conclusion

The revised prompt increases coverage, but these three trials do not establish a routing method achieving >=90% accuracy at >=10% acceptance. The promising discovery patterns did not generalize sufficiently. All measured failures are retained; no test-set tuning or selective omission was used to manufacture a success. Stronger candidates would distinguish explicit clinical facts from generic insurer assertions using separate JEV questions, and validate the combined criterion on another untouched batch rather than reuse this test set.
