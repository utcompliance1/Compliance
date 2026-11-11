# Pass 3: ask JEV for evidence-based automation eligibility

This trial changes the actual JEV question and criteria. JEV chooses AutomateUpheld, AutomateOverturned, or HumanReview from case text. The prompt asks for decisive patient-specific evidence, distinguishes explicit negative findings from missing information, and defers mixed or generic cases. See prompt.json for exact instructions.

Pattern hypothesis: explicit stability / no acute need can resolve intensity-of-care appeals; persistent disease with documented failed alternatives can resolve supported-treatment appeals. The discovery association for “did not” + Upheld + confidence >=0.80 was 84/85 correct. This motivated a semantic question, not a keyword filter or a causal claim.

| Split | Tasks | Accepted | Acceptance | Correct/accepted | Accuracy | 95% Wilson interval |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Calibration (threshold selection) | 600 | 84 | 14.00% | 72/84 | 85.71% | 76.67%–91.63% |
| Held-out evaluation | 600 | 147 | 24.50% | 122/147 | 82.99% | 76.10%–88.21% |

Frozen rule: JEV must choose an automation outcome with native three-choice confidence >= 0. Everything else, including API failure, goes to a human. Human-review cases in evaluation: 453/600. Observed 90% target: NOT MET.

There are 600 live calibration cases and 600 separate live evaluation cases. Evaluation cases are outside all 6,915 cached baseline predictions, with normalized exact text overlap excluded. Calibration selects the highest-coverage threshold with >=92% observed accuracy and >=60 acceptances; fallback maximizes the lower Wilson bound with >=60 acceptances and is explicitly exploratory. No evaluation result changes the prompt or threshold. Calibration intervals are selection-biased; evaluation intervals are approximate and not adjusted for repeated trials. A point estimate above 90% is not a guarantee. Near-duplicate group separation is not verified.

Model: jev-1.13.0. Input: text only, never outcome labels or full_text. Source remains the historical cleaned IMR dataset, not prospective production data. Some text retains clinical-review-style narrative; performance may not transfer to incoming appeals without that narrative. New and old trials use different evaluation batches, so their difference is not a controlled causal estimate of prompt improvement. No human decisions were measured or executed.

Artifacts: plan.json records split IDs and selection procedure; prompt.json and rule.json freeze instructions and threshold; predictions JSONL and decisions.csv permit independent scoring; metrics.json stores results. Reproduce with `.venv/bin/python jev/selective-automation/prompt_trial.py`; completed results are protected against overwrite. Only a zero-success authentication failure may restart in place.

## Matched original-prompt control

An additional 600 successful live calls applied the original focused prompt and the already-fixed pass1 confidence threshold (0.94) to the same evaluation cases. This control was not tuned on their outcomes.

| Method on identical 600 cases | Accepted | Acceptance | Correct | Accuracy | 95% Wilson interval |
| --- | ---: | ---: | ---: | ---: | --- |
| Original prompt + confidence >=0.94 | 66 | 11.00% | 59 | 89.39% | 79.69%–94.77% |
| Revised evidence-based prompt | 147 | 24.50% | 122 | 82.99% | 76.10%–88.21% |

The revised method alone accepted 110 cases that the original method rejected; 87 were correct (79.09%). Thus the revised routing strategy expands acceptance but its extra coverage is not accurate enough. This comparison holds cases and model version fixed, but changes both the question and routing strategy; it does not isolate one instruction or prove clinical causality.

No calibration threshold achieved the required 92% calibration accuracy with at least 60 accepted cases. The predeclared fallback selected threshold 0: all explicit automation choices are retained, while HumanReview choices are rejected. This is an unsuccessful exploratory trial, not a deployment-approved policy. Total successful live calls for this trial including its control: 1,800 (plus authentication failures recorded separately).

`control_plan.json`, `control_predictions.jsonl`, and `control_metrics.json` document the matched reference. Run `../verify.py` from any working directory via Python to independently audit the primary result counts and split separation.
