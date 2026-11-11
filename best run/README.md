# Best run — meeting brief

**We found a selective automation rule that exceeds the target: 45.1% acceptance at 92.8% accuracy among accepted cases.**

The target was **25%+ acceptance and 90%+ accuracy**. The result below is from **1,200 reserved historical cases**, separate from the 1,000 cases used to choose the final rule.

## Results to present

| Measure | Result |
| --- | ---: |
| Cases evaluated | 1,200 |
| Accepted for automation | **541 (45.1%)** |
| Correct accepted decisions | **502 of 541 (92.8%)** |
| Incorrect accepted decisions | 39 |
| Sent to human review | **659 (54.9%)** |
| 95% interval for accepted accuracy | **90.3%–94.7%** |

Accuracy is measured against the **recorded IMR outcome**, not against another model. These are simulated routing decisions; no real claims were automatically adjudicated.

## What changed

A general JEV confidence cutoff was not enough. The most useful new JEV question focuses on **whether the documented clinical facts justify the requested intensity of care**:

- Explicit stability, no acute danger or withdrawal, and suitability for a lower level of care can support sustaining a denial.
- Acute danger, severe impairment, or failed lower-intensity care can support overturning a denial.
- Missing, conflicting, or out-of-scope evidence is marked unresolved.

DistilBERT supplies the broader outcome prediction. This combination expands the accepted subset beyond the narrow JEV specialist alone.

## Exact frozen rule

1. If JEV's intensity-of-care question chooses **Upheld** or **Overturned** with selected-outcome probability **≥0.875**, use that outcome.
2. Otherwise, if DistilBERT's selected-outcome probability is **≥0.90**, use its outcome.
3. Otherwise, **send the case to a human reviewer**.

The JEV specialist takes precedence when eligible, including when the models disagree. These probability cutoffs are model scores; the accuracy interval above is calculated from observed correctness.

| Source of accepted decision | Accepted | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| JEV intensity-of-care specialist | 161 | 151 | 93.8% |
| DistilBERT fallback | 380 | 351 | 92.4% |
| **Combined** | **541** | **502** | **92.8%** |

## One-minute explanation

> “We changed what we ask JEV instead of only tightening a generic confidence threshold. JEV identifies a narrow group where explicit clinical facts resolve the required level of care. We combine that specialist with high-confidence DistilBERT predictions and defer the rest. On 1,200 reserved historical cases, the frozen rule accepted 45% of tasks and was correct on 93% of those accepted. The remaining 55% went to human review.”

## Important scope

This is **retrospective dataset evidence, not a production accuracy guarantee**. DistilBERT outputs were already saved; the final trial made 1,200 new JEV calls. The DistilBERT checkpoint was selected using training/validation data. Calibration chose the hybrid rule; confirmation outcomes did not change it.

The dataset includes clinical-review-style narrative and some descriptions that explicitly state an outcome. Removing the 19 literal-outcome shortcut cases still gives **44.2% acceptance and 92.5% accuracy** (483/522 correct), although the interval becomes **89.95%–94.49%**. Near-duplicate dependence and differences from incoming appeals remain limitations. Human-review accuracy was not measured. The patterns are predictive associations, not proven clinical causes.

## Supporting files

- [Full experiment report](full_trial_report.md)
- [Exact JEV question](jev_prompt.json)
- [Frozen policy](policy.json)
- [Unrounded results](results.json)
- [All 1,200 routing decisions](evaluation_decisions.csv)
- [Source references and file hashes](provenance.json)

This folder is the latest best run from **trial 13**, the final one of ten additional authorized trials. Earlier successful and unsuccessful trials remain in the repository. Independent audit code is in `distilbert/selective-automation/verify.py`.
