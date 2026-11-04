# Jev baseline

**Jev 1.13.0 correctly predicted 4,757 of 6,915 completed cases (68.79%).**
The planned 9,846-case run was interrupted; 2,931 cases were never attempted.
All attempted cases returned valid predictions. This is a partial-run result.

[Back to comparison](../README.md)

## Saved runs

| Run | Prompt | Correct / scored | Accuracy | Sample | Estimated API cost |
| --- | --- | ---: | ---: | --- | ---: |
| [Larger interrupted run](../../jev/full-test-focused/baseline_results_jev.json) | Focused, zero-shot | 4,757 / 6,915 | 68.79% | Completed test prefix | $0.189201 |
| [1,000-case snapshot](../../jev/run-1000/baseline_results_jev.json) | Focused, zero-shot | 695 / 1,000 | 69.50% | First 1,000 test rows from larger run | $0.026348 |
| [Original pilot](../../jev/output/baseline_results_jev.json) | Basic, zero-shot | 132 / 200 | 66.00% | Balanced random sample, seed 42 | $0.004225 |

The snapshot reuses predictions and its cost is already included in the larger
run. Do not add their cases or costs together. The pilot uses different cases
and a different prompt; the score difference does not isolate a prompt effect.

## Method

- Dataset: cleaned IMR appeals, `test` split, revision
  `0abb1e862c71e2b0558b560fceedab3d528759f7`.
- Input: `text` only, one native Choice question per case, with `Upheld` and
  `Overturned` as options. No labeled examples or model training in the selected run.
- Focused criteria cover the requested care, clinical needs, prior treatment,
  alternatives, evidence of benefit, and denial rationale. Missing facts are
  not automatically treated as evidence against the request.
- Prompt development used 100 balanced training-split cases after excluding
  normalized exact-text overlap with the test split. Basic scored 68%, structured
  66%, focused 69%, and focused plus four examples 69%. The tie favored focused
  zero-shot; selection was saved before the larger test run.
- Four concurrent workers were used for the larger run. No confidence filtering
  was applied to its headline accuracy.

See [exact prompts](../../jev/prompts.py),
[selection record](../../jev/prompt_selection.json), and
[runner](../../jev/baseline_eval_jev.py).

## Larger-run results

| Metric | Value |
| --- | ---: |
| Completed / planned cases | 6,915 / 9,846 (70.23%) |
| Accuracy | 68.79% |
| Approximate Wilson 95% interval | 67.69%–69.87% |
| Balanced accuracy (average recall across the two classes) | 68.84% |
| Majority baseline among completed cases | 53.48% |
| Elapsed time | 268.86 seconds |
| Median / p95 client call latency | 146 / 219 ms |
| Input / output tokens | 4,504,782 / 255,762 |

The interval describes uncertainty for the observed cases; it does not correct
for evaluating a dataset prefix. The raw summary's `majority_class_baseline`
is 53.90% for the entire planned split; 53.48% above uses completed cases only.
Its `accuracy_all_requested` of 48.31% divides correct answers by all 9,846
planned cases, including unattempted ones; it is not the scored-case accuracy.

Rows below are actual outcomes; columns are predicted outcomes.

| Actual outcome | Predicted Upheld | Predicted Overturned |
| --- | ---: | ---: |
| Upheld | 2,522 | 1,176 |
| Overturned | 982 | 2,235 |

| Class | Precision | Recall | F1 | Cases |
| --- | ---: | ---: | ---: | ---: |
| Upheld | 71.97% | 68.20% | 70.04% | 3,698 |
| Overturned | 65.52% | 69.47% | 67.44% | 3,217 |

Precision asks how often a predicted label is right; recall asks how many actual
cases of that label are found. F1 combines precision and recall.

## Confidence and cost interpretation

Keeping only predictions with returned confidence at least 0.75 gives **80.98%
accuracy on 39.70% of completed cases**. The other 60.30% remain unresolved by
that rule. This does not raise all-case accuracy to 80.98%. Returned confidence
is not itself a calibrated probability of correctness. See the
[confidence report](../../jev/CONFIDENCE_STATS.md) and
[aggregate statistics](../../jev/confidence_stats.json).

Cost uses the run's saved rate of $0.042 per million input tokens and free output
tokens. It excludes pilot and prompt-development calls. Latency includes network
and client overhead; concurrent wall time is not a per-case latency measurement.

## Reproduction and evidence

Follow the [original setup guide](../../jev/README.md). To generate a new balanced
random sample after installing dependencies and configuring the API key:

```bash
python jev/baseline_eval_jev.py --n 1000 --prompt focused --workers 4 \
  --output-dir jev/baselines-rerun-1000
```

This makes paid API calls and does **not** reproduce the saved 1,000-row prefix.
Use a fresh output directory. No rerun was made for this documentation.

The [1,000-case manifest](../../jev/run-1000/jev_sample.json) and
[predictions](../../jev/run-1000/jev_predictions.jsonl) are tracked. The larger
prediction file is local and excluded from Git; its summary and confidence
aggregates are tracked. Further context: [original results](../../jev/RESULTS.md).
