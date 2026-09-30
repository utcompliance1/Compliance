# Jev accuracy and confidence statistics

**Overall accuracy: 68.79% (4,757/6,915 correct).**

This report covers all **6,915 completed test cases** from the interrupted focused-prompt run. The separately saved 1,000-case snapshot scored 69.50%; it is contained within these cases and is not a separate experiment.

## Overall results

| Metric | Result |
| --- | ---: |
| Scored cases | 6,915 |
| Correct / incorrect | 4,757 / 2,158 |
| Accuracy | 68.79% |
| Approximate 95% Wilson interval | 67.69%–69.87% |
| Balanced accuracy | 68.84% |
| Majority baseline on evaluated cases | 53.48% |
| Upheld / Overturned true labels | 3,698 / 3,217 |
| Failed requests | 0 |
| Requested / unattempted test cases | 9,846 / 2,931 |
| Coverage of the full test split | 70.23% |
| Total elapsed time | 268.86 seconds |
| Median / p95 client latency | 146 / 219 ms |
| Input / output tokens | 4,504,782 / 255,762 |
| Estimated cost of this larger run | $0.189201 |

The run stopped at the user's request. Every attempted case was scored; the remaining cases were unattempted, not API failures. The evaluated cases are a prefix of the test split, not a newly drawn random sample. The full 9,846-case split has a 53.90% majority baseline; this report uses the evaluated prefix's own baseline above.

## Confidence ranges

Each case belongs to exactly one range. Share of cases uses all completed cases as its denominator. Share of errors uses all incorrect predictions as its denominator.

| Returned confidence c | Cases | Share of cases | Correct | Incorrect | Accuracy | Share of all errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 ≤ c < 0.25 | 1,152 | 16.66% | 581 | 571 | 50.43% | 26.46% |
| 0.25 ≤ c < 0.50 | 1,232 | 17.82% | 783 | 449 | 63.56% | 20.81% |
| 0.50 ≤ c < 0.75 | 1,786 | 25.83% | 1,170 | 616 | 65.51% | 28.54% |
| 0.75 ≤ c ≤ 1.00 | 2,745 | 39.70% | 2,223 | 522 | 80.98% | 24.19% |
| **Overall** | **6,915** | **100.00%** | **4,757** | **2,158** | **68.79%** | **100.00%** |

Percentages are rounded, so displayed shares may differ slightly from 100% when summed.

## Accuracy at cumulative confidence cutoffs

These rows overlap: each cutoff includes all cases at or above that confidence.

| Minimum confidence | Retained cases | Share retained | Share excluded | Accuracy on retained cases |
| --- | ---: | ---: | ---: | ---: |
| 0.00 | 6,915 | 100.00% | 0.00% | 68.79% |
| 0.25 | 5,763 | 83.34% | 16.66% | 72.46% |
| 0.50 | 4,531 | 65.52% | 34.48% | 74.88% |
| 0.75 | 2,745 | 39.70% | 60.30% | 80.98% |
| 0.90 | 1,331 | 19.25% | 80.75% | 87.00% |

At a **0.75 cutoff**, accuracy is **80.98%** on **39.70%** of completed cases. The other **60.30%** would need another prediction method or review. This cutoff does not increase accuracy across all cases by itself.

Confidence here is the API's returned `confidence` field, not the selected label's probability. TypeSafe derives it from the probability distribution. An individual score of 0.75 is not a guarantee of 75% accuracy. [Official confidence documentation](https://docs.typesafe.ai/confidence).

## Per-class metrics

| True class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Upheld | 71.97% | 68.20% | 70.04% | 3,698 |
| Overturned | 65.52% | 69.47% | 67.44% | 3,217 |

Confusion matrix (rows are true labels):

| | Predicted Upheld | Predicted Overturned |
| --- | ---: | ---: |
| True Upheld | 2,522 | 1,176 |
| True Overturned | 982 | 2,235 |

## Method and provenance

- Model: `jev-1.13.0`; focused zero-shot prompt selected on separate training-split development cases.
- Dataset: `utcompliance1/imr-appeals-cleaned`, `test` split; revision `0abb1e862c71e2b0558b560fceedab3d528759f7`.
- Input: `text` only; reviewer reasoning in `full_text` and ground-truth decisions were excluded from requests.
- Saved run timestamp: `2026-09-30T03:51:15.594281+00:00`.
- Main metrics source: [interrupted-run summary](full-test-focused/baseline_results_jev.json).
- Counts, confidence ranges, and intervals: [machine-readable statistics](confidence_stats.json).
- Original 1,000-case report: [results and methodology](RESULTS.md).
- Prediction file SHA-256: `f6966acc8405b556fd0855928e6907471680fea97bc96d3f6cf00762d2b1e5e0`.

All confidence statistics were calculated from the existing local `full-test-focused/jev_predictions.jsonl`; no new API calls were made. That larger prediction file is kept locally and excluded from Git; the report publishes aggregate statistics. With that local file available, regenerate using:

```bash
python jev/confidence_report.py
```

The cutoff results describe this evaluated dataset. Select operational thresholds on development data and verify on new cases before treating them as expected future performance.
