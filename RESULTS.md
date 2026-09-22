# LLM Baseline Results

## Setup
- Dataset: `Persius/imr-appeals`, `train` split (64,067 rows)
- Class balance: 53% Upheld / 47% Overturned → majority-class baseline = **53%**
- Eval sample: 6,406 rows (~10%), stratified by `decision`
- Input: `text` field only (see methodology note on `full_text` leakage in README)

## Prompt comparison (dev sample, 63 rows)
| Prompt | Accuracy | Unparsed |
|---|---|---|
| Zero-shot | 61.3% | 1 |
| Few-shot (4 examples) | 66.7% | 0 |

Few-shot selected for the full run based on this comparison.

## Full baseline (eval sample, 6,406 rows, few-shot prompt)
| Model | Accuracy | Unparsed | Time | Cost |
|---|---|---|---|---|
| gpt-5 | 71.9% | 0 (0%) | 10.0 min | $5.59 |
| gpt-5-mini | 66.7% | 191 (3.0%) | 8.8 min | $1.12 |
| gpt-5-nano | 59.2% | 34 (0.5%) | 7.2 min | $0.22 |

All three models exceed the 53% majority-class baseline. Accuracy scales
monotonically with model size. Notably, gpt-5-mini's full-scale accuracy
(66.7%) exactly matches its accuracy on the 63-row dev sample used for
prompt iteration, suggesting the dev sample was representative and the
prompt-selection process generalized well. gpt-5-mini's unparsed rate is
higher than either gpt-5 or gpt-5-nano's, which doesn't track cleanly with
model size and is noted here as an open observation rather than a diagnosed
cause.

Cost/accuracy tradeoff: gpt-5 gains ~5 points of accuracy over gpt-5-mini
for roughly 5x the cost; gpt-5-nano trades ~7-13 points of accuracy for a
further ~5x cost reduction versus gpt-5-mini.

## Error analysis
Qualitative review of dev-sample mismatches surfaced two patterns:
1. Detailed inpatient-admission narratives (with clinical severity) are
   over-predicted as Overturned, even when the true label is Upheld.
2. Terse home-health/personal-care-hours requests, with minimal case detail,
   are over-predicted as Upheld — possibly reflecting missing context in
   `text` that the original reviewer had access to.

## Known data issues (flagged for dataset-cleaning task)
- `appeal_type` has inconsistent capitalization (e.g. "Medical necessity" vs
  "Medical Necessity") that splits what should be one category.
- Some `text` rows are truncated mid-sentence.
