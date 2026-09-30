# Jev baseline results

For all 6,915 completed cases, see [overall metrics and confidence statistics](CONFIDENCE_STATS.md).

## Current result: 1,000 cases with focused criteria

**Accuracy: 69.5% (695/1,000)**. All 1,000 cases returned valid decisions.
Balanced accuracy was 69.45%; the majority-class baseline was 50.3%.
The approximate Wilson 95% interval is 66.6%–72.3%.

The selected `focused` prompt specifies medical necessity for the particular
requested care, appropriate setting/intensity, evidence of benefit, prior
treatment response, alternatives, and the insurer's denial rationale. It
distinguishes the initial insurer denial from the final review decision, and
does not treat missing case facts as negative findings. Its exact criteria are
in [prompts.py](prompts.py).

Prompt selection used 100 balanced training-split development cases, separate
from the test data. Basic criteria scored 68%, detailed structured criteria 66%,
focused criteria 69%, and focused criteria with four vetted examples 69%.
The focused zero-shot version won the tie. The differences are small; the
selection was frozen before the larger evaluation in
[prompt_selection.json](prompt_selection.json).

The full test run was stopped at the user's request. By then **6,915 cases had
completed**, with 68.79% accuracy (4,757 correct) and an estimated cost of
$0.189201. Those predictions and the interrupted summary are preserved in
`full-test-focused/`.

The requested 1,000-case result uses the **first 1,000 test rows in dataset
order**, extracted from those completed predictions without further API calls.
It contains 503 Upheld and 497 Overturned cases. This is a prefix of the test
split, not a seeded random sample or an evaluation of all 9,846 cases.

| Metric | 1,000-case result |
| --- | ---: |
| Accuracy | 69.5% |
| Balanced accuracy | 69.45% |
| Successful requests | 1,000/1,000 |
| Median / p95 client call latency | 157 / 310 ms |
| Estimated cost attributable to these 1,000 calls | $0.026348 |

The subset cost is not the total experiment bill: the interrupted larger run
cost approximately $0.189201, plus the earlier pilot and development calls.
No separate wall time was measured for the extracted subset.

- [1,000-case metrics](run-1000/baseline_results_jev.json)
- [1,000-case predictions](run-1000/jev_predictions.jsonl)
- [1,000-case manifest](run-1000/jev_sample.json)
- [Interrupted larger-run metrics](full-test-focused/baseline_results_jev.json)

To run a new **balanced random** 1,000-case sample instead of this saved prefix:

```bash
jev/.venv/bin/python jev/baseline_eval_jev.py --n 1000 --prompt focused \
  --workers 4 --output-dir jev/random-1000
```

No additional run was started after the request to stop.

## Original 200-case baseline

Live run on September 29, 2026 (America/Chicago), using `jev-1.13.0`.

**Accuracy: 66.0% (132/200)** on a balanced sample from the cleaned IMR appeals
test split. The majority-class baseline for this sample is 50.0%. All 200
requests returned valid decisions; no cases were excluded from scoring.

| Metric | Result |
| --- | ---: |
| Accuracy / balanced accuracy | 66.0% |
| Approximate Wilson 95% accuracy interval | 59.2%–72.2% |
| Successful requests | 200/200 |
| Total evaluation time | 28.30 seconds |
| Median client call latency | 137 ms |
| p95 client call latency | 179 ms |
| Input / output tokens | 100,602 / 7,412 |
| Estimated API cost | $0.004225 (about 0.42 cents) |

Cost uses [TypeSafe's documented pricing](https://docs.typesafe.ai/models):
$0.042 per million input tokens, with free output tokens. Latency includes
network and client overhead; it is not isolated model inference time.

## Method

- `utcompliance1/imr-appeals-cleaned`, `test` split, revision
  `0abb1e862c71e2b0558b560fceedab3d528759f7`.
- 200 cases: 100 Upheld, 100 Overturned; pandas sampling without replacement,
  seed 42, following the existing baselines' class sampling procedure.
- Zero-shot: one native Choice question per case, with Upheld and Overturned
  as the two options. No prompt tuning or labeled examples were used.
- Input was only the `text` case description. `full_text`, the true decision,
  and row metadata were excluded from the API request.
- Low-confidence predictions were retained. Exact settings, wording, and sample
  IDs are in [the sample manifest](output/jev_sample.json).

## Per-class results

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Upheld | 67.0% | 63.0% | 64.9% | 100 |
| Overturned | 65.1% | 69.0% | 67.0% | 100 |

Confusion matrix (rows are true labels):

| | Predicted Upheld | Predicted Overturned |
| --- | ---: | ---: |
| True Upheld | 63 | 37 |
| True Overturned | 31 | 69 |

Jev predicted Overturned 106 times and Upheld 94 times. Mean returned confidence
was 0.627 for correct predictions and 0.437 for incorrect predictions. This is
a descriptive check on this sample, not a calibration study.

## Interpretation

Jev exceeded the balanced sample's majority-class baseline by 16 percentage
points at a total API cost below one cent. This establishes a small initial
baseline for the task, with meaningful uncertainty from the 200-case sample.

The existing 4,000-case results were 73.1% for few-shot GPT-5, 60.6% for
few-shot GPT-5-nano, and 74.65% for fine-tuned DistilBERT. These used different
methods and a larger sample. The Jev pilot cannot establish a model ranking
against those historical results; a controlled comparison needs all models
evaluated on the same saved cases with declared prompts.

All 200 pilot row IDs were verified to belong to the earlier canonical
4,000-case sample, so the saved manifest can support that comparison.

## Local artifacts and reproduction

- [Runner](baseline_eval_jev.py)
- [Summary metrics](output/baseline_results_jev.json)
- [Per-case predictions](output/jev_predictions.jsonl)
- [Setup and methodology](README.md)

```bash
jev/.venv/bin/python jev/baseline_eval_jev.py --n 200 --prompt basic \
  --output-dir jev/rerun-200
```

The sample is deterministic; live model answers can vary across runs. Code,
results, and the write-up are local. Nothing was pushed to GitHub or Hugging Face.
