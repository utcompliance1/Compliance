# IMR Appeals LLM Baseline

Start with [Baselines: results and comparisons](Baselines/README.md) for an
overview and detailed Jev, GPT, and DistilBERT results.

LLM baseline for predicting IMR (Independent Medical Review) appeal outcomes
(`Upheld` vs `Overturned`) from case descriptions, using the
[Persius/imr-appeals](https://huggingface.co/datasets/Persius/imr-appeals) dataset.

## Contents
- `baseline_eval.py` — data loading, sampling, prompt construction (zero-shot and
  few-shot), and the evaluation pipeline used to score OpenAI models against the
  dataset's labeled `decision` field.
- `baseline_results.json` — raw accuracy/timing/cost output from the final run.
- `RESULTS.md` — write-up of findings, methodology notes, and the accuracy comparison
  across model sizes.

## Setup
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
python baseline_eval.py
```

## Methodology notes
- Predictions are made from the dataset's `text` field only. The `full_text` field
  was excluded because it contains the reviewer's own reasoning/conclusion, which
  leaks the label.
- The few-shot prompt uses 4 manually vetted examples (2 Upheld, 2 Overturned),
  selected for completeness (no truncation) and explicit denial rationale.
- Models are queried with `reasoning_effort="minimal"` and evaluated with
  `temperature` at its default (gpt-5-family models do not support overriding it).

## Original xverifier implementation

[Limited GPT-5 results on cleaned and uncleaned inputs](original-implementation-results/xverifier/RESULTS.md), with original per-seed outputs, saved predictions, and reproduction scripts.

[Matched Jev and GPT-5 xverifier pilot with presentation plots](original-implementation-results/jev-xverifier/RESULTS.md).


## Latest selective automation result

The DistilBERT + JEV intensity-of-care specialist accepted **45.1%** of 1,200 reserved historical cases at **92.8% accuracy among accepted cases**, exceeding the 25% / 90% target. [Meeting-ready best run](best%20run/README.md) contains the summary, exact rule, prompt, results and per-case evidence. [Full trial report](distilbert/selective-automation/pass13/README.md) explains calibration, confirmation and limitations.
