# IMR Appeals LLM Baseline

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
