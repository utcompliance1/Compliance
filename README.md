# IMR Appeals Baselines

Baselines for predicting IMR (Independent Medical Review) appeal outcomes
(`Upheld` vs `Overturned`) from case descriptions, using the
[Persius/imr-appeals](https://huggingface.co/datasets/Persius/imr-appeals) dataset.

## Contents
- `week1-results/baseline_eval.py` — data loading, sampling, prompt construction (zero-shot and
  few-shot), and the evaluation pipeline used to score OpenAI models against the
  dataset's labeled `decision` field.
- `week1-results/baseline_results.json` — raw accuracy/timing/cost output from the final run.
- `week1-results/RESULTS.md` — write-up of findings, methodology notes, and the accuracy comparison
  across model sizes.
- `data-processing/` — cleaned dataset pipeline and audit notes.
- `week2-results/` — DistilBERT training and results.
- `jev-results/baseline_eval_jev.py` — zero-shot Jev evaluation on the cleaned
  test split. See [Jev setup and methodology](jev-results/README.md).

## Setup
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
python week1-results/baseline_eval.py
```

## Jev baseline

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set TYPESAFE_API_KEY in .env locally.
python jev-results/baseline_eval_jev.py --n 200
```

The default `focused` prompt was chosen on a separate 100-case development
sample. It specifies medical necessity, evidence of benefit, appropriate care
setting, and denial-rationale criteria. To evaluate the entire cleaned test split:

```bash
python jev-results/baseline_eval_jev.py --full-test --workers 4 \
  --output-dir jev-results/full-test-focused
```

Use `--dry-run` to prepare the sample without an API key or billable calls.
Outputs include accuracy, per-class metrics, confidence/probabilities, latency,
estimated cost, and a reproducible sample manifest. Results are saved locally.

## LLM methodology notes
- Predictions are made from the dataset's `text` field only. The `full_text` field
  was excluded because it contains the reviewer's own reasoning/conclusion, which
  leaks the label.
- The few-shot prompt uses 4 manually vetted examples (2 Upheld, 2 Overturned),
  selected for completeness (no truncation) and explicit denial rationale.
- Models are queried with `reasoning_effort="minimal"` and evaluated with
  `temperature` at its default (gpt-5-family models do not support overriding it).
