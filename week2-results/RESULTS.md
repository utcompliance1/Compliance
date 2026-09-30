# Week 2 Results — Transformer Fine-Tuning & Comparison to LLM Baselines

## Overview
This week's core task was to fine-tune a transformer for text classification
on the IMR appeals task and compare it against the LLM baselines. The
cleaned-dataset LLM re-run (gpt-5 / gpt-5-nano) is a re-run of week 1's
experiment on new data and is documented separately; its headline numbers
are referenced below only for comparison.

## Model & training setup
- Base model: `distilbert-base-uncased` (66M parameters), fine-tuned with a
  2-class classification head (Upheld / Overturned)
- Training data: `utcompliance1/imr-appeals-cleaned`, `train` split (60,096 rows)
- Evaluation data: the same 4,000-row stratified test sample (seed 42) used
  for this week's gpt-5 / gpt-5-nano comparison, ensuring a fair, matched
  comparison across all three models
- Input: `text` field only (same leakage rationale as the LLM pipeline —
  `full_text` contains the reviewer's own reasoning)
- Tokenization: max length 256 tokens, truncation + padding
- Training: 3 epochs, batch size 16, `Trainer` API with
  `load_best_model_at_end=True` (selection metric: accuracy)
- Hardware: Google Colab, T4 GPU

## Training dynamics and checkpoint selection
| Epoch | Train Loss | Val Loss | Accuracy |
|---|---|---|---|
| 1 | 0.501 | 0.534 | 74.05% |
| 2 | 0.358 | 0.567 | **74.65%** |
| 3 | 0.220 | 0.766 | 74.30% |

Training loss decreased monotonically across all 3 epochs, but validation
loss increased after epoch 1 while accuracy peaked at epoch 2 — a clear
overfitting signature past that point. `load_best_model_at_end` correctly
selected the epoch 2 checkpoint rather than the final epoch, which is the
model used for all results below.

## Final evaluation (4,000-row test set)
**Accuracy: 74.65%**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Upheld | 0.74 | 0.77 | 0.75 | 2,000 |
| Overturned | 0.76 | 0.73 | 0.74 | 2,000 |

Confusion matrix:
|  | Predicted Upheld | Predicted Overturned |
|---|---|---|
| **True Upheld** | 1,531 | 469 |
| **True Overturned** | 545 | 1,455 |

Precision and recall are balanced across both classes (no lopsided bias
toward the majority class), indicating the model is genuinely discriminating
between outcomes rather than defaulting to the more common label.

## Comparison to LLM baselines
| Model | Accuracy | Approach |
|---|---|---|
| **DistilBERT (fine-tuned)** | **74.65%** | Supervised fine-tuning on 60,096 labeled examples |
| gpt-5 | 73.1%* | Few-shot prompting, 4 examples, zero training |
| gpt-5-nano | 60.6%* | Few-shot prompting, 4 examples, zero training |

*From this week's cleaned-dataset LLM re-run (see that write-up for full
methodology and cost details).

**Key finding**: the fine-tuned DistilBERT model slightly outperforms gpt-5
(+1.55 pp) despite being a dramatically smaller model (66M parameters vs.
GPT-5's much larger scale) with no reasoning or generation capability at
inference time. This is a useful illustration of a general pattern: with
enough labeled training data (tens of thousands of examples, vs. 4 few-shot
examples for the LLM), a small, purpose-built classifier can match or exceed
a much larger general-purpose model prompted for the same task.

**Cost/effort tradeoff worth noting**: the LLM approach required no training
data preparation and no training time — just prompt design and inference
calls. The transformer required a labeled training set, ~70 minutes of GPU
training time, and hyperparameter attention (e.g., catching the epoch-3
overfitting). In exchange, transformer inference is essentially free and
near-instant per row once trained, versus real per-call API cost for the
LLM models. Which approach is preferable likely depends on whether the task
needs to scale to very high inference volume (favors the transformer) or
needs to stay flexible/promptable without retraining (favors the LLM).

## Error pattern comparison
Unlike gpt-5's error analysis from week 1 (which showed directional bias —
over-predicting "Overturned" on detailed inpatient narratives and
over-predicting "Upheld" on terse home-health requests), the transformer's
errors are more evenly distributed in both directions (469 vs. 545
misclassifications). This suggests the two approaches may have different
failure modes rather than sharing the same blind spots — worth a deeper
qualitative pass on `transformer_mismatches.json` if time allows, comparing
specific misclassified cases against the LLM's mismatches from the same
test rows.

## Artifacts
- Fine-tuned model: [utcompliance1/distilbert-imr-baseline](https://huggingface.co/utcompliance1/distilbert-imr-baseline)
  on the Hugging Face Hub (not stored in this repo — load directly via
  `transformers` rather than retraining)
- `train_transformer.py` — full training/evaluation pipeline
- `transformer_results.json` — accuracy, per-class metrics, confusion matrix
- `transformer_mismatches.json` — all misclassified test examples for
  qualitative review

## Jev exploration
The initial zero-shot Jev pilot scored **66.0% (132/200)** on 200 balanced cases
from the cleaned test split, with 100% API coverage. Evaluation took 28.3 seconds
and cost an estimated $0.004225. This smaller pilot is not a matched comparison
with the 4,000-case results above. See [Jev results and methodology](../jev-results/RESULTS.md).

The later focused-criteria prompt scored **69.5% (695/1,000)** on the first 1,000
test rows, extracted from a larger run stopped at the user's request. The prompt
was selected on separate training-split development data. This prefix sample is
also distinct from the canonical 4,000-case benchmark above.
