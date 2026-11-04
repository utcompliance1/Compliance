# DistilBERT baseline

**Fine-tuned DistilBERT correctly classified 2,986 of 4,000 cases (74.65%).**
That is 63 more correct predictions than GPT-5, a **1.58 percentage-point**
difference using the saved, unrounded scores. The same evaluation cases were
also used to select the best training checkpoint, which limits the comparison.

[Back to comparison](../README.md)

The directory uses the requested spelling `DistillBERT`; the actual model name
is `distilbert-base-uncased` (DistilBERT).

## Training and evaluation

| Setting | Recorded configuration |
| --- | --- |
| Base model | `distilbert-base-uncased`, 66M parameters |
| Task | Two-class classification: Upheld / Overturned |
| Training data | Cleaned IMR appeals `train` split, 60,096 labeled cases |
| Evaluation data | Cleaned `test` split, balanced 4,000-case sample, seed 42 |
| Input | `text` only; `full_text` excluded |
| Tokenization | Maximum 256 tokens, truncation and fixed-length padding |
| Training / evaluation batch size | 16 / 32 |
| Training duration | Three epochs; approximately 70 minutes reported |
| Hardware | Google Colab T4 GPU |
| Checkpoint selection | Highest evaluation accuracy; epoch 2 selected |

The GPT cleaned runner and transformer runner use the same sampling procedure.
The dataset revision and dependency versions are not pinned in these scripts.
Long descriptions are truncated after 256 tokens, which can discard relevant
facts. GPT and Jev do not use this same tokenizer cutoff.

## Training dynamics

| Epoch | Training loss | Evaluation loss | Evaluation accuracy |
| --- | ---: | ---: | ---: |
| 1 | 0.501 | 0.534 | 74.05% |
| 2 | 0.358 | 0.567 | **74.65%** |
| 3 | 0.220 | 0.766 | 74.30% |

Loss measures prediction error, with lower values preferred. Training loss
continued to improve while evaluation loss rose, consistent with overfitting.
The code selected epoch 2 by accuracy, not epoch 1 by loss.

**Evaluation caveat:** the runner passes the 4,000-case test sample as
`eval_dataset`, selects the best epoch on it, and reports final accuracy on it
again. This sample was excluded from gradient training but was used for model
selection. A separate development split and untouched test evaluation are
needed to assess the final model without this selection effect. The 1.58-point
gap alone does not establish statistical significance or a general advantage.

## Errors and class metrics

Rows are actual labels; columns are predicted labels.

| Actual outcome | Predicted Upheld | Predicted Overturned | Total |
| --- | ---: | ---: | ---: |
| Upheld | 1,531 | 469 | 2,000 |
| Overturned | 545 | 1,455 | 2,000 |
| Total | 2,076 | 1,924 | 4,000 |

There were **1,014 errors**. The model missed 545 actual Overturned cases and
469 actual Upheld cases. Recall was 76.55% for Upheld and 72.75% for Overturned,
calculated from the matrix. Its balanced accuracy is also 74.65%, because the
two classes have equal support.

The saved report rounds class metrics to two decimal places:

| Class | Precision | Recall | F1 | Cases |
| --- | ---: | ---: | ---: | ---: |
| Upheld | 0.74 | 0.77 | 0.75 | 2,000 |
| Overturned | 0.76 | 0.73 | 0.74 | 2,000 |

Precision is the fraction of predictions of a label that are correct; recall
is the fraction of actual cases of that label found; F1 combines both.

## Cost and reproduction

No measured training cost, inference cost, or inference latency is saved in the
metrics. Running a local classifier avoids per-request model API charges, but
still consumes compute. It should not be described as measured free inference.

The [training script](../../week2-results/train_transformer.py) contains the
training and evaluation workflow and dependency instructions. Run it in a fresh
working directory with a suitable GPU: it writes `distilbert-imr/`,
`transformer_results.json`, and `transformer_mismatches.json` relative to the
working directory. Its `best_epoch` output field is hard-coded to 2, so verify
the actual selected checkpoint when rerunning. No retraining was performed here.

## Original artifacts

- [Saved metrics](../../week2-results/transformer_results.json)
- [Misclassified examples](../../week2-results/transformer_mismatches.json)
- [Original training report](../../week2-results/RESULTS.md)

The original report identifies the published checkpoint as
`utcompliance1/distilbert-imr-baseline`; weights are not stored in this repository.
