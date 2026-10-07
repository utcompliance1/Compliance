# Week 2 Results: Validation-Selected DistilBERT

## Summary
Last week's DistilBERT result (74.65%) used the same 4,000 test rows both to pick
the checkpoint and to report accuracy, so it was optimistic. This week the model
was retrained with the checkpoint chosen on a validation slice carved from `train`
only, and the test set was scored once after the choice was frozen.

- **Headline: 75.43% accuracy on the full 9,846-row test split** (95% CI 74.57-76.27%,
  natural class prevalence, majority-class baseline 53.9%).
- **Balanced 4,000-row sample (same rows as the gpt-5 / gpt-5-nano runs): 74.52%**
  (95% CI 73.15-75.85%).
- **The selection fix made no measurable difference.** Last week's test-selected model,
  rescored on the same rows, gets 74.60% on the balanced sample and 75.21% on the full
  test split. The intervals overlap almost completely.
- Rows whose text states the outcome have a negligible effect on the result (see below).

## Setup
- Dataset: `utcompliance1/imr-appeals-cleaned`. Input is the `text` field only.
- Validation split: carved from `train` (60,096 rows) by `near_dup_group_id`, with each
  `-1` row treated as its own group, so no near-duplicate group spans train and validation.
  Result: 52,877 train / 7,219 validation (12% of groups, seed 42).
  Class balance: train 28,461 Upheld / 24,416 Overturned; validation 3,754 / 3,465.
- Test: the full cleaned `test` split (9,846 rows), plus the 4,000-row stratified sample
  (2,000 per class, seed 42) used for the LLM comparison. The sample is drawn with the same
  function and seed as `baseline_eval_cleaned.py`.

### Truncation check
On a 2,000-row sample of train, mean length is 141 tokens, median 110, 95th percentile 349,
max 477. **17.3% of cases exceed 256 tokens**, so the max length was raised from 256 to 512,
which removes truncation entirely.

### Rows whose text states the outcome
Some `text` fields contain the outcome itself (for example the literal phrase
"The denial is: Overturned"). Matching the overturn/uphold word families gave:
589 train rows (1.1%), 91 validation rows (1.3%), 115 test rows (1.2%).
These rows were kept in training. Test accuracy is reported with and without them.

## Training
`distilbert-base-uncased` with a 2-class head. Learning rate 2e-5 (was 5e-5), weight decay
0.01 (was 0), 500 warmup steps (was none), batch size 16, up to 10 epochs, early stopping on
**validation loss** with patience 2. fp16 and dynamic padding, one T4 GPU, about 31 minutes.

| Epoch | Val loss | Val accuracy |
|---|---|---|
| 1 | 0.5034 | 75.52% |
| 2 | **0.4728** | 76.53% |
| 3 | 0.5069 | 77.30% |
| 4 | 0.6189 | 76.81% |

Validation loss was lowest at epoch 2 and rose afterward, so early stopping ended the run
after epoch 4 and kept the **epoch-2 checkpoint (checkpoint-6610)**. Epoch 3 had higher
validation accuracy, but selection followed the pre-specified rule (validation loss).
Validation loss did not rise inside the first epoch, so the second-pass options (freezing
lower layers, higher dropout) were not needed.

## Test results (single evaluation, after the checkpoint was frozen)
Wilson 95% intervals. "No shortcut" excludes the 115 test rows described above.

| Subset | n | Validation-selected (new) | Test-selected (old, rescored) | Majority baseline |
|---|---|---|---|---|
| Full test, all rows | 9,846 | **75.43%** [74.57, 76.27] | 75.21% [74.35, 76.05] | 53.90% |
| Balanced 4,000, all rows | 4,000 | 74.52% [73.15, 75.85] | 74.60% [73.23, 75.92] | 50.00% |

The old model was rescored at its original 256-token setting. Last week it was reported at
**74.65%** on the balanced sample. That figure is **test-selected** and is kept here only for
reference. Rescoring today gives 74.60%, a difference of two predictions, attributable to
fp16 and dynamic padding.

### Effect of the shortcut rows
Dropping them lowers full-test accuracy by about 0.2 points (75.43% to 75.19%). By
subtraction the model gets roughly 95% of those 115 rows right, against about 75% elsewhere,
so it does use the literal outcome text when it is present. The rows are too few to move the
headline. The ~95% figure is derived from rounded accuracies and a small sample, and should
be recomputed exactly from `test_predictions.csv` before it is quoted.

## Comparison with the LLM baselines (balanced 4,000 rows)
| Model | Accuracy |
|---|---|
| DistilBERT, validation-selected (new) | 74.52% [73.15, 75.85] |
| DistilBERT, test-selected (old, rescored) | 74.60% [73.23, 75.92] |
| gpt-5 (few-shot) | 73.1% (about [71.7, 74.4]) |
| gpt-5-nano (few-shot) | 60.6% (3,980 scored, 20 unparsed) |

DistilBERT is about 1.4 points ahead of gpt-5, but the intervals overlap and the gpt-5 run
saved only aggregate accuracy, not per-row predictions. **No claim of a difference from
gpt-5 is made yet.** A paired comparison (such as McNemar's test) needs both prediction
vectors, which means rerunning gpt-5 on the same 4,000 rows with predictions saved.

## Caveats
- This week's training used fp16 and dynamic padding, which differ from last week's fp32
  and fixed padding. These changes were made to finish after Colab usage limits interrupted
  an earlier run, and they were not found to affect results (the old model reproduced to
  within two predictions).
- An earlier run with the same configuration was lost to a session reset. Only the rerun is
  reported here, and its validation curve has the same shape (lowest at epoch 2, rising after).
- Confidence intervals treat rows as independent. Near-duplicate structure in the test set
  could make them slightly optimistic.

## Artifacts
- `train_val_selected.py`: group-aware split, training with early stopping
- `eval_test_once.py`: single test evaluation, writes the files below
- `test_metrics.json`: accuracy, intervals, baselines, and per-class recall for all subsets
- `test_predictions.csv`: prediction, class probabilities, and logits for every test row, for
  both models (input for calibration/ECE analysis)
- `training_log.json`: full training and validation log
- Model weights: [utcompliance1/distilbert-imr-val-selected](https://huggingface.co/utcompliance1/distilbert-imr-val-selected) on the Hugging Face Hub

## Open items
1. Whether to also drop shortcut rows from the training set (the effect on test accuracy
   suggests it is probably not needed).
2. Rerun gpt-5 on the 4,000-row sample with per-row predictions saved, to allow a paired test.
3. Calibration (ECE) and the follow-up metric can be computed from the saved probabilities.
