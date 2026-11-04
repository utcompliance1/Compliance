# Baseline results at a glance

These experiments predict whether an Independent Medical Review **upheld** an
insurer's denial or **overturned** it, using only the case description (`text`).
Accuracy means the percentage of scored cases whose predicted outcome matched
the recorded decision. Higher is better.

This folder summarizes existing saved experiments; no new evaluations were run.
Original code, predictions, and metrics remain in their existing locations.

## Main results

| Model | Accuracy | Correct / scored | Approach | Evaluation sample | Recorded API cost |
| --- | ---: | ---: | --- | --- | ---: |
| [DistilBERT](DistillBERT/README.md) | **74.65%** | 2,986 / 4,000 | Trained on 60,096 labeled cases | Balanced 4,000-case sample | Not measured; local compute required |
| [GPT-5](GPT/README.md) | **73.08%** | 2,923 / 4,000 | Four examples in the prompt; no training | Same 4,000-case sampling procedure | $3.5153 |
| [GPT-5-nano](GPT/README.md) | **60.58%** | 2,411 / 3,980 | Four examples in the prompt; no training | Same 4,000 requested; 20 unparsed | $0.1404 |
| [Jev 1.13.0](Jev/README.md) | **68.79%** | 4,757 / 6,915 | Focused criteria; no labeled examples or training | Completed portion of interrupted test run | $0.189201 |

All rows use `utcompliance1/imr-appeals-cleaned`. Costs are historical estimates
saved with each run, not current prices, and exclude development work and training.

## What this tells us

- **DistilBERT has the highest recorded accuracy**, 1.58 percentage points above
  GPT-5 (63 more correct predictions out of 4,000). However, its best checkpoint
  was selected using this same evaluation sample, so the result is not an
  untouched final test and does not establish a definitive model advantage.
- **GPT-5 is the strongest recorded GPT baseline.** Nano cost about 25 times less
  for the 4,000 requested cases, with lower accuracy. Counting its 20 unparsed
  answers as incorrect gives nano **60.28%** accuracy across all requested cases.
- **Jev scored about 69% at low recorded API cost**, but its sample differs from
  the GPT/DistilBERT sample. Its score cannot establish a controlled ranking.
  Its separately reported **69.50% on 1,000 cases** is a subset of the larger run,
  not an independent experiment.

## How to read the comparisons

The GPT/DistilBERT sample contains 2,000 cases per outcome, so always predicting
one outcome would score **50%**, not the full dataset's 53.90%. The completed Jev
sample has a **53.48%** majority baseline. Differences in class balance, case
selection, prompting, and supervised training matter when comparing results.

GPT and DistilBERT use the same seed-42 sampling code; their dataset revision
was not pinned and no shared sample manifest is saved with those summaries.
Jev pins its dataset revision and saves sample metadata. A stronger comparison
would use one fixed manifest for every model and select checkpoints and prompts
on separate development data before evaluating the final test once.

`full_text` is excluded because it includes reviewer reasoning that can reveal
the answer. These results measure agreement with recorded labels on these
samples; they do not measure performance on a new population.

## Detailed results and original evidence

| Folder | What you will find |
| --- | --- |
| [Jev](Jev/README.md) | Interrupted run, 1,000-case subset, pilot, confidence, prompt selection |
| [GPT](GPT/README.md) | Cleaned and original dataset runs, parsing, timing, cost, methodology |
| [DistillBERT](DistillBERT/README.md) | Training, checkpoint selection, class metrics, confusion matrix |

The requested folder spelling is `DistillBERT`; the model's actual name is
**DistilBERT**. Each detailed page links directly to the original metrics and code.
