# GPT baselines

**GPT-5 scored 73.08% on the cleaned 4,000-case sample. GPT-5-nano scored
60.58% on its 3,980 parseable answers, or 60.28% across all 4,000 requests.**

[Back to comparison](../README.md)

## Main results: cleaned dataset

| Model | Correct / scored | Accuracy | Unparsed | Elapsed time | Estimated API cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-5 | 2,923 / 4,000 | 73.08% | 0 | 6.09 min | $3.5153 |
| GPT-5-nano | 2,411 / 3,980 | 60.58% | 20 | 4.06 min | $0.1404 |

Source: [cleaned-run metrics](../../week1-results/baseline_results_cleaned.json).
Correct counts are derived from saved accuracy × scored cases.
GPT-5-mini was not evaluated on the cleaned dataset.

GPT-5's scored accuracy exceeds nano by 12.50 percentage points, at about 25
times the total API cost. Recorded cost per 1,000 requested cases is about
$0.879 for GPT-5 and $0.035 for nano. These are historical token-cost estimates,
not current quotes, and do not include prompt development.

## Method and scoring

- Dataset: `utcompliance1/imr-appeals-cleaned`, `test` split (9,846 rows).
- Sample: 2,000 cases per label, pandas sampling with seed 42, followed by a
  seeded shuffle. The sample's majority baseline is **50%**.
- Prompt: four manually selected labeled examples (two per outcome), followed
  by the current case. No supervised model training was performed.
- Input: only `text`; reviewer reasoning in `full_text` is excluded.
- Requests: `reasoning_effort="minimal"`, maximum 150 completion tokens,
  default temperature, up to 10 concurrent workers.
- Parsing: extract the first `UPHELD` or `OVERTURNED` match from the response.
  Responses with neither are excluded from the saved `accuracy` denominator.

An unparsed answer is a response without an extractable label, not necessarily
an API failure. Counting all requested cases gives GPT-5 73.08% and nano 60.28%.
The saved files do not provide per-class metrics or per-case GPT predictions;
the runner collects mismatches in memory but omits them from the summary JSON.

The prompt was chosen in the earlier experiment; it was not reselected on the
cleaned dataset. The cleaned runner constructs a small development sample but
does not use it to compare prompts in its main execution path.

## Earlier results: original dataset

These results used `Persius/imr-appeals`, a 6,406-case stratified sample from its
`train` split, with an approximately 53% majority baseline. They are historical
context and should not be pooled with the cleaned test results.

| Model | Saved scored accuracy | Requested | Unparsed | Elapsed time | Estimated API cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-5 | 71.85% | 6,406 | 0 | 10.01 min | $5.5902 |
| GPT-5-mini | 66.69% | 6,406 | 191 | 8.82 min | $1.1183 |
| GPT-5-nano | 59.17% | 6,406 | 34 | 7.18 min | $0.2234 |

Sources: [original metrics](../../week1-results/baseline_results.json) and
[original report](../../week1-results/RESULTS.md). That report records a small
development comparison of 61.3% zero-shot versus 66.7% few-shot, motivating the
four-example prompt. Different data and samples prevent attributing later score
changes solely to dataset cleaning.

## Comparison limits and reproduction

DistilBERT uses the same 4,000-case sampling procedure, but also used that sample
to select its best checkpoint. Jev's larger result uses different cases. These
differences limit claims about which approach generalizes best.

The GPT scripts do not pin the dataset revision or dependency versions, so a
future run may differ. Setup and exact prompts are in the
[cleaned runner](../../week1-results/baseline_eval_cleaned.py) and
[original runner](../../week1-results/baseline_eval.py).

For a deliberate rerun, use a separate checkout or change `OUTPUT_FILE` first:
the scripts write directly over their result JSON. Configure `OPENAI_API_KEY`,
install the root requirements plus `python-dotenv`, then run the relevant script.
This incurs API charges. No API calls were made for this documentation.
