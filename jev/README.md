# Jev IMR baseline

Predict `Upheld` versus `Overturned` on a small sample from the cleaned IMR
appeals test split. This uses Jev's native **Choice** question. The selected
`focused` prompt is zero-shot, with explicit outcome criteria and no model
training. Each case gets its own API request.

The focused prompt achieved **69.5% accuracy on 1,000 cases** extracted from an
interrupted larger run. The original 200-case basic prompt scored 66.0% on a
different sample. See [measured results and sampling details](RESULTS.md).

## Setup and run

From the repository root:

```bash
python3 -m venv jev/.venv
source jev/.venv/bin/activate
pip install -r jev/requirements.txt
cp jev/.env.example jev/.env
# Edit jev/.env: TYPESAFE_API_KEY=<your TypeSafe key>
python jev/baseline_eval_jev.py --n 200
```

For every case in the cleaned held-out test split, with its natural class balance:

```bash
python jev/baseline_eval_jev.py --full-test --prompt focused --workers 4 \
  --output-dir jev/full-test-focused
```

`--full-test` and `--n` are mutually exclusive. The full split has 9,846 cases
(5,307 Upheld and 4,539 Overturned); its majority baseline is 53.9%, rather than
the balanced pilot's 50%. Four workers make concurrent calls through the same
SDK connection pool. The first call runs alone to validate the integration.

Create a key in the [TypeSafe console](https://console.typesafe.ai/). Environment
variables take precedence over `jev/.env`. A root `.env` is also supported for
compatibility with the original run. The key is never saved in results;
`jev/.env` and `jev/.venv` are excluded from Git.

For a 100-case run or an additional run, use a separate directory:

```bash
python jev/baseline_eval_jev.py --n 100 --output-dir jev/run-100
```

To download the data and inspect the sample without calling Jev:

```bash
python jev/baseline_eval_jev.py --n 200 --dry-run
```

## Methodology

- Dataset: `utcompliance1/imr-appeals-cleaned`, held-out `test` split (9,846 cases).
  Default revision is pinned to `0abb1e862c71e2b0558b560fceedab3d528759f7`.
- Default: 200 cases, 100 per label, sampled without replacement with seed 42.
  `--n` accepts even sizes. Sampling uses the same pandas procedure as the
  existing cleaned LLM and transformer baselines, with a smaller sample size.
  The sample manifest preserves original test indices and dataset row IDs.
- `--full-test` includes every test row exactly once without rebalancing.
- Model: `jev-1.13.0`, pinned for reproducibility; override with `--model`.
- Input: only `text`. `full_text` contains reviewer reasoning and can leak the
  outcome. The true decision, row IDs, and other case metadata are never sent.
- Prompt: zero-shot, one question asking for the appeal outcome with descriptions
  of the two labels. Its exact wording is saved with the run. Prompts live in
  [prompts.py](prompts.py): `--prompt basic` reproduces the initial short prompt;
  `--prompt structured` supplies explicit review factors, interpretation rules,
  and supporting evidence/boundaries for each outcome; `--prompt focused` uses
  a concise rubric with task-specific criteria (the selected default).
  `--prompt focused-fewshot` adds the four original vetted GPT prompt examples.
- Majority-class baseline on this balanced sample: 50%. Dataset prevalence is
  different, so this pilot's accuracy does not directly estimate accuracy on an
  unbalanced production population.
- Reports accuracy over successfully scored cases, a Wilson 95% interval, API
  coverage, and correct/all-requested accuracy (failures count as incorrect).
  Confusion matrix and per-class metrics use scored cases; rows are true labels,
  columns are predictions. No confidence threshold excludes difficult cases.
- Latency measures the client call, including network time and SDK retries;
  the median and p95 use successful cases. Total time includes failed calls.
- Estimated cost uses returned input tokens at $0.042 per million; output tokens
  are free. Failed calls that do not return usage can add unmeasured cost.

The prior GPT results used four labeled examples and 4,000 cases; the prior
DistilBERT result used supervised training. Comparing those headline scores with
this smaller zero-shot pilot is descriptive. A controlled model comparison
requires evaluating all models on the same saved sample and specifying prompts.

## Criteria and prompt development

The structured prompt identifies the requested care, clinical severity and
functional impairment, prior treatment response and alternatives, the stated
denial rationale, medical necessity for the requested setting/intensity/duration,
and evidence for investigational services. Coverage terms are applied only if
supplied in the description. The rubric explains which facts support each
outcome while guarding against equating the initial denial with the final review,
or treating a missing detail as evidence against the patient.

These are contextual evaluation factors rather than hard coverage rules.
Every case still gets one of the same two labels, including ambiguous cases.

To compare the fixed basic and structured prompts on a development sample:

```bash
python jev/compare_prompts.py --n 100
```

Development uses 100 balanced **training-split** cases, seed 17, after excluding
normalized exact-text overlaps with the entire test split. Both candidates and
the selection rule are written to `development_plan.json` before any calls.
Highest accuracy with complete coverage wins; ties favor the first listed prompt.
Few-shot examples are also checked for exact-text overlap with development and
test cases. The selected prompt is fixed before further test evaluation.

The first comparison tested basic and detailed structured criteria. The second
tested focused criteria and focused criteria with four examples on the same
development cases. Both rounds and their predictions are preserved:

| Prompt | Development accuracy |
| --- | ---: |
| Original basic | 68/100 (68%) |
| Detailed structured rubric | 66/100 (66%) |
| Focused criteria | 69/100 (69%) |
| Focused criteria + four examples | 69/100 (69%) |

The `focused` prompt was selected; the tie favors its zero-shot form. These
differences are small and do not establish a reliable improvement. The choice,
scores, and sample hash were frozen in [prompt_selection.json](prompt_selection.json)
before the full test run. No test results were used to change the selected prompt.
The full test split includes the 200 cases already evaluated in the initial pilot.

The full evaluation was stopped at the user's request after 6,915 completed
cases. `run-1000/` contains the first 1,000 test rows and their existing
predictions, with no new API calls. This saved prefix is distinct from the
runner's balanced random `--n 1000` sampling.

```bash
python jev/compare_prompts.py --n 100 --prompts focused focused-fewshot \
  --output-dir jev/prompt-development-focused
```

The comparison script refuses to reuse an output directory. Use
`--output-dir jev/dev-rerun` for an additional development experiment.

## Outputs

Default destination: `jev/output/`.

| File | Contents |
| --- | --- |
| `jev_sample.json` | Dataset/model/prompt settings, sample hash, IDs, case descriptions and labels |
| `jev_predictions.jsonl` | One record per attempted case, flushed immediately: prediction, probabilities, confidence, usage, latency, or error |
| `baseline_results_jev.json` | Summary metrics and complete/incomplete/interrupted status |

A dry run writes only the sample manifest and never reports measured accuracy.
Existing live results are protected against overwrite; choose another
`--output-dir` for a rerun. Ctrl+C saves a partial summary. Authentication,
billing, model, or request validation errors stop the run. The first case is a
live integration check: if it fails, the remaining cases are left unattempted.
SDK retries handle transient failures with backoff.

## Verification

```bash
python -m unittest discover -s jev -p 'test_*.py' -v
```

These offline checks exercise the real SDK with a simulated HTTP response, verify
text-only requests, deterministic balanced sampling, failure accounting, and
stopping on authentication errors, plus concurrent per-case saving and full-split
class prevalence. They do not measure model performance.

## API references

- [Official HTTP API](https://docs.typesafe.ai/api)
- [Models and pricing](https://docs.typesafe.ai/models)
- [Python SDK](https://docs.typesafe.ai/sdk/python)
