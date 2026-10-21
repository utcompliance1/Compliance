# Original xverifier implementation: limited GPT-5 evaluation

The limited run found no consistent improvement from flipping GPT-5 verdicts.
Across five seeds, mean evaluation accuracy changed from 70.03% to 69.33% on
uncleaned cases and from 70.04% to 70.17% on cleaned cases. These are descriptive
pilot results, not full-dataset estimates or evidence that cleaning improves accuracy.

## Accuracy

| Dataset | Input rows | Evaluation rows per seed | GPT-5 accuracy, mean ± SD | After xverifier, mean ± SD | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uncleaned | 4,884 | 360–384 | 70.03% ± 2.28 pp | 69.33% ± 2.04 pp | −0.70 pp |
| Cleaned / filtered | 3,592 | 290–297 | 70.04% ± 4.23 pp | 70.17% ± 4.69 pp | +0.14 pp |

SD is variation across seeds, not a confidence interval. Each input is split into
verifier training, calibration, and evaluation by the original group-aware draw
procedure; input row counts are not evaluation denominators. The five evaluation
sets overlap, so they must not be summed as independent observations.

## Automatic decisions versus review

Coverage averages include seeds that review every case. Observed error rates
average only seeds with a certified cutoff and automatic decisions.

| Target error level | Uncleaned coverage | Uncleaned observed error | Seeds with cutoff | Cleaned coverage | Cleaned observed error | Seeds with cutoff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 25% | 59.47% | 21.13% | 5/5 | 36.00% | 22.04% | 3/5 |
| 20% | 6.67% | 13.92% | 2/5 | 5.37% | 24.05% | 1/5 |
| 15% | 0% | — | 0/5 | 0% | — | 0/5 |
| 10% | 0% | — | 0/5 | 0% | — | 0/5 |
| 5% | 0% | — | 0/5 | 0% | — | 0/5 |

The cleaned run's observed error at the 20% target exceeded that target on its
evaluation slice. The configured confidence is 95%; it is not a guarantee that
every finite evaluation slice meets its target. Small samples, near-duplicate
dependence, and this pilot's selection also limit interpretation. No cutoff
qualified at targets of 15% or below; all those cases were deferred to review.

## Inputs and scope

- Original implementation: [thiago-cs-ai/xverifier at 241fcf3](https://github.com/thiago-cs-ai/xverifier/tree/241fcf3e068771fd23522b582f72e4e87d07fce8), installed without tracked-source changes. Nothing was pushed to that repository.
- Judge: `gpt-5-2025-08-07`, the snapshot returned for the existing baseline's
  `gpt-5` model. Same four-example prompt, `reasoning_effort=minimal`, and
  `max_completion_tokens=150` as `week1-results/baseline_eval_cleaned.py`.
  The saved prompt is in `gpt_prompt.json`. The parser requires one unambiguous
  whole-word label; all 3,727 saved unique predictions parsed successfully.
- Uncleaned: `Persius/imr-appeals`, revision
  `c68b70e0faf74cd2b5731ea02ad7ba7d99a11b8b`.
- Cleaned: `utcompliance1/imr-appeals-cleaned`, revision
  `0abb1e862c71e2b0558b560fceedab3d528759f7`. The filtered pilot is the
  intersection of the uncleaned pilot with that dataset's retained rows.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, pinned revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, normalized float32, 384
  dimensions, model default 256-token truncation, computed on Apple MPS.
  Longer text can be truncated for embeddings; GPT receives the full `text`.
- Only `text` enters GPT and the embedding model. `full_text`, labels, and
  other case metadata never enter either request. Labels enter only supervised
  verifier fitting/calibration and offline evaluation, as required by xverifier.

The user stopped the full collection and requested a limited run. This pilot
uses all predictions saved at that point, with no additional GPT calls. Collection
started in original train-row order and finished requests concurrently. Joining
saved exact-text hashes back to eligible original rows yields 4,884 uncleaned
rows and 3,592 cleaned rows. Repeated text reuses one GPT prediction, including
duplicates with disagreeing labels. This explains why 3,727 API predictions
cover more original rows. All pilot cases come from the source **train** split;
xverifier creates new held-out evaluation slices within this pilot. This is
**not** evaluation on the dataset's canonical test split and is not a random
sample of the whole dataset. API latency/completion order can affect inclusion.

The original 0.1% GPT prompt-development sample is reconstructed with seed 42.
Those cases, the four few-shot examples, normalized exact matches, and their
near-duplicate groups are excluded before forming verifier inputs. Group IDs
come from the cleaned dataset's annotations, including its quarantined and
dropped-duplicate records. All 73,987 original row IDs, texts, decisions, and
source splits were checked against those annotations. The uncleaned sample
retains duplicates and quarantined/conflicting-label cases apart from these
prompt-overlap exclusions; it is not silently denoised.

## Verifier settings

The unchanged upstream `runner.run` is called once per dataset with seeds
`[42, 123, 456, 789, 1024]`. Settings are frozen before viewing verifier outputs:
`draw` mode; calibration fraction 0.274; evaluation fraction 0.082; linear probe
with C=1, max 256 PCA components, minimum 40 rows per verdict; five-fold
Venn–Abers calibration; flip threshold 0.5; confidence 0.95. Near-duplicate
groups stay together during splitting and cross-fitting.

Each dataset draws its splits separately using the same seeds. Removing cases
changes the groups and draw, so evaluation cases are **not matched** between
cleaned and uncleaned runs. No causal cleaning-effect claim or paired comparison
is justified. The original full-run plan is retained in `run_plan.json` for
provenance; `limited_sample_plan.json` supersedes its full-run scope and split
policy. No settings were chosen using verifier evaluation accuracy.

## Usage and artifacts

Saved requests used 2,101,110 input and 46,844 output tokens. At the prior
baseline's rates ($1.25/M input, $10/M output), that is **$3.09**. This is a
historical-rate estimate, not an invoice; it excludes the integration check and
in-flight requests that were interrupted before their results were saved.

- `limited_summary.json`: all accuracy, flip, coverage, and error statistics.
- `cleaned/` and `uncleaned/`: original per-seed records, native aggregate
  output, and per-seed evaluation predictions.
- `*_sample.csv`: row IDs, source splits, duplicate groups, labels, predictions,
  and text hashes; no case text.
- `gpt_predictions.jsonl.gz`: saved API results, usage, model IDs, and exact-text
  hashes; no API keys or case text.
- `data_manifest.json`, YAML files, and `verification.json`: input/source
  provenance and checks. Large embedding/row files are excluded from Git.
- `prepare_data.py`, `collect_gpt.py`, `run_limited.py`: preparation wrappers;
  verifier mathematics come exclusively from the upstream package.

To replay without more GPT requests, install `requirements-lock.txt` in Python
3.13, then run from this directory in a new checkout without the generated
`cleaned/` and `uncleaned/` output folders:

```bash
python prepare_data.py --work /tmp/xverifier-replay --device mps
python -c 'import gzip,pathlib; pathlib.Path("/tmp/xverifier-replay/gpt_predictions.jsonl").write_bytes(gzip.decompress(pathlib.Path("gpt_predictions.jsonl.gz").read_bytes()))'
python run_limited.py --work /tmp/xverifier-replay
```

Use CPU instead of MPS if needed; floating-point differences can change
embeddings and downstream results. Original input hashes identify this measured
run. The upstream test suite passed all 78 tests before evaluation.
