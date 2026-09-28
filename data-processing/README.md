# IMR Appeals Dataset — Cleaning Pipeline

Cleans the [Persius/imr-appeals](https://huggingface.co/datasets/Persius/imr-appeals)
dataset (73,987 health insurance appeal case summaries): deduplicates,
quarantines cases where near-identical text has conflicting labels, and
publishes a cleaned version. See `clean_imr_appeals.py` for the runnable
pipeline and `case_studies.md` for real example rows from each bucket.

## Quick start

```bash
pip install datasets huggingface_hub scikit-learn pandas pyarrow numpy

# Run locally, no upload
python3 clean_imr_appeals.py --source_repo Persius/imr-appeals --out_dir ./cleaned_output

# Run + push to HuggingFace
hf auth login
python3 clean_imr_appeals.py --source_repo Persius/imr-appeals --out_dir ./cleaned_output \
    --push --target_repo <account>/imr-appeals-cleaned
```

## How near-duplicate detection works

The corpus is heavily templated — most rows read like "A NN-year-old
[sex] enrollee has requested [procedure] for treatment of [condition]" —
so exact string matching alone misses cases that are the same appeal
reworded by a word or two (typos, "requested for" vs "requested coverage
for", added/dropped articles).

1. **Vectorize.** Each row's `text` is vectorized with TF-IDF over word
   1-2 grams (`min_df=2`, `max_df=0.5`, `sublinear_tf=True`). `max_df=0.5`
   downweights the boilerplate phrases (e.g. "has requested", "the health
   insurer has denied") that appear in more than half the corpus, so
   clustering is driven more by the specific procedure/condition terms
   than by shared sentence scaffolding.
2. **Find neighbors.** `sklearn.neighbors.NearestNeighbors` with cosine
   distance finds each row's `k=15` nearest neighbors by TF-IDF vector.
   This keeps the algorithm roughly O(n log n) instead of the O(n²) a
   full pairwise cosine-similarity matrix would need — required to stay
   tractable at ~74k rows.
3. **Cluster.** A union-find structure links row `i` and `j` whenever
   `cosine_similarity(i, j) >= sim_threshold`. Transitivity means a chain
   of near-identical rows all end up in one cluster even if the two most
   distant members in the chain aren't directly similar to each other.

## Why 0.93 as the similarity threshold

0.93 was chosen empirically after spot-checking `audit_near_dup_clusters.csv`
(the pipeline's dump of the 30 largest near-dup clusters, meant for manual
review). At lower thresholds (tested down to ~0.85), the clustering started
**false-merging genuinely unrelated cases** that only share boilerplate
phrasing and procedure names — e.g. two different patients requesting the
same common procedure (like breast tomosynthesis) but for different
underlying conditions or reasoning, pulled into the same cluster purely
because the `max_df` downweighting isn't perfect at catching every
templated phrase. This was confirmed empirically, not just theoretically:
lowering the threshold visibly pulled in rows whose `full_text` reviewer
reasoning had nothing in common.

Raising the threshold above 0.93 would reduce false merges further but
starts missing real near-duplicates (rows differing only by a couple of
words) — 0.93 was the balance point found by manually inspecting cluster
contents at a few threshold values. **This is a tunable, not a fixed
constant** — anyone re-running this pipeline on a different corpus (or a
future version of this one) should re-spot-check
`audit_near_dup_clusters.csv` rather than trusting 0.93 by default.

## Minority-outlier vs. whole-cluster quarantine rule

Once near-dup clusters exist, each cluster is checked for disagreement on
`decision`, `appeal_type`, or `sufficiency_id` (see
`flag_label_conflicts()` in `clean_imr_appeals.py`). Two tiers:

- **Minority outlier** (disagreeing rows are <5% of the cluster): only
  those outlier rows get quarantined (reason suffixed `_minority_outlier`);
  the majority-label rows are left in the clean set. Rationale: a
  22-row-out-of-222 disagreement is more consistent with a handful of
  mislabeled/edge-case entries than with the text itself being
  unpredictive — see Cluster #1152 in `case_studies.md`.
- **Substantial split** (≥5%, closer to even): the **entire cluster** is
  quarantined. Rationale: if the split is closer to 50/50, the input text
  genuinely can't predict either label for that fact pattern, so keeping
  either side in the "clean" set would teach a false signal — see
  Cluster #22 in `case_studies.md`, a word-for-word-identical pair with
  opposite decisions.

The 5% threshold is a parameter (`minority_threshold` in
`flag_label_conflicts()`), not something derived from the data — chosen
as a reasonable cutoff for "handful of outliers" vs. "real ambiguity."

## Known limitations

- **`full_text` doesn't always contain `text` verbatim.** Only 34.2% of
  rows (48,685 do not) have `full_text` literally containing the `text`
  field as a substring. Not yet root-caused — could be reformatting
  (whitespace/punctuation differences the normalization doesn't catch) or
  a real data quality issue where `text` and `full_text` were sourced
  independently. **Open item — needs investigation before being fully
  trusted as "just reformatting."**
- **Boilerplate-collision risk.** Because this corpus redacts
  patient-identifying details, many genuinely different real cases
  collapse to identical or near-identical `text` (e.g. ~180 different
  patients all reading "An enrollee has requested breast tomosynthesis...").
  The near-dup clustering can't fully distinguish "same case duplicated"
  from "different cases, same redacted template" — the minority/majority
  quarantine split is a heuristic to manage this, not a guarantee. Always
  spot-check `audit_near_dup_clusters.csv` before trusting quarantine
  output at face value on a re-run.
- **Threshold sensitivity.** The 0.93 similarity threshold and 5% minority
  threshold were both tuned by hand on this specific corpus. Re-running on
  a different snapshot of the data (or a different dataset entirely) may
  need both re-tuned.
- **Cross-split leakage is best-effort.** Exact-match leakage (2 rows) and
  near-dup leakage (106 rows) between `train`/`test` are both checked, but
  a near-dup leak still depends on the same TF-IDF clustering above, so it
  inherits the same collision/miss risks.

## Output files

| File | Contents |
|---|---|
| `train_clean.parquet` / `test_clean.parquet` | Final cleaned splits |
| `quarantined.parquet` | Rows pulled for conflicting labels or cross-split leakage, with `quarantine_reason` |
| `dropped_exact_duplicates.parquet` | Exact-duplicate rows removed (first occurrence kept in clean set) — kept as a record, not silently discarded |
| `audit_near_dup_clusters.csv` | The 30 largest near-dup clusters, for manual spot-checking |
| `cleaning_report.md` | Full run statistics |

## Dataset statistics

Of 73,987 total rows:

| Category | Rows | % |
|---|---|---|
| Exact duplicates removed | 2,769 | 3.74% |
| Near-duplicates kept (legitimate distinct cases) | 5,647 | 7.63% |
| Near-duplicates quarantined (conflicting labels) | 1,206 | 1.63% |

See `case_studies.md` for real example rows behind each of these numbers.
