# Case Studies — Real Examples From the Data

These are literal rows pulled from the cleaning pipeline's output files
(`dropped_exact_duplicates.parquet`, `audit_near_dup_clusters.csv`,
`quarantined.parquet`), not general descriptions. Each shows the real
`text` and label values, with a note on why it belongs in that bucket.

---

## 1. Similar cases that were removed (exact duplicates)

**Example A** — text_hash `a6e5782...`, 2 rows in `train`, both:
> "A 66-year-old male enrollee has requested reimbursement for the
> DecisionDx-Melanoma Gene Expression Assay provided on 10/12/17. The
> Health Insurer has denied this request indicating that the testing at
> issue was considered investigational for evaluation of the enrollees
> melanoma."

Both rows: `Upheld` / `Experimental` / `sufficiency_id=1`. Word-for-word
identical text, identical labels, and a specific date + named assay make
this look like the same case entered into the source dataset twice rather
than two different patients. One copy kept, the other dropped.

**Example B** — text_hash `95be867...`, 2 rows in `train`, both:
> "A 62-year-old female enrollee has requested reimbursement for digital
> breast tomosynthesis performed on 8/02/16. The Health Insurer has
> denied this request indicating that the service at issue was
> considered investigational for evaluation of the enrollee, who was
> asymptomatic."

Both rows: `Overturned` / `Experimental` / `sufficiency_id=1`. Same
reasoning as Example A — specific procedure date, identical wording,
identical outcome. Genuine duplicate, not a boilerplate collision.

---

## 2. Similar cases that should NOT be removed (legitimate distinct cases)

**Cluster #656** — 15 rows, near-identical short `text`:
> "An enrollee has requested reimbursement for a digital breast
> tomosynthesis for evaluation of the enrollees medical condition."

All labeled `Overturned` / `Experimental/Investigational`. Kept as
distinct (not merged) because this is generic, redacted boilerplate that
many different real patients requesting the same common procedure would
independently produce — it's the *redaction*, not the case, that makes
them look alike. Confirmed by checking `full_text`: the reviewer findings
differ across rows (e.g. one row's `full_text` says "The physician
reviewer found that Tomosy..." while another says "All three physician
reviewers found that t..." — different number of reviewers, different
reasoning), proving these are different real appeals, not one case
duplicated.

**Cluster #1147** — 20 rows, near-identical `text`:
> "An enrollee has requested reimbursement for advanced lipoprotein
> testing."

All labeled `Upheld` / `Experimental/Investigational`, consistent
`sufficiency_id`. Same pattern: a common, short, heavily-redacted request
description shared by many distinct real cases with the same outcome.

**Cluster #950** — 17 rows, near-identical `text` (minor wording variants,
e.g. "male" vs "female" enrollee, "requested for" vs "requested coverage
for"):
> "The parent of a two-year-old male enrollee has requested for speech
> therapy for treatment of the enrollees speech delay."

All labeled `Overturned` / `Medical Necessity`. `full_text` findings again
differ per row (different cited literature, e.g. "Schirmer and..." appears
in only one variant), confirming distinct underlying cases sharing
template language, not duplicates.

---

## 3. Similar cases with different labels (quarantined)

**Cluster #22** — 2 rows in `train`, text is **word-for-word identical**:
> "A 28-year-old female enrollee has requested authorization and coverage
> for autologous chondrocyte implantation and autologous cultured
> chondrocytes implant for left knee. The Health Insurer has denied this
> request indicating that the requested procedure is not medically
> necessary for treatment of the enrollees left knee pain."

One row: `Overturned`. The other: `Upheld`. Same age, same sex, same
procedure, same joint, same denial reason — text alone gives no signal
that could separate these into two different outcomes. Flagged
`conflicting_decision` and the **entire cluster** quarantined, since this
is a clean 50/50 split (not a minority outlier).

**Cluster #1152** — 222 rows, near-identical `text`:
> "An enrollee has requested breast tomosynthesis for evaluation of her
> medical condition."

216 rows `Overturned`, 6 rows `Upheld` (~2.7% of the cluster). Flagged
`conflicting_decision_minority_outlier` — because the disagreement is a
small minority (<5% threshold), only those 6 outlier rows are quarantined;
the other 216 stay in the clean set, since the dominant label for this
fact pattern is still reliable.

---

## Summary statistics (of 73,987 total rows)

| Category | Rows | % of total |
|---|---|---|
| Exact duplicates (removed, first occurrence kept) | 2,769 | 3.74% |
| Near-duplicates kept as legitimate distinct cases | 5,647 | 7.63% |
| Near-duplicates quarantined for conflicting labels | 1,206 | 1.63% |

(Computed directly from `train_clean.parquet` / `test_clean.parquet` /
`quarantined.parquet` / `dropped_exact_duplicates.parquet`, not just the
summary in `cleaning_report.md`.)
