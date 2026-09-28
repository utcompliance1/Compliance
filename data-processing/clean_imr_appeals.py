"""
IMR Appeals dataset cleaning pipeline
======================================
Source: https://huggingface.co/datasets/Persius/imr-appeals
Schema: text, decision, appeal_type, full_text, sufficiency_id  (splits: train, test)

What this does
--------------
1. EXPLORATORY ANALYSIS
   - basic shape / label distributions / text-length stats
   - sanity check that `full_text` really does start with `text`
   - exact-duplicate detection on `text` and on `full_text`
   - cross-split duplicate detection (train row leaking into test)
   - near-duplicate detection via TF-IDF cosine similarity + union-find clustering
   - conflicting-label detection inside near-dup clusters

2. QUARANTINE
   Rows get pulled into a `quarantined` split (with a `quarantine_reason` column)
   when they belong to a cluster of near-identical text that disagrees on
   `decision`, `appeal_type`, or `sufficiency_id`, or when the same text
   appears in both train and test (leakage).

3. CLEAN OUTPUT
   `train_clean` / `test_clean` = original rows minus exact duplicates
   (keep first occurrence) minus quarantined rows, with a
   `dup_group_id` and `near_dup_group_id` column added for transparency
   (so downstream users can still recover/inspect what was collapsed).

4. PUSH TO HUB
   Builds a DatasetDict and calls push_to_hub(). Requires a logged-in
   huggingface_hub session or HF_TOKEN env var, and network access to
   huggingface.co (this script is meant to run in YOUR environment, not
   inside a sandboxed tool container without HF network access).

Usage
-----
    pip install datasets huggingface_hub scikit-learn pandas scipy tqdm

    python clean_imr_appeals.py \
        --source_repo Persius/imr-appeals \
        --target_repo <your-username>/imr-appeals-cleaned \
        --sim_threshold 0.90 \
        --push

Run with --push omitted first to just get the local report + cleaned
parquet files under ./cleaned_output/ before deciding to publish.
"""
import argparse
import hashlib
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Union-Find for clustering near-duplicates
# --------------------------------------------------------------------------
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def normalize_text(s: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation-heavy noise."""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Step 1: Load
# --------------------------------------------------------------------------
def load_data(source_repo: str) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(source_repo)
    frames = []
    for split_name in ds.keys():
        df = ds[split_name].to_pandas()
        df["split"] = split_name
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full["row_id"] = full.index
    full["appeal_type"] = full["appeal_type"].replace({"Medical necessity": "Medical Necessity"})
    return full


# --------------------------------------------------------------------------
# Step 2: Exploratory analysis
# --------------------------------------------------------------------------
def explore(df: pd.DataFrame) -> dict:
    report = {}
    report["n_rows"] = len(df)
    report["n_rows_by_split"] = df["split"].value_counts().to_dict()
    report["decision_counts"] = df["decision"].value_counts().to_dict()
    report["appeal_type_counts"] = df["appeal_type"].value_counts().to_dict()
    report["sufficiency_id_counts"] = df["sufficiency_id"].value_counts().to_dict()
    report["text_len_chars"] = df["text"].str.len().describe().to_dict()
    report["full_text_len_chars"] = df["full_text"].str.len().describe().to_dict()

    # sanity check: does full_text actually contain text (allowing for the
    # "Summary Reviewer" / "Summary Reviewer N" prefix observed in samples)?
    def contains_text(row):
        return normalize_text(row["text"]) in normalize_text(row["full_text"])

    contains = df.apply(contains_text, axis=1)
    report["pct_full_text_contains_text"] = float(contains.mean())
    report["n_full_text_missing_text"] = int((~contains).sum())

    # null / empty checks
    report["n_null_text"] = int(df["text"].isna().sum())
    report["n_empty_text"] = int((df["text"].str.strip() == "").sum())

    return report


# --------------------------------------------------------------------------
# Step 3: Exact duplicate detection
# --------------------------------------------------------------------------
def find_exact_duplicates(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Returns df with a `<col>_hash` and `<col>_dup_group_id` column.
    Group id is -1 for rows that are unique (no duplicates)."""
    norm = df[col].fillna("").map(normalize_text)
    hashes = norm.map(sha1)
    df[f"{col}_hash"] = hashes

    group_map = {}
    group_id_col = np.full(len(df), -1, dtype=int)
    next_id = 0
    hash_to_indices = defaultdict(list)
    for i, h in enumerate(hashes):
        hash_to_indices[h].append(i)

    for h, idxs in hash_to_indices.items():
        if len(idxs) > 1:
            for i in idxs:
                group_id_col[i] = next_id
            next_id += 1

    df[f"{col}_dup_group_id"] = group_id_col
    return df


def cross_split_leakage(df: pd.DataFrame, col: str) -> pd.Series:
    """Rows whose <col>_hash appears in more than one split (EXACT text match)."""
    hash_splits = df.groupby(f"{col}_hash")["split"].nunique()
    leaking_hashes = set(hash_splits[hash_splits > 1].index)
    return df[f"{col}_hash"].isin(leaking_hashes)


def near_dup_cross_split_leakage(df: pd.DataFrame, group_col: str) -> pd.Series:
    """
    Rows in a NEAR-duplicate cluster (not just exact-hash match) that spans
    more than one split. Catches cases like a train row and a test row that
    are the same underlying case reworded by one word (e.g. '...therapy.'
    vs '...therapy services.'), which the exact-hash check above misses.
    """
    group_splits = df.groupby(group_col)["split"].nunique()
    leaking_groups = set(group_splits[(group_splits > 1) & (group_splits.index != -1)].index)
    return df[group_col].isin(leaking_groups)


# --------------------------------------------------------------------------
# Step 4: Near-duplicate detection (TF-IDF cosine similarity + clustering)
# --------------------------------------------------------------------------
def find_near_duplicates(df: pd.DataFrame, text_col: str = "text",
                          sim_threshold: float = 0.90,
                          max_neighbors: int = 15) -> np.ndarray:
    """
    Returns an array of near-dup group ids aligned to df's row order.
    -1 means the row was not linked to any other row above threshold.

    Uses TF-IDF (word 1-2 grams) + cosine similarity via sklearn's
    NearestNeighbors so this stays roughly O(n log n) instead of O(n^2),
    which matters at ~74k rows.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    texts = df[text_col].fillna("").tolist()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.5,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)

    n = X.shape[0]
    k = min(max_neighbors, n)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    uf = UnionFind(n)
    for i in range(n):
        for dist, j in zip(distances[i], indices[i]):
            if j == i:
                continue
            sim = 1 - dist
            if sim >= sim_threshold:
                uf.union(i, j)

    roots = np.array([uf.find(i) for i in range(n)])
    # relabel roots that actually have >1 member as 0..k-1, everyone else -1
    root_counts = pd.Series(roots).value_counts()
    multi_roots = set(root_counts[root_counts > 1].index)
    relabel = {r: gid for gid, r in enumerate(sorted(multi_roots))}
    group_ids = np.array([relabel.get(r, -1) for r in roots])
    return group_ids


# --------------------------------------------------------------------------
# Step 5: Conflict detection within near-dup clusters
# --------------------------------------------------------------------------
def flag_label_conflicts(df: pd.DataFrame, group_col: str,
                          label_cols=("decision", "appeal_type", "sufficiency_id"),
                          minority_threshold: float = 0.05):
    """
    For every near-dup group (excluding -1 = singleton), check whether
    label_cols disagree across members.

    Two tiers, per Kashish's call:
      - If the disagreeing rows are a SMALL minority of the group
        (< minority_threshold, default 5%), only THOSE outlier rows are
        flagged (e.g. conflicting_decision_minority_outlier) and the
        majority-label rows stay clean.
      - If the split is more substantial (closer to even), the ENTIRE
        group is flagged (e.g. conflicting_decision) since we can't trust
        either label for that fact pattern.

    Returns a Series aligned to df with a conflict_reason string or None.
    A row can accumulate reasons from multiple label_cols (joined with '+').
    """
    reasons = pd.Series([None] * len(df), index=df.index, dtype=object)

    def add_reason(idx, text):
        if reasons.loc[idx] is None:
            reasons.loc[idx] = text
        else:
            reasons.loc[idx] = reasons.loc[idx] + "+" + text

    for gid, idxs in df.groupby(group_col).groups.items():
        if gid == -1 or len(idxs) < 2:
            continue
        sub = df.loc[idxs]
        n = len(sub)

        for col in label_cols:
            counts = sub[col].value_counts(dropna=False)
            if len(counts) <= 1:
                continue  # no disagreement on this column

            mode_val = counts.idxmax()
            minority_idx = sub.index[sub[col] != mode_val]
            minority_frac = len(minority_idx) / n

            if minority_frac < minority_threshold:
                # small number of outliers: flag just those rows
                for idx in minority_idx:
                    add_reason(idx, f"conflicting_{col}_minority_outlier")
            else:
                # substantial split: flag the whole group
                for idx in idxs:
                    add_reason(idx, f"conflicting_{col}")

    return reasons


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def run_pipeline(source_repo: str, sim_threshold: float, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {source_repo} ...")
    df = load_data(source_repo)

    print("Running exploratory analysis ...")
    report = explore(df)

    print("Finding exact duplicates on `text` and `full_text` ...")
    df = find_exact_duplicates(df, "text")
    df = find_exact_duplicates(df, "full_text")

    report["n_exact_dup_groups_text"] = int((df["text_dup_group_id"] >= 0).sum() and
                                             df.loc[df["text_dup_group_id"] >= 0, "text_dup_group_id"].nunique())
    report["n_exact_dup_rows_text"] = int((df["text_dup_group_id"] >= 0).sum())

    print("Checking for cross-split leakage ...")
    df["text_cross_split_leak"] = cross_split_leakage(df, "text")
    report["n_cross_split_leak_rows"] = int(df["text_cross_split_leak"].sum())

    print(f"Finding near-duplicates (cosine >= {sim_threshold}) ... this is the slow step")
    df["near_dup_group_id"] = find_near_duplicates(df, "text", sim_threshold)
    report["n_near_dup_groups"] = int(len(set(df.loc[df["near_dup_group_id"] >= 0, "near_dup_group_id"])))
    report["n_near_dup_rows"] = int((df["near_dup_group_id"] >= 0).sum())

    print("Checking for near-duplicate cross-split leakage ...")
    df["near_dup_cross_split_leak"] = near_dup_cross_split_leakage(df, "near_dup_group_id")
    report["n_near_dup_cross_split_leak_rows"] = int(df["near_dup_cross_split_leak"].sum())

    print("Flagging label conflicts within near-dup clusters ...")
    df["conflict_reason"] = flag_label_conflicts(df, "near_dup_group_id")
    report["n_conflict_rows"] = int(df["conflict_reason"].notna().sum())
    report["conflict_reason_counts"] = df["conflict_reason"].value_counts().to_dict()

    # ---- Build quarantine reason (priority: label conflict > any cross-split leak) ----
    df["quarantine_reason"] = None
    df.loc[df["near_dup_cross_split_leak"], "quarantine_reason"] = "near_dup_cross_split_leakage"
    df.loc[df["text_cross_split_leak"], "quarantine_reason"] = "cross_split_leakage"
    df.loc[df["conflict_reason"].notna(), "quarantine_reason"] = df.loc[
        df["conflict_reason"].notna(), "conflict_reason"
    ]

    quarantined = df[df["quarantine_reason"].notna()].copy()

    # ---- Build clean set: drop quarantined + drop repeat exact dups (keep first) ----
    clean = df[df["quarantine_reason"].isna()].copy()
    clean = clean.sort_values("row_id")
    clean["_is_first_of_exact_dup"] = ~clean.duplicated(subset=["text_hash"], keep="first")
    dropped_exact_dups = clean[~clean["_is_first_of_exact_dup"]].copy()
    dropped_exact_dups = dropped_exact_dups.drop(columns=["_is_first_of_exact_dup"])
    clean = clean[clean["_is_first_of_exact_dup"]].drop(columns=["_is_first_of_exact_dup"])

    report["n_rows_quarantined"] = len(quarantined)
    report["n_rows_dropped_as_exact_dup_after_quarantine"] = len(dropped_exact_dups)
    report["n_rows_clean_final"] = len(clean)

    # ---- Write outputs ----
    for split_name, split_df in clean.groupby("split"):
        split_df.to_parquet(os.path.join(out_dir, f"{split_name}_clean.parquet"), index=False)
    quarantined.to_parquet(os.path.join(out_dir, "quarantined.parquet"), index=False)
    dropped_exact_dups.to_parquet(os.path.join(out_dir, "dropped_exact_duplicates.parquet"), index=False)

    write_report(report, os.path.join(out_dir, "cleaning_report.md"))
    print(f"\nDone. Outputs written to {out_dir}/")
    return df, clean, quarantined, report


def write_report(report: dict, path: str):
    lines = ["# IMR Appeals — Dataset Cleaning Report\n"]
    lines.append("## Overview\n")
    lines.append(f"- Total rows loaded: {report['n_rows']}")
    lines.append(f"- Rows by split: {report['n_rows_by_split']}")
    lines.append(f"- Decision label distribution: {report['decision_counts']}")
    lines.append(f"- Appeal type distribution: {report['appeal_type_counts']}")
    lines.append(f"- sufficiency_id distribution: {report['sufficiency_id_counts']}")
    lines.append(f"- % of rows where full_text contains text verbatim: "
                 f"{report['pct_full_text_contains_text']:.2%} "
                 f"({report['n_full_text_missing_text']} rows do not)")
    lines.append("")
    lines.append("## Deduplication\n")
    lines.append(f"- Exact-duplicate rows on `text` (normalized): {report['n_exact_dup_rows_text']} "
                 f"in {report['n_exact_dup_groups_text']} groups")
    lines.append(f"- Rows with identical `text` appearing in BOTH train and test (exact leakage): "
                 f"{report['n_cross_split_leak_rows']}")
    lines.append(f"- Rows in a near-duplicate cluster that spans BOTH train and test "
                 f"(near-dup leakage, e.g. one-word rewording): "
                 f"{report['n_near_dup_cross_split_leak_rows']}")
    lines.append(f"- Near-duplicate rows (TF-IDF cosine, clustered): {report['n_near_dup_rows']} "
                 f"in {report['n_near_dup_groups']} groups")
    lines.append("")
    lines.append("## Quarantine (conflicting labels on near-identical cases)\n")
    lines.append(f"- Rows quarantined: {report['n_rows_quarantined']}")
    lines.append(f"- Breakdown by reason: {report['conflict_reason_counts']}")
    lines.append("")
    lines.append("## Final Output\n")
    lines.append(f"- Rows dropped as exact duplicates (kept first occurrence): "
                 f"{report['n_rows_dropped_as_exact_dup_after_quarantine']}")
    lines.append(f"- Final clean row count: {report['n_rows_clean_final']}")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------
# Step 6: Push to Hub
# --------------------------------------------------------------------------
def push_to_hub(clean: pd.DataFrame, quarantined: pd.DataFrame,
                 dropped_exact_dups: pd.DataFrame, target_repo: str):
    from datasets import Dataset, DatasetDict

    # Force these two columns to a consistent nullable-string dtype across
    # ALL dataframes before building datasets. Without this, a dataframe
    # where the column is 100% empty (e.g. `clean`, which by definition has
    # no conflict_reason) gets inferred as an Arrow "null" type, while a
    # dataframe with actual text (e.g. `quarantined`) gets "string" type --
    # and push_to_hub refuses to combine splits with mismatched schemas.
    for d in (clean, quarantined, dropped_exact_dups):
        d["conflict_reason"] = d["conflict_reason"].astype("string")
        d["quarantine_reason"] = d["quarantine_reason"].astype("string")

    dd = DatasetDict()
    for split_name, split_df in clean.groupby("split"):
        dd[split_name] = Dataset.from_pandas(split_df.reset_index(drop=True))
    dd["quarantined"] = Dataset.from_pandas(quarantined.reset_index(drop=True))
    dd["dropped_exact_duplicates"] = Dataset.from_pandas(dropped_exact_dups.reset_index(drop=True))

    dd.push_to_hub(target_repo)
    print(f"Pushed to https://huggingface.co/datasets/{target_repo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_repo", default="Persius/imr-appeals")
    parser.add_argument("--target_repo", default=None,
                         help="e.g. your-username/imr-appeals-cleaned (required if --push)")
    parser.add_argument("--sim_threshold", type=float, default=0.93,
                         help="Cosine similarity threshold for near-dup clustering. "
                              "NOTE: this corpus is heavily templated ('A NN-year-old "
                              "[sex] enrollee has requested...'), so TF-IDF cosine sim "
                              "can false-merge unrelated cases that only share "
                              "boilerplate phrasing if this is set too low. Always "
                              "spot-check audit_near_dup_clusters.csv before trusting "
                              "the quarantine output at face value.")
    parser.add_argument("--out_dir", default="./cleaned_output")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    df, clean, quarantined, report = run_pipeline(
        args.source_repo, args.sim_threshold, args.out_dir
    )

    # Export the largest near-dup clusters for manual spot-checking, since
    # threshold-based clustering on templated legal/medical text can produce
    # false positives that are worth eyeballing before trusting quarantine.
    audit_rows = df[df["near_dup_group_id"] >= 0].sort_values("near_dup_group_id")
    cluster_sizes = audit_rows.groupby("near_dup_group_id").size().sort_values(ascending=False)
    top_clusters = cluster_sizes.head(30).index
    audit_sample = audit_rows[audit_rows["near_dup_group_id"].isin(top_clusters)][
        ["near_dup_group_id", "text", "decision", "appeal_type", "sufficiency_id", "split"]
    ]
    audit_sample.to_csv(os.path.join(args.out_dir, "audit_near_dup_clusters.csv"), index=False)
    print(f"Wrote {len(top_clusters)} largest near-dup clusters to "
          f"{args.out_dir}/audit_near_dup_clusters.csv for manual review.")

    if args.push:
        if not args.target_repo:
            print("ERROR: --target_repo is required when using --push", file=sys.stderr)
            sys.exit(1)
        dropped = pd.read_parquet(os.path.join(args.out_dir, "dropped_exact_duplicates.parquet"))
        push_to_hub(clean, quarantined, dropped, args.target_repo)