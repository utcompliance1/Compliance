"""
Single frozen test evaluation for the validation-selected DistilBERT.

Run this ONCE, after the checkpoint and settings are frozen. It scores:
  * the full cleaned test split (natural prevalence)      -> headline number
  * the 4,000-row balanced sample (same rows as the gpt-5 / gpt-5-nano run)
each with and without "shortcut" rows (text that states the outcome), and it
saves a prediction plus class probabilities for EVERY test row, so ECE, paired
tests, or any later metric can be computed from the saved file without
rerunning the model.

It also scores last week's test-selected model on the same rows (it was
trained with max_length=256) for a like-for-like comparison. That model is not
being chosen or tuned here, so it does not break the one-evaluation rule.

Usage:
    python eval_test_once.py <val_selected_model_dir> [output_dir]
"""

import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from scipy.special import softmax
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

if len(sys.argv) < 2:
    sys.exit("usage: python eval_test_once.py <val_selected_model_dir> [output_dir]")

MODEL_DIR = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "./eval_out"
os.makedirs(OUT, exist_ok=True)

SEED = 42
N_BALANCED = 4000
LABELS = ["Upheld", "Overturned"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
OLD_MODEL = "utcompliance1/distilbert-imr-baseline"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
test = load_dataset("utcompliance1/imr-appeals-cleaned", split="test").to_pandas()
test = test.reset_index(drop=True)
test["orig_idx"] = np.arange(len(test))
assert set(test["decision"].unique()) <= set(LABELS), "unexpected labels"
y = test["decision"].map(LABEL2ID).values
print(f"test rows: {len(test)}")

# rows whose text states the outcome
pat = re.compile(r"\b(overturn\w*|uphold\w*|uphel\w*)\b", re.IGNORECASE)
test["has_shortcut"] = test["text"].apply(lambda t: bool(pat.search(t)))
print("shortcut rows in test:", int(test["has_shortcut"].sum()))


def stratified_sample_n(dataframe, n, seed=42):
    """Identical to baseline_eval_cleaned.py, so the sample matches the LLM run."""
    per_class = n // dataframe["decision"].nunique()
    frames = [
        g.sample(min(per_class, len(g)), random_state=seed)
        for _, g in dataframe.groupby("decision")
    ]
    return pd.concat(frames).sample(frac=1, random_state=seed).reset_index(drop=True)


bal = stratified_sample_n(test, N_BALANCED, SEED)
test["in_balanced"] = test["orig_idx"].isin(set(bal["orig_idx"]))
print("balanced sample:", int(test["in_balanced"].sum()),
      bal["decision"].value_counts().to_dict(), "(expected 2000 / 2000)")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_logits(model_ref, max_len, texts):
    tok = AutoTokenizer.from_pretrained(model_ref)
    model = AutoModelForSequenceClassification.from_pretrained(model_ref)
    ds = Dataset.from_dict({"text": list(texts)})
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_len),
                batched=True)
    args = TrainingArguments(
        output_dir=os.path.join(OUT, "tmp"),
        per_device_eval_batch_size=64,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, data_collator=DataCollatorWithPadding(tok))
    return trainer.predict(ds).predictions


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


def summarize(mask, pred):
    n = int(mask.sum())
    if n == 0:
        return None
    k = int((pred[mask] == y[mask]).sum())
    lo, hi = wilson(k, n)
    counts = np.bincount(y[mask], minlength=2)
    recall = {}
    for c in (0, 1):
        sel = mask & (y == c)
        recall[LABELS[c]] = float((pred[sel] == c).mean()) if sel.any() else None
    return {
        "n": n,
        "accuracy": k / n,
        "ci95": [lo, hi],
        "majority_baseline": float(counts.max() / n),
        "recall": recall,
    }


subsets = {
    "full_test_all": np.ones(len(test), dtype=bool),
    "full_test_no_shortcut": ~test["has_shortcut"].values,
    "balanced4000_all": test["in_balanced"].values,
    "balanced4000_no_shortcut": test["in_balanced"].values & ~test["has_shortcut"].values,
}

models = {
    "val_selected": (MODEL_DIR, 512),
    "old_test_selected": (OLD_MODEL, 256),
}

id_cols = [c for c in ("row_id",) if c in test.columns]
table = test[["orig_idx"] + id_cols + ["decision", "has_shortcut", "in_balanced"]].copy()
results = {}

for name, (ref, max_len) in models.items():
    try:
        logits = predict_logits(ref, max_len, test["text"])
    except Exception as e:  # the comparison model is optional
        if name == "val_selected":
            raise
        print(f"[skip] could not score {name}: {e}")
        continue
    probs = softmax(logits, axis=1)
    pred = logits.argmax(axis=1)
    table[f"{name}_pred"] = [LABELS[i] for i in pred]
    table[f"{name}_p_upheld"] = probs[:, 0]
    table[f"{name}_p_overturned"] = probs[:, 1]
    table[f"{name}_logit_upheld"] = logits[:, 0]
    table[f"{name}_logit_overturned"] = logits[:, 1]
    results[name] = {s: summarize(m, pred) for s, m in subsets.items()}

# ---------------------------------------------------------------------------
# Save and print
# ---------------------------------------------------------------------------
table.to_csv(os.path.join(OUT, "test_predictions.csv"), index=False)
with open(os.path.join(OUT, "test_metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n=== RESULTS ===")
for name, res in results.items():
    print(f"\n{name}")
    for s, r in res.items():
        if r is None:
            continue
        print(f"  {s:26s} n={r['n']:5d}  acc={r['accuracy']:.4f}  "
              f"95% CI [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]  "
              f"majority={r['majority_baseline']:.4f}")
print(f"\nSaved per-row predictions and metrics to {OUT}")
