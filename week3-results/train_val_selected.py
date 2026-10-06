"""
Validation-selected DistilBERT on utcompliance1/imr-appeals-cleaned.

Checkpoint selection and early stopping use a validation slice carved from
`train` only (grouped by near_dup_group_id; -1 rows are their own group).
The test split is not touched here.

Usage:
    python train_val_selected.py [output_dir]

Safe to rerun: if checkpoints exist in output_dir, training resumes from the
newest one.
"""

import os
import sys
import glob
import json
import re

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 512
SEED = 42
VAL_FRAC = 0.12
OUT = sys.argv[1] if len(sys.argv) > 1 else "./imr_distilbert_v2"

set_seed(SEED)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

df = load_dataset("utcompliance1/imr-appeals-cleaned", split="train").to_pandas()
df = df.reset_index(drop=True)

df["split_group_id"] = df["near_dup_group_id"].astype(object)
singleton = df["split_group_id"] == -1
df.loc[singleton, "split_group_id"] = [f"singleton_{i}" for i in df.index[singleton]]

rng = np.random.RandomState(SEED)
groups = df["split_group_id"].unique()
rng.shuffle(groups)
val_groups = set(groups[: int(len(groups) * VAL_FRAC)])
is_val = df["split_group_id"].isin(val_groups)

val_df = df[is_val].reset_index(drop=True)
train_df = df[~is_val].reset_index(drop=True)
print(f"train={len(train_df)} val={len(val_df)}  (expected 52877 / 7219)")

pat = re.compile(r"\b(overturn\w*|uphold\w*|uphel\w*)\b", re.IGNORECASE)
for d in (train_df, val_df):
    d["has_shortcut"] = d["text"].apply(lambda t: bool(pat.search(t)))
print("shortcut rows: train", int(train_df["has_shortcut"].sum()),
      "val", int(val_df["has_shortcut"].sum()))

label2id = {"Upheld": 0, "Overturned": 1}
id2label = {0: "Upheld", 1: "Overturned"}
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def to_hf(frame):
    ds = Dataset.from_pandas(frame[["text", "decision"]])
    ds = ds.map(lambda b: {"label": [label2id[s] for s in b["decision"]]}, batched=True)
    return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LEN),
                  batched=True)


train_hf, val_hf = to_hf(train_df), to_hf(val_df)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id
)


def compute_metrics(p):
    return {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=-1))}


args = TrainingArguments(
    output_dir=OUT,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=200,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_hf,
    eval_dataset=val_hf,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

has_ckpt = bool(glob.glob(os.path.join(OUT, "checkpoint-*")))
print("Resuming from checkpoint" if has_ckpt else "Starting fresh")
trainer.train(resume_from_checkpoint=has_ckpt)

print("Best checkpoint:", trainer.state.best_model_checkpoint)
print("Best validation loss:", trainer.state.best_metric)

best_dir = os.path.join(OUT, "best")
trainer.save_model(best_dir)
tokenizer.save_pretrained(best_dir)
with open(os.path.join(OUT, "training_log.json"), "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)
print("Saved best model to", best_dir)
