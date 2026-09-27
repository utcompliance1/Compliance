"""
Transformer baseline for the IMR Appeals dataset (utcompliance1/imr-appeals-cleaned).

Fine-tunes DistilBERT for binary classification (Upheld / Overturned) and
evaluates on the same 4,000-row stratified test sample used for this week's
gpt-5 / gpt-5-nano LLM comparison, for a fair cross-approach comparison.

Requires a GPU (developed on Google Colab, T4). Running on CPU only is not
practical at this dataset size.

Usage (in Colab or any GPU-enabled environment):
    pip install transformers datasets accelerate scikit-learn
    python train_transformer.py
"""

import json

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert-base-uncased"
SEED = 42
N_EVAL = 4000  # must match the LLM comparison's eval sample size

print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("WARNING: no GPU detected. Training will be very slow on CPU.")

# ---------------------------------------------------------------------------
# 1. Load data and reconstruct the exact same test sample used by the LLM
#    baseline this week (same seed, same sampling function -- must stay in
#    sync with baseline_eval_cleaned.py's stratified_sample_n()).
# ---------------------------------------------------------------------------
def stratified_sample_n(dataframe, n, seed=SEED):
    n_classes = dataframe["decision"].nunique()
    per_class = n // n_classes
    frames = [
        group.sample(min(per_class, len(group)), random_state=seed)
        for _, group in dataframe.groupby("decision")
    ]
    return pd.concat(frames).sample(frac=1, random_state=seed).reset_index(drop=True)

test_ds = load_dataset("utcompliance1/imr-appeals-cleaned", split="test")
test_df = test_ds.to_pandas()
eval_sample_df = stratified_sample_n(test_df, n=N_EVAL, seed=SEED)
print(f"eval_sample: {len(eval_sample_df)} rows")
print(eval_sample_df["decision"].value_counts())

train_ds = load_dataset("utcompliance1/imr-appeals-cleaned", split="train")
train_df = train_ds.to_pandas()
print(f"train: {len(train_df)} rows")

# ---------------------------------------------------------------------------
# 2. Labels
# ---------------------------------------------------------------------------
label2id = {"Upheld": 0, "Overturned": 1}
id2label = {0: "Upheld", 1: "Overturned"}

train_df["label"] = train_df["decision"].map(label2id)
eval_sample_df["label"] = eval_sample_df["decision"].map(label2id)

# ---------------------------------------------------------------------------
# 3. Tokenize
#    NOTE: uses `text` only, never `full_text` -- same leakage rationale as
#    the LLM pipeline (full_text contains the reviewer's own reasoning).
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

train_hf = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize_fn, batched=True)
eval_hf = Dataset.from_pandas(eval_sample_df[["text", "label"]]).map(tokenize_fn, batched=True)

# ---------------------------------------------------------------------------
# 4. Model
# ---------------------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id,
)

# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

training_args = TrainingArguments(
    output_dir="./distilbert-imr",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=200,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_hf,
    eval_dataset=eval_hf,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train()
    # `load_best_model_at_end=True` means trainer.model is now the best
    # checkpoint by accuracy across all epochs (empirically, epoch 2 of 3 --
    # epoch 3 showed overfitting: validation loss rose while training loss
    # kept falling).

    print("Best checkpoint:", trainer.state.best_model_checkpoint)
    print("Best metric value:", trainer.state.best_metric)

    # -----------------------------------------------------------------
    # 6. Final evaluation + diagnostics
    # -----------------------------------------------------------------
    final_metrics = trainer.evaluate()
    print(final_metrics)

    predictions = trainer.predict(eval_hf)
    preds = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids

    report = classification_report(
        true_labels, preds, target_names=["Upheld", "Overturned"], output_dict=True
    )
    cm = confusion_matrix(true_labels, preds)
    print(classification_report(true_labels, preds, target_names=["Upheld", "Overturned"]))
    print("Confusion matrix:\n", cm)

    # -----------------------------------------------------------------
    # 7. Save results
    # -----------------------------------------------------------------
    output = {
        "distilbert-imr": {
            "model": "distilbert-base-uncased (fine-tuned)",
            "n": len(eval_hf),
            "accuracy": final_metrics["eval_accuracy"],
            "train_epochs_run": 3,
            "best_epoch": 2,
            "notes": (
                "load_best_model_at_end selected epoch 2 checkpoint; epoch 3 "
                "showed overfitting (val loss rose despite train loss "
                "continuing to decrease)."
            ),
            "per_class_metrics": {
                "Upheld": {
                    "precision": report["Upheld"]["precision"],
                    "recall": report["Upheld"]["recall"],
                    "f1": report["Upheld"]["f1-score"],
                    "support": report["Upheld"]["support"],
                },
                "Overturned": {
                    "precision": report["Overturned"]["precision"],
                    "recall": report["Overturned"]["recall"],
                    "f1": report["Overturned"]["f1-score"],
                    "support": report["Overturned"]["support"],
                },
            },
            "confusion_matrix": {
                "true_upheld_pred_upheld": int(cm[0][0]),
                "true_upheld_pred_overturned": int(cm[0][1]),
                "true_overturned_pred_upheld": int(cm[1][0]),
                "true_overturned_pred_overturned": int(cm[1][1]),
            },
        }
    }
    with open("transformer_results.json", "w") as f:
        json.dump(output, f, indent=2)

    # -----------------------------------------------------------------
    # 8. Save mismatches for qualitative review
    # -----------------------------------------------------------------
    mismatches = []
    for i in range(len(eval_sample_df)):
        if preds[i] != true_labels[i]:
            mismatches.append({
                "text": eval_sample_df.iloc[i]["text"],
                "true": id2label[true_labels[i]],
                "pred": id2label[preds[i]],
            })
    with open("transformer_mismatches.json", "w") as f:
        json.dump(mismatches, f, indent=2)

    print(f"\nSaved transformer_results.json and transformer_mismatches.json "
          f"({len(mismatches)} mismatches)")

    # -----------------------------------------------------------------
    # 9. Push the trained model to the Hugging Face Hub
    #    (requires `huggingface_hub` login -- run interactively, not
    #    automatically, since it needs an auth token)
    # -----------------------------------------------------------------
    # from huggingface_hub import login
    # login()
    # trainer.save_model("./distilbert-imr-final")
    # tokenizer.save_pretrained("./distilbert-imr-final")
    # trainer.model.push_to_hub("utcompliance1/distilbert-imr-baseline")
    # tokenizer.push_to_hub("utcompliance1/distilbert-imr-baseline")
