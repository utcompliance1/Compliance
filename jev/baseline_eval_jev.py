"""Small, zero-shot Jev baseline on the cleaned IMR appeals test split.

Run from the repository root:
    python jev/baseline_eval_jev.py --n 200
    python jev/baseline_eval_jev.py --n 100 --dry-run
"""

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from typesafe_sdk import RetryPolicy, TypeSafeClient, TypeSafeError
from prompts import DEFAULT_PROMPT, PROMPTS

JEV_DIR = Path(__file__).resolve().parent
ROOT = JEV_DIR.parent
DATASET = "utcompliance1/imr-appeals-cleaned"
# Pin the dataset and model so later releases do not silently change the run.
REVISION = "0abb1e862c71e2b0558b560fceedab3d528759f7"
MODEL = "jev-1.13.0"
LABELS = ("Overturned", "Upheld")
INPUT_PRICE_PER_MILLION = 0.042  # https://docs.typesafe.ai/models
QUESTIONS = PROMPTS[DEFAULT_PROMPT]


def load_local_env():
    """Prefer jev/.env; retain compatibility with the original root .env."""
    load_dotenv(JEV_DIR / ".env", override=False)
    load_dotenv(ROOT / ".env", override=False)


def make_sample(dataframe, n, seed=42, index_field="test_index"):
    """Match the existing baselines' balanced pandas sampling for even n."""
    if n < 2 or n % 2:
        raise ValueError("--n must be an even integer >= 2 (equal cases per label).")
    if set(dataframe["decision"].unique()) != set(LABELS):
        raise ValueError("Dataset must contain exactly Upheld and Overturned labels.")
    frames = []
    for _, group in dataframe.groupby("decision"):
        if len(group) < n // 2:
            raise ValueError("Not enough cases to sample the requested size without replacement.")
        frames.append(group.sample(n // 2, random_state=seed))
    sample = pd.concat(frames).sample(frac=1, random_state=seed)
    return case_records(sample, index_field)


def case_records(dataframe, index_field="test_index"):
    """Preserve case identifiers while omitting label-leaking full_text."""
    if not dataframe["decision"].isin(LABELS).all():
        raise ValueError("Unexpected decision labels.")
    if not dataframe["text"].map(lambda text: isinstance(text, str) and bool(text.strip())).all():
        raise ValueError("Sample contains an empty or non-string case description.")
    # Keep the original split index and row_id for comparisons. Do not keep full_text.
    return [
        {
            index_field: int(index),
            "row_id": int(row["row_id"]),
            "text": row["text"],
            "decision": row["decision"],
        }
        for index, row in dataframe.iterrows()
    ]


def sample_hash(sample):
    data = json.dumps(sample, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def predict(client, case_text, model, questions=None):
    """Send only text; labels and dataset metadata never enter the request."""
    response = client.system_one(state=case_text, questions=questions or QUESTIONS, model=model)
    answer = response.choices["decision"]
    probabilities = answer.probabilities
    if answer.choice not in LABELS or set(probabilities) != set(LABELS):
        raise ValueError("Jev returned unexpected decision labels.")
    values = [answer.confidence, *probabilities.values()]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Jev returned invalid probabilities or confidence.")
    if not math.isclose(sum(probabilities.values()), 1, abs_tol=0.01):
        raise ValueError("Jev probabilities do not sum to approximately one.")
    return {
        "pred": answer.choice,
        "probabilities": dict(probabilities),
        "confidence": answer.confidence,
        "model": response.model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def wilson_interval(correct, n):
    """95% binomial interval; useful for a small pilot sample."""
    if not n:
        return None
    z = 1.959963984540054
    p = correct / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - half_width, center + half_width]


def summarize(sample, records, elapsed):
    successful = [row for row in records if row.get("pred") in LABELS]
    n = len(sample)
    scored = len(successful)
    correct = sum(row["pred"] == row["true"] for row in successful)
    matrix = {label: {pred: 0 for pred in LABELS} for label in LABELS}
    for row in successful:
        matrix[row["true"]][row["pred"]] += 1
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[true][label] for true in LABELS)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else None
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        per_class[label] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support,
        }
    recalls = [item["recall"] for item in per_class.values()]
    input_tokens = sum(row.get("input_tokens", 0) for row in records)
    output_tokens = sum(row.get("output_tokens", 0) for row in records)
    latencies = pd.Series([row["latency_seconds"] for row in successful], dtype=float)
    return {
        "n": n,
        "attempted": len(records),
        "scored": scored,
        "failed": len(records) - scored,
        "unattempted": n - len(records),
        "correct": correct,
        "coverage": scored / n,
        "accuracy": correct / scored if scored else None,
        "accuracy_95pct_wilson": wilson_interval(correct, scored),
        "accuracy_all_requested": correct / n,
        "balanced_accuracy": sum(recalls) / 2 if None not in recalls else None,
        "majority_class_baseline": max(Counter(row["decision"] for row in sample).values()) / n,
        "confusion_matrix": matrix,
        "per_class": per_class,
        "elapsed_seconds": elapsed,
        "latency_seconds": {
            "p50": float(latencies.quantile(0.5)) if scored else None,
            "p95": float(latencies.quantile(0.95)) if scored else None,
        },
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "estimated_cost_usd": input_tokens / 1e6 * INPUT_PRICE_PER_MILLION,
        "cost_note": "Estimate from returned usage; failed calls without usage may add cost.",
    }


def evaluate(sample, client, model, predictions_path, questions=None, max_workers=1):
    """Flush each result to disk and stop on errors requiring configuration fixes."""
    records = []
    interrupted = False
    start = time.perf_counter()
    progress_every = 100 if len(sample) >= 1000 else 20

    def run_one(row):
        record = {**row, "true": row["decision"]}
        call_start = time.perf_counter()
        fatal = False
        try:
            record.update(predict(client, row["text"], model, questions))
        except (TypeSafeError, ValueError, KeyError) as error:
            # Avoid logging response bodies or secrets in exception strings.
            status = getattr(error, "status", None)
            record.update(pred=None, error=type(error).__name__, http_status=status)
            fatal = status in {400, 401, 402, 403, 404, 422} or isinstance(error, (ValueError, KeyError))
            print(f"Case {row['row_id']} failed: {type(error).__name__}, HTTP {status}", flush=True)
        record["latency_seconds"] = time.perf_counter() - call_start
        return record, fatal

    with predictions_path.open("x", encoding="utf-8") as output:
        def save(record):
            records.append(record)
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            if len(records) % progress_every == 0 or len(records) == len(sample):
                print(f"Completed {len(records)}/{len(sample)} cases", flush=True)

        try:
            # Validate the first live call before concurrent work.
            first, fatal = run_one(sample[0])
            save(first)
            if first.get("pred") in LABELS and not fatal:
                remaining = iter(sample[1:])
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    pending = {}
                    for _ in range(max_workers):
                        row = next(remaining, None)
                        if row is not None:
                            pending[executor.submit(run_one, row)] = row
                    while pending:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            pending.pop(future)
                            record, case_fatal = future.result()
                            save(record)
                            fatal = fatal or case_fatal
                        # Drain in-flight calls after a fatal error, and stop submitting.
                        if not fatal:
                            for _ in done:
                                row = next(remaining, None)
                                if row is not None:
                                    pending[executor.submit(run_one, row)] = row
        except KeyboardInterrupt:
            interrupted = True
            print("Interrupted; saving results for completed cases.", flush=True)
    metrics = summarize(sample, records, time.perf_counter() - start)
    return metrics, interrupted, sorted({row["model"] for row in records if "model" in row})


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--n", type=int, default=200, help="Even balanced sample size (default: 200)")
    size.add_argument("--full-test", action="store_true", help="Evaluate every cleaned test case with its natural class balance")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--prompt", choices=PROMPTS, default=DEFAULT_PROMPT,
                        help="Versioned instructions and criteria (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent API calls (default: 1)")
    parser.add_argument("--revision", default=REVISION, help="Hugging Face dataset commit")
    parser.add_argument("--output-dir", type=Path, default=JEV_DIR / "output")
    parser.add_argument("--dry-run", action="store_true", help="Prepare sample without making API calls")
    args = parser.parse_args(argv)
    questions = PROMPTS[args.prompt]
    if args.n < 2 or args.n % 2:
        parser.error("--n must be an even integer >= 2")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    load_local_env()
    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not args.dry_run and not api_key:
        parser.error("Set TYPESAFE_API_KEY in your environment or jev/.env. Use --dry-run to prepare data without a key.")
    output_dir = args.output_dir.resolve()
    results_path = output_dir / "baseline_results_jev.json"
    predictions_path = output_dir / "jev_predictions.jsonl"
    if not args.dry_run and (results_path.exists() or predictions_path.exists()):
        parser.error("Results already exist. Choose a new --output-dir to preserve the earlier run.")

    print(f"Loading {DATASET} test split at {args.revision}", flush=True)
    dataset = load_dataset(DATASET, revision=args.revision, split="test")
    frame = dataset.to_pandas()
    sample = case_records(frame) if args.full_test else make_sample(frame, args.n, args.seed)
    metadata = {
        "dataset": DATASET, "revision": args.revision, "split": "test",
        "dataset_fingerprint": dataset._fingerprint,
        "dataset_rows": len(dataset),
        "dataset_class_counts": dict(Counter(dataset["decision"])),
        "input_field": "text",
        "prompt_mode": "few-shot" if args.prompt == "focused-fewshot" else "zero-shot",
        "prompt_name": args.prompt,
        "sampling": "entire test split, natural prevalence" if args.full_test else "equal per class, without replacement, pandas random_state",
        "n": len(sample), "seed": None if args.full_test else args.seed,
        "sample_sha256": sample_hash(sample), "workers": args.workers,
        "class_counts": dict(Counter(row["decision"] for row in sample)),
        "requested_model": args.model, "questions": questions,
        "input_price_usd_per_million_tokens": INPUT_PRICE_PER_MILLION,
        "pricing_source": "https://docs.typesafe.ai/models",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "jev_sample.json"
    # A dry run cannot change the manifest belonging to an existing live run.
    if results_path.exists() or predictions_path.exists():
        parser.error("Output directory contains a live run. Choose a new --output-dir.")
    write_json(sample_path, {"metadata": metadata, "sample": sample})
    print(f"Prepared {len(sample)} cases: {metadata['class_counts']}; sample saved to {sample_path}", flush=True)
    if args.dry_run:
        print("Dry run complete. No API calls made and no accuracy measured.")
        return 0

    with TypeSafeClient(api_key=api_key, base_url="https://api.typesafe.ai", timeout=30,
                        retry=RetryPolicy(max_retries=3)) as client:
        metrics, interrupted, models = evaluate(sample, client, args.model, predictions_path, questions, args.workers)
    complete = metrics["scored"] == len(sample)
    result = {
        "status": "complete" if complete else "interrupted" if interrupted else "incomplete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata, "resolved_models": models, **metrics,
    }
    write_json(results_path, result)
    if metrics["accuracy"] is not None:
        print(f"Jev accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['scored']} scored); "
              f"coverage: {metrics['coverage']:.1%}; estimated cost: ${metrics['estimated_cost_usd']:.6f}")
    print(f"Status: {result['status']}; results saved to {results_path}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
