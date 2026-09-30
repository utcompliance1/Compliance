"""Build confidence statistics from saved predictions without making API calls."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from baseline_eval_jev import JEV_DIR, wilson_interval, write_json


def metrics(rows, total, total_errors):
    n = len(rows)
    correct = sum(row["pred"] == row["true"] for row in rows)
    return {
        "cases": n, "share_of_cases": n / total,
        "correct": correct, "incorrect": n - correct,
        "accuracy": correct / n if n else None,
        "accuracy_95pct_wilson": wilson_interval(correct, n),
        "share_of_errors": (n - correct) / total_errors if total_errors else 0,
    }


def percentage(value):
    return f"{value:.2%}" if value is not None else "N/A"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=JEV_DIR / "full-test-focused")
    args = parser.parse_args()
    source = args.source_dir.resolve()
    predictions = source / "jev_predictions.jsonl"
    raw = predictions.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    summary = json.loads((source / "baseline_results_jev.json").read_text())
    n = len(rows)
    assert n and n == summary["scored"] == summary["attempted"]
    assert len({row["row_id"] for row in rows}) == n
    assert all(row["pred"] in ("Upheld", "Overturned") for row in rows)
    assert all(math.isfinite(row["confidence"]) and 0 <= row["confidence"] <= 1 for row in rows)
    correct = sum(row["pred"] == row["true"] for row in rows)
    assert correct == summary["correct"]
    errors = n - correct
    bins = []
    for low, high in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)):
        selected = [row for row in rows if low <= row["confidence"] and
                    (row["confidence"] <= high if high == 1 else row["confidence"] < high)]
        label = f"{low:.2f} ≤ c {'≤' if high == 1 else '<'} {high:.2f}"
        bins.append({"range": label, **metrics(selected, n, errors)})
    assert sum(item["cases"] for item in bins) == n
    assert sum(item["correct"] for item in bins) == correct
    thresholds = []
    for threshold in (0, .25, .5, .75, .9):
        selected = [row for row in rows if row["confidence"] >= threshold]
        thresholds.append({"minimum_confidence": threshold, **metrics(selected, n, errors)})
    counts = dict(Counter(row["true"] for row in rows))
    result = {
        "model": summary["resolved_models"], "prompt": summary["metadata"]["prompt_name"],
        "dataset": summary["metadata"]["dataset"], "revision": summary["metadata"]["revision"],
        "source_run_status": summary["status"], "source_run_utc": summary["created_at_utc"],
        "source_predictions_sha256": hashlib.sha256(raw).hexdigest(),
        "cases": n, "correct": correct, "incorrect": errors, "accuracy": correct / n,
        "accuracy_95pct_wilson": wilson_interval(correct, n),
        "balanced_accuracy": summary["balanced_accuracy"],
        "class_counts": counts, "majority_baseline_evaluated_cases": max(counts.values()) / n,
        "confidence_ranges": bins, "cumulative_thresholds": thresholds,
        "confusion_matrix": summary["confusion_matrix"], "per_class": summary["per_class"],
        "elapsed_seconds": summary["elapsed_seconds"], "latency_seconds": summary["latency_seconds"],
        "total_input_tokens": summary["total_input_tokens"], "total_output_tokens": summary["total_output_tokens"],
        "estimated_cost_usd": summary["estimated_cost_usd"],
        "requested_cases": summary["n"], "unattempted_cases": summary["unattempted"],
        "dataset_coverage": summary["coverage"], "failed_requests": summary["failed"],
    }
    write_json(JEV_DIR / "confidence_stats.json", result)
    lines = [
        "# Jev accuracy and confidence statistics", "",
        f"**Overall accuracy: {correct / n:.2%} ({correct:,}/{n:,} correct).**",
        "",
        f"This report covers all **{n:,} completed test cases** from the interrupted focused-prompt run. "
        "The separately saved 1,000-case snapshot scored 69.50%; it is contained within these cases and is not a separate experiment.",
        "", "## Overall results", "",
        "| Metric | Result |", "| --- | ---: |",
        f"| Scored cases | {n:,} |", f"| Correct / incorrect | {correct:,} / {errors:,} |",
        f"| Accuracy | {correct / n:.2%} |",
        f"| Approximate 95% Wilson interval | {percentage(result['accuracy_95pct_wilson'][0])}–{percentage(result['accuracy_95pct_wilson'][1])} |",
        f"| Balanced accuracy | {summary['balanced_accuracy']:.2%} |",
        f"| Majority baseline on evaluated cases | {result['majority_baseline_evaluated_cases']:.2%} |",
        f"| Upheld / Overturned true labels | {counts['Upheld']:,} / {counts['Overturned']:,} |",
        f"| Failed requests | {summary['failed']} |",
        f"| Requested / unattempted test cases | {summary['n']:,} / {summary['unattempted']:,} |",
        f"| Coverage of the full test split | {summary['coverage']:.2%} |",
        f"| Total elapsed time | {summary['elapsed_seconds']:.2f} seconds |",
        f"| Median / p95 client latency | {summary['latency_seconds']['p50'] * 1000:.0f} / {summary['latency_seconds']['p95'] * 1000:.0f} ms |",
        f"| Input / output tokens | {summary['total_input_tokens']:,} / {summary['total_output_tokens']:,} |",
        f"| Estimated cost of this larger run | ${summary['estimated_cost_usd']:.6f} |",
        "", "The run stopped at the user's request. Every attempted case was scored; the remaining cases were unattempted, not API failures. "
        "The evaluated cases are a prefix of the test split, not a newly drawn random sample. "
        "The full 9,846-case split has a 53.90% majority baseline; this report uses the evaluated prefix's own baseline above.",
        "", "## Confidence ranges", "",
        "Each case belongs to exactly one range. Share of cases uses all completed cases as its denominator. "
        "Share of errors uses all incorrect predictions as its denominator.", "",
        "| Returned confidence c | Cases | Share of cases | Correct | Incorrect | Accuracy | Share of all errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in bins:
        lines.append(f"| {item['range']} | {item['cases']:,} | {percentage(item['share_of_cases'])} | "
                     f"{item['correct']:,} | {item['incorrect']:,} | {percentage(item['accuracy'])} | {percentage(item['share_of_errors'])} |")
    lines += [f"| **Overall** | **{n:,}** | **100.00%** | **{correct:,}** | **{errors:,}** | **{correct/n:.2%}** | **100.00%** |",
              "", "Percentages are rounded, so displayed shares may differ slightly from 100% when summed.",
              "", "## Accuracy at cumulative confidence cutoffs", "",
              "These rows overlap: each cutoff includes all cases at or above that confidence.", "",
              "| Minimum confidence | Retained cases | Share retained | Share excluded | Accuracy on retained cases |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for item in thresholds:
        lines.append(f"| {item['minimum_confidence']:.2f} | {item['cases']:,} | {percentage(item['share_of_cases'])} | "
                     f"{percentage(1-item['share_of_cases'])} | {percentage(item['accuracy'])} |")
    lines += ["", "At a **0.75 cutoff**, accuracy is **80.98%** on **39.70%** of completed cases. "
              "The other **60.30%** would need another prediction method or review. "
              "This cutoff does not increase accuracy across all cases by itself.",
              "", "Confidence here is the API's returned `confidence` field, not the selected label's probability. "
              "TypeSafe derives it from the probability distribution. An individual score of 0.75 is not a guarantee of 75% accuracy. "
              "[Official confidence documentation](https://docs.typesafe.ai/confidence).",
              "", "## Per-class metrics", "",
              "| True class | Precision | Recall | F1 | Support |", "| --- | ---: | ---: | ---: | ---: |"]
    for label in ("Upheld", "Overturned"):
        item = summary["per_class"][label]
        lines.append(f"| {label} | {percentage(item['precision'])} | {percentage(item['recall'])} | {percentage(item['f1'])} | {item['support']:,} |")
    matrix = summary["confusion_matrix"]
    lines += ["", "Confusion matrix (rows are true labels):", "",
              "| | Predicted Upheld | Predicted Overturned |", "| --- | ---: | ---: |",
              f"| True Upheld | {matrix['Upheld']['Upheld']:,} | {matrix['Upheld']['Overturned']:,} |",
              f"| True Overturned | {matrix['Overturned']['Upheld']:,} | {matrix['Overturned']['Overturned']:,} |",
              "", "## Method and provenance", "",
              "- Model: `jev-1.13.0`; focused zero-shot prompt selected on separate training-split development cases.",
              f"- Dataset: `{result['dataset']}`, `test` split; revision `{result['revision']}`.",
              "- Input: `text` only; reviewer reasoning in `full_text` and ground-truth decisions were excluded from requests.",
              f"- Saved run timestamp: `{result['source_run_utc']}`.",
              "- Main metrics source: [interrupted-run summary](full-test-focused/baseline_results_jev.json).",
              "- Counts, confidence ranges, and intervals: [machine-readable statistics](confidence_stats.json).",
              "- Original 1,000-case report: [results and methodology](RESULTS.md).",
              f"- Prediction file SHA-256: `{result['source_predictions_sha256']}`.",
              "", "All confidence statistics were calculated from the existing local `full-test-focused/jev_predictions.jsonl`; "
              "no new API calls were made. That larger prediction file is kept locally and excluded from Git; "
              "the report publishes aggregate statistics. With that local file available, regenerate using:", "",
              "```bash", "python jev/confidence_report.py", "```", "",
              "The cutoff results describe this evaluated dataset. Select operational thresholds on development data "
              "and verify on new cases before treating them as expected future performance.", ""]
    (JEV_DIR / "CONFIDENCE_STATS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated CONFIDENCE_STATS.md and confidence_stats.json from {n:,} saved predictions.")


if __name__ == "__main__":
    main()
