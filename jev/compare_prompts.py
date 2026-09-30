"""Select a prompt on train-split development cases, keeping test cases fixed.

Run: python jev/compare_prompts.py --n 100
"""

import argparse
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from typesafe_sdk import RetryPolicy, TypeSafeClient

from baseline_eval_jev import DATASET, MODEL, REVISION, JEV_DIR, evaluate, load_local_env, make_sample, sample_hash, write_json
from prompts import PROMPTS


def text_key(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--prompts", nargs="+", choices=PROMPTS, default=["basic", "structured"])
    parser.add_argument("--output-dir", type=Path, default=JEV_DIR / "prompt-development")
    args = parser.parse_args()
    if args.n < 2 or args.n % 2:
        parser.error("--n must be an even integer >= 2")
    load_local_env()
    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        parser.error("Set TYPESAFE_API_KEY locally.")
    folder = args.output_dir.resolve()
    if folder.exists():
        parser.error("Development directory exists; choose a new --output-dir.")

    train = load_dataset(DATASET, revision=REVISION, split="train").to_pandas()
    test = load_dataset(DATASET, revision=REVISION, split="test")
    # Exclude normalized exact-text overlaps across the entire test split.
    test_texts = {text_key(text) for text in test["text"]}
    eligible = train.loc[~train["text"].map(text_key).isin(test_texts)]
    sample = make_sample(eligible, args.n, args.seed, index_field="train_index")
    # The copied examples must not be evaluation cases, including dev cases.
    if "focused-fewshot" in args.prompts:
        examples = PROMPTS["focused-fewshot"]["decision"]["instructions"]["labeled_examples"]
        example_texts = {text_key(example["case"]) for example in examples}
        if example_texts & (test_texts | {text_key(row["text"]) for row in sample}):
            parser.error("A few-shot example overlaps evaluation data; remove it before running.")
    folder.mkdir(parents=True)
    plan = {
        "dataset": DATASET, "revision": REVISION, "split": "train",
        "n": args.n, "seed": args.seed, "sample_sha256": sample_hash(sample),
        "excluded_train_rows_matching_test_text": len(train) - len(eligible),
        "sample": sample, "model": MODEL,
        "prompts": {name: PROMPTS[name] for name in args.prompts},
        "selection_rule": "Highest dev accuracy with complete coverage; ties favor first listed prompt.",
        "test_policy": "Freeze selected prompt before rerunning the original 200 test cases; no further test-based tuning.",
    }
    # Record both candidates and the selection rule before any calls.
    write_json(folder / "development_plan.json", plan)
    results = {}
    with TypeSafeClient(api_key=api_key, base_url="https://api.typesafe.ai", timeout=30,
                        retry=RetryPolicy(max_retries=3)) as client:
        for name in args.prompts:
            questions = PROMPTS[name]
            print(f"Development prompt: {name}", flush=True)
            metrics, interrupted, models = evaluate(sample, client, MODEL, folder / f"{name}_predictions.jsonl", questions)
            results[name] = {"resolved_models": models, **metrics}
            write_json(folder / f"{name}_results.json", results[name])
            if interrupted or metrics["coverage"] != 1:
                print("Incomplete development run; no prompt selected.")
                return 1
            print(f"{name}: {metrics['accuracy']:.1%} ({metrics['correct']}/{args.n})", flush=True)
    selected = max(results, key=lambda name: results[name]["accuracy"])
    comparison = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_prompt": selected, "selection_rule": plan["selection_rule"],
        "sample_sha256": plan["sample_sha256"], "split": "train", "n": args.n,
        "results": results,
    }
    write_json(folder / "prompt_comparison.json", comparison)
    print(f"Frozen prompt selected on development data: {selected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
