"""Run the unchanged upstream implementation on the frozen completed GPT sample."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import yaml

from collect_gpt import eligible_cases, prompt_spec, text_hash
from xverifier.runner import run

HERE = Path(__file__).resolve().parent
SEEDS = [42, 123, 456, 789, 1024]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    source = args.work / "gpt_predictions.jsonl"
    records = [json.loads(line) for line in source.read_text().splitlines()]
    predictions = pd.DataFrame(records)
    assert predictions.text_sha256.is_unique
    full = pd.read_parquet(args.work / "cases.parquet")
    eligible, _ = eligible_cases(full, prompt_spec())
    eligible["text_sha256"] = eligible.text.map(text_hash)
    sample = eligible.merge(predictions[["text_sha256", "pred"]], on="text_sha256", how="inner", validate="many_to_one")
    sample = sample.sort_values("row_id").reset_index(drop=True)
    sample.to_csv(args.work / "limited_cases.csv", index=False)
    rows_dir = HERE / "rows"
    rows_dir.mkdir(exist_ok=True)
    embeddings = np.load(args.work / "embeddings.npz", allow_pickle=False)
    positions = pd.Series(np.arange(len(embeddings["row_id"])), index=embeddings["row_id"])
    vectors = embeddings["representation"]
    assert len(positions) == len(full)
    label_map = {"Upheld": 0, "Overturned": 1}
    summary = {
        "status": "complete_limited_run",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling": "All GPT predictions saved before user stopped full collection; eligible cases joined by exact text SHA-256, sorted by original row_id. Completion/prefix sample, not a random population sample. No further API calls.",
        "prediction_cache_sha256": sha(source),
        "unique_api_predictions": len(records),
        "unparsed_unique_predictions": int(predictions.pred.isna().sum()),
        "source_splits": sample.split.value_counts().to_dict(),
        "api_usage": {
            "input_tokens": sum(r["usage"]["prompt_tokens"] for r in records),
            "output_tokens": sum(r["usage"]["completion_tokens"] for r in records),
            "cost_estimate_usd_at_prior_baseline_rates": sum(r["usage"]["prompt_tokens"] * 1.25e-6 + r["usage"]["completion_tokens"] * 10e-6 for r in records),
            "note": "Estimate using prior baseline rates; excludes integration check and interrupted requests not saved in cache. Not an invoice.",
        },
        "datasets": {},
    }
    # Freeze the sample before inspecting any verifier outputs.
    (HERE / "limited_sample_plan.json").write_text(json.dumps({
        k: v for k, v in summary.items() if k != "datasets"
    }, indent=2) + "\n")
    with gzip.GzipFile(filename=str(HERE / "gpt_predictions.jsonl.gz"), mode="wb", mtime=0) as f:
        f.write(source.read_bytes())
    (HERE / "data_manifest.json").write_bytes((args.work / "data_manifest.json").read_bytes())
    for name, subset in (("uncleaned", sample), ("cleaned", sample[sample.clean])):
        subset = subset[subset.pred.notna()].copy()
        assert subset.row_id.is_unique and len(subset) >= 100
        assert subset.decision.isin(label_map).all() and subset.pred.isin(label_map).all()
        # Join the embedding by row ID; the row order is never assumed.
        index = positions.loc[subset.row_id].to_numpy()
        assert np.array_equal(embeddings["row_id"][index], subset.row_id.to_numpy())
        path = rows_dir / f"{name}.npz"
        np.savez(path, representation=vectors[index],
                 label=subset.decision.map(label_map).to_numpy(dtype=np.int8),
                 verdict=subset.pred.map(label_map).to_numpy(dtype=np.int8),
                 group=subset.near_dup_group_id.to_numpy(dtype=np.int64))
        subset[["row_id", "text_sha256", "split", "near_dup_group_id", "decision", "pred"]].to_csv(HERE / f"{name}_sample.csv", index=False)
        config = {
            "study": f"imr-gpt5-{name}-limited",
            "seeds": SEEDS,
            "frame": {"path": f"rows/{name}.npz", "sha256": {"path": sha(path)}},
            "split": {"mode": "draw", "calibration_fraction": .274, "evaluation_fraction": .082},
            "estimator": {"probe": {"family": "linear", "C": 1., "max_components": 256, "min_rows_per_verdict": 40},
                          "calibration": {"method": "venn_abers", "cross_fit_folds": 5}},
            "flip": {"threshold": .5},
            "decide": {"error_levels": [.25, .20, .15, .10, .05], "confidence": .95},
        }
        config_path = HERE / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"Running original xverifier: {name}, {len(subset)} rows, {len(SEEDS)} seeds", flush=True)
        results = run(config_path, HERE / name)
        runs = []
        for seed, result in results.items():
            agreement = result["agreement"]["test"].astype(bool)
            flipped = result["flipped"]["test"].astype(bool)
            correct = ~result["error"]["test"].astype(bool)
            metrics = {
                "seed": seed, "rows": result["rows"],
                "judge_accuracy": float(agreement.mean()),
                "verifier_accuracy": float(correct.mean()),
                "flipped": int(flipped.sum()),
                "flips_helped": int((flipped & ~agreement).sum()),
                "flips_hurt": int((flipped & agreement).sum()),
            }
            runs.append(metrics)
            test_indices = result["index"]["test"]
            output = subset.iloc[test_indices][["row_id", "decision", "pred"]].copy()
            output["confidence"] = result["confidence"]["test"]
            output["flipped"] = flipped
            output["verifier_correct"] = correct
            output.to_csv(HERE / name / f"test_predictions_{seed}.csv", index=False)
        aggregate = json.loads((HERE / name / "aggregate.json").read_text())
        summary["datasets"][name] = {
            "input_rows": len(subset), "source_splits": subset.split.value_counts().to_dict(),
            "sample_judge_accuracy": float((subset.decision == subset.pred).mean()),
            "test_judge_accuracy_mean": float(np.mean([r["judge_accuracy"] for r in runs])),
            "test_verifier_accuracy_mean": float(np.mean([r["verifier_accuracy"] for r in runs])),
            "test_judge_accuracy_std": float(np.std([r["judge_accuracy"] for r in runs], ddof=1)),
            "test_verifier_accuracy_std": float(np.std([r["verifier_accuracy"] for r in runs], ddof=1)),
            "runs": runs, "operating_points": aggregate["error_levels"],
        }
        print(json.dumps(summary["datasets"][name], indent=2), flush=True)
    (HERE / "limited_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Both limited evaluations complete", flush=True)


if __name__ == "__main__":
    main()
