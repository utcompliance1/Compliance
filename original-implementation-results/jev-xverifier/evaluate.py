"""Matched GPT/Jev evaluation through the original xverifier.runner.run API."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from collect_jev import HERE, ROOT, SEEDS, prepare, sha
from xverifier.runner import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    args = p.parse_args()
    sample = prepare(args.work)
    cache = args.work / "jev_limited_predictions.jsonl"
    saved = pd.DataFrame([json.loads(line) for line in cache.read_text().splitlines()])
    assert saved.text_sha256.is_unique
    assert set(sample.text_sha256) <= set(saved.text_sha256), "Limited collection incomplete"
    sample = sample.merge(saved[["text_sha256", "pred", "probabilities"]].rename(columns={"pred": "jev_pred"}),
                          on="text_sha256", how="left", validate="many_to_one").sort_values("row_id")
    sample["jev_probability"] = sample.apply(lambda r: r.probabilities[r.jev_pred], axis=1)
    assert sample.jev_pred.notna().all() and sample.gpt_pred.notna().all()
    assert sample.jev_probability.between(.5, 1).all()
    sample[["row_id", "text_sha256", "decision", "split", "near_dup_group_id", "clean", "gpt_pred", "jev_pred", "jev_probability"]].to_csv(HERE / "matched_sample.csv", index=False)
    with gzip.GzipFile(filename=str(HERE / "jev_predictions.jsonl.gz"), mode="wb", mtime=0) as f:
        f.write(cache.read_bytes())
    embed = np.load(args.work / "embeddings.npz", allow_pickle=False)
    position = pd.Series(np.arange(len(embed["row_id"])), index=embed["row_id"])
    label_map = {"Upheld": 0, "Overturned": 1}
    rows = HERE / "rows"
    rows.mkdir(exist_ok=True)
    summary = {
        "status": "complete_limited_matched_comparison", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "unique_jev_predictions": len(saved),
        "new_jev_cost_estimate_usd": float(saved.input_tokens.sum()) * .042e-6,
        "jev_input_tokens": int(saved.input_tokens.sum()),
        "sample_rows": len(sample), "cleaned_sample_rows": int(sample.clean.sum()),
        "notes": "Matched GPT/Jev cases within each dataset, five identical group-blocked draws. Nonrandom limited source-train sample. Seed SD is not a confidence interval. Filtered/unfiltered evaluation draws differ. Confidence sweep is diagnostic, not certification or tuning.",
        "experiments": {},
    }
    reference_splits = {}
    sweep = []
    for dataset, subset in (("uncleaned", sample), ("cleaned", sample[sample.clean])):
        subset = subset.sort_values("row_id").reset_index(drop=True)
        index = position.loc[subset.row_id].to_numpy()
        assert np.array_equal(embed["row_id"][index], subset.row_id.to_numpy())
        for judge in ("gpt", "jev"):
            name = f"{judge}-{dataset}"
            path = rows / f"{name}.npz"
            np.savez(path, representation=embed["representation"][index],
                     label=subset.decision.map(label_map).to_numpy(dtype=np.int8),
                     verdict=subset[f"{judge}_pred"].map(label_map).to_numpy(dtype=np.int8),
                     group=subset.near_dup_group_id.to_numpy(dtype=np.int64))
            config = {
                "study": f"matched-{name}", "seeds": SEEDS,
                "frame": {"path": f"rows/{name}.npz", "sha256": {"path": sha(path)}},
                "split": {"mode": "draw", "calibration_fraction": .274, "evaluation_fraction": .082},
                "estimator": {"probe": {"family": "linear", "C": 1., "max_components": 256, "min_rows_per_verdict": 40},
                              "calibration": {"method": "venn_abers", "cross_fit_folds": 5}},
                "flip": {"threshold": .5},
                "decide": {"error_levels": [.25, .20, .15, .10, .05], "confidence": .95},
            }
            config_path = HERE / f"{name}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            print(f"Running {name}: {len(subset)} rows, five seeds", flush=True)
            results = run(config_path, HERE / name)
            per_seed = []
            for seed, result in results.items():
                split_key = (dataset, seed)
                indices = result["index"]
                if judge == "gpt":
                    reference_splits[split_key] = indices
                else:
                    assert all(np.array_equal(indices[role], reference_splits[split_key][role]) for role in ("train", "calibration", "test"))
                # Confirm group and exact text are disjoint across all three roles.
                for a, b in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
                    left, right = subset.iloc[indices[a]], subset.iloc[indices[b]]
                    lg = set(left.loc[left.near_dup_group_id >= 0, "near_dup_group_id"])
                    rg = set(right.loc[right.near_dup_group_id >= 0, "near_dup_group_id"])
                    assert lg.isdisjoint(rg)
                    assert set(left.text_sha256).isdisjoint(set(right.text_sha256))
                test = subset.iloc[indices["test"]]
                output = test[["row_id", "text_sha256", "decision", "gpt_pred", "jev_pred", "jev_probability"]].copy()
                output["confidence"] = result["confidence"]["test"]
                output["base_correct"] = result["agreement"]["test"].astype(bool)
                output["verifier_correct"] = ~result["error"]["test"].astype(bool)
                output["flipped"] = result["flipped"]["test"].astype(bool)
                output.to_csv(HERE / name / f"test_predictions_{seed}.csv", index=False)
                metrics = {"seed": seed, "rows": result["rows"],
                           "base_accuracy": float(output.base_correct.mean()),
                           "verifier_accuracy": float(output.verifier_correct.mean()),
                           "flips": int(output.flipped.sum()),
                           "flips_helped": int((output.flipped & ~output.base_correct).sum()),
                           "flips_hurt": int((output.flipped & output.base_correct).sum())}
                per_seed.append(metrics)
                for threshold in np.r_[np.linspace(.5, .95, 46), .975, .99, .995, 1.]:
                    series = [("xverifier", output.confidence, output.verifier_correct)]
                    if judge == "jev":
                        series.append(("native_jev", output.jev_probability, output.base_correct))
                    for method, confidence, correct in series:
                        mask = confidence >= threshold - 1e-12
                        n = int(mask.sum())
                        sweep.append({"dataset": dataset, "judge": judge, "method": method, "seed": seed,
                                      "threshold": float(threshold), "total": len(output), "decided": n,
                                      "coverage": n / len(output),
                                      "accuracy": float(correct[mask].mean()) if n else None,
                                      "correct": int(correct[mask].sum())})
            aggregate = json.loads((HERE / name / "aggregate.json").read_text())
            summary["experiments"][name] = {
                "dataset": dataset, "judge": judge, "input_rows": len(subset),
                "base_accuracy_mean": float(np.mean([r["base_accuracy"] for r in per_seed])),
                "base_accuracy_sd": float(np.std([r["base_accuracy"] for r in per_seed], ddof=1)),
                "verifier_accuracy_mean": float(np.mean([r["verifier_accuracy"] for r in per_seed])),
                "verifier_accuracy_sd": float(np.std([r["verifier_accuracy"] for r in per_seed], ddof=1)),
                "accuracy_delta_pp": float(np.mean([r["verifier_accuracy"] - r["base_accuracy"] for r in per_seed])) * 100,
                "per_seed": per_seed, "operating_points": aggregate["error_levels"],
            }
            print(f"{name}: {summary['experiments'][name]['base_accuracy_mean']:.2%} -> {summary['experiments'][name]['verifier_accuracy_mean']:.2%}", flush=True)
    pd.DataFrame(sweep).to_csv(HERE / "confidence_sweep.csv", index=False)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (HERE / "verification.json").write_text(json.dumps({
        "row_id_embedding_join_verified": True, "gpt_jev_roles_identical_within_dataset": True,
        "group_and_exact_text_roles_disjoint": True, "complete_jev_prediction_coverage": True,
        "native_upstream_runner_used": True, "upstream_commit": "241fcf3e068771fd23522b582f72e4e87d07fce8",
        "sample_sha256": sha(HERE / "matched_sample.csv"), "jev_cache_sha256": sha(cache),
    }, indent=2) + "\n")
    print("All matched limited evaluations complete", flush=True)


if __name__ == "__main__":
    main()
