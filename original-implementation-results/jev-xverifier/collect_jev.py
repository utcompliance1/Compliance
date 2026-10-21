"""Limited, checkpointed Jev collection on the prior GPT pilot's case pool."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import pandas as pd
from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient, RetryPolicy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "jev"))
from baseline_eval_jev import predict
from prompts import PROMPTS

MODEL = "jev-1.13.0"
SEEDS = [42, 123, 456, 789, 1024]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(work):
    prior = ROOT / "original-implementation-results/xverifier"
    full = pd.read_parquet(work / "cases.parquet")
    normalized = full.text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    dev_ids = set()
    for p in (ROOT / "jev").glob("prompt-development*/development_plan.json"):
        dev_ids.update(int(x["row_id"]) for x in json.loads(p.read_text())["sample"])
    overlap = full.row_id.isin(dev_ids)
    overlap |= normalized.isin(set(normalized[overlap]))
    groups = set(full.loc[overlap & (full.near_dup_group_id >= 0), "near_dup_group_id"])
    overlap |= full.near_dup_group_id.isin(groups)
    prior_rows = pd.read_csv(prior / "uncleaned_sample.csv")
    sample = full[full.row_id.isin(set(prior_rows.row_id)) & ~overlap].copy().sort_values("row_id")
    sample["text_sha256"] = sample.text.map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    sample = sample.merge(prior_rows[["row_id", "pred"]].rename(columns={"pred": "gpt_pred"}), on="row_id", validate="one_to_one")
    assert sample.row_id.is_unique and len(sample) <= 4884
    prompt = PROMPTS["focused"]
    (HERE / "jev_prompt.json").write_text(json.dumps(prompt, indent=2) + "\n")
    plan = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "prompt": "focused", "prompt_sha256": sha(HERE / "jev_prompt.json"),
        "upstream_commit": "241fcf3e068771fd23522b582f72e4e87d07fce8",
        "source_gpt_sample_sha256": sha(prior / "uncleaned_sample.csv"),
        "sample_policy": "Prior limited GPT pool; exclude both Jev prompt-development rounds, normalized text overlaps, and whole near-duplicate groups. Existing GPT exclusions remain inherited. Same rows, representations, groups and per-dataset splits for GPT and Jev.",
        "source_rows": len(prior_rows), "eligible_rows": len(sample),
        "excluded_jev_development_overlap_rows": sorted(set(prior_rows.row_id) - set(sample.row_id)),
        "unique_texts": int(sample.text_sha256.nunique()), "cleaned_rows": int(sample.clean.sum()),
        "seeds": SEEDS, "calibration_fraction": .274, "evaluation_fraction": .082,
        "error_levels": [.25, .20, .15, .10, .05], "confidence": .95,
        "input": "text only; never full_text, labels or row metadata",
        "scope": "Limited nonrandom completion/prefix sample from source train; new group-held-out verifier evaluation slices, not canonical test. No full run.",
    }
    path = HERE / "run_plan.json"
    if not path.exists():
        path.write_text(json.dumps(plan, indent=2) + "\n")
    else:
        old = json.loads(path.read_text())
        assert old["prompt_sha256"] == plan["prompt_sha256"] and old["eligible_rows"] == len(sample)
    sample.to_parquet(work / "jev_limited_cases.parquet", index=False)
    return sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--env-file", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=4000, help="Maximum new unique API calls; keeps collection limited")
    args = p.parse_args()
    sample = prepare(args.work)
    if args.env_file:
        load_dotenv(args.env_file, override=False)
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY is required")
    cache = args.work / "jev_limited_predictions.jsonl"
    saved = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            record = json.loads(line)
            assert record["model_requested"] == MODEL
            saved[record["text_sha256"]] = record
    todo = [r for r in sample.drop_duplicates("text_sha256")[["text_sha256", "text"]].to_dict("records") if r["text_sha256"] not in saved]
    todo = todo[:args.limit]
    print(f"Limited Jev: {len(sample)} rows; {len(saved)} cached; {len(todo)} new unique calls", flush=True)
    with TypeSafeClient(api_key=key, base_url="https://api.typesafe.ai", timeout=30,
                        retry=RetryPolicy(max_retries=3)) as client, cache.open("a") as stream:
        def query(row):
            result = predict(client, row["text"], MODEL, PROMPTS["focused"])
            if result["model"] != MODEL:
                raise ValueError("Unexpected resolved Jev model")
            return {"text_sha256": row["text_sha256"], "model_requested": MODEL, **result}
        def save(record):
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            saved[record["text_sha256"]] = record
        if todo:
            save(query(todo.pop(0)))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            iterator = iter(todo)
            pending = set()
            def fill():
                while len(pending) < args.workers:
                    row = next(iterator, None)
                    if row is None:
                        break
                    pending.add(pool.submit(query, row))
            fill()
            try:
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in done:
                        save(f.result())
                    if len(saved) % 250 < len(done):
                        print(f"Saved {len(saved)}/{sample.text_sha256.nunique()} unique predictions", flush=True)
                    fill()
            except BaseException:
                for f in pending:
                    f.cancel()
                raise
    print("Limited Jev collection complete", flush=True)


if __name__ == "__main__":
    main()
