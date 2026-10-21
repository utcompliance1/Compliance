"""Checkpointed GPT baseline predictions, shared across cleaned/uncleaned inputs."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import pandas as pd
from openai import OpenAI

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MODEL = "gpt-5-2025-08-07"


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def prompt_spec():
    # Read constants without importing the old script (it makes API clients/downloads).
    tree = ast.parse((ROOT / "week1-results/baseline_eval_cleaned.py").read_text())
    wanted = {"FEWSHOT_EXAMPLES", "PROMPT_FEWSHOT_HEADER"}
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in wanted:
                values[name] = ast.literal_eval(node.value)
    assert set(values) == wanted
    return values


def build_prompt(text, spec):
    parts = [spec["PROMPT_FEWSHOT_HEADER"]]
    for example in spec["FEWSHOT_EXAMPLES"]:
        parts.append(f"Case:\n{example['text']}\nAnswer: {example['decision']}\n")
    parts.append(f"Case:\n{text}\nAnswer:")
    return "\n".join(parts)


def eligible_cases(full, spec):
    normalized = full.text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    # Reconstruct the original 0.1% development sample, preserving row IDs.
    train = full.loc[full.split == "train"]
    dev_ids = set()
    for _, g in train.groupby("decision"):
        dev_ids.update(g.sample(max(1, int(len(g) * .001)), random_state=42).row_id)
    example_texts = {re.sub(r"\s+", " ", e["text"].lower()).strip()
                     for e in spec["FEWSHOT_EXAMPLES"]}
    chosen = full.row_id.isin(dev_ids) | normalized.isin(example_texts)
    # Exclude normalized exact matches and full near-duplicate groups of these rows.
    chosen |= normalized.isin(set(normalized[chosen]))
    groups = set(full.loc[chosen & (full.near_dup_group_id >= 0), "near_dup_group_id"])
    chosen |= full.near_dup_group_id.isin(groups)
    return full.loc[~chosen].copy(), {
        "original_development_row_ids": sorted(int(i) for i in dev_ids),
        "excluded_prompt_overlap_row_ids": full.loc[chosen, "row_id"].astype(int).tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    spec = prompt_spec()
    spec_path = HERE / "gpt_prompt.json"
    serialized = json.dumps(spec, sort_keys=True, ensure_ascii=False)
    digest = text_hash(serialized)
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    cases = pd.read_parquet(args.work / "cases.parquet")
    cases, exclusions = eligible_cases(cases, spec)
    cases["text_sha256"] = cases.text.map(text_hash)
    plan = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "reasoning_effort": "minimal", "max_completion_tokens": 150,
        "prompt_sha256": digest, "workers": args.workers,
        "source_prompt": "week1-results/baseline_eval_cleaned.py",
        "seeds": [42, 123, 456, 789, 1024],
        "error_levels": [.25, .20, .15, .10, .05],
        "confidence": .95, "unique_texts": int(cases.text_sha256.nunique()),
        "exclusions": exclusions,
        "policy": "Original train/test; all eligible test rows retained. Remove train rows sharing a near-duplicate group or normalized exact text with test, separately per dataset. Never use full_text. Unparsed/failed verdicts excluded and counted. No test-driven tuning.",
    }
    plan_path = HERE / "run_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text())
        assert existing["model"] == MODEL and existing["prompt_sha256"] == digest
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    cache = args.work / "gpt_predictions.jsonl"
    saved = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            r = json.loads(line)
            assert r["model_requested"] == MODEL and r["prompt_sha256"] == digest
            saved[r["text_sha256"]] = r
    unique = cases.drop_duplicates("text_sha256")
    todo = [r for r in unique[["text", "text_sha256"]].to_dict("records") if r["text_sha256"] not in saved]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(saved)} cached; {len(todo)} unique texts to query", flush=True)
    client = OpenAI(timeout=60, max_retries=3)

    def predict(row):
        start = time.monotonic()
        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": build_prompt(row["text"], spec)}],
            max_completion_tokens=150, reasoning_effort="minimal")
        raw = (response.choices[0].message.content or "").strip().upper()
        matches = set(re.findall(r"\b(?:UPHELD|OVERTURNED)\b", raw))
        pred = {"UPHELD": "Upheld", "OVERTURNED": "Overturned"}[matches.pop()] if len(matches) == 1 else None
        return {"text_sha256": row["text_sha256"], "model_requested": MODEL,
                "model": response.model, "prompt_sha256": digest, "pred": pred,
                "raw": raw, "finish_reason": response.choices[0].finish_reason,
                "usage": response.usage.model_dump(), "response_id": response.id,
                "latency_seconds": time.monotonic() - start}

    # First call validates auth/model/parameters before scheduling concurrent work.
    start = time.monotonic()
    with cache.open("a") as stream:
        def save(record):
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            saved[record["text_sha256"]] = record

        if todo:
            save(predict(todo.pop(0)))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            iterator = iter(todo)
            pending = {}
            def fill():
                while len(pending) < args.workers:
                    row = next(iterator, None)
                    if row is None:
                        break
                    pending[pool.submit(predict, row)] = row
            fill()
            try:
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future)
                        save(future.result())
                    if len(saved) % 100 < len(done):
                        print(f"Completed {len(saved)}/{len(unique)}; elapsed {time.monotonic()-start:.0f}s", flush=True)
                    fill()
            except BaseException:
                for f in pending:
                    f.cancel()
                raise
    print(f"Saved {len(saved)} unique predictions", flush=True)


if __name__ == "__main__":
    main()
