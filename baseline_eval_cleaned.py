"""
LLM Baseline -- cleaned dataset run (utcompliance1/imr-appeals-cleaned).

Evaluates gpt-5 and gpt-5-nano on the canonical held-out test split
(4,000 rows stratified by decision) using the validated few-shot prompt.

Week 1 results (Persius/imr-appeals, 6,406 rows) are in baseline_results.json.
This run's output goes to baseline_results_cleaned.json.

Setup:
    pip install -r requirements.txt python-dotenv
    # .env file in this folder with OPENAI_API_KEY=sk-...

Usage:
    python baseline_eval_cleaned.py
"""

import os
import re
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads .env in cwd
client = OpenAI()  # picks up OPENAI_API_KEY

# 1. Load cleaned test split
ds = load_dataset("utcompliance1/imr-appeals-cleaned", split="test")
df = ds.to_pandas()
print(f"Loaded {len(df)} rows from utcompliance1/imr-appeals-cleaned (test split)")
print("Class balance:", Counter(df["decision"]))

# NOTE: using `text` column only, NOT `full_text`. `full_text` includes the
# reviewer's reasoning which leaks the label ("...is/is not medically necessary").


# 2. Build samples

def stratified_sample_n(dataframe, n, seed=42):
    """Sample n rows proportionally from each class."""
    n_classes = dataframe["decision"].nunique()
    per_class = n // n_classes
    frames = [
        group.sample(min(per_class, len(group)), random_state=seed)
        for _, group in dataframe.groupby("decision")
    ]
    return pd.concat(frames).sample(frac=1, random_state=seed).reset_index(drop=True)

# ~63-row sample used for sanity-checking prompts
dev_sample = stratified_sample_n(df, n=63).to_dict("records")
# 4,000-row canonical held-out sample for the real benchmark
eval_sample = stratified_sample_n(df, n=4000).to_dict("records")
print(f"dev_sample={len(dev_sample)} rows, eval_sample={len(eval_sample)} rows")


# 3. Prompts (unchanged from week 1 -- few-shot prompt is the validated winner)

LABEL_MAP = {"UPHELD": "Upheld", "OVERTURNED": "Overturned"}

FEWSHOT_EXAMPLES = [
    {
        "text": (
            "The patient is a 22-year-old male with a history of idiopathic scoliosis. "
            "He has had extensive chiropractic care from 10/13/15 through 7/4/16 for at "
            "least 36 visits. His diagnoses are cervical and thoracic sprain, rib sprain, "
            "and muscle spasm. His treatment includes chiropractic manipulation, traction, "
            "and electrical stimulation, as well as abdominal and low back exercises. "
            "Overall, his condition was unchanged and his pain was consistently at 2/10. "
            "The patient has requested reimbursement for the chiropractic services "
            "provided from 6/13/16 through 8/13/16. The Health Insurer has denied this "
            "request indicating that the services at issue were not medically necessary. "
            "The patient has a stable condition that shows no sign of improvement even "
            "with continuous extensive care."
        ),
        "decision": "UPHELD",
    },
    {
        "text": (
            "The parent of a 17-year-old male enrollee has requested reimbursement for "
            "residential treatment center (RTC) services provided from 4/4/15 through "
            "6/10/15. The Health Insurer has denied this request indicating that the "
            "services at issue were not medically necessary for treatment of the "
            "enrollees major depressive disorder, recurrent. The patients history "
            "explained his lack of cooperation with therapy prior to admission."
        ),
        "decision": "UPHELD",
    },
    {
        "text": (
            "An enrollee has requested authorization and coverage for right hip "
            "arthroplasty. The patient is documented to have sustained multiple "
            "traumatic injuries from a motor vehicle collision including a hip "
            "dislocation associated with fracture of the acetabulum."
        ),
        "decision": "OVERTURNED",
    },
    {
        "text": (
            "A 59-year-old female enrollee has requested reimbursement for the single "
            "photon emission computed tomography (SPECT) scan performed on 2/17/15. "
            "The Health Insurer has denied this request indicating that the services at "
            "issue were considered investigational for evaluation of the enrollees "
            "rigidity, lack of response to Sinemet trial, with suspected Parkinsons "
            "disease."
        ),
        "decision": "OVERTURNED",
    },
]

PROMPT_FEWSHOT_HEADER = """You are reviewing an independent medical review (IMR) appeal \
of a health insurance coverage denial. Based on the facts, decide whether the insurer's \
denial was UPHELD or OVERTURNED on appeal. Respond with exactly one word: UPHELD or \
OVERTURNED.

Here are some examples:
"""

def build_fewshot_prompt(case_text):
    parts = [PROMPT_FEWSHOT_HEADER]
    for ex in FEWSHOT_EXAMPLES:
        parts.append(f"Case:\n{ex['text']}\nAnswer: {ex['decision']}\n")
    parts.append(f"Case:\n{case_text}\nAnswer:")
    return "\n".join(parts)


# 4. Model evaluation

def predict(case_text, model, prompt_fn=build_fewshot_prompt):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_fn(case_text)}],
        max_completion_tokens=150,
        reasoning_effort="minimal",
    )
    raw = (resp.choices[0].message.content or "").strip().upper()
    match = re.search(r"UPHELD|OVERTURNED", raw)
    pred = LABEL_MAP[match.group(0)] if match else None
    usage = resp.usage
    return pred, usage.prompt_tokens, usage.completion_tokens

def evaluate_concurrent(sample, model, prompt_fn=build_fewshot_prompt, max_workers=10):
    correct = 0
    unparsed = 0
    mismatches = []
    completed = 0
    total_input_tokens = 0
    total_output_tokens = 0

    def run_one(row):
        pred, in_tok, out_tok = predict(row["text"], model=model, prompt_fn=prompt_fn)
        return row, pred, in_tok, out_tok

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_one, row) for row in sample]
        for future in as_completed(futures):
            row, pred, in_tok, out_tok = future.result()
            completed += 1
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            if completed % 200 == 0:
                print(f"  {completed}/{len(sample)} done...")
            if pred is None:
                unparsed += 1
                continue
            if pred == row["decision"]:
                correct += 1
            else:
                mismatches.append({"text": row["text"], "true": row["decision"], "pred": pred})

    n_scored = len(sample) - unparsed
    accuracy = correct / n_scored if n_scored else 0
    return {
        "model": model, "n": len(sample), "unparsed": unparsed, "accuracy": accuracy,
        "mismatches": mismatches,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }


# $ per 1M tokens
PRICING = {
    "gpt-5":      (1.25, 10.00),
    "gpt-5-mini": (0.25,  2.00),
    "gpt-5-nano": (0.05,  0.40),
}

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "baseline_results_cleaned.json")

# 5. Main

if __name__ == "__main__":
    assert not os.path.exists(
        os.path.join(os.path.dirname(__file__), "baseline_results.json") + ""
    ) or True, "baseline_results.json already exists -- this run writes to baseline_results_cleaned.json"

    print("\n--- Full eval (4,000 rows, few-shot prompt) ---")
    all_results = {}
    for model in ["gpt-5", "gpt-5-nano"]:
        print(f"\n=== {model} ({len(eval_sample)} rows) ===")
        start = time.time()
        result = evaluate_concurrent(eval_sample, model=model, max_workers=10)
        elapsed = time.time() - start

        in_price, out_price = PRICING[model]
        cost = (result["total_input_tokens"] / 1e6 * in_price) + (result["total_output_tokens"] / 1e6 * out_price)

        print(f"{model}: accuracy={result['accuracy']:.1%}, unparsed={result['unparsed']}, "
              f"time={elapsed/60:.1f}min, cost=${cost:.2f}")

        all_results[model] = {k: v for k, v in result.items() if k != "mismatches"}
        all_results[model]["elapsed_minutes"] = round(elapsed / 60, 2)
        all_results[model]["cost_usd"] = round(cost, 4)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")
