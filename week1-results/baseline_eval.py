"""
LLM Baseline for the IMR Appeals dataset (Persius/imr-appeals).

Predicts whether a denied insurance claim's IMR decision was
"Upheld" (denial stands) or "Overturned" (denial reversed), based
only on the appeal case description -- not the reviewer's reasoning.

Setup:
    pip install -r requirements.txt
    export OPENAI_API_KEY="..."

Usage:
    python baseline_eval.py
"""

import os
import re
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from datasets import load_dataset
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY

# 1. Load data
ds = load_dataset("Persius/imr-appeals", split="train")
df = ds.to_pandas()
print(f"Loaded {len(df)} rows")
print("Class balance:", Counter(df["decision"]))

# NOTE: I used the `text` column, NOT `full_text`. `full_text` includes the
# reviewer's reasoning/findings, which leaks the answer (it
# states "...is/is not medically necessary", which maps almost
# directly onto the decision). I confirmed this by manual inspection of sample
# rows before building the pipeline.


# 2. Build samples
def stratified_sample(dataframe, frac, seed=42):
    return (
        dataframe.groupby("decision", group_keys=False)
        .apply(lambda g: g.sample(max(1, int(len(g) * frac)), random_state=seed))
        .sample(frac=1, random_state=seed)  # shuffle
        .reset_index(drop=True)
    )

# small sample used for prompt iteration and comparison
dev_sample = stratified_sample(df, frac=0.001).to_dict("records")   # about 63 rows
# the real baseline sample used for the final model comparison
eval_sample = stratified_sample(df, frac=0.10).to_dict("records")   # about 6,406 rows


# 3. Prompts
LABEL_MAP = {"UPHELD": "Upheld", "OVERTURNED": "Overturned"}

PROMPT_TEMPLATE = """You are reviewing an independent medical review (IMR) appeal \
of a health insurance coverage denial. Based on the facts below, decide whether \
the insurer's denial was UPHELD or OVERTURNED on appeal.

Case:
{case_text}

Respond with exactly one word: UPHELD or OVERTURNED."""

def build_prompt(case_text):
    return PROMPT_TEMPLATE.format(case_text=case_text)

# Few-shot version, 4 clean cases which I chose for testing. Pulled from `text`
# only, never `full_text`, for the same leakage reason from before.
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

# 4. Model Evaluation (with token/cost tracking)

def predict(case_text, model="gpt-5-mini", prompt_fn=build_prompt):
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

def evaluate(sample, model="gpt-5-mini", prompt_fn=build_prompt):
    """Sequential evaluator -- used for small dev-sample comparisons."""
    correct = 0
    unparsed = 0
    mismatches = []
    for row in sample:
        pred, _, _ = predict(row["text"], model=model, prompt_fn=prompt_fn)
        if pred is None:
            unparsed += 1
            continue
        if pred == row["decision"]:
            correct += 1
        else:
            mismatches.append({"text": row["text"], "true": row["decision"], "pred": pred})
    n_scored = len(sample) - unparsed
    accuracy = correct / n_scored if n_scored else 0
    return {"model": model, "n": len(sample), "unparsed": unparsed, "accuracy": accuracy, "mismatches": mismatches}

def evaluate_concurrent(sample, model="gpt-5-mini", prompt_fn=build_fewshot_prompt, max_workers=10):
    """Concurrent evaluator -- used for the full eval_sample run."""
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
        "mismatches": mismatches, "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }

# $ per 1M tokens: Pricing for models
PRICING = {
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
}

# 5. Main

if __name__ == "__main__":
    print("--- Dev sample, zero-shot prompt ---")
    dev_zeroshot = evaluate(dev_sample, model="gpt-5-mini", prompt_fn=build_prompt)
    print(f"accuracy={dev_zeroshot['accuracy']:.1%}, unparsed={dev_zeroshot['unparsed']}")

    print("\n--- Dev sample, few-shot prompt ---")
    dev_fewshot = evaluate(dev_sample, model="gpt-5-mini", prompt_fn=build_fewshot_prompt)
    print(f"accuracy={dev_fewshot['accuracy']:.1%}, unparsed={dev_fewshot['unparsed']}")
    # Few-shot outperformed zero-shot on the dev sample (66.7% vs 61.3%) and
    # is used as the prompt for the full run below.

    print("\n--- Full eval sample, by model size (few-shot prompt) ---")
    all_results = {}
    for model in ["gpt-5", "gpt-5-mini", "gpt-5-nano"]:
        print(f"=== Running {model} on full eval sample ({len(eval_sample)} rows) ===")
        start = time.time()
        result = evaluate_concurrent(eval_sample, model=model, prompt_fn=build_fewshot_prompt, max_workers=10)
        elapsed = time.time() - start

        in_price, out_price = PRICING[model]
        cost = (result["total_input_tokens"] / 1e6 * in_price) + (result["total_output_tokens"] / 1e6 * out_price)

        print(f"{model}: accuracy={result['accuracy']:.1%}, unparsed={result['unparsed']}, "
              f"time={elapsed/60:.1f}min, cost=${cost:.2f}")
        all_results[model] = {k: v for k, v in result.items() if k != "mismatches"}
        all_results[model]["elapsed_minutes"] = elapsed / 60
        all_results[model]["cost_usd"] = cost

    os.makedirs("results", exist_ok=True)
    with open("baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to baseline_results.json")
