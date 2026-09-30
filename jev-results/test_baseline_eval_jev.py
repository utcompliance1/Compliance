"""Offline checks for sampling, API integration, and metric denominators."""

import json
import tempfile
import unittest
from pathlib import Path

import httpx2
import pandas as pd
from typesafe_sdk import RetryPolicy, TypeSafeClient

from baseline_eval_jev import MODEL, case_records, evaluate, make_sample, predict, summarize
from prompts import BASIC, PROMPTS


def response(choice="Upheld"):
    return {
        "model": MODEL,
        "answers": {"decision": {
            "type": "choice", "choice": choice, "confidence": 0.8,
            "probabilities": {"Upheld": 0.9, "Overturned": 0.1},
        }},
        "usage": {"input_tokens": 300, "output_tokens": 30},
    }


class BaselineTests(unittest.TestCase):
    def test_full_split_keeps_natural_class_balance_and_all_cases(self):
        frame = pd.DataFrame([
            {"row_id": i, "text": f"Case {i}", "decision": "Upheld" if i < 3 else "Overturned",
             "full_text": "LEAK"} for i in range(5)
        ])
        rows = case_records(frame)
        self.assertEqual(len(rows), 5)
        self.assertEqual(sum(row["decision"] == "Upheld" for row in rows), 3)
        self.assertTrue(all("full_text" not in row for row in rows))

    def test_concurrent_evaluation_saves_every_case_once(self):
        def handle(request):
            return httpx2.Response(200, json=response())

        sample = [{"row_id": i, "text": f"Case {i}", "decision": "Upheld"} for i in range(20)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.jsonl"
            with TypeSafeClient(api_key="test-only", transport=httpx2.MockTransport(handle)) as client:
                metrics, interrupted, _ = evaluate(sample, client, MODEL, path, max_workers=4)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual({row["row_id"] for row in rows}, set(range(20)))
        self.assertEqual(len(rows), 20)
        self.assertEqual(metrics["scored"], 20)
        self.assertFalse(interrupted)

    def test_sampling_is_balanced_reproducible_and_preserves_ids(self):
        frame = pd.DataFrame([
            {"text": f"Case {i}", "row_id": 100 + i,
             "decision": "Upheld" if i < 20 else "Overturned", "full_text": "LEAK"}
            for i in range(30)
        ])
        sample = make_sample(frame, 10)
        self.assertEqual(sample, make_sample(frame, 10))
        self.assertEqual(sum(row["decision"] == "Upheld" for row in sample), 5)
        self.assertEqual(len({row["row_id"] for row in sample}), 10)
        for row in sample:
            self.assertEqual(row["row_id"], 100 + row["test_index"])
            self.assertNotIn("full_text", row)
        for n in (1, 3, 22):
            with self.assertRaises(ValueError):
                make_sample(frame, n)

    def test_real_sdk_request_and_response_contract(self):
        def handle(request):
            body = json.loads(request.content)
            self.assertEqual(str(request.url), "https://api.typesafe.ai/v1/systemone")
            self.assertEqual(body["state"], "The case description")
            self.assertEqual(set(body), {"model", "state", "questions"})
            self.assertEqual(set(body["questions"]["decision"]["criteria"]), {"Upheld", "Overturned"})
            self.assertNotIn("full_text", request.content.decode())
            return httpx2.Response(200, json=response())

        with TypeSafeClient(api_key="test-only", base_url="https://api.typesafe.ai",
                            transport=httpx2.MockTransport(handle)) as client:
            result = predict(client, "The case description", MODEL)
        self.assertEqual(result["pred"], "Upheld")
        self.assertEqual(result["input_tokens"], 300)
        self.assertEqual(result["probabilities"]["Overturned"], 0.1)

    def test_prompt_selection_reaches_sdk_without_changing_case_input(self):
        seen = []

        def handle(request):
            seen.append(json.loads(request.content))
            return httpx2.Response(200, json=response())

        with TypeSafeClient(api_key="test-only", transport=httpx2.MockTransport(handle)) as client:
            for questions in PROMPTS.values():
                predict(client, "Fixed case description", MODEL, questions)
        self.assertEqual([body["questions"] for body in seen], list(PROMPTS.values()))
        self.assertEqual({body["state"] for body in seen}, {"Fixed case description"})
        # The original experimental condition remains reproducible.
        self.assertIsInstance(BASIC["decision"]["instructions"], str)
        self.assertIsInstance(PROMPTS["structured"]["decision"]["criteria"]["Upheld"], dict)

    def test_failed_cases_do_not_inflate_accuracy_or_coverage(self):
        sample = [{"decision": "Upheld"}, {"decision": "Overturned"}] * 2
        records = [
            {"true": "Upheld", "pred": "Upheld", "input_tokens": 100, "latency_seconds": 0.1},
            {"true": "Overturned", "pred": "Upheld", "input_tokens": 200, "latency_seconds": 0.2},
            {"true": "Upheld", "pred": None, "latency_seconds": 0.3},
        ]
        metrics = summarize(sample, records, 1)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["accuracy_all_requested"], 0.25)
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual((metrics["failed"], metrics["unattempted"]), (1, 1))
        self.assertEqual(metrics["confusion_matrix"]["Overturned"]["Upheld"], 1)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["estimated_cost_usd"], 0.0000126)
        self.assertIsNone(summarize(sample, [], 0)["accuracy"])

    def test_auth_failure_stops_after_first_call_and_saves_record(self):
        calls = []

        def handle(request):
            calls.append(request)
            return httpx2.Response(401, json={"error": "Unauthorized"})

        sample = [{"row_id": i, "text": "Case", "decision": "Upheld"} for i in range(4)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.jsonl"
            with TypeSafeClient(api_key="test-only", transport=httpx2.MockTransport(handle),
                                retry=RetryPolicy(max_retries=0)) as client:
                metrics, interrupted, models = evaluate(sample, client, MODEL, path)
            self.assertEqual(len(calls), 1)
            self.assertEqual(json.loads(path.read_text())["http_status"], 401)
        self.assertEqual((metrics["failed"], metrics["unattempted"]), (1, 3))
        self.assertFalse(interrupted)
        self.assertEqual(models, [])


if __name__ == "__main__":
    unittest.main()
