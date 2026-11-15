"""CPU integration checks; synthetic fixtures, no downloads or live model calls."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from experiment import read_jsonl, sha256_file, write_json
from prepare_evaluation import freeze_retriever
from evaluate import complete_query, main as evaluate_main


class FakeResponse:
    def __init__(self, payload, error=None):
        self.payload, self.error = payload, error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.payload


class FakeSession:
    """Only the HTTP boundary is substituted; real preparation/parsing/analysis run."""
    def __init__(self, name):
        self.name, self.headers, self.calls = name, {}, 0

    def get(self, url, timeout):
        return FakeResponse({"data": [{"id": self.name}]})

    def post(self, url, json, timeout):
        self.calls += 1
        assert json["model"] == self.name
        assert json["temperature"] == 0
        assert json["seed"] == 0
        order = list(range(1, 101))
        if self.name.startswith("epoch_random"):
            order.reverse()
        reply = " > ".join(f"[{i}]" for i in order)
        return FakeResponse({"choices": [{"message": {"content": reply}, "finish_reason": "stop"}],
                             "usage": {"completion_tokens": 399}})


class PipelineTests(unittest.TestCase):
    def test_freezing_checks_candidate_integrity_and_never_overwrites(self):
        row = dict(query="q", retrieved=[1, 2], corpusids=[2])
        corpus = {1: "First abstract", 2: "Second abstract"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "inputs.jsonl"
            freeze_retriever([row], corpus, output, expected_count=1, k=2)
            saved = read_jsonl(output)[0]
            self.assertEqual(saved["pre_reranked"], [1, 2])
            self.assertIn("[2] Second abstract", saved["prompt"])
            with self.assertRaises(FileExistsError):
                freeze_retriever([row], corpus, output, expected_count=1, k=2)
            with self.assertRaisesRegex(ValueError, "unique"):
                freeze_retriever([dict(row, retrieved=[1, 1])], corpus,
                                 Path(directory) / "bad.jsonl", expected_count=1, k=2)
            with self.assertRaisesRegex(ValueError, "Duplicate query"):
                freeze_retriever([row, row], corpus, Path(directory) / "duplicate.jsonl", expected_count=2, k=2)

    def test_api_errors_propagate_without_reducing_candidate_pool(self):
        class ErrorSession:
            calls = 0

            def post(self, *args, **kwargs):
                self.calls += 1
                return FakeResponse({}, RuntimeError("context length exceeded"))

        session = ErrorSession()
        row = dict(prompt="test", pre_reranked=list(range(100)), corpusids=[1])
        with self.assertRaisesRegex(RuntimeError, "context length"):
            complete_query(session, "http://fixture/v1", "fixture", row, 1024)
        self.assertEqual(session.calls, 1)
        self.assertEqual(len(row["pre_reranked"]), 100)

    def test_frozen_inputs_evaluation_resume_and_analysis_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            corpus = {i: f"Title: Document {i}\nAbstract: Test text" for i in range(1, 101)}
            rows = [dict(query=f"Question {i}", query_set="fixture", specificity=1,
                         retrieved=list(range(1, 101)), corpusids=[100]) for i in range(597)]
            files = {}
            for retriever in ("bm25", "grit", "qwen3"):
                path = inputs / f"{retriever}.jsonl"
                freeze_retriever(rows, corpus, path)
                files[retriever] = {"input_sha256": sha256_file(path), "queries": 597}
            write_json(inputs / "manifest.json", {"files": files, "fixture": True})
            for policy in ("deterministic", "epoch_random"):
                name = f"{policy}_seed42"
                run = root / "runs" / name
                write_json(run / "training_manifest.json", {
                    "status": "complete", "config": {"policy": policy, "seed": 42, "smoke_steps": 0},
                    "packages": {"fixture": True},
                    "preflight": {"schedule_sha256": "matched", "chat_template_sha256": "matched"},
                })
                session = FakeSession(name)
                argv = ["evaluate.py", "--run", str(run), "--inputs", str(inputs), "--served-model", name]
                with patch.object(sys, "argv", argv), patch.dict(os.environ, {"TIE_VLLM_VERSION": "fixture"}), \
                        patch("requests.Session", return_value=session), contextlib.redirect_stdout(io.StringIO()):
                    evaluate_main()
                    self.assertEqual(session.calls, 3 * 597)
                    evaluate_main()  # Same protocol resumes without making a second inference call.
                    self.assertEqual(session.calls, 3 * 597)
                summary = json.loads((run / "evaluation/summary.json").read_text())
                self.assertEqual(summary["bm25"]["R@5"], int(policy == "epoch_random"))
                self.assertEqual(summary["bm25"]["exact_permutation_rate"], 1)

            output = root / "analysis"
            command = [sys.executable, "-B", str(HERE / "analyze.py"), "--runs", str(root / "runs"),
                       "--seeds", "42", "--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("PILOT", result.stdout)
            with (output / "paired_uncertainty.csv").open() as handle:
                comparisons = list(csv.DictReader(handle))
            self.assertEqual(len(comparisons), 12)
            for row in comparisons:
                if row["metric"] in ("R@5", "R@20"):
                    self.assertEqual(float(row["delta_pp"]), 100)
                    self.assertEqual(float(row["ci_low_pp"]), 100)
            self.assertTrue(json.loads((output / "analysis_manifest.json").read_text())["pilot"])

            # File integrity is enforced before comparing a tampered run.
            damaged = root / "runs/epoch_random_seed42/evaluation/bm25.jsonl"
            with damaged.open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            command[-1] = str(root / "analysis_tampered")
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Evaluation output changed", result.stderr)


if __name__ == "__main__":
    unittest.main()
