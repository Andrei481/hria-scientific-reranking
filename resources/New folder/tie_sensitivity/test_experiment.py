"""CPU checks for the scientific controls; run: python -m unittest -v."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class BundleExists(unittest.TestCase):
    def test_experiment_module_exists(self):
        self.assertTrue((HERE / "experiment.py").exists(), "experiment implementation is missing")


@unittest.skipUnless((HERE / "experiment.py").exists(), "implementation not created yet")
class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("experiment", HERE / "experiment.py")
        cls.e = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.e)

    def test_deterministic_preserves_prompt_position_inside_grades(self):
        self.assertEqual(self.e.target_indices([2, 0, 2, 1, 0], "deterministic", 42, "x", 0), [1, 3, 4, 2, 5])

    def test_randomization_changes_only_ties_and_is_reproducible(self):
        labels = [2] * 7 + [1] * 7 + [0] * 7
        first = self.e.target_indices(labels, "epoch_random", 42, "x", 0)
        second = self.e.target_indices(labels, "epoch_random", 42, "x", 1)
        self.assertNotEqual(first, second)
        self.assertEqual(first, self.e.target_indices(labels, "epoch_random", 42, "x", 0))
        for order in (first, second):
            self.assertEqual(sorted(order), list(range(1, 22)))
            self.assertEqual([labels[i - 1] for i in order], [2] * 7 + [1] * 7 + [0] * 7)
        self.assertEqual(self.e.target_indices(labels, "fixed_random", 42, "x", 0), self.e.target_indices(labels, "fixed_random", 42, "x", 1))

    def test_unknown_policy_and_invalid_labels_are_rejected(self):
        for labels, policy in [([3, 0], "deterministic"), ([1, 0], "typo"), ([], "deterministic")]:
            with self.assertRaises(ValueError):
                self.e.target_indices(labels, policy, 42, "x", 0)

    def test_schedule_visits_every_example_once_in_each_logical_epoch(self):
        schedule = self.e.epoch_schedule(7, 2, 42)
        self.assertEqual(len(schedule), 14)
        for epoch in (0, 1):
            self.assertEqual(sorted(i for e, i in schedule if e == epoch), list(range(7)))
        self.assertTrue(all(e == 0 for e, _ in schedule[:7]))
        self.assertTrue(all(e == 1 for e, _ in schedule[7:]))

    def test_audit_preserves_actual_51_document_lists(self):
        row = dict(query="q", docs=["d"] * 51, labels=[2] * 10 + [0] * 41,
                   candidate_ids=list(range(51)), source_row_id=0, variant_id=0)
        audit = self.e.audit_rows([row])
        self.assertEqual(audit["list_size_histogram"], {"51": 1})
        self.assertEqual(audit["lists_above_five_positives"], 1)
        bad = dict(row, candidate_ids=[0] * 51)
        with self.assertRaises(ValueError):
            self.e.audit_rows([bad])

    def test_duplicate_example_keys_are_rejected(self):
        row = dict(query="q", docs=["d"], labels=[2], candidate_ids=[5], source_row_id=0, variant_id=0)
        with self.assertRaises(ValueError):
            self.e.audit_rows([row, row])

    def test_parser_measures_raw_errors_before_repair(self):
        order, audit = self.e.parse_ranking("[3] > [3] > [99] > [1]", 4)
        self.assertEqual(order, [2, 0, 1, 3])
        self.assertEqual(audit["duplicate_ids"], 1)
        self.assertEqual(audit["invalid_ids"], 1)
        self.assertEqual(audit["missing_ids"], 2)
        self.assertFalse(audit["exact_permutation"])
        self.assertFalse(audit["fallback"])
        _, clean = self.e.parse_ranking("[2] > [1]", 2)
        self.assertTrue(clean["exact_permutation"])
        _, extra = self.e.parse_ranking("Here is my ranking: [2] > [1]", 2)
        self.assertTrue(extra["exact_permutation"])
        self.assertFalse(extra["format_only"])
        order, empty = self.e.parse_ranking("no ranking", 3)
        self.assertEqual(order, [0, 1, 2])
        self.assertTrue(empty["fallback"])

    def test_metrics_use_full_gold_set_and_log_discount(self):
        scores = self.e.litsearch_metrics([3, 2, 1], [1, 3])
        self.assertEqual(scores["R@5"], 1.0)
        self.assertAlmostEqual(scores["nDCG@20"], 1.5 / (1 + 1 / __import__("math").log2(3)))

    def test_response_masking_keeps_prompt_and_padding_out_of_loss(self):
        # A deterministic character tokenizer is sufficient to check the boundary
        # contract without downloading a model. It returns complete chat prefixes.
        class Tokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                text = "".join(m["role"] + ":" + m["content"] + "|" for m in messages)
                return text + ("assistant:" if add_generation_prompt else "")
            def __call__(self, text, **kwargs):
                return {"input_ids": [ord(c) for c in text]}
        row = dict(query="q", docs=["a", "b"], labels=[0, 2], source_row_id=0, variant_id=0)
        encoded = self.e.encode_example(row, Tokenizer(), "deterministic", 42, 0, 10000)
        supervised = "".join(chr(i) for i in encoded["labels"] if i != -100)
        self.assertEqual(supervised, "[2] > [1]|")
        self.assertIn(-100, encoded["labels"])
        with self.assertRaises(ValueError):
            self.e.encode_example(row, Tokenizer(), "deterministic", 42, 0, 10)

    def test_manifest_rejects_wrong_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lists.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            manifest = {"data_sha256": "0" * 64}
            with self.assertRaises(ValueError):
                self.e.check_data_hash(path, manifest)

    def test_scheduled_training_and_padding(self):
        from train import ScheduledDataset, ResponseCollator
        import torch
        class Tokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                return "".join(m["role"] + ":" + m["content"] + "|" for m in messages) + ("assistant:" if add_generation_prompt else "")
            def __call__(self, text, **kwargs):
                return {"input_ids": [ord(c) for c in text]}
        rows = [dict(query="q", docs=["a"] * 12, labels=[2] * 6 + [0] * 6, source_row_id=0, variant_id=0)]
        dataset = ScheduledDataset(rows, Tokenizer(), "epoch_random", 42, 10000)
        self.assertEqual(len(dataset), 2)
        self.assertNotEqual(dataset[0]["labels"], dataset[1]["labels"])
        features = [dict(input_ids=[1, 2, 3], attention_mask=[1, 1, 1], labels=[-100, 2, 3]),
                    dict(input_ids=[1, 4], attention_mask=[1, 1], labels=[-100, 4])]
        batch = ResponseCollator(0)(features)
        self.assertEqual(batch["labels"].tolist(), [[-100, 2, 3], [-100, 4, -100]])
        # Real backward pass checks the padded-label contract with causal shifting.
        layer = torch.nn.Linear(5, 5)
        logits = layer(torch.nn.functional.one_hot(batch["input_ids"], 5).float())
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 5), batch["labels"][:, 1:].reshape(-1))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


class AnalysisTests(unittest.TestCase):
    def test_analysis_module_exists(self):
        self.assertTrue((HERE / "analyze.py").exists(), "paired analysis implementation missing")

    @unittest.skipUnless((HERE / "analyze.py").exists(), "analysis not yet created")
    def test_paired_bootstrap_and_holm(self):
        from analyze import paired_stats, holm
        identical = paired_stats([0.0] * 30, 42, replicates=100)
        self.assertEqual((identical["delta"], identical["ci_low"], identical["ci_high"], identical["p"]), (0, 0, 0, 1))
        improvement = paired_stats([0.1] * 30, 42, replicates=100)
        self.assertAlmostEqual(improvement["delta"], 0.1)
        self.assertLess(improvement["p"], 0.05)
        self.assertEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    @unittest.skipUnless((HERE / "analyze.py").exists(), "analysis not yet created")
    def test_alignment_rejects_candidate_mismatch(self):
        from analyze import align_rows
        a = dict(qid="q", pre_reranked=[1, 2], corpusids=[1], retrieved=[2, 1])
        with self.assertRaises(ValueError):
            align_rows([a], [dict(a, pre_reranked=[2, 1])], expected=1)
        with self.assertRaises(ValueError):
            align_rows([a, a], [a, a], expected=2)
        self.assertEqual(align_rows([a], [dict(a, retrieved=[1, 2])], expected=1)[0]["qid"], "q")


if __name__ == "__main__":
    unittest.main()
