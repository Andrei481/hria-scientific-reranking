"""CPU checks for our export orchestration; no model download or Unsloth import."""
import json
import importlib
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train


class ExportTests(unittest.TestCase):
    def test_reload_runs_outside_parent_unsloth_patches(self):
        exporter = importlib.import_module("export_checkpoint")
        verify = getattr(exporter, "verify_export_in_subprocess", None)
        self.assertTrue(callable(verify), "Reload must run in a clean subprocess")
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from transformers.models.llama.modeling_llama import LlamaAttention
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        import types

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "final"
            config = LlamaConfig(vocab_size=8, hidden_size=16, intermediate_size=32,
                                 num_hidden_layers=1, num_attention_heads=2,
                                 num_key_value_heads=2, max_position_embeddings=64,
                                 bos_token_id=1, eos_token_id=2, pad_token_id=0)
            LlamaForCausalLM(config).save_pretrained(checkpoint)
            backend = Tokenizer(WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2,
                                           "Rank": 3, "these": 4, "passages": 5,
                                           "by": 6, "relevance": 7}, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                               bos_token="[BOS]", eos_token="[EOS]", pad_token="[UNK]")
            tokenizer.save_pretrained(checkpoint)
            # Reproduce the important boundary: the parent has globally patched
            # Llama attention, while the child must reload with native Transformers.
            with patch.dict(sys.modules, {"unsloth": types.ModuleType("unsloth")}), \
                    patch.object(LlamaAttention, "forward", side_effect=AssertionError("parent patch leaked")):
                verify(checkpoint, device="cpu")
            report = json.loads((checkpoint / "reload_check.json").read_text())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["device"], "cpu")
            self.assertFalse(report["unsloth_imported"])
            self.assertEqual(report["logits_shape"][-1], 8)

    def test_verifier_refuses_already_patched_process(self):
        script = HERE / "verify_checkpoint.py"
        self.assertTrue(script.is_file(), "Missing isolated verifier")
        code = (
            "import runpy, sys, types; "
            "sys.modules['unsloth'] = types.ModuleType('unsloth'); "
            f"sys.argv = [{str(script)!r}, '--checkpoint', 'missing', '--device', 'cpu']; "
            f"runpy.run_path({str(script)!r}, run_name='__main__')"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fresh Python process", result.stderr)

    def test_export_only_refuses_existing_output_before_loading_model(self):
        script = HERE / "export_checkpoint.py"
        self.assertTrue(script.is_file(), "Missing export-only command")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "existing"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("preserve")
            result = subprocess.run(
                [sys.executable, "-B", str(script), "--run", str(root / "missing"),
                 "--output", str(output)], capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing to overwrite", result.stderr)
            self.assertEqual(marker.read_text(), "preserve")

    def test_export_only_rejects_unpinned_revision_before_loading_model(self):
        script = HERE / "export_checkpoint.py"
        self.assertTrue(script.is_file(), "Missing export-only command")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "training_manifest.json").write_text(json.dumps({
                "status": "smoke_only", "config": {"revision": "main", "quantization": "none"},
            }))
            result = subprocess.run([sys.executable, "-B", str(script), "--run", str(root)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Pinned 40-character", result.stderr)
            self.assertFalse((root / "export_check").exists())

    def test_saves_loaded_weights_and_preserves_premerge_parameter_count(self):
        self.assertTrue(callable(getattr(train, "save_exact_checkpoint", None)),
                        "Missing exact-base export helper")
        import torch

        # The GPU/PEFT boundary is replaced by a tiny, stateful model. In particular,
        # merging mutates the wrapper's parameters, as the real PEFT API can do.
        class Merged(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(weight, requires_grad=False)

            def save_pretrained(self, directory, *, safe_serialization, max_shard_size):
                self_settings = dict(safe_serialization=safe_serialization, max_shard_size=max_shard_size)
                path = Path(directory)
                path.mkdir(parents=True)
                torch.save(self.state_dict(), path / "weights.pt")
                (path / "settings.json").write_text(json.dumps(self_settings))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Parameter(torch.tensor([10.0, 20.0]), requires_grad=False)
                self.adapter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

            def save_pretrained(self, directory):
                path = Path(directory)
                path.mkdir(parents=True)
                torch.save(self.adapter.detach().clone(), path / "adapter.pt")

            def merge_and_unload(self, *, safe_merge):
                if not safe_merge:
                    raise AssertionError("Export must request PEFT's finite-weight guard")
                merged = Merged(self.base.detach() + self.adapter.detach())
                self.adapter = None
                return merged

            def save_pretrained_merged(self, *args, **kwargs):
                raise AssertionError("Mirror-resolving exporter must not be used")

        class Tokenizer:
            def save_pretrained(self, directory):
                (Path(directory) / "tokenizer.json").write_text('{"fixture": true}')

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            count = train.save_exact_checkpoint(Model(), Tokenizer(), root)
            self.assertEqual(count, 2)
            self.assertEqual(torch.load(root / "adapter/adapter.pt", weights_only=True).tolist(), [1.0, 2.0])
            weights = torch.load(root / "final/weights.pt", weights_only=True)
            self.assertEqual(weights["weight"].tolist(), [11.0, 22.0])
            self.assertEqual(json.loads((root / "final/settings.json").read_text()),
                             {"safe_serialization": True, "max_shard_size": "5GB"})
            self.assertTrue((root / "adapter/tokenizer.json").is_file())
            self.assertTrue((root / "final/tokenizer.json").is_file())


if __name__ == "__main__":
    unittest.main()
