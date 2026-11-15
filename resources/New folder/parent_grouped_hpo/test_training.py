import importlib.util
from pathlib import Path
import unittest


class TrainingTests(unittest.TestCase):
    def test_gemma_loading_is_text_only(self):
        m = self.module()
        self.assertTrue(hasattr(m, 'model_interface'), 'model-family-aware loading is required')
        backend, flags = m.model_interface('google/gemma-3-27b-it')
        self.assertEqual(backend, 'FastModel')
        self.assertFalse(flags['finetune_vision_layers'])
        self.assertTrue(flags['finetune_language_layers'])
        self.assertEqual(m.model_interface('meta-llama/Llama-3.1-8B-Instruct'), ('FastLanguageModel', {}))

    def test_trainable_vision_and_unmaterialized_weights_are_rejected(self):
        m = self.module()
        self.assertTrue(hasattr(m, 'validate_trainable_model'), 'loaded-model checks are required')
        class Param:
            def __init__(self, trainable, meta=False):
                self.requires_grad, self.is_meta = trainable, meta
        class Model:
            def __init__(self, pairs):
                self.pairs = pairs
            def named_parameters(self):
                return iter(self.pairs)
        m.validate_trainable_model(Model([('language.lora_A', Param(True)), ('vision_tower.weight', Param(False))]))
        for pairs in [[('vision_tower.lora_A', Param(True))], [('language.lora_A', Param(True, True))], [('language.weight', Param(False))]]:
            with self.assertRaises(ValueError):
                m.validate_trainable_model(Model(pairs))

    def test_gemma_targets_exclude_vision_projections(self):
        m = self.module()
        self.assertTrue(hasattr(m, 'language_target_modules'))
        class Model:
            def named_modules(self):
                return iter((name, object()) for name in [
                    'model.language_model.layers.0.self_attn.q_proj',
                    'model.language_model.layers.0.mlp.down_proj',
                    'model.vision_tower.layers.0.self_attn.q_proj',
                    'model.multi_modal_projector'])
        self.assertEqual(m.language_target_modules(Model()), [
            'model.language_model.layers.0.self_attn.q_proj',
            'model.language_model.layers.0.mlp.down_proj'])

    def module(self):
        path = Path(__file__).with_name('train.py')
        self.assertTrue(path.exists(), 'training runner is required')
        spec = importlib.util.spec_from_file_location('hpo_train', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_dataset_uses_only_supplied_partition_and_masks_prompt(self):
        m = self.module()
        encoded = [{'input_ids': [1, 2, 3, 4], 'response_start': 2},
                   {'input_ids': [5, 6, 7], 'response_start': 1}]
        dataset = m.EncodedDataset(encoded, [1])
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0], {'input_ids': [5, 6, 7], 'attention_mask': [1, 1, 1], 'labels': [-100, 6, 7]})
        self.assertNotIn('labels', encoded[1])

    def test_run_location_separates_smoke_and_refit(self):
        m = self.module()
        root = Path('/tmp/hpo')
        self.assertNotEqual(m.output_directory(root, 'trial', 'grouped', 42, 'a', 0),
                            m.output_directory(root, 'trial', 'grouped', 42, 'a', 2))
        self.assertNotEqual(m.output_directory(root, 'trial', 'grouped', 42, 'a', 0),
                            m.output_directory(root, 'refit', 'grouped', 42, 'a', 0))


if __name__ == '__main__':
    unittest.main()
