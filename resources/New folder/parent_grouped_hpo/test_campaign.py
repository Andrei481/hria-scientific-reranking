"""CPU campaign checks; no GPU training or experimental scores are produced."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from protocol import ROOT, load_config, read_json, trial_parameters, write_json


class CampaignTests(unittest.TestCase):
    def module(self):
        path = ROOT / 'campaign.py'
        self.assertTrue(path.exists(), 'four-model grouped-selection campaign is required')
        spec = importlib.util.spec_from_file_location('hpo_campaign', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_grid_covers_all_models_and_includes_reference(self):
        configs = self.module().campaign_configs()
        self.assertEqual(set(configs), {'llama8b', 'gemma12b', 'qwen14b', 'gemma27b'})
        with tempfile.TemporaryDirectory() as tmp:
            for name, config in configs.items():
                p = Path(tmp) / (name + '.json')
                write_json(p, config)
                checked = load_config(p)
                settings = [trial_parameters(checked, t['id']) for t in checked['trials']]
                self.assertEqual(len(settings), 12)
                self.assertEqual(len({(s['learning_rate'], s['lora_rank'], s['lora_dropout']) for s in settings}), 12)
                self.assertTrue(all(s['lora_alpha'] == 2 * s['lora_rank'] for s in settings))
                self.assertEqual(trial_parameters(checked, checked['reference_trial']), checked['shared'])
        self.assertEqual(sum(len(c['trials']) * len(c['split_seeds']) for c in configs.values()), 144)
        self.assertTrue(configs['gemma27b']['load_in_4bit'])
        self.assertTrue(all(not c['load_in_4bit'] for k, c in configs.items() if k != 'gemma27b'))

    def test_prepare_is_immutable_and_status_does_not_invent_results(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'campaign'
            m.prepare_campaign(root)
            with self.assertRaises(FileExistsError):
                m.prepare_campaign(root)
            report = m.campaign_status(root)
            self.assertFalse(report['selection_complete'])
            self.assertFalse(report['refits_complete'])
            self.assertEqual(len(report['models']), 4)
            self.assertTrue(all(r['status'] == 'selection_pending' for r in report['models'].values()))
            self.assertIn('external', report['scope'].lower())
            commands = m.stage_commands(root, 'search')
            self.assertEqual(len(commands), 4)
            self.assertTrue(all('--execute' in c for c in commands))
            parallel = m.stage_commands(root, 'search', gpus='0,1,2,3,4,5,6,7')
            self.assertTrue(all(c[-2:] == ['--gpus', '0,1,2,3,4,5,6,7'] for c in parallel))
            config = read_json(root / 'llama8b' / 'config.json')
            config['shared']['epochs'] = 1
            write_json(root / 'llama8b' / 'config.json', config)
            with self.assertRaises(ValueError):
                m.campaign_status(root)

    def test_scope_cannot_be_reduced_by_editing_manifest(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'campaign'
            m.prepare_campaign(root)
            manifest = read_json(root / 'campaign.json')
            del manifest['models']['gemma27b']
            write_json(root / 'campaign.json', manifest)
            with self.assertRaisesRegex(ValueError, 'model coverage'):
                m.campaign_status(root)


if __name__ == '__main__':
    unittest.main()
