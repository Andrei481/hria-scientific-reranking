"""CPU integration tests. Synthetic trial scores exist only in temp directories."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from analyze import collect_results, load_selection, main as analyze_main
from protocol import (ROOT, digest, load_resources, read_json, run_name, sha256_file,
                      summarize_losses, trial_parameters, verify_study, write_json)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.artifacts = Path(self.temporary.name) / 'artifacts'
        subprocess.run([sys.executable, '-B', str(ROOT / 'prepare.py'), '--output', str(self.artifacts)],
                       check=True, capture_output=True, text=True)
        self.config, self.study, self.rows = verify_study(self.artifacts)

    def synthetic_results(self):
        tokenfile = self.artifacts / 'tokens' / 'encoded.jsonl.gz'
        tokenfile.parent.mkdir()
        tokenfile.write_bytes(b'SYNTHETIC TEST FIXTURE - NOT TRAINING TOKENS')
        tokenhash = sha256_file(tokenfile)
        write_json(tokenfile.parent / 'manifest.json', {'study_sha256': digest(self.study),
                   'tokens_sha256': tokenhash, 'revision': 'a' * 40})
        for seed in self.config['split_seeds']:
            split = read_json(self.artifacts / 'splits' / f'grouped_{seed}.json')
            for ordinal, trial in enumerate(self.config['trials']):
                directory = self.artifacts / 'runs' / run_name('grouped', seed, trial['id'])
                scores = [{'example_id': self.rows[i]['example_id'], 'parent_id': self.rows[i]['parent_id'],
                           'nll': 1.0 + ordinal, 'response_tokens': 10} for i in split['indices']['validation']]
                write_json(directory / 'validation_scores.json', scores)
                write_json(directory / 'result.json', {
                    'study_sha256': digest(self.study), 'tokens_sha256': tokenhash,
                    'protocol': 'grouped', 'split_seed': seed, 'trial_id': trial['id'],
                    'mode': 'trial', 'smoke_steps': 0, 'model': self.config['model'], 'revision': 'a' * 40,
                    'training_seed': self.config['training_seed'], 'hyperparameters': trial_parameters(self.config, trial['id']),
                    'training_lists': len(split['indices']['train']), 'training_parents': 80,
                    'packages': {'fixture': 'not-a-real-environment'}, 'status': 'complete',
                    'validation_scores_sha256': sha256_file(directory / 'validation_scores.json'),
                    'validation': summarize_losses(scores),
                })

    def test_actual_data_and_split_audit(self):
        parents, rows, docs, manifest = load_resources()
        self.assertEqual((len(parents), len(rows), len(docs)), (100, 4494, 8477))
        self.assertEqual(manifest['audit']['lists_above_five_positives'], 1392)
        fixed_test = None
        for seed in self.config['split_seeds']:
            grouped = read_json(self.artifacts / 'splits' / f'grouped_{seed}.json')
            self.assertEqual(grouped['train_validation_parent_overlap'], [])
            self.assertEqual(len(grouped['parent_ids']['test']), 10)
            if fixed_test is not None:
                self.assertEqual(fixed_test, grouped['parent_ids']['test'])
            fixed_test = grouped['parent_ids']['test']

    def test_analysis_and_selection_round_trip(self):
        self.synthetic_results()
        old = sys.argv
        try:
            sys.argv = ['analyze.py', '--artifacts', str(self.artifacts)]
            with contextlib.redirect_stdout(io.StringIO()):
                analyze_main()
        finally:
            sys.argv = old
        selection = load_selection(self.artifacts)
        self.assertEqual(selection['summary']['grouped']['winner'], 'lr_5e-6')
        self.assertFalse(selection['summary']['grouped']['reference_retained'])
        self.assertTrue((self.artifacts / 'analysis' / 'validation_summary.csv').exists())
        path = self.artifacts / 'runs' / 'grouped' / 'split_42' / 'lr_5e-6' / 'result.json'
        result = read_json(path)
        result['packages'] = {'fixture': 'different-environment'}
        write_json(path, result)
        with self.assertRaisesRegex(ValueError, 'package versions differ'):
            collect_results(self.artifacts, ['grouped'])

    def test_wrong_validation_partition_rejected_even_with_updated_hash(self):
        self.synthetic_results()
        directory = self.artifacts / 'runs' / 'grouped' / 'split_42' / 'lr_5e-6'
        scores = read_json(directory / 'validation_scores.json')
        scores[0]['example_id'] = 'wrong-parent-example'
        write_json(directory / 'validation_scores.json', scores)
        result = read_json(directory / 'result.json')
        result['validation_scores_sha256'] = sha256_file(directory / 'validation_scores.json')
        write_json(directory / 'result.json', result)
        with self.assertRaisesRegex(ValueError, 'validation examples'):
            collect_results(self.artifacts, ['grouped'])

    def test_configuration_mutation_rejected(self):
        path = self.artifacts / 'config.json'
        config = read_json(path)
        config['training_seed'] = 123
        write_json(path, config)
        with self.assertRaisesRegex(ValueError, 'Frozen configuration'):
            verify_study(self.artifacts)


if __name__ == '__main__':
    unittest.main()
