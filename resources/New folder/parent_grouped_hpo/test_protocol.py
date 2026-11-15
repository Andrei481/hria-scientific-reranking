import importlib.util
import math
from pathlib import Path
import unittest


class ProtocolTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).with_name('protocol.py')
        self.assertTrue(path.exists(), 'protocol implementation is required')
        spec = importlib.util.spec_from_file_location('hpo_protocol', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def fixture(self):
        parents = [{'parent_id': i, 'category': f'c{i // 20}'} for i in range(100)]
        rows = [{'example_id': f'{i}:{v}', 'parent_id': i} for i in range(100) for v in range(3)]
        return parents, rows

    def test_grouped_split_keeps_all_children_and_fixed_test(self):
        m = self.module()
        parents, rows = self.fixture()
        first = m.make_split(parents, rows, 42, 'grouped')
        second = m.make_split(parents, rows, 2026, 'grouped')
        self.assertEqual(first['parent_ids']['test'], second['parent_ids']['test'])
        for split in [first, second]:
            self.assertEqual([len(split['parent_ids'][s]) for s in m.PARTS], [80, 10, 10])
            self.assertEqual([len(split['indices'][s]) for s in m.PARTS], [240, 30, 30])
            self.assertEqual(split['train_validation_parent_overlap'], [])
        self.assertNotEqual(first['indices']['validation'], second['indices']['validation'])
        self.assertEqual(first, m.make_split(parents, rows, 42, 'grouped'))

    def test_list_control_preserves_test_and_permits_parent_overlap(self):
        m = self.module()
        parents, rows = self.fixture()
        grouped = m.make_split(parents, rows, 42, 'grouped')
        control = m.make_split(parents, rows, 42, 'list_control')
        self.assertEqual(grouped['indices']['test'], control['indices']['test'])
        self.assertGreater(len(control['train_validation_parent_overlap']), 0)
        self.assertFalse(set(control['parent_ids']['test']) & set(control['parent_ids']['train']))

    def test_categories_do_not_all_use_identical_positional_draws(self):
        m = self.module()
        parents, rows = self.fixture()
        split = m.make_split(parents, rows, 42, 'grouped')
        draws = {tuple(pid % 20 for pid in split['parent_ids']['test'] if pid // 20 == c) for c in range(5)}
        self.assertGreater(len(draws), 1)

    def test_invalid_seeds_and_nonfinite_parameters_are_rejected(self):
        import json
        import tempfile
        m = self.module()
        original = json.loads(Path(__file__).with_name('config.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            for changed in [{'split_seeds': ['../bad']}, {'shared': {**original['shared'], 'weight_decay': math.nan}}]:
                path.write_text(json.dumps({**original, **changed}))
                with self.assertRaises(ValueError):
                    m.load_config(path)

    def test_parent_macro_does_not_overweight_large_parents(self):
        m = self.module()
        scores = [{'parent_id': 1, 'nll': 1., 'response_tokens': 2},
                  {'parent_id': 1, 'nll': 3., 'response_tokens': 2},
                  {'parent_id': 2, 'nll': 8., 'response_tokens': 1}]
        result = m.summarize_losses(scores)
        self.assertEqual(result['parent_macro_nll'], 5.)
        self.assertEqual(result['list_macro_nll'], 4.)
        self.assertEqual(result['token_micro_nll'], 3.2)

    def test_rejects_nonfinite_scores(self):
        m = self.module()
        with self.assertRaises(ValueError):
            m.summarize_losses([{'parent_id': 1, 'nll': math.nan, 'response_tokens': 2}])

    def test_parent_mapping_ignores_reordered_source_ids(self):
        m = self.module()
        parents = [{'candidate_pool': [1, 2], 'query_type': 'A'},
                   {'candidate_pool': [3, 4], 'query_type': 'B'}]
        aspects = [{'aspect_id': 'a', 'query_text': 'qa', 'candidate_pool': [2, 1],
                    'candidate_scores': [2, 0], 'original_query_type': 'A'},
                   {'aspect_id': 'b', 'query_text': 'qb', 'candidate_pool': [3, 4],
                    'candidate_scores': [1, 0], 'original_query_type': 'B'}]
        row = {'aspect_id': 'b', 'query': 'qb', 'source_row_id': 0,
               'candidate_ids': [4, 3], 'labels': [0, 1], 'original_query_type': 'B'}
        self.assertEqual(m.resolve_parent(row, m.parent_mapping(parents, aspects)), 1)
        row['labels'] = [2, 1]
        with self.assertRaises(ValueError):
            m.resolve_parent(row, m.parent_mapping(parents, aspects))

    def test_reused_aspect_id_is_resolved_by_candidate_membership(self):
        m = self.module()
        parents = [{'candidate_pool': [1, 2], 'query_type': 'A'},
                   {'candidate_pool': [3, 4], 'query_type': 'A'}]
        aspects = [{'aspect_id': 'shared', 'query_text': 'q', 'candidate_pool': p['candidate_pool'],
                    'candidate_scores': [1, 0], 'original_query_type': 'A'} for p in parents]
        row = {'aspect_id': 'shared', 'query': 'q', 'candidate_ids': [3, 4],
               'labels': [1, 0], 'original_query_type': 'A'}
        self.assertEqual(m.resolve_parent(row, m.parent_mapping(parents, aspects)), 1)

    def test_selection_requires_complete_grid_and_excludes_smoke(self):
        m = self.module()
        self.assertTrue(hasattr(m, 'select_results'), 'selection implementation required')
        cfg = {'split_seeds': [42, 2026], 'trials': [{'id': 'old'}, {'id': 'new'}],
               'selection_metric': 'parent_macro_nll', 'reference_trial': 'old'}
        records = [{'protocol': 'grouped', 'split_seed': seed, 'trial_id': trial,
                    'status': 'complete', 'validation': {'parent_macro_nll': score}}
                   for seed in cfg['split_seeds'] for trial, score in [('old', 2.), ('new', 1.)]]
        result = m.select_results(cfg, records, ['grouped'])
        self.assertEqual(result['grouped']['winner'], 'new')
        self.assertFalse(result['grouped']['reference_retained'])
        with self.assertRaises(ValueError):
            m.select_results(cfg, records[:-1], ['grouped'])
        records[0]['status'] = 'smoke_only'
        with self.assertRaises(ValueError):
            m.select_results(cfg, records, ['grouped'])

    def test_response_encoding_masks_prompt_and_keeps_target(self):
        m = self.module()
        class Tokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                return 'P:' + messages[0]['content'] + ':A:' + (messages[1]['content'] + ':END' if len(messages) == 2 else '')
            def __call__(self, text, **kwargs):
                return {'input_ids': list(text.encode())}
        row = {'example_id': 'a', 'parent_id': 1, 'candidate_ids': [7, 9], 'labels': [0, 2], 'query': 'q'}
        encoded = m.encode_row(row, {'7': 'one', '9': 'two'}, Tokenizer())
        target = bytes(encoded['input_ids'][encoded['response_start']:]).decode()
        self.assertEqual(target, '[2] > [1]:END')


if __name__ == '__main__':
    unittest.main()
