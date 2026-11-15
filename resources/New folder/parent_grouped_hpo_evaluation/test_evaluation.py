"""CPU checks; synthetic responses are not benchmark results."""
import importlib.util
import ast
import json
from pathlib import Path
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import unittest
from unittest.mock import patch


class EvaluationTests(unittest.TestCase):
    def module(self, name='external_protocol'):
        path = Path(__file__).with_name(name + '.py')
        self.assertTrue(path.exists(), 'external evaluation helpers are required')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_repair_keeps_first_valid_then_original_missing_order(self):
        m = self.module()
        order, audit = m.parse_ranking('[3] > [3] > [99] > [1]', 4)
        self.assertEqual(order, [2, 0, 1, 3])
        self.assertEqual((audit['duplicate_ids'], audit['invalid_ids'], audit['missing_ids']), (1, 1, 2))
        self.assertEqual(m.parse_ranking('no ranking', 3)[0], [0, 1, 2])

    def test_litsearch_metrics_use_all_gold_documents(self):
        m = self.module()
        scores = m.litsearch_metrics([1, 3, 5], [1, 2])
        self.assertEqual(scores['R@5'], 0.5)
        self.assertAlmostEqual(scores['nDCG@5'], 1 / (1 + 1 / __import__('math').log2(3)))

    def test_csfcube_modes_preserve_original_or_explicit_facet(self):
        m = self.module()
        papers = {'q': {'title': 'Query', 'abstract': ['One.', 'Two.']},
                  'a': {'title': 'Candidate', 'abstract': ['Three.', 'Four.']}}
        original = m.csfcube_prompt('method', 'q', ['a'], papers, 'original')
        self.assertIn("Abstract: ['One.', 'Two.']", original)
        self.assertNotIn('Facet:', original)
        facet = m.csfcube_prompt('method', 'q', ['a'], papers, 'facet')
        self.assertIn('Abstract: One. Two.', facet)
        self.assertIn('Facet: Candidates that solve problems using a similar technique or experimental setup.', facet)
        self.assertNotEqual(facet, m.csfcube_prompt('result', 'q', ['a'], papers, 'facet'))

    def test_resume_rejects_wrong_queries_and_repaired_ranking(self):
        m = self.module()
        row = {'qid': 'q1', 'pre_reranked': ['a', 'b'], 'prompt': 'prompt', 'corpusids': ['a']}
        good = {'qid': 'q1', 'input_sha256': m.digest(row), 'raw_response': '[2] > [1]',
                'retrieved': ['b', 'a']}
        m.validate_prefix([row], [good])
        for broken in [{**good, 'qid': 'q2'}, {**good, 'retrieved': ['a', 'b']}]:
            with self.assertRaises(ValueError):
                m.validate_prefix([row], [broken])

    def test_adapter_hashes_detect_modification(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'adapter_model.safetensors'
            path.write_bytes(b'fixture')
            hashes = {path.name: m.sha256_file(path)}
            m.verify_files(path.parent, hashes)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                m.verify_files(path.parent, hashes)

    def test_server_command_pins_model_and_explicit_rank128_adapter(self):
        m = self.module('serve')
        command = m.server_command({'model': 'meta-llama/Llama-3.1-8B-Instruct', 'revision': 'a'*40,
                                    'adapter': '/data/adapter', 'lora_rank': 128, 'max_model_len': 131072}, 8000)
        self.assertEqual(command[command.index('--revision')+1], 'a'*40)
        self.assertEqual(command[command.index('--max-lora-rank')+1], '128')
        self.assertEqual(command[command.index('--tokenizer')+1], '/data/adapter')
        adapter_spec = command[command.index('--lora-modules')+1]
        self.assertTrue(adapter_spec.startswith('hpo_grouped_'))
        self.assertTrue(adapter_spec.endswith('=/data/adapter'))

    def test_real_csfcube_cases_keep_complete_candidate_pools(self):
        m = self.module('prepare')
        locations = [Path(__file__).resolve().parents[i] / 'CSFCube' for i in (1, 2)]
        gold = next((p for p in locations if (p / 'abstracts-csfcube-preds.jsonl').exists()), None)
        if gold is None:
            self.skipTest('CSFCube checkout not present beside bundle or in repository root')
        rows = m.csfcube_rows(gold, 'original')
        self.assertEqual(sum(len(v) for v in rows.values()), 50)
        for facet, cases in rows.items():
            expected = json.loads((gold / f'test-pid2anns-csfcube-{facet}.json').read_text())
            for row in cases:
                self.assertEqual(row['pre_reranked'], expected[row['paper_id']]['cands'])

    def test_http_evaluation_explicit_adapter_and_resume(self):
        m = self.module('evaluate')
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({'data': [{'id': 'hpo_base'}, {'id': 'hpo_grouped'}]}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                body = json.dumps({'choices': [{'message': {'content': '[2] > [2] > [9]'},
                                                 'finish_reason': 'length'}], 'usage': {}}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'raw.jsonl'
                rows = [{'qid': 'one', 'pre_reranked': ['a', 'b', 'c'], 'prompt': 'fixture'}]
                url = f'http://127.0.0.1:{server.server_port}/v1'
                p = self.module()
                cohort_alias = p.model_aliases({'adapter': 'new-refit'})['grouped']
                with self.assertRaisesRegex(ValueError, 'this cohort model'):
                    p.require_model(url, cohort_alias)
                first = m.evaluate_dataset(rows, path, url, 'hpo_grouped', 1024)
                self.assertEqual(first[0]['retrieved'], ['b', 'a', 'c'])
                self.assertEqual(first[0]['finish_reason'], 'length')
                second = m.evaluate_dataset(rows, path, url, 'hpo_grouped', 1024)
                self.assertEqual(first, second)
                self.assertEqual(len(requests), 1)
                self.assertEqual(requests[0]['model'], 'hpo_grouped')
                self.assertEqual(requests[0]['max_tokens'], 1024)
                self.assertEqual(requests[0]['temperature'], 0)
                path.write_text('{"truncated":', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Incomplete final record'):
                    m.evaluate_dataset(rows, path, url, 'hpo_grouped', 1024)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_prompt_matches_retained_original_notebook(self):
        m = self.module()
        notebook = Path(__file__).resolve().parents[1] / 'csfcube_eval_original.ipynb'
        if not notebook.exists():
            self.skipTest('Retained original notebook is only needed for repository parity check')
        functions = []
        for cell in json.loads(notebook.read_text(encoding='utf-8'))['cells']:
            source = ''.join(cell.get('source', []))
            if cell['cell_type'] == 'code' and 'def build_rerank_prompt' in source:
                functions.extend(n for n in ast.parse(source).body
                                 if isinstance(n, ast.FunctionDef) and n.name == 'build_rerank_prompt')
        self.assertEqual(len(functions), 1)
        namespace = {}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(notebook), 'exec'), namespace)
        papers = {'q': {'title': 'Query', 'abstract': ['One.', 'Two.']},
                  'a': {'title': 'Candidate', 'abstract': ['Three.', 'Four.']}}
        self.assertEqual(m.csfcube_prompt('method', 'q', ['a'], papers, 'original'),
                         namespace['build_rerank_prompt']('q', ['a'], papers))

    def test_official_metrics_with_and_without_numpy_asfarray(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('NumPy required for official evaluator integration check')
        m = self.module('score_csfcube')
        p = self.module()
        locations = [Path(__file__).resolve().parents[i] / 'CSFCube' for i in (1, 2)]
        gold = next((x for x in locations if (x / 'queries-release.csv').exists()), None)
        if gold is None:
            self.skipTest('CSFCube checkout required for official evaluator integration check')
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for facet in p.FACETS:
                pools = p.read_json(gold / f'test-pid2anns-csfcube-{facet}.json')
                ranks = {qid: [[pid, i] for i, pid in enumerate(pool['cands'])] for qid, pool in pools.items()}
                p.write_json(output / f'test-pid2pool-csfcube-fixture-{facet}-ranked.json', ranks)
            report, _ = m.run_official(gold, output, 'fixture', 'all')
            self.assertIn('EVAL SPLIT: test', report)
            self.assertIn('Total queries: 50', report)
            original = getattr(np, 'asfarray', None)
            if original is not None:
                del np.asfarray
            try:
                compatibility_report, env = m.run_official(gold, output, 'fixture', 'all')
                self.assertTrue(env['asfarray_compatibility'])
                self.assertEqual(report, compatibility_report)
                self.assertFalse(hasattr(np, 'asfarray'))
            finally:
                if original is not None:
                    np.asfarray = original

    def test_main_separates_smoke_and_complete_and_resumes(self):
        m, p = self.module('evaluate'), self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = {}
            for key in ('litsearch_bm25', 'litsearch_grit', 'litsearch_qwen3',
                        'csfcube_background', 'csfcube_method', 'csfcube_result'):
                rows = [{'qid': str(i), 'paper_id': str(i), 'pre_reranked': ['a', 'b'],
                         'prompt': 'fixture', **({'corpusids': ['b']} if key.startswith('litsearch') else {})}
                        for i in (1, 2)]
                path = root / f'{key}.jsonl'
                path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
                datasets[key] = {'path': str(path), 'queries': 2, 'max_tokens': 32}
            manifest = {'datasets': datasets}
            aliases = p.model_aliases(manifest)
            serving = {'manifest_sha256': p.digest(manifest), 'adapter_alias': aliases['grouped'],
                       'base_alias': aliases['base'], 'url': 'http://fixture/v1'}
            p.write_json(root / 'serving.json', serving)
            response = {'choices': [{'message': {'content': '[2] > [1]'}, 'finish_reason': 'stop'}]}
            with patch.object(m, 'verify_manifest', return_value=manifest), \
                 patch.object(m, 'require_model'), patch.object(m, 'request_json', return_value=response) as calls, \
                 patch('builtins.print'):
                with patch('sys.argv', ['evaluate.py', '--artifacts', str(root), '--limit', '1']):
                    m.main()
                smoke = root / 'results_smoke' / 'grouped'
                self.assertEqual(p.read_json(smoke / 'summary.json')['status'], 'smoke_only')
                self.assertEqual(list(smoke.glob('*-ranked.json')), [])
                self.assertEqual(calls.call_count, 6)
                with patch('sys.argv', ['evaluate.py', '--artifacts', str(root)]):
                    m.main()
                    self.assertEqual(calls.call_count, 18)
                    m.main()
                    self.assertEqual(calls.call_count, 18)
                full = root / 'results' / 'grouped'
                summary = p.read_json(full / 'summary.json')
                self.assertEqual(summary['status'], 'complete')
                self.assertEqual(len(summary['datasets']), 6)
                self.assertEqual(len(list(full.glob('*-ranked.json'))), 3)
                self.assertEqual(summary['datasets']['litsearch_bm25']['metrics']['R@5'], 1)


if __name__ == '__main__':
    unittest.main()
