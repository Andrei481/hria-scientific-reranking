"""Exercise GPU scheduling using real lightweight subprocesses, without CUDA."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import run_matrix


class ParallelTests(unittest.TestCase):
    def test_gpu_ids_must_be_nonempty_unique_indices(self):
        self.assertTrue(hasattr(run_matrix, 'parse_gpus'), 'GPU selection is not implemented')
        self.assertEqual(run_matrix.parse_gpus('0,2,7'), ['0', '2', '7'])
        for value in ['', '0,0', '0,00', '-1', '0,', 'gpu']:
            with self.assertRaises(ValueError):
                run_matrix.parse_gpus(value)

    def jobs(self, directory, fail=False):
        script = directory / 'worker.py'
        script.write_text('''import json, os, pathlib, sys, time
root, name, fail = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
gpu = os.environ['CUDA_VISIBLE_DEVICES']
lock = root / ('gpu_' + gpu)
with lock.open('x'):
    pass
start = time.monotonic()
time.sleep(0.2 if fail == 'yes' else 0.6)
end = time.monotonic()
(root / (name + '.json')).write_text(json.dumps(dict(gpu=gpu, start=start, end=end)))
lock.unlink()
print('finished ' + name, flush=True)
sys.exit(3 if fail == 'yes' else 0)
''', encoding='utf-8')
        return [(str(i), [sys.executable, '-B', str(script), str(directory), str(i),
                         'yes' if fail and i == 0 else 'no']) for i in range(4)]

    def test_parallel_workers_are_isolated_and_gpu_is_reused(self):
        self.assertTrue(hasattr(run_matrix, 'execute_parallel'), 'Parallel execution is not implemented')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_matrix.execute_parallel(self.jobs(root), ['0', '3'], root / 'logs')
            results = [json.loads((root / f'{i}.json').read_text()) for i in range(4)]
            self.assertEqual({r['gpu'] for r in results}, {'0', '3'})
            self.assertLess(max(results[0]['start'], results[1]['start']),
                            min(results[0]['end'], results[1]['end']))
            for gpu in ['0', '3']:
                runs = sorted([r for r in results if r['gpu'] == gpu], key=lambda r: r['start'])
                for first, second in zip(runs, runs[1:]):
                    self.assertLessEqual(first['end'], second['start'])
            self.assertEqual(len(list((root / 'logs').glob('*.log'))), 4)

    def test_failure_stops_queue_but_preserves_active_work(self):
        self.assertTrue(hasattr(run_matrix, 'execute_parallel'), 'Parallel execution is not implemented')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                run_matrix.execute_parallel(self.jobs(root, fail=True), ['0', '1'], root / 'logs')
            self.assertTrue((root / '0.json').exists())
            self.assertTrue((root / '1.json').exists())
            self.assertFalse((root / '2.json').exists())
            self.assertFalse((root / '3.json').exists())


if __name__ == '__main__':
    unittest.main()
