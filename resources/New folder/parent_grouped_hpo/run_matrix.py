"""Print or execute fresh-checkpoint trials, optionally one per selected GPU."""
import argparse
from collections import deque
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

from protocol import ROOT, PROTOCOLS, digest, read_json, run_name, verify_study, verify_tokens


def parse_gpus(value):
    items = value.split(',')
    if any(not item.strip().isascii() or not item.strip().isdigit() for item in items):
        raise ValueError('--gpus must contain comma-separated nonnegative GPU indices')
    devices = [str(int(item.strip())) for item in items]
    if len(set(devices)) != len(devices):
        raise ValueError('--gpus must not contain duplicate GPU indices')
    return devices


def execute_parallel(jobs, gpus, log_dir):
    """Drain a queue; stop dispatch on failure, letting already active jobs finish."""
    log_dir.mkdir(parents=True, exist_ok=True)
    pending, available = deque(jobs), deque(gpus)
    active, failures = {}, []
    try:
        while active or (pending and not failures):
            # Collect every finished worker before dispatching any more work.
            for gpu, (process, stream, name) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                del active[gpu]
                available.append(gpu)
                print(f'GPU {gpu}: {name} exited {code}', flush=True)
                if code:
                    failures.append((name, code))
            while pending and available and not failures:
                gpu = available.popleft()
                name, command = pending.popleft()
                log = log_dir / (name.replace('/', '_').replace('\\', '_') + '.log')
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = gpu
                env['PYTHONUNBUFFERED'] = '1'
                stream = log.open('x', encoding='utf-8')
                try:
                    process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                except BaseException:
                    stream.close()
                    raise
                active[gpu] = (process, stream, name)
                print(f'GPU {gpu}: started {name}; log: {log}', flush=True)
            if active:
                time.sleep(0.1)
    finally:
        # Keyboard interruption or a launch error must not orphan GPU workers.
        for process, stream, _ in active.values():
            if process.poll() is None:
                process.terminate()
        for process, stream, _ in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            finally:
                stream.close()
    if failures:
        raise RuntimeError(f'Trials failed: {failures}; {len(pending)} queued trials not started. See {log_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--protocols', choices=PROTOCOLS, nargs='+', default=['grouped'])
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--gpus', help='GPU indices, e.g. 0,1,2,3; one trial per GPU. Overrides inherited CUDA_VISIBLE_DEVICES.')
    args = parser.parse_args()
    gpus = parse_gpus(args.gpus) if args.gpus is not None else None
    config, study, _ = verify_study(args.artifacts)
    if len(args.protocols) != len(set(args.protocols)):
        raise ValueError('Duplicate protocols')
    tokens = verify_tokens(args.artifacts, study) if args.execute else None
    completed = count = 0
    jobs = []
    for protocol in args.protocols:
        for seed in config['split_seeds']:
            for trial in config['trials']:
                count += 1
                path = args.artifacts / 'runs' / run_name(protocol, seed, trial['id']) / 'result.json'
                if args.execute and path.exists():
                    r = read_json(path)
                    if r['status'] != 'complete' or r['study_sha256'] != digest(study) or r['tokens_sha256'] != tokens['tokens_sha256']:
                        raise ValueError(f'Existing run has incompatible provenance: {path}')
                    completed += 1
                    print(f'Already complete: {path}', flush=True)
                    continue
                command = [sys.executable, '-B', str(ROOT / 'train.py'), '--artifacts', str(args.artifacts.resolve()),
                           '--protocol', protocol, '--split-seed', str(seed), '--trial', trial['id']]
                jobs.append((run_name(protocol, seed, trial['id']), command))
                print(shlex.join(command), flush=True)
    if gpus:
        print(f'Queue capacity: {len(gpus)} trials; GPU indices: {",".join(gpus)}. Each child sees exactly one GPU.', flush=True)
    if args.execute and jobs:
        if gpus:
            log_root = args.artifacts / 'logs'
            log_root.mkdir(parents=True, exist_ok=True)
            log_dir = Path(tempfile.mkdtemp(prefix='matrix_', dir=log_root))
            execute_parallel(jobs, gpus, log_dir)
        else:
            for _, command in jobs:
                subprocess.run(command, check=True)
    print(f'{count} planned trials; {completed} existing completed trials skipped. ' + ('Execution finished.' if args.execute else 'Dry run only.'))


if __name__ == '__main__':
    main()
