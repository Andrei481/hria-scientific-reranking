"""Score verified complete rankings with the original CSFCube evaluator and folds."""
import argparse
import contextlib
import importlib
import io
from pathlib import Path
import runpy
import sys

from external_protocol import (FACETS, digest, model_aliases, read_json, read_jsonl, sha256_file,
                               validate_prefix, verify_manifest, write_json)


def run_official(root, output, alias, facet):
    sys.path.insert(0, str(root / 'eval_scripts'))
    rm = importlib.import_module('rank_metrics')
    # NumPy 2 removed asfarray. The original evaluator uses it only on numeric
    # relevance arrays; float asarray is equivalent for this input domain.
    patched = not hasattr(rm.np, 'asfarray')
    if patched:
        rm.np.asfarray = lambda values, dtype=float: rm.np.asarray(values, dtype=dtype)
    old_argv = sys.argv
    sys.argv = ['ranking_eval.py', 'eval_pool_ranking', '--gold_path', str(root),
                '--ranked_path', str(output), '--experiment', alias, '--facet', facet]
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            runpy.run_path(str(root / 'eval_scripts/ranking_eval.py'), run_name='__main__')
    finally:
        sys.argv = old_argv
        sys.path.pop(0)
        if patched:
            del rm.np.asfarray
    return captured.getvalue(), {'numpy': rm.np.__version__, 'asfarray_compatibility': patched}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', required=True, type=Path)
    parser.add_argument('--model', choices=['grouped', 'base'], default='grouped')
    args = parser.parse_args()
    manifest = verify_manifest(args.artifacts)
    output = args.artifacts / 'results' / args.model
    summary, protocol = read_json(output / 'summary.json'), read_json(output / 'protocol.json')
    alias = model_aliases(manifest)[args.model]
    experiment = 'hpo_' + args.model
    serving = read_json(args.artifacts / 'serving.json')
    expected = {'manifest_sha256': digest(manifest), 'serving_sha256': digest(serving),
                'model_alias': alias, 'limit': 0, 'temperature': 0, 'top_p': 1, 'seed': 0}
    if protocol != expected or summary['status'] != 'complete' or summary['protocol_sha256'] != digest(protocol):
        raise ValueError('Scoring requires a complete full evaluation with matching provenance')
    for facet in FACETS:
        key = 'csfcube_' + facet
        rows = read_jsonl(manifest['datasets'][key]['path'])
        path = output / f'{key}.jsonl'
        done = read_jsonl(path)
        if len(done) != len(rows) or sha256_file(path) != summary['datasets'][key]['raw_sha256']:
            raise ValueError('CSFCube output is incomplete or changed')
        validate_prefix(rows, done)
        if any(r['model_alias'] != alias for r in done):
            raise ValueError('Incorrect model alias in saved outputs')
        ranks = {row['paper_id']: [[pid, rank] for rank, pid in enumerate(r['retrieved'])]
                 for row, r in zip(rows, done)}
        if read_json(output / f'test-pid2pool-csfcube-{experiment}-{facet}-ranked.json') != ranks:
            raise ValueError('Official ranking input differs from raw generation')
    for facet in ('all', *FACETS):
        report, environment = run_official(Path(manifest['csfcube_root']), output, experiment, facet)
        (output / f'csfcube_{facet}_metrics.txt').write_text(report, encoding='utf-8')
        print(f'Facet: {facet}\n{report}')
    write_json(output / 'metrics_environment.json', environment)


if __name__ == '__main__':
    main()
