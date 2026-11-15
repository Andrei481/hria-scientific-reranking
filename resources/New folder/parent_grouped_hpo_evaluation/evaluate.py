"""Generate rankings for frozen pools, retaining raw outputs and safe prefix resume."""
import argparse
import json
from pathlib import Path

from external_protocol import (digest, litsearch_metrics, model_aliases, parse_ranking, read_json, read_jsonl,
                               request_json, require_model, sha256_file, validate_prefix, verify_manifest, write_json)


def evaluate_dataset(rows, path, url, alias, max_tokens):
    # Resume only from complete, newline-terminated JSONL records.
    if path.exists() and path.stat().st_size:
        with path.open('rb') as f:
            f.seek(-1, 2)
            if f.read(1) != b'\n':
                raise ValueError(f'Incomplete final record in {path}; preserve file before recovery')
    done = read_jsonl(path) if path.exists() else []
    validate_prefix(rows, done)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as output:
        for row in rows[len(done):]:
            response = request_json(url + '/chat/completions', {
                'model': alias, 'messages': [{'role': 'user', 'content': row['prompt']}],
                'temperature': 0, 'top_p': 1, 'seed': 0, 'max_tokens': max_tokens, 'stream': False})
            choice = response['choices'][0]
            raw = choice['message']['content']
            if not isinstance(raw, str):
                raise ValueError('Server returned non-text content; no output recorded')
            order, audit = parse_ranking(raw, len(row['pre_reranked']))
            record = {'qid': row['qid'], 'input_sha256': digest(row), 'raw_response': raw,
                      'retrieved': [row['pre_reranked'][i] for i in order], 'parser': audit,
                      'finish_reason': choice.get('finish_reason'), 'usage': response.get('usage'),
                      'model_alias': alias}
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            output.flush()
            done.append(record)
            if len(done) % 25 == 0 or len(done) == len(rows):
                print(f'{path.stem}: {len(done)}/{len(rows)}', flush=True)
    return done


def summarize(rows, done):
    validate_prefix(rows, done)
    audits = [parse_ranking(r['raw_response'], len(row['pre_reranked']))[1]
              for row, r in zip(rows, done)]
    summary = {'queries': len(done), 'exact_permutations': sum(a['exact_permutation'] for a in audits),
               'fallbacks': sum(a['fallback'] for a in audits),
               'outputs_with_missing_ids': sum(a['missing_ids'] > 0 for a in audits),
               'length_finishes': sum(r['finish_reason'] == 'length' for r in done)}
    if rows and 'corpusids' in rows[0]:
        values = [litsearch_metrics(r['retrieved'], row['corpusids']) for row, r in zip(rows, done)]
        summary['metrics'] = {k: sum(v[k] for v in values) / len(values) for k in values[0]}
        inputs = [litsearch_metrics(row['pre_reranked'], row['corpusids']) for row in rows]
        summary['retriever_metrics'] = {k: sum(v[k] for v in inputs) / len(inputs) for k in inputs[0]}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', required=True, type=Path)
    parser.add_argument('--model', choices=['grouped', 'base'], default='grouped')
    parser.add_argument('--limit', type=int, default=0, help='Positive: smoke only, this many queries per pool')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    manifest = verify_manifest(args.artifacts)
    serving = read_json(args.artifacts / 'serving.json')
    if serving['manifest_sha256'] != digest(manifest):
        raise ValueError('Serving configuration belongs to a different evaluation manifest')
    alias = serving['adapter_alias' if args.model == 'grouped' else 'base_alias']
    expected_alias = model_aliases(manifest)[args.model]
    if alias != expected_alias:
        raise ValueError('Unexpected model alias')
    require_model(serving['url'], alias)
    output = args.artifacts / ('results_smoke' if args.limit else 'results') / args.model
    output.mkdir(parents=True, exist_ok=True)
    protocol = {'manifest_sha256': digest(manifest), 'serving_sha256': digest(serving),
                'model_alias': alias, 'limit': args.limit, 'temperature': 0, 'top_p': 1, 'seed': 0}
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if read_json(protocol_path) != protocol:
            raise ValueError('Cannot mix evaluation protocols or smoke limits')
    elif any(output.iterdir()):
        raise ValueError('Output directory contains files without protocol provenance')
    write_json(protocol_path, protocol)
    summary = {'status': 'smoke_only' if args.limit else 'complete', 'model_alias': alias,
               'protocol_sha256': digest(protocol), 'datasets': {}}
    for key, dataset in manifest['datasets'].items():
        rows = read_jsonl(dataset['path'])
        if len(rows) != dataset['queries']:
            raise ValueError('Frozen input count mismatch')
        if args.limit:
            rows = rows[:args.limit]
        path = output / f'{key}.jsonl'
        done = evaluate_dataset(rows, path, serving['url'], alias, dataset['max_tokens'])
        if any(r['model_alias'] != alias for r in done):
            raise ValueError('Saved outputs have the wrong model alias')
        summary['datasets'][key] = {**summarize(rows, done), 'raw_sha256': sha256_file(path)}
        if key.startswith('csfcube_') and not args.limit:
            facet = key.removeprefix('csfcube_')
            ranks = {row['paper_id']: [[pid, rank] for rank, pid in enumerate(r['retrieved'])]
                     for row, r in zip(rows, done)}
            write_json(output / f'test-pid2pool-csfcube-hpo_{args.model}-{facet}-ranked.json', ranks)
    # Written only when all six pools have completed. Smoke never creates official ranking files.
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))
    print(f'Results: {output}')


if __name__ == '__main__':
    main()
