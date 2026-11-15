"""Verify the completed HPO refit and preflight frozen external evaluation inputs."""
import argparse
import json
from pathlib import Path
import sys

from external_protocol import (FACETS, csfcube_prompt, digest, read_json, read_jsonl,
                               sha256_file, source_hashes, verify_files, write_json)


def csfcube_rows(root, mode):
    papers = {str(r['paper_id']): r for r in read_jsonl(root / 'abstracts-csfcube-preds.jsonl')}
    result = {}
    for facet in FACETS:
        rows = []
        for qid, pool in read_json(root / f'test-pid2anns-csfcube-{facet}.json').items():
            ids = list(map(str, pool['cands']))
            if len(ids) != len(set(ids)) or len(ids) != len(pool['relevance_adju']):
                raise ValueError('Invalid CSFCube candidate pool')
            rows.append({'qid': f'{qid}_{facet}', 'paper_id': qid, 'facet': facet, 'pre_reranked': ids,
                         'prompt': csfcube_prompt(facet, qid, ids, papers, mode)})
        result[facet] = rows
    if sum(map(len, result.values())) != 50:
        raise ValueError('Expected all 50 CSFCube query-facet cases')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hpo-bundle', required=True, type=Path)
    parser.add_argument('--hpo-artifacts', required=True, type=Path)
    parser.add_argument('--litsearch-inputs', required=True, type=Path)
    parser.add_argument('--csfcube', required=True, type=Path)
    parser.add_argument('--csfcube-prompt', required=True, choices=['original', 'facet'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-model-len', type=int, default=131072)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh external-evaluation output directory')
    args.hpo_artifacts = args.hpo_artifacts.resolve()
    args.csfcube = args.csfcube.resolve()
    args.litsearch_inputs = args.litsearch_inputs.resolve()
    args.output = args.output.resolve()
    sys.path.insert(0, str(args.hpo_bundle.resolve()))
    import protocol as hpo
    from analyze import load_selection
    config, study, training_rows = hpo.verify_study(args.hpo_artifacts)
    selection = load_selection(args.hpo_artifacts)
    winner = selection['summary']['grouped']['winner']
    refit = args.hpo_artifacts / 'refit' / 'grouped' / winner
    result = read_json(refit / 'result.json')
    tokens = hpo.verify_tokens(args.hpo_artifacts, study)
    expected = {'status': 'complete', 'mode': 'refit', 'protocol': 'grouped', 'trial_id': winner,
                'model': config['model'], 'revision': tokens['revision'], 'smoke_steps': 0,
                'study_sha256': hpo.digest(study), 'tokens_sha256': tokens['tokens_sha256'],
                'training_lists': len(training_rows), 'training_parents': len({r['parent_id'] for r in training_rows}),
                'training_seed': config['training_seed'], 'hyperparameters': hpo.trial_parameters(config, winner)}
    if any(result.get(k) != v for k, v in expected.items()):
        raise ValueError('Refit does not match the selected complete grouped study')
    if config['model'] != 'meta-llama/Llama-3.1-8B-Instruct' or config.get('load_in_4bit', False):
        raise ValueError('This serving recipe is for the BF16 Llama 3.1 8B refit')
    adapter = refit / 'adapter'
    if not result.get('adapter_files') or 'adapter_config.json' not in result['adapter_files']:
        raise ValueError('Refit has no complete adapter manifest')
    verify_files(adapter, result['adapter_files'])
    if read_json(adapter / 'adapter_config.json')['r'] != result['hyperparameters']['lora_rank']:
        raise ValueError('Adapter rank differs from refit')

    frozen = read_json(args.litsearch_inputs / 'manifest.json')
    if frozen['max_k'] != 100 or frozen['candidate_word_limit'] != 300:
        raise ValueError('Expected frozen top-100/300-word LitSearch inputs')
    datasets, all_rows, identities = {}, {}, []
    for retriever in ('bm25', 'grit', 'qwen3'):
        path = args.litsearch_inputs / f'{retriever}.jsonl'
        if sha256_file(path) != frozen['files'][retriever]['input_sha256']:
            raise ValueError('LitSearch input hash mismatch')
        rows = read_jsonl(path)
        ids = {r['qid'] for r in rows}
        if len(rows) != 597 or len(ids) != 597 or any(len(r['pre_reranked']) != 100 or len(set(r['pre_reranked'])) != 100 for r in rows):
            raise ValueError('Expected 597 distinct queries with 100 distinct candidates')
        identities.append(ids)
        key = 'litsearch_' + retriever
        datasets[key] = {'path': str(path), 'sha256': sha256_file(path), 'queries': 597, 'max_tokens': 1024}
        all_rows[key] = rows
    if not identities[0] == identities[1] == identities[2]:
        raise ValueError('LitSearch retrievers have different query identities')
    csf = csfcube_rows(args.csfcube, args.csfcube_prompt)
    args.output.mkdir(parents=True)
    for facet, rows in csf.items():
        key = 'csfcube_' + facet
        path = args.output / f'{key}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        datasets[key] = {'path': str(path), 'sha256': sha256_file(path), 'queries': len(rows), 'max_tokens': 4096}
        all_rows[key] = rows

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(adapter), local_files_only=True)
    audit, overflow = {}, []
    for key, rows in all_rows.items():
        lengths = []
        for i, row in enumerate(rows, 1):
            ids = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                                                tokenize=True, add_generation_prompt=True)
            lengths.append(len(ids))
            if len(ids) + datasets[key]['max_tokens'] > args.max_model_len:
                overflow.append({'dataset': key, 'qid': row['qid'], 'prompt_tokens': len(ids)})
            if i % 100 == 0:
                print(f'{key}: preflight {i}/{len(rows)}', flush=True)
        audit[key] = {'queries': len(rows), 'maximum_prompt_tokens': max(lengths), 'max_tokens': datasets[key]['max_tokens']}
    write_json(args.output / 'token_audit.json', {'datasets': audit, 'overflow': overflow})
    if overflow:
        raise ValueError('Context overflow; no success manifest written. Do not truncate or shorten candidate pools.')
    csf_files = ['abstracts-csfcube-preds.jsonl', 'evaluation_splits.json', 'queries-release.csv',
                 'eval_scripts/ranking_eval.py', 'eval_scripts/rank_metrics.py']
    csf_files.extend(f'test-pid2anns-csfcube-{facet}.json' for facet in FACETS)
    write_json(args.output / 'manifest.json', {
        'model': config['model'], 'revision': result['revision'], 'adapter': str(adapter),
        'adapter_files': result['adapter_files'], 'lora_rank': result['hyperparameters']['lora_rank'],
        'refit_result': str(refit / 'result.json'), 'refit_result_sha256': sha256_file(refit / 'result.json'),
        'source_hashes': source_hashes(), 'max_model_len': args.max_model_len, 'datasets': datasets,
        'litsearch_manifest': frozen, 'csfcube_root': str(args.csfcube), 'csfcube_prompt': args.csfcube_prompt,
        'csfcube_files': {name: sha256_file(args.csfcube / name) for name in csf_files},
        'tokenizer_vocab_sha256': digest(tokenizer.get_vocab()), 'token_audit': audit,
        'interpretation': 'new external evaluation cohort; historical score changes are not isolated LR effects',
    })
    print(f'Prepared {args.output}: 1791 LitSearch requests and 50 CSFCube requests; no generation performed.')


if __name__ == '__main__':
    main()
