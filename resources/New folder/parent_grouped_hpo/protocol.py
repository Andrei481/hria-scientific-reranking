"""Auditable, CPU-only data and selection primitives."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

ROOT = Path(__file__).resolve().parent
PARTS = ('train', 'validation', 'test')
PROTOCOLS = ('grouped', 'list_control')
METRICS = ('parent_macro_nll', 'list_macro_nll', 'token_micro_nll')
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def read_json(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_hashes():
    return {p.name: sha256_file(p) for p in sorted(ROOT.glob('*.py')) if not p.name.startswith('test_')}


def pool_key(pool):
    return tuple(sorted(map(int, pool)))


def parent_mapping(parents, aspects):
    lookup = {}
    for i, p in enumerate(parents):
        key = pool_key(p['candidate_pool'])
        if key in lookup:
            raise ValueError('Ambiguous parent pools')
        lookup[key] = i
    result = defaultdict(list)
    for a in aspects:
        aid = str(a['aspect_id'])
        pid = lookup[pool_key(a['candidate_pool'])]
        if a['original_query_type'] != parents[pid]['query_type']:
            raise ValueError('Parent category mismatch')
        result[aid].append({
            'parent_id': pid, 'query': a['query_text'], 'category': a['original_query_type'],
            'scores': dict(zip(map(int, a['candidate_pool']), map(int, a['candidate_scores']))),
        })
    return result


def resolve_parent(row, mapping):
    if len(row['candidate_ids']) != len(row['labels']):
        raise ValueError('Unaligned candidate labels')
    matches = [a['parent_id'] for a in mapping[str(row['aspect_id'])]
               if row['query'] == a['query'] and row['original_query_type'] == a['category']
               and all(a['scores'].get(int(cid)) == label for cid, label in zip(row['candidate_ids'], row['labels']))]
    if len(matches) != 1:
        raise ValueError(f'Expected unique parent match, found {matches}')
    return matches[0]


def audit_rows(rows):
    keys, lengths, positives = set(), Counter(), []
    by_aspect = defaultdict(list)
    for row in rows:
        key = row['example_id']
        if key in keys:
            raise ValueError('Duplicate example ID')
        keys.add(key)
        ids, labels = row['candidate_ids'], row['labels']
        if not ids or len(ids) != len(labels) or len(set(ids)) != len(ids):
            raise ValueError('Invalid candidate list')
        if any(type(v) is not int or v not in (0, 1, 2) for v in labels) or not any(labels):
            raise ValueError('Invalid relevance labels')
        if not row['query'].strip():
            raise ValueError('Empty query')
        by_aspect[(row['parent_id'], str(row['aspect_id']))].append(row)
        lengths[str(len(ids))] += 1
        positives.append(sum(v > 0 for v in labels))
    if not rows:
        raise ValueError('Empty data')
    for group in by_aspect.values():
        if len(group) != 3 or len({r['parent_id'] for r in group}) != 1:
            raise ValueError('Expected three variants and one parent per aspect')
        if len({pool_key(r['candidate_ids']) for r in group}) != 3:
            raise ValueError('Variant pools are not set-unique')
    return {'examples': len(rows), 'parents': len({r['parent_id'] for r in rows}),
            'aspects': len(by_aspect), 'list_size_histogram': dict(lengths),
            'lists_above_five_positives': sum(n > 5 for n in positives),
            'maximum_positives': max(positives)}


def load_resources(directory=None):
    directory = Path(directory or ROOT / 'resources')
    manifest = read_json(directory / 'manifest.json')
    for name, expected in manifest['files'].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f'Resource hash mismatch: {name}')
    with gzip.open(directory / 'lists.jsonl.gz', 'rt', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    with gzip.open(directory / 'documents.json.gz', 'rt', encoding='utf-8') as f:
        documents = json.load(f)
    if audit_rows(rows) != manifest['audit']:
        raise ValueError('Resource audit mismatch')
    for row in rows:
        if any(not documents.get(str(cid), '').strip() for cid in row['candidate_ids']):
            raise ValueError('Missing document text')
    return read_json(directory / 'parents.json'), rows, documents, manifest


def make_split(parents, rows, seed, protocol, test_seed=42):
    if protocol not in PROTOCOLS:
        raise ValueError('Unknown protocol')
    categories = defaultdict(list)
    for p in parents:
        categories[p['category']].append(p['parent_id'])
    if len(parents) != 100 or len(categories) != 5 or any(len(v) != 20 for v in categories.values()):
        raise ValueError('Expected five categories of twenty parents')
    ids = {s: set() for s in PARTS}
    for category in sorted(categories):
        ordered = sorted(categories[category])
        random.Random(int(digest(['test', test_seed, category])[:16], 16)).shuffle(ordered)
        test = set(ordered[18:])
        other = sorted(categories[category])
        random.Random(int(digest(['split', seed, category])[:16], 16)).shuffle(other)
        other = [pid for pid in other if pid not in test]
        ids['train'].update(other[:16])
        ids['validation'].update(other[16:])
        ids['test'].update(test)
    known = set.union(*ids.values())
    if len(known) != 100 or any(r['parent_id'] not in known for r in rows):
        raise ValueError('Unknown/duplicated parents')
    indices = {s: [i for i, r in enumerate(rows) if r['parent_id'] in ids[s]] for s in PARTS}
    if protocol == 'list_control':
        other = sorted(indices['train'] + indices['validation'])
        random.Random(seed).shuffle(other)
        nval = math.ceil(len(other) / 9)
        indices['validation'] = sorted(other[:nval])
        indices['train'] = sorted(other[nval:])
        ids = {s: {rows[i]['parent_id'] for i in indices[s]} for s in PARTS}
    if any(set(indices[a]) & set(indices[b]) for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise ValueError('Overlapping examples')
    if sorted(sum(indices.values(), [])) != list(range(len(rows))):
        raise ValueError('Split does not cover the data exactly once')
    if ids['test'] & (ids['train'] | ids['validation']):
        raise ValueError('Internal-test parent leakage')
    overlap = sorted(ids['train'] & ids['validation'])
    if protocol == 'grouped' and overlap:
        raise ValueError('Grouped train/validation leakage')
    return {'protocol': protocol, 'split_seed': seed, 'test_seed': test_seed,
            'indices': indices, 'parent_ids': {s: sorted(ids[s]) for s in PARTS},
            'train_validation_parent_overlap': overlap,
            'counts': {s: {'parents': len(ids[s]), 'lists': len(indices[s]),
                          'aspects': len({(rows[i]['parent_id'], rows[i].get('aspect_id', rows[i]['example_id'])) for i in indices[s]})}
                       for s in PARTS}}


def summarize_losses(scores):
    if not scores:
        raise ValueError('No evaluation scores')
    groups = defaultdict(list)
    for r in scores:
        if not math.isfinite(r['nll']) or r['nll'] < 0 or r['response_tokens'] <= 0:
            raise ValueError('Invalid evaluation loss')
        groups[r['parent_id']].append(r['nll'])
    return {'parent_macro_nll': statistics.mean(statistics.mean(v) for v in groups.values()),
            'list_macro_nll': statistics.mean(r['nll'] for r in scores),
            'token_micro_nll': sum(r['nll'] * r['response_tokens'] for r in scores) / sum(r['response_tokens'] for r in scores),
            'parents': len(groups), 'lists': len(scores)}


def load_config(path):
    config = read_json(path)
    required = {'study_kind', 'model', 'split_seeds', 'test_seed', 'training_seed', 'max_length',
                'selection_metric', 'reference_trial', 'shared', 'trials'}
    if set(config) - {'load_in_4bit'} != required:
        raise ValueError(f'Config requires {sorted(required)}; only load_in_4bit is optional')
    if type(config.get('load_in_4bit', False)) is not bool:
        raise ValueError('load_in_4bit must be a boolean')
    if config['study_kind'] not in ('new_sensitivity', 'original_grid_replay'):
        raise ValueError('Invalid study kind')
    if config['selection_metric'] not in METRICS:
        raise ValueError('Unsupported selection metric')
    if not config['split_seeds'] or len(set(config['split_seeds'])) != len(config['split_seeds']):
        raise ValueError('Empty/duplicated split seeds')
    if any(type(s) is not int or not 0 <= s < 2**32 for s in config['split_seeds'] + [config['test_seed'], config['training_seed']]):
        raise ValueError('Seeds must be unsigned 32-bit integers')
    if not isinstance(config['model'], str) or not config['model'].strip():
        raise ValueError('Invalid model identifier')
    allowed = {'learning_rate', 'epochs', 'lora_rank', 'lora_alpha', 'lora_dropout',
               'batch_size', 'gradient_accumulation', 'warmup_steps', 'weight_decay'}
    if set(config['shared']) != allowed:
        raise ValueError('Shared hyperparameter keys are incomplete or unknown')
    seen = set()
    for t in config['trials']:
        if set(t) - (allowed | {'id'}) or not re.fullmatch(r'[A-Za-z0-9_-]+', t['id']) or t['id'] in seen:
            raise ValueError('Invalid/duplicated trial')
        seen.add(t['id'])
        hp = {**config['shared'], **{k: v for k, v in t.items() if k != 'id'}}
        for key in ['learning_rate', 'epochs', 'lora_rank', 'lora_alpha', 'batch_size', 'gradient_accumulation']:
            if not isinstance(hp[key], (int, float)) or not math.isfinite(hp[key]) or hp[key] <= 0:
                raise ValueError(f'Invalid {key}')
        for key in ['epochs', 'lora_rank', 'lora_alpha', 'batch_size', 'gradient_accumulation', 'warmup_steps']:
            if type(hp[key]) is not int:
                raise ValueError(f'{key} must be an integer')
        if not 0 <= hp['lora_dropout'] < 1 or hp['warmup_steps'] < 0 or not math.isfinite(hp['weight_decay']) or hp['weight_decay'] < 0:
            raise ValueError('Invalid dropout/warmup/decay')
    if len(seen) < 2 or config['reference_trial'] not in seen:
        raise ValueError('At least two trials including the reference are required')
    if type(config['max_length']) is not int or config['max_length'] < 32:
        raise ValueError('Invalid max_length')
    return config


def trial_parameters(config, trial_id):
    trial = next((t for t in config['trials'] if t['id'] == trial_id), None)
    if trial is None:
        raise ValueError('Unknown trial ID')
    return {**config['shared'], **{k: v for k, v in trial.items() if k != 'id'}}


def run_name(protocol, seed, trial_id):
    return f'{protocol}/split_{seed}/{trial_id}'


def prompt_text(row, documents):
    passages = '\n\n---\n\n'.join(f'[{i}] {documents[str(cid)]}' for i, cid in enumerate(row['candidate_ids'], 1))
    return ('You are an expert academic paper reranker. '
            f'Your task is to re-order the given list of passages (from [1] to [{len(row["candidate_ids"])}]) '
            'based on their relevance to the query. Respond with only the ranking and nothing else.\n\n'
            'Example output for 8 passages:\n[2] > [5] > [4] > [8] > [6] > [1] > [3] > [7]\n\n'
            f'Query: {row["query"]}\n\nPassages:\n{passages}\n\nYour ranking (most to least relevant):')


def encode_row(row, documents, tokenizer):
    order = sorted(range(len(row['labels'])), key=lambda i: (-row['labels'][i], i))
    response = ' > '.join(f'[{i + 1}]' for i in order)
    messages = [{'role': 'user', 'content': prompt_text(row, documents)}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': response}], tokenize=False, add_generation_prompt=False)
    prefix_ids = tokenizer(prefix, add_special_tokens=False, truncation=False)['input_ids']
    ids = tokenizer(full, add_special_tokens=False, truncation=False)['input_ids']
    if ids[:len(prefix_ids)] != prefix_ids or len(ids) <= len(prefix_ids):
        raise ValueError('Unsafe response mask or empty target')
    return {'input_ids': ids, 'response_start': len(prefix_ids), 'example_id': row['example_id'], 'parent_id': row['parent_id']}


def verify_study(artifacts):
    artifacts = Path(artifacts)
    config = load_config(artifacts / 'config.json')
    study = read_json(artifacts / 'study.json')
    if study['config_sha256'] != digest(config):
        raise ValueError('Frozen configuration was edited; prepare a fresh study')
    if study['source_hashes'] != source_hashes():
        raise ValueError('Experiment code changed; prepare a fresh study before running comparisons')
    _, rows, _, manifest = load_resources()
    if study['resource_manifest_sha256'] != digest(manifest):
        raise ValueError('Resources changed')
    for name, expected in study['splits'].items():
        if digest(read_json(artifacts / 'splits' / f'{name}.json')) != expected:
            raise ValueError(f'Split changed: {name}')
    return config, study, rows


def verify_tokens(artifacts, study):
    artifacts = Path(artifacts)
    manifest = read_json(artifacts / 'tokens' / 'manifest.json')
    if manifest['study_sha256'] != digest(study):
        raise ValueError('Token preflight belongs to a different study')
    if sha256_file(artifacts / 'tokens' / 'encoded.jsonl.gz') != manifest['tokens_sha256']:
        raise ValueError('Token cache changed')
    return manifest


def select_results(config, records, protocols):
    expected = {(p, s, t['id']) for p in protocols for s in config['split_seeds'] for t in config['trials']}
    indexed = {}
    for r in records:
        key = (r['protocol'], r['split_seed'], r['trial_id'])
        if key in indexed or r['status'] != 'complete':
            raise ValueError('Duplicate or incomplete/smoke result')
        indexed[key] = r
    if set(indexed) != expected:
        raise ValueError(f'Incomplete or unexpected trial grid: missing={sorted(expected - set(indexed))}')
    summary = {}
    for protocol in protocols:
        trials = []
        for trial in config['trials']:
            scores = [indexed[(protocol, seed, trial['id'])]['validation'][config['selection_metric']] for seed in config['split_seeds']]
            if any(not math.isfinite(v) or v < 0 for v in scores):
                raise ValueError('Non-finite or invalid selection score')
            trials.append({'trial_id': trial['id'], 'scores_by_split': dict(zip(map(str, config['split_seeds']), scores)),
                           'mean': statistics.mean(scores), 'sd_across_splits': statistics.stdev(scores) if len(scores) > 1 else None})
        trials.sort(key=lambda r: (r['mean'], r['trial_id']))
        summary[protocol] = {'winner': trials[0]['trial_id'], 'reference_retained': trials[0]['trial_id'] == config['reference_trial'],
                             'trials': trials, 'selection_metric': config['selection_metric'],
                             'winner_per_split': {str(s): min(config['trials'], key=lambda t: (indexed[(protocol, s, t['id'])]['validation'][config['selection_metric']], t['id']))['id'] for s in config['split_seeds']}}
    return summary
