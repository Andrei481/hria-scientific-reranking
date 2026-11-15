"""Shared external-evaluation checks; independent of frozen HPO source files."""
import hashlib
import json
import math
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parent
FACETS = ('background', 'method', 'result')
FACET_INSTRUCTIONS = {
    'background': 'Candidates asking a similar question or motivated by the same problem.',
    'method': 'Candidates that solve problems using a similar technique or experimental setup.',
    # Preserve the spelling in the retained improved notebook for prompt parity.
    'result': 'Cadndiates that arrive at similar findings or demonstrate similar outcomes.',
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_hashes():
    return {p.name: sha256_file(p) for p in sorted(ROOT.glob('*.py')) if not p.name.startswith('test_')}


def model_aliases(manifest):
    suffix = digest(manifest)[:20]
    return {'grouped': 'hpo_grouped_' + suffix, 'base': 'hpo_base_' + suffix}


def require_model(url, alias):
    if alias not in {m['id'] for m in request_json(url + '/models')['data']}:
        raise ValueError(f'Server does not advertise this cohort model: {alias}')


def verify_files(directory, hashes):
    for name, expected in hashes.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe relative file name')
        if sha256_file(Path(directory) / name) != expected:
            raise ValueError(f'File changed: {name}')


def verify_manifest(artifacts):
    manifest = read_json(Path(artifacts) / 'manifest.json')
    if manifest['source_hashes'] != source_hashes():
        raise ValueError('External evaluation code changed; preserve this cohort and prepare a new one')
    if sha256_file(manifest['refit_result']) != manifest['refit_result_sha256']:
        raise ValueError('Refit result changed')
    verify_files(manifest['adapter'], manifest['adapter_files'])
    verify_files(manifest['csfcube_root'], manifest['csfcube_files'])
    for dataset in manifest['datasets'].values():
        if sha256_file(dataset['path']) != dataset['sha256']:
            raise ValueError('Frozen evaluation input changed')
    return manifest


def parse_ranking(text, count):
    seen, valid = set(), []
    duplicate = invalid = 0
    for value in map(int, re.findall(r'\[(\d+)\]', text)):
        if not 1 <= value <= count:
            invalid += 1
        elif value in seen:
            duplicate += 1
        else:
            seen.add(value)
            valid.append(value - 1)
    missing = count - len(valid)
    audit = {'duplicate_ids': duplicate, 'invalid_ids': invalid, 'missing_ids': missing,
             'exact_permutation': not (duplicate or invalid or missing), 'fallback': not valid}
    return valid + [i for i in range(count) if i + 1 not in seen], audit


def litsearch_metrics(retrieved, relevant):
    relevant = set(relevant)
    result = {}
    for k in (5, 20):
        result[f'R@{k}'] = len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0
        dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(retrieved[:k]) if d in relevant)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
        result[f'nDCG@{k}'] = dcg / ideal if ideal else 0.0
    return result


def csfcube_prompt(facet, qid, candidates, papers, mode):
    def format_paper(record):
        abstract = record['abstract']
        if mode == 'facet':
            if isinstance(abstract, list):
                abstract = ' '.join(str(s).strip() for s in abstract if str(s).strip())
            else:
                abstract = str(abstract).strip()
        return f"Title: {record['title']}\nAbstract: {abstract}"
    if mode not in ('original', 'facet'):
        raise ValueError('Choose original or facet CSFCube prompt explicitly')
    query = format_paper(papers[qid])
    if mode == 'facet':
        query = ' '.join(query.split())
    passages = '\n\n---\n\n'.join(f"[{i}] {' '.join(format_paper(papers[p]).split())}"
                                    for i, p in enumerate(candidates, 1))
    qualifier = ' under a specified facet' if mode == 'facet' else ''
    facet_text = f'Facet: {FACET_INSTRUCTIONS[facet]}\n\n' if mode == 'facet' else ''
    return ('You are an expert academic paper reranker. '
            f'Your task is to re-order the given list of passages (from [1] to [{len(candidates)}]) '
            f'based on their relevance to the query{qualifier}. Respond with only the ranking and nothing else.\n\n'
            'Example output for 8 passages:\n[2] > [5] > [4] > [8] > [6] > [1] > [3] > [7]\n\n'
            f'Query: {query}\n\nPassages:\n{passages}\n\n{facet_text}Your ranking (most to least relevant):')


def validate_prefix(rows, done):
    if len(done) > len(rows):
        raise ValueError('More outputs than inputs')
    for row, result in zip(rows, done):
        if result['qid'] != row['qid'] or result['input_sha256'] != digest(row):
            raise ValueError('Saved outputs do not match frozen query order')
        order, _ = parse_ranking(result['raw_response'], len(row['pre_reranked']))
        if result['retrieved'] != [row['pre_reranked'][i] for i in order]:
            raise ValueError('Saved ranking differs from its raw response')


def request_json(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)
