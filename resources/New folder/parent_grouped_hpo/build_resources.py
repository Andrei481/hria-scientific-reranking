"""Maintainer utility: package the existing data; no sampling or model calls."""
import argparse
import gzip
import json
from pathlib import Path

from protocol import ROOT, audit_rows, parent_mapping, read_json, resolve_parent, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'resources')
    args = parser.parse_args()
    paths = {
        'frozen_lists': args.repository / 'New folder/modernbert_supervised_baseline/modernbert_supervised_baseline/dorismae_listwise_4494.jsonl',
        'original_parents': args.repository / 'DorisMAE/Query.json',
        'aspects': args.repository / 'HRIA/reranking_dataset_doris_mae.jsonl',
    }
    if sha256_file(paths['frozen_lists']) != '7c649c816c78778777afcddc011dafd345a6847d2fd6d4e082b58bad75e5bc4f':
        raise ValueError('Unexpected frozen-list source')
    if args.output.exists():
        raise FileExistsError(f'Resource directory already exists: {args.output}')
    parents = read_json(paths['original_parents'])
    with paths['aspects'].open(encoding='utf-8') as f:
        mapping = parent_mapping(parents, [json.loads(line) for line in f])
    rows, documents = [], {}
    with paths['frozen_lists'].open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            row['parent_id'] = resolve_parent(row, mapping)
            row['example_id'] = f'{row["source_row_id"]}:{row["variant_id"]}'
            if len(row['docs']) != len(row['candidate_ids']):
                raise ValueError('Unaligned document texts')
            for cid, text in zip(row['candidate_ids'], row.pop('docs')):
                cid = str(cid)
                if cid in documents and documents[cid] != text:
                    raise ValueError(f'Conflicting document text {cid}')
                documents[cid] = text
            rows.append(row)
    audit = audit_rows(rows)
    if audit['examples'] != 4494 or audit['parents'] != 100:
        raise ValueError('Unexpected dataset size')
    args.output.mkdir(parents=True)
    with gzip.open(args.output / 'lists.jsonl.gz', 'wt', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    with gzip.open(args.output / 'documents.json.gz', 'wt', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False)
    write_json(args.output / 'parents.json', [{'parent_id': i, 'category': p['query_type'],
        'query_text': p['query_text']} for i, p in enumerate(parents)])
    write_json(args.output / 'manifest.json', {
        'source_files': {k: {'relative_path': str(p.relative_to(args.repository)), 'sha256': sha256_file(p)} for k, p in paths.items()},
        'files': {name: sha256_file(args.output / name) for name in ['lists.jsonl.gz', 'documents.json.gz', 'parents.json']},
        'audit': audit, 'unique_documents': len(documents),
        'parent_mapping': 'aspect_id -> verified query, category, candidate grades -> unique full parent pool',
        'cohort': 'new selection-sensitivity cohort using retained frozen lists; historical checkpoint input identity not established',
    })
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
