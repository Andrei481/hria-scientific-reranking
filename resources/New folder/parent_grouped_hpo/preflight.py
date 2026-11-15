"""Pin a tokenizer and cache complete response-masked examples without truncation."""
import argparse
import gzip
import json
from pathlib import Path
import re

from protocol import ROOT, digest, encode_row, load_resources, sha256_file, verify_study, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--revision', required=True, help='40-character model commit SHA')
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        raise ValueError('Use an exact model commit, not main')
    config, study, rows = verify_study(args.artifacts)
    _, _, documents, _ = load_resources()
    directory = args.artifacts / 'tokens'
    if directory.exists():
        raise FileExistsError('Token cache already exists; preserve it and use a new study directory')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=args.revision)
    if not tokenizer.chat_template:
        raise ValueError('Model has no chat template')
    directory.mkdir()
    lengths, overflow, response_tokens = [], [], []
    with gzip.open(directory / 'encoded.jsonl.gz', 'wt', encoding='utf-8') as f:
        for i, row in enumerate(rows):
            encoded = encode_row(row, documents, tokenizer)
            length = len(encoded['input_ids'])
            lengths.append(length)
            response_tokens.append(length - encoded['response_start'])
            if length > config['max_length']:
                overflow.append({'example_id': row['example_id'], 'tokens': length})
            f.write(json.dumps(encoded) + '\n')
            if (i + 1) % 250 == 0:
                print(f'Tokenized {i + 1}/{len(rows)}', flush=True)
    audit = {'examples': len(rows), 'max_tokens': max(lengths), 'min_tokens': min(lengths),
             'mean_tokens': sum(lengths) / len(lengths), 'min_response_tokens': min(response_tokens),
             'overlength_examples': len(overflow), 'overlength_details': overflow}
    write_json(directory / 'audit.json', audit)
    if overflow:
        raise ValueError(f'{len(overflow)} complete examples exceed max_length; maximum={max(lengths)}. '
                         'No successful preflight manifest was written. Read README before changing the protocol.')
    write_json(directory / 'manifest.json', {
        'study_sha256': digest(study), 'model': config['model'], 'revision': args.revision,
        'tokens_sha256': sha256_file(directory / 'encoded.jsonl.gz'),
        'tokenizer_vocab_sha256': digest(tokenizer.get_vocab()), 'chat_template_sha256': digest(tokenizer.chat_template),
        'audit': audit,
    })
    print(f'Preflight passed: {len(rows)} complete examples; max tokens {max(lengths)}')


if __name__ == '__main__':
    main()
