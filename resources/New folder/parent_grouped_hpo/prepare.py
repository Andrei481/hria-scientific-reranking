"""Freeze experiment settings and generate grouped/control split manifests."""
import argparse
from pathlib import Path

from protocol import ROOT, PROTOCOLS, digest, load_config, load_resources, make_split, source_hashes, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts')
    args = parser.parse_args()
    config = load_config(args.config)
    parents, rows, _, resource = load_resources()
    if args.output.exists():
        raise FileExistsError('Use a fresh artifact directory; preparation is immutable')
    args.output.mkdir(parents=True)
    write_json(args.output / 'config.json', config)
    for protocol in PROTOCOLS:
        for seed in config['split_seeds']:
            split = make_split(parents, rows, seed, protocol, config['test_seed'])
            write_json(args.output / 'splits' / f'{protocol}_{seed}.json', split)
            print(protocol, seed, split['counts'], 'overlap:', len(split['train_validation_parent_overlap']))
    write_json(args.output / 'study.json', {
        'config_sha256': digest(config), 'resource_manifest_sha256': digest(resource),
        'source_hashes': source_hashes(),
        'splits': {f'{p}_{s}': digest(make_split(parents, rows, s, p, config['test_seed'])) for p in PROTOCOLS for s in config['split_seeds']},
        'audit': resource['audit'],
    })
    print(f'Prepared {args.output}; no training or selection has been performed.')


if __name__ == '__main__':
    main()
