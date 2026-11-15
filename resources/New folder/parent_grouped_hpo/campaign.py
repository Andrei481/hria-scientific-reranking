"""Prepare and orchestrate an optional four-model grouped-selection campaign."""
import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import sys

from analyze import load_selection
from protocol import (ROOT, digest, load_config, read_json, sha256_file, trial_parameters,
                      verify_study, verify_tokens, write_json)
from run_matrix import parse_gpus


MODELS = {
    'llama8b': 'meta-llama/Llama-3.1-8B-Instruct',
    'gemma12b': 'google/gemma-3-12b-it',
    'qwen14b': 'Qwen/Qwen2.5-14B-Instruct',
    'gemma27b': 'google/gemma-3-27b-it',
}
SCOPE = ('New grouped-selection sensitivity over learning rate, LoRA rank and dropout. '
         'External LitSearch/CSFCube evaluation is separate and is not established by campaign status. '
         'This grid is not a verified replay of historical HPO.')


def campaign_configs(max_length=16384):
    base = load_config(ROOT / 'config.json')
    trials = []
    for (label, rate), rank, dropout in itertools.product(
            [('5e-6', 5e-6), ('1e-5', 1e-5), ('2e-5', 2e-5)], [64, 128], [0.0, 0.05]):
        trials.append({'id': f'lr_{label}_r{rank}_d{int(dropout * 100):02d}',
                       'learning_rate': rate, 'lora_rank': rank, 'lora_alpha': 2 * rank,
                       'lora_dropout': dropout})
    configs = {}
    for name, model in MODELS.items():
        config = deepcopy(base)
        config.update(model=model, study_kind='new_sensitivity', max_length=max_length,
                      trials=deepcopy(trials), reference_trial='lr_1e-5_r128_d05',
                      load_in_4bit=name == 'gemma27b')
        if trial_parameters(config, config['reference_trial']) != config['shared']:
            raise ValueError('The base config reference differs from the campaign reference; review the grid first')
        configs[name] = config
    return configs


def prepare_campaign(root, max_length=16384):
    root = Path(root)
    configs = campaign_configs(max_length)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {'scope': SCOPE, 'protocol': 'grouped', 'models': {}}
    for name, config in configs.items():
        config_path = root / 'configs' / f'{name}.json'
        write_json(config_path, config)
        subprocess.run([sys.executable, '-B', str(ROOT / 'prepare.py'), '--config', str(config_path),
                        '--output', str(root / name)], check=True, capture_output=True, text=True)
        manifest['models'][name] = {'model': config['model'], 'config_sha256': digest(config),
                                   'study_sha256': digest(read_json(root / name / 'study.json'))}
    write_json(root / 'campaign.json', manifest)
    return manifest


def verified_campaign(root):
    root = Path(root)
    manifest = read_json(root / 'campaign.json')
    if set(manifest['models']) != set(MODELS):
        raise ValueError('Campaign model coverage differs from the four-model protocol')
    if manifest['protocol'] != 'grouped' or manifest['scope'] != SCOPE:
        raise ValueError('Campaign scope changed')
    studies = {}
    for name, model in MODELS.items():
        config, study, rows = verify_study(root / name)
        expected = {'model': model, 'config_sha256': digest(config), 'study_sha256': digest(study)}
        if config['model'] != model or manifest['models'][name] != expected:
            raise ValueError(f'Campaign study identity changed: {name}')
        studies[name] = (config, study, rows)
    return manifest, studies


def campaign_status(root):
    root = Path(root)
    manifest, studies = verified_campaign(root)
    report = {'scope': manifest['scope'], 'selection_complete': False, 'refits_complete': False, 'models': {}}
    for name, (config, study, rows) in studies.items():
        item = {'status': 'selection_pending', 'selection_complete': False, 'refit_complete': False,
                'expected_trials': len(config['trials']) * len(config['split_seeds'])}
        report['models'][name] = item
        if not (root / name / 'analysis' / 'selection.json').exists():
            continue
        selection = load_selection(root / name)
        if 'grouped' not in selection['summary']:
            raise ValueError('Campaign requires grouped selection')
        winner = selection['summary']['grouped']['winner']
        item.update(status='refit_pending', selection_complete=True, winner=winner)
        directory = root / name / 'refit' / 'grouped' / winner
        if not (directory / 'result.json').exists():
            continue
        result = read_json(directory / 'result.json')
        tokens = verify_tokens(root / name, study)
        expected = {'status': 'complete', 'mode': 'refit', 'protocol': 'grouped', 'trial_id': winner,
                    'smoke_steps': 0, 'study_sha256': digest(study), 'tokens_sha256': tokens['tokens_sha256'],
                    'model': config['model'], 'revision': tokens['revision'],
                    'training_seed': config['training_seed'], 'hyperparameters': trial_parameters(config, winner),
                    'training_lists': len(rows), 'training_parents': len({r['parent_id'] for r in rows})}
        if any(result.get(k) != v for k, v in expected.items()) or not result.get('adapter_files'):
            raise ValueError(f'Invalid refit provenance: {name}')
        for filename, expected_hash in result['adapter_files'].items():
            if Path(filename).name != filename or sha256_file(directory / 'adapter' / filename) != expected_hash:
                raise ValueError(f'Refit adapter changed: {name}')
        item.update(status='refit_complete_external_pending', refit_complete=True)
    report['selection_complete'] = all(m['selection_complete'] for m in report['models'].values())
    report['refits_complete'] = all(m['refit_complete'] for m in report['models'].values())
    return report


def stage_commands(root, stage, gpus=None):
    root = Path(root).resolve()
    _, studies = verified_campaign(root)
    if stage not in ('smoke', 'search', 'analyze', 'refit'):
        raise ValueError('Unknown campaign stage')
    if gpus is not None:
        parse_gpus(gpus)
        if stage != 'search':
            raise ValueError('--gpus is for search only; expose one GPU for smoke/refit')
    commands = []
    for name, (config, _, _) in studies.items():
        common = ['--artifacts', str(root / name)]
        if stage == 'search':
            command = [sys.executable, '-B', str(ROOT / 'run_matrix.py'), *common, '--protocols', 'grouped', '--execute']
            if gpus is not None:
                command.extend(['--gpus', gpus])
        elif stage == 'analyze':
            command = [sys.executable, '-B', str(ROOT / 'analyze.py'), *common, '--protocols', 'grouped']
        elif stage == 'smoke':
            command = [sys.executable, '-B', str(ROOT / 'train.py'), *common, '--protocol', 'grouped',
                       '--split-seed', str(config['split_seeds'][0]), '--trial', config['reference_trial'], '--smoke-steps', '2']
        else:
            command = [sys.executable, '-B', str(ROOT / 'train.py'), *common, '--protocol', 'grouped',
                       '--mode', 'refit', '--split-seed', str(config['split_seeds'][0])]
        commands.append(command)
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'status', 'smoke', 'search', 'analyze', 'refit'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--max-length', type=int, default=16384, help='Preparation only')
    parser.add_argument('--gpus', help='Search only; comma-separated GPU indices')
    parser.add_argument('--execute', action='store_true', help='Run printed stage commands; models run sequentially')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare_campaign(args.root, args.max_length)
        print(f'Prepared four-model campaign: {args.root}; 144 grouped trials, no training performed.')
    elif args.stage == 'status':
        print(json.dumps(campaign_status(args.root), indent=2))
    else:
        for command in stage_commands(args.root, args.stage, args.gpus):
            print(shlex.join(command), flush=True)
            if args.execute:
                subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
