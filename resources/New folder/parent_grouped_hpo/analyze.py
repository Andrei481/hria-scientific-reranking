"""Select hyperparameters from complete validation-only grids; never use test scores."""
import argparse
import csv
from pathlib import Path

from protocol import (ROOT, PROTOCOLS, digest, read_json, run_name, select_results, sha256_file,
                      summarize_losses, trial_parameters, verify_study, verify_tokens, write_json)


def collect_results(artifacts, protocols):
    artifacts = Path(artifacts)
    config, study, rows = verify_study(artifacts)
    tokens = verify_tokens(artifacts, study)
    records, hashes = [], {}
    environment = None
    for p in protocols:
        for seed in config['split_seeds']:
            split = read_json(artifacts / 'splits' / f'{p}_{seed}.json')
            expected = {rows[i]['example_id']: rows[i]['parent_id'] for i in split['indices']['validation']}
            for trial in config['trials']:
                name = run_name(p, seed, trial['id'])
                path = artifacts / 'runs' / name / 'result.json'
                r = read_json(path)
                if (r['study_sha256'], r['tokens_sha256']) != (digest(study), tokens['tokens_sha256']):
                    raise ValueError('Mixed study or tokenization provenance')
                if (r['protocol'], r['split_seed'], r['trial_id']) != (p, seed, trial['id']):
                    raise ValueError('Run identity mismatch')
                if (r['mode'] != 'trial' or r['smoke_steps'] != 0 or r['model'] != config['model']
                    or r['revision'] != tokens['revision'] or r['training_seed'] != config['training_seed']
                    or r['hyperparameters'] != trial_parameters(config, trial['id'])
                    or r['training_lists'] != len(split['indices']['train'])
                    or r['training_parents'] != len(split['parent_ids']['train'])):
                    raise ValueError('Trial settings differ from the frozen protocol')
                if environment is None:
                    environment = r['packages']
                elif environment != r['packages']:
                    raise ValueError('Training package versions differ across trials')
                scorepath = path.parent / 'validation_scores.json'
                if r['validation_scores_sha256'] != sha256_file(scorepath):
                    raise ValueError('Validation scores changed')
                scores = read_json(scorepath)
                if len(scores) != len(expected) or {s['example_id']: s['parent_id'] for s in scores} != expected:
                    raise ValueError('Incomplete/duplicated/wrong validation examples')
                if summarize_losses(scores) != r['validation']:
                    raise ValueError('Validation summary mismatch')
                records.append(r)
                hashes[name] = sha256_file(path)
    return config, study, tokens, records, hashes


def load_selection(artifacts):
    selection = read_json(Path(artifacts) / 'analysis' / 'selection.json')
    config, study, tokens, records, hashes = collect_results(artifacts, selection['protocols'])
    summary = select_results(config, records, selection['protocols'])
    if selection['result_hashes'] != hashes or selection['summary'] != summary or selection['study_sha256'] != digest(study):
        raise ValueError('Selection no longer matches the completed trial grid')
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--protocols', nargs='+', choices=PROTOCOLS, default=['grouped'])
    args = parser.parse_args()
    if len(set(args.protocols)) != len(args.protocols):
        raise ValueError('Duplicate protocols')
    config, study, tokens, records, hashes = collect_results(args.artifacts, args.protocols)
    summary = select_results(config, records, args.protocols)
    directory = args.artifacts / 'analysis'
    directory.mkdir(exist_ok=True)
    selection = {'study_sha256': digest(study), 'protocols': args.protocols,
                 'study_kind': config['study_kind'], 'model': config['model'], 'revision': tokens['revision'],
                 'reference_trial': config['reference_trial'], 'summary': summary, 'result_hashes': hashes}
    write_json(directory / 'selection.json', selection)
    with (directory / 'validation_summary.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['protocol', 'trial_id', 'mean', 'sd_across_splits', 'scores_by_split'])
        writer.writeheader()
        for p, s in summary.items():
            for trial in s['trials']:
                writer.writerow({'protocol': p, **trial})
    lines = ['Parent-grouped HPO results', '', f"Model: {config['model']} at {tokens['revision']}",
             f"Study: {config['study_kind']}; selection metric: {config['selection_metric']} (lower is better).", '',
             '| Protocol | Trial | Mean validation NLL | SD across splits |', '|---|---|---:|---:|']
    for p, s in summary.items():
        for t in s['trials']:
            sd = 'n/a' if t['sd_across_splits'] is None else f"{t['sd_across_splits']:.6f}"
            lines.append(f"| {p} | {t['trial_id']} | {t['mean']:.6f} | {sd} |")
        lines.extend(['', f"{p}: selected {s['winner']}; reference retained: {s['reference_retained']}.",
                      f"Winners by split: {s['winner_per_split']}", ''])
    if set(PROTOCOLS) <= set(summary):
        lines.append(f"Grouped and list-control winners agree: {summary['grouped']['winner'] == summary['list_control']['winner']}.")
    lines.extend(['', 'These are validation-selection results, not external benchmark improvements or an equivalence test.',
                  'The default grid tests learning-rate sensitivity only. Prior historical HPO is not established by this study.',
                  'Split-level SD is descriptive; the training sets overlap and it is not a standard error over independent experiments.'])
    (directory / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
