"""Fresh-base HPO trial, selected-checkpoint internal test, or fresh all-parent refit."""
import argparse
import gzip
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from protocol import (ROOT, PROTOCOLS, TARGET_MODULES, digest, read_json, run_name,
                      sha256_file, summarize_losses, trial_parameters, verify_study, verify_tokens, write_json)


def model_interface(model_name):
    if 'gemma-3-' in model_name.lower():
        return 'FastModel', {'finetune_vision_layers': False, 'finetune_language_layers': True,
                             'finetune_attention_modules': True, 'finetune_mlp_modules': True}
    return 'FastLanguageModel', {}


def language_target_modules(model):
    # Full paths prevent shared suffixes such as q_proj from targeting vision.
    targets = [name for name, _ in model.named_modules()
               if 'language_model' in name.split('.') and name.rsplit('.', 1)[-1] in TARGET_MODULES]
    if not targets:
        raise ValueError('No language-model projections found; refusing unscoped multimodal LoRA')
    return targets


def validate_trainable_model(model):
    trainable = 0
    for name, parameter in model.named_parameters():
        if parameter.is_meta:
            raise ValueError(f'Unmaterialized model parameter: {name}')
        if parameter.requires_grad:
            if any(part in name.lower() for part in ('vision', 'visual', 'multi_modal_projector')):
                raise ValueError(f'Vision parameter must remain frozen: {name}')
            if 'lora_' not in name:
                raise ValueError(f'Unexpected trainable non-LoRA parameter: {name}')
            trainable += 1
    if not trainable:
        raise ValueError('No trainable adapter parameters')
    return trainable


class EncodedDataset:
    def __init__(self, encoded, indices):
        self.encoded, self.indices = encoded, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.encoded[self.indices[index]]
        ids, start = row['input_ids'], row['response_start']
        return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': [-100] * start + ids[start:]}


class ResponseCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        import torch
        width = max(len(f['input_ids']) for f in features)
        result = {k: [] for k in ['input_ids', 'attention_mask', 'labels']}
        for f in features:
            padding = width - len(f['input_ids'])
            for key, value in [('input_ids', self.pad_id), ('attention_mask', 0), ('labels', -100)]:
                result[key].append(f[key] + [value] * padding)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}


def output_directory(artifacts, mode, protocol, seed, trial, smoke_steps):
    if mode == 'refit':
        return Path(artifacts) / 'refit' / protocol / trial
    category = 'smoke' if smoke_steps else ('tests' if mode == 'test' else 'runs')
    return Path(artifacts) / category / run_name(protocol, seed, trial)


def package_versions():
    result = {}
    for name in ['torch', 'transformers', 'unsloth', 'unsloth_zoo', 'peft', 'accelerate', 'datasets', 'trl',
                 'numpy', 'tokenizers', 'bitsandbytes']:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def evaluate_losses(model, dataset, encoded, indices, collator):
    import torch
    model.eval()
    scores = []
    with torch.no_grad():
        for position, index in enumerate(indices):
            row = encoded[index]
            feature = dataset[position]
            batch = {k: v.to(model.device) for k, v in collator([feature]).items()}
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = float(model(**batch).loss.detach().float().cpu())
            scores.append({'example_id': row['example_id'], 'parent_id': row['parent_id'], 'nll': loss,
                           'response_tokens': len(row['input_ids']) - row['response_start']})
            if not math.isfinite(loss):
                raise ValueError(f'Non-finite loss for {row["example_id"]}')
            if (position + 1) % 50 == 0:
                print(f'Evaluated {position + 1}/{len(indices)}', flush=True)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--mode', choices=['trial', 'test', 'refit'], default='trial')
    parser.add_argument('--protocol', choices=PROTOCOLS, default='grouped')
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--trial', help='Required for trials; selected automatically for test/refit')
    parser.add_argument('--smoke-steps', type=int, default=0)
    parser.add_argument('--export-merged', action='store_true', help='For refit only; write a standalone checkpoint')
    args = parser.parse_args()
    if args.smoke_steps < 0 or (args.smoke_steps and args.mode != 'trial'):
        raise ValueError('Smoke steps are allowed only for trial mode')
    if args.export_merged and args.mode != 'refit':
        raise ValueError('Merged export is only available for final refit')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This protocol requires one process and one GPU')
    config, study, rows = verify_study(args.artifacts)
    tokens = verify_tokens(args.artifacts, study)
    if args.split_seed not in config['split_seeds']:
        raise ValueError('Split seed is not in the frozen config')
    if args.mode != 'trial':
        from analyze import load_selection
        selection = load_selection(args.artifacts)
        if args.protocol not in selection['summary']:
            raise ValueError('Protocol has not been analyzed')
        selected = selection['summary'][args.protocol]['winner']
        if args.trial and args.trial != selected:
            raise ValueError('Test/refit must use the selected configuration')
        args.trial = selected
    if not args.trial:
        raise ValueError('Specify --trial for trial mode')
    hp = trial_parameters(config, args.trial)
    split = read_json(args.artifacts / 'splits' / f'{args.protocol}_{args.split_seed}.json')
    directory = output_directory(args.artifacts, args.mode, args.protocol, args.split_seed, args.trial, args.smoke_steps)
    if directory.exists():
        raise FileExistsError(f'Run directory exists: {directory}. Preserve completed/failed runs; see README.')
    with gzip.open(args.artifacts / 'tokens' / 'encoded.jsonl.gz', 'rt', encoding='utf-8') as f:
        encoded = [json.loads(line) for line in f]
    if len(encoded) != len(rows) or any((r['example_id'], r['parent_id']) != (e['example_id'], e['parent_id']) for r, e in zip(rows, encoded)):
        raise ValueError('Token cache row order/identity mismatch')

    # Unsloth patches must be installed before importing Transformers/PyTorch.
    import unsloth
    import torch
    from transformers import BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one BF16-capable CUDA GPU with CUDA_VISIBLE_DEVICES')
    set_seed(config['training_seed'])
    backend_name, peft_flags = model_interface(config['model'])
    backend = getattr(unsloth, backend_name)
    load_in_4bit = config.get('load_in_4bit', False)
    load_options = {}
    if backend_name == 'FastModel':
        load_options.update(load_in_16bit=not load_in_4bit, load_in_8bit=False, full_finetuning=False)
    if load_in_4bit:
        load_options['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    model, processor = backend.from_pretrained(
        model_name=config['model'], revision=tokens['revision'], max_seq_length=config['max_length'],
        dtype=torch.bfloat16, load_in_4bit=load_in_4bit, use_exact_model_name=True, **load_options,
    )
    tokenizer = getattr(processor, 'tokenizer', processor)
    if digest(tokenizer.get_vocab()) != tokens['tokenizer_vocab_sha256'] or digest(tokenizer.chat_template) != tokens['chat_template_sha256']:
        raise ValueError('Loaded tokenizer differs from preflight')
    resolved = getattr(model.config, '_commit_hash', None)
    if resolved and resolved != tokens['revision']:
        raise ValueError('Loaded base model revision differs from preflight')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    if args.mode == 'test':
        from peft import PeftModel
        trial_dir = args.artifacts / 'runs' / run_name(args.protocol, args.split_seed, args.trial)
        result = read_json(trial_dir / 'result.json')
        for name, expected in result['adapter_files'].items():
            if sha256_file(trial_dir / 'adapter' / name) != expected:
                raise ValueError('Selected adapter changed')
        model = PeftModel.from_pretrained(model, str(trial_dir / 'adapter'), is_trainable=False)
    else:
        targets = TARGET_MODULES
        if backend_name == 'FastModel':
            targets = '^(?:' + '|'.join(re.escape(n) for n in language_target_modules(model)) + ')$'
        model = backend.get_peft_model(
            model, r=hp['lora_rank'], lora_alpha=hp['lora_alpha'], lora_dropout=hp['lora_dropout'],
            target_modules=targets, bias='none', use_gradient_checkpointing='unsloth',
            random_state=config['training_seed'], **peft_flags,
        )
        validate_trainable_model(model)
    model.config.use_cache = False
    collator = ResponseCollator(tokenizer.pad_token_id)
    directory.mkdir(parents=True)
    provenance = {'study_sha256': digest(study), 'tokens_sha256': tokens['tokens_sha256'],
                  'model': config['model'], 'revision': tokens['revision'], 'mode': args.mode,
                  'protocol': args.protocol, 'split_seed': args.split_seed, 'trial_id': args.trial,
                  'training_seed': config['training_seed'], 'hyperparameters': hp, 'packages': package_versions(),
                  'smoke_steps': args.smoke_steps, 'model_backend': backend_name,
                  'load_in_4bit': load_in_4bit, 'vision_finetuning': False}
    write_json(directory / 'started.json', provenance)
    if args.mode == 'test':
        indices = split['indices']['test']
        scores = evaluate_losses(model, EncodedDataset(encoded, indices), encoded, indices, collator)
        write_json(directory / 'test_scores.json', scores)
        write_json(directory / 'test_result.json', {**provenance, 'status': 'complete', 'test': summarize_losses(scores),
                    'interpretation': 'selected 80-parent checkpoint; no further tuning on this test'})
        print(f'Internal-test results saved: {directory}')
        return

    train_indices = list(range(len(rows))) if args.mode == 'refit' else split['indices']['train']
    training_args = TrainingArguments(
        output_dir=str(directory / 'trainer'), num_train_epochs=hp['epochs'], max_steps=args.smoke_steps or -1,
        per_device_train_batch_size=hp['batch_size'], gradient_accumulation_steps=hp['gradient_accumulation'],
        learning_rate=hp['learning_rate'], warmup_steps=hp['warmup_steps'], weight_decay=hp['weight_decay'],
        lr_scheduler_type='cosine', optim='adamw_torch', bf16=True, fp16=False, tf32=False,
        seed=config['training_seed'], data_seed=config['training_seed'], dataloader_num_workers=0,
        dataloader_drop_last=False, remove_unused_columns=False, report_to='none',
        logging_steps=25, save_strategy='no', eval_strategy='no',
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=EncodedDataset(encoded, train_indices),
                      data_collator=collator, processing_class=tokenizer)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    trained = trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    memory = {'allocated_gib': torch.cuda.max_memory_allocated() / 2**30, 'reserved_gib': torch.cuda.max_memory_reserved() / 2**30}
    model.save_pretrained(str(directory / 'adapter'))
    tokenizer.save_pretrained(str(directory / 'adapter'))
    adapter_files = {p.name: sha256_file(p) for p in sorted((directory / 'adapter').iterdir()) if p.is_file()}
    result = {**provenance, 'status': 'smoke_only' if args.smoke_steps else 'complete',
              'training_lists': len(train_indices), 'training_parents': len({rows[i]['parent_id'] for i in train_indices}),
              'training_seconds': elapsed, 'optimizer_steps': trainer.state.global_step, 'train_metrics': trained.metrics,
              'peak_training_memory': memory, 'adapter_files': adapter_files,
              'gpu_name': torch.cuda.get_device_name(0),
              'nvidia_smi': subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                                           capture_output=True, text=True, check=True).stdout.strip()}
    if args.mode == 'trial':
        indices = split['indices']['validation']
        if args.smoke_steps:
            indices = indices[:2]
        scores = evaluate_losses(model, EncodedDataset(encoded, indices), encoded, indices, collator)
        write_json(directory / 'validation_scores.json', scores)
        result['validation'] = summarize_losses(scores)
        result['validation_scores_sha256'] = sha256_file(directory / 'validation_scores.json')
    elif args.export_merged:
        if load_in_4bit:
            model.save_pretrained_merged(str(directory / 'final'), processor, save_method='merged_16bit')
        else:
            merged = model.merge_and_unload(safe_merge=True)
            merged.save_pretrained(str(directory / 'final'), safe_serialization=True, max_shard_size='5GB')
            processor.save_pretrained(str(directory / 'final'))
        result['merged_checkpoint'] = str((directory / 'final').resolve())
    write_json(directory / 'result.json', result)
    print(f'Saved {result["status"]} {args.mode}: {directory}', flush=True)


if __name__ == '__main__':
    main()
