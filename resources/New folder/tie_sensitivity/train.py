"""Run one matched Llama condition, or preflight all its tokenized targets."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import subprocess
import time

from experiment import (
    POLICIES, audit_rows, check_data_hash, encode_example, epoch_schedule,
    example_id, read_jsonl, sha256_file, source_hashes, target_indices, write_json,
)


class ScheduledDataset:
    """Two logical epochs represented by one immutable, explicit schedule."""
    def __init__(self, rows, tokenizer, policy, seed, max_length, epochs=2):
        self.rows, self.tokenizer = rows, tokenizer
        self.policy, self.seed, self.max_length = policy, seed, max_length
        self.schedule = epoch_schedule(len(rows), epochs, seed)

    def __len__(self):
        return len(self.schedule)

    def __getitem__(self, index):
        epoch, row = self.schedule[index]
        return encode_example(self.rows[row], self.tokenizer, self.policy, self.seed, epoch, self.max_length)


class ResponseCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        import torch
        length = max(len(item["input_ids"]) for item in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad = length - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [self.pad_id] * pad)
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad)
            batch["labels"].append(item["labels"] + [-100] * pad)
        return {key: torch.tensor(values, dtype=torch.long) for key, values in batch.items()}


def package_versions():
    result = {}
    for name in ("torch", "transformers", "trl", "unsloth", "unsloth_zoo", "peft", "accelerate", "datasets", "bitsandbytes"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def save_exact_checkpoint(model, tokenizer, directory):
    """Export the loaded base + adapter without resolving another base repository."""
    directory = Path(directory)
    # PEFT merging removes adapter modules; count and save them before merging.
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.save_pretrained(str(directory / "adapter"))
    tokenizer.save_pretrained(str(directory / "adapter"))
    merged_model = model.merge_and_unload(safe_merge=True)
    merged_model.save_pretrained(
        str(directory / "final"), safe_serialization=True, max_shard_size="5GB",
    )
    tokenizer.save_pretrained(str(directory / "final"))
    return trainable_parameters


def make_config(args, data_hash):
    return {
        "model": args.model, "revision": args.revision,
        "data_sha256": data_hash, "seed": args.seed, "policy": args.policy,
        "max_length": args.max_length, "logical_epochs": 2,
        "batch_size": 2, "gradient_accumulation": 1, "learning_rate": 1e-5,
        "warmup_steps": 20, "weight_decay": 0.01, "scheduler": "cosine",
        "lora_rank": 128, "lora_alpha": 256, "lora_dropout": 0.05,
        "precision": "bf16", "quantization": "none",
        "export_method": "peft_merge_and_unload_loaded_base",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "epoch_implementation": "two fixed logical epochs in one sequential Trainer pass",
        "smoke_steps": args.smoke_steps, "source_hashes": source_hashes(),
    }


def preflight(dataset, directory, config, tokenizer):
    stats = [{"examples": 0, "input_tokens": 0, "response_tokens": 0, "max_tokens": 0} for _ in range(2)]
    hashes = [hashlib.sha256(), hashlib.sha256()]
    for index, (epoch, row_index) in enumerate(dataset.schedule):
        item = dataset[index]
        row = dataset.rows[row_index]
        target = target_indices(row["labels"], dataset.policy, dataset.seed, example_id(row), epoch)
        hashes[epoch].update(json.dumps([example_id(row), target]).encode())
        stat = stats[epoch]
        stat["examples"] += 1
        stat["input_tokens"] += len(item["input_ids"])
        stat["response_tokens"] += sum(label != -100 for label in item["labels"])
        stat["max_tokens"] = max(stat["max_tokens"], len(item["input_ids"]))
        if (index + 1) % 500 == 0:
            print(f"Preflight: {index + 1}/{len(dataset)} presentations", flush=True)
    changed = sum(
        target_indices(r["labels"], dataset.policy, dataset.seed, example_id(r), 0)
        != target_indices(r["labels"], dataset.policy, dataset.seed, example_id(r), 1)
        for r in dataset.rows
    )
    if dataset.policy == "epoch_random" and not changed:
        raise ValueError("No targets changed between epochs")
    result = {
        "config": config, "epochs": stats, "changed_targets_between_epochs": changed,
        "epoch_target_sha256": [h.hexdigest() for h in hashes],
        "schedule_sha256": hashlib.sha256(json.dumps(dataset.schedule).encode()).hexdigest(),
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "packages": package_versions(),
    }
    write_json(directory / "preflight.json", result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "runs")
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 2026, 3407), required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--revision", required=True, help="Pinned 40-character HF commit SHA")
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0, help="2-step GPU validation; never a paper result")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("--revision must be a commit SHA, not main")
    if args.smoke_steps < 0:
        raise ValueError("Negative smoke steps")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This matched protocol is single-process/single-GPU only")
    manifest = json.loads((args.data / "manifest.json").read_text())
    data_hash = check_data_hash(args.data / "lists.jsonl", manifest)
    rows = read_jsonl(args.data / "lists.jsonl")
    audit_rows(rows)
    if len(rows) != 4494:
        raise ValueError("Expected all 4,494 frozen lists")
    config = make_config(args, data_hash)
    suffix = "_SMOKE" if args.smoke_steps else ""
    directory = args.output / f"{args.policy}_seed{args.seed}{suffix}"
    if (directory / "training_manifest.json").exists() or (directory / "final").exists():
        raise FileExistsError(f"Completed run already exists: {directory}")
    directory.mkdir(parents=True, exist_ok=True)

    # Unsloth must be imported before Transformers/PyTorch in GPU runs.
    if not args.preflight_only:
        from unsloth import FastLanguageModel
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if not tokenizer.chat_template:
        raise ValueError("The selected tokenizer has no chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset = ScheduledDataset(rows, tokenizer, args.policy, args.seed, args.max_length)
    preflight_path = directory / "preflight.json"
    if args.preflight_only or not preflight_path.exists():
        checked = preflight(dataset, directory, config, tokenizer)
    else:
        checked = json.loads(preflight_path.read_text())
        if checked["config"] != config:
            raise ValueError("Preflight settings/source changed; use a fresh output root or repeat preflight")
    if args.preflight_only:
        return

    import torch
    from torch.utils.data import SequentialSampler
    from transformers import Trainer, TrainingArguments, set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA GPU required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    if (directory / "started.json").exists():
        raise FileExistsError("Interrupted run exists. Keep it for audit and select a fresh --output directory")
    set_seed(args.seed)
    model, loaded_tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        revision=args.revision,
        max_seq_length=args.max_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        use_exact_model_name=True,
    )
    if loaded_tokenizer.get_vocab() != tokenizer.get_vocab() or loaded_tokenizer.chat_template != tokenizer.chat_template:
        raise ValueError("Unsloth tokenizer differs from preflight tokenizer")
    model = FastLanguageModel.get_peft_model(
        model, r=128, lora_alpha=256, lora_dropout=0.05, bias="none",
        target_modules=config["target_modules"], use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    model.config.use_cache = False

    class ScheduledTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

    arguments = TrainingArguments(
        output_dir=str(directory / "checkpoints"),
        # The dataset already contains TWO logical epochs. Do not set this to 2.
        num_train_epochs=1, max_steps=args.smoke_steps or -1,
        per_device_train_batch_size=2, gradient_accumulation_steps=1,
        learning_rate=1e-5, warmup_steps=20, weight_decay=0.01,
        lr_scheduler_type="cosine", optim="adamw_torch",
        bf16=True, fp16=False, tf32=False, seed=args.seed, data_seed=args.seed,
        dataloader_num_workers=0, dataloader_drop_last=False,
        remove_unused_columns=False, report_to="none", logging_steps=25,
        save_strategy="steps", save_steps=2247, save_total_limit=2,
    )
    trainer = ScheduledTrainer(
        model=model, args=arguments, train_dataset=dataset,
        data_collator=ResponseCollator(tokenizer.pad_token_id), processing_class=tokenizer,
    )
    environment = package_versions()
    write_json(directory / "started.json", {"config": config, "packages": environment})
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = trainer.train()
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start
    memory = {"allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    expected_steps = args.smoke_steps or 4494
    if trainer.state.global_step != expected_steps:
        raise RuntimeError(f"Unexpected optimizer steps: {trainer.state.global_step}, expected {expected_steps}")
    trainable_parameters = save_exact_checkpoint(model, tokenizer, directory)
    hardware = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    write_json(directory / "training_manifest.json", {
        "status": "smoke_only" if args.smoke_steps else "complete", "config": config,
        "packages": environment, "hardware": hardware,
        "training_seconds": training_seconds, "peak_training_vram": memory,
        "optimizer_steps": trainer.state.global_step,
        "preflight": checked, "train_metrics": result.metrics,
        "checkpoint": str((directory / "final").resolve()),
        "trainable_parameters": trainable_parameters,
    })
    print(f"Saved {'SMOKE (exclude from paper)' if args.smoke_steps else 'completed'} run: {directory}")


if __name__ == "__main__":
    main()
