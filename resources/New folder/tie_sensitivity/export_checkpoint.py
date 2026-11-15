"""Re-export a saved adapter against its pinned base, without retraining.

Writes a separate export_check directory and never changes the original run.
This is a deployment check, not a new training/evaluation result.
"""
import argparse
import gc
import json
from pathlib import Path
import re
import subprocess
import sys

from experiment import sha256_file, source_hashes, write_json
from train import package_versions, save_exact_checkpoint


def verify_export_in_subprocess(checkpoint, device="cuda"):
    """Do not let Unsloth's process-global patches reach native model reload."""
    subprocess.run([
        sys.executable, "-B", str(Path(__file__).with_name("verify_checkpoint.py")),
        "--checkpoint", str(checkpoint), "--device", device,
    ], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New directory; default: RUN/export_check")
    args = parser.parse_args()
    output = args.output or args.run / "export_check"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    manifest_path = args.run / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = manifest["config"]
    if manifest["status"] not in ("complete", "smoke_only"):
        raise ValueError("Expected a completed training or smoke run")
    if not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("Pinned 40-character model revision required")
    if config["quantization"] != "none":
        raise ValueError("This export check supports the unquantized matched protocol only")
    adapter = args.run / "adapter"
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Saved adapter not found: {adapter}")
    adapter_hashes = {p.name: sha256_file(p) for p in sorted(adapter.iterdir()) if p.is_file()}

    # Match training's import order and loader. No optimizer or Trainer is created.
    from unsloth import FastLanguageModel
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Expose exactly one BF16 CUDA GPU with CUDA_VISIBLE_DEVICES")
    base, _ = FastLanguageModel.from_pretrained(
        model_name=config["model"], revision=config["revision"],
        max_seq_length=config["max_length"], dtype=torch.bfloat16,
        load_in_4bit=False, use_exact_model_name=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(adapter), local_files_only=True)
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
    model.eval()
    trainable_parameters = save_exact_checkpoint(model, tokenizer, output)
    del model, base
    gc.collect()
    torch.cuda.empty_cache()

    # Unsloth patches Transformer classes globally. Native reload must happen in
    # another interpreter, not in this process after importing Unsloth.
    verify_export_in_subprocess(output / "final")
    write_json(output / "export_manifest.json", {
        "status": "export_and_reload_checked", "training_manifest_sha256": sha256_file(manifest_path),
        "source_adapter_sha256": adapter_hashes,
        "model": config["model"], "revision": config["revision"],
        "export_method": "peft_merge_and_unload_loaded_base",
        "trainable_parameters_before_merge": trainable_parameters,
        "packages": package_versions(), "source_hashes": source_hashes(),
        "reload_check": "finite logits on one short input; not an equivalence or quality test",
    })
    print(f"Exact-base export and reload check passed: {output / 'final'}")


if __name__ == "__main__":
    main()
