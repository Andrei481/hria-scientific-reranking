"""Reload a local checkpoint in native Transformers, without importing Unsloth."""
import argparse
import json
from pathlib import Path
import sys


def require_unpatched_process():
    if any(name.split(".")[0] in ("unsloth", "unsloth_zoo") for name in sys.modules):
        raise RuntimeError("Run this verifier in a fresh Python process that has not imported Unsloth")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    require_unpatched_process()
    if not (args.checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint config not found: {args.checkpoint}")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    require_unpatched_process()
    if args.device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Expose exactly one BF16 CUDA GPU with CUDA_VISIBLE_DEVICES")
        dtype, device = torch.bfloat16, "cuda:0"
        placement = {"device_map": {"": 0}}
    else:
        dtype, device, placement = torch.float32, "cpu", {}
    # Transformers 5 renamed torch_dtype to dtype; retain CPU-test compatibility
    # with the older Transformers installed on the development workstation.
    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        str(args.checkpoint), local_files_only=True, **{dtype_key: dtype}, **placement,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True)
    inputs = tokenizer(
        "Rank these passages by relevance.", return_tensors="pt", return_token_type_ids=False,
    ).to(device)
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits
    if not torch.isfinite(logits).all().item():
        raise RuntimeError("Reloaded checkpoint produced nonfinite logits")
    require_unpatched_process()
    report = {
        "status": "passed", "checkpoint": str(args.checkpoint.resolve()),
        "device": args.device, "unsloth_imported": False,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "logits_shape": list(logits.shape),
        "scope": "native Transformers reload and finite logits; not equivalence or ranking quality",
    }
    (args.checkpoint / "reload_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Checkpoint reload check passed: {args.checkpoint}")


if __name__ == "__main__":
    main()
