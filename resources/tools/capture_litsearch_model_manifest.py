from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import request


def fetch_served_model_name(endpoint_url: str) -> str:
    url = f"{endpoint_url.rstrip('/')}/v1/models"
    with request.urlopen(url, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    model_ids = [
        str(item.get("id", "")).strip()
        for item in payload.get("data", [])
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    ]
    if len(model_ids) != 1:
        raise ValueError(f"expected exactly one served model ID, got {model_ids!r}")
    return model_ids[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture a LitSearch model manifest.")
    parser.add_argument("--endpoint-url", required=True)
    parser.add_argument(
        "--model-family",
        required=True,
        choices=("gemma3_27b", "llama3", "gemma3_12b", "qwen2.5"),
    )
    parser.add_argument("--variant", required=True, choices=("base", "ft"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--quantization", required=True)
    parser.add_argument("--chat-template", required=True)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    served_model_name = fetch_served_model_name(args.endpoint_url)
    payload = {
        "model_family": args.model_family,
        "variant": args.variant,
        "model_id": served_model_name,
        "checkpoint": args.checkpoint,
        "quantization": args.quantization,
        "chat_template": args.chat_template,
        "endpoint": {
            "url": args.endpoint_url.rstrip("/"),
            "served_model_name": served_model_name,
        },
        "inference": {
            "temperature": 0.0,
            "max_output_tokens": args.max_output_tokens,
            "prompt_repetition": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
