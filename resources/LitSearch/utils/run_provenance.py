from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "model_family",
        "variant",
        "model_id",
        "checkpoint",
        "quantization",
        "chat_template",
        "endpoint",
        "inference",
    ):
        if key not in payload:
            raise ValueError(f"model manifest missing field: {key}")
    if payload["variant"] not in {"base", "ft"}:
        raise ValueError("variant must be base or ft")
    for key in ("url", "served_model_name"):
        if not str(payload["endpoint"].get(key, "")).strip():
            raise ValueError(f"model manifest missing endpoint.{key}")
    inference = payload["inference"]
    if inference.get("temperature") != 0.0:
        raise ValueError("temperature must be 0.0 for deterministic reranking")
    if (
        not isinstance(inference.get("max_output_tokens"), int)
        or inference["max_output_tokens"] <= 0
    ):
        raise ValueError("max_output_tokens must be a positive integer")
    if inference.get("prompt_repetition") is not False:
        raise ValueError("prompt_repetition must be false for the current reranker")
    return payload


def assert_corpus_coverage(
    rows: list[dict[str, Any]], corpusid_to_text: dict[int, str]
) -> None:
    candidate_ids = set()
    for row in rows:
        canonical_ids = []
        for doc_id in row.get("retrieved", []):
            try:
                canonical_id = int(doc_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid retrieved corpus ID: {doc_id!r}") from exc
            canonical_ids.append(canonical_id)
            candidate_ids.add(canonical_id)
        row["retrieved"] = canonical_ids
    missing = sorted(candidate_ids - set(corpusid_to_text))
    if missing:
        raise ValueError(f"missing corpus text for IDs: {missing[:20]}")
    blank = sorted(
        doc_id for doc_id in candidate_ids if not corpusid_to_text[doc_id].strip()
    )
    if blank:
        raise ValueError(f"blank corpus text for IDs: {blank[:20]}")


def build_rerank_run_manifest(
    *,
    retrieval_input: Path,
    model_manifest_path: Path,
    model_manifest: dict[str, Any],
    dataset_path: str,
    dataset_revision: str,
    dataset_fingerprint: str,
    max_k: int,
    reranker_source: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "retrieval_input": {
            "path": str(retrieval_input),
            "sha256": sha256_file(retrieval_input),
        },
        "model_manifest": {
            "path": str(model_manifest_path),
            "sha256": sha256_file(model_manifest_path),
            "payload": model_manifest,
        },
        "dataset": {
            "path": dataset_path,
            "revision": dataset_revision,
            "fingerprint": dataset_fingerprint,
        },
        "reranker": {
            "max_k": max_k,
            "source_sha256": sha256_file(reranker_source),
        },
    }


def ensure_resume_compatible(path: Path, expected: dict[str, Any]) -> None:
    if not path.exists():
        raise ValueError(f"existing output has no run manifest: {path}")
    actual = json.loads(path.read_text(encoding="utf-8"))
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
    if canonical(actual) != canonical(expected):
        raise ValueError("resume manifest mismatch; use a new output directory")
