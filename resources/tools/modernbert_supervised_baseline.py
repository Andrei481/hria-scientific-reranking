"""Shared, dependency-light helpers for the controlled ModernBERT baseline.

The notebooks under ``New folder/modernbert_supervised_baseline`` import this
module.  Keeping data construction and output-schema validation here makes the
controlled experiment auditable and lets the lightweight parts be unit tested
without downloading a model.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from copy import deepcopy
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


ScoreFunction = Callable[[list[tuple[str, str]]], Sequence[float]]


def resolve_existing_path(
    *,
    env_name: str,
    candidates: Sequence[str | Path],
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve an explicit environment override or the first existing candidate."""
    environment = os.environ if environ is None else environ
    configured = environment.get(env_name)
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.exists():
            return configured_path
        raise FileNotFoundError(
            f"{env_name} points to a missing path: {configured_path}"
        )

    checked = [Path(candidate).expanduser() for candidate in candidates]
    for candidate in checked:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate {env_name}. Checked: "
        + ", ".join(str(candidate) for candidate in checked)
    )


def stable_seed(row_id: int, query_text: str) -> int:
    """Reproduce the seed used by the original LLM pool augmentation."""
    digest = hashlib.md5(
        f"{row_id}||{query_text}".encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    return int(digest[:8], 16)


def pool_signature(documents: Sequence[tuple[Any, int]]) -> tuple[int, ...]:
    """Set-based pool identity, matching the LLM training notebook."""
    return tuple(sorted(int(document_id) for document_id, _ in documents))


def sample_unique_pools_for_query(
    row: Mapping[str, Any],
    *,
    rng: random.Random,
    n_variants: int = 3,
    target_size: int = 50,
    max_positives: int = 5,
    hard_neg_top: int | None = 100,
    max_total_tries: int = 3000,
) -> list[list[tuple[int, int]]]:
    """Sample pools with the same algorithm used for LLM supervision."""
    pool = [int(document_id) for document_id in row["candidate_pool"]]
    scores = [int(score) for score in row["candidate_scores"]]
    if len(pool) != len(scores):
        raise ValueError("candidate_pool and candidate_scores have different lengths.")

    negatives = [(document_id, score) for document_id, score in zip(pool, scores) if score == 0]
    positives_1 = [(document_id, score) for document_id, score in zip(pool, scores) if score == 1]
    positives_2 = [(document_id, score) for document_id, score in zip(pool, scores) if score >= 2]
    all_positives = positives_2 + positives_1
    # This is deliberately computed before hard-negative truncation to reproduce
    # the fill behavior in train_llama_seed.ipynb exactly.
    all_documents = all_positives + negatives

    if hard_neg_top is not None and len(negatives) > hard_neg_top:
        negatives = negatives[:hard_neg_top]
    if not all_documents:
        return []

    modes = ["only_2", "only_1", "mixed", "any", "any", "any"]
    seen: set[tuple[int, ...]] = set()
    variants: list[list[tuple[int, int]]] = []

    tries = 0
    while len(variants) < n_variants and tries < max_total_tries:
        mode = modes[tries % len(modes)]
        tries += 1

        positives: list[tuple[int, int]] = []
        if mode == "only_2":
            count = min(len(positives_2), max_positives)
            positives = rng.sample(positives_2, count) if count else []
        elif mode == "only_1":
            count = min(len(positives_1), max_positives)
            positives = rng.sample(positives_1, count) if count else []
        elif mode == "mixed":
            count_2 = min(len(positives_2), max_positives // 2)
            count_1 = min(len(positives_1), max_positives - count_2)
            if count_2:
                positives.extend(rng.sample(positives_2, count_2))
            if count_1:
                positives.extend(rng.sample(positives_1, count_1))
        else:
            count = min(len(all_positives), max_positives)
            positives = rng.sample(all_positives, count) if count else []

        needed = target_size - len(positives)
        if needed < 0:
            positives = positives[:target_size]
            needed = 0
        count_negatives = min(len(negatives), needed)
        selected = positives + (
            rng.sample(negatives, count_negatives) if count_negatives else []
        )

        if not positives and all_positives and max_positives > 0:
            selected.insert(0, rng.choice(all_positives))

        if len(selected) < target_size:
            used_ids = {document_id for document_id, _ in selected}
            remaining = [
                document
                for document in all_documents
                if document[0] not in used_ids
            ]
            selected.extend(remaining[: target_size - len(selected)])

        signature = pool_signature(selected)
        if signature in seen:
            continue
        seen.add(signature)
        rng.shuffle(selected)
        variants.append(selected)

    return variants


def _lookup_text(corpus_lookup: Mapping[Any, str], document_id: int) -> str | None:
    text = corpus_lookup.get(document_id)
    if text is None:
        text = corpus_lookup.get(str(document_id))
    if text is None or not str(text).strip():
        return None
    return str(text)


def materialize_listwise_supervision(
    rows: Iterable[Mapping[str, Any]],
    corpus_lookup: Mapping[Any, str],
    *,
    n_variants: int = 3,
    target_size: int = 50,
    max_positives: int = 5,
    hard_neg_top: int | None = 100,
    max_total_tries: int = 3000,
    document_word_limit: int = 300,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create labeled lists from the exact DORIS-MAE-derived supervision."""
    input_rows = [dict(row) for row in rows]
    eligible: list[tuple[int, dict[str, Any]]] = []
    removed_ids: list[int] = []

    for row_id, row in enumerate(input_rows):
        pool = row.get("candidate_pool", [])
        scores = row.get("candidate_scores", [])
        if len(pool) != len(scores):
            raise ValueError(
                f"Row {row_id}: candidate_pool and candidate_scores have different lengths."
            )
        if not any(int(score) > 0 for score in scores):
            removed_ids.append(row_id)
            continue
        missing = [
            int(document_id)
            for document_id in pool
            if _lookup_text(corpus_lookup, int(document_id)) is None
        ]
        if missing:
            raise ValueError(
                f"Missing corpus text for row {row_id}; first IDs: {missing[:20]}"
            )
        eligible.append((row_id, row))

    examples: list[dict[str, Any]] = []
    variant_counts: dict[str, int] = {}
    for row_id, row in eligible:
        query = str(row["query_text"])
        variants = sample_unique_pools_for_query(
            row,
            rng=random.Random(stable_seed(row_id, query)),
            n_variants=n_variants,
            target_size=target_size,
            max_positives=max_positives,
            hard_neg_top=hard_neg_top,
            max_total_tries=max_total_tries,
        )
        variant_counts[str(row_id)] = len(variants)
        if len(variants) != n_variants:
            raise ValueError(
                f"Row {row_id} produced {len(variants)} unique pools, expected {n_variants}."
            )

        for variant_id, selected in enumerate(variants):
            candidate_ids = [document_id for document_id, _ in selected]
            labels = [score for _, score in selected]
            documents = [
                " ".join(
                    _lookup_text(corpus_lookup, document_id).split()[
                        :document_word_limit
                    ]
                )
                for document_id in candidate_ids
            ]
            examples.append(
                {
                    "query": query,
                    "docs": documents,
                    "labels": labels,
                    "candidate_ids": candidate_ids,
                    "source_row_id": row_id,
                    "aspect_id": str(row.get("aspect_id", "")),
                    "original_query_type": str(row.get("original_query_type", "")),
                    "variant_id": variant_id,
                }
            )

    audit = {
        "input_rows": len(input_rows),
        "eligible_rows": len(eligible),
        "removed_zero_positive_rows": len(removed_ids),
        "removed_source_row_ids": removed_ids,
        "variants_per_eligible_row": n_variants,
        "target_documents_per_list": target_size,
        "maximum_positives_per_list": max_positives,
        "hard_negative_prefix": hard_neg_top,
        "document_word_limit": document_word_limit,
        "output_examples": len(examples),
        "list_size_histogram": {
            str(size): sum(len(example["docs"]) == size for example in examples)
            for size in sorted({len(example["docs"]) for example in examples})
        },
        "minimum_generated_variants": min(variant_counts.values(), default=0),
        "maximum_generated_variants": max(variant_counts.values(), default=0),
    }
    return examples, audit


def rerank_litsearch_rows(
    rows: Iterable[Mapping[str, Any]],
    corpus_lookup: Mapping[Any, str],
    score_function: ScoreFunction,
) -> list[dict[str, Any]]:
    """Pointwise-score and rerank each existing LitSearch candidate pool."""
    outputs: list[dict[str, Any]] = []
    for row_index, original in enumerate(rows):
        output = deepcopy(dict(original))
        query = str(output["query"])
        candidate_ids = [int(document_id) for document_id in output["retrieved"]]
        documents: list[str] = []
        for document_id in candidate_ids:
            document = _lookup_text(corpus_lookup, document_id)
            if document is None:
                raise ValueError(
                    f"Missing corpus text for LitSearch row {row_index}, ID {document_id}."
                )
            documents.append(document)

        scores = [float(value) for value in score_function(list(zip([query] * len(documents), documents)))]
        if len(scores) != len(candidate_ids):
            raise ValueError("The score function returned the wrong number of scores.")
        order = sorted(range(len(candidate_ids)), key=lambda index: (-scores[index], index))
        reranked = [candidate_ids[index] for index in order]
        if len(reranked) != len(set(reranked)) or set(reranked) != set(candidate_ids):
            raise ValueError(f"Candidate-set corruption in LitSearch row {row_index}.")

        output["pre_reranked"] = candidate_ids
        output["retrieved"] = reranked
        output["reranker_scores"] = [scores[index] for index in order]
        outputs.append(output)
    return outputs


def build_csfcube_rankings(
    annotations: Mapping[str, Mapping[str, Any]],
    paper_lookup: Mapping[Any, str],
    score_function: ScoreFunction,
) -> dict[str, list[list[Any]]]:
    """Create the exact ranked-pool JSON schema consumed by ranking_eval.py."""
    rankings: dict[str, list[list[Any]]] = {}
    for query_id, annotation in annotations.items():
        query_text = _lookup_text(paper_lookup, int(query_id))
        if query_text is None:
            raise ValueError(f"Missing CSFCube query text for paper {query_id}.")
        candidate_ids = [str(document_id) for document_id in annotation["cands"]]
        documents: list[str] = []
        for candidate_id in candidate_ids:
            document = _lookup_text(paper_lookup, int(candidate_id))
            if document is None:
                raise ValueError(
                    f"Missing CSFCube candidate text for paper {candidate_id}."
                )
            documents.append(document)

        scores = [float(value) for value in score_function(list(zip([query_text] * len(documents), documents)))]
        if len(scores) != len(candidate_ids):
            raise ValueError("The score function returned the wrong number of scores.")
        order = sorted(range(len(candidate_ids)), key=lambda index: (-scores[index], index))
        ranked = [[candidate_ids[index], scores[index]] for index in order]
        if len(ranked) != len({item[0] for item in ranked}) or {
            item[0] for item in ranked
        } != set(candidate_ids):
            raise ValueError(f"Candidate-set corruption for CSFCube query {query_id}.")
        rankings[str(query_id)] = ranked
    return rankings


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(names: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def reproducibility_manifest() -> dict[str, Any]:
    """Capture software, Python, OS, GPU, and CUDA details without failing on CPU."""
    manifest: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(
            [
                "torch",
                "sentence-transformers",
                "transformers",
                "datasets",
                "accelerate",
                "numpy",
                "scipy",
            ]
        ),
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        manifest["nvidia_smi"] = [
            line.strip() for line in result.stdout.splitlines() if line.strip()
        ]
    except (FileNotFoundError, subprocess.SubprocessError):
        manifest["nvidia_smi"] = []
    return manifest
