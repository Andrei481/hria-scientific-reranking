#!/usr/bin/env python3
"""Prepare deterministic listwise training instances from DORIS-MAE data."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hria_reranking.data.aspects import aggregate_full_queries, decompose_queries
from hria_reranking.data.pools import PoolConfig, sample_unique_pools
from hria_reranking.inference.prompts import format_listwise_prompt


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="DORIS-MAE JSON or aspect-row JSONL"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output training-instance JSONL"
    )
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--target-size", type=int, default=100)
    parser.add_argument("--max-positives", type=int, default=5)
    parser.add_argument(
        "--positive-mode",
        choices=("cycle", "only_2", "only_1", "mixed", "any"),
        default="cycle",
    )
    parser.add_argument(
        "--full-query",
        action="store_true",
        help="Use original queries and max-aggregate labels across their aspects",
    )
    return parser.parse_args()


def _read_records(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return json.loads(path.read_text(encoding="utf-8"))


def _corpus_lookup(records: Any) -> dict[int, str]:
    if not isinstance(records, Mapping):
        return {}
    lookup: dict[int, str] = {}
    for document in records.get("Corpus", []):
        title = str(document.get("title", "")).replace("_", " ")
        abstract = document.get("original_abstract", document.get("abstract", ""))
        lookup[int(document["abstract_id"])] = f"Title: {title}\nAbstract: {abstract}"
    return lookup


def _training_rows(records: Any, full_query: bool) -> list[dict[str, Any]]:
    if isinstance(records, Mapping) and "Query" in records:
        rows = (
            aggregate_full_queries(records)
            if full_query
            else decompose_queries(records)
        )
    elif isinstance(records, list) and not full_query:
        rows = [dict(row) for row in records]
    else:
        raise ValueError(
            "--full-query requires a DORIS-MAE JSON object, and JSONL input "
            "must contain aspect rows"
        )

    positive_rows = [
        row
        for row in rows
        if any(int(score) > 0 for score in row["candidate_scores"])
    ]
    for row_id, row in enumerate(positive_rows):
        row["row_id"] = row_id
    return positive_rows


def _passage_texts(
    row: Mapping[str, Any],
    selected: list[tuple[Any, int]],
    corpus: Mapping[int, str],
) -> list[str]:
    if "passages" in row:
        by_document = {
            int(document_id): str(text)
            for document_id, text in zip(row["candidate_pool"], row["passages"])
        }
    else:
        by_document = corpus
    return [
        by_document.get(int(document_id), "Error: Document text not found.")
        for document_id, _ in selected
    ]


def _target_permutation(selected: list[tuple[Any, int]]) -> list[int]:
    return sorted(
        range(1, len(selected) + 1),
        key=lambda identifier: (-int(selected[identifier - 1][1]), identifier),
    )


def main() -> int:
    args = _parse_arguments()
    if not args.input.is_file():
        print(f"error: input file does not exist: {args.input}", file=sys.stderr)
        return 2

    records = _read_records(args.input)
    rows = _training_rows(records, args.full_query)
    corpus = _corpus_lookup(records)
    config = PoolConfig(
        variants=args.variants,
        target_size=args.target_size,
        max_positives=args.max_positives,
        positive_mode=args.positive_mode,
    )

    output_rows: list[dict[str, Any]] = []
    unique_documents: set[int] = set()
    for row in rows:
        for variant_id, selected in enumerate(sample_unique_pools(row, config)):
            candidate_pool = [document_id for document_id, _ in selected]
            candidate_scores = [int(score) for _, score in selected]
            passages = _passage_texts(row, selected, corpus)
            prompt = format_listwise_prompt(str(row["query_text"]), passages)
            target = _target_permutation(selected)
            target_text = " > ".join(f"[{identifier}]" for identifier in target)
            unique_documents.update(int(document_id) for document_id in candidate_pool)
            output_rows.append(
                {
                    "row_id": row["row_id"],
                    "variant_id": variant_id,
                    "query_text": row["query_text"],
                    "aspect_id": row.get("aspect_id"),
                    "original_query_type": row.get("original_query_type"),
                    "candidate_pool": candidate_pool,
                    "candidate_scores": candidate_scores,
                    "prompt": prompt,
                    "target_permutation": target,
                    "conversations": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": target_text},
                    ],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        for output_row in output_rows:
            output_file.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "rows": len(rows),
                "prompts": len(output_rows),
                "unique_documents": len(unique_documents),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
