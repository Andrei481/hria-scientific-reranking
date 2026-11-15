from __future__ import annotations

import argparse
import hashlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from litsearch_document_disjoint import (
    DEFAULT_LEFT_SET,
    DEFAULT_RIGHT_SET,
    filter_entry,
    load_jsonl_or_json,
    load_overlap_blacklist,
    parse_match_types,
    safe_list_of_ints,
    write_json,
    write_jsonl,
)


def query_identity(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("query_set", "")),
        str(row.get("query", "")),
        int(row.get("specificity", 0)),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_output_stem(path: Path) -> str:
    name = path.name
    for suffix in (".jsonl", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return re.sub(r"\.top\d+$", "", name)


def prepare_rows(
    rows: list[dict[str, Any]],
    blacklist: set[int],
    *,
    source_depth: int,
    target_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    if source_depth < target_k or target_k <= 0:
        raise ValueError("source_depth must be >= target_k > 0")

    identities: set[tuple[str, str, int]] = set()
    kept_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    removed_mentions = 0

    for row in rows:
        identity = query_identity(row)
        if identity in identities:
            raise ValueError(f"duplicate query identity: {identity!r}")
        identities.add(identity)
        if "pre_reranked" in row:
            raise ValueError(f"pre_reranked input is not allowed: {identity!r}")

        retrieved = safe_list_of_ints(row.get("retrieved", []))
        if len(retrieved) < source_depth:
            raise ValueError(
                f"query {identity!r} has {len(retrieved)} retrieved candidates; "
                f"need at least {source_depth}"
            )
        source = retrieved[:source_depth]
        if len(set(source)) != len(source):
            raise ValueError(f"duplicate candidate in query {identity!r}")

        source_row = deepcopy(row)
        source_row["retrieved"] = source
        kept, excluded = filter_entry(source_row, blacklist, exclude_gold_overlap=True)
        if excluded is not None:
            excluded_rows.append(excluded)
            continue
        assert kept is not None

        clean = safe_list_of_ints(kept["retrieved"])
        if len(clean) < target_k:
            raise ValueError(
                f"query {identity!r} has only {len(clean)} clean candidates; "
                f"need {target_k}"
            )
        removed = [doc_id for doc_id in source if doc_id in blacklist]
        kept["retrieved"] = clean[:target_k]
        kept["document_disjoint"].update(
            {
                "removed_retrieved_ids": sorted(set(removed)),
                "removed_retrieved_count": len(removed),
                "source_depth": source_depth,
                "target_k": target_k,
            }
        )
        removed_mentions += len(removed)
        kept_rows.append(kept)

    stats = {
        "input_queries": len(rows),
        "kept_queries": len(kept_rows),
        "excluded_queries": len(excluded_rows),
        "blacklisted_doc_ids": len(blacklist),
        "removed_retrieved_mentions": removed_mentions,
        "source_depth": source_depth,
        "target_k": target_k,
    }
    return kept_rows, excluded_rows, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare fixed-depth strict LitSearch document-disjoint inputs."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--overlap-details", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-depth", type=int, default=300)
    parser.add_argument("--target-k", type=int, default=100)
    parser.add_argument("--dataset-revision", default="9573fb2")
    parser.add_argument("--match-types", default="text_hash")
    parser.add_argument("--left-set", default=DEFAULT_LEFT_SET)
    parser.add_argument("--right-set", default=DEFAULT_RIGHT_SET)
    parser.add_argument("--expected-blacklist-size", type=int, default=227)
    parser.add_argument("--expected-input-queries", type=int, default=597)
    parser.add_argument("--expected-kept-queries", type=int, default=595)
    parser.add_argument("--expected-excluded-queries", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    match_types = parse_match_types(args.match_types)
    blacklist = load_overlap_blacklist(
        args.overlap_details,
        match_types=match_types,
        left_set=args.left_set,
        right_set=args.right_set,
    )
    if len(blacklist) != args.expected_blacklist_size:
        raise ValueError(
            f"expected {args.expected_blacklist_size} blacklisted IDs, got {len(blacklist)}"
        )

    rows = load_jsonl_or_json(args.input)
    kept, excluded, stats = prepare_rows(
        rows, blacklist, source_depth=args.source_depth, target_k=args.target_k
    )
    if stats["input_queries"] != args.expected_input_queries:
        raise ValueError(
            f"expected {args.expected_input_queries} input queries, "
            f"got {stats['input_queries']}"
        )
    if len(kept) != args.expected_kept_queries:
        raise ValueError(
            f"expected {args.expected_kept_queries} retained queries, got {len(kept)}"
        )
    if stats["excluded_queries"] != args.expected_excluded_queries:
        raise ValueError(
            f"expected {args.expected_excluded_queries} excluded queries, "
            f"got {stats['excluded_queries']}"
        )
    if any(set(safe_list_of_ints(row["corpusids"])) & blacklist for row in kept):
        raise AssertionError("retained gold document intersects blacklist")
    if any(set(safe_list_of_ints(row["retrieved"])) & blacklist for row in kept):
        raise AssertionError("retained candidate intersects blacklist")
    if any(len(safe_list_of_ints(row["retrieved"])) != args.target_k for row in kept):
        raise AssertionError("retained candidate list has wrong depth")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = derive_output_stem(args.input)
    output_path = args.output_dir / f"{stem}.document_disjoint.jsonl"
    excluded_path = args.output_dir / f"{stem}.excluded_overlap_queries.jsonl"
    manifest_path = args.output_dir / f"{stem}.document_disjoint.manifest.json"
    write_jsonl(output_path, kept)
    write_jsonl(excluded_path, excluded)
    manifest = {
        "schema_version": 1,
        "selection": {
            "match_types": sorted(match_types),
            "left_set": args.left_set,
            "right_set": args.right_set,
        },
        "configuration": {
            "source_depth": args.source_depth,
            "target_k": args.target_k,
            "exclude_gold_overlap": True,
            "dataset_revision": args.dataset_revision,
        },
        "counts": stats,
        "files": {
            "input": {"path": str(args.input), "sha256": sha256_file(args.input)},
            "overlap_details": {
                "path": str(args.overlap_details),
                "sha256": sha256_file(args.overlap_details),
            },
            "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
            "excluded": {
                "path": str(excluded_path),
                "sha256": sha256_file(excluded_path),
            },
        },
    }
    write_json(manifest_path, manifest)
    print(f"Prepared {len(kept)} clean queries: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
