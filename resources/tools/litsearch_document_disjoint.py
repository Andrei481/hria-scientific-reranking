from __future__ import annotations

import argparse
import csv
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


DEFAULT_LEFT_SET = "DORIS actual prompt candidates"
DEFAULT_RIGHT_SET = "LitSearch text docs"
DEFAULT_MATCH_TYPES = ("text_hash",)


def safe_list_of_ints(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(value)]


def recall_at_k(retrieved: list[int], relevant: list[int], k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def ndcg_at_k(retrieved: list[int], relevant: list[int], k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0
    relevant_set = set(relevant)
    dcg = 0.0
    for idx, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(idx + 2.0)
    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(idx + 2.0) for idx in range(ideal_count))
    return dcg / idcg if idcg else 0.0


def categorize(entry: dict[str, Any]) -> str:
    query_set = str(entry.get("query_set", "")).lower()
    specificity = int(entry.get("specificity", 0))
    if "inline" in query_set:
        prefix = "inline"
    elif query_set in {"manual_acl", "manual_iclr"}:
        prefix = "author"
    else:
        prefix = "other"
    return f"{prefix}_broad" if specificity == 0 else f"{prefix}_specific"


def aggregate_metrics(per_query_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not per_query_metrics:
        return {"count": 0, "R@20": 0.0, "R@5": 0.0, "NDCG@20": 0.0, "NDCG@5": 0.0}
    count = len(per_query_metrics)
    return {
        "count": count,
        "R@20": sum(row["r20"] for row in per_query_metrics) / count,
        "R@5": sum(row["r5"] for row in per_query_metrics) / count,
        "NDCG@20": sum(row["n20"] for row in per_query_metrics) / count,
        "NDCG@5": sum(row["n5"] for row in per_query_metrics) / count,
    }


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, float]]] = {
        "inline_broad": [],
        "inline_specific": [],
        "author_broad": [],
        "author_specific": [],
    }
    per_query_records: list[dict[str, float]] = []

    for entry in rows:
        retrieved = safe_list_of_ints(entry.get("retrieved", []))
        relevant = safe_list_of_ints(entry.get("corpusids", []))
        per_query = {
            "r20": recall_at_k(retrieved, relevant, 20),
            "r5": recall_at_k(retrieved, relevant, 5),
            "n20": ndcg_at_k(retrieved, relevant, 20),
            "n5": ndcg_at_k(retrieved, relevant, 5),
        }
        per_query_records.append(per_query)
        buckets.setdefault(categorize(entry), []).append(per_query)

    broad = buckets.get("inline_broad", []) + buckets.get("author_broad", [])
    specific = buckets.get("inline_specific", []) + buckets.get("author_specific", [])

    return {
        "inline_broad": aggregate_metrics(buckets.get("inline_broad", [])),
        "inline_specific": aggregate_metrics(buckets.get("inline_specific", [])),
        "author_broad": aggregate_metrics(buckets.get("author_broad", [])),
        "author_specific": aggregate_metrics(buckets.get("author_specific", [])),
        "broad_avg": aggregate_metrics(broad),
        "specific_avg": aggregate_metrics(specific),
        "overall_avg": aggregate_metrics(per_query_records),
    }


def load_jsonl_or_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
        if not first:
            return []
        handle.seek(0)
        if first.lstrip().startswith("["):
            return json.load(handle)
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_overlap_blacklist(
    path: Path,
    *,
    match_types: set[str] | None = None,
    left_set: str = DEFAULT_LEFT_SET,
    right_set: str = DEFAULT_RIGHT_SET,
) -> set[int]:
    selected_match_types = match_types or set(DEFAULT_MATCH_TYPES)
    blacklist: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("left_set") != left_set:
                continue
            if row.get("right_set") != right_set:
                continue
            if row.get("match_type") not in selected_match_types:
                continue
            right_doc_id = str(row.get("right_doc_id", "")).strip()
            if right_doc_id:
                blacklist.add(int(right_doc_id))
    return blacklist


def filter_entry(
    entry: dict[str, Any],
    blacklist: set[int],
    *,
    exclude_gold_overlap: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    relevant = safe_list_of_ints(entry.get("corpusids", []))
    retrieved = safe_list_of_ints(entry.get("retrieved", []))
    pre_reranked = safe_list_of_ints(entry.get("pre_reranked", []))
    overlapping_gold = sorted(set(relevant) & blacklist)

    if overlapping_gold and exclude_gold_overlap:
        return None, {
            "reason": "gold_overlap",
            "query": entry.get("query", ""),
            "query_set": entry.get("query_set", ""),
            "specificity": entry.get("specificity", ""),
            "corpusids": relevant,
            "overlapping_gold_ids": overlapping_gold,
            "removed_retrieved_ids": sorted(set(retrieved) & blacklist),
        }

    kept = deepcopy(entry)
    kept["retrieved"] = [doc_id for doc_id in retrieved if doc_id not in blacklist]
    if "pre_reranked" in kept:
        kept["pre_reranked"] = [doc_id for doc_id in pre_reranked if doc_id not in blacklist]
    removed_retrieved = sorted(set(retrieved) & blacklist)
    kept["document_disjoint"] = {
        "removed_retrieved_ids": removed_retrieved,
        "removed_retrieved_count": len(removed_retrieved),
        "overlapping_gold_ids": overlapping_gold,
    }
    return kept, None


def filter_rows(
    rows: list[dict[str, Any]],
    blacklist: set[int],
    *,
    exclude_gold_overlap: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    kept_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    original_retrieved_count = 0
    filtered_retrieved_count = 0

    for entry in rows:
        original_retrieved_count += len(safe_list_of_ints(entry.get("retrieved", [])))
        kept, excluded = filter_entry(
            entry,
            blacklist,
            exclude_gold_overlap=exclude_gold_overlap,
        )
        if excluded is not None:
            excluded_rows.append(excluded)
            continue
        if kept is not None:
            filtered_retrieved_count += len(safe_list_of_ints(kept.get("retrieved", [])))
            kept_rows.append(kept)

    return kept_rows, excluded_rows, {
        "input_queries": len(rows),
        "kept_queries": len(kept_rows),
        "excluded_queries": len(excluded_rows),
        "blacklisted_doc_ids": len(blacklist),
        "original_retrieved_mentions": original_retrieved_count,
        "filtered_retrieved_mentions": filtered_retrieved_count,
        "removed_retrieved_mentions": original_retrieved_count - filtered_retrieved_count,
    }


def metrics_for_json(summary: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for name, metrics in summary.items():
        payload[name] = {
            "count": int(metrics["count"]),
            "R@20": metrics["R@20"],
            "R@5": metrics["R@5"],
            "NDCG@20": metrics["NDCG@20"],
            "NDCG@5": metrics["NDCG@5"],
            "R@20_percent": round(metrics["R@20"] * 100, 4),
            "R@5_percent": round(metrics["R@5"] * 100, 4),
            "NDCG@20_percent": round(metrics["NDCG@20"] * 100, 4),
            "NDCG@5_percent": round(metrics["NDCG@5"] * 100, 4),
        }
    return payload


def output_stem(input_path: Path) -> str:
    name = input_path.name
    for suffix in (".jsonl", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return input_path.stem


def parse_match_types(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create and evaluate a LitSearch document-disjoint subset."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--overlap-details",
        default=Path("leakage_analysis_outputs/prompt_level_overlap_details.csv"),
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--match-types", default="text_hash")
    parser.add_argument("--left-set", default=DEFAULT_LEFT_SET)
    parser.add_argument("--right-set", default=DEFAULT_RIGHT_SET)
    parser.add_argument(
        "--exclude-gold-overlap",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    blacklist = load_overlap_blacklist(
        args.overlap_details,
        match_types=parse_match_types(args.match_types),
        left_set=args.left_set,
        right_set=args.right_set,
    )
    rows = load_jsonl_or_json(args.input)
    kept_rows, excluded_rows, run_stats = filter_rows(
        rows,
        blacklist,
        exclude_gold_overlap=args.exclude_gold_overlap,
    )
    metrics = metrics_for_json(evaluate_rows(kept_rows))

    stem = output_stem(args.input)
    filtered_path = args.output_dir / f"{stem}.document_disjoint.jsonl"
    excluded_path = args.output_dir / f"{stem}.excluded_overlap_queries.jsonl"
    metrics_path = args.output_dir / f"{stem}.document_disjoint.metrics.json"
    summary_path = args.output_dir / f"{stem}.document_disjoint.summary.json"

    write_jsonl(filtered_path, kept_rows)
    write_jsonl(excluded_path, excluded_rows)
    write_json(metrics_path, metrics)
    write_json(
        summary_path,
        {
            "input": str(args.input),
            "overlap_details": str(args.overlap_details),
            "match_types": sorted(parse_match_types(args.match_types)),
            "left_set": args.left_set,
            "right_set": args.right_set,
            "exclude_gold_overlap": args.exclude_gold_overlap,
            **run_stats,
            "metrics": metrics,
        },
    )

    overall = metrics["overall_avg"]
    print(f"Input queries: {run_stats['input_queries']}")
    print(f"Kept document-disjoint queries: {run_stats['kept_queries']}")
    print(f"Excluded queries with overlapping gold docs: {run_stats['excluded_queries']}")
    print(f"Removed retrieved document mentions: {run_stats['removed_retrieved_mentions']}")
    print(
        "Overall document-disjoint: "
        f"R@20={overall['R@20_percent']:.2f}, "
        f"R@5={overall['R@5_percent']:.2f}, "
        f"NDCG@20={overall['NDCG@20_percent']:.2f}, "
        f"NDCG@5={overall['NDCG@5_percent']:.2f}"
    )
    print(f"Wrote filtered rows to: {filtered_path}")
    print(f"Wrote metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
