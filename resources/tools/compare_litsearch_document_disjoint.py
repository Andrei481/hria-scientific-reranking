from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from litsearch_document_disjoint import (
    load_jsonl_or_json,
    ndcg_at_k,
    recall_at_k,
    write_json,
)
from prepare_litsearch_document_disjoint import query_identity


METRIC_KEYS = ("R@5", "R@20", "NDCG@5", "NDCG@20")


def metric_record(row: dict) -> dict[str, float]:
    retrieved = [int(value) for value in row.get("retrieved", [])]
    relevant = [int(value) for value in row.get("corpusids", [])]
    return {
        "R@5": recall_at_k(retrieved, relevant, 5),
        "R@20": recall_at_k(retrieved, relevant, 20),
        "NDCG@5": ndcg_at_k(retrieved, relevant, 5),
        "NDCG@20": ndcg_at_k(retrieved, relevant, 20),
    }


def index_rows(rows: list[dict], label: str) -> dict[tuple[str, str, int], dict]:
    indexed = {}
    for row in rows:
        key = query_identity(row)
        if key in indexed:
            raise ValueError(f"duplicate query identity in {label}: {key!r}")
        indexed[key] = row
    return indexed


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap(
    base: list[float], ft: list[float], *, resamples: int, seed: int
) -> dict[str, float]:
    if len(base) != len(ft) or not base:
        raise ValueError("paired bootstrap requires equal non-empty samples")
    deltas = [ft_value - base_value for base_value, ft_value in zip(base, ft)]
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        means.append(
            sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        )
    return {
        "mean": sum(deltas) / len(deltas),
        "ci95_low": percentile(means, 0.025),
        "ci95_high": percentile(means, 0.975),
        "resamples": resamples,
        "seed": seed,
    }


def summarize(records: list[dict[str, float]]) -> dict[str, float | int]:
    return {
        "count": len(records),
        **{
            metric: sum(record[metric] for record in records) / len(records)
            for metric in METRIC_KEYS
        },
    }


def compare_rows(
    *,
    model_family: str,
    retriever: str,
    original_base: list[dict],
    original_ft: list[dict],
    strict_base: list[dict],
    strict_ft: list[dict],
    resamples: int,
    seed: int,
) -> dict:
    indexed = {
        "original_base": index_rows(original_base, "original_base"),
        "original_ft": index_rows(original_ft, "original_ft"),
        "strict_base": index_rows(strict_base, "strict_base"),
        "strict_ft": index_rows(strict_ft, "strict_ft"),
    }
    strict_keys = set(indexed["strict_base"])
    if strict_keys != set(indexed["strict_ft"]):
        raise ValueError("strict query identities differ between base and FT")
    for label in ("original_base", "original_ft"):
        missing = strict_keys - set(indexed[label])
        if missing:
            raise ValueError(f"{label} is missing strict queries: {sorted(missing)!r}")

    ordered_keys = sorted(strict_keys)
    for key in ordered_keys:
        gold_sets = {
            tuple(int(value) for value in indexed[label][key].get("corpusids", []))
            for label in indexed
        }
        if len(gold_sets) != 1:
            raise ValueError(f"gold labels differ for query {key!r}")

    condition_sources = {
        "original_same_queries": (indexed["original_base"], indexed["original_ft"]),
        "document_disjoint": (indexed["strict_base"], indexed["strict_ft"]),
    }
    conditions = {}
    per_query = []
    for condition, (base_index, ft_index) in condition_sources.items():
        base_records = [metric_record(base_index[key]) for key in ordered_keys]
        ft_records = [metric_record(ft_index[key]) for key in ordered_keys]
        deltas = {}
        for metric in METRIC_KEYS:
            seed_material = (
                f"{model_family}|{retriever}|{condition}|{metric}|{seed}"
            ).encode("utf-8")
            metric_seed = int(hashlib.sha256(seed_material).hexdigest()[:8], 16)
            deltas[metric] = paired_bootstrap(
                [record[metric] for record in base_records],
                [record[metric] for record in ft_records],
                resamples=resamples,
                seed=metric_seed,
            )
        conditions[condition] = {
            "base": summarize(base_records),
            "ft": summarize(ft_records),
            "ft_minus_base": deltas,
        }
        for key, base_record, ft_record in zip(
            ordered_keys, base_records, ft_records
        ):
            for metric in METRIC_KEYS:
                per_query.append(
                    {
                        "query_set": key[0],
                        "query": key[1],
                        "specificity": key[2],
                        "condition": condition,
                        "model_family": model_family,
                        "retriever": retriever,
                        "metric": metric,
                        "base": base_record[metric],
                        "ft": ft_record[metric],
                        "delta": ft_record[metric] - base_record[metric],
                    }
                )
    return {
        "model_family": model_family,
        "retriever": retriever,
        "query_count": len(ordered_keys),
        "conditions": conditions,
        "per_query": per_query,
    }


def load_run_input_sha256(output_path: Path) -> str:
    manifest_path = Path(f"{output_path}.run_manifest.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return str(payload["retrieval_input"]["sha256"])


def validate_strict_input_identity(base_rows: list[dict], ft_rows: list[dict]) -> None:
    base_index = index_rows(base_rows, "strict_base")
    ft_index = index_rows(ft_rows, "strict_ft")
    if set(base_index) != set(ft_index):
        raise ValueError("strict query identities differ between base and FT")
    for key in base_index:
        if "pre_reranked" not in base_index[key]:
            raise ValueError(f"strict input missing pre_reranked for query {key!r}")
        if "pre_reranked" not in ft_index[key]:
            raise ValueError(f"strict input missing pre_reranked for query {key!r}")
        base_candidates = [
            int(value) for value in base_index[key]["pre_reranked"]
        ]
        ft_candidates = [int(value) for value in ft_index[key]["pre_reranked"]]
        if base_candidates != ft_candidates:
            raise ValueError(
                f"strict input candidate order differs for query {key!r}"
            )


def compare_spec(
    spec: dict,
    *,
    resamples: int,
    seed: int,
) -> dict:
    paths = {
        name: Path(spec[name])
        for name in (
            "original_base",
            "original_ft",
            "document_disjoint_base",
            "document_disjoint_ft",
        )
    }
    strict_base = load_jsonl_or_json(paths["document_disjoint_base"])
    strict_ft = load_jsonl_or_json(paths["document_disjoint_ft"])
    validate_strict_input_identity(strict_base, strict_ft)
    if len(strict_base) != 595:
        raise ValueError(
            f"expected 595 strict queries, got {len(strict_base)}"
        )
    base_sha = load_run_input_sha256(paths["document_disjoint_base"])
    ft_sha = load_run_input_sha256(paths["document_disjoint_ft"])
    if base_sha != ft_sha:
        raise ValueError("strict base and FT retrieval input checksums differ")
    result = compare_rows(
        model_family=spec["model_family"],
        retriever=spec["retriever"],
        original_base=load_jsonl_or_json(paths["original_base"]),
        original_ft=load_jsonl_or_json(paths["original_ft"]),
        strict_base=strict_base,
        strict_ft=strict_ft,
        resamples=resamples,
        seed=seed,
    )
    result["retrieval_input_sha256"] = base_sha
    result["paths"] = {name: str(path) for name, path in paths.items()}
    return result


def expansion_decision(comparisons: list[dict]) -> dict:
    triggers = []
    for comparison in comparisons:
        original = comparison["conditions"]["original_same_queries"]["ft_minus_base"]
        strict = comparison["conditions"]["document_disjoint"]["ft_minus_base"]
        for metric in METRIC_KEYS:
            original_delta = original[metric]["mean"]
            strict_delta = strict[metric]["mean"]
            sign = lambda value: (value > 0.0) - (value < 0.0)
            change_pp = abs(strict_delta - original_delta) * 100.0
            if sign(original_delta) != sign(strict_delta) or change_pp > 1.0:
                triggers.append(
                    {
                        "model_family": comparison["model_family"],
                        "retriever": comparison["retriever"],
                        "metric": metric,
                        "original_delta_percentage_points": original_delta * 100.0,
                        "strict_delta_percentage_points": strict_delta * 100.0,
                        "absolute_change_percentage_points": change_pp,
                    }
                )
    return {"run_secondary_models": bool(triggers), "triggers": triggers}


def write_outputs(output_dir: Path, comparisons: list[dict], decision: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "comparison.json",
        {
            "schema_version": 1,
            "comparisons": comparisons,
            "expansion_decision": decision,
        },
    )
    with (output_dir / "aggregate_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "model_family",
            "retriever",
            "condition",
            "variant",
            "count",
            *METRIC_KEYS,
            *(f"{metric}_percent" for metric in METRIC_KEYS),
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for comparison in comparisons:
            for condition, condition_result in comparison["conditions"].items():
                for variant in ("base", "ft"):
                    metrics = condition_result[variant]
                    writer.writerow(
                        {
                            "model_family": comparison["model_family"],
                            "retriever": comparison["retriever"],
                            "condition": condition,
                            "variant": variant,
                            "count": metrics["count"],
                            **{metric: metrics[metric] for metric in METRIC_KEYS},
                            **{
                                f"{metric}_percent": metrics[metric] * 100.0
                                for metric in METRIC_KEYS
                            },
                        }
                    )
    with (output_dir / "paired_deltas.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "model_family",
            "retriever",
            "condition",
            "metric",
            "mean",
            "ci95_low",
            "ci95_high",
            "mean_percentage_points",
            "ci95_low_percentage_points",
            "ci95_high_percentage_points",
            "resamples",
            "seed",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for comparison in comparisons:
            for condition, condition_result in comparison["conditions"].items():
                for metric, delta in condition_result["ft_minus_base"].items():
                    writer.writerow(
                        {
                            "model_family": comparison["model_family"],
                            "retriever": comparison["retriever"],
                            "condition": condition,
                            "metric": metric,
                            "mean": delta["mean"],
                            "ci95_low": delta["ci95_low"],
                            "ci95_high": delta["ci95_high"],
                            "mean_percentage_points": delta["mean"] * 100.0,
                            "ci95_low_percentage_points": delta["ci95_low"] * 100.0,
                            "ci95_high_percentage_points": delta["ci95_high"]
                            * 100.0,
                            "resamples": delta["resamples"],
                            "seed": delta["seed"],
                        }
                    )
    with (output_dir / "per_query_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "query_set",
            "query",
            "specificity",
            "condition",
            "model_family",
            "retriever",
            "metric",
            "base",
            "ft",
            "delta",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for comparison in comparisons:
            writer.writerows(comparison["per_query"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare strict LitSearch reruns.")
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260808)
    args = parser.parse_args(argv)
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    comparisons = [
        compare_spec(
            spec,
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        )
        for spec in matrix["comparisons"]
    ]
    write_outputs(args.output_dir, comparisons, expansion_decision(comparisons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
