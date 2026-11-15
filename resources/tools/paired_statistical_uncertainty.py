from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.litsearch_document_disjoint import ndcg_at_k, recall_at_k


LITSEARCH_METRICS = ("R@5", "R@20", "NDCG@20")
CSFCUBE_METRICS = ("R@20", "NDCG%20")


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def _litsearch_identity(row: dict) -> tuple[str, str, int]:
    return (
        str(row["query_set"]),
        str(row["query"]),
        int(row["specificity"]),
    )


def _index_litsearch(rows: list[dict], label: str) -> dict[tuple[str, str, int], dict]:
    indexed = {}
    for row in rows:
        identity = _litsearch_identity(row)
        if identity in indexed:
            raise ValueError(f"duplicate query identity in {label}: {identity!r}")
        indexed[identity] = row
    return indexed


def _litsearch_metric_values(row: dict, ranking_field: str) -> dict[str, float]:
    if ranking_field not in row:
        raise ValueError(f"missing ranking field {ranking_field!r}")
    retrieved = [int(value) for value in row[ranking_field]]
    relevant = [int(value) for value in row.get("corpusids", [])]
    return {
        "R@5": recall_at_k(retrieved, relevant, 5),
        "R@20": recall_at_k(retrieved, relevant, 20),
        "NDCG@20": ndcg_at_k(retrieved, relevant, 20),
    }


def load_litsearch_pair(
    reference_path: Path,
    candidate_path: Path,
    *,
    reference_field: str = "retrieved",
    candidate_field: str = "retrieved",
) -> dict[str, list | dict]:
    """Align full LitSearch runs and return paired per-query metric values."""
    reference_index = _index_litsearch(_load_jsonl(reference_path), "reference")
    candidate_index = _index_litsearch(_load_jsonl(candidate_path), "candidate")
    if set(reference_index) != set(candidate_index):
        missing_reference = sorted(set(candidate_index) - set(reference_index))
        missing_candidate = sorted(set(reference_index) - set(candidate_index))
        raise ValueError(
            "query identities differ; "
            f"missing_reference={missing_reference[:3]!r}, "
            f"missing_candidate={missing_candidate[:3]!r}"
        )

    identities = sorted(reference_index)
    result: dict[str, list | dict] = {
        "query_ids": ["|".join(map(str, identity)) for identity in identities]
    }
    metric_pairs = {
        metric: {"reference": [], "candidate": []}
        for metric in LITSEARCH_METRICS
    }
    for identity in identities:
        reference_row = reference_index[identity]
        candidate_row = candidate_index[identity]
        if [int(value) for value in reference_row.get("corpusids", [])] != [
            int(value) for value in candidate_row.get("corpusids", [])
        ]:
            raise ValueError(f"gold labels differ for query {identity!r}")
        if (
            "pre_reranked" in reference_row
            and "pre_reranked" in candidate_row
            and [int(value) for value in reference_row["pre_reranked"]]
            != [int(value) for value in candidate_row["pre_reranked"]]
        ):
            raise ValueError(f"candidate pool order differs for query {identity!r}")

        reference_metrics = _litsearch_metric_values(reference_row, reference_field)
        candidate_metrics = _litsearch_metric_values(candidate_row, candidate_field)
        for metric in LITSEARCH_METRICS:
            metric_pairs[metric]["reference"].append(reference_metrics[metric])
            metric_pairs[metric]["candidate"].append(candidate_metrics[metric])

    result.update(metric_pairs)
    return result


def load_litsearch_aggregate_pair(
    reference_paths: list[Path],
    candidate_paths: list[Path],
) -> dict[str, list | dict]:
    """Average per-query LitSearch metrics over matched retriever runs."""
    if len(reference_paths) != len(candidate_paths) or not reference_paths:
        raise ValueError("matched non-empty reference/candidate path lists are required")
    pairs = [
        load_litsearch_pair(reference_path, candidate_path)
        for reference_path, candidate_path in zip(reference_paths, candidate_paths)
    ]
    query_ids = pairs[0]["query_ids"]
    if any(pair["query_ids"] != query_ids for pair in pairs[1:]):
        raise ValueError("query order differs across LitSearch retrievers")

    result: dict[str, list | dict] = {"query_ids": query_ids}
    for metric in LITSEARCH_METRICS:
        reference_matrix = np.asarray(
            [pair[metric]["reference"] for pair in pairs], dtype=float
        )
        candidate_matrix = np.asarray(
            [pair[metric]["candidate"] for pair in pairs], dtype=float
        )
        result[metric] = {
            "reference": reference_matrix.mean(axis=0).tolist(),
            "candidate": candidate_matrix.mean(axis=0).tolist(),
        }
    return result


def _load_csfcube_index(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {}
    for row in rows:
        case_id = str(row.get("paper_id", ""))
        if not case_id:
            raise ValueError(f"missing paper_id in {path}")
        if case_id in indexed:
            raise ValueError(f"duplicate CSFCube case in {path}: {case_id}")
        indexed[case_id] = row
    return indexed


def _csfcube_parent_id(case_id: str) -> str:
    parent, separator, facet = case_id.rpartition("_")
    if not separator or not parent or facet not in {"background", "method", "result"}:
        raise ValueError(f"invalid CSFCube query-facet ID: {case_id!r}")
    return parent


def load_csfcube_pair(
    reference_path: Path,
    candidate_path: Path,
) -> dict[str, list | dict]:
    """Align CSFCube cases and derive unique-paper cluster identifiers."""
    reference_index = _load_csfcube_index(reference_path)
    candidate_index = _load_csfcube_index(candidate_path)
    if set(reference_index) != set(candidate_index):
        raise ValueError("CSFCube case identities differ between systems")

    case_ids = sorted(reference_index)
    result: dict[str, list | dict] = {
        "case_ids": case_ids,
        "cluster_ids": [_csfcube_parent_id(case_id) for case_id in case_ids],
    }
    metric_columns = {"R@20": "recall", "NDCG%20": "ndcg%20"}
    for metric, column in metric_columns.items():
        result[metric] = {
            "reference": [float(reference_index[case_id][column]) for case_id in case_ids],
            "candidate": [float(candidate_index[case_id][column]) for case_id in case_ids],
        }
    return result


def _cluster_sums_and_counts(
    differences: list[float] | np.ndarray,
    cluster_ids: list[str] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(differences) != len(cluster_ids) or not len(differences):
        raise ValueError("differences and cluster_ids must be equal and non-empty")

    grouped: OrderedDict[str, list[float]] = OrderedDict()
    for cluster_id, difference in zip(cluster_ids, differences):
        grouped.setdefault(str(cluster_id), []).append(float(difference))

    sums = np.asarray([sum(values) for values in grouped.values()], dtype=float)
    counts = np.asarray([len(values) for values in grouped.values()], dtype=int)
    return sums, counts


def cluster_bootstrap_means(
    *,
    differences: list[float] | np.ndarray,
    cluster_ids: list[str] | np.ndarray,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap clusters and retain every observation inside sampled clusters."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    cluster_sums, cluster_counts = _cluster_sums_and_counts(
        differences, cluster_ids
    )
    cluster_count = len(cluster_sums)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, cluster_count, size=(replicates, cluster_count))
    numerators = cluster_sums[sampled].sum(axis=1)
    denominators = cluster_counts[sampled].sum(axis=1)
    return numerators / denominators


def paired_inference(
    *,
    reference: list[float] | np.ndarray,
    candidate: list[float] | np.ndarray,
    cluster_ids: list[str] | np.ndarray,
    bootstrap_replicates: int,
    permutation_replicates: int,
    bootstrap_seed: int,
    permutation_seed: int,
) -> dict[str, float | int]:
    """Estimate candidate-minus-reference uncertainty with paired clusters."""
    reference_values = np.asarray(reference, dtype=float)
    candidate_values = np.asarray(candidate, dtype=float)
    if reference_values.ndim != 1 or candidate_values.ndim != 1:
        raise ValueError("reference and candidate must be one-dimensional")
    if len(reference_values) != len(candidate_values) or not len(reference_values):
        raise ValueError("reference and candidate must be equal and non-empty")
    if len(cluster_ids) != len(reference_values):
        raise ValueError("one cluster ID is required per paired case")
    if not np.isfinite(reference_values).all() or not np.isfinite(
        candidate_values
    ).all():
        raise ValueError("scores must be finite")
    if permutation_replicates <= 0:
        raise ValueError("permutation_replicates must be positive")

    differences = candidate_values - reference_values
    bootstrap_means = cluster_bootstrap_means(
        differences=differences,
        cluster_ids=cluster_ids,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])

    cluster_sums, _ = _cluster_sums_and_counts(differences, cluster_ids)
    observed = float(differences.mean())
    rng = np.random.default_rng(permutation_seed)
    extreme = 0
    completed = 0
    batch_size = 2_000
    while completed < permutation_replicates:
        current_batch = min(batch_size, permutation_replicates - completed)
        signs = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(current_batch, len(cluster_sums)),
        )
        permuted = (signs * cluster_sums).sum(axis=1) / len(differences)
        extreme += int(np.count_nonzero(np.abs(permuted) >= abs(observed)))
        completed += current_batch

    return {
        "n_cases": len(differences),
        "n_clusters": len(cluster_sums),
        "reference_mean": float(reference_values.mean()),
        "candidate_mean": float(candidate_values.mean()),
        "delta": observed,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_raw": (extreme + 1) / (permutation_replicates + 1),
        "bootstrap_replicates": bootstrap_replicates,
        "permutation_replicates": permutation_replicates,
        "bootstrap_seed": bootstrap_seed,
        "permutation_seed": permutation_seed,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm--Bonferroni adjusted p-values in input order."""
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted_sorted: list[float] = []
    running_maximum = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running_maximum = max(running_maximum, candidate)
        adjusted_sorted.append(running_maximum)

    adjusted = [0.0] * count
    for index, value in zip(order, adjusted_sorted):
        adjusted[index] = value
    return adjusted


def apply_holm_by_family(rows: list[dict]) -> list[dict]:
    """Apply Holm correction independently to each family-and-metric group."""
    corrected = [dict(row) for row in rows]
    grouped_indices: OrderedDict[tuple[str, str], list[int]] = OrderedDict()
    for index, row in enumerate(corrected):
        key = (str(row["family"]), str(row["metric"]))
        grouped_indices.setdefault(key, []).append(index)

    for indices in grouped_indices.values():
        adjusted = holm_adjust([float(corrected[index]["p_raw"]) for index in indices])
        for index, adjusted_p in zip(indices, adjusted):
            corrected[index]["p_holm"] = adjusted_p
    return corrected


def _derived_seed(master_seed: int, material: str) -> int:
    digest = hashlib.sha256(f"{master_seed}|{material}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def run_analysis(
    config: dict,
    workspace_root: Path,
) -> tuple[list[dict], list[dict]]:
    """Run configured paired analyses and return result and missing-input rows."""
    settings = config["settings"]
    master_seed = int(settings["master_seed"])
    bootstrap_replicates = int(settings["bootstrap_replicates"])
    permutation_replicates = int(settings["permutation_replicates"])
    result_rows: list[dict] = []
    missing_inputs: list[dict] = []

    for contrast in config["contrasts"]:
        dataset = str(contrast["dataset"])
        family = str(contrast["family"])
        comparison = str(contrast["comparison"])
        if dataset == "litsearch_aggregate":
            reference_paths = [
                workspace_root / path for path in contrast["reference"]
            ]
            candidate_paths = [
                workspace_root / path for path in contrast["candidate"]
            ]
            input_paths = [*reference_paths, *candidate_paths]
        else:
            reference_path = workspace_root / contrast["reference"]
            candidate_path = workspace_root / contrast["candidate"]
            input_paths = [reference_path, candidate_path]
        missing_paths = [
            str(path)
            for path in input_paths
            if not path.is_file()
        ]
        if missing_paths:
            missing_inputs.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "comparison": comparison,
                    "missing_paths": missing_paths,
                }
            )
            continue

        if dataset == "litsearch":
            paired = load_litsearch_pair(
                reference_path,
                candidate_path,
                reference_field=str(contrast.get("reference_field", "retrieved")),
                candidate_field=str(contrast.get("candidate_field", "retrieved")),
            )
            case_ids = list(paired["query_ids"])
            cluster_ids = case_ids
            metrics = LITSEARCH_METRICS
        elif dataset == "litsearch_aggregate":
            paired = load_litsearch_aggregate_pair(reference_paths, candidate_paths)
            case_ids = list(paired["query_ids"])
            cluster_ids = case_ids
            metrics = LITSEARCH_METRICS
        elif dataset == "csfcube":
            paired = load_csfcube_pair(reference_path, candidate_path)
            case_ids = list(paired["case_ids"])
            cluster_ids = list(paired["cluster_ids"])
            metrics = CSFCUBE_METRICS
        else:
            raise ValueError(f"unsupported dataset: {dataset!r}")

        expected_cases = contrast.get("expected_cases")
        if expected_cases is not None and len(case_ids) != int(expected_cases):
            raise ValueError(
                f"{comparison}: expected {expected_cases} paired cases, got {len(case_ids)}"
            )
        expected_clusters = contrast.get("expected_clusters")
        actual_clusters = len(set(cluster_ids))
        if expected_clusters is not None and actual_clusters != int(expected_clusters):
            raise ValueError(
                f"{comparison}: expected {expected_clusters} clusters, got {actual_clusters}"
            )

        for metric in metrics:
            seed_material = f"{dataset}|{family}|{comparison}|{metric}"
            bootstrap_seed = _derived_seed(master_seed, f"bootstrap|{seed_material}")
            permutation_seed = _derived_seed(master_seed, f"permutation|{seed_material}")
            inference = paired_inference(
                reference=paired[metric]["reference"],
                candidate=paired[metric]["candidate"],
                cluster_ids=cluster_ids,
                bootstrap_replicates=bootstrap_replicates,
                permutation_replicates=permutation_replicates,
                bootstrap_seed=bootstrap_seed,
                permutation_seed=permutation_seed,
            )
            result_rows.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "comparison": comparison,
                    "model": contrast.get("model", ""),
                    "retriever": contrast.get("retriever", ""),
                    "reference_label": contrast.get("reference_label", "reference"),
                    "candidate_label": contrast.get("candidate_label", "candidate"),
                    "metric": metric,
                    **inference,
                    "reference_mean_percentage": inference["reference_mean"] * 100.0,
                    "candidate_mean_percentage": inference["candidate_mean"] * 100.0,
                    "delta_percentage_points": inference["delta"] * 100.0,
                    "ci95_low_percentage_points": inference["ci95_low"] * 100.0,
                    "ci95_high_percentage_points": inference["ci95_high"] * 100.0,
                }
            )

    expected_by_family = {
        str(key): int(value)
        for key, value in config["expected_contrasts_per_family"].items()
    }
    grouped: OrderedDict[tuple[str, str], list[int]] = OrderedDict()
    for index, row in enumerate(result_rows):
        grouped.setdefault((row["family"], row["metric"]), []).append(index)
    for (family, _metric), indices in grouped.items():
        actual_comparisons = len({result_rows[index]["comparison"] for index in indices})
        expected_comparisons = expected_by_family[family]
        complete = actual_comparisons == expected_comparisons
        adjusted = (
            holm_adjust([float(result_rows[index]["p_raw"]) for index in indices])
            if complete
            else [None] * len(indices)
        )
        for index, adjusted_p in zip(indices, adjusted):
            result_rows[index]["family_complete"] = complete
            result_rows[index]["p_holm"] = adjusted_p

    return result_rows, missing_inputs


def write_outputs(
    output_dir: Path,
    *,
    rows: list[dict],
    missing_inputs: list[dict],
    config: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "paired_uncertainty.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "paired_uncertainty.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "missing_inputs.json").write_text(
        json.dumps(missing_inputs, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap, clustered permutation tests, and Holm correction."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write partial results instead of returning an error for missing inputs.",
    )
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows, missing_inputs = run_analysis(config, args.workspace_root)
    write_outputs(
        args.output_dir,
        rows=rows,
        missing_inputs=missing_inputs,
        config=config,
    )
    print(f"Wrote {len(rows)} metric comparisons to {args.output_dir}")
    if missing_inputs:
        print(f"Missing inputs for {len(missing_inputs)} configured contrasts")
        return 0 if args.allow_missing else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
