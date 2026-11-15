"""Matched LitSearch quality/compliance summaries and paired query uncertainty."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np

from experiment import litsearch_metrics, parse_ranking, read_jsonl, sha256_file, stable_seed, write_json

RETRIEVERS = ("bm25", "grit", "qwen3")
QUALITY = ("R@5", "R@20", "nDCG@20")
COMPLIANCE = ("exact_permutation", "fallback", "missing_id_rate", "duplicate_id_rate", "invalid_id_rate", "mean_missing_ids")


def align_rows(reference, candidate, expected=597):
    left = {r["qid"]: r for r in reference}
    right = {r["qid"]: r for r in candidate}
    if len(left) != len(reference) or len(right) != len(candidate):
        raise ValueError("Duplicate query IDs")
    if len(left) != expected or set(left) != set(right):
        raise ValueError("Missing, extra, or unaligned evaluation queries")
    aligned = []
    for qid in sorted(left):
        a, b = left[qid], right[qid]
        if a["pre_reranked"] != b["pre_reranked"] or a["corpusids"] != b["corpusids"]:
            raise ValueError("Different candidate order or gold labels in paired comparison")
        for row in (a, b):
            if len(row["retrieved"]) != len(row["pre_reranked"]) or set(row["retrieved"]) != set(row["pre_reranked"]):
                raise ValueError("Saved ranking is not a permutation of its candidate pool")
        aligned.append(b)
    return aligned


def paired_stats(differences, seed, replicates=10000):
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Expected a nonempty finite vector of query differences")
    rng = np.random.default_rng(seed)
    means, extreme = [], 0
    observed = float(values.mean())
    for start in range(0, replicates, 250):
        size = min(250, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means.extend(values[indices].mean(axis=1).tolist())
        signs = rng.choice([-1, 1], size=(size, len(values)))
        extreme += int(np.sum(np.abs((signs * values).mean(axis=1)) >= abs(observed) - 1e-12))
    low, high = np.quantile(means, [0.025, 0.975])
    return {"delta": observed, "ci_low": float(low), "ci_high": float(high), "p": (extreme + 1) / (replicates + 1)}


def holm(values):
    result = [0.0] * len(values)
    previous = 0.0
    for rank, index in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        previous = max(previous, min(1.0, (len(values) - rank) * values[index]))
        result[index] = previous
    return result


def measured_values(row):
    order, audit = parse_ranking(row["raw_response"], len(row["pre_reranked"]))
    if [row["pre_reranked"][i] for i in order] != row["retrieved"]:
        raise ValueError("Saved ranking does not match repair of saved raw output")
    return {
        **litsearch_metrics(row["retrieved"], row["corpusids"]),
        "exact_permutation": float(audit["exact_permutation"]),
        "fallback": float(audit["fallback"]),
        "missing_id_rate": float(audit["missing_ids"] > 0),
        "duplicate_id_rate": float(audit["duplicate_ids"] > 0),
        "invalid_id_rate": float(audit["invalid_ids"] > 0),
        "mean_missing_ids": audit["missing_ids"],
    }


def load_run(root, policy, seed):
    directory = root / f"{policy}_seed{seed}"
    training_path = directory / "training_manifest.json"
    training = json.loads(training_path.read_text())
    if training["status"] != "complete" or training["config"]["smoke_steps"]:
        raise ValueError("Incomplete or smoke-only training")
    if training["config"]["policy"] != policy or training["config"]["seed"] != seed:
        raise ValueError("Directory name and recorded training condition disagree")
    evaluation = directory / "evaluation"
    protocol_path = evaluation / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    complete = json.loads((evaluation / "complete.json").read_text())
    if protocol["limit"] or protocol["training_manifest_sha256"] != sha256_file(training_path):
        raise ValueError("Evaluation is a smoke run or refers to a different training manifest")
    if complete["protocol_sha256"] != sha256_file(protocol_path):
        raise ValueError("Evaluation protocol changed after completion")
    values = {}
    for retriever in RETRIEVERS:
        path = evaluation / f"{retriever}.jsonl"
        if sha256_file(path) != complete["results"][retriever]:
            raise ValueError("Evaluation output changed after completion")
        values[retriever] = read_jsonl(path)
    return training, protocol, values


def comparable_settings(training, protocol):
    config = {k: v for k, v in training["config"].items() if k not in ("policy", "seed")}
    inference = {k: v for k, v in protocol.items() if k not in ("training_manifest_sha256", "served_model", "url")}
    return {"config": config, "inference": inference, "packages": training["packages"],
            "chat_template_sha256": training["preflight"]["chat_template_sha256"]}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path(__file__).parent / "runs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026, 3407])
    parser.add_argument("--comparisons", nargs="+", choices=("epoch_random", "fixed_random"), default=["epoch_random"])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "analysis")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.comparisons)) != len(args.comparisons):
        raise ValueError("Seeds/comparisons must be unique")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an empty analysis output directory")
    loaded, summaries, common_settings = {}, [], None
    for policy in ["deterministic"] + args.comparisons:
        for seed in args.seeds:
            training, protocol, outputs = load_run(args.runs, policy, seed)
            settings = comparable_settings(training, protocol)
            if common_settings is None:
                common_settings = settings
            elif settings != common_settings:
                raise ValueError(f"Unmatched data/model/environment/inference settings: {policy}, {seed}")
            arrays, ids = {}, None
            for retriever in RETRIEVERS:
                ordered = align_rows(outputs[retriever], outputs[retriever])
                query_ids = [r["qid"] for r in ordered]
                if ids is None:
                    ids = query_ids
                elif ids != query_ids:
                    raise ValueError("Retrievers have different query sets")
                arrays[retriever] = np.array([[measured_values(r)[m] for m in QUALITY + COMPLIANCE] for r in ordered])
            arrays["retriever_average"] = np.mean([arrays[r] for r in RETRIEVERS], axis=0)
            loaded[(policy, seed)] = (training, outputs, arrays)
            for retriever, array in arrays.items():
                for column, metric in enumerate(QUALITY + COMPLIANCE):
                    scale = 1 if metric == "mean_missing_ids" else 100
                    summaries.append({"policy": policy, "seed": seed, "retriever": retriever, "metric": metric,
                                      "value": float(array[:, column].mean()) * scale,
                                      "unit": "candidate count" if scale == 1 else "percent"})
    comparisons = []
    for policy in args.comparisons:
        for seed in args.seeds:
            reference, candidate = loaded[("deterministic", seed)], loaded[(policy, seed)]
            if reference[0]["preflight"]["schedule_sha256"] != candidate[0]["preflight"]["schedule_sha256"]:
                raise ValueError("Matched conditions saw different data schedules")
            for retriever in RETRIEVERS:
                align_rows(reference[1][retriever], candidate[1][retriever])
        for retriever in (*RETRIEVERS, "retriever_average"):
            for column, metric in enumerate(QUALITY):
                # Average each query over seeds FIRST; do not pretend 3*597 independent queries.
                differences = np.mean([
                    loaded[(policy, s)][2][retriever][:, column] - loaded[("deterministic", s)][2][retriever][:, column]
                    for s in args.seeds
                ], axis=0)
                stats = paired_stats(differences, stable_seed(20260815, policy, retriever, metric))
                comparisons.append({"candidate": policy, "reference": "deterministic", "retriever": retriever,
                                    "metric": metric, "seeds": len(args.seeds), "queries": 597,
                                    "delta_pp": 100 * stats["delta"], "ci_low_pp": 100 * stats["ci_low"],
                                    "ci_high_pp": 100 * stats["ci_high"], "p": stats["p"], "p_holm": 0.0})
    for metric in QUALITY:
        family = [r for r in comparisons if r["metric"] == metric]
        for row, adjusted in zip(family, holm([r["p"] for r in family])):
            row["p_holm"] = adjusted
    aggregate = []
    for policy in ["deterministic"] + args.comparisons:
        for retriever in (*RETRIEVERS, "retriever_average"):
            for metric in QUALITY + COMPLIANCE:
                values = [r["value"] for r in summaries if r["policy"] == policy and r["retriever"] == retriever and r["metric"] == metric]
                aggregate.append({"policy": policy, "retriever": retriever, "metric": metric,
                                  "seed_count": len(values), "mean": statistics.mean(values),
                                  "sample_sd": statistics.stdev(values) if len(values) > 1 else ""})
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "per_seed.csv", summaries)
    write_csv(args.output / "mean_sd.csv", aggregate)
    write_csv(args.output / "paired_uncertainty.csv", comparisons)
    write_json(args.output / "analysis_manifest.json", {
        "seeds": args.seeds, "comparisons": args.comparisons, "replicates": 10000,
        "settings": common_settings,
        "interpretation": "Query bootstrap and sign-flips condition on these trained checkpoints; seed SD is separate. No equivalence inference.",
        "multiplicity": "Holm within each metric across all selected policies and four retriever summaries",
        "pilot": len(args.seeds) < 3,
    })
    print(f"Saved analysis to {args.output}. {'PILOT: fewer than three seeds.' if len(args.seeds) < 3 else ''}")
    for row in comparisons:
        if row["retriever"] == "retriever_average":
            print(f"{row['candidate']} {row['metric']}: {row['delta_pp']:+.3f} pp, CI [{row['ci_low_pp']:.3f}, {row['ci_high_pp']:.3f}], Holm p={row['p_holm']:.4f}")


if __name__ == "__main__":
    main()
