from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BUDGETS = (10, 25, 50, 75, 100)
SEEDS = (42, 2026, 3407)
RETRIEVERS = ("bm25", "grit", "qwen3")
METRICS = ("R@20", "R@5", "nDCG@20")
RUN_PATTERN = re.compile(
    r"^Llama-3\.1-8B-Instruct_variants_3_target_50_pos_5_"
    r"parent_budget_(10|25|50|75|100)_seed_(42|2026|3407)$"
)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR.parent / "New folder" / "results_review"
DEFAULT_BASELINE_ROOT = (
    SCRIPT_DIR.parent
    / "reproducibility_package"
    / "results"
    / "litsearch"
    / "metrics"
    / "reranking"
    / "llama3"
    / "base"
    / "k_100"
)

# Reconstructed with the learning-curve notebook's parent-selection algorithm
# from DorisMAE/Query.json, Annotation.json, and aspect_id2aspect.json. The full
# budget reproduces the known total of 1,498 eligible aspect rows.
ASPECT_ROWS = {
    42: {10: 144, 25: 348, 50: 705, 75: 1108, 100: 1498},
    2026: {10: 147, 25: 406, 50: 782, 75: 1137, 100: 1498},
    3407: {10: 160, 25: 389, 50: 753, 75: 1124, 100: 1498},
}


def training_scale(budget: int, seed: int) -> dict[str, int]:
    aspect_rows = ASPECT_ROWS[seed][budget]
    augmented_lists = 3 * aspect_rows
    # Two epochs, batch size two, one device and no gradient accumulation.
    optimizer_steps = 2 * math.ceil(augmented_lists / 2)
    return {
        "parent_queries": budget,
        "aspect_rows": aspect_rows,
        "augmented_lists": augmented_lists,
        "optimizer_steps": optimizer_steps,
    }


def _recall_at_k(retrieved: list[int], relevant: list[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]).intersection(relevant)) / len(relevant)


def _ndcg_at_k(retrieved: list[int], relevant: list[int], k: int) -> float:
    relevant_set = set(relevant)
    dcg = sum(
        1.0 / math.log2(index + 2.0)
        for index, document_id in enumerate(retrieved[:k])
        if document_id in relevant_set
    )
    idcg = sum(
        1.0 / math.log2(index + 2.0)
        for index in range(min(len(relevant_set), k))
    )
    return dcg / idcg if idcg else 0.0


def _evaluate_file(path: Path, expected_queries: int) -> dict[str, float]:
    per_query = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            retrieved = [int(value) for value in row.get("retrieved", [])]
            pre_reranked = [int(value) for value in row.get("pre_reranked", [])]
            relevant = [int(value) for value in row.get("corpusids", [])]
            if (
                len(retrieved) != 100
                or len(set(retrieved)) != 100
                or set(retrieved) != set(pre_reranked)
            ):
                raise ValueError(
                    f"{path}:{line_number} is an incomplete or invalid reranking"
                )
            per_query.append(
                {
                    "R@20": 100.0 * _recall_at_k(retrieved, relevant, 20),
                    "R@5": 100.0 * _recall_at_k(retrieved, relevant, 5),
                    "nDCG@20": 100.0 * _ndcg_at_k(retrieved, relevant, 20),
                }
            )

    if len(per_query) != expected_queries:
        raise ValueError(
            f"{path} has {len(per_query)} queries; expected {expected_queries}"
        )
    return {
        metric: statistics.mean(row[metric] for row in per_query)
        for metric in METRICS
    }


def aggregate_learning_curve(
    results_root: str | Path,
    *,
    expected_queries: int = 597,
) -> dict[int, dict[str, float | int | dict[int, dict[str, float]]]]:
    results_root = Path(results_root)
    run_metrics: dict[tuple[int, int], dict[str, dict[str, float]]] = {}

    for run_directory in results_root.iterdir():
        if not run_directory.is_dir():
            continue
        match = RUN_PATTERN.fullmatch(run_directory.name)
        if not match:
            continue
        budget, seed = (int(value) for value in match.groups())
        retriever_metrics = {}
        for retriever in RETRIEVERS:
            matches = list(
                run_directory.glob(
                    "*/k_100/"
                    f"LitSearch.title_abstract.{retriever}.reranked.jsonl"
                )
            )
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one k=100 {retriever} result under "
                    f"{run_directory}, found {len(matches)}"
                )
            retriever_metrics[retriever] = _evaluate_file(
                matches[0], expected_queries
            )
        run_metrics[(budget, seed)] = retriever_metrics

    expected_runs = {(budget, seed) for budget in BUDGETS for seed in SEEDS}
    if set(run_metrics) != expected_runs:
        missing = sorted(expected_runs.difference(run_metrics))
        extra = sorted(set(run_metrics).difference(expected_runs))
        raise ValueError(f"Incomplete learning-curve grid; missing={missing}, extra={extra}")

    summaries = {}
    for budget in BUDGETS:
        seed_values = {}
        for seed in SEEDS:
            seed_values[seed] = {
                metric: statistics.mean(
                    run_metrics[(budget, seed)][retriever][metric]
                    for retriever in RETRIEVERS
                )
                for metric in METRICS
            }
        summary: dict[str, float | int | dict[int, dict[str, float]]] = {
            "seed_count": len(seed_values),
            "seed_values": seed_values,
        }
        for metric in METRICS:
            values = [seed_values[seed][metric] for seed in SEEDS]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_sd"] = statistics.stdev(values)
        summaries[budget] = summary

    return summaries


def load_zero_shot_baseline(
    metrics_root: str | Path = DEFAULT_BASELINE_ROOT,
) -> dict[str, float]:
    metrics_root = Path(metrics_root)
    values = {metric: [] for metric in METRICS}
    json_keys = {
        "R@20": "R@20_percent",
        "R@5": "R@5_percent",
        "nDCG@20": "NDCG@20_percent",
    }
    for retriever in RETRIEVERS:
        path = (
            metrics_root
            / f"LitSearch.title_abstract.{retriever}.reranked.metrics.json"
        )
        with path.open("r", encoding="utf-8") as handle:
            overall = json.load(handle)["overall_avg"]
        for metric in METRICS:
            values[metric].append(float(overall[json_keys[metric]]))
    return {metric: statistics.mean(values[metric]) for metric in METRICS}


def _save_figure(figure, output_base: str | Path) -> list[Path]:
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_base.with_suffix(".pdf"), output_base.with_suffix(".png")]
    figure.savefig(outputs[0], bbox_inches="tight")
    figure.savefig(outputs[1], dpi=300, bbox_inches="tight")
    plt.close(figure)
    return outputs


def plot_learning_curve(
    summaries: dict[int, dict],
    output_base: str | Path,
    *,
    baseline: dict[str, float],
) -> list[Path]:
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.35), sharey=True)
    styles = {
        "R@20": {"color": "#4C78A8", "marker": "o"},
        "R@5": {"color": "#F58518", "marker": "s"},
        "nDCG@20": {"color": "#54A24B", "marker": "^"},
    }
    panels = (
        ("parent_queries", "Independent parent queries"),
        ("aspect_rows", "Aspect-level examples"),
        ("optimizer_steps", "Optimizer steps"),
    )

    for axis, (scale_key, x_label) in zip(axes, panels):
        for metric in METRICS:
            x_means = []
            y_means = []
            y_errors = []
            for budget in BUDGETS:
                x_values = [
                    training_scale(budget, seed)[scale_key] for seed in SEEDS
                ]
                y_values = [
                    summaries[budget]["seed_values"][seed][metric]
                    for seed in SEEDS
                ]
                x_means.append(statistics.mean(x_values))
                y_means.append(summaries[budget][f"{metric}_mean"])
                y_errors.append(summaries[budget][f"{metric}_sd"])
                if scale_key != "parent_queries":
                    axis.scatter(
                        x_values,
                        y_values,
                        s=8,
                        color=styles[metric]["color"],
                        alpha=0.18,
                        linewidths=0,
                    )

            axis.errorbar(
                x_means,
                y_means,
                yerr=y_errors,
                label=metric,
                color=styles[metric]["color"],
                marker=styles[metric]["marker"],
                markersize=3.3,
                linewidth=1.1,
                capsize=1.8,
            )
            axis.scatter(
                [0],
                [baseline[metric]],
                marker=styles[metric]["marker"],
                s=18,
                facecolors="white",
                edgecolors=styles[metric]["color"],
                linewidths=0.9,
                zorder=3,
            )

        axis.set_xlabel(x_label)
        axis.grid(color="#d8d8d8", linewidth=0.4)
        axis.set_axisbelow(True)
        axis.set_ylim(44, 79)
        axis.margins(x=0.04)

    axes[0].set_ylabel("LitSearch performance (%)")
    axes[0].annotate(
        "zero-shot",
        xy=(0, baseline["nDCG@20"]),
        xytext=(4, -2),
        textcoords="offset points",
        fontsize=6.3,
        color="#555555",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93), pad=0.45, w_pad=0.65)
    return _save_figure(figure, output_base)


def generate_all(
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    output_base: str | Path = SCRIPT_DIR / "fig_parent_query_learning_curve",
) -> tuple[dict[int, dict], list[Path]]:
    summaries = aggregate_learning_curve(results_root)
    outputs = plot_learning_curve(
        summaries,
        output_base,
        baseline=load_zero_shot_baseline(),
    )
    return summaries, outputs


if __name__ == "__main__":
    curve, paths = generate_all()
    print("budget,R@20,R@5,nDCG@20")
    for budget in BUDGETS:
        values = [
            f"{curve[budget][f'{metric}_mean']:.2f}+/-"
            f"{curve[budget][f'{metric}_sd']:.2f}"
            for metric in METRICS
        ]
        print(f"{budget}," + ",".join(values))
    for path in paths:
        print(path)
