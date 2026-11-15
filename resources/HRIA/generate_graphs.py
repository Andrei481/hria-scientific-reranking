from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt


METRIC_LABELS = ["R@20", "R@5", "nDCG@20"]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LITSEARCH_RESULTS = SCRIPT_DIR / "LitSearch" / "results"

HEATMAP_MODELS = [
    ("llama3", "Llama3.1-8B"),
    ("gemma3_12b", "Gemma3-12B"),
    ("qwen2.5", "Qwen2.5-14B"),
    ("gemma3_27b", "Gemma3-27B"),
]

HEATMAP_RETRIEVERS = [
    ("bm25", "BM25"),
    ("grit", "GRIT"),
    ("qwen3", "Qwen3"),
]

METRIC_KEYS = ["R@20", "R@5", "NDCG@20"]


def _load_metrics(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _overall_metric_percent(path, metric):
    return _load_metrics(path)["overall_avg"][metric] * 100.0


def _group_metric_percent(path, group, metric):
    return _load_metrics(path)[group][metric] * 100.0


def litsearch_gain_matrix(results_root=DEFAULT_LITSEARCH_RESULTS):
    reranking_root = Path(results_root) / "reranking"
    rows = []

    for model_dir, model_label in HEATMAP_MODELS:
        for retriever_key, retriever_label in HEATMAP_RETRIEVERS:
            base_path = (
                reranking_root
                / model_dir
                / "base"
                / "k_100"
                / f"LitSearch.title_abstract.{retriever_key}.reranked.metrics.json"
            )
            ft_path = (
                reranking_root
                / model_dir
                / "FT"
                / "k_100"
                / f"LitSearch.title_abstract.{retriever_key}.reranked.metrics.json"
            )

            gains = [
                _overall_metric_percent(ft_path, metric)
                - _overall_metric_percent(base_path, metric)
                for metric in METRIC_KEYS
            ]
            rows.append((f"{model_label}/{retriever_label}", gains))

    labels, data = zip(*rows)
    return list(labels), np.array(data)


def gpt4o_comparison_data(results_root=DEFAULT_LITSEARCH_RESULTS):
    results_root = Path(results_root)
    qwen3_retrieval = results_root / "retrieval" / "LitSearch.title_abstract.qwen3.metrics.json"
    gemma27_ft_qwen3 = (
        results_root
        / "reranking"
        / "gemma3_27b"
        / "FT"
        / "k_100"
        / "LitSearch.title_abstract.qwen3.reranked.metrics.json"
    )

    return {
        # External reference values reported by the LitSearch paper.
        "GPT-4o + GritLM": [75.3, 79.2, 85.9],
        "Qwen3 retrieval": [
            _group_metric_percent(qwen3_retrieval, "broad_avg", "R@20"),
            _group_metric_percent(qwen3_retrieval, "specific_avg", "R@5"),
            _group_metric_percent(qwen3_retrieval, "specific_avg", "R@20"),
        ],
        "Gemma27 FT + Qwen3": [
            _group_metric_percent(gemma27_ft_qwen3, "broad_avg", "R@20"),
            _group_metric_percent(gemma27_ft_qwen3, "specific_avg", "R@5"),
            _group_metric_percent(gemma27_ft_qwen3, "specific_avg", "R@20"),
        ],
    }


def _save_figure(fig, base_path):
    base_path = Path(base_path)
    pdf_path = base_path.with_suffix(".pdf")
    png_path = base_path.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [pdf_path, png_path]


def plot_gpt4o_comparison(base_path, results_root=DEFAULT_LITSEARCH_RESULTS):
    metrics = ["Broad R@20", "Spec. R@5", "Spec. R@20"]
    configs = gpt4o_comparison_data(results_root)
    x = np.arange(len(metrics))
    width = 0.24

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
        }
    )
    fig, ax = plt.subplots(figsize=(6.8, 2.35))

    styles = [
        {"color": "#4C78A8", "hatch": ""},
      {"color": "#F58518", "hatch": ""},
      {"color": "#54A24B", "hatch": ""},
    ]
    for idx, ((name, values), style) in enumerate(zip(configs.items(), styles)):
        offset = (idx - 1) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=name,
            color=style["color"],
            edgecolor="black",
            linewidth=0.6,
            hatch=style["hatch"],
        )
        for xpos, value in zip(x + offset, values):
            ax.text(xpos, value + 0.45, f"{value:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Recall (%)")
    ax.set_xticks(x, metrics)
    ax.set_ylim(65, 89)
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
    fig.tight_layout(pad=0.35)
    return _save_figure(fig, base_path)


def plot_litsearch_gain_heatmap(base_path, results_root=DEFAULT_LITSEARCH_RESULTS):
    row_labels, data = litsearch_gain_matrix(results_root)
    vmax = 17

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 7,
        }
    )
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(METRIC_LABELS)), METRIC_LABELS)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)

    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            text_color = "white" if abs(value) >= 10 else "black"
            ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=7, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("FT - base (percentage points)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout(pad=0.35)
    return _save_figure(fig, base_path)


def generate_all(output_dir=Path("."), results_root=DEFAULT_LITSEARCH_RESULTS):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    outputs.extend(plot_gpt4o_comparison(output_dir / "fig_gpt4o_comparison", results_root))
    outputs.extend(plot_litsearch_gain_heatmap(output_dir / "fig_litsearch_ft_gain_heatmap", results_root))
    return outputs


if __name__ == "__main__":
    for output in generate_all():
        print(output)
