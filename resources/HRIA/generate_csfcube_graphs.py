from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "CSFCube" / "results"

MODELS = [
    ("llama3", "Llama3.1-8B"),
    ("gemma3_12b", "Gemma3-12B"),
    ("qwen2.5", "Qwen2.5-14B"),
    ("gemma3_27b", "Gemma3-27B"),
]

STATES = ["base", "FT"]
FACETS = ["background", "method", "result", "all"]
METRIC_NAMES = [
    "r_precision",
    "precision@20",
    "recall@20",
    "ndcg",
    "ndcg@20",
    "ndcg%20",
]


def parse_results_file(path):
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    current_facet = None
    current_split = None

    for i, line in enumerate(lines):
        if line.startswith("python ") and "--facet " in line:
            match = re.search(r"--facet\s+(\w+)", line)
            current_facet = match.group(1) if match else None
        elif line.startswith("EVAL SPLIT:"):
            current_split = line.split(":", 1)[1].strip()
        elif line.startswith("R-Precision;"):
            values = [float(x.strip()) for x in lines[i + 1].split(",")]
            record = {
                "facet": current_facet,
                "split": current_split,
            }
            record.update(dict(zip(METRIC_NAMES, values)))
            records.append(record)

    return records


def load_all_results(results_root=DEFAULT_RESULTS_ROOT):
    results_root = Path(results_root)
    data = {}

    for model_dir, model_label in MODELS:
        data[model_label] = {}
        for state in STATES:
            path = results_root / model_dir / state / "results.txt"
            data[model_label][state] = parse_results_file(path)

    return data


def get_record(data, model_label, state, facet, split):
    for record in data[model_label][state]:
        if record["facet"] == facet and record["split"] == split:
            return record
    raise KeyError(f"Missing record: {model_label} {state} {facet} {split}")


def _save_figure(fig, base_path):
    base_path = Path(base_path)
    pdf_path = base_path.with_suffix(".pdf")
    png_path = base_path.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [pdf_path, png_path]


def plot_test_main_metrics(base_path, results_root=DEFAULT_RESULTS_ROOT):
    data = load_all_results(results_root)
    model_labels = [label for _, label in MODELS]
    metrics = [
        ("ndcg%20", "nDCG%@20"),
        ("recall@20", "R@20"),
    ]

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), sharey=False)
    x = np.arange(len(model_labels))
    width = 0.36

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        base_values = [
            get_record(data, model, "base", "all", "test")[metric_key]
            for model in model_labels
        ]
        ft_values = [
            get_record(data, model, "FT", "all", "test")[metric_key]
            for model in model_labels
        ]

        ax.bar(
            x - width / 2,
            base_values,
            width,
            label="base",
            color="#4C78A8",
            edgecolor="black",
            linewidth=0.6,
        )
        ax.bar(
            x + width / 2,
            ft_values,
            width,
            label="FT",
            color="#F58518",
            edgecolor="black",
            linewidth=0.6,
        )

        for xpos, value in zip(x - width / 2, base_values):
            ax.text(xpos, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=6)
        for xpos, value in zip(x + width / 2, ft_values):
            ax.text(xpos, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=6)

        ax.set_title(metric_label)
        ax.set_xticks(x, model_labels, rotation=20, ha="right")
        ax.grid(axis="y", color="#d0d0d0", linewidth=0.4)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Score")
    axes[0].legend(loc="upper left", frameon=False)
    fig.tight_layout(pad=0.45)
    return _save_figure(fig, base_path)


def csfcube_facet_gain_matrix(results_root=DEFAULT_RESULTS_ROOT):
    data = load_all_results(results_root)
    model_labels = [label for _, label in MODELS]
    rows = []

    for model in model_labels:
        gains = []
        for facet in FACETS:
            base_value = round(get_record(data, model, "base", facet, "test")["ndcg%20"], 3)
            ft_value = round(get_record(data, model, "FT", facet, "test")["ndcg%20"], 3)
            gains.append(ft_value - base_value)
        rows.append(gains)

    return model_labels, np.array(rows)


def plot_facet_gain_heatmap(base_path, results_root=DEFAULT_RESULTS_ROOT):
    model_labels, data = csfcube_facet_gain_matrix(results_root)
    facet_labels = ["Background", "Method", "Result", "All"]
    vmax = max(0.11, float(np.max(np.abs(data))))

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(facet_labels)), facet_labels)
    ax.set_yticks(np.arange(len(model_labels)), model_labels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)

    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            text_color = "white" if abs(value) >= 0.075 else "black"
            ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=7, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("FT - base nDCG%@20", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout(pad=0.4)
    return _save_figure(fig, base_path)


def generate_all(output_dir=Path("."), results_root=DEFAULT_RESULTS_ROOT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    outputs.extend(plot_test_main_metrics(output_dir / "fig_csfcube_test_main_metrics_v2", results_root))
    outputs.extend(plot_facet_gain_heatmap(output_dir / "fig_csfcube_facet_ndcg_gain_heatmap_v2", results_root))
    return outputs


if __name__ == "__main__":
    for output in generate_all():
        print(output)
