"""Render submission figures at their final manuscript width (vector PDF/SVG).

Run from the repository root: python scripts/render_submission_figures.py
Data are loaded through the original research plotting modules; only layout,
lettering, and styles are changed here.
"""

from pathlib import Path
import importlib.util

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 5.15  # Approximately 131 mm, matching sn-jnl's text width.


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "resources/HRIA" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(fig, stem):
    # Fixed canvas preserves final-sized font sizes when included at textwidth.
    fig.savefig(ROOT / f"{stem}.pdf")
    svg = ROOT / f"{stem}.svg"
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def pipeline():
    fig = plt.figure(figsize=(WIDTH, 3.35))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, WIDTH), ylim=(0, 3.35))
    ax.axis("off")

    def group(y, height, title, color):
        ax.add_patch(FancyBboxPatch((.025, y), 5.10, height,
                     boxstyle="round,pad=0.005,rounding_size=0.08",
                     facecolor=color, edgecolor="#66717b", linewidth=.65))
        ax.text(.13, y + height - .15, title, size=9, weight="bold", va="center")

    def box(x, y, title, detail, fill="white"):
        ax.add_patch(FancyBboxPatch((x, y), 1.49, .59,
                     boxstyle="round,pad=0.015,rounding_size=0.045",
                     facecolor=fill, edgecolor="#41494f", linewidth=.8))
        ax.text(x + .745, y + .425, title, size=8.5, ha="center", va="center", weight="bold")
        ax.text(x + .745, y + .19, detail, size=8.5, ha="center", va="center", linespacing=1.25)

    def arrow(start, end, **kwargs):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                     mutation_scale=9, color="#303840", linewidth=.9, **kwargs))

    group(1.40, 1.92, "Training pipeline", "#eaf2f8")
    xs = [.12, 1.83, 3.54]
    box(xs[0], 2.38, "DORIS-MAE", "100 parent queries\nGraded candidate pools")
    box(xs[1], 2.38, "Aspect decomposition", "Aspect and sub-aspect\nranking instances")
    box(xs[2], 2.38, "Candidate sampling", "Target list size K = 50")
    box(xs[2], 1.62, "Pool augmentation", "Up to 3 distinct pools\nper aspect")
    box(xs[1], 1.62, "Listwise fine-tuning", "LoRA / QLoRA\nRanking prompts")
    box(xs[0], 1.62, "Fine-tuned reranker", "Adapted model", "#d9e8f2")
    for left in xs[:2]:
        arrow((left + 1.51, 2.675), (left + 1.69, 2.675))
    arrow((4.285, 2.36), (4.285, 2.23))
    for right in xs[1:]:
        arrow((right - .02, 1.915), (right - .20, 1.915))

    group(.025, 1.20, "Inference and evaluation\nLitSearch / CSFCube", "#f5eee8")
    box(xs[0], .14, "Candidate retrieval", "Unseen query\nBM25 / dense / pool")
    box(xs[1], .14, "Reranking", "Top-K titles + abstracts\nBase / fine-tuned model")
    box(xs[2], .14, "Ranked output", "Relevance ordering\nEvaluation metrics")
    for left in xs[:2]:
        arrow((left + 1.51, .435), (left + 1.69, .435))
    arrow((1.57, 1.60), (3.05, .75), linestyle="--")
    save(fig, "Fig1")


def learning_curve():
    source = load_module("learning_curve", "generate_parent_query_learning_curve.py")
    summaries = source.aggregate_learning_curve(source.DEFAULT_RESULTS_ROOT)
    baseline = source.load_zero_shot_baseline(
        ROOT / "resources/LitSearch/results/reranking/llama3/base/k_100"
    )
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.48), sharey=True)
    styles = [("#245786", "o", "-"), ("#a94c00", "s", "--"), ("#28703c", "^", "-.")]
    panels = [("parent_queries", "(a) Parent queries", [0, 50, 100]),
              ("aspect_rows", "(b) Aspect rows", [0, 750, 1500]),
              ("optimizer_steps", "(c) Optimizer steps", [0, 2000, 4000])]
    for ax, (scale, label, ticks) in zip(axes, panels):
        for metric, (color, marker, line) in zip(source.METRICS, styles):
            xs, ys, errors = [], [], []
            for budget in source.BUDGETS:
                seed_x = [source.training_scale(budget, seed)[scale] for seed in source.SEEDS]
                seed_y = [summaries[budget]["seed_values"][seed][metric] for seed in source.SEEDS]
                xs.append(np.mean(seed_x))
                ys.append(summaries[budget][f"{metric}_mean"])
                errors.append(summaries[budget][f"{metric}_sd"])
                ax.scatter(seed_x, seed_y, s=8, color=color, alpha=.25, linewidths=0)
            ax.errorbar(xs, ys, yerr=errors, color=color, marker=marker,
                        linestyle=line, markersize=3, linewidth=.9, capsize=1.8, label=metric)
            ax.scatter([0], [baseline[metric]], marker=marker, s=18,
                       facecolors="white", edgecolors=color, linewidths=.9, zorder=3)
        ax.set_xlabel(label, fontsize=8.5, labelpad=6)
        ax.set_xticks(ticks)
        ax.tick_params(axis="both", labelsize=8.5, pad=2)
        ax.set_ylim(44, 79)
        ax.set_yticks([45, 55, 65, 75])
        ax.margins(x=.07)
        ax.grid(color="#d8d8d8", linewidth=.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("LitSearch performance (%)", fontsize=9, labelpad=4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.55, 1.0),
               ncol=3, frameon=False, fontsize=9, columnspacing=1.1, handlelength=1.8)
    fig.text(.55, .015, "Open markers: zero-shot reference", ha="center", fontsize=8.5)
    fig.subplots_adjust(left=.115, right=.975, bottom=.25, top=.82, wspace=.19)
    save(fig, "Fig2")


def gpt_comparison():
    source = load_module("comparison", "generate_graphs.py")
    configs = source.gpt4o_comparison_data(ROOT / "resources/LitSearch/results")
    fig, ax = plt.subplots(figsize=(WIDTH, 2.85))
    x, width = np.arange(3), .25
    styles = [("#85adc9", ""), ("#ecc38b", "///"), ("#93bd9c", "...")]
    for index, ((name, values), (color, hatch)) in enumerate(zip(configs.items(), styles)):
        name = name.replace("Gemma27", "Gemma 3 27B")
        positions = x + (index - 1) * width
        ax.bar(positions, values, width, label=name, color=color, edgecolor="#333333",
               linewidth=.65, hatch=hatch)
        for xpos, value in zip(positions, values):
            ax.text(xpos, value + .5, f"{value:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x, ["Broad R@20", "Spec. R@5", "Spec. R@20"], fontsize=9)
    ax.set_ylabel("Recall (%)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_ylim(65, 89)
    ax.grid(axis="y", color="#d0d0d0", linewidth=.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.56), ncol=1, frameon=False,
              fontsize=8.5, borderaxespad=0, labelspacing=.35)
    fig.subplots_adjust(left=.12, right=.98, bottom=.14, top=.66)
    save(fig, "FigS2")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    pipeline()
    learning_curve()
    gpt_comparison()
    print("Rendered Fig1, Fig2, and FigS2 as final-width PDF and editable SVG.")
