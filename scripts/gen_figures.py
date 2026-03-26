"""
Generate placeholder figures for the FinDocOCR paper.

Produces three PDFs in paper/figures/:
  - pipeline.pdf      HunyuanOCR + Dual LoRA block diagram
  - dataset_stats.pdf Horizontal bar chart of training pair counts per feature
  - ablation.pdf      Grouped bar chart with placeholder (zero) results

Usage:
    python scripts/gen_figures.py
"""

import pathlib

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# ── Output directory ──────────────────────────────────────────────────────────
_SCRIPT_DIR: pathlib.Path = pathlib.Path(__file__).resolve().parent
_FIGURES_DIR: pathlib.Path = _SCRIPT_DIR.parent / "paper" / "figures"

# ── Dataset statistics constants ──────────────────────────────────────────────
_FEATURE_LABELS: list[str] = [
    "Complex Tables",
    "Strikethrough",
    "Underline",
    "Colored Text",
]
_FEATURE_COUNTS: list[int] = [3565, 5446, 5000, 10000]

# ── Ablation constants ────────────────────────────────────────────────────────
_METRIC_LABELS: list[str] = ["Strike F1", "Under. F1", "Color Acc.", "TEDS"]
_MODEL_LABELS: list[str] = ["Vanilla", "LoRA", "Dual LoRA"]
_ABLATION_VALUES: list[list[float]] = [
    [0.0, 0.0, 0.0, 0.0],  # Vanilla
    [0.0, 0.0, 0.0, 0.0],  # LoRA
    [0.0, 0.0, 0.0, 0.0],  # Dual LoRA
]


def gen_pipeline(figures_dir: pathlib.Path) -> pathlib.Path:
    """Generate a block-diagram overview of HunyuanOCR + Dual LoRA."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # ── Main component boxes (top row) ────────────────────────────────────────
    components: list[tuple[float, str]] = [
        (1.0, "ViT Encoder\n(0.4B)"),
        (4.2, "MLP\nConnector"),
        (7.0, "LLM Decoder\n(0.5B)"),
    ]
    box_width, box_height = 2.2, 1.0
    box_y = 2.5
    for x, label in components:
        rect = mpatches.FancyBboxPatch(
            (x, box_y), box_width, box_height,
            boxstyle="round,pad=0.1",
            linewidth=1.5,
            edgecolor="#333333",
            facecolor="#dce9f7",
        )
        ax.add_patch(rect)
        ax.text(
            x + box_width / 2, box_y + box_height / 2,
            label, ha="center", va="center", fontsize=9, fontweight="bold",
        )

    # ── Arrows between main components ───────────────────────────────────────
    arrow_props = dict(arrowstyle="-|>", color="#555555", lw=1.5)
    ax.annotate("", xy=(4.2, box_y + 0.5), xytext=(3.2, box_y + 0.5),
                arrowprops=arrow_props)
    ax.annotate("", xy=(7.0, box_y + 0.5), xytext=(6.4, box_y + 0.5),
                arrowprops=arrow_props)

    # ── Dual LoRA adapter boxes (bottom row, under LLM) ──────────────────────
    adapter_labels: list[str] = ["A", "B", "C", "D"]
    adapter_colors: list[str] = ["#fde8c8", "#fde8c8", "#d8f0d8", "#d8f0d8"]
    adapter_w, adapter_h = 0.7, 0.6
    adapter_y = 1.3
    adapter_xs: list[float] = [6.85, 7.65, 8.45, 9.0]
    for i, (ax_x, alabel, acolor) in enumerate(
        zip(adapter_xs, adapter_labels, adapter_colors)
    ):
        rect = mpatches.FancyBboxPatch(
            (ax_x, adapter_y), adapter_w, adapter_h,
            boxstyle="round,pad=0.05",
            linewidth=1.2,
            edgecolor="#888888",
            facecolor=acolor,
        )
        ax.add_patch(rect)
        ax.text(
            ax_x + adapter_w / 2, adapter_y + adapter_h / 2,
            alabel, ha="center", va="center", fontsize=9,
        )

    # Vertical connector from LLM box to adapters
    ax.annotate(
        "", xy=(8.1, adapter_y + adapter_h), xytext=(8.1, box_y),
        arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.2),
    )

    # Legend for adapter groups
    ax.text(6.6, 0.85, "Magnitude (A,B)", fontsize=7, color="#b87c00")
    ax.text(8.3, 0.85, "Direction (C,D)", fontsize=7, color="#2a7a2a")

    ax.set_title("HunyuanOCR + Dual LoRA", fontsize=11, fontweight="bold", pad=8)

    out_path = figures_dir / "pipeline.pdf"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def gen_dataset_stats(figures_dir: pathlib.Path) -> pathlib.Path:
    """Generate a horizontal bar chart of training pair counts per feature."""
    fig, ax = plt.subplots(figsize=(6, 3))

    colors: list[str] = ["#5b9bd5", "#ed7d31", "#a9d18e", "#ffc000"]
    bars = ax.barh(
        _FEATURE_LABELS, _FEATURE_COUNTS,
        color=colors, edgecolor="#444444", linewidth=0.8, height=0.55,
    )

    # Count labels at end of each bar
    for bar, count in zip(bars, _FEATURE_COUNTS):
        ax.text(
            bar.get_width() + 80, bar.get_y() + bar.get_height() / 2,
            f"{count:,}", va="center", ha="left", fontsize=9,
        )

    ax.set_xlabel("Training Pairs", fontsize=9)
    ax.set_xlim(0, max(_FEATURE_COUNTS) * 1.18)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("FinDocOCR Training Pairs by Feature", fontsize=10, fontweight="bold")

    out_path = figures_dir / "dataset_stats.pdf"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def gen_ablation(figures_dir: pathlib.Path) -> pathlib.Path:
    """Generate a grouped bar chart with placeholder (zero) ablation results."""
    n_metrics = len(_METRIC_LABELS)
    n_models = len(_MODEL_LABELS)
    bar_width = 0.22
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    colors: list[str] = ["#bdbdbd", "#5b9bd5", "#ed7d31"]

    for i, (model, vals, color) in enumerate(
        zip(_MODEL_LABELS, _ABLATION_VALUES, colors)
    ):
        offsets = x + (i - (n_models - 1) / 2) * bar_width
        ax.bar(
            offsets, vals, bar_width,
            label=model, color=color, edgecolor="#444444", linewidth=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(_METRIC_LABELS, fontsize=9)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Ablation Results (placeholder)", fontsize=10, fontweight="bold")
    ax.text(
        0.5, 0.5, "Results pending GPU training",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=11, color="#aaaaaa", style="italic",
    )

    out_path = figures_dir / "ablation.pdf"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pipeline_path = gen_pipeline(_FIGURES_DIR)
    print(f"Generated: {pipeline_path}")

    stats_path = gen_dataset_stats(_FIGURES_DIR)
    print(f"Generated: {stats_path}")

    ablation_path = gen_ablation(_FIGURES_DIR)
    print(f"Generated: {ablation_path}")


if __name__ == "__main__":
    main()
