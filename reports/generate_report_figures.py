"""Regenerate the report figures with a cleaner, more consistent visual style.

Run from the project root after the evaluation JSON files and training histories
have been produced:

    source .venv/bin/activate
    python generate_report_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"


COLORS = {
    "ink": "#1F2A37",
    "subtle": "#6B7280",
    "grid": "#D9E1EA",
    "panel": "#F7FAFD",
    "blue": "#2F6DB2",
    "blue_light": "#78A9E0",
    "orange": "#F28E2B",
    "green": "#59A14F",
    "teal": "#76B7B2",
    "red": "#E15759",
    "gray": "#8A94A6",
    "gold": "#C58F1F",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def set_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 1.1,
            "axes.facecolor": COLORS["panel"],
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.9,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def make_axes(size=(6.4, 4.0)):
    fig, ax = plt.subplots(figsize=size, constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(COLORS["panel"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return fig, ax


def annotate_vertical_bars(ax, bars, fmt="{:.1f}", dy=3):
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            fmt.format(value),
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=COLORS["ink"],
            fontweight="semibold",
        )


def annotate_horizontal_bars(ax, bars, fmt="{:.2f}", dx=4):
    for bar in bars:
        value = bar.get_width()
        ax.annotate(
            fmt.format(value),
            xy=(value, bar.get_y() + bar.get_height() / 2),
            xytext=(dx, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9.5,
            color=COLORS["ink"],
            fontweight="semibold",
        )


def pretty_extractive_name(name: str) -> str:
    mapping = {
        "hf_deepset_roberta-base-squad2": "RoBERTa-base",
        "hf_bert-large-uncased-whole-word-masking-finetuned-squad": "BERT-large",
        "custom_scratch_encoder_squadv2": "Scratch encoder",
        "hf_distilbert-base-uncased-distilled-squad": "DistilBERT",
    }
    return mapping.get(name, name.replace("hf_", "").replace("_", " "))


def pretty_generative_name(name: str) -> str:
    mapping = {
        "our_hybrid_decoder": "Hybrid",
        "t5-small": "T5-small",
        "t5-base": "T5-base",
        "google_flan-t5-small": "Flan-T5-small",
    }
    return mapping.get(name, name.replace("_", " "))


def training_history_plot(history_path: Path, title: str, out_path: Path, color: str) -> None:
    history = load_json(history_path)
    epochs = [int(item["epoch"]) for item in history]
    losses = [float(item["train_loss"]) for item in history]

    fig, ax = make_axes(size=(6.35, 3.95))

    if len(epochs) > 1:
        ax.plot(
            epochs,
            losses,
            color=color,
            linewidth=3.0,
            marker="o",
            markersize=9,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=2.0,
        )
    else:
        ax.scatter(
            epochs,
            losses,
            s=130,
            color=color,
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )

    for epoch, loss in zip(epochs, losses):
        ax.annotate(
            f"{loss:.3f}",
            xy=(epoch, loss),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color=COLORS["ink"],
            fontweight="semibold",
        )

    start = losses[0]
    final = losses[-1]
    delta = final - start
    summary_lines = [
        f"Epochs logged: {len(epochs)}",
        f"Start loss: {start:.3f}",
        f"Final loss: {final:.3f}",
    ]
    if len(epochs) > 1:
        summary_lines.append(f"Delta: {delta:+.3f}")
    else:
        summary_lines.append("Single logged epoch")

    ax.text(
        0.02,
        0.06,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        color=COLORS["ink"],
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#EEF4FB",
            "edgecolor": "#C9D5E4",
            "linewidth": 1.0,
        },
    )

    ax.set_title(title, pad=12, fontweight="semibold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")

    epoch_min = min(epochs)
    epoch_max = max(epochs)
    if epoch_min == epoch_max:
        ax.set_xlim(epoch_min - 0.5, epoch_max + 0.5)
    else:
        ax.set_xlim(epoch_min - 0.15, epoch_max + 0.15)

    if len(losses) == 1:
        pad = max(0.03, abs(final) * 0.08)
    else:
        span = max(losses) - min(losses)
        pad = max(0.03, span * 0.30)
    ax.set_ylim(min(losses) - pad, max(losses) + pad)

    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def extractive_baseline_plot(data_path: Path, out_path: Path) -> None:
    data = load_json(data_path)
    rows = data["ranking_by_best_f1"]

    labels = [pretty_extractive_name(row["name"]) for row in rows]
    em = []
    f1 = []
    for row in rows:
        model = next(item for item in data["models"] if item["name"] == row["name"])
        summary = model["summary"]
        em.append(float(summary["best_exact"]))
        f1.append(float(summary["best_f1"]))

    x = list(range(len(labels)))
    width = 0.34

    fig, ax = make_axes(size=(7.4, 4.15))
    em_bars = ax.bar([i - width / 2 for i in x], em, width=width, color=COLORS["blue"], label="EM")
    f1_bars = ax.bar([i + width / 2 for i in x], f1, width=width, color=COLORS["orange"], label="F1")

    annotate_vertical_bars(ax, em_bars, fmt="{:.1f}")
    annotate_vertical_bars(ax, f1_bars, fmt="{:.1f}")

    ax.set_title("Extractive QA baselines on SQuAD v2", pad=12, fontweight="semibold")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=14, ha="right")
    ax.set_ylim(0, max(max(em), max(f1)) * 1.16)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.legend(loc="upper right", ncol=2)
    ax.text(
        0.02,
        0.95,
        "Higher is better",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        color=COLORS["subtle"],
    )

    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def generative_comparison_plot(data_path: Path, out_path: Path) -> None:
    data = load_json(data_path)
    rows = data["ranking_by_f1"]

    labels = []
    scores = []
    colors = []
    for row in rows:
        model = next(item for item in data["models"] if item["name"] == row["name"])
        labels.append(pretty_generative_name(row["name"]))
        scores.append(float(row["f1"]))
        if model["name"] == "our_hybrid_decoder":
            colors.append(COLORS["blue"])
        elif model["name"] == "t5-base":
            colors.append(COLORS["green"])
        elif model["name"] == "t5-small":
            colors.append(COLORS["gray"])
        else:
            colors.append(COLORS["teal"])

    fig, ax = make_axes(size=(7.1, 4.0))
    bars = ax.bar(labels, scores, color=colors, width=0.72)
    annotate_vertical_bars(ax, bars, fmt="{:.1f}")

    ax.set_title("Generative QA F1 on the shared 1k validation subset", pad=12, fontweight="semibold")
    ax.set_ylabel("F1")
    ax.set_ylim(0, max(scores) * 1.16)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.tick_params(axis="x", rotation=12)
    ax.text(
        0.02,
        0.95,
        "Best external baseline: T5-base",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        color=COLORS["subtle"],
    )

    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def output_length_plot(data_path: Path, out_path: Path) -> None:
    data = load_json(data_path)
    rows = data["ranking_by_f1"]

    labels = []
    values = []
    colors = []
    for row in rows:
        model = next(item for item in data["models"] if item["name"] == row["name"])
        labels.append(pretty_generative_name(row["name"]))
        values.append(float(model["avg_output_len"]))
        if model["name"] == "our_hybrid_decoder":
            colors.append(COLORS["blue"])
        elif model["name"] == "t5-base":
            colors.append(COLORS["green"])
        elif model["name"] == "t5-small":
            colors.append(COLORS["gray"])
        else:
            colors.append(COLORS["teal"])

    fig, ax = make_axes(size=(7.1, 3.85))
    bars = ax.barh(labels, values, color=colors, height=0.62)
    annotate_horizontal_bars(ax, bars, fmt="{:.2f}")

    ax.set_title("Average generated length", pad=12, fontweight="semibold")
    ax.set_xlabel("Tokens")
    ax.set_xlim(0, max(values) * 1.25)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.invert_yaxis()
    ax.text(
        0.98,
        0.95,
        "Shorter is more concise",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.2,
        color=COLORS["subtle"],
    )

    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def _panel_background(ax, title: str, panel_label: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.015, 0.03),
            0.97,
            0.94,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.2,
            edgecolor="#CBD5E1",
            facecolor="#FCFEFF",
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.03, 0.875),
            0.94,
            0.075,
            boxstyle="round,pad=0.01,rounding_size=0.018",
            linewidth=0,
            facecolor="#2A2E35",
        )
    )
    ax.text(
        0.5,
        0.912,
        title,
        ha="center",
        va="center",
        color="white",
        fontsize=14.2,
        fontweight="semibold",
    )
    ax.text(
        0.045,
        0.912,
        panel_label,
        ha="left",
        va="center",
        color="#B8C2D0",
        fontsize=12.0,
        fontweight="semibold",
    )


def _rounded_box(ax, x, y, w, h, text, facecolor, edgecolor="#7B8794", text_color="#1F2937", fontsize=9.6):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.014,rounding_size=0.012",
            linewidth=1.2,
            edgecolor=edgecolor,
            facecolor=facecolor,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        fontweight="medium",
        linespacing=1.15,
    )


def _arrow(ax, start, end, color="#6B7280", lw=1.5, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def architecture_overview_figure(out_path: Path) -> None:
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(14.2, 7.8),
        dpi=300,
        constrained_layout=True,
    )

    # Top panel: encoder + extractive QA.
    _panel_background(ax_top, "Encoder + extractive QA", "A")
    y = 0.60
    h = 0.16
    boxes = [
        (0.04, 0.16, "Input pair\nquestion + context\ninput_ids\ntoken_type_ids\nattention_mask", COLORS["green"], 9.0),
        (0.23, 0.16, "Embeddings\ntoken + position +\nsegment\nLayerNorm + dropout", COLORS["green"], 9.0),
        (0.43, 0.23, "12 Transformer blocks\npre-norm residual self-attn\nFFN 3072, 12 heads, hidden 768", COLORS["green"], 8.8),
        (0.69, 0.13, "Final\nLayerNorm", COLORS["green"], 9.2),
        (0.85, 0.11, "Contextual states\nshape: L × 768", COLORS["green"], 9.0),
    ]

    for x, w, text, fc, size in boxes:
        _rounded_box(ax_top, x, y, w, h, text, fc, fontsize=size)

    for (x1, w1, _, _, _), (x2, _, _, _, _) in zip(boxes[:-1], boxes[1:]):
        _arrow(ax_top, (x1 + w1, y + h / 2), (x2, y + h / 2))

    head_x, head_y, head_w, head_h = 0.67, 0.28, 0.25, 0.16
    _rounded_box(
        ax_top,
        head_x,
        head_y,
        head_w,
        head_h,
        "Extractive QA head\nLinear(768→2)\nstart/end logits",
        "#FBEFDE",
        edgecolor="#B79A78",
        fontsize=9.0,
    )
    _arrow(ax_top, (boxes[-1][0] + boxes[-1][1] / 2, y), (head_x + head_w / 2, head_y + head_h))

    out_x, out_y, out_w, out_h = 0.68, 0.09, 0.23, 0.12
    _rounded_box(
        ax_top,
        out_x,
        out_y,
        out_w,
        out_h,
        "Extracted answer\nor no-answer fallback",
        "#EEF2F7",
        edgecolor="#C0C8D4",
        fontsize=8.9,
    )
    _arrow(ax_top, (head_x + head_w / 2, head_y), (out_x + out_w / 2, out_y + out_h))

    ax_top.text(
        0.5,
        0.195,
        "The encoder is the shared representation source for both QA modes.",
        ha="center",
        va="center",
        fontsize=9.2,
        color=COLORS["subtle"],
    )

    # Bottom panel: decoder + generative QA.
    _panel_background(ax_bottom, "Decoder + generative QA", "B")
    y2 = 0.60
    boxes2 = [
        (0.04, 0.16, "Target prefix\nBOS + previous tokens\ncausal mask", COLORS["blue_light"], 9.0),
        (0.23, 0.16, "Decoder embeddings\ntoken + position\nLayerNorm + dropout", COLORS["blue_light"], 9.0),
        (0.43, 0.25, "Decoder stack\nStandard: 4 TransformerDecoderLayer blocks\nHybrid: 4 custom blocks\nself-attn + cross-attn + FFN", COLORS["blue_light"], 8.6),
        (0.71, 0.13, "Final\nLayerNorm", COLORS["blue_light"], 9.2),
        (0.86, 0.11, "Tied LM\nhead", COLORS["blue_light"], 9.2),
    ]

    for x, w, text, fc, size in boxes2:
        _rounded_box(ax_bottom, x, y2, w, h, text, fc, fontsize=size)

    for (x1, w1, _, _, _), (x2, _, _, _, _) in zip(boxes2[:-1], boxes2[1:]):
        _arrow(ax_bottom, (x1 + w1, y2 + h / 2), (x2, y2 + h / 2))

    bridge_x, bridge_y, bridge_w, bridge_h = 0.36, 0.30, 0.24, 0.13
    _rounded_box(
        ax_bottom,
        bridge_x,
        bridge_y,
        bridge_w,
        bridge_h,
        "Shared encoder memory\nLinear(768→512) bridge",
        "#EEF2F7",
        edgecolor="#C0C8D4",
        fontsize=8.9,
    )
    _arrow(ax_bottom, (bridge_x + bridge_w / 2, bridge_y + bridge_h), (boxes2[2][0] + boxes2[2][1] / 2, y2))

    out2_x, out2_y, out2_w, out2_h = 0.76, 0.09, 0.20, 0.12
    _rounded_box(
        ax_bottom,
        out2_x,
        out2_y,
        out2_w,
        out2_h,
        "Generated answer\nor no-answer fallback",
        "#EEF2F7",
        edgecolor="#C0C8D4",
        fontsize=8.9,
    )
    _arrow(ax_bottom, (boxes2[4][0] + boxes2[4][1] / 2, y2), (out2_x + out2_w / 2, out2_y + out2_h))

    ax_bottom.text(
        0.5,
        0.195,
        "The decoder stays aligned to encoder evidence through cross-attention.",
        ha="center",
        va="center",
        fontsize=9.2,
        color=COLORS["subtle"],
    )

    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def main() -> None:
    set_theme()
    FIGURES.mkdir(exist_ok=True)

    architecture_overview_figure(FIGURES / "architecture_overview.png")

    training_history_plot(
        ROOT / "checkpoints_generative_qa_hybrid_span_restart1_20260426_010636" / "train_history.json",
        "Primary decoder training loss",
        FIGURES / "hybrid_prev_loss.png",
        COLORS["blue"],
    )
    training_history_plot(
        ROOT / "checkpoints_generative_qa_hybrid_span_noans_upweight_20260426_111025" / "train_history.json",
        "NoAns x3 continuation loss",
        FIGURES / "hybrid_noans3_loss.png",
        COLORS["red"],
    )
    extractive_baseline_plot(ROOT / "comparison_squadv2_results.json", FIGURES / "extractive_baselines.png")
    generative_comparison_plot(
        ROOT / "comparison_generative_seq2seq_20260426_121748.json",
        FIGURES / "generative_comparison.png",
    )
    output_length_plot(ROOT / "comparison_generative_seq2seq_20260426_121748.json", FIGURES / "output_length.png")

    print("Regenerated report figures in:", FIGURES)


if __name__ == "__main__":
    main()