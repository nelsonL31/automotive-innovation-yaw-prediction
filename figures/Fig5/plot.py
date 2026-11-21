"""Plot 0.6 s endpoint gain distributions for the six estimator contrasts."""
from pathlib import Path
from decimal import Decimal
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patheffects as effects
import numpy as np

ROOT = Path(__file__).resolve().parent
INK = "#172E40"
GRID = "#D8E1E7"
UNIT = "#505767"
SURFACE = ("sealed_id", "sealed_surface")
ARMS = ("pgnn_ref", "pgnn_matched")
CONDITIONS = ("gain", "gain0", "gain_per_unit_stacked_norm")
# Each condition has the same row, colour and sample shape in both panels.
PALETTE = ("#0072B2", "#D55E00", "#009E73")
LABEL_COLOURS = ("#005D91", "#A54900", "#007353")
SAMPLE_MARKERS = ("o", "^", "s")
LOCATIONS = (6.28, 5.33, 4.38, 2.68, 1.73, .78)
RANGES = (135, 108)


def read_csv():
    with (ROOT / "local_gains.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = ("group", "arm", "condition", "record", "seed", "origin_within_record_seed")
    assert len(rows) == len({tuple(r[k] for k in keys) for r in rows}) == 1458
    return rows


def configure(chinese):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei" if chinese else "Arial", "DejaVu Sans"],
        "font.size": 9.3, "mathtext.fontset": "stixsans", "axes.unicode_minus": True,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "svg.fonttype": "none", "pdf.fonttype": 42, "text.color": INK,
        "axes.labelcolor": INK, "ytick.color": "#394C5D", "xtick.color": "#394C5D",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelsize": 9.4, "xtick.labelsize": 8.6, "ytick.labelsize": 8.6,
        "axes.edgecolor": "#82919D", "axes.linewidth": .55, "xtick.major.size": 2.6,
        "ytick.major.size": 2.6, "xtick.major.width": .55, "ytick.major.width": .55,
    })


def median_label(value):
    rounded = Decimal(f"{value:.3e}")
    return format(rounded, "f")


def plot_distribution(ax, values, row, colour, marker):
    values = np.sort(values)
    rank = np.arange(len(values))
    jitter = .115 * (2 * ((rank * .61803398875) % 1) - 1)
    ax.scatter(values, row + jitter, marker=marker, c=colour, alpha=.54,
               linewidth=0, s=6.0, zorder=2)
    low, median, high = np.percentile(values, [25, 50, 75])
    ax.plot([low, high], [row, row], color=colour, lw=3.0,
            solid_capstyle="round", zorder=4)
    ax.plot(median, row, marker="D", color=colour, mec="white",
            mew=.8, ms=6.6, ls="none", zorder=5)
    return ax.annotate(median_label(median), (median, row), xytext=(0, 3.7),
                       textcoords="offset points", ha="center", va="bottom",
                       fontsize=8.3, color=INK, zorder=6,
                       path_effects=[effects.withStroke(linewidth=1.5, foreground="white")])


def draw(rows, chinese=False):
    configure(chinese)
    fig = plt.figure(figsize=(7.20, 4.06))
    names = (["完整导数", "截断刚度路径", "持续偏置"] if chinese
             else ["Full derivative", "Stiffness path cut", "Persistent bias"])
    titles = ["同路面", "跨路面"] if chinese else ["Same-surface drives", "Cross-surface drives"]
    arm_labels = ["参考 GRU", "同设定 MLP"] if chinese else ["Reference GRU", "Matched MLP"]
    # Sparse labels keep the entire observed range legible in ordinary decimal notation.
    exponents = [-16, -8, -3, 0, 3]
    ticks = [10 ** k for k in exponents]
    tick_labels = [format(Decimal(1).scaleb(k), "f") for k in exponents]
    annotations = []
    for panel, (group, left) in enumerate(zip(SURFACE, [.188, .599])):
        ax = fig.add_axes([left, .18, .359, .625])
        ax.text(0, 1.075, f"({'ab'[panel]})  {titles[panel]}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=10.5, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xlim(1e-18, 1e3)
        ax.set_ylim(.23, 7.14)
        ax.axvspan(1, 1e3, facecolor="#F1F1F6", zorder=-3)
        ax.axvline(1, color=UNIT, ls=(0, (2, 1.8)), lw=1.06, zorder=3,
                   path_effects=[effects.withStroke(linewidth=1.9, foreground="white")])
        ax.set_axisbelow(True)
        ax.set_xticks(ticks, tick_labels)
        ax.tick_params(axis="x", labelsize=8.0)
        # Set the longest decimal on a second tier so it reads independently
        # from the right-edge tick of the neighbouring panel.
        ax.xaxis.get_major_ticks()[0].set_pad(10.5)
        ax.get_xticklabels()[-1].set_ha("right")
        ax.minorticks_off()
        ax.grid(axis="x", lw=.5, color=GRID)
        ax.set_yticks(LOCATIONS, names * 2 if panel == 0 else [""] * 6)
        if panel == 0:
            for text, colour in zip(ax.get_yticklabels(), LABEL_COLOURS * 2):
                text.set_color(colour)
        ax.tick_params(axis="y", length=0, pad=4.0)
        ax.spines["left"].set_visible(False)
        ax.axhline(3.77, color="#CBD7DF", lw=.6, zorder=-1)
        for arm_number, arm in enumerate(ARMS):
            top = arm_number * 3
            ax.text(0, LOCATIONS[top] + .55, arm_labels[arm_number],
                    transform=ax.get_yaxis_transform(), fontweight="bold",
                    fontsize=8.8, color=INK)
            for k, condition in enumerate(CONDITIONS):
                values = np.array([float(r["gain"]) for r in rows
                                   if (r["group"], r["arm"], r["condition"])
                                   == (group, arm, condition)])
                assert len(values) == RANGES[panel] and np.all((values > 1e-18) & (values < 1e3))
                label = plot_distribution(ax, values, LOCATIONS[top + k], PALETTE[k], SAMPLE_MARKERS[k])
                annotations.append(label)

    xlabel = ("0.6 s 端点增益（无量纲，对数轴）" if chinese
              else "0.6 s endpoint gain (dimensionless, log scale)")
    fig.text(.563, .055, xlabel, ha="center", fontsize=9.5)
    legend = [
        tuple(Line2D([], [], marker=marker, color=colour, ls="none", ms=3.6)
              for marker, colour in zip(SAMPLE_MARKERS, PALETTE)),
        Line2D([], [], color="#354F61", lw=3, solid_capstyle="round"),
        Line2D([], [], marker="D", color="#354F61", ls="none", ms=4.8),
        Line2D([], [], color=UNIT, ls=(0, (2, 1.8)), lw=1.06),
    ]
    legend_labels = (["逐起点样本", "四分位距", "中位数", "单位增益"] if chinese
                     else ["Sample", "Interquartile range", "Median", "Unit gain"])
    fig.legend(legend, legend_labels, loc="upper center", bbox_to_anchor=(.54, .998),
               frameon=False, ncol=4, columnspacing=1.30, handlelength=2.0,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=.20)},
               handletextpad=.37, fontsize=8.4)
    # Keep the labels above their own medians. Only shift the longest labels
    # inward if the centred lettering would reach the edge of its panel.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pixels_per_point = fig.dpi / 72
    for label in annotations:
        box = label.get_window_extent(renderer)
        edge = label.axes.get_window_extent(renderer)
        overflow = max(0, (box.x1 - edge.x1) / pixels_per_point + 3.0)
        if overflow:
            label.set_position((-overflow, 3.7))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(Path.cwd() / f"Fig5.{ext}", dpi=450)
    plt.close(fig)


if __name__ == "__main__":
    data = read_csv()
    draw(data, chinese=False)
