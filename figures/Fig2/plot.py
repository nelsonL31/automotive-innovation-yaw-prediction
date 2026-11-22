"""Figure 2. KIT yaw-rate errors by input information and steering excursion."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, to_rgb
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter


HERE = Path(__file__).resolve().parent
WIDTH = 6.35
INK = "#243440"
BLUE = "#21668C"
ORANGE = "#B9502A"
TEAL = "#147D73"
GRID = "#DDE4E8"
MODES = ("hold", "speed_only", "steer_only", "replay")
MODE_LABELS = ("Hold\nboth", "Replay\nspeed only", "Replay\nsteering only", "Replay\nboth")
METHODS = {
    "data": ("Data only", INK, "o", "solid"),
    "ode": ("ODE-PINN", TEAL, "s", (0, (4, 2))),
    "arx": ("ARX", ORANGE, "^", (0, (1.2, 1.5))),
}
DRIVE_NAMES = {
    "dynamic_driving_asphalt_a_cobble_1.mat": "Asphalt A + cobble",
    "dynamic_driving_asphalt_b_1.mat": "Asphalt B",
    "dynamic_driving_cobble_1.mat": "Cobble",
    "dynmic_asphalt_b_cobble_1.mat": "Asphalt B + cobble",
}


def read_csv(filename):
    with (HERE / filename).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def configure_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.8, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": "#798891",
        "axes.linewidth": .65, "axes.labelsize": 8.8,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.8,
        "xtick.major.width": .65, "ytick.major.width": .65,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "vsd-fig12",
        "axes.unicode_minus": True, "figure.facecolor": "white",
    })


def panel_title(ax, letter, title):
    for x, text, size, weight in [(0, f"({letter})", 10.9, "bold"), (18, title, 9.2, "bold")]:
        ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(x, 7),
                    textcoords="offset points", ha="left", va="baseline",
                    fontsize=size, fontweight=weight, annotation_clip=False)


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=.6)
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(length=2.6, pad=3)


def draw_methods(ax, values):
    for method, (name, color, marker, linestyle) in METHODS.items():
        ax.plot(range(len(MODES)), values[method], label=name, marker=marker,
                ls=linestyle, color=color, lw=1.5, ms=4.8, mfc="white", mew=1.15)
    ax.set_xticks(range(len(MODES)), MODE_LABELS)
    ax.set_xlim(-.19, 3.25)
    ax.set_ylabel("Yaw-rate RMSE (rad/s)")
    style_axes(ax)


def method_legend(fig, ax, y):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, y),
               ncol=3, frameon=False, columnspacing=2.3, handletextpad=.6,
               handlelength=2.6, borderaxespad=0)


def save(fig, stem):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(Path.cwd() / f"{stem}.{ext}", dpi=600, facecolor="white")
    plt.close(fig)


def drive_differences(ax, rows):
    for row in rows:
        hold, steer = float(row["hold"]), float(row["steer_only"])
        changes_sign = hold * steer < 0
        ax.plot([0, 1], [hold, steer], color="#6A7D88", lw=1.2,
                linestyle="solid" if changes_sign else (0, (3, 2)), zorder=2)
        for x, y in [(0, hold), (1, steer)]:
            color = BLUE if y < 0 else ORANGE if y > 0 else INK
            ax.plot(x, y, ls="none", marker="o", ms=4.4 if x else 3.1, mec=color, mew=1.1,
                    mfc=color if x == 1 else "white", zorder=3)
        ax.annotate(DRIVE_NAMES[row["method_or_record"]], xy=(1, steer),
                    xytext=(5, 0), textcoords="offset points", va="center", fontsize=8.3,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": .2})
    ax.axhline(0, color=INK, lw=.9, zorder=1)
    ax.set_xlim(-.13, 2.39)
    ax.set_ylim(-.132, .132)
    ax.set_xticks([0, 1], [MODE_LABELS[0], MODE_LABELS[2]])
    ax.set_yticks([-.1, -.05, 0, .05, .1])
    ax.set_ylabel("RMSE difference (rad/s)\nODE-PINN − data only")
    style_axes(ax)


def steering_strata(ax, strata):
    bounds = sorted({(float(r["min_rad"]), float(r["max_rad"]), int(r["trips"]))
                     for r in strata})
    for method, offset in [("data", -.095), ("ode", .095)]:
        _, color, marker, _ = METHODS[method]
        rows = sorted((r for r in strata if r["method"] == method),
                      key=lambda r: float(r["min_rad"]))
        ax.plot(np.arange(3) + offset, [float(r["yaw_rmse_radps"]) for r in rows],
                ls="none", color=color, marker=marker, ms=5.1, mew=1.25, mfc="white")
    labels = []
    for index, (lower, upper, drives) in enumerate(bounds):
        lower_label = "0" if lower == 0 else f"{lower:.2f}"
        boundary = f"≥{lower_label}" if index == len(bounds) - 1 else f"{lower_label}–{upper:.2f}"
        labels.append(f"{boundary}\n{drives} drives")
    ax.set_xticks(range(3), labels)
    ax.set_xlim(-.48, 2.48)
    ax.set_ylim(0, .455)
    ax.set_yticks(np.arange(5) * .1)
    ax.set_xlabel("Steering excursion (rad)")
    ax.set_ylabel("Yaw-rate RMSE (rad/s)")
    style_axes(ax)


def plot_kit():
    factor = read_csv("input_conditions.csv")
    strata = read_csv("steering_excursion.csv")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.72))
    fig.subplots_adjust(left=.113, right=.985, top=.856, bottom=.147,
                        hspace=.60, wspace=.38)
    for ax, letter, title in zip(axes[0], ("a", "b"),
                                 ("Same surface · 5 drives", "Cross-surface · 4 drives")):
        values = {r["method_or_record"].replace("linear_ARX", "arx"):
                  [float(r[m]) for m in MODES] for r in factor if r["panel"] == letter}
        draw_methods(ax, values)
        ax.set_ylim(0, .275)
        ax.set_yticks([0, .05, .10, .15, .20, .25])
        panel_title(ax, letter, title)
    method_legend(fig, axes[0, 0], .983)
    drive_differences(axes[1, 0], [r for r in factor if r["panel"] == "c"])
    panel_title(axes[1, 0], "c", "Paired cross-surface drives")
    steering_strata(axes[1, 1], strata)
    panel_title(axes[1, 1], "d", "Steering excursion · double hold")
    save(fig, "Fig2")



if __name__ == "__main__":
    configure_style()
    plot_kit()
