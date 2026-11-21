"""Figure 6. Paired 0.6 s endpoint errors for six LMSD driving records."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
METHODS = ("data", "ode", "pgnn_ref")
STATES = ("vx", "vy", "yaw_rate")
METRICS = tuple(f"{state}_{kind}" for state in STATES for kind in ("RMSE", "max"))
RECORDS = ("FLPG_Track7_5", "FLPG_VDA_DoubleLane1", "FLPG_VDA_Sinesweep4",
           "FLPG_VDA_Slalom3", "Kristalpark_ConstRadCorn2", "Residential_Area_2")
FIELDS = ("record", "method", *METRICS)
INK = "#263341"
MEAN = "#202733"
GRID = "#E1E7EA"
# Labels retain the suffixes of the released records. Shape and dash pattern
# reinforce colour for the six paired observations in every panel.
DRIVE_STYLES = {
    "FLPG_Track7_5": ("Track 7_5", "#1369CE", "o", "-"),
    "FLPG_VDA_DoubleLane1": ("Double lane 1", "#E55218", "s", (0, (4, 1.55))),
    "FLPG_VDA_Sinesweep4": ("Sine sweep 4", "#00884D", "^", (0, (4, 1.3, 1, 1.3))),
    "FLPG_VDA_Slalom3": ("Slalom 3", "#8844C8", "D", (0, (1.1, 1.25))),
    "Kristalpark_ConstRadCorn2": ("Constant radius 2", "#CD1670", "v", (0, (6, 1.7))),
    "Residential_Area_2": ("Residential 2", "#BC8200", "P", (0, (2.4, 1.15))),
}


def load_and_verify():
    with (HERE / "record_errors.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    paired = {(row["record"], row["method"]): row for row in rows}
    expected = {(d, m) for d in RECORDS for m in METHODS}
    if len(rows) != len(expected) or set(paired) != expected:
        raise ValueError("Expected six complete test drives for each of the three models")
    data = {k: np.array([[float(paired[d, m][k]) for m in METHODS] for d in RECORDS]) for k in METRICS}
    if any(not np.isfinite(v).all() or np.min(v) < 0 for v in data.values()):
        raise ValueError("Absolute endpoint errors must be finite and non-negative")
    mean = {k: values.mean(axis=0) for k, values in data.items()}
    checked = set()
    with (HERE / "aggregate_errors.csv").open(encoding="utf-8", newline="") as file:
        for r in csv.DictReader(file):
            if r["control"] not in ("data", "pgnn_ref"):
                continue
            k, control = r["metric"], r["control"]
            assert (r["records"], r["seeds"]) == ("6", "3")
            np.testing.assert_allclose(mean[k][1], float(r["ode_error"]), rtol=1e-12)
            np.testing.assert_allclose(mean[k][METHODS.index(control)], float(r["baseline_error"]), rtol=1e-12)
            checked.add((k, control))
    assert checked == {(k, m) for k in METRICS for m in ("data", "pgnn_ref")}
    return data


def plot_lmsd_pairs():
    data = load_and_verify()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5, "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.labelcolor": INK, "text.color": INK, "axes.edgecolor": "#87979E",
        "xtick.color": INK, "ytick.color": INK, "axes.linewidth": .65,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": .55, "ytick.major.width": .55,
    })
    fig, axes = plt.subplots(2, 3, figsize=(6.35, 4.30), sharex=True)
    fig.subplots_adjust(left=.093, right=.988, top=.887, bottom=.250, hspace=.260, wspace=.345)
    titles = ("Longitudinal velocity", "Lateral velocity", "Yaw rate")
    units = ("m/s", "m/s", "rad/s")
    ticks = {"vx_RMSE": [0, .2, .4, .6], "vy_RMSE": [0, .1, .2, .3, .4],
             "yaw_rate_RMSE": [0, .05, .1, .15, .2], "vx_max": [0, .5, 1, 1.5, 2],
             "vy_max": [0, .3, .6, .9, 1.2], "yaw_rate_max": [0, .1, .2, .3]}
    upper = {"vx_RMSE": .67, "vy_RMSE": .40, "yaw_rate_RMSE": .20, "vx_max": 2., "vy_max": 1.2, "yaw_rate_max": .355}
    for row, suffix in enumerate(("RMSE", "max")):
        for col, state in enumerate(STATES):
            ax = axes[row, col]
            key = f"{state}_{suffix}"
            values = data[key]
            if values.max() > upper[key]:
                raise ValueError(f"Out-of-range value in {key}")
            ax.set_axisbelow(True)
            ax.grid(axis="y", color=GRID, linewidth=.48)
            ax.set_xlim(-.22, 2.22)
            ax.set_ylim(0., upper[key])
            ax.set_yticks(ticks[key])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
            ax.tick_params(axis="y", length=0, pad=3, labelsize=8.0)
            ax.tick_params(axis="x", length=0, pad=5, labelsize=8.0)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_linewidth(.60)
            ax.set_xticks([0, 1, 2], ["Data-only\nMLP", "ODE-PINN", "Ref.\nPGNN"])
            if not row:
                ax.text(.5, 1.157, titles[col], transform=ax.transAxes, ha="center",
                        va="center", fontsize=9.45, fontweight="bold")
                ax.text(.5, 1.052, units[col], transform=ax.transAxes, ha="center",
                        va="center", fontsize=8.35, color="#4E5D6C")
            if not col:
                ax.set_ylabel(("RMSE", "MaxAE")[row], fontsize=9.4, fontweight="bold", labelpad=4)
            for drive, triplet in zip(RECORDS, values):
                label, color, marker, dash = DRIVE_STYLES[drive]
                track, = ax.plot(range(3), triplet, label=label, color=color,
                                 linewidth=1.20, linestyle=dash, marker=marker,
                                 markersize=4.45, markeredgecolor="white", markeredgewidth=.40,
                                 solid_capstyle="round", dash_capstyle="round", zorder=2)
                track.set_gid(f"{key}-{drive}")
            average, = ax.plot(range(3), values.mean(axis=0), color=MEAN, linewidth=1.95,
                               label="Drive mean", marker="o", markersize=4.8,
                               markerfacecolor="white", markeredgewidth=1.25, zorder=3)
            average.set_gid(f"{key}-mean")
    # Use the plotted artists themselves as legend keys. Matplotlib fills
    # columns first, so arrange them for left-to-right reading in two rows.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = (0, 3, 1, 4, 2, 5, 6)
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center",
               bbox_to_anchor=(.516, .016), ncol=4, frameon=False, handlelength=2.2,
               columnspacing=1.24, handletextpad=.48, labelspacing=.66,
               borderaxespad=0, fontsize=8.1)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(Path.cwd() / f"Fig6.{ext}", dpi=600, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    plot_lmsd_pairs()
