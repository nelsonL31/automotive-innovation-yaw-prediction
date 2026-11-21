"""Figure 7. Paired changes in one-step and 15-step errors on LVMS.

Run beside racing_errors.csv with Python 3, NumPy, pandas and Matplotlib.
The two errors for each method and its reproduced PGNN comparator use
the same training/test pair and are averaged over the same three seeds.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIRECTORY = Path(__file__).resolve().parent
RUNS = ("run4_2", "run2_1", "run4_1")
# Descriptive name, point label, colour, symbol.
MODELS = {
    "ode": ("ODE-PINN", "ODE", "#007D79", "o"),
    "pde": ("PDE-PINN", "PDE", "#714BB9", "s"),
    "joint": ("ODE–PDE-PINN", "Joint", "#BA4C16", "^"),
}
TESTS = ("A", "B")
AXIS_METRICS = ("yaw_rate_RMSE", "FDE_m")
INK, MUTED = "#192D3A", "#536775"
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8.5, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": "#9AAAB3", "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": .65, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.size": 2.8, "ytick.major.size": 2.8, "xtick.major.width": .6,
    "ytick.major.width": .6, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
}


def get_pairs():
    data = pd.read_csv(DIRECTORY / "racing_errors.csv")
    data = data.loc[data.metric.isin(AXIS_METRICS)].copy()
    if not (data.baseline_error > 0).all():
        raise ValueError("The reproduced PGNN error must be positive.")
    calculated = 100 * (data.method_error / data.baseline_error - 1)
    np.testing.assert_allclose(calculated, data.change_percent, rtol=1e-10, atol=1e-10)
    data["run"] = data.source.map({f"Putnam_park2023_{run}.csv": run for run in RUNS})
    data["test_id"] = data.test.map({f"LVMS_23_01_04_{s}.csv": s for s in TESTS})
    pairs = data.pivot(index=["run", "arm", "test_id"], columns="metric", values="change_percent")
    expected = pd.MultiIndex.from_product([RUNS, MODELS, TESTS], names=pairs.index.names)
    if len(pairs) != len(expected) or not pairs.index.sort_values().equals(expected.sort_values()):
        raise ValueError("Expected 18 paired observations across three runs, three models and two tests.")
    pairs = pairs.reindex(index=expected, columns=AXIS_METRICS)
    if not np.isfinite(pairs.to_numpy()).all():
        raise ValueError("Incomplete or non-finite coordinates.")
    return pairs


def point_style(arm, record):
    _, _, color, shape = MODELS[arm]
    return dict(ls="none", marker=shape, ms=5.7, mec=color,
                mfc="white" if record=="A" else color, mew=1.15)


# These offsets refer to the side of each real point. No plotted position is jittered.
PLACEMENT = {
    "run4_2": {"ode": ("n", "s"), "pde": ("s", "ne"), "joint": ("e", "e-low")},
    "run2_1": {"ode": ("s", "e"), "pde": ("e", "n"), "joint": ("n", "ne")},
    "run4_1": {"ode": ("n", "se"), "pde": ("s", "n"), "joint": ("nw", "s-left")},
}
POSITIONS = {
    "n": (0, 5.5, "center", "bottom"), "s": (0,-5.5,"center","top"),
    "e": (5.4,0,"left","center"), "w": (-5.4,0,"right","center"),
    "ne": (3.5,3.6,"left","bottom"), "nw": (-3.5,3.6,"right","bottom"),
    "se": (3.5,-3.6,"left","top"), "sw": (-3.5,-3.6,"right","top"),
    "e-low": (5.6,-2.0,"left","center"),
    "s-left": (-3.0,-5.6,"center","top"),
}


def fde_label(short, value):
    # Two decimals retain the sign and magnitude of the two near-zero changes.
    magnitude = f"{value:+.2f}" if abs(value) < 1 else f"{value:+.1f}"
    return short + " " + magnitude.replace("-", "−") + "%"


def add_key(fig):
    # Both marker fills in each key are reused verbatim from the data.
    # The first symbol corresponds to the upper row, the second to the lower.
    key = fig.add_axes([.11, .015, .86, .077])
    key.set_xlim(0,1)
    key.set_ylim(0,1)
    key.set_axis_off()
    for x,(arm,(full,_,_,_)) in zip((.010,.324,.641),MODELS.items()):
        for dx,record in [(0, "A"),(.027,"B")]:
            mark,=key.plot(x+dx,.56,**point_style(arm,record))
            mark.set_gid(f"legend-{arm}-{record}")
        key.text(x+.048,.56,full,va="center",fontsize=8.55,color=INK)


def make_figure():
    pairs=get_pairs()
    with plt.rc_context(STYLE):
        fig, axes=plt.subplots(2,3,figsize=(6.35,4.94),sharex=True,sharey=True)
        fig.subplots_adjust(left=.125,right=.980,top=.903,bottom=.214,hspace=.245,wspace=.148)
        for i,run in enumerate(RUNS):
            top=axes[0,i]
            top.set_title(f"({'abc'[i]})  Putnam {run}", loc="left", fontsize=9.25,
                          weight="bold", pad=8)
            for j,record in enumerate(TESTS):
                ax=axes[j,i]
                ax.set_xlim(-108,0)
                ax.set_ylim(-72,84)
                ax.set_xticks([-100,-75,-50,-25,0])
                ax.set_yticks([-50,0,50])
                ax.tick_params(axis="both",labelsize=7.75,pad=2.3)
                ax.spines.left.set_visible(i==0)
                ax.spines.bottom.set_visible(j==1)
                ax.tick_params(axis="y",left=i==0)
                ax.tick_params(axis="x",bottom=j==1)
                ax.grid(axis="y",color="#DEE7EB",lw=.58)
                ax.set_axisbelow(True)
                ax.axhline(0,lw=.90,color="#7F929C")
                for arm,(_,short,color,_) in MODELS.items():
                    x,y=pairs.loc[(run,arm,record)]
                    mark,=ax.plot(x,y,zorder=4,**point_style(arm,record))
                    mark.set_gid(f"point-{run}-{arm}-{record}")
                    dx,dy,ha,va=POSITIONS[PLACEMENT[run][arm][j]]
                    text = fde_label(short,y)
                    if (run, arm, record) == ("run4_2", "joint", "A"):
                        text = text.replace(" ", "\n")
                    lbl=ax.annotate(text,(x,y),xytext=(dx,dy),
                                    textcoords="offset points",ha=ha,va=va,
                                    color=color,fontsize=7.65,fontweight="medium",zorder=5,
                                    bbox=dict(boxstyle="square,pad=.018",ec="none",fc="white"))
                    lbl.set_gid(f"label-{run}-{arm}-{record}")
        for j, record in enumerate(TESTS):
            ext=axes[j,0].get_position()
            fig.text(.019,(ext.y0+ext.y1)/2,f"LVMS-{record}",ha="left",va="center",
                     rotation=90,fontsize=9.6,fontweight="bold")
        fig.text(.125,.984,"15-step FDE change (%)",fontsize=8.8,ha="left",va="top",color=MUTED)
        fig.text(.55,.116,"One-step yaw-rate RMSE change (%)",ha="center",fontsize=8.9)
        add_key(fig)
        return fig


def plot_track():
    fig=make_figure()
    with plt.rc_context(STYLE):
        for ext in ("png","svg","pdf"):
            fig.savefig(Path.cwd() / f"Fig7.{ext}", dpi=600, facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    plot_track()
