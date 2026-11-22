"""Draw Figure 4 in English and Chinese from its adjacent source CSV.

Run with Python 3, NumPy, and Matplotlib. The Chinese figure uses Microsoft YaHei.
"""
from argparse import ArgumentParser
from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parent
INK = "#291921"
MUTED = "#594A4A"
GRID = "#EBE1DF"
HOLD_COLOR = "#EEA023"
REPLAY_COLOR = "#BA2855"
STIFFNESS = [0.7, 1.0, 1.3]
RAMP = LinearSegmentedColormap.from_list(
    "rmse_change",
    [(0.00, "#642042"), (0.09, "#9F2855"), (0.18, "#D43B4A"),
     (0.34, "#E9563E"), (0.54, "#F47B35"), (0.645, "#F69836"),
     (0.765, "#FFCA4E"), (0.90, "#FFE495"), (1.00, "#FFF4D5")],
)
NORM = Normalize(-90, 0)

def read_results():
    with (ROOT / "tyre_simulation.csv").open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    items = {(r["tyre"], float(r["front"]), float(r["rear"]), r["mode"]): r for r in rows}
    assert len(rows) == len(items) == 72
    assert all(int(row["paired_phases"]) == 24 for row in rows)
    for row in rows:
        changed = 100 * (float(row["ode"]) / float(row["data"]) - 1)
        np.testing.assert_allclose(changed, float(row["ODE_vs_data_percent"]), atol=1e-11)
    return items

def luminance(color):
    rgb = np.asarray(to_rgb(color))
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055)**2.4)
    return np.dot(linear, [.2126, .7152, .0722])

def set_style(chinese):
    plt.rcParams.update({
        "font.family":"sans-serif",
        "font.sans-serif":["Microsoft YaHei" if chinese else "Arial","DejaVu Sans"],
        "font.size":8.8, "axes.labelsize":9.0, "xtick.labelsize":8.4,
        "ytick.labelsize":8.7, "text.color":INK, "axes.labelcolor":INK,
        "xtick.color":MUTED, "ytick.color":MUTED, "axes.edgecolor":"#A39A8F",
        "axes.linewidth":.58, "hatch.linewidth":.45,
        "svg.fonttype":"none", "pdf.fonttype":42, "axes.unicode_minus":True,
    })

def heading(fig, y, label):
    fig.text(.028, y, label, fontsize=10.7, fontweight="bold", ha="left", va="top")

def plot_absolute(fig, items, zh):
    heading(fig,.970,
            "(a)  标称刚度线性轮胎的预测误差" if zh else "(a)  Prediction error at nominal linear-tyre stiffness")
    keys = [Patch(facecolor=HOLD_COLOR,linewidth=0),
            Patch(facecolor=REPLAY_COLOR,edgecolor="white",linewidth=.2,hatch="///")]
    fig.legend(keys, ["转角保持","转角回放"] if zh else ["Steering held","Steering replayed"],
               frameon=False,ncol=2,loc="upper right",bbox_to_anchor=(.976,.975),
               fontsize=8.6,handlelength=1.35,handleheight=.78,columnspacing=1.0,handletextpad=.45)
    modes = [("hold","steer_only"),("speed_only","replay")]
    group_names = ["车速保持","车速回放"] if zh else ["Speed held","Speed replayed"]
    names = ["无物理约束","ODE-PINN","ARX"] if zh else ["Data only","ODE-PINN","ARX"]
    for j, (s0,s1) in enumerate(modes):
        x0 = [.128,.627][j]
        ax = fig.add_axes((x0,.631,.344,.228))
        ax.set_xlim(0,.20); ax.set_ylim(-.48,2.48)
        ax.set_xticks([0,.05,.1,.15,.2],["0","0.05","0.10","0.15","0.20"])
        ax.set_yticks([2,1,0],names)
        ax.grid(axis="x",color=GRID,lw=.55); ax.set_axisbelow(True)
        ax.spines[["left","right","top"]].set_visible(False)
        ax.tick_params(axis="y",length=0,pad=4.0)
        ax.tick_params(axis="x",length=2.2,pad=3.0)
        ax.text(0,1.095,group_names[j],ha="left",va="center",transform=ax.transAxes,
                fontsize=9.7,color=INK,fontweight="bold")
        for k,method in enumerate(["data","ode","arx"]):
            y0=2-k
            for mode,delta,color,hatch in [(s0,.195,HOLD_COLOR,None),(s1,-.195,REPLAY_COLOR,"///")]:
                v=float(items["linear_slip",1.,1.,mode][method])
                ax.barh(y0+delta,v,height=.31,color=color,edgecolor="white",
                        linewidth=.25,hatch=hatch,zorder=2)
                ax.annotate(f"{v:.4f}",(v,y0+delta),xytext=(2.5,0),
                            textcoords="offset points",va="center",ha="left",fontsize=8.1,color=INK)
    fig.text(.546,.548,"0.6 s 横摆角速度 RMSE（rad/s）" if zh else "0.6 s yaw-rate RMSE (rad/s)",
             ha="center",va="center",fontsize=9.3)

def make_tile(fig, rect, items, tyre, mode, y_ticks):
    ax = fig.add_axes(rect)
    values=np.array([[float(items[tyre,f,r,mode]["ODE_vs_data_percent"])
                      for r in STIFFNESS] for f in STIFFNESS])
    assert np.all((values > -90) & (values < 0))
    ax.imshow(values,origin="lower",cmap=RAMP,norm=NORM,aspect="auto")
    ax.set_xticks(range(3),[f"{f:.1f}" for f in STIFFNESS])
    ax.set_yticks(range(3),[f"{f:.1f}" for f in STIFFNESS] if y_ticks else [""]*3)
    ax.tick_params(axis="both",length=0,pad=3.0,labelsize=8.1)
    ax.spines[:].set_visible(False)
    for boundary in [.5,1.5]:
        ax.axvline(boundary,color="white",lw=.8); ax.axhline(boundary,color="white",lw=.8)
    for row,col in np.ndindex(values.shape):
        value=values[row,col]
        lum=luminance(RAMP(NORM(value)))
        textcolor = "white" if 1.05/(lum+.05) > (lum+.05)/(luminance(INK)+.05) else INK
        ax.text(col,row,f"{value:.1f}".replace("-","−"),ha="center",va="center",
                fontsize=8.7,weight="bold" if row==col==1 else "normal",color=textcolor)
        if row==col==1:
            ax.add_patch(Rectangle((.53,.53),.94,.94,fill=False,edgecolor=textcolor,lw=.9))

def plot_stiffness(fig, items, zh):
    heading(fig,.477,
            "(b)  ODE-PINN 对前后轴刚度变化的敏感性" if zh else "(b)  ODE-PINN sensitivity to axle stiffness")
    fig.text(.973,.457, "刚度倍率：行 = 前轴，列 = 后轴" if zh else
             "Stiffness multipliers: front axle in rows, rear axle in columns",
             ha="right",va="center",fontsize=8.1,color=MUTED)
    xs = [.068,.290,.554,.776]
    w=.197
    for j,tyre in enumerate(["linear_slip","saturated"]):
        name=(["线性轮胎","饱和轮胎"] if zh else ["Linear tyres","Saturating tyres"])[j]
        start=xs[j*2]; end=xs[j*2+1]+w
        fig.text((start+end)/2,.382,name,ha="center",va="center",fontsize=9.6,color=INK,fontweight="bold")
        # A short rule on each side groups the two input conditions for one tyre law.
        fig.add_artist(plt.Line2D([start,start+.102],[.382,.382],transform=fig.transFigure,color=GRID,lw=.9))
        fig.add_artist(plt.Line2D([end-.102,end],[.382,.382],transform=fig.transFigure,color=GRID,lw=.9))
        for m,mode in enumerate(["hold","replay"]):
            x=xs[2*j+m]
            mode_name=(["车速与转角均保持","车速与转角均回放"] if zh else ["Hold both inputs","Replay both inputs"])[m]
            fig.text(x+w/2,.328,mode_name,ha="center",va="center",fontsize=8.4,color=MUTED)
            make_tile(fig,(x,.139,w,.156),items,tyre,mode,y_ticks=(m==0))
    label=("ODE-PINN 相对无物理约束网络的 RMSE 变化（%）" if zh else
           "ODE-PINN RMSE change relative to data only (%)")
    fig.text(.032,.051,label,ha="left",va="center",fontsize=8.4,color=INK)
    cax=fig.add_axes((.514,.057,.455,.017))
    cb=fig.colorbar(ScalarMappable(norm=NORM,cmap=RAMP),cax=cax,
                    orientation="horizontal",ticks=[-90,-75,-60,-45,-30,-15,0])
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7.9,length=2,pad=2)

def main():
    items=read_results()
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    zh = parser.parse_args().language == "zh"
    set_style(zh)
    fig=plt.figure(figsize=(7.5,5.25),facecolor="white")
    plot_absolute(fig,items,zh)
    plot_stiffness(fig,items,zh)
    for ext in ["png","pdf","svg"]:
        fig.savefig(Path.cwd()/f"Fig4.{ext}", dpi=600 if ext == "png" else 480)
    plt.close(fig)

if __name__=="__main__":
    main()
