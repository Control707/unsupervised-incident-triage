"""Rebuild poster-sized temporal figures from the original saved project tables.

Run from the repository root with .venv-q2/bin/python poster_work/redraw_temporal.py.
Only the six exports in poster_work/hires_figures/ are written. No raster input is used.
Physical canvases are specified in poster points, giving the exact requested ratios.
"""
from pathlib import Path
import hashlib
import json
import tempfile
import os

# Keep matplotlib's incidental cache outside the project; install nothing.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "poster-temporal-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "poster_work/hires_figures"
ROLLING_SOURCE = ROOT / "paper_q2_results/comparisons/baseline_excess_2021Q2_vs_2021Q1.csv"
POOLING_SOURCE = ROOT / "final_experiments/exp8_it_coarse_vs_monthly_shifts.csv"
DPI = 240
BLUE, PURPLE, RED, BLACK = "#1768ac", "#7b4397", "#c62828", "#171717"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 24,
    "axes.labelsize": 25, "axes.titlesize": 27, "axes.titleweight": "bold",
    "axes.labelcolor": BLACK, "text.color": BLACK, "xtick.color": BLACK,
    "ytick.color": BLACK, "xtick.labelsize": 22, "ytick.labelsize": 22,
    "axes.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "savefig.facecolor": "white",
    "figure.facecolor": "white", "savefig.bbox": None,
})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(ax, axis="both"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis=axis, color="#dedede", lw=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(length=6, pad=8)


def save(fig, stem, canvas_pt):
    """Never use tight bounding-box export: page size is the panel contract."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Every visible text object (including legends) must fit the fixed page.
    from matplotlib.text import Text
    bounds = fig.bbox
    for text in fig.findobj(match=Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1, (stem, text.get_text(), box)
            assert box.x1 <= bounds.x1 + 1 and box.y1 <= bounds.y1 + 1, (stem, text.get_text(), box)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=DPI, bbox_inches=None,
                    metadata={"Creator": "Matplotlib; redraw from saved project data"}
                    if extension != "png" else {"Software": "Matplotlib; redraw from saved project data"})
    with Image.open(OUT / f"{stem}.png") as image:
        pixels = image.size
    result = {"canvas_pt": canvas_pt, "canvas_in": list(fig.get_size_inches()),
              "target_aspect": canvas_pt[0] / canvas_pt[1], "png_pixels": pixels,
              "png_dpi": DPI, "files": [str(OUT / f"{stem}.{ext}") for ext in ("png", "pdf", "svg")]}
    plt.close(fig)
    return result


def rolling(d):
    fig = plt.figure(figsize=(1032 / 72, 780 / 72), dpi=100)
    top = fig.add_axes([0.115, 0.53, 0.855, 0.30])
    bottom = fig.add_axes([0.115, 0.135, 0.855, 0.30], sharex=top)
    fig.text(0.115, 0.958, "14-day rolling trajectories", fontsize=28, fontweight="bold", va="top")
    handles = []
    counts = {}
    for cid, label, color in ((1, "R1: pickle / dtype", BLUE), (3, "R3: task_cast", PURPLE)):
        s = d[d.Cluster_ID == cid].sort_values("Window_End")
        line, = top.plot(s.Window_End, s.Target_Prevalence * 100, color=color, lw=3.1)
        baseline = s.Baseline_Prevalence.iloc[0] * 100
        top.axhline(baseline, color=color, lw=2, ls=(0, (1.5, 2.5)))
        excess, = bottom.plot(s.Window_End, s.Excess_PP, color=color, lw=3.1)
        # Exact saved values, no smoothing/resampling/refitting.
        np.testing.assert_array_equal(line.get_ydata(), (s.Target_Prevalence * 100).to_numpy())
        np.testing.assert_array_equal(excess.get_ydata(), s.Excess_PP.to_numpy())
        np.testing.assert_array_equal(mdates.date2num(line.get_xdata()), mdates.date2num(s.Window_End))
        handles.append(Line2D([], [], color=color, lw=3.1, label=label))
        counts[str(cid)] = len(s)
    handles.append(Line2D([], [], color=BLACK, lw=2, ls=(0, (1.5, 2.5)), label="Q1 baseline"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.103, 0.922),
               ncol=3, fontsize=23, frameon=False, columnspacing=1.5, handlelength=2)
    top.set_ylabel("Q2 share (%)", labelpad=12)
    top.set_ylim(-2, 60)
    top.set_yticks([0, 20, 40, 60])
    top.tick_params(labelbottom=False)
    bottom.set_ylabel("Excess vs. Q1 (pp)", labelpad=12)
    bottom.set_ylim(-4, 27)
    bottom.set_yticks([0, 10, 20])
    bottom.axhline(0, color=BLACK, lw=1.4, ls="--")
    r1 = d[d.Cluster_ID == 1]
    r3 = d[d.Cluster_ID == 3]
    above = (r1.Excess_PP > 0).mean() * 100
    peak = r3.loc[r3.Excess_PP.idxmax()]
    fig.text(0.115, 0.466, f"R1 above Q1 level on {above:.1f}% of Q2 days", fontsize=23,
             fontweight="bold", color=BLUE, va="center")
    bottom.annotate(f"R3 peak: +{peak.Excess_PP:.1f} pp\n{peak.Window_End:%b} {peak.Window_End.day}",
                    xy=(peak.Window_End, peak.Excess_PP),
                    xytext=(pd.Timestamp("2021-06-03"), 23.3),
                    fontsize=22, color=PURPLE, fontweight="bold", va="center",
                    arrowprops={"arrowstyle": "->", "color": PURPLE, "lw": 1.8},
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    bottom.set_xlim(pd.Timestamp("2021-03-30"), pd.Timestamp("2021-07-02"))
    bottom.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15]))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    bottom.set_xlabel("2021 Q2 window end · 14-day trailing window", labelpad=15, fontsize=24)
    for ax in (top, bottom):
        clean(ax)
    result = save(fig, "rolling", [1032, 780])
    result["verification"] = {"points_per_series_per_panel": counts,
                              "exact_artist_array_checks": "passed",
                              "R1_days_above_Q1_pct": above,
                              "R3_peak_excess_pp": float(peak.Excess_PP),
                              "R3_peak_date": str(peak.Window_End.date())}
    return result


def pooling(e):
    fig = plt.figure(figsize=(1416 / 72, 760 / 72), dpi=100)
    a = fig.add_axes([0.068, 0.295, 0.375, 0.585])
    b = fig.add_axes([0.605, 0.295, 0.355, 0.585])
    names = {"IT-03": "IT03 Install", "IT-10": "IT10 RSA/Java", "IT-05": "IT05 File Crash",
             "IT-07": "IT07 OneDrive", "IT-02": "IT02 DevOps", "IT-08": "IT08 F5 BIG-IP"}
    # All six original series; distinct markers and dashes make colors nonessential.
    styles = {"IT-03": (RED, "o", "-"), "IT-10": (BLUE, "s", "--"),
              "IT-05": ("#a55387", "^", "-."), "IT-07": ("#454545", "v", "--"),
              "IT-02": (PURPLE, "D", "-"), "IT-08": (BLACK, "P", ":")}
    handles = []
    for cid, (color, marker, linestyle) in styles.items():
        row = e.loc[cid]
        values = row[["P_Jan_Pct", "P_Feb_Pct", "P_Mar_Pct"]].to_numpy(dtype=float)
        sig = "*" if row.Monthly_Q_BH_FDR < .10 else ""
        line, = a.plot([0, 1, 2], values, color=color, marker=marker, ls=linestyle,
                       lw=3 if cid == "IT-03" else 2.4, ms=10, label=names[cid] + sig)
        np.testing.assert_array_equal(line.get_ydata(), values)
        handles.append(line)
    r = e.loc["IT-03"]
    pooled = (r.Count_Feb + r.Count_Mar) / 851 * 100  # Same saved denominator as original.
    a.plot([1, 2], [pooled, pooled], color=RED, lw=2.5, ls=(0, (2, 2)))
    a.annotate(f"IT03 Feb+Mar pooled: {pooled:.1f}%\nJan: {r.P_Jan_Pct:.1f}%  ({r.Coarse_Delta_FebMar_vs_Jan_PP:+.2f} pp)",
               xy=(1.48, pooled), xytext=(-0.02, 27.0), fontsize=23,
               color=RED, fontweight="bold", va="top",
               arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.8,
                           "connectionstyle": "arc3,rad=-0.10"})
    a.set_xlim(-.12, 2.12)
    a.set_ylim(-.5, 29)
    a.set_yticks([0, 5, 10, 15, 20, 25])
    a.set_xticks([0, 1, 2], ["Jan", "Feb", "Mar"])
    a.set_ylabel("Monthly share of tickets (%)", labelpad=12)
    a.set_title("(a) Pooling cancellation", loc="left", pad=24)
    # Fixed footer regions leave chart area unobstructed by either legend.
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.061, .225),
               ncol=2, frameon=False, fontsize=22, columnspacing=1.2,
               handlelength=2.3, handletextpad=.6, labelspacing=.6)
    fig.text(.068, .045, "* March shift: BH-FDR q < 0.10", fontsize=22)
    flag = ["IT-03", "IT-02", "IT-08", "IT-10"]
    y = np.arange(len(flag))[::-1]
    coarse = e.loc[flag, "Coarse_Delta_FebMar_vs_Jan_PP"].to_numpy()
    monthly = e.loc[flag, "Monthly_Delta_Mar_vs_JanFeb_PP"].to_numpy()
    lo = e.loc[flag, "Monthly_CI95_Low_PP"].to_numpy()
    hi = e.loc[flag, "Monthly_CI95_High_PP"].to_numpy()
    h = .31
    c_bars = b.barh(y + h / 2, coarse, h, color="#c4c4c4", edgecolor="#555555", lw=1.0)
    m_bars = b.barh(y - h / 2, monthly, h,
                    color=[RED if m > 0 else BLUE for m in monthly], edgecolor=BLACK, lw=1.0)
    errors = b.errorbar(monthly, y - h / 2, xerr=[monthly - lo, hi - monthly],
                       fmt="none", ecolor=BLACK, lw=1.8, capsize=6, capthick=1.8)
    np.testing.assert_array_equal([bar.get_width() for bar in c_bars], coarse)
    np.testing.assert_array_equal([bar.get_width() for bar in m_bars], monthly)
    for segment, low, high in zip(errors.lines[2][0].get_segments(), lo, hi):
        np.testing.assert_allclose(segment[:, 0], [low, high], rtol=0, atol=1e-14)
    # A dedicated q-value column, not text over bars or CI whiskers.
    for yy, cid in zip(y, flag):
        b.text(16, yy - h / 2, f"{e.loc[cid, 'Monthly_Q_BH_FDR']:.3f}",
               fontsize=22, ha="center", va="center")
    b.text(16, 3.64, "FDR q", fontsize=22, ha="center", fontweight="bold")
    b.axvline(0, color=BLACK, lw=1.3)
    b.set_ylim(-.65, 3.92)
    b.set_xlim(-9.5, 20.2)
    b.set_xticks([-8, -4, 0, 4, 8, 12])
    b.set_yticks(y, [names[c].replace(" ", "\n", 1) for c in flag])
    b.tick_params(axis="y", length=0, pad=12)
    b.set_xlabel("Shift in share (pp) · 95% CI", labelpad=15, fontsize=24)
    b.set_title("(b) Monthly test reveals shifts", loc="left", pad=24)
    clean(a, "y")
    clean(b, "x")
    # Fixed-position legend rows have explicit clearance from the axis label and footer.
    for yy, color, text in [(.175, "#c4c4c4", "Coarse: (Feb+Mar) − Jan; none significant"),
                             (.120, RED, "Monthly: Mar − (Jan+Feb); q < 0.10")]:
        fig.add_artist(Rectangle((.605, yy - .010), .026, .020,
                                 transform=fig.transFigure, facecolor=color,
                                 edgecolor=BLACK, lw=1.0))
        fig.text(.644, yy, text, fontsize=22, va="center")
    fig.text(.605, .055, "Whiskers: monthly 95% CI · q: BH-FDR", fontsize=22, va="center")
    result = save(fig, "pooling", [1416, 760])
    result["verification"] = {"monthly_series": list(styles), "monthly_points": 18,
                              "bar_clusters": flag, "coarse_bars": coarse.tolist(),
                              "monthly_bars": monthly.tolist(), "CI95_low": lo.tolist(),
                              "CI95_high": hi.tolist(),
                              "FDR_q_unrounded": e.loc[flag, "Monthly_Q_BH_FDR"].tolist(),
                              "IT03_pooled_pct": pooled,
                              "exact_artist_array_checks": "passed; CI endpoints tolerance 1e-14"}
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [ROLLING_SOURCE, POOLING_SOURCE,
              ROOT / "figures/scripts/make_v3_figure3_r1_r3.py",
              ROOT / "figures/scripts/make_v4_figure4_pooling_cancellation.py"]
    before = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    d = pd.read_csv(ROLLING_SOURCE, parse_dates=["Window_End"])
    e = pd.read_csv(POOLING_SOURCE).set_index("Cluster_ID")
    report = {"rolling": rolling(d), "pooling": pooling(e)}
    after = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    assert before == after, "An original input changed during redraw"
    report["source_sha256_unchanged"] = after
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
