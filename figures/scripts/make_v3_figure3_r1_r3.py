"""Figure 3 (v3): 14 day rolling trajectories in Q2 2021 for R1 and R3 only.

Data: paper_q2_results/comparisons/baseline_excess_2021Q2_vs_2021Q1.csv (saved pipeline output).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
d = pd.read_csv(ROOT / "paper_q2_results/comparisons/baseline_excess_2021Q2_vs_2021Q1.csv", parse_dates=["Window_End"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.linewidth": 0.8})
series = {1: ("R1: pickle / dtype (chronic plateau)", "#1f77b4"), 3: ("R3: task_cast (May sprint)", "#9467bd")}
fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.4, 4.25), sharex=True, gridspec_kw={"hspace": 0.12})
for cid, (lab, col) in series.items():
    s = d[d.Cluster_ID == cid].sort_values("Window_End")
    a1.plot(s.Window_End, s.Target_Prevalence * 100, color=col, lw=2.0, label=lab)
    a1.axhline(s.Baseline_Prevalence.iloc[0] * 100, color=col, lw=1.0, ls=":")
    a2.plot(s.Window_End, s.Excess_PP, color=col, lw=2.0)
r1 = d[d.Cluster_ID == 1]; r3 = d[d.Cluster_ID == 3]
share_above = (r1.Excess_PP > 0).mean() * 100
pk = r3.loc[r3.Excess_PP.idxmax()]
a1.set_ylabel("Q2 share (%)", fontweight="bold")
a1.plot([], [], color="#666666", lw=1.0, ls=":", label="Q1 baseline share")
a1.legend(loc="upper right", fontsize=7.8, frameon=True, framealpha=0.95)
a1.set_ylim(-2, 72)
a2.axhline(0, color="black", lw=0.9, ls="--")
a2.set_ylabel("Excess vs. Q1 (pp)", fontweight="bold")
a2.set_ylim(-4, 26)
a2.annotate(f"R1 above its Q1 level on {share_above:.1f}% of Q2 days", xy=(0.30, 0.97), xycoords="axes fraction",
            fontsize=7.8, color="#1f77b4", fontweight="bold", va="top")
a2.annotate(f"R3 peak: +{pk.Excess_PP:.1f} pp ({pk.Window_End:%b} {pk.Window_End.day})",
            xy=(pk.Window_End, pk.Excess_PP), xytext=(pk.Window_End + pd.Timedelta(days=8), pk.Excess_PP + 3.5),
            fontsize=7.8, color="#9467bd", fontweight="bold", arrowprops=dict(arrowstyle="->", color="#9467bd", lw=0.9))
a2.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15]))
a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
a2.set_xlabel("2021 Q2 daily window end (14 day trailing window)", fontweight="bold")
for a in (a1, a2):
    a.grid(True, color="#dddddd", lw=0.6)
    a.set_axisbelow(True)
out = ROOT / "figures/v3_figure3_r1_r3_rolling.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
print(out, f"R1 above {share_above:.1f}%", f"R3 peak {pk.Excess_PP:.2f} on {pk.Window_End.date()}")
