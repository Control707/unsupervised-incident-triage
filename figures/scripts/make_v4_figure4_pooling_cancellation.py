"""Figure 4 (v4): pooling cancellation on the synthetic IT benchmark and the monthly test that fixes it.

Data: final_experiments/exp8_it_coarse_vs_monthly_shifts.csv (saved pipeline output).
(a) monthly shares of the six clusters shown in the original Figure 4;
(b) coarse shift (Jan vs. Feb+Mar pooled) next to the monthly shift (Mar vs. Jan+Feb) for the four clusters that
    the monthly test flags at q < 0.10.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
e = pd.read_csv(ROOT / "final_experiments/exp8_it_coarse_vs_monthly_shifts.csv").set_index("Cluster_ID")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.8})
names = {"IT-03": "IT03 Install", "IT-10": "IT10 RSA/Java", "IT-05": "IT05 File Crash",
         "IT-07": "IT07 OneDrive", "IT-02": "IT02 DevOps", "IT-08": "IT08 F5 BIG-IP"}
style = {"IT-03": ("#d62728", "o", "-", 2.4), "IT-10": ("#1f77b4", "s", "--", 1.6), "IT-05": ("#e377c2", "^", "-.", 1.4),
         "IT-07": ("#2ca02c", "v", "--", 1.4), "IT-02": ("#9467bd", "D", "-", 1.6), "IT-08": ("#333333", "P", ":", 1.6)}
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.5})
x = [0, 1, 2]
for cid, (col, mk, ls, lw) in style.items():
    r = e.loc[cid]
    sig = "*" if r.Monthly_Q_BH_FDR < 0.10 else ""
    a.plot(x, [r.P_Jan_Pct, r.P_Feb_Pct, r.P_Mar_Pct], color=col, marker=mk, ls=ls, lw=lw, ms=5,
           label=f"{names[cid]}{sig}")
r3 = e.loc["IT-03"]
pooled = (r3.Count_Feb + r3.Count_Mar) / (851) * 100
a.plot([1, 2], [pooled, pooled], color="#d62728", lw=1.2, ls=(0, (2, 2)))
a.annotate(f"Feb and Mar pooled: {pooled:.1f}%\nvs. Jan {r3.P_Jan_Pct:.1f}% ({r3.Coarse_Delta_FebMar_vs_Jan_PP:+.2f} pp)",
           xy=(1.5, pooled), xytext=(0.05, 25.0), fontsize=7.4, color="#b11d1d", fontweight="bold",
           arrowprops=dict(arrowstyle="->", color="#b11d1d", lw=0.9))
a.set_xticks(x, ["Jan", "Feb", "Mar"])
a.set_ylabel("Monthly share of tickets (%)", fontweight="bold")
a.set_ylim(-0.5, 29)
a.set_title("(a) Pooling cancellation (IT03)", fontsize=9.5, fontweight="bold")
a.legend(fontsize=6.6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False,
         handlelength=2.2, columnspacing=1.0, title="* = significant March shift (q < 0.10)", title_fontsize=6.6)
flag = ["IT-03", "IT-02", "IT-08", "IT-10"]
y = np.arange(len(flag))[::-1]
coarse = [e.loc[c, "Coarse_Delta_FebMar_vs_Jan_PP"] for c in flag]
monthly = [e.loc[c, "Monthly_Delta_Mar_vs_JanFeb_PP"] for c in flag]
lo = [e.loc[c, "Monthly_CI95_Low_PP"] for c in flag]
hi = [e.loc[c, "Monthly_CI95_High_PP"] for c in flag]
h = 0.36
b.barh(y + h / 2, coarse, h, color="#bdbdbd", edgecolor="#555555", lw=0.6, label="Coarse: Jan vs. Feb+Mar (none significant)")
b.barh(y - h / 2, monthly, h, color=["#d62728" if m > 0 else "#1f77b4" for m in monthly], edgecolor="#333333", lw=0.6,
       label="Monthly: Mar vs. Jan+Feb ($q<0.10$)")
b.errorbar(monthly, y - h / 2, xerr=[np.subtract(monthly, lo), np.subtract(hi, monthly)], fmt="none", ecolor="#222222", lw=0.8, capsize=2)
for yy, c, m, hh, ll in zip(y, flag, monthly, hi, lo):
    b.text(hh + 0.6 if m > 0 else 0.6, yy - h / 2, f"q = {e.loc[c, 'Monthly_Q_BH_FDR']:.3f}", fontsize=6.8, va="center", ha="left")
b.axvline(0, color="black", lw=0.8)
b.set_yticks(y, [names[c].replace(" ", "\n", 1) for c in flag])
b.set_xlim(-9.5, 17.5)
b.set_xlabel("Shift in share (pp), 95% CI", fontweight="bold")
b.set_title("(b) Monthly test reveals the shifts", fontsize=9.5, fontweight="bold")
b.legend(fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
for ax in (a, b):
    ax.grid(True, color="#e2e2e2", lw=0.6); ax.set_axisbelow(True)
out = ROOT / "figures/v4_figure4_pooling_cancellation.pdf"
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
print(out, "pooled", round(pooled, 2))
