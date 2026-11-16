"""Generate section-aligned, high-legibility figures so every panel of every figure
belongs strictly to the section where that figure is placed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
warnings.filterwarnings("ignore")

import hdbscan
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "final_experiments"
FIG_DIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#bbbbbb"
plt.rcParams["axes.linewidth"] = 1.0


def make_section5_spatial_figure():
    """Figure 2 (Section 5): (a) UMAP Dimensionality Sweep, (b) HDBSCAN Mega-Cluster Resolution, (c) 2D UMAP Manifold."""
    dim_df = pd.read_csv(OUT_DIR / "exp3a_umap_dimension_ablation.csv")
    hdb_df = pd.read_csv(OUT_DIR / "exp3c_hdbscan_sensitivity.csv")
    npz = np.load(ROOT / "paper_q2_results" / "joint_embeddings.npz")
    embeddings = npz["embeddings"]
    umap_5d = npz["umap_5d"]

    umap_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.12, metric="cosine", random_state=42).fit_transform(embeddings)
    labels = hdbscan.HDBSCAN(min_cluster_size=16, min_samples=5, max_cluster_size=120, metric="euclidean", cluster_selection_method="eom").fit_predict(umap_5d)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), dpi=300)

    # Panel (a): UMAP Dimensionality Sweep (Section 5.2)
    ax1 = axes[0]
    d_eom = dim_df[dim_df["Method"] == "Bounded-EOM (max=120)"].copy()
    x_labels = [str(d) if d < 384 else "384" for d in d_eom["Dimensions_d"]]
    x_idx = np.arange(len(d_eom))
    ax1.plot(x_idx, d_eom["Sil_Manifold"], marker="o", markersize=6, lw=2.3, color="#0275d8", label="Silhouette (Manifold)")
    ax1.plot(x_idx, d_eom["Topic_Quality"], marker="s", markersize=6, lw=2.3, color="#2ca02c", label="Topic Quality (Cv×TD)")
    ax1.plot(x_idx, d_eom["Noise_Ratio"], marker="^", markersize=6, lw=2.0, linestyle="--", color="#d9534f", label="Noise Ratio")
    ax1.axvline(2, color="#0275d8", linestyle=":", lw=1.6, alpha=0.85, label="Selected d = 5")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(x_labels, fontsize=9.8, fontweight="bold")
    ax1.tick_params(axis="y", labelsize=10.0)
    ax1.set_xlabel("UMAP Ambient Dimension d", fontsize=11.0, fontweight="bold")
    ax1.set_ylabel("Score / Proportion", fontsize=11.0, fontweight="bold")
    ax1.set_title("(a) UMAP Dimensionality Sweep", fontsize=11.8, fontweight="bold")
    ax1.set_ylim(-0.06, 1.06)
    ax1.legend(loc="lower left", ncol=2, fontsize=7.9, columnspacing=0.8, handletextpad=0.4, frameon=True, facecolor="white", framealpha=0.95)

    # Panel (b): HDBSCAN Mega-Cluster Resolution (Unbounded EOM vs Bounded EOM)
    ax2 = axes[1]
    eom_sub = hdb_df[hdb_df["Selection_Method"] == "EOM (Unbounded)"].sort_values("Min_Cluster_Size")
    beom_sub = hdb_df[hdb_df["Selection_Method"] == "Bounded-EOM (max=120)"].sort_values("Min_Cluster_Size")
    mcs_vals = eom_sub["Min_Cluster_Size"].to_numpy()
    x_m = np.arange(len(mcs_vals))

    ax2.plot(x_m, beom_sub["Sil_5D_Manifold"], marker="o", markersize=6, lw=2.4, color="#0275d8", label="5D Sil. (Bounded EOM)")
    ax2.plot(x_m, eom_sub["Sil_5D_Manifold"], marker="o", markersize=5.5, lw=2.0, linestyle="--", color="#5bc0de", label="5D Sil. (Unbounded EOM)")
    ax2.plot(x_m, eom_sub["Largest_Cluster_Share_Total"], marker="s", markersize=6, lw=2.3, linestyle="--", color="#d9534f", label="Max Cluster Share (Unbounded)")
    ax2.plot(x_m, beom_sub["Largest_Cluster_Share_Total"], marker="^", markersize=6, lw=2.3, color="#2ca02c", label="Max Cluster Share (Bounded)")
    ax2.axvline(3, color="#333333", linestyle=":", lw=1.5, alpha=0.85)
    ax2.set_xticks(x_m)
    ax2.set_xticklabels([str(v) for v in mcs_vals], fontsize=9.8, fontweight="bold")
    ax2.tick_params(axis="y", labelsize=10.0)
    ax2.set_xlabel("HDBSCAN Min Cluster Size m_cl", fontsize=11.0, fontweight="bold")
    ax2.set_ylabel("Silhouette / Corpus Share", fontsize=11.0, fontweight="bold")
    ax2.set_title("(b) Mega-Cluster Resolution (EOM vs. BEOM)", fontsize=11.4, fontweight="bold")
    ax2.set_ylim(0.04, 0.62)
    ax2.legend(bbox_to_anchor=(0.98, 0.11), loc="lower right", fontsize=7.8, frameon=True, facecolor="white", framealpha=0.95)

    # Panel (c): 2D UMAP Manifold Clusters (Section 5.2)
    ax3 = axes[2]
    mask = labels != -1
    ax3.scatter(umap_2d[~mask, 0], umap_2d[~mask, 1], c="#d9d9d9", s=14, alpha=0.55, label="Noise C_-1 (38.1%)")
    ax3.scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=labels[mask], cmap="tab20", s=22, alpha=0.88, edgecolors="none")

    callouts = [
        (12, "T12: Speech", (4.4, 4.7)),
        (4, "T2: Task Cast", (5.2, 2.35)),
        (10, "T7: Checksums", (7.6, 5.3)),
        (3, "T4: PyArrow", (9.3, 6.2)),
        (5, "T8: Pool .map", (11.1, 4.1)),
        (0, "T0: DupKeys", (5.7, 1.25)),
        (1, "T1: Columns", (10.5, 2.35)),
    ]
    for cid, text_lbl, (tx, ty) in callouts:
        pts = umap_2d[labels == cid]
        if len(pts) > 0:
            cx, cy = float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
            ax3.annotate(
                text_lbl, xy=(cx, cy), xytext=(tx, ty),
                fontsize=8.6, fontweight="bold", ha="center", va="center",
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#333333"),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#333333", alpha=0.94, lw=0.9)
            )
    ax3.set_title("(c) 2D UMAP Manifold (Bounded EOM)", fontsize=11.8, fontweight="bold")
    ax3.set_xlabel("UMAP Dimension 1", fontsize=11.0, fontweight="bold")
    ax3.set_ylabel("UMAP Dimension 2", fontsize=11.0, fontweight="bold")
    ax3.tick_params(axis="both", labelsize=10.0)
    ax3.set_xlim(2.8, 12.8)
    ax3.set_ylim(0.7, 7.3)
    ax3.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95, fontsize=8.4)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure2_spatial_discovery.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure2_spatial_discovery.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/figure2_spatial_discovery.pdf and .png")


def make_section6_temporal_prospective_figure():
    """Figure 4 (Sections 6.2 & 6.3): (a) Coarse Windows Mask March Shift, (b) Lookback Window Sensitivity (w=7..28d), (c) Dose-Response Floor."""
    exp8_df = pd.read_csv(OUT_DIR / "exp8_it_coarse_vs_monthly_shifts.csv")
    win_df = pd.read_csv(OUT_DIR / "exp3d_rolling_window_sensitivity.csv")
    dose_df = pd.read_csv(OUT_DIR / "exp5_dose_response_perturbation.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), dpi=300)

    # Panel (a): Non-Monotonic Monthly Trajectories Explaining Coarse Cancellation (Section 6.2)
    ax1 = axes[0]
    months_x = np.array([0, 1, 2])
    plot_specs = [
        ("IT-03", "IT03: Install**", "#d9534f", "o", "-", 2.5),
        ("IT-10", "IT10: RSA/Java*", "#0275d8", "s", "--", 2.2),
        ("IT-05", "IT05: File Crash", "#e07a5f", "^", "-.", 2.0),
        ("IT-07", "IT07: OneDrive", "#5cb85c", "v", "--", 2.0),
        ("IT-02", "IT02: DevOps**", "#9467bd", "D", "-", 2.2),
        ("IT-08", "IT08: F5 BIG-IP**", "#333333", "P", ":", 2.0),
    ]
    for cid, lbl, col, mk, ls, lw in plot_specs:
        row = exp8_df[exp8_df["Cluster_ID"] == cid].iloc[0]
        vals = [row["P_Jan_Pct"], row["P_Feb_Pct"], row["P_Mar_Pct"]]
        ax1.plot(months_x, vals, marker=mk, markersize=6, color=col, linestyle=ls, lw=lw, label=lbl)
    # Show IT-03 Jan baseline to illustrate how Feb lull (11.6%) + Mar surge (21.5%) cancel out
    ax1.axhline(16.02, color="#d9534f", linestyle=":", lw=1.2, alpha=0.75)
    ax1.annotate(
        "Feb Lull (11.6%)\nCancels Mar Surge",
        xy=(1, 11.59), xytext=(1.08, 17.8),
        fontsize=8.0, fontweight="bold", color="#b52b27", ha="center",
        arrowprops=dict(arrowstyle="->", lw=1.0, color="#b52b27"),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d9534f", alpha=0.94, lw=0.8)
    )
    ax1.set_xticks(months_x)
    ax1.set_xticklabels(["Jan (Base)", "Feb", "Mar"], fontsize=10.0, fontweight="bold")
    ax1.tick_params(axis="y", labelsize=10.0)
    ax1.set_ylabel("Monthly Topic Share (%)", fontsize=11.0, fontweight="bold")
    ax1.set_title("(a) Monthly Cancellation Waves (IT)", fontsize=11.5, fontweight="bold")
    ax1.set_ylim(-0.8, 29.0)
    ax1.legend(loc="upper left", ncol=2, fontsize=7.6, columnspacing=0.6, handletextpad=0.4, frameon=True, facecolor="white", framealpha=0.94)

    # Panel (b): Rolling Window Lookback Sensitivity w in {7, 14, 21, 28} Days
    ax2 = axes[1]
    r3_win = win_df[win_df["Cluster_ID"] == "R3"].sort_values("Window_Days_w")
    r1_win = win_df[win_df["Cluster_ID"] == "R1"].sort_values("Window_Days_w")
    ws = r3_win["Window_Days_w"].to_numpy()
    ax2.plot(ws, r3_win["Peak_Prevalence_Pct"], marker="o", markersize=6.5, lw=2.4, color="#9467bd", label="R3 Peak Share (%)")
    ax2.plot(ws, r1_win["Daily_Roughness_PP"] * 4.0, marker="s", markersize=6.5, lw=2.3, linestyle="--", color="#e07a5f", label="R1 Daily Roughness (×4 pp)")
    ax2.plot(ws, r3_win["Daily_Roughness_PP"] * 4.0, marker="^", markersize=6.5, lw=2.1, linestyle="-.", color="#2b5c8f", label="R3 Daily Roughness (×4 pp)")
    ax2.axvline(14, color="#2ca02c", linestyle=":", lw=1.7, label="Selected w = 14 Days")
    ax2.set_xticks([7, 14, 21, 28])
    ax2.set_xticklabels(["7d", "14d", "21d", "28d"], fontsize=10.0, fontweight="bold")
    ax2.tick_params(axis="y", labelsize=10.0)
    ax2.set_xlabel("Trailing Window Lookback w (Days)", fontsize=11.0, fontweight="bold")
    ax2.set_ylabel("Peak Share (%) / Scaled Roughness", fontsize=10.8, fontweight="bold")
    ax2.set_title("(b) Rolling Window Bias and Variance", fontsize=11.5, fontweight="bold")
    ax2.set_ylim(0, 26.2)
    ax2.legend(loc="upper right", fontsize=8.0, frameon=True, facecolor="white", framealpha=0.95)

    # Panel (c): Dose-Response Outbreak Sensitivity Floor (Section 6.3)
    ax3 = axes[2]
    d12 = dose_df[(dose_df["HDBSCAN_Min_Cluster_Size"] == 12) & (dose_df["Anomaly_Domain"] == "CUDA / NCCL Distributed OOM")].sort_values("Injected_Count_M")
    d16 = dose_df[(dose_df["HDBSCAN_Min_Cluster_Size"] == 16) & (dose_df["Anomaly_Domain"] == "CUDA / NCCL Distributed OOM")].sort_values("Injected_Count_M")

    ax3.plot(d12["Injected_Count_M"], d12["Cluster_Recall"] * 100.0, marker="s", markersize=6, lw=2.3, linestyle="--", color="#2ca02c", label="Recall (m_cl=12)")
    ax3.plot(d16["Injected_Count_M"], d16["Cluster_Recall"] * 100.0, marker="o", markersize=6, lw=2.5, color="#0275d8", label="Recall (m_cl=16)")
    ax3.plot(d16["Injected_Count_M"], d16["Top1_Detected"].astype(float) * 100.0, marker="^", markersize=6, lw=2.1, linestyle="-.", color="#d9534f", label="Top-1 (m_cl=16)")
    ax3.axvline(16, color="#333333", linestyle=":", lw=1.4)
    ax3.axvline(20, color="#d9534f", linestyle=":", lw=1.4)
    ax3.set_xlabel("Injected Q2 Outage Tickets (M)", fontsize=11.0, fontweight="bold")
    ax3.set_ylabel("Detection Sensitivity (%)", fontsize=11.0, fontweight="bold")
    ax3.set_title("(c) Dose Response Outbreak Floor", fontsize=11.5, fontweight="bold")
    ax3.tick_params(axis="both", labelsize=10.0)
    ax3.set_ylim(-5, 108)
    ax3.legend(bbox_to_anchor=(0.99, 0.14), loc="lower right", fontsize=7.6, handlelength=1.2, handletextpad=0.35, borderpad=0.3, frameon=True, facecolor="white", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure4_temporal_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure4_temporal_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure4_temporal_and_prospective.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure4_temporal_and_prospective.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/figure4_temporal_sensitivity.pdf and .png")


def make_section7_extrinsic_figure():
    """Figure 5 (Section 7.1): Single-column Extrinsic Bug Issue Ratio vs. Median MTTR Hours with Regime Zones."""
    ext_df = pd.read_csv(OUT_DIR / "exp6_extrinsic_validation_r10.csv")

    fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=300)
    ext_non_noise = ext_df[ext_df["Cluster_ID"] != "Noise (-1)"].copy()

    # Shaded workflow regime zones to provide visual interpretation beyond raw table values
    ax.axvspan(46, 81, ymin=0.0, ymax=0.58, color="#fde0dc", alpha=0.42, zorder=0)
    ax.text(79.8, 98, "Regime III: User Bug Outbreaks\n(Cross Platform Repro)", fontsize=7.6, fontweight="bold", color="#9c1f1f", ha="right", va="top")

    ax.axhspan(108, 174, xmin=0.0, xmax=0.58, color="#e8eaf6", alpha=0.48, zorder=0)
    ax.text(2.5, 170, "Regime II: Schema Design PRs\n(Multiweek Architecture Sign Off)", fontsize=7.6, fontweight="bold", color="#283593", ha="left", va="top")

    ax.axvspan(2, 25, ymin=0.0, ymax=0.56, color="#e8f5e9", alpha=0.45, zorder=0)
    ax.text(2.8, 101.5, "Regime I: Routine CI PRs\n(Fast Merge, Low Backlog)", fontsize=7.6, fontweight="bold", color="#1b5e20", ha="left", va="top")

    sc = ax.scatter(
        ext_non_noise["Issue_Ratio_Pct"],
        ext_non_noise["Median_MTTR_Hours"],
        s=ext_non_noise["Total_Count"] * 1.6 + 60,
        c=ext_non_noise["Open_Unresolved_Pct"],
        cmap="YlOrRd",
        edgecolor="#222222",
        lw=1.2,
        alpha=0.92,
        zorder=3,
    )
    offsets = {
        "R1": (12, 4), "R6": (-46, 12), "R5": (8, -4), "R3": (8, -4),
        "R9": (8, -11), "R2": (8, -3), "R7": (-85, -8), "R8": (-75, 5),
        "R4": (-80, -4), "R0": (-45, -14)
    }
    short_names = {
        "R1": "R1 (pickle/dtype)",
        "R6": "R6 (numpy/core)",
        "R5": "R5 (docs)",
        "R3": "R3 (task_cast)",
        "R9": "R9 (pdf/duorc)",
        "R2": "R2 (rename_col)",
        "R7": "R7 (coco/cards)",
        "R8": "R8 (speech)",
        "R4": "R4 (cer_metric)",
        "R0": "R0 (dup_keys)",
    }
    for _, row in ext_non_noise.iterrows():
        cid_lbl = row["Cluster_ID"]
        dx, dy = offsets.get(cid_lbl, (6, 4))
        ax.annotate(
            short_names.get(cid_lbl, cid_lbl),
            (row["Issue_Ratio_Pct"], row["Median_MTTR_Hours"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.6,
            fontweight="bold",
            zorder=4,
        )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Open Unresolved (%)", fontsize=10.0, fontweight="bold")
    cbar.ax.tick_params(labelsize=9.5)
    ax.set_xlabel("User Bug Issue Share (%, Excl. PRs)", fontsize=10.8, fontweight="bold")
    ax.set_ylabel("Median Resolution Time (Hours)", fontsize=10.8, fontweight="bold")
    ax.set_title("Workflow Regime Stratification: Bug Share vs. MTTR", fontsize=11.2, fontweight="bold")
    ax.tick_params(axis="both", labelsize=10.0)
    ax.set_xlim(0, 82)
    ax.set_ylim(5, 178)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure5_extrinsic_mttr.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure5_extrinsic_mttr.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/figure5_extrinsic_mttr.pdf and .png")


def make_figure3_q2_rolling_and_excess():
    """Regenerate Figure 3 (Q2 rolling prevalence and Q1-baseline excess) with (a)/(b) titles,
    unclipped [-14.8, 23.5] y-axis, and callouts for R1, R3, R0, and R6.
    """
    df_ex = pd.read_csv(ROOT / "paper_q2_results" / "comparisons" / "baseline_excess_2021Q2_vs_2021Q1.csv")
    df_ex["Date"] = pd.to_datetime(df_ex["Window_End"])

    top_clusters = [
        (1, "R1: pickle/dtype", "#1f77b4"),
        (6, "R6: numpy/core", "#ff7f0e"),
        (0, "R0: dup_keys", "#2ca02c"),
        (7, "R7: coco/cards", "#d62728"),
        (3, "R3: task_cast", "#9467bd"),
        (5, "R5: sphinx_docs", "#8c564b"),
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.6, 4.8), dpi=300, sharex=True)

    for cid, lbl, col in top_clusters:
        sub = df_ex[df_ex["Cluster_ID"] == cid].sort_values("Date")
        ax1.plot(sub["Date"], sub["Target_Prevalence"] * 100.0, lw=2.1, label=lbl, color=col)
        ax2.plot(sub["Date"], sub["Excess_PP"], lw=2.1, label=lbl, color=col)

    ax1.set_title("(a) 14-Day Trailing Rolling Share in Q2 (%)", fontsize=10.8, fontweight="bold", pad=4)
    ax1.set_ylabel("Q2 Share (%)", fontsize=10.8, fontweight="bold")
    ax1.tick_params(axis="both", labelsize=9.8)
    ax1.set_ylim(-1.5, 56)
    ax1.legend(
        bbox_to_anchor=(0.5, 1.18), loc="lower center", ncol=3,
        fontsize=8.6, frameon=True, facecolor="white", framealpha=0.95
    )
    ax1.annotate(
        "R1 Peak: 50.0% (Apr 15)", xy=(pd.to_datetime("2021-04-15"), 50.0),
        xytext=(pd.to_datetime("2021-04-25"), 46.5),
        arrowprops=dict(arrowstyle="->", lw=1.3, color="#1f77b4"),
        fontsize=9.0, fontweight="bold", color="#1f77b4"
    )
    ax1.annotate(
        "R0 Late Surge:\n12.7% (Jun 28)", xy=(pd.to_datetime("2021-06-28"), 12.73),
        xytext=(pd.to_datetime("2021-05-27"), 21.5),
        arrowprops=dict(arrowstyle="->", lw=1.3, color="#1b781b"),
        fontsize=8.8, fontweight="bold", color="#1b781b"
    )

    ax2.set_title("(b) Baseline-Relative Excess Over Q1 Share (pp)", fontsize=10.8, fontweight="bold", pad=4)
    ax2.axhline(0, color="#111111", lw=1.2, linestyle="--")
    ax2.set_ylabel("Excess vs. Q1 (pp)", fontsize=10.8, fontweight="bold")
    ax2.set_xlabel("2021 Q2 Daily Window Endpoint", fontsize=10.8, fontweight="bold")
    ax2.tick_params(axis="both", labelsize=9.8)
    ax2.set_ylim(-14.8, 23.5)
    ax2.annotate(
        "R3 Spike: +16.3 pp\n(May 26)", xy=(pd.to_datetime("2021-05-26"), 16.25),
        xytext=(pd.to_datetime("2021-04-26"), 14.5),
        arrowprops=dict(arrowstyle="->", lw=1.3, color="#7b4fa3"),
        fontsize=9.0, fontweight="bold", color="#7b4fa3"
    )
    ax2.annotate(
        "R6 Resolved (< Q1 on 91/91d)", xy=(pd.to_datetime("2021-05-03"), -12.55),
        xytext=(pd.to_datetime("2021-05-12"), -13.3),
        arrowprops=dict(arrowstyle="->", lw=1.3, color="#d95f02"),
        fontsize=8.8, fontweight="bold", color="#d95f02"
    )

    ax2.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15]))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure3_q2_rolling_and_excess.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure3_q2_rolling_and_excess.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/figure3_q2_rolling_and_excess.pdf and .png")


if __name__ == "__main__":
    make_section5_spatial_figure()
    make_figure3_q2_rolling_and_excess()
    make_section6_temporal_prospective_figure()
    make_section7_extrinsic_figure()
