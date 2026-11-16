"""Generate the Midterm Report Pipeline Architecture & Evaluation Diagram."""

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["mathtext.fontset"] = "dejavusans"


def make_midterm_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(14.2, 4.5), dpi=300)
    ax.set_xlim(0, 136)
    ax.set_ylim(0, 41)
    ax.axis("off")

    def draw_box(x, y, w, h, title, lines, facecolor="#f8f9fa", edgecolor="#2b5c8f", title_bg="#2b5c8f", title_fs=9.3, body_fs=8.0):
        body = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.2,rounding_size=0.65",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5, zorder=2
        )
        ax.add_patch(body)
        hdr_h = 3.7
        hdr = FancyBboxPatch(
            (x, y + h - hdr_h), w, hdr_h,
            boxstyle="round,pad=0.2,rounding_size=0.65",
            facecolor=title_bg, edgecolor=edgecolor, linewidth=1.4, zorder=3
        )
        ax.add_patch(hdr)
        ax.text(
            x + w / 2, y + h - hdr_h / 2, title,
            ha="center", va="center", fontsize=title_fs, fontweight="bold", color="white", zorder=4
        )
        ty = y + h - hdr_h - 1.45
        for line in lines:
            is_bullet = line.startswith("•")
            ax.text(
                x + (0.65 if is_bullet else 1.25), ty, line,
                ha="left", va="top", fontsize=body_fs,
                fontweight="bold" if is_bullet else "normal",
                color="#1a1a1a", zorder=4
            )
            ty -= 2.4

    def draw_arrow(x1, y1, x2, y2, label="", color="#222222"):
        arr = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>,head_length=5.2,head_width=3.2",
            color=color, linewidth=1.7, zorder=5
        )
        ax.add_patch(arr)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 1.55
            ax.text(
                mx, my, label, ha="center", va="center", fontsize=7.8,
                fontweight="bold", color=color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85)
            )

    # Top row: 5 Sequential Pipeline Stages with generous 5.5-unit gaps
    y_top = 17.5
    h_top = 21.8
    w_box = 22.0
    gap = 5.6
    xs = [1.2 + i * (w_box + gap) for i in range(5)]

    draw_box(
        xs[0], y_top, w_box, h_top,
        "1. SBERT Embedding",
        [
            "• ITSM Corpus (N = 1,338):",
            "  Summary + Description",
            "  lowercased & normalized",
            "• all-MiniLM-L6-v2:",
            r"  Dense vectors $e_i \in \mathbb{R}^{384}$",
            r"  Unit norm $\|e_i\|_2 = 1$"
        ],
        facecolor="#eef4fa", edgecolor="#1f4e78", title_bg="#1f4e78"
    )

    draw_box(
        xs[1], y_top, w_box, h_top,
        "2. UMAP Reduction",
        [
            "• Nonlinear Manifold:",
            "  384D → 5D projection",
            r"  $u_i = \mathrm{UMAP}(e_i;\, d=5, k=15)$",
            "• Metric & Topology:",
            "  Cosine distance, fixed seed;",
            "  avoids 384D concentration"
        ],
        facecolor="#eefaf2", edgecolor="#216e39", title_bg="#216e39"
    )

    draw_box(
        xs[2], y_top, w_box, h_top,
        "3. Joint HDBSCAN",
        [
            r"• Joint Space $D = D_a \cup D_b$:",
            "  Prevents label drift across",
            "  observation periods",
            "• Density Clustering:",
            r"  $m_{\mathrm{cl}} = 20,\; m_{\mathrm{pts}} = 6$;",
            r"  isolates Noise $y_i = -1$ (19.7%)"
        ],
        facecolor="#eef8f8", edgecolor="#1b696a", title_bg="#1b696a"
    )

    draw_box(
        xs[3], y_top, w_box, h_top,
        "4. c-TF-IDF Topics",
        [
            "• Cluster Pooling:",
            r"  Pool docs in each cluster $c$",
            "• Class-Based Weights:",
            r"  $W_{t,c} = \mathrm{tf}_{t,c}\log(1 + A/\mathrm{cf}_t)$",
            "• Representation:",
            "  Top 10 keywords per topic"
        ],
        facecolor="#fff9ed", edgecolor="#b25912", title_bg="#b25912"
    )

    draw_box(
        xs[4], y_top, w_box, h_top,
        "5. Temporal Shift (ΔK)",
        [
            "• Window Proportions:",
            r"  $P(k \mid D_a)$ vs. $P(k \mid D_b)$",
            r"  $\Delta K = P(k \mid D_b) - P(k \mid D_a)$",
            "• Issue Classification:",
            r"  Surge: $\Delta K \geq 0.025$",
            r"  Chronic: $P_A, P_B \geq 0.04$"
        ],
        facecolor="#f7f0fa", edgecolor="#5c2d82", title_bg="#5c2d82"
    )

    arrow_labels = ["384D", "5D", "Clusters", "Topics"]
    for i in range(4):
        draw_arrow(
            xs[i] + w_box + 0.45, y_top + h_top / 2 - 1.2,
            xs[i + 1] - 0.45, y_top + h_top / 2 - 1.2,
            label=arrow_labels[i]
        )

    # Bottom container: Midterm Experimental Setup & Evaluation
    total_w = xs[-1] + w_box - xs[0]
    outer = FancyBboxPatch(
        (xs[0], 1.0), total_w, 13.6,
        boxstyle="round,pad=0.25,rounding_size=0.8",
        facecolor="#fcfcfc", edgecolor="#666666", linewidth=1.3, linestyle="--", zorder=1
    )
    ax.add_patch(outer)
    ax.text(
        xs[0] + total_w / 2, 13.15,
        "Midterm Experimental Partitioning, Baseline Benchmark, and Synthetic Perturbation Evaluation",
        ha="center", va="center", fontsize=9.3, fontweight="bold", color="#333333", zorder=4
    )

    b_y = 2.0
    b_h = 9.5
    b_gap = 6.8
    b_w = (total_w - 4.0 - 2 * b_gap) / 3.0
    b_xs = [xs[0] + 2.0 + i * (b_w + b_gap) for i in range(3)]

    sub_boxes = [
        (
            "Temporal Window Partitioning",
            "1,380 raw → 1,338 valid tickets (Jan–Mar 2021)\n"
            r"Baseline $D_a$ (Jan, $N_a=487$) vs. Obs. $D_b$ (Feb–Mar, $N_b=851$)",
            "#f0f4f8", "#1f4e78", "#1f4e78"
        ),
        (
            "Baseline Benchmark Comparison",
            "TF-IDF (2,500 terms) + K-Means (K = 10) + shared c-TF-IDF\n"
            r"Evaluated via $C_v$ Coherence, Diversity ($TD$), & Silhouette",
            "#f4f4f6", "#495057", "#495057"
        ),
        (
            "Perturbation Sensitivity Test",
            r"Inject +30 Kubernetes OOM error tickets into Period $D_b$" + "\n"
            r"Projected into UMAP+HDBSCAN → Ranked #1 ($\Delta K = +0.0216$)",
            "#fdf2f2", "#9e2a2b", "#9e2a2b"
        ),
    ]

    for idx, (stitle, sbody, sfc, sec, stc) in enumerate(sub_boxes):
        bx = b_xs[idx]
        sb = FancyBboxPatch(
            (bx, b_y), b_w, b_h,
            boxstyle="round,pad=0.2,rounding_size=0.55",
            facecolor=sfc, edgecolor=sec, linewidth=1.3, zorder=2
        )
        ax.add_patch(sb)
        ax.text(
            bx + b_w / 2, b_y + b_h - 2.1, stitle,
            ha="center", va="center", fontsize=8.8, fontweight="bold", color=stc, zorder=3
        )
        ax.text(
            bx + b_w / 2, b_y + 3.2, sbody,
            ha="center", va="center", fontsize=7.9, color="#1a1a1a", zorder=3, linespacing=1.4
        )

    draw_arrow(b_xs[0] + b_w + 0.45, b_y + b_h / 2 - 0.5, b_xs[1] - 0.45, b_y + b_h / 2 - 0.5, label="Compare", color="#444444")
    draw_arrow(b_xs[1] + b_w + 0.45, b_y + b_h / 2 - 0.5, b_xs[2] - 0.45, b_y + b_h / 2 - 0.5, label="Validate", color="#9e2a2b")

    fig.tight_layout(pad=0.2)
    out_png = FIG_DIR / "midterm_pipeline_diagram.png"
    out_pdf = FIG_DIR / "midterm_pipeline_diagram.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png} and {out_pdf}")


if __name__ == "__main__":
    make_midterm_pipeline_diagram()
