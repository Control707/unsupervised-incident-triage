#!/usr/bin/env python3
"""Poster-specific faithful redraw of paper Figure 5; no inference or model download.

Run from project root: PYTHONDONTWRITEBYTECODE=1 .venv-q2/bin/python poster_work/redraw_payoff.py
Only final payoff.{png,pdf,svg} and payoff.provenance.json are written.
IT percentages are recovered from the original PDF's colored vector rectangles,
NOT inferred from team purity, keywords, or rerun clustering. Original plotted
one-decimal shares (rather than unrounded count ratios) are retained exactly.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "poster_work/hires_figures"
PDF = ROOT / "figures/v3_figure5_payoff_triage.pdf"
WIDTH, HEIGHT, DPI = 25.0, 25.0 * 563 / 1800, 240
IT_IDS = ["IT-00", "IT-02", "IT-01", "IT-03", "IT-10", "IT-07", "IT-08", "IT-12", "IT-13"]
IT_NAMES = ["Training request", "DevOps config request", "Licensing request", "SW install request", "RSA/Java outage", "OneDrive/Ping error", "F5 BIG-IP outage", "CrowdStrike alert", "IAM role outage"]
GH_NAMES = ["Duplicate keys", "Pickle / dtype", "Rename column", "Task cast", "CER metric", "Documentation", "NumPy", "COCO / cards", "Speech / audio", "PDF / DUORC"]
PALETTE = ["#2ca02c", "#f0ad4e", "#d9534f"]


def recover_pdf_vectors():
    # pypdf exists in the host Python, not the prescribed plotting environment.
    # This subprocess only reads the source PDF and returns compact JSON.
    reader = r'''
import json, sys
from pypdf import PdfReader
from pypdf.generic import ContentStream
p = PdfReader(sys.argv[1]).pages[0]
ops = ContentStream(p.get_contents(), p.pdf).operations
cols = [.17254903, .9411765, .8509804]
bars = [[], [], []]
for i, (args, op) in enumerate(ops):
    if op != b'rg': continue
    for j, color in enumerate(cols):
        if abs(float(args[0]) - color) > 1e-7: continue
        path = []
        for ar, op2 in ops[i+1:i+10]:
            if op2 in (b'm', b'l'): path.append([float(x) for x in ar])
            if op2 == b'h': break
        # Original bar area; excludes same-colored legend swatches.
        if path and 500 < path[0][0] < 710 and 35 < path[0][1] < 200:
            bars[j].append({'operation': i, 'path': path})
assert all(len(x)==9 for x in bars), [len(x) for x in bars]
x0 = bars[0][0]['path'][0][0]
x100 = bars[0][0]['path'][1][0]
scale = x100 - x0
for group in bars:
    for b in group:
        b['share_from_pdf'] = (b['path'][1][0]-b['path'][0][0])/scale*100
# Scatter circles occur in R0...R9 order. Recover their point-area sizes.
scatter = []
for i, (args, op) in enumerate(ops):
    if op != b'rg' or not 600 <= i <= 730: continue
    path = [(ar, op2) for ar, op2 in ops[i+1:i+15] if op2 in (b'm', b'c')]
    if len(path) != 9: continue
    bottom = [float(x) for x in path[0][0]]
    center_y = float(path[2][0][-1])
    radius = center_y-bottom[1]
    scatter.append({'rgb': [float(x) for x in args], 'center': [bottom[0], center_y], 'radius_pt': radius, 'size_pt2': 4*radius*radius})
assert len(scatter)==10, len(scatter)
print(json.dumps({'bars':bars,'x0':x0,'x100':x100,'scatter':scatter,'pdf_page_pt':list(map(float,p.mediabox))}))
'''
    return json.loads(subprocess.check_output(["/opt/homebrew/bin/python3", "-c", reader, str(PDF)], text=True))


def main():
    vectors = recover_pdf_vectors()
    e6 = pd.read_csv(ROOT / "final_experiments/exp6_extrinsic_validation_r10.csv").set_index("Cluster_ID")
    e9 = pd.read_csv(ROOT / "final_experiments/exp9_it_extrinsic_triage_and_routing.csv").set_index("Cluster_ID")
    raw = pd.read_csv(ROOT / "github_issues_180d.csv", keep_default_na=False)
    assignments = pd.read_csv(ROOT / "paper_q2_results/topic_assignments.csv")
    merged = raw.merge(assignments[["id", "hdbscan_cluster"]], on="id", validate="one_to_one")
    assert len(merged) == len(raw) == len(assignments) == 881
    gh = e6.loc[[f"R{i}" for i in range(10)]].copy()
    gh["External_Author_Pct_Raw"] = [100 * (merged.loc[merged.hdbscan_cluster == i, "author_association"] == "NONE").mean() for i in range(10)]
    # Original PDF colors reveal one-decimal author-share rounding (R8 is diagnostic).
    gh["External_Author_Pct"] = gh.External_Author_Pct_Raw.round(1)
    counts = [int((merged.hdbscan_cluster == i).sum()) for i in range(10)]
    assert counts == gh.Total_Count.tolist()
    shares_raw = np.array([[vectors["bars"][j][i]["share_from_pdf"] for j in range(3)] for i in range(9)])
    shares = np.round(shares_raw, 1)
    assert np.max(np.abs(shares - shares_raw)) < 0.00004
    assert np.allclose(shares.sum(axis=1), 100)
    it = e9.loc[IT_IDS]
    assert np.array_equal(np.round(100*it.Unassigned_Tickets_Routed/it.Total_Tickets, 1), shares[:, 2])
    # The recorded cluster sizes uniquely recover integer counts for these shares.
    outcome_counts = np.rint(shares * it.Total_Tickets.to_numpy()[:, None]/100).astype(int)
    assert np.array_equal(outcome_counts.sum(axis=1), it.Total_Tickets)
    assert np.allclose(np.round(outcome_counts / it.Total_Tickets.to_numpy()[:, None]*100, 1), shares)
    assert np.array_equal(outcome_counts[:, 2], it.Unassigned_Tickets_Routed)
    # Original circle area is affine in total record count: 60 + 1.55*n.
    original_sizes = np.array([x["size_pt2"] for x in vectors["scatter"]])
    sizes = 60 + 1.55 * gh.Total_Count.to_numpy()
    assert np.max(np.abs(original_sizes - sizes)) < 0.002
    norm = colors.Normalize(0, 56)
    cmap = matplotlib.colormaps["YlOrRd"]
    assert np.allclose(cmap(norm(gh.External_Author_Pct))[:, :3], [s["rgb"] for s in vectors["scatter"]], atol=1e-6)
    # Verify CSV x/y against original PDF scatter geometry (including original y=5 baseline).
    original_xy = np.array([s["center"] for s in vectors["scatter"]])
    expected_x = 43.34746 + gh.Issue_Ratio_Pct.to_numpy()/87 * 269.28099
    expected_y = 34.706838 + (gh.Median_MTTR_Hours.to_numpy()-5)/173 * 213.46
    assert np.allclose(original_xy[:, 0], expected_x, atol=0.0001)
    assert np.allclose(original_xy[:, 1], expected_y, atol=0.0001)

    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":22,
        "axes.labelsize":24,"xtick.labelsize":22,"ytick.labelsize":22,
        "axes.titlesize":28,"svg.fonttype":"none","pdf.fonttype":42,
        "axes.edgecolor":"#9da8b5","axes.linewidth":1.0,
        "savefig.bbox":None,"figure.facecolor":"white"})
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI)
    # Allocations: GitHub 42%, enterprise IT 58%. All coordinates fixed.
    fig.text(.015, .935, "(a) GitHub: internal sprints vs. user bugs", fontsize=28, weight="bold", color="#172d43")
    fig.text(.442, .935, "(b) Enterprise IT: the Tier 1 triage blind spot", fontsize=28, weight="bold", color="#172d43")
    ax = fig.add_axes([.060, .165, .212, .565])
    ax.set_xlim(0,87); ax.set_ylim(5,175)
    ax.set_xticks([0,20,40,60,80]); ax.set_yticks([20,60,100,140])
    ax.set_axisbelow(True); ax.grid(color="#dfe4e9", linewidth=1)
    # Same contextual regions as paper figure, unobtrusive behind data.
    ax.fill_between([0,50],110,175,color="#e8eaf6",alpha=.55,zorder=0)
    ax.fill_between([46,87],5,86,color="#fde0dd",alpha=.44,zorder=0)
    # Enlarge all areas by the same factor: original affine count encoding retained.
    area_scale = 4.0
    sc = ax.scatter(gh.Issue_Ratio_Pct, gh.Median_MTTR_Hours,
        s=sizes*area_scale,c=gh.External_Author_Pct,cmap=cmap,norm=norm,
        edgecolors="#27323d",linewidths=1.7,zorder=3)
    ax.set_xlabel("Issue share (%, excluding PRs)",labelpad=13)
    ax.set_ylabel("Median resolution time (hours)",labelpad=12)
    # IDs only on scatter; full names in an adjacent readable key.
    offsets = [(0,-31),(22,10),(-16,16),(11,13),(19,-9),(10,14),
               (16,-19),(-26,-20),(-13,28),(13,15)]
    scatter_labels=[]
    for i, ((_, row), offset) in enumerate(zip(gh.iterrows(), offsets)):
        scatter_labels.append(ax.annotate(f"R{i}",(row.Issue_Ratio_Pct,row.Median_MTTR_Hours),
            xytext=offset,textcoords="offset points",fontsize=22,weight="bold",color="#243545",
            ha="right" if offset[0]<0 else "left",va="center",
            arrowprops={"arrowstyle":"-","color":"#7c8791","lw":.9},zorder=5))
    key_texts=[]
    fig.text(.287,.723,"Topic key",fontsize=24,weight="bold",color="#172d43")
    for i,name in enumerate(GH_NAMES):
        key_texts.append(fig.text(.287,.659-i*.055,f"R{i}  {name}",fontsize=22,color="#263747"))
    cax=fig.add_axes([.053,.831,.195,.027])
    cb=fig.colorbar(sc,cax=cax,orientation="horizontal",ticks=[0,20,40,56])
    cb.ax.tick_params(labelsize=22,length=3,pad=5)
    fig.text(.049,.877,"External authors (NONE, %)",fontsize=22,color="#263747")
    fig.text(.283,.877,"Bubble area: record count",fontsize=22,color="#263747")
    size_handles=[plt.scatter([],[],s=(60+1.55*n)*area_scale,facecolor="#dce4ec",edgecolor="#27323d") for n in [20,100,298]]
    fig.legend(size_handles,["20","100","298"],loc="center",bbox_to_anchor=(.348,.812),
        ncol=3,frameon=False,fontsize=22,handletextpad=.35,columnspacing=.7,borderpad=0)

    ax2=fig.add_axes([.610,.165,.244,.565])
    ax2.set_xlim(0,100); ax2.set_ylim(8.65,-.65)
    ax2.set_xticks([0,25,50,75,100]); ax2.set_yticks(np.arange(9))
    ax2.set_yticklabels([f"{cid.replace('-', '')} · {name}" for cid,name in zip(IT_IDS,IT_NAMES)],fontsize=22)
    ax2.tick_params(axis="y",length=0,pad=12)
    ax2.set_xlabel("Share of cluster tickets at intake (%)",labelpad=13)
    ax2.grid(axis="x",color="#dfe4e9",linewidth=1); ax2.set_axisbelow(True)
    left=np.zeros(9)
    for j,color in enumerate(PALETTE):
        ax2.barh(np.arange(9),shares[:,j],left=left,color=color,edgecolor="#27323d",linewidth=1.2,height=.56)
        left+=shares[:,j]
    ax2.axhline(3.5,color="#536574",linestyle=(0,(5,4)),linewidth=1.4)
    ax2.spines[["top","right","left"]].set_visible(False)
    legend_labels=["Reached a specialist queue at intake", "Sent to generic TIER 2 holding queue", "Orphaned in Unassigned (no resolution)"]
    fig.legend([Patch(facecolor=c,edgecolor="#27323d") for c in PALETTE],legend_labels,
        loc="upper left",bbox_to_anchor=(.451,.897),frameon=False,fontsize=22,
        labelspacing=.3,handlelength=1.35,borderpad=0)
    fig.text(.868,.747,"Hops · grouped",fontsize=22,weight="bold",color="#263747")
    for i,(_,row) in enumerate(it.iterrows()):
        hops=f"{row.Mean_Reassigned_Count:.2f}"
        if i>=4:hops+=f" · {int(row.Unassigned_Tickets_Routed)}"
        ax2.text(102.1,i,hops,fontsize=22,weight="bold",color="#28613b" if i<4 else "#ac3531",va="center",clip_on=False)
    # Detailed provenance ships separately; leave the poster's existing source caption unchanged.

    # Renderer-based checks: IDs are legible, disjoint and inside the fixed canvas.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    label_boxes=[t.get_window_extent(renderer).expanded(1.04,1.08) for t in scatter_labels]
    overlaps=[]
    for i in range(10):
        for j in range(i):
            if label_boxes[i].overlaps(label_boxes[j]):overlaps.append([i,j])
    assert not overlaps, f"Overlapping scatter labels: {overlaps}"
    all_text=[t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
    outside=[]
    for t in all_text:
        box=t.get_window_extent(renderer)
        if box.width and (box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.x1+.5 or box.y1>fig.bbox.y1+.5):outside.append(t.get_text())
    assert not outside, f"Text outside fixed canvas: {outside}"
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ["png","pdf","svg"]:
        fig.savefig(OUT/f"payoff.{ext}",dpi=DPI,facecolor="white",bbox_inches=None)
    from PIL import Image
    with Image.open(OUT/"payoff.png") as im:
        pixel_size=list(im.size)
        assert pixel_size==[6000,1876],pixel_size
    inputs=["figures/v3_figure5_payoff_triage.pdf","final_experiments/exp6_extrinsic_validation_r10.csv","final_experiments/exp9_it_extrinsic_triage_and_routing.csv","github_issues_180d.csv","paper_q2_results/topic_assignments.csv"]
    provenance={"purpose":"Poster-specific faithful redraw of original Figure 5; all 10 GitHub points and original 9 IT bars preserved",
        "canvas":{"inches":[WIDTH,HEIGHT],"dpi":DPI,"png_pixels":pixel_size,"target_aspect":"1800/563","fixed_canvas":True,"bbox_inches":None},
        "inputs_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs},
        "it_source_note":"Exact original plotted one-decimal percentages recovered from colored PDF vector rectangle widths. No clustering rerun, model download, invented shares, or purity-as-share substitution. Integer outcome counts are consistency checks, not a replacement for plotted percentages.",
        "it":[{"id":cid,"name":name,"n":int(it.loc[cid,"Total_Tickets"]),"specialist_pct":float(shares[i,0]),"tier2_pct":float(shares[i,1]),"unassigned_pct":float(shares[i,2]),"integer_counts_check":outcome_counts[i].tolist(),"mean_hops":float(it.loc[cid,"Mean_Reassigned_Count"]),"grouped_unassigned":int(it.loc[cid,"Unassigned_Tickets_Routed"])} for i,(cid,name) in enumerate(zip(IT_IDS,IT_NAMES))],
        "github_source_note":"x=Exp6 Issue_Ratio_Pct; y=Exp6 Median_MTTR_Hours; color=literal author_association NONE among ALL records (issues and PRs) joined 1:1 by id to saved topic_assignments, rounded to one decimal exactly as original PDF colors (R8 diagnostically confirms rounding); keep_default_na=False is essential. Original YlOrRd color normalization 0–56 retained. Original size 60+1.55*Total_Count is uniformly scaled by 4 for poster legibility.",
        "github":[{"id":cid,"name":GH_NAMES[i],"n":int(row.Total_Count),"issue_share_pct":float(row.Issue_Ratio_Pct),"median_resolution_hours":float(row.Median_MTTR_Hours),"external_author_pct":float(row.External_Author_Pct),"external_author_pct_unrounded":float(row.External_Author_Pct_Raw)} for i,(cid,row) in enumerate(gh.iterrows())],
        "pdf_vector_evidence":vectors,
        "checks":{"github_join_records":len(merged),"github_cluster_sizes_match_exp6":True,"github_xy_match_pdf":True,"github_color_match_pdf":True,"github_bubble_sizes_match_pdf":True,"it_rounded_width_max_error_pct":float(np.max(np.abs(shares-shares_raw))),"it_row_sums_100":True,"it_unassigned_counts_match_exp9":True,"it_integer_counts_reproduce_widths":True,"scatter_label_overlaps":overlaps,"out_of_canvas_text":outside,"fontsize_min_pt":22},
        "outputs_sha256":{f"payoff.{ext}":hashlib.sha256((OUT/f"payoff.{ext}").read_bytes()).hexdigest() for ext in ["png","pdf","svg"]}}
    (OUT/"payoff.provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(json.dumps({"outputs":[str(OUT/f"payoff.{e}") for e in ["png","pdf","svg"]],"checks":provenance["checks"],"png_pixels":pixel_size},indent=2))

if __name__=="__main__":main()
