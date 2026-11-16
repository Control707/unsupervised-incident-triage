"""Tri-model benchmark with rolling-window and calendar-quarter topic tracking.

Run ``python experiment_quarterly_with_lda.py --help`` for temporal settings.
Topic identities come from one retrospective joint-space fit, not per-window fits.
"""

import argparse
import os
import json
from pathlib import Path

from temporal_tracking import (
    TemporalConfig, analyze_temporal_topics, export_temporal_analysis,
    focus_topic_quarter, export_focus_quarter,
)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', default='/content/github_issues_180d.csv' if os.path.exists('/content/github_issues_180d.csv') else 'github_issues_180d.csv')
parser.add_argument('--window-days', type=int, default=14, help='Trailing calendar days per window (default: 14)')
parser.add_argument('--step-days', type=int, default=1, help='Days between rolling endpoints (default: 1)')
parser.add_argument('--min-window-tickets', type=int, default=1, help='Suppress low-support rolling prevalence below this count')
parser.add_argument('--surge-threshold', type=float, default=0.025, help='Absolute prevalence change threshold (0.025 = 2.5 percentage points)')
parser.add_argument('--persistence-threshold', type=float, default=0.035)
parser.add_argument('--temporal-output-dir', default='temporal_results')
parser.add_argument('--focus-quarter', default='2021Q2', help='Within-quarter trajectory focus; paper default: 2021Q2')
parser.add_argument('--no-temporal-plots', action='store_true')
args = parser.parse_args()
temporal_config = TemporalConfig(
    window_days=args.window_days, step_days=args.step_days,
    min_window_tickets=args.min_window_tickets,
    surge_threshold=args.surge_threshold, persistence_threshold=args.persistence_threshold,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import umap
import hdbscan

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

print("=== Step 1: Loading the Dated Technical Incident Stream ===")
df = pd.read_csv(args.data)
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True, format='mixed').dt.tz_localize(None)
invalid_dates = int(df['datetime'].isna().sum())
if invalid_dates:
    print(f"Dropping {invalid_dates} records with invalid dates before embedding.")
df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
if df.empty:
    raise ValueError('The input must contain records with valid datetimes.')
df['text'] = df['text'].fillna('').astype(str)
df['quarter'] = df['datetime'].dt.to_period('Q').astype(str)
observed_quarters = sorted(df['quarter'].unique())
if len(observed_quarters) < 2:
    raise ValueError('The legacy two-period benchmark needs at least two observed quarters; use temporal_tracking.py for single-quarter tracking.')
# The benchmark contrasts the latest two observed quarters. Tracking below includes all quarters.
BASELINE_QUARTER, OBSERVATION_QUARTER = observed_quarters[-2:]
N_Q1 = (df['quarter'] == BASELINE_QUARTER).sum()
N_Q2 = (df['quarter'] == OBSERVATION_QUARTER).sum()
print(f"Loaded {len(df)} records across {len(observed_quarters)} observed calendar quarters.")
print(f"Benchmark baseline {BASELINE_QUARTER}: {N_Q1} issues")
print(f"Benchmark observation {OBSERVATION_QUARTER}: {N_Q2} issues")

docs = df['text'].tolist()

print("\n=== Step 2: Dense Transformer Embeddings (SBERT) ===")
import torch
embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
embeddings = embed_model.encode(docs, show_progress_bar=True, batch_size=64, normalize_embeddings=True)
print(f"Computed dense semantic vectors: {embeddings.shape} (384D)")

print("\n=== Step 3: Dimensionality Reduction via UMAP (384D -> 5D and 2D) ===")
umap_5d = umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric='cosine', random_state=42)
embeddings_5d = umap_5d.fit_transform(embeddings)

umap_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
embeddings_2d = umap_2d.fit_transform(embeddings)
df['umap_x'] = embeddings_2d[:, 0]
df['umap_y'] = embeddings_2d[:, 1]

print("\n=== Step 4: Model Execution ===")
K_CLUSTERS = 10

# 4A. Classical Generative Baseline: Latent Dirichlet Allocation (LDA)
print("Running Classical Generative Baseline: LDA (Blei et al., 2003)...")
count_vec = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=2500)
X_counts = count_vec.fit_transform(docs)
lda = LatentDirichletAllocation(n_components=K_CLUSTERS, max_iter=25, random_state=42, learning_method='batch')
lda_doc_topic = lda.fit_transform(X_counts)
df['lda_cluster'] = np.argmax(lda_doc_topic, axis=1)

# Extract top keywords for LDA
lda_vocab = np.array(count_vec.get_feature_names_out())
lda_top_words = {}
for topic_idx, topic in enumerate(lda.components_):
    top_indices = topic.argsort()[::-1][:10]
    lda_top_words[topic_idx] = lda_vocab[top_indices].tolist()

# 4B. Discriminative Baseline: Sparse TF-IDF + K-Means
print("Running Discriminative Baseline: Sparse K-Means on TF-IDF...")
tfidf_vec = TfidfVectorizer(max_features=2500, stop_words='english', ngram_range=(1, 2))
X_tfidf = tfidf_vec.fit_transform(docs)
kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
df['kmeans_cluster'] = kmeans.fit_predict(X_tfidf)

# 4C. Proposed Pipeline: Joint HDBSCAN on 5D UMAP Manifold
print("Running Proposed Pipeline: Joint HDBSCAN on 5D UMAP Manifold...")
hdb = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean', cluster_selection_method='eom')
df['hdbscan_cluster'] = hdb.fit_predict(embeddings_5d)
num_hdb_clusters = len(set(df['hdbscan_cluster'])) - (1 if -1 in df['hdbscan_cluster'] else 0)
noise_count = (df['hdbscan_cluster'] == -1).sum()
print(f"HDBSCAN discovered {num_hdb_clusters} clusters with {noise_count} noise points ({noise_count/len(df)*100:.1f}%)")

print("\n=== Step 5: c-TF-IDF Topic Representations ===")
def compute_c_tfidf(documents_by_cluster, max_words=10):
    cluster_ids = sorted(list(documents_by_cluster.keys()))
    if not cluster_ids:
        return {}
    cluster_docs = [" ".join(documents_by_cluster[c]) for c in cluster_ids]
    
    cv = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=2500)
    X = cv.fit_transform(cluster_docs)
    words = np.array(cv.get_feature_names_out())
    
    tf = np.asarray(X.toarray(), dtype=np.float64)
    t = tf.sum(axis=1, keepdims=True)
    w = tf / np.maximum(t, 1.0)
    
    f = np.asarray((X > 0).sum(axis=0), dtype=np.float64).ravel()
    idf = np.log((len(cluster_ids) + 1.0) / (f + 1.0)) + 1.0
    c_tfidf_matrix = w * idf
    
    top_words = {}
    for idx, c in enumerate(cluster_ids):
        sorted_indices = np.argsort(c_tfidf_matrix[idx])[::-1][:max_words]
        top_words[c] = words[sorted_indices].tolist()
    return top_words

# c-TF-IDF for K-Means
km_docs = {c: [] for c in range(K_CLUSTERS)}
for _, row in df.iterrows():
    km_docs[row['kmeans_cluster']].append(row['text'])
km_top_words = compute_c_tfidf(km_docs, max_words=10)

# c-TF-IDF for HDBSCAN (excluding noise -1)
hdb_docs = {c: [] for c in set(df['hdbscan_cluster']) if c != -1}
for _, row in df.iterrows():
    if row['hdbscan_cluster'] != -1:
        hdb_docs[row['hdbscan_cluster']].append(row['text'])
hdb_top_words = compute_c_tfidf(hdb_docs, max_words=10)

print("\n=== Step 6: Quantitative Benchmark (LDA vs K-Means vs Proposed) ===")
def evaluate_coherence_and_diversity(top_words_dict, embed_model):
    all_words = []
    coherence_scores = []
    for c, words in top_words_dict.items():
        all_words.extend(words)
        w_emb = embed_model.encode(words, normalize_embeddings=True)
        sim_matrix = np.dot(w_emb, w_emb.T)
        n = len(words)
        upper_idx = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[upper_idx]
        coherence_scores.append(np.mean(pairwise_sims) if len(pairwise_sims) > 0 else 0)
    
    mean_coherence = float(np.mean(coherence_scores))
    topic_diversity = float(len(set(all_words)) / len(all_words)) if all_words else 0.0
    return mean_coherence, topic_diversity

lda_coherence, lda_diversity = evaluate_coherence_and_diversity(lda_top_words, embed_model)
km_coherence, km_diversity = evaluate_coherence_and_diversity(km_top_words, embed_model)
hdb_coherence, hdb_diversity = evaluate_coherence_and_diversity(hdb_top_words, embed_model)

# Silhouette Scores
non_noise = df['hdbscan_cluster'] != -1
hdb_sil = float(silhouette_score(embeddings_5d[non_noise], df.loc[non_noise, 'hdbscan_cluster']))
km_sil = float(silhouette_score(X_tfidf.toarray(), df['kmeans_cluster']))
lda_sil = float(silhouette_score(lda_doc_topic, df['lda_cluster']))

print(f"1. Classical LDA:     Coherence = {lda_coherence:.4f} | Diversity = {lda_diversity:.4f} | Silhouette = {lda_sil:.4f}")
print(f"2. Baseline K-Means:  Coherence = {km_coherence:.4f} | Diversity = {km_diversity:.4f} | Silhouette = {km_sil:.4f}")
print(f"3. Proposed Pipeline: Coherence = {hdb_coherence:.4f} | Diversity = {hdb_diversity:.4f} | Silhouette = {hdb_sil:.4f}")

print(f"\n=== Step 7: Benchmark Shift ({BASELINE_QUARTER} vs {OBSERVATION_QUARTER}) ===")
shift_records = []
for c, words in hdb_top_words.items():
    cnt_q1 = ((df['quarter'] == BASELINE_QUARTER) & (df['hdbscan_cluster'] == c)).sum()
    cnt_q2 = ((df['quarter'] == OBSERVATION_QUARTER) & (df['hdbscan_cluster'] == c)).sum()
    
    P_Q1 = cnt_q1 / N_Q1
    P_Q2 = cnt_q2 / N_Q2
    delta_K = P_Q2 - P_Q1
    pct_change = ((P_Q2 - P_Q1) / P_Q1 * 100) if P_Q1 > 0 else None
    
    if delta_K >= temporal_config.surge_threshold:
        category = "Emerging Surge"
    elif delta_K <= -temporal_config.surge_threshold:
        category = "Declining"
    elif P_Q1 >= temporal_config.persistence_threshold and P_Q2 >= temporal_config.persistence_threshold:
        category = "Persistent Chronic Root Cause"
    else:
        category = "Stable Operational Routine"
        
    topic_label = " / ".join(words[:4])
    shift_records.append({
        'Cluster_ID': c,
        'Top_Keywords': ", ".join(words[:6]),
        'Topic_Label': topic_label,
        'Count_Q1': int(cnt_q1),
        'Count_Q2': int(cnt_q2),
        'P_Q1': round(float(P_Q1), 4),
        'P_Q2': round(float(P_Q2), 4),
        'Delta_K': round(float(delta_K), 4),
        'Pct_Change': round(float(pct_change), 1) if pct_change is not None else None,
        'Category': category
    })

shift_df = pd.DataFrame(shift_records).sort_values('Delta_K', ascending=False).reset_index(drop=True)
print(shift_df[['Cluster_ID', 'Topic_Label', 'P_Q1', 'P_Q2', 'Delta_K', 'Category']])

print("\n=== Step 7B: Rolling Windows and All-Quarter Topic Trajectories ===")
topic_labels = {int(c): ' / '.join(words[:4]) for c, words in hdb_top_words.items()}
temporal_analysis = analyze_temporal_topics(df, topic_labels=topic_labels, config=temporal_config)
temporal_paths = export_temporal_analysis(
    temporal_analysis, output_dir=args.temporal_output_dir,
    make_plots=not args.no_temporal_plots,
)
assignment_dir = Path(args.temporal_output_dir)
assignment_columns = [c for c in ['id', 'datetime', 'quarter', 'hdbscan_cluster', 'lda_cluster', 'kmeans_cluster'] if c in df.columns]
df[assignment_columns].to_csv(assignment_dir / 'topic_assignments.csv', index=False)
(assignment_dir / 'topic_labels.json').write_text(json.dumps(topic_labels, indent=2), encoding='utf-8')
quarter_focus = focus_topic_quarter(temporal_analysis, quarter=args.focus_quarter)
temporal_paths.update(export_focus_quarter(
    quarter_focus, output_dir=args.temporal_output_dir,
    make_plots=not args.no_temporal_plots, top_n=6,
))
print(f"Within-quarter trajectories for {args.focus_quarter} (same input corpus):")
print(quarter_focus['trajectories'].to_string(index=False))
print(f"Temporal outputs and reusable per-ticket assignments saved to {assignment_dir}")
print('Tracking is retrospective: joint-space topics are fitted on the whole input corpus.')

print("\n=== Step 8: Controlled Anomaly Perturbation Recall Experiment ===")
synthetic_anomaly_texts = [
    "CUDA out of memory in PyTorch distributed training: NCCL kernel timeout failure during backpropagation",
    "GPU memory allocation failed OOM error in torch distributed training NCCL timeout",
    "PyTorch CUDA out of memory RuntimeError: CUDA error: out of memory in backward pass",
    "NCCL connection timeout in multi-GPU distributed data parallel training PyTorch CUDA failure",
    "Distributed PyTorch worker crashed with CUDA OOM memory limit exceeded in backward step"
] * 5  # 25 synthetic critical regression issues

synth_emb = embed_model.encode(synthetic_anomaly_texts, normalize_embeddings=True)
synth_5d = umap_5d.transform(synth_emb)

synth_df = pd.DataFrame({
    'text': synthetic_anomaly_texts,
    'quarter': OBSERVATION_QUARTER,
    'datetime': pd.Period(OBSERVATION_QUARTER, freq='Q').end_time.normalize()
})

benchmark_mask = df['quarter'].isin([BASELINE_QUARTER, OBSERVATION_QUARTER])
combined_df = pd.concat([df.loc[benchmark_mask, ['text', 'quarter']], synth_df[['text', 'quarter']]], ignore_index=True)
combined_5d = np.vstack([embeddings_5d[benchmark_mask.to_numpy()], synth_5d])

hdb_synth = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean', cluster_selection_method='eom')
combined_df['cluster'] = hdb_synth.fit_predict(combined_5d)

N_Q1_s = (combined_df['quarter'] == BASELINE_QUARTER).sum()
N_Q2_s = (combined_df['quarter'] == OBSERVATION_QUARTER).sum()

synth_shifts = []
for c in set(combined_df['cluster']):
    if c == -1: continue
    cq1 = ((combined_df['quarter'] == BASELINE_QUARTER) & (combined_df['cluster'] == c)).sum()
    cq2 = ((combined_df['quarter'] == OBSERVATION_QUARTER) & (combined_df['cluster'] == c)).sum()
    pq1 = cq1 / N_Q1_s
    pq2 = cq2 / N_Q2_s
    synth_shifts.append({'cluster': c, 'delta': pq2 - pq1, 'count_Q2': cq2})

synth_shift_df = pd.DataFrame(synth_shifts).sort_values('delta', ascending=False).reset_index(drop=True)
print("Top 3 detected temporal shifts after anomaly injection in Q2:")
print(synth_shift_df.head(3))

print("\n=== Step 9: Publication Figures Generation ===")
os.makedirs("figures", exist_ok=True)

# Figure 1: 2D UMAP Semantic Manifold (180-day stream)
plt.figure(figsize=(10, 7), dpi=300)
non_noise = df['hdbscan_cluster'] != -1

plt.scatter(df.loc[~non_noise, 'umap_x'], df.loc[~non_noise, 'umap_y'], 
            c='#e0e0e0', s=16, alpha=0.5, label='Unassigned Outliers (Noise)')

plt.scatter(df.loc[non_noise, 'umap_x'], df.loc[non_noise, 'umap_y'], 
            c=df.loc[non_noise, 'hdbscan_cluster'], cmap='tab10', s=30, alpha=0.85)

for c, words in list(hdb_top_words.items())[:6]:
    cx = df.loc[df['hdbscan_cluster'] == c, 'umap_x'].median()
    cy = df.loc[df['hdbscan_cluster'] == c, 'umap_y'].median()
    plt.annotate(f"Topic {c}:\n{words[0]}, {words[1]}", (cx, cy),
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#777777", alpha=0.9, lw=0.8),
                 fontsize=8.5, ha='center', weight='bold')

plt.title("2D UMAP Manifold of 180-Day Technical Incident Stream (HDBSCAN)", fontsize=13, weight='bold', pad=12)
plt.xlabel("UMAP Dimension 1", fontsize=11)
plt.ylabel("UMAP Dimension 2", fontsize=11)
plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
plt.tight_layout()
plt.savefig("figures/figure1_umap_clusters_180d.png", dpi=300)
plt.close()
print("Saved figures/figure1_umap_clusters_180d.png")

# Figure 2: Quarterly Temporal Shift (Q1 -> Q2)
plt.figure(figsize=(11, 6), dpi=300)
category_colors = {
    'Emerging Surge': '#d9534f',
    'Persistent Chronic Root Cause': '#f0ad4e',
    'Stable Operational Routine': '#5bc0de',
    'Declining': '#5cb85c'
}
bar_colors = [category_colors.get(cat, '#333333') for cat in shift_df['Category']]

plt.barh(range(len(shift_df)), shift_df['Delta_K'], color=bar_colors, edgecolor='#444444', lw=0.6)
plt.yticks(range(len(shift_df)), [f"T{row['Cluster_ID']}: {row['Topic_Label'][:26]}" for _, row in shift_df.iterrows()], fontsize=9)
plt.axvline(0, color='black', linewidth=1.0, linestyle='--')
plt.xlabel("Quarterly Representation Shift Delta (ΔK = P(k|Q2) - P(k|Q1))", fontsize=11, weight='bold')
plt.title(f"Quarterly Incident Shift Detection ({BASELINE_QUARTER} vs. {OBSERVATION_QUARTER})", fontsize=13, weight='bold', pad=12)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, label=label) for label, color in category_colors.items()]
plt.legend(handles=legend_elements, loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
plt.tight_layout()
plt.savefig("figures/figure2_temporal_shift_q1_q2.png", dpi=300)
plt.close()
print("Saved figures/figure2_temporal_shift_q1_q2.png")

# Figure 3: Tri-Model Benchmark: Classical LDA vs K-Means vs Proposed
plt.figure(figsize=(8, 5.5), dpi=300)
metrics = ['Topic Coherence', 'Topic Diversity', 'Silhouette Score']
lda_vals = [lda_coherence, lda_diversity, lda_sil]
km_vals = [km_coherence, km_diversity, km_sil]
prop_vals = [hdb_coherence, hdb_diversity, hdb_sil]

x = np.arange(len(metrics))
width = 0.25

plt.bar(x - width, lda_vals, width, label='Classical LDA (Blei et al.)', color='#d9534f', edgecolor='#333333', lw=0.6)
plt.bar(x, km_vals, width, label='Baseline (TF-IDF + K-Means)', color='#6c757d', edgecolor='#333333', lw=0.6)
plt.bar(x + width, prop_vals, width, label='Proposed (SBERT + UMAP + HDB)', color='#0275d8', edgecolor='#333333', lw=0.6)

for i in range(len(metrics)):
    plt.text(i - width, lda_vals[i] + 0.02, f"{lda_vals[i]:.2f}", ha='center', fontsize=8.5)
    plt.text(i, km_vals[i] + 0.02, f"{km_vals[i]:.2f}", ha='center', fontsize=8.5)
    plt.text(i + width, prop_vals[i] + 0.02, f"{prop_vals[i]:.2f}", ha='center', fontsize=8.5, weight='bold')

plt.ylabel("Score", fontsize=11, weight='bold')
plt.title("Tri-Model Benchmark: LDA vs. K-Means vs. Proposed Framework", fontsize=13, weight='bold', pad=12)
plt.xticks(x, metrics, fontsize=10, weight='bold')
plt.ylim(0, max(max(lda_vals), max(km_vals), max(prop_vals)) + 0.18)
plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
plt.tight_layout()
plt.savefig("figures/figure3_tri_model_benchmark.png", dpi=300)
plt.close()
print("Saved figures/figure3_tri_model_benchmark.png")

print("\n=== Step 10: Saving Summary and Data Artifacts ===")
shift_df.to_csv("quarterly_temporal_shifts.csv", index=False)

summary = {
    "dataset": str(args.data),
    "total_span_days": int((df['datetime'].max().normalize() - df['datetime'].min().normalize()).days + 1),
    "baseline_quarter": BASELINE_QUARTER,
    "observation_quarter": OBSERVATION_QUARTER,
    "temporal_tracking": temporal_analysis['summary'],
    "within_quarter_tracking": quarter_focus['summary'],
    "temporal_outputs": {key: str(value) for key, value in temporal_paths.items()},
    "legacy_field_note": "Q1/Q2 fields below denote the selected baseline/observation quarters, respectively.",
    "total_issues": int(len(df)),
    "Q1_baseline_issues": int(N_Q1),
    "Q2_observation_issues": int(N_Q2),
    "models": {
        "classical_lda": {
            "coherence": round(lda_coherence, 4),
            "diversity": round(lda_diversity, 4),
            "silhouette": round(lda_sil, 4)
        },
        "baseline_kmeans": {
            "coherence": round(km_coherence, 4),
            "diversity": round(km_diversity, 4),
            "silhouette": round(km_sil, 4)
        },
        "proposed_pipeline": {
            "coherence": round(hdb_coherence, 4),
            "diversity": round(hdb_diversity, 4),
            "silhouette": round(hdb_sil, 4),
            "num_topics": int(num_hdb_clusters),
            "noise_ratio": round(float(noise_count / len(df)), 4)
        }
    },
    "topics": [
        {
            "id": int(row['Cluster_ID']),
            "label": row['Topic_Label'],
            "keywords": row['Top_Keywords'],
            "p_q1": row['P_Q1'],
            "p_q2": row['P_Q2'],
            "delta_k": row['Delta_K'],
            "category": row['Category']
        } for _, row in shift_df.iterrows()
    ]
}

with open("quarterly_results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Quarterly experiment with LDA complete! All deliverables generated.")
