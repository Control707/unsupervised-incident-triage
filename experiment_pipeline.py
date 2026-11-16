import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import umap
import hdbscan

# Set styling for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

print("=== Step 1: Loading and Preprocessing Technical Incident Dataset ===")
data_path = "/content/it_incident_tickets.csv" if os.path.exists("/content/it_incident_tickets.csv") else "it_incident_tickets.csv"
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} incident tickets.")

# Clean and combine text
df['text'] = df['short_description'].fillna('') + ". " + df['content'].fillna('')
df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)

# Define baseline period A (Jan 2021) and observation period B (Feb-Mar 2021)
df['period'] = df['datetime'].apply(lambda d: 'A' if d.month == 1 and d.year == 2021 else ('B' if (d.month in [2, 3] and d.year == 2021) else 'Other'))
df_eval = df[df['period'].isin(['A', 'B'])].copy().reset_index(drop=True)
print(f"Filtered to temporal evaluation windows: Period A (Baseline): {(df_eval['period'] == 'A').sum()} tickets, Period B (Observation): {(df_eval['period'] == 'B').sum()} tickets.")

print("\n=== Step 2: Dense Document Embeddings (SBERT) ===")
embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1' else 'cpu')
docs = df_eval['text'].tolist()
embeddings = embed_model.encode(docs, show_progress_bar=True, batch_size=64, normalize_embeddings=True)
print(f"Generated dense embeddings of shape: {embeddings.shape} (384-dimensional)")

print("\n=== Step 3: Dimensionality Reduction via UMAP (384 -> 5D) ===")
# 5D for density clustering (addresses curse of dimensionality)
umap_reducer_5d = umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric='cosine', random_state=42)
embeddings_5d = umap_reducer_5d.fit_transform(embeddings)

# 2D for publication-quality scatter plot visualization
umap_reducer_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
embeddings_2d = umap_reducer_2d.fit_transform(embeddings)
df_eval['umap_x'] = embeddings_2d[:, 0]
df_eval['umap_y'] = embeddings_2d[:, 1]

print("\n=== Step 4: Clustering Pipeline Execution ===")
# 4A: Baseline: Sparse TF-IDF + K-Means
print("Running Baseline: Sparse K-Means on TF-IDF...")
tfidf_vec = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
X_tfidf = tfidf_vec.fit_transform(docs)

# Find optimal or comparable K (e.g., K=10)
K_CLUSTERS = 10
kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
df_eval['kmeans_cluster'] = kmeans.fit_predict(X_tfidf)

# 4B: Proposed: Joint HDBSCAN Clustering on 5D UMAP Space
print("Running Proposed: HDBSCAN on 5D UMAP Manifold...")
hdb = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=6, metric='euclidean', cluster_selection_method='eom')
df_eval['hdbscan_cluster'] = hdb.fit_predict(embeddings_5d)

num_clusters = len(set(df_eval['hdbscan_cluster'])) - (1 if -1 in df_eval['hdbscan_cluster'] else 0)
noise_count = (df_eval['hdbscan_cluster'] == -1).sum()
print(f"HDBSCAN discovered {num_clusters} dense clusters. Noise/Outlier points: {noise_count} ({noise_count/len(df_eval)*100:.1f}%)")

print("\n=== Step 5: Class-based TF-IDF (c-TF-IDF) Topic Representation ===")
def compute_c_tfidf(documents_by_cluster, max_words=10):
    cluster_ids = sorted(list(documents_by_cluster.keys()))
    cluster_docs = [" ".join(documents_by_cluster[c]) for c in cluster_ids]
    
    cv = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=2500)
    X = cv.fit_transform(cluster_docs)
    words = np.array(cv.get_feature_names_out())
    
    # Calculate c-TF-IDF using standard 2D numpy arrays
    tf = np.asarray(X.toarray(), dtype=np.float64)
    t = tf.sum(axis=1, keepdims=True)
    w = tf / np.maximum(t, 1.0)
    
    # Inverse cluster frequency
    f = np.asarray((X > 0).sum(axis=0), dtype=np.float64).ravel()
    idf = np.log((len(cluster_ids) + 1.0) / (f + 1.0)) + 1.0
    c_tfidf_matrix = w * idf  # element-wise broadcast
    
    top_words = {}
    for idx, c in enumerate(cluster_ids):
        sorted_indices = np.argsort(c_tfidf_matrix[idx])[::-1][:max_words]
        top_words[c] = words[sorted_indices].tolist()
    return top_words

# Compute c-TF-IDF for HDBSCAN clusters (excluding noise -1)
hdb_docs = {c: [] for c in set(df_eval['hdbscan_cluster']) if c != -1}
for _, row in df_eval.iterrows():
    if row['hdbscan_cluster'] != -1:
        hdb_docs[row['hdbscan_cluster']].append(row['text'])
hdb_top_words = compute_c_tfidf(hdb_docs, max_words=10)

# Compute c-TF-IDF for KMeans clusters
km_docs = {c: [] for c in range(K_CLUSTERS)}
for _, row in df_eval.iterrows():
    km_docs[row['kmeans_cluster']].append(row['text'])
km_top_words = compute_c_tfidf(km_docs, max_words=10)

print("\n=== Step 6: Quantitative Metrics Calculation ===")
def evaluate_coherence_and_diversity(top_words_dict, embed_model):
    all_words = []
    coherence_scores = []
    
    for c, words in top_words_dict.items():
        all_words.extend(words)
        # Pairwise cosine similarity among top word embeddings
        w_emb = embed_model.encode(words, normalize_embeddings=True)
        sim_matrix = np.dot(w_emb, w_emb.T)
        # Upper triangle mean (excluding diagonal)
        n = len(words)
        upper_idx = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[upper_idx]
        coherence_scores.append(np.mean(pairwise_sims) if len(pairwise_sims) > 0 else 0)
    
    mean_coherence = float(np.mean(coherence_scores))
    topic_diversity = float(len(set(all_words)) / len(all_words)) if all_words else 0.0
    return mean_coherence, topic_diversity

hdb_coherence, hdb_diversity = evaluate_coherence_and_diversity(hdb_top_words, embed_model)
km_coherence, km_diversity = evaluate_coherence_and_diversity(km_top_words, embed_model)

# Silhouette Scores
non_noise = df_eval['hdbscan_cluster'] != -1
hdb_sil_5d = float(silhouette_score(embeddings_5d[non_noise], df_eval.loc[non_noise, 'hdbscan_cluster']))
km_sil = float(silhouette_score(X_tfidf.toarray(), df_eval['kmeans_cluster']))

print(f"Baseline K-Means:   Coherence = {km_coherence:.4f} | Diversity = {km_diversity:.4f} | Silhouette = {km_sil:.4f}")
print(f"Proposed UMAP+HDB:  Coherence = {hdb_coherence:.4f} | Diversity = {hdb_diversity:.4f} | Silhouette = {hdb_sil_5d:.4f}")

print("\n=== Step 7: Temporal Shift & Root Cause Persistence Analysis ===")
# Calculate proportional representation in period A vs B for HDBSCAN clusters
N_A = (df_eval['period'] == 'A').sum()
N_B = (df_eval['period'] == 'B').sum()

shift_records = []
for c, words in hdb_top_words.items():
    count_A = ((df_eval['period'] == 'A') & (df_eval['hdbscan_cluster'] == c)).sum()
    count_B = ((df_eval['period'] == 'B') & (df_eval['hdbscan_cluster'] == c)).sum()
    
    P_A = count_A / N_A
    P_B = count_B / N_B
    delta_K = P_B - P_A
    pct_change = ((P_B - P_A) / P_A * 100) if P_A > 0 else float('inf')
    
    # Classify shift dynamics
    if delta_K >= 0.025:
        category = "Emerging Surge"
    elif delta_K <= -0.025:
        category = "Declining / Resolved"
    elif P_A >= 0.04 and P_B >= 0.04:
        category = "Persistent Root Cause"
    else:
        category = "Stable Routine"
        
    topic_label = " / ".join(words[:4])
    shift_records.append({
        'Cluster_ID': c,
        'Top_Keywords': ", ".join(words[:6]),
        'Topic_Label': topic_label,
        'Count_Period_A': int(count_A),
        'Count_Period_B': int(count_B),
        'P_A': round(float(P_A), 4),
        'P_B': round(float(P_B), 4),
        'Delta_K': round(float(delta_K), 4),
        'Pct_Change': round(float(pct_change), 1) if pct_change != float('inf') else 999.0,
        'Category': category
    })

shift_df = pd.DataFrame(shift_records).sort_values('Delta_K', ascending=False).reset_index(drop=True)
print(shift_df[['Cluster_ID', 'Topic_Label', 'P_A', 'P_B', 'Delta_K', 'Category']])

print("\n=== Step 8: Synthetic Perturbation / Shift Sensitivity Experiment ===")
# Inject synthetic surge into Period B to test detection sensitivity
synthetic_anomaly_texts = [
    "Kubernetes CrashLoopBackOff: container killed due to out of memory OOMKilled limit in cluster production",
    "Pod evicted: Kubernetes memory limit exceeded OOMKilled in production namespace",
    "Kubernetes node memory pressure CrashLoopBackOff container killed repeatedly",
    "Container terminating OOMKilled status code 137 in Kubernetes ingress deployment",
    "Production Pod crashing with CrashLoopBackOff OOMKilled error memory allocation failed"
] * 6  # 30 injected anomaly incident tickets

synth_df = pd.DataFrame({
    'text': synthetic_anomaly_texts,
    'period': 'B',
    'datetime': pd.to_datetime('2021-03-25 10:00:00')
})

# Embed and project injected items
synth_emb = embed_model.encode(synthetic_anomaly_texts, normalize_embeddings=True)
synth_5d = umap_reducer_5d.transform(synth_emb)

# Combine and test detection
combined_eval = pd.concat([df_eval[['text', 'period']], synth_df[['text', 'period']]], ignore_index=True)
all_embeddings_5d = np.vstack([embeddings_5d, synth_5d])

# Re-run HDBSCAN on injected stream
hdb_synth = hdbscan.HDBSCAN(min_cluster_size=12, min_samples=5, metric='euclidean', cluster_selection_method='eom')
combined_eval['cluster'] = hdb_synth.fit_predict(all_embeddings_5d)

N_A_synth = (combined_eval['period'] == 'A').sum()
N_B_synth = (combined_eval['period'] == 'B').sum()

synth_shift = []
for c in set(combined_eval['cluster']):
    if c == -1: continue
    cnt_a = ((combined_eval['period'] == 'A') & (combined_eval['cluster'] == c)).sum()
    cnt_b = ((combined_eval['period'] == 'B') & (combined_eval['cluster'] == c)).sum()
    p_a = cnt_a / N_A_synth
    p_b = cnt_b / N_B_synth
    synth_shift.append({'cluster': c, 'delta': p_b - p_a, 'count_B': cnt_b})

synth_shift_df = pd.DataFrame(synth_shift).sort_values('delta', ascending=False).reset_index(drop=True)
print("Top 3 detected temporal shifts after anomaly injection:")
print(synth_shift_df.head(3))

print("\n=== Step 9: Generating Publication-Quality Figures ===")
os.makedirs("figures", exist_ok=True)

# Figure 1: 2D UMAP Semantic Clusters
plt.figure(figsize=(10, 7), dpi=300)
palette = sns.color_palette("tab10", n_colors=num_clusters)
non_noise = df_eval['hdbscan_cluster'] != -1

# Plot noise in light gray
plt.scatter(df_eval.loc[~non_noise, 'umap_x'], df_eval.loc[~non_noise, 'umap_y'], 
            c='#e0e0e0', s=15, alpha=0.5, label='Unassigned Noise / Outliers')

# Plot clusters
scatter = plt.scatter(df_eval.loc[non_noise, 'umap_x'], df_eval.loc[non_noise, 'umap_y'], 
                      c=df_eval.loc[non_noise, 'hdbscan_cluster'], cmap='tab10', s=28, alpha=0.85)

# Annotate top cluster centroids
for c, words in list(hdb_top_words.items())[:6]:
    cx = df_eval.loc[df_eval['hdbscan_cluster'] == c, 'umap_x'].median()
    cy = df_eval.loc[df_eval['hdbscan_cluster'] == c, 'umap_y'].median()
    plt.annotate(f"Topic {c}:\n{words[0]}, {words[1]}", (cx, cy),
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#777777", alpha=0.9, lw=0.8),
                 fontsize=8.5, ha='center', weight='bold')

plt.title("Dense 2D UMAP Manifold of Technical Incident Clusters (HDBSCAN)", fontsize=13, weight='bold', pad=12)
plt.xlabel("UMAP Dimension 1", fontsize=11)
plt.ylabel("UMAP Dimension 2", fontsize=11)
plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
plt.tight_layout()
plt.savefig("figures/figure1_umap_clusters.png", dpi=300)
plt.close()
print("Saved figures/figure1_umap_clusters.png")

# Figure 2: Temporal Shift Delta Bar Chart
plt.figure(figsize=(11, 6), dpi=300)
category_colors = {
    'Emerging Surge': '#d9534f',        # Red/Coral
    'Persistent Root Cause': '#f0ad4e', # Amber/Gold
    'Stable Routine': '#5bc0de',        # Blue
    'Declining / Resolved': '#5cb85c'   # Green
}
bar_colors = [category_colors.get(cat, '#333333') for cat in shift_df['Category']]

bars = plt.barh(range(len(shift_df)), shift_df['Delta_K'], color=bar_colors, edgecolor='#444444', lw=0.6)
plt.yticks(range(len(shift_df)), [f"T{row['Cluster_ID']}: {row['Topic_Label'][:28]}" for _, row in shift_df.iterrows()], fontsize=9)
plt.axvline(0, color='black', linewidth=1.0, linestyle='--')
plt.xlabel("Proportional Topic Shift Delta (ΔK = P(k|Db) - P(k|Da))", fontsize=11, weight='bold')
plt.title("Temporal Shift Detection Across Incident Observation Windows (Da -> Db)", fontsize=13, weight='bold', pad=12)

# Custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, label=label) for label, color in category_colors.items()]
plt.legend(handles=legend_elements, loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
plt.tight_layout()
plt.savefig("figures/figure2_temporal_shift.png", dpi=300)
plt.close()
print("Saved figures/figure2_temporal_shift.png")

# Figure 3: Baseline vs. Proposed Comparison
plt.figure(figsize=(7, 5), dpi=300)
metrics = ['Topic Coherence', 'Topic Diversity', 'Silhouette Score']
baseline_vals = [km_coherence, km_diversity, km_sil]
proposed_vals = [hdb_coherence, hdb_diversity, hdb_sil_5d]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, baseline_vals, width, label='Baseline (TF-IDF + K-Means)', color='#6c757d', edgecolor='#333333', lw=0.6)
plt.bar(x + width/2, proposed_vals, width, label='Proposed (SBERT + UMAP + HDBSCAN)', color='#0275d8', edgecolor='#333333', lw=0.6)

for i in range(len(metrics)):
    plt.text(i - width/2, baseline_vals[i] + 0.02, f"{baseline_vals[i]:.2f}", ha='center', fontsize=9)
    plt.text(i + width/2, proposed_vals[i] + 0.02, f"{proposed_vals[i]:.2f}", ha='center', fontsize=9, weight='bold')

plt.ylabel("Score", fontsize=11, weight='bold')
plt.title("Quantitative Model Benchmark Comparison", fontsize=13, weight='bold', pad=12)
plt.xticks(x, metrics, fontsize=10, weight='bold')
plt.ylim(0, max(max(baseline_vals), max(proposed_vals)) + 0.15)
plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
plt.tight_layout()
plt.savefig("figures/figure3_baseline_vs_proposed.png", dpi=300)
plt.close()
print("Saved figures/figure3_baseline_vs_proposed.png")

print("\n=== Step 10: Saving Structured Results and Artifacts ===")
shift_df.to_csv("temporal_shift_table.csv", index=False)

summary = {
    "num_tickets": int(len(df_eval)),
    "period_A_tickets": int(N_A),
    "period_B_tickets": int(N_B),
    "num_topics_discovered": int(num_clusters),
    "noise_ratio": round(float(noise_count / len(df_eval)), 4),
    "baseline_kmeans": {
        "coherence": round(km_coherence, 4),
        "diversity": round(km_diversity, 4),
        "silhouette": round(km_sil, 4)
    },
    "proposed_pipeline": {
        "coherence": round(hdb_coherence, 4),
        "diversity": round(hdb_diversity, 4),
        "silhouette": round(hdb_sil_5d, 4)
    },
    "topics": [
        {
            "id": int(row['Cluster_ID']),
            "label": row['Topic_Label'],
            "keywords": row['Top_Keywords'],
            "delta_k": row['Delta_K'],
            "category": row['Category']
        } for _, row in shift_df.iterrows()
    ]
}

with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("All experimental results, tables, and figures have been generated successfully!")
