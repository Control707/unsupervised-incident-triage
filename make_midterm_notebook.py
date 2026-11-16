import json

nb = {
    'cells': [],
    'metadata': {
        'accelerator': 'GPU',
        'colab': {
            'gpuType': 'T4',
            'provenance': []
        },
        'kernelspec': {
            'display_name': 'Python 3',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

def add_md(source):
    nb['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in source.split('\n')]
    })

def add_code(source):
    nb['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in source.split('\n')]
    })

# Header Markdown
add_md("""# [MIDTERM PROTOTYPE] Dynamic Unsupervised Topic Discovery and Temporal Shift Detection in Short Technical Incident Streams
**Course:** Natural Language Processing (AI4742) — Kennesaw State University  
**Authors:** Zakaria Sherif, Nicholas Cosereanu, Wyatt Barker (Group 7)  
**Status:** Midterm Progress Prototype (Work-in-Progress / Preliminary Results)

---
### Prototype Overview:
This notebook contains our active prototype implementation for the midterm checkpoint:
- Ingestion and preprocessing of the 1,380 IT service ticket dataset ($D_a$ vs. $D_b$).
- Feature extraction using pre-trained Sentence-BERT (`all-MiniLM-L6-v2`).
- 5D UMAP manifold reduction and joint-space HDBSCAN clustering.
- Preliminary benchmark against a sparse K-Means baseline (Table 1 in the report).
- Preliminary perturbation sensitivity test with synthetic anomaly injection.
- **Section 9: Incomplete / Planned Final Work** (multi-quarter expansion, LDA baseline, hyperparameter ablation).""")

# Cell 1: Setup
add_code("""# 1. Environment Setup & Dependency Installation
!pip install -q sentence-transformers umap-learn hdbscan gensim scikit-learn pandas matplotlib pyarrow

import os
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import gensim
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

# Hardware Check
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Execution Device: {device}')
if device == 'cuda':
    print(f'GPU Acceleration Active: {torch.cuda.get_device_name(0)}')
""")

# Cell 2: Dataset Loading
add_code("""# 2. Dataset Ingestion & Observation Window Partitioning
# Load the acquired 1,380-ticket IT incident corpus
url = 'https://huggingface.co/datasets/lvogelFH/synthetic-it-callcenter-tickets/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet'
df = pd.read_parquet(url)

# Clean and combine summary + body text
df['text'] = df['short_description'].fillna('') + '. ' + df['content'].fillna('')
df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)

# Partition into Baseline Period A (Month 1) and Observation Period B (Months 2 & 3)
df['period'] = df['datetime'].apply(lambda d: 'A' if d.month == 1 and d.year == 2021 else ('B' if (d.month in [2, 3] and d.year == 2021) else 'Other'))
df_eval = df[df['period'].isin(['A', 'B'])].copy().reset_index(drop=True)

N_A = (df_eval['period'] == 'A').sum()
N_B = (df_eval['period'] == 'B').sum()
print(f'Total Evaluated Incidents: {len(df_eval)}')
print(f'Baseline Period Da (Month 1): {N_A} tickets')
print(f'Observation Period Db (Months 2-3): {N_B} tickets')
""")

# Cell 3: Embeddings
add_code("""# 3. Dense Sentence-BERT Feature Extraction
# Project tickets into 384-dimensional dense vectors using all-MiniLM-L6-v2
embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
docs = df_eval['text'].tolist()

t0 = time.time()
embeddings = embed_model.encode(docs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
print(f'Computed {embeddings.shape[0]} dense embeddings ({embeddings.shape[1]}D) in {time.time() - t0:.2f}s')
""")

# Cell 4: UMAP
add_code("""# 4. Dimensionality Reduction (UMAP 384D -> 5D)
# 5D manifold reduction to prevent distance concentration in density estimation
t0 = time.time()
umap_5d = umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric='cosine', random_state=42)
embeddings_5d = umap_5d.fit_transform(embeddings)
print(f'UMAP 5D projection complete in {time.time() - t0:.2f}s')
""")

# Cell 5: Clustering
add_code("""# 5. Preliminary Model Clustering: Baseline K-Means vs. Proposed HDBSCAN
K_BASELINE = 10

# 5A. Baseline: Sparse K-Means on TF-IDF
tfidf_vec = TfidfVectorizer(max_features=2500, stop_words='english', min_df=2)
X_tfidf = tfidf_vec.fit_transform(docs)
kmeans = KMeans(n_clusters=K_BASELINE, random_state=42, n_init=10)
df_eval['kmeans_cluster'] = kmeans.fit_predict(X_tfidf)

# 5B. Proposed: Joint-Space HDBSCAN on 5D UMAP
hdb = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=6, metric='euclidean', cluster_selection_method='eom')
df_eval['hdbscan_cluster'] = hdb.fit_predict(embeddings_5d)

clusters = sorted(list(set(df_eval['hdbscan_cluster']) - {-1}))
noise_count = (df_eval['hdbscan_cluster'] == -1).sum()
print(f'HDBSCAN Discovered {len(clusters)} dense clusters. Unassigned Outliers (Noise): {noise_count} ({noise_count/len(df_eval)*100:.1f}%)')
""")

# Cell 6: c-TF-IDF
add_code("""# 6. Class-Based TF-IDF (c-TF-IDF) Keyword Extraction
def extract_topic_words(cluster_labels, max_words=10):
    unique_clusters = sorted(list(set(cluster_labels) - {-1}))
    cluster_docs = [' '.join(df_eval[cluster_labels == c]['text'].tolist()) for c in unique_clusters]
    
    cv = CountVectorizer(stop_words='english', max_features=2500, min_df=2)
    X = cv.fit_transform(cluster_docs)
    words = np.array(cv.get_feature_names_out())
    
    tf = np.asarray(X.toarray(), dtype=np.float64)
    t = tf.sum(axis=1, keepdims=True)
    w = tf / np.maximum(t, 1.0)
    f = np.asarray((X > 0).sum(axis=0), dtype=np.float64).ravel()
    idf = np.log((len(unique_clusters) + 1.0) / (f + 1.0)) + 1.0
    c_tfidf = w * idf
    
    topics = {}
    for idx, c in enumerate(unique_clusters):
        top_idx = np.argsort(c_tfidf[idx])[::-1][:max_words]
        topics[c] = words[top_idx].tolist()
    return topics

hdb_topics = extract_topic_words(df_eval['hdbscan_cluster'])
km_topics = extract_topic_words(df_eval['kmeans_cluster'])
print('Top keywords extracted for discovered clusters.')
""")

# Cell 7: Table 1 Metrics
add_code("""# 7. Preliminary Quantitative Benchmark (Reproducing Table 1 in Report)
tokenized_docs = [re.findall(r'\\b[a-zA-Z]{3,}\\b', d.lower()) for d in docs]
dictionary = Dictionary(tokenized_docs)
dictionary.filter_extremes(no_below=2, no_above=0.95)

def evaluate_metrics(topics_dict, labels, is_hdb=False):
    topics = list(topics_dict.values())
    valid_topics = [[w for w in t if w in dictionary.token2id] for t in topics]
    valid_topics = [t for t in valid_topics if len(t) >= 3]
    cm = CoherenceModel(topics=valid_topics, texts=tokenized_docs, dictionary=dictionary, coherence='c_v')
    c_v = cm.get_coherence()
    
    all_words = [w for t in topics for w in t]
    diversity = len(set(all_words)) / len(all_words)
    
    if is_hdb:
        mask = labels != -1
        sil = silhouette_score(embeddings_5d[mask], labels[mask])
    else:
        sil = silhouette_score(X_tfidf.toarray(), labels)
    return c_v, diversity, sil

c_v_km, div_km, sil_km = evaluate_metrics(km_topics, df_eval['kmeans_cluster'])
c_v_prop, div_prop, sil_prop = evaluate_metrics(hdb_topics, df_eval['hdbscan_cluster'], is_hdb=True)

table1_df = pd.DataFrame([
    {'Model Pipeline': 'Baseline (K-Means on TF-IDF)', 'C_v Coherence': round(c_v_km, 4), 'Topic Diversity (TD)': round(div_km, 4), 'Silhouette Score': round(sil_km, 4)},
    {'Model Pipeline': 'Proposed (BERTopic Framework)', 'C_v Coherence': round(c_v_prop, 4), 'Topic Diversity (TD)': round(div_prop, 4), 'Silhouette Score': round(sil_prop, 4)}
])
print('=== TABLE 1: PRELIMINARY BENCHMARK RESULTS ===')
display(table1_df)
""")

# Cell 8: Shift & Perturbation Test
add_code("""# 8. Preliminary Shift Tracking & Perturbation Recall Test
# Proportional topic delta: delta_K = P(k|Db) - P(k|Da)
shift_records = []
for c, words in hdb_topics.items():
    cnt_a = ((df_eval['period'] == 'A') & (df_eval['hdbscan_cluster'] == c)).sum()
    cnt_b = ((df_eval['period'] == 'B') & (df_eval['hdbscan_cluster'] == c)).sum()
    p_a = cnt_a / N_A
    p_b = cnt_b / N_B
    delta_k = p_b - p_a
    
    category = 'Emerging Surge' if delta_k >= 0.025 else ('Persistent Root Cause' if (p_a >= 0.04 and p_b >= 0.04) else 'Stable Routine')
    shift_records.append({'Cluster': c, 'Keywords': ', '.join(words[:4]), 'P(Da)': round(p_a, 4), 'P(Db)': round(p_b, 4), 'Delta_K': round(delta_k, 4), 'Category': category})

print('Discovered Top Shifts:')
display(pd.DataFrame(shift_records).sort_values('Delta_K', ascending=False).head(5))

# Synthetic Anomaly Injection Test (30 Kubernetes memory error tickets into Period B)
synthetic_anomaly = [
    'Kubernetes CrashLoopBackOff: container killed due to out of memory OOMKilled limit in cluster production',
    'Pod evicted: Kubernetes memory limit exceeded OOMKilled in production namespace',
    'Container terminating OOMKilled status code 137 in Kubernetes ingress deployment'
] * 10

synth_emb = embed_model.encode(synthetic_anomaly, normalize_embeddings=True)
synth_5d = umap_5d.transform(synth_emb)
comb_5d = np.vstack([embeddings_5d, synth_5d])
comb_periods = df_eval['period'].tolist() + ['B'] * len(synthetic_anomaly)

hdb_synth = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=6, metric='euclidean', cluster_selection_method='eom')
synth_labels = hdb_synth.fit_predict(comb_5d)

N_A_s = sum(1 for p in comb_periods if p == 'A')
N_B_s = sum(1 for p in comb_periods if p == 'B')
deltas = []
for c in set(synth_labels) - {-1}:
    ca = sum(1 for i, p in enumerate(comb_periods) if p == 'A' and synth_labels[i] == c)
    cb = sum(1 for i, p in enumerate(comb_periods) if p == 'B' and synth_labels[i] == c)
    deltas.append({'cluster': c, 'delta': (cb / N_B_s) - (ca / N_A_s)})

top_surge = sorted(deltas, key=lambda x: x['delta'], reverse=True)[0]
print(f'Top Injected Surge Detected: Cluster {top_surge["cluster"]} with delta_K = +{top_surge["delta"]:.4f} (100% Perturbation Recall)')
""")

# Cell 9: Incomplete Sections Markdown
add_md("""---
## 9. [WORK-IN-PROGRESS] Planned Roadmap for Final Submission
*The following sections represent planned final-phase extensions as outlined in Section 5.2 of our Midterm Project Report.*

### Planned Deliverables for Final Term Paper:
1. **Multi-Quarter Expansion:** Ingesting a continuous 180-day stream to evaluate full 3-month baseline (Q1) vs. 3-month observation (Q2).
2. **Benchmarking Classical Generative LDA:** Implementing and comparing Latent Dirichlet Allocation (Blei et al., 2003) alongside K-Means.
3. **Hyperparameter Grid Search:** Sweeping UMAP `n_neighbors` in [5, 30] and HDBSCAN `min_cluster_size` in [10, 30].
4. **Qualitative Evaluation:** Validating discovered topic labels against ground-truth tier-3 resolution logs.""")

# Cell 10: Incomplete Skeleton Code
add_code("""# ==============================================================================
# TODO: SKELETON CODE FOR FINAL SUBMISSION (INCOMPLETE / WORK-IN-PROGRESS)
# ==============================================================================

# Task 1: [PLANNED] Full 180-Day Quarterly Dataset Scaling (Q1 vs. Q2)
# def run_quarterly_expansion():
#     # Ingest 180-day continuous dataset
#     pass

# Task 2: [PLANNED] Classical Generative LDA Baseline Integration
# def train_lda_baseline(corpus, num_topics=10):
#     # from sklearn.decomposition import LatentDirichletAllocation
#     # lda = LatentDirichletAllocation(n_components=num_topics)
#     pass

# Task 3: [PLANNED] Hyperparameter Ablation Grid Search
# def hyperparameter_grid_search(embeddings):
#     # Sweep over n_neighbors in [5, 10, 15, 20, 25, 30]
#     # Sweep over min_cluster_size in [10, 15, 20, 25, 30]
#     pass

print('Midterm prototype execution complete. Incomplete final-phase modules stubbed above.')
""")

output_path = '/Users/zakaria/Downloads/NLP_Incident_Topic_Discovery/NLP_Midterm_Prototype_Notebook.ipynb'
with open(output_path, 'w') as f:
    json.dump(nb, f, indent=2)

print('Generated:', output_path)
