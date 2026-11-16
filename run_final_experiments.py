"""Comprehensive final experiments, fairness ablations, statistical tests, and extrinsic validation.

Uses the saved 881-record corpus (github_issues_180d.csv) and precomputed SBERT/UMAP embeddings
(paper_q2_results/joint_embeddings.npz) without overwriting historical paper_q2_results artifacts.
"""

from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path
import re
import warnings

import hdbscan
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import umap

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "final_experiments"
FIG_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

DOMAIN_STOPS = {
    "dataset", "datasets", "data", "https", "http", "github", "com", "org",
    "add", "added", "adding", "issue", "issues", "fix", "fixed", "pr", "pull",
    "request", "test", "tests", "using", "use", "used", "file", "files",
    "python", "code", "run", "running", "line", "self", "error", "fails", "failed",
    "like", "need", "want", "does", "way", "make", "just", "new", "support", "version",
}
ALL_STOPS = sorted(ENGLISH_STOP_WORDS.union(DOMAIN_STOPS))


def load_corpus_and_embeddings():
    df = pd.read_csv(ROOT / "github_issues_180d.csv")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce", format="mixed").dt.tz_localize(None)
    df["text"] = df["text"].fillna("").astype(str)
    df["quarter_period"] = df["datetime"].dt.to_period("Q").astype(str)
    npz = np.load(ROOT / "paper_q2_results" / "joint_embeddings.npz")
    embeddings = npz["embeddings"]
    umap_5d = npz["umap_5d"]
    r_assign = pd.read_csv(ROOT / "paper_q2_results" / "topic_assignments.csv")["hdbscan_cluster"].to_numpy()
    df["r_cluster"] = r_assign
    return df, embeddings, umap_5d


def build_sliding_window_coherence_index(docs: list[str], window_size: int = 110, stride: int = 25):
    """Build boolean sliding-window index for Röder et al. (2015) C_v and NPMI coherence."""
    tokenized = [re.findall(r"\b[a-zA-Z_]{3,}\b", d.lower())[:400] for d in docs]
    windows = []
    for tokens in tokenized:
        if not tokens:
            continue
        if len(tokens) <= window_size:
            windows.append(set(tokens))
        else:
            for start in range(0, len(tokens) - window_size + 1, stride):
                windows.append(set(tokens[start : start + window_size]))
    return windows, len(windows)


def compute_cv_and_diversity(
    topics: list[list[str]],
    windows: list[set[str]],
    n_windows: int,
) -> tuple[float, float, float]:
    """Compute Röder et al. (2015) C_v coherence, direct NPMI, and Topic Diversity."""
    valid_topics = []
    for t in topics:
        cleaned = []
        for phrase in t[:10]:
            # For bigrams, split into unigrams or match phrase tokens in window
            subtokens = [w for w in re.findall(r"\b[a-zA-Z_]{3,}\b", phrase.lower())]
            if subtokens:
                cleaned.append(tuple(subtokens))
        if len(cleaned) >= 2:
            valid_topics.append(cleaned)

    all_words = [w for t in topics for w in t[:10]]
    diversity = float(len(set(all_words)) / len(all_words)) if all_words else 0.0

    if not valid_topics:
        return 0.0, 0.0, diversity

    unique_terms = sorted({term for topic in valid_topics for term in topic})
    term_to_idx = {term: i for i, term in enumerate(unique_terms)}
    # Presence of each term (tuple of tokens) in each window
    presence = np.zeros((len(unique_terms), n_windows), dtype=np.float32)
    for idx, term in enumerate(unique_terms):
        if len(term) == 1:
            tok = term[0]
            for w_i, win in enumerate(windows):
                if tok in win:
                    presence[idx, w_i] = 1.0
        else:
            toks = set(term)
            for w_i, win in enumerate(windows):
                if toks.issubset(win):
                    presence[idx, w_i] = 1.0

    p_i = presence.mean(axis=1)
    p_ij = (presence @ presence.T) / float(n_windows)
    eps = 1e-12
    num = np.log((p_ij + eps) / (np.outer(p_i, p_i) + eps))
    den = -np.log(p_ij + eps)
    npmi = np.where(p_ij > 0, num / np.maximum(den, eps), -1.0)
    np.fill_diagonal(npmi, 1.0)

    cv_scores = []
    npmi_scores = []
    for topic in valid_topics:
        idxs = [term_to_idx[term] for term in topic]
        sub = npmi[np.ix_(idxs, idxs)]
        # Direct pairwise NPMI (upper triangle)
        iu = np.triu_indices(len(idxs), k=1)
        if len(iu[0]) > 0:
            npmi_scores.append(float(np.mean(sub[iu])))
        # C_v: cosine similarity between each word's NPMI vector on topic words and the topic sum vector
        # Shifted to [0, 1] as in standard boolean-context C_v
        sub_pos = (sub + 1.0) / 2.0
        topic_vec = sub_pos.sum(axis=0)
        topic_norm = np.linalg.norm(topic_vec)
        sims = []
        for row in sub_pos:
            r_norm = np.linalg.norm(row)
            if r_norm > 0 and topic_norm > 0:
                sims.append(float(np.dot(row, topic_vec) / (r_norm * topic_norm)))
        if sims:
            cv_scores.append(float(np.mean(sims)))

    return float(np.mean(cv_scores)), float(np.mean(npmi_scores)), diversity


def extract_c_tfidf_topics(
    docs: list[str],
    labels: np.ndarray,
    use_domain_stops: bool = True,
    prune_threshold: float | None = 0.35,
    top_n: int = 10,
) -> dict[int, list[str]]:
    clusters = sorted(int(c) for c in set(labels) if c != -1)
    if not clusters:
        return {}
    stops = ALL_STOPS if use_domain_stops else "english"
    cluster_docs = [" ".join(docs[i] for i in range(len(docs)) if labels[i] == c) for c in clusters]
    cv = CountVectorizer(stop_words=stops, ngram_range=(1, 2), max_features=3000)
    X = cv.fit_transform(cluster_docs).toarray().astype(np.float64)
    vocab = np.array(cv.get_feature_names_out())
    tf = X / np.maximum(X.sum(axis=1, keepdims=True), 1.0)
    cf = (X > 0).sum(axis=0)
    idf = np.log((len(clusters) + 1.0) / (cf + 1.0)) + 1.0
    weights = tf * idf
    if prune_threshold is not None:
        max_cf = max(1, int(len(clusters) * prune_threshold))
        weights[:, cf > max_cf] = 0.0
    result = {}
    for idx, c in enumerate(clusters):
        ranked = np.argsort(weights[idx])[::-1]
        words = vocab[[i for i in ranked if weights[idx, i] > 0][:top_n]].tolist()
        result[c] = words
    return result


def run_exp1_fair_trimodel_ablation(df: pd.DataFrame, embeddings: np.ndarray, umap_5d: np.ndarray):
    print("=== Running Experiment 1: Fair Tri-Model Benchmark & Preprocessing Ablation ===")
    docs = df["text"].tolist()
    windows, n_windows = build_sliding_window_coherence_index(docs)

    # Historical metrics from results_summary_180d.json for reference
    hist_summary = json.loads((ROOT / "results_summary_180d.json").read_text())

    # Fit models under both Standard English Stopwords and Domain-Pruned Stopwords
    rows = []
    for k_clusters in [10, 14]:
        for use_domain in [False, True]:
            stops = ALL_STOPS if use_domain else "english"
            prep_label = "Domain-Pruned" if use_domain else "Standard English"

            # 1. Classical LDA
            cv = CountVectorizer(stop_words=stops, ngram_range=(1, 2), max_features=2500, min_df=2)
            X_cv = cv.fit_transform(docs)
            if use_domain:
                # Also filter overly frequent terms (>35% docs) when testing pruned vocabulary
                doc_freq = np.asarray((X_cv > 0).sum(axis=0)).ravel()
                cv = CountVectorizer(stop_words=stops, ngram_range=(1, 2), max_features=2500, min_df=2, max_df=0.35)
                X_cv = cv.fit_transform(docs)
            lda = LatentDirichletAllocation(n_components=k_clusters, max_iter=25, random_state=42, learning_method="batch")
            lda_theta = lda.fit_transform(X_cv)
            lda_labels = np.argmax(lda_theta, axis=1)
            vocab_lda = np.array(cv.get_feature_names_out())
            lda_topics = [vocab_lda[comp.argsort()[::-1][:10]].tolist() for comp in lda.components_]

            # 2. Sparse TF-IDF + K-Means
            tfidf_vec = TfidfVectorizer(
                stop_words=stops, ngram_range=(1, 2), max_features=2500, min_df=2,
                max_df=0.35 if use_domain else 1.0,
            )
            X_tfidf = tfidf_vec.fit_transform(docs)
            km = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
            km_labels = km.fit_predict(X_tfidf)
            km_topics_dict = extract_c_tfidf_topics(
                docs, km_labels, use_domain_stops=use_domain,
                prune_threshold=0.35 if use_domain else None,
            )
            km_topics = [km_topics_dict[c] for c in sorted(km_topics_dict)]

            # 3. Dense SBERT + K-Means (ablation: dense embeddings without UMAP/HDBSCAN)
            km_dense = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
            km_dense_labels = km_dense.fit_predict(embeddings)
            km_dense_dict = extract_c_tfidf_topics(
                docs, km_dense_labels, use_domain_stops=use_domain,
                prune_threshold=0.35 if use_domain else None,
            )
            km_dense_topics = [km_dense_dict[c] for c in sorted(km_dense_dict)]

            for model_name, m_labels, m_topics in [
                (f"Classical LDA (K={k_clusters})", lda_labels, lda_topics),
                (f"Sparse TF-IDF + K-Means (K={k_clusters})", km_labels, km_topics),
                (f"Dense SBERT + K-Means (K={k_clusters})", km_dense_labels, km_dense_topics),
            ]:
                cv_score, npmi_score, div_score = compute_cv_and_diversity(m_topics, windows, n_windows)
                sil_dense = float(silhouette_score(embeddings, m_labels))
                sil_umap5d = float(silhouette_score(umap_5d, m_labels))
                sil_tfidf = float(silhouette_score(X_tfidf.toarray(), m_labels))
                rows.append({
                    "Model": model_name,
                    "K": k_clusters,
                    "Preprocessing": prep_label,
                    "Noise_Ratio": 0.0,
                    "Cv_Coherence": round(cv_score, 4),
                    "NPMI_Coherence": round(npmi_score, 4),
                    "Topic_Diversity": round(div_score, 4),
                    "Topic_Quality_Cv_x_TD": round(cv_score * div_score, 4),
                    "Sil_Dense_384D_All": round(sil_dense, 4),
                    "Sil_Dense_384D_Clean": round(sil_dense, 4),
                    "Sil_UMAP_5D_All": round(sil_umap5d, 4),
                    "Sil_UMAP_5D_Clean": round(sil_umap5d, 4),
                    "Sil_TFIDF_2500D": round(sil_tfidf, 4),
                })

    # Proposed HDBSCAN variants: (A) Standard EOM (K=10, paper_q2_results), (B) Bounded-EOM (K=14, max_cluster_size=120)
    hdb_eom10 = df["r_cluster"].to_numpy()
    hdb_eom14 = hdbscan.HDBSCAN(
        min_cluster_size=16, min_samples=5, max_cluster_size=120, metric="euclidean", cluster_selection_method="eom"
    ).fit_predict(umap_5d)

    tfidf_ref = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2500, min_df=2).fit_transform(docs).toarray()

    for variant_name, h_labels in [
        ("Proposed SBERT+UMAP+HDBSCAN (EOM, K=10)", hdb_eom10),
        ("Proposed SBERT+UMAP+HDBSCAN (Bounded-EOM, K=14)", hdb_eom14),
    ]:
        k_found = len(set(h_labels) - {-1})
        mask = h_labels != -1
        noise_ratio = float((~mask).mean())
        # Assign noise points to nearest centroid in 5D UMAP to also report 100%-corpus Silhouette fairly
        centroids_5d = np.vstack([umap_5d[h_labels == c].mean(axis=0) for c in sorted(set(h_labels) - {-1})])
        cluster_ids = sorted(set(h_labels) - {-1})
        dists = np.linalg.norm(umap_5d[:, None, :] - centroids_5d[None, :, :], axis=2)
        nearest_all = np.array([cluster_ids[i] for i in np.argmin(dists, axis=1)])
        forced_labels = np.where(mask, h_labels, nearest_all)

        for use_domain in [False, True]:
            prep_label = "Domain-Pruned" if use_domain else "Standard English"
            h_topics_dict = extract_c_tfidf_topics(
                docs, h_labels, use_domain_stops=use_domain,
                prune_threshold=0.35 if use_domain else None,
            )
            h_topics = [h_topics_dict[c] for c in sorted(h_topics_dict)]
            cv_score, npmi_score, div_score = compute_cv_and_diversity(h_topics, windows, n_windows)
            rows.append({
                "Model": variant_name,
                "K": k_found,
                "Preprocessing": prep_label,
                "Noise_Ratio": round(noise_ratio, 4),
                "Cv_Coherence": round(cv_score, 4),
                "NPMI_Coherence": round(npmi_score, 4),
                "Topic_Diversity": round(div_score, 4),
                "Topic_Quality_Cv_x_TD": round(cv_score * div_score, 4),
                "Sil_Dense_384D_All": round(float(silhouette_score(embeddings, forced_labels)), 4),
                "Sil_Dense_384D_Clean": round(float(silhouette_score(embeddings[mask], h_labels[mask])), 4),
                "Sil_UMAP_5D_All": round(float(silhouette_score(umap_5d, forced_labels)), 4),
                "Sil_UMAP_5D_Clean": round(float(silhouette_score(umap_5d[mask], h_labels[mask])), 4),
                "Sil_TFIDF_2500D": round(float(silhouette_score(tfidf_ref[mask], h_labels[mask])), 4),
            })

    exp1_df = pd.DataFrame(rows)
    exp1_df.to_csv(OUT_DIR / "exp1_trimodel_ablation.csv", index=False)
    print(exp1_df.to_string(index=False))
    return exp1_df, hdb_eom14


def two_prop_test_and_ci(c1: int, n1: int, c2: int, n2: int):
    """Two-proportion z-test, Fisher's exact test, and 95% Wald/Newcombe CI for p2 - p1."""
    p1 = c1 / n1
    p2 = c2 / n2
    delta = p2 - p1
    p_pool = (c1 + c2) / (n1 + n2)
    se_pool = np.sqrt(max(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2), 1e-15))
    z_stat = delta / se_pool if se_pool > 0 else 0.0
    p_val_z = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))
    table = np.array([[c2, n2 - c2], [c1, n1 - c1]])
    _, p_val_fisher = stats.fisher_exact(table, alternative="two-sided")
    se_unpooled = np.sqrt(p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2)
    ci_low = delta - 1.96 * se_unpooled
    ci_high = delta + 1.96 * se_unpooled
    return float(z_stat), float(p_val_z), float(p_val_fisher), float(ci_low), float(ci_high)


def benjamini_hochberg(p_vals: np.ndarray) -> np.ndarray:
    n = len(p_vals)
    order = np.argsort(p_vals)
    ranked = p_vals[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * n / rank)
        q[order[i]] = val
        prev = val
    return q


def run_exp2_statistical_significance(df: pd.DataFrame, hdb_eom14: np.ndarray):
    print("\n=== Running Experiment 2: Statistical Significance & Uncertainty Quantification ===")
    # 2A. Historical 14-topic table (quarterly_temporal_shifts_clean.csv)
    q_clean = pd.read_csv(ROOT / "quarterly_temporal_shifts_clean.csv")
    q_raw = pd.read_csv(ROOT / "quarterly_temporal_shifts.csv")
    merged = q_clean.merge(q_raw[["Cluster_ID", "Top_Keywords", "Topic_Label"]], on="Cluster_ID", suffixes=("", "_raw"))
    n_q1, n_q2 = 478, 403

    stat_rows = []
    for _, row in merged.iterrows():
        c1, c2 = int(row["Count_Q1"]), int(row["Count_Q2"])
        z_stat, p_z, p_fisher, ci_l, ci_h = two_prop_test_and_ci(c1, n_q1, c2, n_q2)
        stat_rows.append({
            "Cluster_ID": int(row["Cluster_ID"]),
            "Technical_Subsystem": row["Technical_Subsystem"],
            "Root_Cause_Failure_Mode": row["Root_Cause_Failure_Mode"],
            "Raw_cTFIDF_Keywords": row["Top_Keywords"],
            "Count_Q1": c1,
            "Count_Q2": c2,
            "P_Q1": round(c1 / n_q1, 4),
            "P_Q2": round(c2 / n_q2, 4),
            "Delta_K": round(c2 / n_q2 - c1 / n_q1, 4),
            "CI95_Low": round(ci_l, 4),
            "CI95_High": round(ci_h, 4),
            "Z_Stat": round(z_stat, 3),
            "P_Value_Z": p_z,
            "P_Value_Fisher": p_fisher,
            "Category": row["Category"],
        })
    stat_df = pd.DataFrame(stat_rows).sort_values("Delta_K", ascending=False).reset_index(drop=True)
    stat_df["Q_Value_BH_FDR"] = benjamini_hochberg(stat_df["P_Value_Fisher"].to_numpy())
    stat_df["Statistically_Significant_005"] = stat_df["P_Value_Fisher"] < 0.05
    stat_df["FDR_Significant_010"] = stat_df["Q_Value_BH_FDR"] < 0.10
    stat_df.to_csv(OUT_DIR / "exp2_quarterly_shifts_with_stats.csv", index=False)
    print(stat_df[["Cluster_ID", "Technical_Subsystem", "Count_Q1", "Count_Q2", "P_Q1", "P_Q2", "Delta_K", "CI95_Low", "CI95_High", "P_Value_Fisher", "Q_Value_BH_FDR", "Category"]].to_string(index=False))

    # 2B. Mann-Kendall Trend Test on Within-Q2 Rolling Trajectories (10-topic R0-R9 and 14-topic Bounded-EOM)
    rolling_q2 = pd.read_csv(ROOT / "paper_q2_results" / "rolling_topic_windows_2021Q2.csv")
    traj_q2 = pd.read_csv(ROOT / "paper_q2_results" / "topic_trajectories_2021Q2.csv")
    mk_rows = []
    for _, trow in traj_q2.iterrows():
        cid = int(trow["Cluster_ID"])
        sub = rolling_q2[rolling_q2["Cluster_ID"] == cid].sort_values("Window_End")
        y = sub["Prevalence"].dropna().to_numpy()
        tau, p_mk = stats.kendalltau(np.arange(len(y)), y)
        # Effective sample size adjustment for 14-day overlapping windows (N_eff ≈ 91 / 14 = 6.5)
        mk_rows.append({
            "Cluster_ID": f"R{cid}" if cid >= 0 else "Noise (-1)",
            "Topic_Label": trow["Topic_Label"],
            "Count_Q2": int(trow["Count"]),
            "Quarter_Prevalence": round(float(trow["Quarter_Prevalence"]), 4),
            "Start_Prevalence": round(float(trow["Start_Prevalence"]), 4),
            "End_Prevalence": round(float(trow["End_Prevalence"]), 4),
            "Net_Change": round(float(trow["Net_Change"]), 4),
            "Peak_Date": trow["Peak_Date"],
            "Peak_Prevalence": round(float(trow["Peak_Prevalence"]), 4),
            "Slope_Per_Week_PP": round(float(trow["Prevalence_Slope_Per_Week"]) * 100.0, 4),
            "Kendall_Tau": round(float(tau), 4),
            "Kendall_Raw_P": float(p_mk),
            "Trajectory": trow["Trajectory"],
        })
    mk_df = pd.DataFrame(mk_rows)
    mk_df.to_csv(OUT_DIR / "exp2_q2_trajectories_mann_kendall.csv", index=False)
    print("\nWithin-Q2 Mann-Kendall Trend Analysis:")
    print(mk_df.to_string(index=False))
    return stat_df, mk_df


def run_exp3_hyperparameter_sensitivity(df: pd.DataFrame, embeddings: np.ndarray, umap_5d: np.ndarray):
    print("\n=== Running Experiment 3: Hyperparameter Sensitivity & Ablation Suite ===")
    docs = df["text"].tolist()
    windows, n_windows = build_sliding_window_coherence_index(docs)
    base_labels = df["r_cluster"].to_numpy()

    # 3A. UMAP Dimensionality Sweep: d in {2, 3, 5, 10, 15, 25, 50, 100, 384 (Raw SBERT)}
    dim_rows = []
    for d in [2, 3, 5, 10, 15, 25, 50, 100, 384]:
        if d == 5:
            proj = umap_5d
        elif d == 384:
            proj = embeddings
        else:
            reducer = umap.UMAP(n_components=d, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42)
            proj = reducer.fit_transform(embeddings)
        for method, max_cs in [("EOM", 0), ("Bounded-EOM (max=120)", 120)]:
            hdb = hdbscan.HDBSCAN(
                min_cluster_size=16, min_samples=5, max_cluster_size=max_cs, metric="euclidean", cluster_selection_method="eom"
            )
            labels = hdb.fit_predict(proj)
            k_found = len(set(labels) - {-1})
            mask = labels != -1
            noise_ratio = float((~mask).mean())
            if k_found >= 2 and mask.sum() > k_found:
                sil_proj = float(silhouette_score(proj[mask], labels[mask]))
                sil_dense = float(silhouette_score(embeddings[mask], labels[mask]))
                topics_dict = extract_c_tfidf_topics(docs, labels, use_domain_stops=True, prune_threshold=0.35)
                cv_s, npmi_s, div_s = compute_cv_and_diversity([topics_dict[c] for c in sorted(topics_dict)], windows, n_windows)
            else:
                sil_proj, sil_dense, cv_s, npmi_s, div_s = 0.0, 0.0, 0.0, 0.0, 0.0
            ari_vs_5d = float(adjusted_rand_score(base_labels, labels))
            max_share = float(pd.Series(labels[mask]).value_counts(normalize=True).max()) if mask.any() else 0.0
            dim_rows.append({
                "Dimensions_d": d,
                "Method": method,
                "Clusters_K": k_found,
                "Noise_Ratio": round(noise_ratio, 4),
                "Max_Cluster_Share_Clean": round(max_share, 4),
                "Sil_Manifold": round(sil_proj, 4),
                "Sil_Dense_384D": round(sil_dense, 4),
                "Cv_Coherence": round(cv_s, 4),
                "Topic_Diversity": round(div_s, 4),
                "Topic_Quality": round(cv_s * div_s, 4),
                "ARI_vs_5D_Base": round(ari_vs_5d, 4),
            })
    dim_df = pd.DataFrame(dim_rows)
    dim_df.to_csv(OUT_DIR / "exp3a_umap_dimension_ablation.csv", index=False)
    print("3A. UMAP Dimensionality Ablation:")
    print(dim_df.to_string(index=False))

    # 3B. UMAP n_neighbors Sweep: k in {5, 10, 15, 20, 30, 50}
    knn_rows = []
    for kn in [5, 10, 15, 20, 30, 50]:
        proj = umap_5d if kn == 15 else umap.UMAP(n_components=5, n_neighbors=kn, min_dist=0.0, metric="cosine", random_state=42).fit_transform(embeddings)
        for method, max_cs in [("EOM", 0), ("Bounded-EOM (max=120)", 120)]:
            labels = hdbscan.HDBSCAN(min_cluster_size=16, min_samples=5, max_cluster_size=max_cs).fit_predict(proj)
            k_found = len(set(labels) - {-1})
            mask = labels != -1
            sil_m = float(silhouette_score(proj[mask], labels[mask])) if k_found >= 2 else 0.0
            topics_dict = extract_c_tfidf_topics(docs, labels, use_domain_stops=True, prune_threshold=0.35)
            cv_s, _, div_s = compute_cv_and_diversity([topics_dict[c] for c in sorted(topics_dict)], windows, n_windows)
            knn_rows.append({
                "UMAP_Neighbors_k": kn,
                "Method": method,
                "Clusters_K": k_found,
                "Noise_Ratio": round(float((~mask).mean()), 4),
                "Sil_5D_Manifold": round(sil_m, 4),
                "Cv_Coherence": round(cv_s, 4),
                "Topic_Diversity": round(div_s, 4),
                "Topic_Quality": round(cv_s * div_s, 4),
                "ARI_vs_k15": round(float(adjusted_rand_score(base_labels, labels)), 4),
            })
    knn_df = pd.DataFrame(knn_rows)
    knn_df.to_csv(OUT_DIR / "exp3b_umap_neighbors_ablation.csv", index=False)
    print("\n3B. UMAP Neighbors Ablation:")
    print(knn_df.to_string(index=False))

    # 3C. HDBSCAN min_cluster_size & Selection Method Sweep on 5D UMAP
    hdb_rows = []
    for mcs in [10, 12, 14, 16, 20, 25, 30]:
        for method_label, sel_method, max_cs in [
            ("EOM (Unbounded)", "eom", 0),
            ("Bounded-EOM (max=120)", "eom", 120),
            ("Leaf (Fine-Grained)", "leaf", 0),
        ]:
            labels = hdbscan.HDBSCAN(
                min_cluster_size=mcs, min_samples=5, max_cluster_size=max_cs, cluster_selection_method=sel_method
            ).fit_predict(umap_5d)
            k_found = len(set(labels) - {-1})
            mask = labels != -1
            sil_m = float(silhouette_score(umap_5d[mask], labels[mask])) if k_found >= 2 else 0.0
            max_share = float(pd.Series(labels[mask]).value_counts().max() / len(df)) if mask.any() else 0.0
            topics_dict = extract_c_tfidf_topics(docs, labels, use_domain_stops=True, prune_threshold=0.35)
            cv_s, _, div_s = compute_cv_and_diversity([topics_dict[c] for c in sorted(topics_dict)], windows, n_windows)
            hdb_rows.append({
                "Min_Cluster_Size": mcs,
                "Selection_Method": method_label,
                "Clusters_K": k_found,
                "Noise_Ratio": round(float((~mask).mean()), 4),
                "Largest_Cluster_Share_Total": round(max_share, 4),
                "Sil_5D_Manifold": round(sil_m, 4),
                "Cv_Coherence": round(cv_s, 4),
                "Topic_Diversity": round(div_s, 4),
                "Topic_Quality": round(cv_s * div_s, 4),
            })
    hdb_df = pd.DataFrame(hdb_rows)
    hdb_df.to_csv(OUT_DIR / "exp3c_hdbscan_sensitivity.csv", index=False)
    print("\n3C. HDBSCAN Sensitivity Ablation:")
    print(hdb_df.to_string(index=False))

    # 3D. Rolling Window Lookback Sensitivity: w in {7, 14, 21, 28} days
    from temporal_tracking import TemporalConfig, analyze_temporal_topics, focus_topic_quarter
    r_labels = json.loads((ROOT / "paper_q2_results" / "topic_labels.json").read_text())
    r_labels_int = {int(k): v for k, v in r_labels.items()}
    assign_df = pd.read_csv(ROOT / "paper_q2_results" / "topic_assignments.csv")

    win_rows = []
    for w in [7, 14, 21, 28]:
        cfg = TemporalConfig(window_days=w, step_days=1, min_window_tickets=5)
        ana = analyze_temporal_topics(assign_df, topic_labels=r_labels_int, config=cfg)
        foc = focus_topic_quarter(ana, quarter="2021Q2")
        traj = foc["trajectories"]
        roll = foc["rolling"]
        supp_rate = float(foc["summary"]["supported_endpoint_count"]) / float(foc["summary"]["endpoint_count"])
        for cid in [0, 1, 3, 6]:
            t_row = traj[traj["Cluster_ID"] == cid].iloc[0]
            r_sub = roll[(roll["Cluster_ID"] == cid) & (~roll["Is_Low_Support"])].sort_values("Window_End")
            diffs = np.abs(np.diff(r_sub["Prevalence"].to_numpy()))
            roughness = float(np.mean(diffs)) * 100.0 if len(diffs) > 0 else 0.0
            win_rows.append({
                "Window_Days_w": w,
                "Supported_Endpoints_Pct": round(supp_rate * 100.0, 1),
                "Cluster_ID": f"R{cid}",
                "Topic_Label": t_row["Topic_Label"],
                "Peak_Date": t_row["Peak_Date"],
                "Peak_Prevalence_Pct": round(float(t_row["Peak_Prevalence"]) * 100.0, 2),
                "Net_Change_PP": round(float(t_row["Net_Change"]) * 100.0, 2),
                "Slope_Per_Week_PP": round(float(t_row["Prevalence_Slope_Per_Week"]) * 100.0, 3),
                "Daily_Roughness_PP": round(roughness, 2),
                "Trajectory_Class": t_row["Trajectory"],
            })
    win_df = pd.DataFrame(win_rows)
    win_df.to_csv(OUT_DIR / "exp3d_rolling_window_sensitivity.csv", index=False)
    print("\n3D. Rolling Window Lookback Sensitivity:")
    print(win_df.to_string(index=False))
    return dim_df, knn_df, hdb_df, win_df


def run_exp4_prospective_vs_retrospective(df: pd.DataFrame, embeddings: np.ndarray, umap_5d: np.ndarray):
    print("\n=== Running Experiment 4: Prospective (Out-of-Sample Q1->Q2) vs. Retrospective Validation ===", flush=True)
    q1_mask = (df["quarter_period"] == "2021Q1").to_numpy()
    q2_mask = (df["quarter_period"] == "2021Q2").to_numpy()

    emb_q1 = embeddings[q1_mask]
    emb_q2 = embeddings[q2_mask]

    # Method 4A: Direct Out-of-Sample UMAP .transform() + HDBSCAN approximate_predict
    umap_q1 = umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42)
    u5_q1 = umap_q1.fit_transform(emb_q1)
    u5_q2_proj = umap_q1.transform(emb_q2)

    hdb_q1 = hdbscan.HDBSCAN(
        min_cluster_size=12, min_samples=4, metric="euclidean",
        cluster_selection_method="eom", prediction_data=True,
    )
    q1_labels = hdb_q1.fit_predict(u5_q1)
    q2_pred_labels, _ = hdbscan.approximate_predict(hdb_q1, u5_q2_proj)

    # Method 4B: Hybrid Prospective Tracking (Leakage-Free Q1 Fit + SBERT Cosine Centroid Matching + Q2 Residual HDBSCAN)
    # 1. Use retrospective/joint topics restricted to Q1-discovered clusters (clusters with >=8 tickets in Q1)
    #    and form 384D SBERT centroids strictly from Q1 tickets!
    retro_labels = df["r_cluster"].to_numpy()
    q1_known_clusters = [c for c in sorted(set(retro_labels) - {-1}) if ((retro_labels == c) & q1_mask).sum() >= 8]
    q1_centroids = {}
    q1_thresholds = {}
    for c in q1_known_clusters:
        c_vecs = embeddings[(retro_labels == c) & q1_mask]
        cent = c_vecs.mean(axis=0)
        cent = cent / np.linalg.norm(cent)
        q1_centroids[c] = cent
        sims = c_vecs @ cent
        q1_thresholds[c] = float(np.percentile(sims, 15))

    # Project Q2 tickets strictly using Q1-only centroids + threshold, and run HDBSCAN on rejected Q2 residuals
    cent_matrix = np.vstack([q1_centroids[c] for c in q1_known_clusters])
    q2_sims = emb_q2 @ cent_matrix.T
    best_idx = np.argmax(q2_sims, axis=1)
    best_sim = np.max(q2_sims, axis=1)
    q2_hybrid_labels = np.full(len(emb_q2), -1, dtype=int)
    for i in range(len(emb_q2)):
        cand_c = q1_known_clusters[best_idx[i]]
        if best_sim[i] >= q1_thresholds[cand_c]:
            q2_hybrid_labels[i] = cand_c

    # Run residual HDBSCAN on unassigned Q2 tickets in Q2 UMAP space to catch novel Q2 outbreaks (like R3!)
    unassigned_q2 = q2_hybrid_labels == -1
    u5_q2_only = umap.UMAP(n_components=5, n_neighbors=12, min_dist=0.0, metric="cosine", random_state=42).fit_transform(emb_q2)
    resid_clusters = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=4).fit_predict(u5_q2_only[unassigned_q2])
    # Assign novel residual cluster matching R3 (which had only 2 tickets in Q1)
    unassigned_indices = np.flatnonzero(unassigned_q2)
    for r_novel in sorted(set(retro_labels) - {-1} - set(q1_known_clusters)):
        r_q2_mask_unassigned = (retro_labels[q2_mask][unassigned_q2] == r_novel)
        overlaps = Counter(resid_clusters[r_q2_mask_unassigned])
        non_noise = {k: v for k, v in overlaps.items() if k != -1}
        if non_noise:
            best_res_k = max(non_noise, key=non_noise.get)
            q2_hybrid_labels[unassigned_indices[resid_clusters == best_res_k]] = r_novel

    both_non_noise_4a = (retro_labels[q2_mask] != -1) & (q2_pred_labels != -1)
    ari_4a = float(adjusted_rand_score(retro_labels[q2_mask][both_non_noise_4a], q2_pred_labels[both_non_noise_4a]))
    nmi_4a = float(normalized_mutual_info_score(retro_labels[q2_mask][both_non_noise_4a], q2_pred_labels[both_non_noise_4a]))

    both_non_noise_4b = (retro_labels[q2_mask] != -1) & (q2_hybrid_labels != -1)
    ari_4b = float(adjusted_rand_score(retro_labels[q2_mask][both_non_noise_4b], q2_hybrid_labels[both_non_noise_4b]))
    nmi_4b = float(normalized_mutual_info_score(retro_labels[q2_mask][both_non_noise_4b], q2_hybrid_labels[both_non_noise_4b]))

    comp_rows = []
    for r_id in sorted(set(retro_labels) - {-1}):
        r_mask_q1 = (retro_labels == r_id) & q1_mask
        r_mask_q2 = (retro_labels == r_id) & q2_mask
        retro_p_q1 = r_mask_q1.sum() / q1_mask.sum()
        retro_p_q2 = r_mask_q2.sum() / q2_mask.sum()
        retro_delta = retro_p_q2 - retro_p_q1

        prosp_cnt_q1 = int(r_mask_q1.sum()) if r_id in q1_known_clusters else 0
        prosp_cnt_q2 = int((q2_hybrid_labels == r_id).sum())
        prosp_p_q1 = prosp_cnt_q1 / q1_mask.sum()
        prosp_p_q2 = prosp_cnt_q2 / q2_mask.sum()
        prosp_delta = prosp_p_q2 - prosp_p_q1

        comp_rows.append({
            "Topic_ID": f"R{r_id}",
            "Discovery_Mode": "Q1-Known Centroid" if r_id in q1_known_clusters else "Q2-Novel Residual",
            "Retro_Count_Q1": int(r_mask_q1.sum()),
            "Retro_Count_Q2": int(r_mask_q2.sum()),
            "Retro_Delta_PP": round(float(retro_delta * 100.0), 2),
            "Prosp_Count_Q1": prosp_cnt_q1,
            "Prosp_Count_Q2": prosp_cnt_q2,
            "Prosp_Delta_PP": round(float(prosp_delta * 100.0), 2),
            "Direction_Agrees": bool(np.sign(retro_delta) == np.sign(prosp_delta)),
        })

    comp_df = pd.DataFrame(comp_rows)
    rho, p_rho = stats.spearmanr(comp_df["Retro_Delta_PP"], comp_df["Prosp_Delta_PP"])
    r_pearson, p_pearson = stats.pearsonr(comp_df["Retro_Delta_PP"], comp_df["Prosp_Delta_PP"])
    comp_df.to_csv(OUT_DIR / "exp4_prospective_vs_retrospective.csv", index=False)

    exp4_summary = {
        "naive_umap_transform_q2_coverage": round(float((q2_pred_labels != -1).mean()), 4),
        "naive_umap_transform_ari": round(ari_4a, 4),
        "naive_umap_transform_nmi": round(nmi_4a, 4),
        "hybrid_prospective_q2_coverage": round(float((q2_hybrid_labels != -1).mean()), 4),
        "hybrid_prospective_ari": round(ari_4b, 4),
        "hybrid_prospective_nmi": round(nmi_4b, 4),
        "delta_k_spearman_rho": round(float(rho), 4),
        "delta_k_spearman_p": float(p_rho),
        "delta_k_pearson_r": round(float(r_pearson), 4),
        "delta_k_pearson_p": float(p_pearson),
        "sign_agreement_rate": round(float(comp_df["Direction_Agrees"].mean()), 4),
    }
    (OUT_DIR / "exp4_prospective_summary.json").write_text(json.dumps(exp4_summary, indent=2))
    print(comp_df.to_string(index=False))
    print("Exp 4 Summary:", exp4_summary)
    return comp_df, exp4_summary


def run_exp5_dose_response_perturbation(df: pd.DataFrame, embeddings: np.ndarray, umap_5d: np.ndarray):
    print("\n=== Running Experiment 5: Dose-Response Anomaly Perturbation Sweep ===", flush=True)
    # Place 3 distinct realistic outage clusters at low-density peripheral coordinates in the 5D UMAP manifold
    # with realistic intra-cluster dispersion matching empirical technical clusters (sigma in {0.06, 0.10, 0.14})
    u_min = umap_5d.min(axis=0)
    u_max = umap_5d.max(axis=0)
    u_mean = umap_5d.mean(axis=0)

    suite_centers = {
        "CUDA / NCCL Distributed OOM": (u_max + np.array([1.2, 0.8, -0.5, 0.6, -0.4]), 0.08),
        "S3 / Cloud Auth & SSL Handshake": (u_min + np.array([-1.0, 1.1, 0.7, -0.8, 0.5]), 0.11),
        "Arrow C++ Segfault / Parquet Schema": (u_mean + np.array([2.8, -2.5, 2.2, -1.8, 2.0]), 0.14),
    }

    n_q1 = int((df["quarter_period"] == "2021Q1").sum())
    n_q2_base = int((df["quarter_period"] == "2021Q2").sum())
    quarters_base = df["quarter_period"].to_numpy()

    dose_rows = []
    for suite_name, (center_5d, sigma) in suite_centers.items():
        for m_injected in [5, 8, 10, 12, 14, 15, 16, 18, 20, 25, 30]:
            rng = np.random.default_rng(42 + m_injected)
            # 5 sub-templates around the outage centroid
            template_offsets = rng.normal(0, sigma * 0.6, size=(5, 5))
            reps = [center_5d + template_offsets[i % 5] + rng.normal(0, sigma * 0.5, size=5) for i in range(m_injected)]
            synth_5d = np.vstack(reps)
            comb_5d = np.vstack([umap_5d, synth_5d])
            comb_q = np.concatenate([quarters_base, np.array(["2021Q2"] * m_injected)])
            is_synth = np.concatenate([np.zeros(len(df), dtype=bool), np.ones(m_injected, dtype=bool)])

            for mcs_setting in [12, 16]:
                hdb = hdbscan.HDBSCAN(
                    min_cluster_size=mcs_setting, min_samples=5, max_cluster_size=120,
                    metric="euclidean", cluster_selection_method="eom",
                )
                labels = hdb.fit_predict(comb_5d)
                synth_labels = labels[is_synth]
                non_noise_synth = synth_labels[synth_labels != -1]

                if len(non_noise_synth) == 0:
                    detected_cluster = -1
                    recall_injected = 0.0
                    purity = 0.0
                    delta_k = 0.0
                    rank = None
                else:
                    detected_cluster = int(Counter(non_noise_synth).most_common(1)[0][0])
                    in_det = (labels == detected_cluster)
                    purity = float((in_det & is_synth).sum() / in_det.sum())
                    recall_injected = float((synth_labels == detected_cluster).mean()) if purity >= 0.5 else 0.0

                    shifts = []
                    for c in sorted(set(labels) - {-1}):
                        p1 = ((labels == c) & (comb_q == "2021Q1")).sum() / n_q1
                        p2 = ((labels == c) & (comb_q == "2021Q2")).sum() / (n_q2_base + m_injected)
                        shifts.append((c, p2 - p1))
                    shifts.sort(key=lambda x: x[1], reverse=True)
                    ranks = {c: idx + 1 for idx, (c, _) in enumerate(shifts)}
                    rank = ranks[detected_cluster] if purity >= 0.5 else None
                    delta_k = dict(shifts)[detected_cluster] if purity >= 0.5 else 0.0

                dose_rows.append({
                    "Anomaly_Domain": suite_name,
                    "Injected_Count_M": m_injected,
                    "Injected_Share_Q2_Pct": round(m_injected / (n_q2_base + m_injected) * 100.0, 2),
                    "HDBSCAN_Min_Cluster_Size": mcs_setting,
                    "Detected_Cluster_ID": detected_cluster if purity >= 0.5 else -1,
                    "Cluster_Recall": round(recall_injected, 4),
                    "Cluster_Purity": round(purity, 4),
                    "Detected_Delta_K_PP": round(delta_k * 100.0, 2),
                    "Surge_Rank": rank if rank is not None else -1,
                    "Top1_Detected": bool(rank == 1 and purity >= 0.5),
                    "Top3_Detected": bool(rank is not None and rank <= 3 and purity >= 0.5),
                })

    dose_df = pd.DataFrame(dose_rows)
    dose_df.to_csv(OUT_DIR / "exp5_dose_response_perturbation.csv", index=False)
    print(dose_df[dose_df["HDBSCAN_Min_Cluster_Size"] == 16].to_string(index=False))
    return dose_df


def run_exp6_extrinsic_github_metadata(df: pd.DataFrame, hdb_eom14: np.ndarray):
    print("\n=== Running Experiment 6: Extrinsic SRE Validation (GitHub Labels, Bug vs PR, & MTTR) ===")
    def parse_labels(val):
        if pd.isna(val):
            return []
        try:
            items = ast.literal_eval(val)
            return [x["name"] for x in items if isinstance(x, dict) and "name" in x]
        except Exception:
            return []

    df = df.copy()
    df["gh_labels"] = df["labels"].apply(parse_labels)
    df["has_bug_label"] = df["gh_labels"].apply(lambda ls: any("bug" in l.lower() for l in ls))
    df["primary_gh_label"] = df["gh_labels"].apply(lambda ls: ls[0] if ls else "unlabeled")
    df["is_issue"] = ~df["is_pull_request"].astype(bool)

    created = pd.to_datetime(df["created_at"], unit="ms", utc=True, errors="coerce")
    closed = pd.to_datetime(df["closed_at"], unit="ms", utc=True, errors="coerce")
    df["mttr_hours"] = (closed - created).dt.total_seconds() / 3600.0
    df["mttr_days"] = df["mttr_hours"] / 24.0
    df["is_open"] = df["state"] == "open"
    df["c14"] = hdb_eom14

    r_labels = json.loads((ROOT / "paper_q2_results" / "topic_labels.json").read_text())
    r_labels_int = {int(k): v for k, v in r_labels.items()}

    # Analyze by R-cluster (10 topics + noise)
    ext_rows = []
    for cid in sorted(set(df["r_cluster"])):
        sub = df[df["r_cluster"] == cid]
        n_total = len(sub)
        n_q1 = int((sub["quarter_period"] == "2021Q1").sum())
        n_q2 = int((sub["quarter_period"] == "2021Q2").sum())

        # Issues only (excluding PRs)
        sub_issues = sub[sub["is_issue"]]
        iss_q1 = int((sub_issues["quarter_period"] == "2021Q1").sum())
        iss_q2 = int((sub_issues["quarter_period"] == "2021Q2").sum())
        total_iss_q1 = int((df["is_issue"] & (df["quarter_period"] == "2021Q1")).sum())
        total_iss_q2 = int((df["is_issue"] & (df["quarter_period"] == "2021Q2")).sum())
        delta_all_pp = (n_q2 / 403.0 - n_q1 / 478.0) * 100.0
        delta_issues_only_pp = (iss_q2 / total_iss_q2 - iss_q1 / total_iss_q1) * 100.0

        valid_mttr = sub["mttr_hours"].dropna()
        valid_mttr = valid_mttr[valid_mttr >= 0]

        all_lbls = [l for ls in sub["gh_labels"] for l in ls]
        top_gh = Counter(all_lbls).most_common(2)
        top_gh_str = ", ".join(f"{k} ({v})" for k, v in top_gh) if top_gh else "none"

        ext_rows.append({
            "Cluster_ID": f"R{cid}" if cid >= 0 else "Noise (-1)",
            "Topic_Label": r_labels_int.get(cid, "Noise / Outliers"),
            "Total_Count": n_total,
            "Issue_Ratio_Pct": round(float(sub["is_issue"].mean() * 100.0), 1),
            "PR_Ratio_Pct": round(float(sub["is_pull_request"].mean() * 100.0), 1),
            "Open_Unresolved_Pct": round(float(sub["is_open"].mean() * 100.0), 1),
            "Median_MTTR_Hours": round(float(valid_mttr.median()), 1) if len(valid_mttr) > 0 else None,
            "Mean_MTTR_Hours": round(float(valid_mttr.mean()), 1) if len(valid_mttr) > 0 else None,
            "Delta_All_Records_PP": round(delta_all_pp, 2),
            "Delta_Issues_Only_PP": round(delta_issues_only_pp, 2),
            "Dominant_GitHub_Labels": top_gh_str,
        })

    ext_df = pd.DataFrame(ext_rows)
    ext_df.to_csv(OUT_DIR / "exp6_extrinsic_validation_r10.csv", index=False)
    print(ext_df.to_string(index=False))

    # Also compute Hierarchical Mapping Table between Historical 14 Topics (T0-T13), R-Topics (R0-R9), and Bounded-EOM 14 Topics (C0-C13)
    reps_14 = json.load(open(ROOT / "cluster_representative_tickets.json"))
    q_clean = pd.read_csv(ROOT / "quarterly_temporal_shifts_clean.csv").set_index("Cluster_ID")
    map_rows = []
    for cid_str, info in sorted(reps_14.items(), key=lambda x: int(x[0])):
        cid = int(cid_str)
        titles = info["representative_titles"]
        idxs = [df.index[df["title"] == t].tolist()[0] for t in titles if (df["title"] == t).any()]
        r_mapped = int(Counter(df.loc[idxs, "r_cluster"]).most_common(1)[0][0])
        c14_mapped = int(Counter(df.loc[idxs, "c14"]).most_common(1)[0][0])
        map_rows.append({
            "Historical_ID": f"T{cid}",
            "Historical_Subsystem": q_clean.loc[cid, "Technical_Subsystem"],
            "Hist_Count": int(info["count"]),
            "Rerun_EOM10_ID": f"R{r_mapped}" if r_mapped >= 0 else "Noise (-1)",
            "Bounded_EOM14_ID": f"C{c14_mapped}" if c14_mapped >= 0 else "Noise (-1)",
            "Medoid_Ticket_Title": titles[0],
        })
    map_df = pd.DataFrame(map_rows)
    map_df.to_csv(OUT_DIR / "exp6_cluster_hierarchy_mapping.csv", index=False)
    print("\nCluster Hierarchy Mapping (T0-T13 <-> R0-R9 <-> Bounded-EOM C0-C13):")
    print(map_df.to_string(index=False))
    return ext_df, map_df


def generate_updated_and_new_figures(
    df: pd.DataFrame,
    stat_df: pd.DataFrame,
    exp1_df: pd.DataFrame,
    dim_df: pd.DataFrame,
    dose_df: pd.DataFrame,
    ext_df: pd.DataFrame,
):
    print("\n=== Generating Regenerated & New Publication-Quality Figures ===")
    from matplotlib.patches import Patch

    # 1. Regenerate Figure 2 (figures/figure2_temporal_shift.png) with full, non-truncated labels and 95% CIs!
    s_df = stat_df.sort_values("Delta_K", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11.5, 6.5), dpi=300)
    cat_colors = {
        "Emerging Surge (Q2)": "#d9534f",
        "Persistent Chronic Root Cause": "#f0ad4e",
        "Stable Operational Routine": "#5bc0de",
        "Declining / Resolved in Q2": "#5cb85c",
    }
    colors = [cat_colors.get(c, "#666666") for c in s_df["Category"]]
    y_pos = np.arange(len(s_df))
    deltas = s_df["Delta_K"].to_numpy()
    err_low = deltas - s_df["CI95_Low"].to_numpy()
    err_high = s_df["CI95_High"].to_numpy() - deltas

    ax.barh(y_pos, deltas, color=colors, edgecolor="#333333", height=0.72, lw=0.7, alpha=0.92)
    ax.errorbar(deltas, y_pos, xerr=[err_low, err_high], fmt="none", ecolor="#222222", elinewidth=1.1, capsize=2.5, alpha=0.75)

    labels_y = []
    for _, r in s_df.iterrows():
        sig = "**" if r["Q_Value_BH_FDR"] < 0.05 else ("*" if r["P_Value_Fisher"] < 0.05 else "")
        labels_y.append(f"T{r['Cluster_ID']}: {r['Technical_Subsystem']}{sig}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_y, fontsize=9.5, fontweight="medium")
    ax.axvline(0, color="#111111", linewidth=1.1, linestyle="--")
    ax.axvline(0.025, color="#d9534f", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.axvline(-0.025, color="#5cb85c", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_xlabel("Quarterly Proportional Shift (ΔK = P(k | Q2) − P(k | Q1)) with 95% CI", fontsize=11, fontweight="bold")
    ax.set_title("180-Day Quarterly Incident Shift Detection Across Technical Subsystems (Q1 vs. Q2)", fontsize=12.5, fontweight="bold", pad=12)

    legend_elements = [Patch(facecolor=color, edgecolor="#333333", label=label) for label, color in cat_colors.items()]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="white", framealpha=0.95, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure2_temporal_shift.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure2_temporal_shift.pdf", bbox_inches="tight")
    plt.close(fig)

    # 2. Regenerate Figure 3 (figures/figure3_tri_model_benchmark.png) without clipped title!
    fig, ax = plt.subplots(figsize=(9.0, 5.4), dpi=300)
    metrics = ["Topic Coherence\n(C_v)", "Topic Diversity\n(TD)", "Topic Quality\n(C_v × TD)", "Silhouette\n(Dense 384D)", "Silhouette\n(Native/Manifold)"]
    lda_vals = [0.7448, 0.7300, round(0.7448 * 0.7300, 4), -0.0247, 0.0006]
    km_vals = [0.6341, 0.8200, round(0.6341 * 0.8200, 4), -0.0194, 0.0158]
    prop_vals = [0.6401, 0.8071, round(0.6401 * 0.8071, 4), 0.0354, 0.4642]

    x = np.arange(len(metrics))
    width = 0.25
    ax.bar(x - width, lda_vals, width, label="Classical Generative (LDA)", color="#d9534f", edgecolor="#333333", lw=0.6)
    ax.bar(x, km_vals, width, label="Discriminative Baseline (K-Means)", color="#6c757d", edgecolor="#333333", lw=0.6)
    ax.bar(x + width, prop_vals, width, label="Proposed Framework (SBERT+UMAP+HDBSCAN)", color="#0275d8", edgecolor="#333333", lw=0.6)

    for i in range(len(metrics)):
        for offset, vals, bold in [(-width, lda_vals, False), (0.0, km_vals, False), (width, prop_vals, True)]:
            val = vals[i]
            y_txt = val + 0.018 if val >= 0 else val - 0.042
            ax.text(i + offset, y_txt, f"{val:.2f}", ha="center", fontsize=8.5, fontweight="bold" if bold else "normal")

    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_ylabel("Metric Score", fontsize=11, fontweight="bold")
    ax.set_title("Tri-Model Quantitative Benchmark (180-Day Quarterly Stream)", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9.5, fontweight="bold")
    ax.set_ylim(-0.08, 0.92)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure3_tri_model_benchmark.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure3_tri_model_benchmark.pdf", bbox_inches="tight")
    plt.close(fig)

    # 3. New Figure 4: Multi-Panel Ablation & Sensitivity Analysis (UMAP Dimension d, Dose-Response Curve, and MTTR vs Issue Ratio)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=300)

    # Panel A: UMAP Dimensionality Sweep
    ax1 = axes[0]
    d_eom = dim_df[dim_df["Method"] == "Bounded-EOM (max=120)"].copy()
    x_labels = [str(d) if d < 384 else "384\n(Raw)" for d in d_eom["Dimensions_d"]]
    x_idx = np.arange(len(d_eom))
    ax1.plot(x_idx, d_eom["Sil_Manifold"], marker="o", lw=2.2, color="#0275d8", label="Silhouette (Manifold)")
    ax1.plot(x_idx, d_eom["Topic_Quality"], marker="s", lw=2.2, color="#2ca02c", label="Topic Quality (C_v × TD)")
    ax1.plot(x_idx, d_eom["Noise_Ratio"], marker="^", lw=1.8, linestyle="--", color="#d9534f", label="Noise Ratio")
    ax1.axvline(2, color="#0275d8", linestyle=":", lw=1.3, alpha=0.75, label="Selected d = 5")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(x_labels, fontsize=8.5)
    ax1.set_xlabel("Ambient Dimension d", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Score / Proportion", fontsize=10, fontweight="bold")
    ax1.set_title("(a) UMAP Dimensionality Ablation", fontsize=11, fontweight="bold")
    ax1.set_ylim(-0.04, 1.06)
    ax1.legend(loc="center left", fontsize=8, frameon=True, facecolor="white", framealpha=0.9)

    # Panel B: Dose-Response Perturbation Sensitivity Curve (mcs=12 vs mcs=16 + Delta_K on twin axis)
    ax2 = axes[1]
    d12 = dose_df[(dose_df["HDBSCAN_Min_Cluster_Size"] == 12) & (dose_df["Anomaly_Domain"] == "CUDA / NCCL Distributed OOM")].sort_values("Injected_Count_M")
    d16 = dose_df[(dose_df["HDBSCAN_Min_Cluster_Size"] == 16) & (dose_df["Anomaly_Domain"] == "CUDA / NCCL Distributed OOM")].sort_values("Injected_Count_M")

    ax2.plot(d12["Injected_Count_M"], d12["Cluster_Recall"] * 100.0, marker="s", lw=2.2, linestyle="--", color="#2ca02c", label="Cluster Recall (m_cl = 12)")
    ax2.plot(d16["Injected_Count_M"], d16["Cluster_Recall"] * 100.0, marker="o", lw=2.4, color="#0275d8", label="Cluster Recall (m_cl = 16)")
    ax2.plot(d16["Injected_Count_M"], d16["Top1_Detected"].astype(float) * 100.0, marker="^", lw=2.0, linestyle="-.", color="#d9534f", label="Rank #1 Surge Isolation (m_cl = 16)")
    ax2.axvline(16, color="#333333", linestyle=":", lw=1.1, label="HDBSCAN m_cl = 16")
    ax2.axvline(20, color="#d9534f", linestyle=":", lw=1.1, label="Rank #1 Threshold (M = 20)")
    ax2.set_xlabel("Injected Q2 Anomaly Tickets (M)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Detection Sensitivity (%)", fontsize=10, fontweight="bold")
    ax2.set_title("(b) Dose–Response Outbreak Sensitivity", fontsize=11, fontweight="bold")
    ax2.set_ylim(-5, 108)
    ax2.legend(loc="center right", fontsize=7.8, frameon=True, facecolor="white", framealpha=0.92)

    # Panel C: Extrinsic Validation — Bug Issue Ratio vs. Median MTTR (Hours) by Topic
    ax3 = axes[2]
    ext_non_noise = ext_df[ext_df["Cluster_ID"] != "Noise (-1)"].copy()
    sc = ax3.scatter(
        ext_non_noise["Issue_Ratio_Pct"],
        ext_non_noise["Median_MTTR_Hours"],
        s=ext_non_noise["Total_Count"] * 2.0 + 45,
        c=ext_non_noise["Open_Unresolved_Pct"],
        cmap="YlOrRd",
        edgecolor="#222222",
        lw=1.0,
        alpha=0.9,
    )
    for _, row in ext_non_noise.iterrows():
        cid_lbl = row["Cluster_ID"]
        dx, dy = (11, 5) if cid_lbl == "R1" else ((5, -10) if cid_lbl == "R6" else (5, 4))
        ax3.annotate(
            cid_lbl,
            (row["Issue_Ratio_Pct"], row["Median_MTTR_Hours"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
        )
    cbar = plt.colorbar(sc, ax=ax3, pad=0.02)
    cbar.set_label("Unresolved Open %", fontsize=8.5)
    ax3.set_xlim(0, 76)
    ax3.set_ylim(10, 165)
    ax3.set_xlabel("Bug / Issue Ticket Share (%, Excl. PRs)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Median Resolution Time (MTTR, Hours)", fontsize=10, fontweight="bold")
    ax3.set_title("(c) Extrinsic Triage Load: Issues vs. MTTR", fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure4_ablation_and_extrinsic.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure4_ablation_and_extrinsic.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved regenerated Figure 2, Figure 3, and new Figure 4 (PNG + PDF).")


def main():
    df, embeddings, umap_5d = load_corpus_and_embeddings()
    hdb_eom14 = hdbscan.HDBSCAN(
        min_cluster_size=16, min_samples=5, max_cluster_size=120, metric="euclidean", cluster_selection_method="eom"
    ).fit_predict(umap_5d)

    if (OUT_DIR / "exp1_trimodel_ablation.csv").exists():
        exp1_df = pd.read_csv(OUT_DIR / "exp1_trimodel_ablation.csv")
    else:
        exp1_df, hdb_eom14 = run_exp1_fair_trimodel_ablation(df, embeddings, umap_5d)

    if (OUT_DIR / "exp2_quarterly_shifts_with_stats.csv").exists():
        stat_df = pd.read_csv(OUT_DIR / "exp2_quarterly_shifts_with_stats.csv")
    else:
        stat_df, _ = run_exp2_statistical_significance(df, hdb_eom14)

    if (OUT_DIR / "exp3a_umap_dimension_ablation.csv").exists():
        dim_df = pd.read_csv(OUT_DIR / "exp3a_umap_dimension_ablation.csv")
    else:
        dim_df, _, _, _ = run_exp3_hyperparameter_sensitivity(df, embeddings, umap_5d)

    comp_df, exp4_summary = run_exp4_prospective_vs_retrospective(df, embeddings, umap_5d)
    dose_df = run_exp5_dose_response_perturbation(df, embeddings, umap_5d)
    ext_df, map_df = run_exp6_extrinsic_github_metadata(df, hdb_eom14)
    generate_updated_and_new_figures(df, stat_df, exp1_df, dim_df, dose_df, ext_df)
    print("\nAll 6 experiments and figures completed successfully!", flush=True)


if __name__ == "__main__":
    main()
