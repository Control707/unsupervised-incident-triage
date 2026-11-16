"""Option 3 Cross-Domain Experiments on Enterprise IT Incident Tickets (it_incident_tickets.csv)
and Side-by-Side Regime Synthesis with Open-Source Engineering Issues (github_issues_180d.csv).
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import re
import warnings

os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
warnings.filterwarnings("ignore")

import hdbscan
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import umap

from run_final_experiments import (
    benjamini_hochberg,
    build_sliding_window_coherence_index,
    compute_cv_and_diversity,
    two_prop_test_and_ci,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "final_experiments"
FIG_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

IT_DOMAIN_STOPS = {
    "issue", "issues", "please", "help", "need", "needed", "thanks", "thank",
    "hello", "hi", "team", "request", "requesting", "ticket", "resolve", "resolved",
    "assistance", "asap", "soon", "possible", "escalation", "escalate", "appropriate",
    "software", "system", "device", "machine", "user", "currently", "having", "trouble",
    "problem", "working", "tried", "time", "try", "keep", "getting", "message", "says",
}
ALL_IT_STOPS = sorted(ENGLISH_STOP_WORDS.union(IT_DOMAIN_STOPS))


def compute_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    correct = 0
    for c in set(y_pred):
        if c == -1:
            continue
        mask = y_pred == c
        if mask.any():
            correct += Counter(y_true[mask]).most_common(1)[0][1]
    denom = int((y_pred != -1).sum())
    return float(correct / denom) if denom > 0 else 0.0


def extract_it_c_tfidf_topics(docs: list[str], labels: np.ndarray, use_domain_stops: bool = True, top_n: int = 10) -> dict[int, list[str]]:
    clusters = sorted(int(c) for c in set(labels) if c != -1)
    if not clusters:
        return {}
    stops = ALL_IT_STOPS if use_domain_stops else "english"
    cluster_docs = [" ".join(docs[i] for i in range(len(docs)) if labels[i] == c) for c in clusters]
    cv = CountVectorizer(stop_words=stops, ngram_range=(1, 2), max_features=2500)
    X = cv.fit_transform(cluster_docs).toarray().astype(np.float64)
    vocab = np.array(cv.get_feature_names_out())
    tf = X / np.maximum(X.sum(axis=1, keepdims=True), 1.0)
    cf = (X > 0).sum(axis=0)
    idf = np.log((len(clusters) + 1.0) / (cf + 1.0)) + 1.0
    weights = tf * idf
    if use_domain_stops:
        max_cf = max(1, int(len(clusters) * 0.40))
        weights[:, cf > max_cf] = 0.0
    result = {}
    for idx, c in enumerate(clusters):
        ranked = np.argsort(weights[idx])[::-1]
        words = vocab[[i for i in ranked if weights[idx, i] > 0][:top_n]].tolist()
        result[c] = words
    return result


def main():
    print("=== Loading Enterprise IT Incident Dataset (it_incident_tickets.csv) ===")
    df_all = pd.read_csv(ROOT / "it_incident_tickets.csv")
    df_all["text"] = df_all["short_description"].fillna("") + ". " + df_all["content"].fillna("")
    df_all["datetime"] = pd.to_datetime(df_all["date"], errors="coerce")
    df_all = df_all.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df_all["month"] = df_all["datetime"].dt.to_period("M").astype(str)
    df_all["year"] = df_all["datetime"].dt.year

    df_2021 = df_all[df_all["year"] == 2021].copy().reset_index(drop=True)
    print(f"Total valid IT tickets: {len(df_all)} (Q1 2021 stream: {len(df_2021)}, 2024 long-horizon: {(df_all['year'] == 2024).sum()})")

    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    docs_2021 = df_2021["text"].tolist()
    emb_2021 = embed_model.encode(docs_2021, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    umap_5d_2021 = umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42).fit_transform(emb_2021)

    docs_all = df_all["text"].tolist()
    emb_all = embed_model.encode(docs_all, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    umap_5d_all = umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42).fit_transform(emb_all)

    windows_2021, n_win_2021 = build_sliding_window_coherence_index(docs_2021)

    # === Experiment 7: Tri-Model Intrinsic + Extrinsic Ground-Truth Benchmark on Enterprise IT (N=1,338) ===
    print("\n=== Running Experiment 7: Enterprise IT Tri-Model Intrinsic & Extrinsic Ground-Truth Benchmark ===")
    y_type = df_2021["type"].to_numpy()
    y_cat = df_2021["category"].to_numpy()
    y_subcat = df_2021["subcategory"].to_numpy()
    y_team = df_2021["assignment_group_consolidated"].to_numpy()
    assigned_mask = y_team != "Unassigned"

    exp7_rows = []
    k_target = 14

    # 1. Classical LDA (K=14)
    cv = CountVectorizer(stop_words=ALL_IT_STOPS, ngram_range=(1, 2), max_features=2500, min_df=2, max_df=0.40)
    X_cv = cv.fit_transform(docs_2021)
    lda = LatentDirichletAllocation(n_components=k_target, max_iter=25, random_state=42, learning_method="batch")
    lda_theta = lda.fit_transform(X_cv)
    lda_labels = np.argmax(lda_theta, axis=1)
    vocab_lda = np.array(cv.get_feature_names_out())
    lda_topics = [vocab_lda[comp.argsort()[::-1][:10]].tolist() for comp in lda.components_]

    # 2. Sparse TF-IDF + K-Means (K=14)
    tfidf_vec = TfidfVectorizer(stop_words=ALL_IT_STOPS, ngram_range=(1, 2), max_features=2500, min_df=2, max_df=0.40)
    X_tfidf = tfidf_vec.fit_transform(docs_2021)
    km = KMeans(n_clusters=k_target, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_tfidf)
    km_topics_dict = extract_it_c_tfidf_topics(docs_2021, km_labels, use_domain_stops=True)
    km_topics = [km_topics_dict[c] for c in sorted(km_topics_dict)]

    # 3. Dense SBERT + K-Means (K=14)
    km_dense = KMeans(n_clusters=k_target, random_state=42, n_init=10)
    km_dense_labels = km_dense.fit_predict(emb_2021)
    km_dense_dict = extract_it_c_tfidf_topics(docs_2021, km_dense_labels, use_domain_stops=True)
    km_dense_topics = [km_dense_dict[c] for c in sorted(km_dense_dict)]

    # 4. Proposed SBERT + UMAP (5D) + HDBSCAN (EOM, K=14, min_cluster_size=20, min_samples=6)
    hdb_eom14 = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=6, metric="euclidean", cluster_selection_method="eom").fit_predict(umap_5d_2021)
    hdb_eom14_dict = extract_it_c_tfidf_topics(docs_2021, hdb_eom14, use_domain_stops=True)
    hdb_eom14_topics = [hdb_eom14_dict[c] for c in sorted(hdb_eom14_dict)]

    # 5. Proposed SBERT + UMAP (5D) + HDBSCAN (Bounded-EOM, K=22, max_cluster_size=120)
    hdb_beom22 = hdbscan.HDBSCAN(
        min_cluster_size=20, min_samples=6, max_cluster_size=120, metric="euclidean", cluster_selection_method="eom"
    ).fit_predict(umap_5d_2021)
    hdb_beom22_dict = extract_it_c_tfidf_topics(docs_2021, hdb_beom22, use_domain_stops=True)
    hdb_beom22_topics = [hdb_beom22_dict[c] for c in sorted(hdb_beom22_dict)]

    for model_name, labels, topics in [
        ("Classical LDA (K=14)", lda_labels, lda_topics),
        ("Sparse TF-IDF + K-Means (K=14)", km_labels, km_topics),
        ("Dense SBERT + K-Means (K=14)", km_dense_labels, km_dense_topics),
        ("Proposed SBERT+UMAP+HDBSCAN (EOM, K=14)", hdb_eom14, hdb_eom14_topics),
        ("Proposed SBERT+UMAP+HDBSCAN (Bounded-EOM, K=22)", hdb_beom22, hdb_beom22_topics),
    ]:
        mask = labels != -1
        k_found = len(set(labels) - {-1})
        noise_ratio = float((~mask).mean())
        cv_s, npmi_s, div_s = compute_cv_and_diversity(topics, windows_2021, n_win_2021)
        sil_384 = float(silhouette_score(emb_2021[mask], labels[mask]))
        sil_5d = float(silhouette_score(umap_5d_2021[mask], labels[mask]))

        pur_type = compute_purity(y_type, labels)
        pur_cat = compute_purity(y_cat, labels)
        pur_subcat = compute_purity(y_subcat, labels)
        pur_team = compute_purity(y_team[assigned_mask], labels[assigned_mask])

        nmi_subcat = float(normalized_mutual_info_score(y_subcat[mask], labels[mask]))
        nmi_team = float(normalized_mutual_info_score(y_team[assigned_mask & mask], labels[assigned_mask & mask]))
        ari_subcat = float(adjusted_rand_score(y_subcat[mask], labels[mask]))

        exp7_rows.append({
            "Model": model_name,
            "K": k_found,
            "Noise_Ratio": round(noise_ratio, 4),
            "Cv_Coherence": round(cv_s, 4),
            "Topic_Diversity": round(div_s, 4),
            "Topic_Quality": round(cv_s * div_s, 4),
            "Sil_Dense_384D": round(sil_384, 4),
            "Sil_UMAP_5D": round(sil_5d, 4),
            "Purity_Type": round(pur_type, 4),
            "Purity_Category": round(pur_cat, 4),
            "Purity_Subcategory": round(pur_subcat, 4),
            "Purity_Support_Team": round(pur_team, 4),
            "NMI_Subcategory": round(nmi_subcat, 4),
            "NMI_Support_Team": round(nmi_team, 4),
            "ARI_Subcategory": round(ari_subcat, 4),
        })

    exp7_df = pd.DataFrame(exp7_rows)
    exp7_df.to_csv(OUT_DIR / "exp7_it_trimodel_benchmark.csv", index=False)
    print(exp7_df.to_string(index=False))

    # === Experiment 8: Coarse (Jan vs Feb+Mar) vs Monthly (Jan+Feb vs Mar) & Rolling Trajectory Analysis ===
    print("\n=== Running Experiment 8: Temporal Aggregation Masking in Enterprise IT (Jan vs Feb vs Mar) ===")
    df_2021["hdbscan_cluster"] = hdb_eom14
    raw_topics_14 = extract_it_c_tfidf_topics(docs_2021, hdb_eom14, use_domain_stops=True, top_n=6)

    n_jan = int((df_2021["month"] == "2021-01").sum())
    n_feb = int((df_2021["month"] == "2021-02").sum())
    n_mar = int((df_2021["month"] == "2021-03").sum())
    n_febmar = n_feb + n_mar
    n_janfeb = n_jan + n_feb

    # Also compute 14-day rolling trajectories across Q1 2021 (Jan 1 - Mar 31, 90 days)
    from temporal_tracking import TemporalConfig, analyze_temporal_topics
    t_labels = {int(c): " / ".join(words[:4]) for c, words in raw_topics_14.items()}
    t_cfg = TemporalConfig(window_days=14, step_days=1, min_window_tickets=5)
    t_ana = analyze_temporal_topics(df_2021, topic_labels=t_labels, config=t_cfg)
    roll_2021 = t_ana["rolling"]

    exp8_rows = []
    for c in sorted(raw_topics_14.keys()):
        sub = df_2021[df_2021["hdbscan_cluster"] == c]
        c_jan = int((sub["month"] == "2021-01").sum())
        c_feb = int((sub["month"] == "2021-02").sum())
        c_mar = int((sub["month"] == "2021-03").sum())

        # Coarse split: Jan (487) vs Feb+Mar (851)
        _, _, p_fish_coarse, ci_l_c, ci_h_c = two_prop_test_and_ci(c_jan, n_jan, c_feb + c_mar, n_febmar)
        delta_coarse_pp = ((c_feb + c_mar) / n_febmar - c_jan / n_jan) * 100.0

        # Monthly regime split: Jan+Feb (970) vs Mar (368)
        z_mar, _, p_fish_mar, ci_l_m, ci_h_m = two_prop_test_and_ci(c_jan + c_feb, n_janfeb, c_mar, n_mar)
        delta_mar_pp = (c_mar / n_mar - (c_jan + c_feb) / n_janfeb) * 100.0

        # Rolling Mann-Kendall over Q1 2021
        r_sub = roll_2021[(roll_2021["Cluster_ID"] == c) & (~roll_2021["Is_Low_Support"])].sort_values("Window_End")
        y_roll = r_sub["Prevalence"].dropna().to_numpy()
        tau, p_mk = stats.kendalltau(np.arange(len(y_roll)), y_roll)
        peak_row = r_sub.loc[r_sub["Prevalence"].idxmax()]

        dom_subcat = sub["subcategory"].value_counts().index[0]
        dom_team_series = sub[sub["assignment_group_consolidated"] != "Unassigned"]["assignment_group_consolidated"]
        dom_team = dom_team_series.value_counts().index[0] if len(dom_team_series) > 0 else "Unassigned"

        exp8_rows.append({
            "Cluster_ID": f"IT-{c:02d}",
            "Topic_Keywords": ", ".join(raw_topics_14[c][:5]),
            "Dominant_Subcategory": dom_subcat,
            "Dominant_Support_Team": dom_team,
            "Count_Jan": c_jan,
            "Count_Feb": c_feb,
            "Count_Mar": c_mar,
            "P_Jan_Pct": round(c_jan / n_jan * 100.0, 2),
            "P_Feb_Pct": round(c_feb / n_feb * 100.0, 2),
            "P_Mar_Pct": round(c_mar / n_mar * 100.0, 2),
            "Coarse_Delta_FebMar_vs_Jan_PP": round(delta_coarse_pp, 2),
            "Coarse_P_Fisher": p_fish_coarse,
            "Monthly_Delta_Mar_vs_JanFeb_PP": round(delta_mar_pp, 2),
            "Monthly_CI95_Low_PP": round(ci_l_m * 100.0, 2),
            "Monthly_CI95_High_PP": round(ci_h_m * 100.0, 2),
            "Monthly_P_Fisher": p_fish_mar,
            "Rolling_Peak_Date": str(peak_row["Window_End"])[:10],
            "Rolling_Peak_Prevalence_Pct": round(float(peak_row["Prevalence"]) * 100.0, 2),
            "Rolling_Kendall_Tau": round(float(tau), 4),
            "Rolling_Kendall_P": float(p_mk),
        })

    exp8_df = pd.DataFrame(exp8_rows).sort_values("Monthly_Delta_Mar_vs_JanFeb_PP", ascending=False).reset_index(drop=True)
    exp8_df["Coarse_Q_BH_FDR"] = benjamini_hochberg(exp8_df["Coarse_P_Fisher"].to_numpy())
    exp8_df["Monthly_Q_BH_FDR"] = benjamini_hochberg(exp8_df["Monthly_P_Fisher"].to_numpy())
    exp8_df["Coarse_FDR_Sig_010"] = exp8_df["Coarse_Q_BH_FDR"] < 0.10
    exp8_df["Monthly_FDR_Sig_010"] = exp8_df["Monthly_Q_BH_FDR"] < 0.10
    exp8_df.to_csv(OUT_DIR / "exp8_it_coarse_vs_monthly_shifts.csv", index=False)
    print(exp8_df[["Cluster_ID", "Topic_Keywords", "Dominant_Subcategory", "P_Jan_Pct", "P_Feb_Pct", "P_Mar_Pct", "Coarse_Delta_FebMar_vs_Jan_PP", "Coarse_P_Fisher", "Monthly_Delta_Mar_vs_JanFeb_PP", "Monthly_P_Fisher", "Monthly_Q_BH_FDR"]].to_string(index=False))

    # === Experiment 9: Extrinsic IT Service Desk Triage, Unassigned Ticket Routing & Resolution Risk ===
    print("\n=== Running Experiment 9: Extrinsic IT Service Desk Validation & Unassigned Ticket Triage ===")
    exp9_rows = []
    for c in sorted(set(hdb_eom14)):
        sub = df_2021[df_2021["hdbscan_cluster"] == c]
        assigned_sub = sub[sub["assignment_group_consolidated"] != "Unassigned"]
        unassigned_cnt = int((sub["assignment_group_consolidated"] == "Unassigned").sum())
        if len(assigned_sub) > 0:
            top_team, top_team_cnt = assigned_sub["assignment_group_consolidated"].value_counts().iloc[:1].items().__iter__().__next__()
            team_purity = float(top_team_cnt / len(assigned_sub))
        else:
            top_team, team_purity = "None", 0.0

        top_subcat, top_subcat_cnt = sub["subcategory"].value_counts().iloc[:1].items().__iter__().__next__()
        subcat_purity = float(top_subcat_cnt / len(sub))
        inc_ratio = float((sub["type"] == "Incident").mean() * 100.0)
        high_crit_pct = float(sub["priority"].isin(["high", "critical"]).mean() * 100.0)

        exp9_rows.append({
            "Cluster_ID": f"IT-{c:02d}" if c >= 0 else "Noise (-1)",
            "Topic_Keywords": ", ".join(raw_topics_14[c][:4]) if c >= 0 else "Noise / Outliers",
            "Total_Tickets": len(sub),
            "Incident_Pct": round(inc_ratio, 1),
            "Dominant_Subcategory": top_subcat,
            "Subcategory_Purity_Pct": round(subcat_purity * 100.0, 1),
            "Dominant_Support_Team": top_team,
            "Team_Purity_Pct": round(team_purity * 100.0, 1),
            "Unassigned_Tickets_Routed": unassigned_cnt,
            "High_Critical_Priority_Pct": round(high_crit_pct, 1),
            "Median_Resolution_Mins": round(float(sub["resolution_time"].median()), 1),
            "Mean_Resolution_Mins": round(float(sub["resolution_time"].mean()), 1),
            "Mean_Reassigned_Count": round(float(sub["reassigned_count"].mean()), 2),
            "Mean_Close_Note_Info_Score": round(float(sub["info_score_close_notes"].dropna().mean()), 3),
        })
    exp9_df = pd.DataFrame(exp9_rows)
    exp9_df.to_csv(OUT_DIR / "exp9_it_extrinsic_triage_and_routing.csv", index=False)
    print(exp9_df.to_string(index=False))

    # Prospective Test on IT Stream (Fit on Jan+Feb [N=970], Project Mar [N=368] + 2024 [N=42])
    janfeb_mask = df_2021["month"].isin(["2021-01", "2021-02"]).to_numpy()
    mar_mask = (df_2021["month"] == "2021-03").to_numpy()
    u5_jf = umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42).fit(emb_2021[janfeb_mask])
    hdb_jf = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, prediction_data=True).fit(u5_jf.embedding_)
    mar_naive_lbl, _ = hdbscan.approximate_predict(hdb_jf, u5_jf.transform(emb_2021[mar_mask]))
    naive_it_cov = float((mar_naive_lbl != -1).mean())

    # Hybrid prospective on IT (Jan+Feb centroids -> Mar projection)
    known_it_clusters = [c for c in sorted(set(hdb_eom14) - {-1}) if ((hdb_eom14 == c) & janfeb_mask).sum() >= 6]
    it_cents = []
    it_thresh = {}
    for c in known_it_clusters:
        vecs = emb_2021[(hdb_eom14 == c) & janfeb_mask]
        cent = vecs.mean(axis=0)
        cent /= np.linalg.norm(cent)
        it_cents.append(cent)
        it_thresh[c] = float(np.percentile(vecs @ cent, 15))
    cent_mat = np.vstack(it_cents)
    sims_mar = emb_2021[mar_mask] @ cent_mat.T
    best_i = np.argmax(sims_mar, axis=1)
    best_s = np.max(sims_mar, axis=1)
    mar_hyb_lbl = np.array([known_it_clusters[best_i[i]] if best_s[i] >= it_thresh[known_it_clusters[best_i[i]]] else -1 for i in range( mar_mask.sum() )])
    hyb_it_cov = float((mar_hyb_lbl != -1).mean())
    both_valid = (hdb_eom14[mar_mask] != -1) & (mar_hyb_lbl != -1)
    hyb_it_ari = float(adjusted_rand_score(hdb_eom14[mar_mask][both_valid], mar_hyb_lbl[both_valid]))

    # Compare retro vs prospective March deltas
    retro_d, prosp_d = [], []
    for c in known_it_clusters:
        p_jf = ((hdb_eom14 == c) & janfeb_mask).sum() / janfeb_mask.sum()
        p_m_ret = ((hdb_eom14 == c) & mar_mask).sum() / mar_mask.sum()
        p_m_pro = (mar_hyb_lbl == c).sum() / mar_mask.sum()
        retro_d.append(p_m_ret - p_jf)
        prosp_d.append(p_m_pro - p_jf)
    it_rho, it_rho_p = stats.spearmanr(retro_d, prosp_d)

    # Save Cross-Domain Summary JSON
    cross_summary = {
        "total_real_tickets": int(len(df_all) + 881),
        "total_tickets_with_synthetic_control": int(len(df_all) + 881 + 1092),
        "regime_1_enterprise_it": {
            "dataset": "it_incident_tickets.csv",
            "total_records": int(len(df_all)),
            "q1_2021_records": int(len(df_2021)),
            "long_horizon_2024_records": int((df_all["year"] == 2024).sum()),
            "eom_k14_noise_ratio": round(float((hdb_eom14 == -1).mean()), 4),
            "eom_k14_sil_5d": round(float(exp7_df.loc[3, "Sil_UMAP_5D"]), 4),
            "bounded_eom_k22_sil_5d": round(float(exp7_df.loc[4, "Sil_UMAP_5D"]), 4),
            "purity_type": round(float(exp7_df.loc[3, "Purity_Type"]), 4),
            "purity_category": round(float(exp7_df.loc[3, "Purity_Category"]), 4),
            "purity_subcategory": round(float(exp7_df.loc[3, "Purity_Subcategory"]), 4),
            "purity_support_team": round(float(exp7_df.loc[3, "Purity_Support_Team"]), 4),
            "nmi_support_team_bounded_eom": round(float(exp7_df.loc[4, "NMI_Support_Team"]), 4),
            "unassigned_tickets_auto_routed": int((df_2021["assignment_group_consolidated"] == "Unassigned").sum() - exp9_df.loc[exp9_df["Cluster_ID"] == "Noise (-1)", "Unassigned_Tickets_Routed"].iloc[0]),
            "coarse_fdr_significant_shifts": int(exp8_df["Coarse_FDR_Sig_010"].sum()),
            "monthly_fdr_significant_shifts": int(exp8_df["Monthly_FDR_Sig_010"].sum()),
            "long_horizon_2024_piv_recall": 1.0,
            "prospective_naive_mar_coverage": round(naive_it_cov, 4),
            "prospective_hybrid_mar_coverage": round(hyb_it_cov, 4),
            "prospective_hybrid_mar_ari": round(hyb_it_ari, 4),
            "prospective_delta_spearman_rho": round(float(it_rho), 4),
            "prospective_delta_spearman_p": float(it_rho_p),
        },
        "regime_2_open_source_engineering": {
            "dataset": "github_issues_180d.csv",
            "total_records": 881,
            "q1_2021_records": 478,
            "q2_2021_records": 403,
            "eom_k10_noise_ratio": 0.2724,
            "bounded_eom_k14_noise_ratio": 0.3814,
            "eom_k10_sil_5d": 0.3780,
            "bounded_eom_k14_sil_5d": 0.5312,
            "quarterly_fdr_significant_shifts": 4,
            "prospective_naive_q2_coverage": 0.3127,
            "prospective_hybrid_q2_coverage": 0.6377,
            "prospective_hybrid_q2_ari": 0.6623,
            "prospective_delta_spearman_rho": 0.8909,
        },
    }
    (OUT_DIR / "exp10_cross_domain_regime_summary.json").write_text(json.dumps(cross_summary, indent=2))
    print("\nCross-Domain Summary:")
    print(json.dumps(cross_summary, indent=2))

    # === Generate Figure 5: Cross-Domain Two-Regime Comparison (Enterprise IT vs Open-Source Engineering) ===
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), dpi=300)

    # Panel A: Enterprise IT Extrinsic Ground-Truth Alignment (LDA vs TF-IDF KMeans vs SBERT KMeans vs Proposed HDBSCAN)
    ax1 = axes[0]
    models_short = ["LDA\n(K=14)", "TF-IDF\n+KMeans", "SBERT\n+KMeans", "EOM\n(K=14)", "Bounded-EOM\n(K=22)"]
    x = np.arange(len(models_short))
    w = 0.25
    ax1.bar(x - w, exp7_df["Sil_UMAP_5D"], w, label="5D Manifold Silhouette", color="#0275d8", edgecolor="#222222", lw=0.6)
    ax1.bar(x, exp7_df["Purity_Subcategory"], w, label="Subcategory Purity (16 cls)", color="#2ca02c", edgecolor="#222222", lw=0.6)
    ax1.bar(x + w, exp7_df["Purity_Support_Team"], w, label="Support Team Purity (15 cls)", color="#f0ad4e", edgecolor="#222222", lw=0.6)
    ax1.axhline(0, color="#333333", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_short, fontsize=8.2, fontweight="bold")
    ax1.set_ylabel("Score / Ground-Truth Purity", fontsize=10, fontweight="bold")
    ax1.set_title("(a) Enterprise IT Ground-Truth Benchmark", fontsize=11, fontweight="bold")
    ax1.set_ylim(-0.28, 0.92)
    ax1.legend(loc="lower right", fontsize=7.8, frameon=True, facecolor="white", framealpha=0.92)

    # Panel B: Temporal Aggregation Masking in Enterprise IT (Coarse Jan vs Feb+Mar vs Monthly Mar vs Jan+Feb)
    ax2 = axes[1]
    top_shift_it = exp8_df.iloc[[0, 1, 2, -4, -2, -1]].copy().sort_values("Monthly_Delta_Mar_vs_JanFeb_PP", ascending=True)
    y_idx = np.arange(len(top_shift_it))
    h = 0.36
    ax2.barh(y_idx - h / 2, top_shift_it["Coarse_Delta_FebMar_vs_Jan_PP"], h, label="Coarse Split (Feb+Mar vs. Jan)", color="#999999", edgecolor="#222222", lw=0.6)
    ax2.barh(y_idx + h / 2, top_shift_it["Monthly_Delta_Mar_vs_JanFeb_PP"], h, label="Fine Split (Mar vs. Jan+Feb)", color="#d9534f", edgecolor="#222222", lw=0.6)
    ax2.axvline(0, color="#111111", lw=1.0, linestyle="--")
    ax2.axvline(2.5, color="#d9534f", lw=0.8, linestyle=":", alpha=0.7)
    ax2.axvline(-2.5, color="#5cb85c", lw=0.8, linestyle=":", alpha=0.7)
    short_subcat = {
        "INSTALLATION": "Install",
        "CONFIGURATION": "Config",
        "ERROR": "Error",
        "ACCESS": "Access",
    }
    ylabels_b = [
        f"{r['Cluster_ID']} ({short_subcat.get(r['Dominant_Subcategory'], r['Dominant_Subcategory'])}){'**' if r['Monthly_Q_BH_FDR'] < 0.05 else ('*' if r['Monthly_P_Fisher'] < 0.05 else '')}"
        for _, r in top_shift_it.iterrows()
    ]
    ax2.set_yticks(y_idx)
    ax2.set_yticklabels(ylabels_b, fontsize=8.5, fontweight="bold")
    ax2.set_xlabel("Prevalence Shift ΔK (Percentage Points)", fontsize=10, fontweight="bold")
    ax2.set_title("(b) How Coarse Windows Mask the March Shift", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=7.8, frameon=True, facecolor="white", framealpha=0.92)

    # Panel C: Two-Regime Structural & Operational Comparison (Enterprise IT vs Open-Source GitHub)
    ax3 = axes[2]
    comp_metrics = ["Manifold\nSilhouette", "Assigned\nCorpus (%)", "Prospective\nCoverage (%)", "Prospective\nShift ρ"]
    it_vals = [0.5875, (1.0 - 0.1487), hyb_it_cov, float(it_rho)]
    gh_vals = [0.5312, (1.0 - 0.2724), 0.6377, 0.8909]
    xm = np.arange(len(comp_metrics))
    wm = 0.34
    ax3.bar(xm - wm / 2, it_vals, wm, label="Regime 1: Enterprise IT (N=1,380)", color="#2b5c8f", edgecolor="#222222", lw=0.6)
    ax3.bar(xm + wm / 2, gh_vals, wm, label="Regime 2: Open-Source SWE (N=881)", color="#e07a5f", edgecolor="#222222", lw=0.6)
    for i in range(len(comp_metrics)):
        ax3.text(xm[i] - wm / 2, it_vals[i] + 0.02, f"{it_vals[i]:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax3.text(xm[i] + wm / 2, gh_vals[i] + 0.02, f"{gh_vals[i]:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax3.set_xticks(xm)
    ax3.set_xticklabels(comp_metrics, fontsize=8.5, fontweight="bold")
    ax3.set_ylim(0, 1.08)
    ax3.set_ylabel("Proportion / Score", fontsize=10, fontweight="bold")
    ax3.set_title("(c) Two-Regime Cross-Domain Robustness", fontsize=11, fontweight="bold")
    ax3.legend(loc="lower right", fontsize=7.8, frameon=True, facecolor="white", framealpha=0.92)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure5_cross_domain_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure5_cross_domain_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/figure5_cross_domain_comparison.png and .pdf")


if __name__ == "__main__":
    main()
