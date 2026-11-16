"""Reproduce within-Q2 trajectories from the paper's saved Q1-Q2 corpus.

This runner does not fetch or expand the dataset, inject synthetic tickets, or
rerun/overwrite the existing LDA/K-Means benchmark results. It fits one shared
SBERT -> UMAP -> HDBSCAN topic space on the saved corpus, saves assignments and
provenance, then focuses rolling-window endpoints on a configurable quarter.
A fresh fit may have different numeric cluster IDs from historical runs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='github_issues_180d.csv')
    parser.add_argument('--quarter', default='2021Q2')
    parser.add_argument('--compare-baseline', help='Optionally export both comparison methods against this YYYYQn baseline')
    parser.add_argument('--output-dir', default='paper_q2_results')
    parser.add_argument('--window-days', type=int, default=14)
    parser.add_argument('--step-days', type=int, default=1)
    parser.add_argument('--min-window-tickets', type=int, default=5)
    parser.add_argument('--top-n', type=int, default=6)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--allow-other-corpus', action='store_true', help='Allow input other than the 881-record paper corpus')
    args = parser.parse_args(argv)

    import numpy as np
    import pandas as pd
    import hdbscan
    import umap
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    from temporal_tracking import (
        TemporalConfig, analyze_temporal_topics, export_temporal_analysis,
        focus_topic_quarter, export_focus_quarter,
    )

    config = TemporalConfig(window_days=args.window_days, step_days=args.step_days,
                            min_window_tickets=args.min_window_tickets)
    source = Path(args.data)
    frame = pd.read_csv(source)
    required = {'id', 'datetime', 'text'}
    if not required.issubset(frame.columns):
        parser.error('Input requires id, datetime, and text columns.')
    frame['datetime'] = pd.to_datetime(frame['datetime'], utc=True, errors='coerce', format='mixed').dt.tz_localize(None)
    if frame['datetime'].isna().any() or frame['id'].isna().any() or frame['id'].duplicated().any():
        parser.error('Paper reproducibility requires valid dates and unique nonmissing ticket IDs.')
    frame['quarter'] = frame['datetime'].dt.to_period('Q').astype(str)
    quarter_counts = {str(k): int(v) for k, v in frame['quarter'].value_counts().sort_index().items()}
    if not args.allow_other_corpus and (len(frame) != 881 or quarter_counts != {'2021Q1': 478, '2021Q2': 403}):
        parser.error('Expected the unchanged 881-record Q1-Q2 paper corpus; use --allow-other-corpus deliberately for other data.')
    # Preserve input row order for the joint fit, as in the original acquisition;
    # temporal_tracking orders calendar endpoints without changing ticket identity.
    docs = frame['text'].fillna('').astype(str).tolist()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print(f'Joint fit: {len(frame)} unchanged records; quarters: {quarter_counts}', flush=True)
    embedder = SentenceTransformer(args.model, device=args.device)
    embeddings = embedder.encode(docs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    reducer = umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric='cosine', random_state=42)
    reduced = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=16, min_samples=5, metric='euclidean', cluster_selection_method='eom')
    frame['hdbscan_cluster'] = clusterer.fit_predict(reduced)
    topics = sorted(int(c) for c in frame['hdbscan_cluster'].unique() if c != -1)
    if not topics:
        parser.error('The shared model found no non-noise topics; no paper topic figure can be produced.')
    print(f'Fit discovered {len(topics)} topics; noise tickets: {int((frame.hdbscan_cluster == -1).sum())}', flush=True)

    # Data-derived labels, never historical numeric-ID assumptions or LLM diagnoses.
    domain_stops = {
        'dataset', 'datasets', 'data', 'https', 'http', 'github', 'com', 'org',
        'add', 'added', 'adding', 'issue', 'issues', 'fix', 'fixed', 'pr', 'pull',
        'request', 'test', 'tests', 'using', 'use', 'used', 'file', 'files',
        'python', 'code', 'run', 'running', 'line', 'self', 'error', 'fails', 'failed',
        'like', 'need', 'want', 'does', 'way', 'make', 'just', 'new', 'support', 'version',
    }
    vectorizer = CountVectorizer(stop_words=sorted(ENGLISH_STOP_WORDS.union(domain_stops)),
                                 ngram_range=(1, 2), max_features=3000)
    cluster_docs = [' '.join(frame.loc[frame.hdbscan_cluster == c, 'text'].fillna('').astype(str)) for c in topics]
    counts = vectorizer.fit_transform(cluster_docs).toarray().astype(float)
    vocab = np.array(vectorizer.get_feature_names_out())
    frequency = (counts > 0).sum(axis=0)
    weights = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    weights *= np.log((len(topics) + 1.0) / (frequency + 1.0)) + 1.0
    weights[:, frequency > max(1, int(len(topics) * 0.35))] = 0.0
    labels, representations = {}, []
    for index, topic in enumerate(topics):
        ranked = np.argsort(weights[index])[::-1]
        keywords = vocab[[i for i in ranked if weights[index, i] > 0][:10]].tolist()
        labels[topic] = ' / '.join(keywords[:4]) if keywords else f'Topic {topic}'
        positions = np.flatnonzero(frame.hdbscan_cluster.to_numpy() == topic)
        centroid = embeddings[positions].mean(axis=0)
        representatives = positions[np.argsort(embeddings[positions] @ centroid)[::-1][:3]]
        representations.append({
            'Cluster_ID': topic, 'Topic_Label': labels[topic], 'Keywords': keywords,
            'Count': int(len(positions)),
            'Representative_Tickets': [
                {'id': int(frame.iloc[i]['id']), 'title': str(frame.iloc[i].get('title', '')),
                 'datetime': frame.iloc[i]['datetime'].isoformat()}
                for i in representatives
            ],
        })

    assignment_columns = ['id', 'datetime', 'quarter', 'hdbscan_cluster']
    frame[assignment_columns].to_csv(output / 'topic_assignments.csv', index=False)
    (output / 'topic_labels.json').write_text(json.dumps(labels, indent=2), encoding='utf-8')
    (output / 'topic_representations.json').write_text(json.dumps(representations, indent=2), encoding='utf-8')
    # Store vectors to support exact future temporal reruns and audit, not just aggregates.
    np.savez_compressed(output / 'joint_embeddings.npz', ticket_ids=frame.id.to_numpy(), embeddings=embeddings, umap_5d=reduced)
    analysis = analyze_temporal_topics(frame, topic_labels=labels, config=config)
    paths = export_temporal_analysis(analysis, output, make_plots=False)
    focus = focus_topic_quarter(analysis, quarter=args.quarter)
    paths.update(export_focus_quarter(focus, output, make_plots=not args.no_plots, top_n=args.top_n))
    if args.compare_baseline:
        from trajectory_comparisons import compare_quarter_trajectories, export_trajectory_comparisons
        comparison = compare_quarter_trajectories(analysis, baseline_quarter=args.compare_baseline, target_quarter=args.quarter)
        paths.update(export_trajectory_comparisons(comparison, output / 'comparisons',
                                                 make_plots=not args.no_plots, top_n=args.top_n))
    versions = {}
    for package in ['numpy', 'pandas', 'sentence-transformers', 'torch', 'transformers', 'umap-learn', 'hdbscan', 'scikit-learn', 'matplotlib']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest = {
        'dataset_path': str(source), 'dataset_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'record_count': len(frame), 'quarter_counts': quarter_counts, 'focus_quarter': args.quarter,
        'comparison_baseline': args.compare_baseline,
        'synthetic_records_added': 0, 'data_expanded': False,
        'model': args.model, 'device': args.device, 'row_order': 'unchanged input CSV order',
        'umap': {'n_components': 5, 'n_neighbors': 15, 'min_dist': 0.0, 'metric': 'cosine', 'random_state': 42},
        'hdbscan': {'min_cluster_size': 16, 'min_samples': 5, 'metric': 'euclidean', 'cluster_selection_method': 'eom'},
        'num_topics': len(topics), 'noise_count': int((frame.hdbscan_cluster == -1).sum()),
        'python': platform.python_version(), 'packages': versions,
        'historical_comparability': 'Fresh shared-model fit on the same records. Numeric IDs and memberships may differ from historical tables; labels are derived from this run. Existing benchmark outputs are not overwritten.',
        'paths': paths,
    }
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    print(focus['trajectories'].to_string(index=False))
    print(json.dumps(paths, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
