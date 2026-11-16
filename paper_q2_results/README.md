# Real same-data Q2 trajectory run

These are **real outputs from the original saved 881-record corpus**, not the synthetic software examples.

- Input: `github_issues_180d.csv`, SHA-256 `a09b8a48dff17f963aee00b7d2ad2c46202cc0fe4dd055efa63549fe9af388ee`.
- Original records: 478 in Q1 2021 and 403 in Q2 2021; no added or synthetic records.
- Shared model fit: MiniLM → 5D UMAP → HDBSCAN, documented in `run_manifest.json`.
- Q2 focus: April 1–June 30, 2021, 91 daily endpoints; 14-day trailing windows, minimum support 5.
- Early-April windows retain March history; no quarter reset.
- All 91 endpoints meet support; no startup-partial Q2 windows.
- Monthly totals: April 139, May 130, June 134 = 403 Q2 records.

## Historical-model limitation

This is a **fresh shared-model fit on the same data**. It found **10 non-noise topics and 240 noise records** overall (105 noise records in Q2), not the historical report's 14-topic solution. Old per-ticket assignments were unavailable. Numeric IDs in these files refer ONLY to this run; the paper prefixes them with **R** to distinguish them from historical benchmark IDs. Existing benchmark tables were not overwritten. Do not claim these curves reproduce the exact historical topic memberships.

## Main artifacts

- `topic_trajectories_2021Q2.png` and `.pdf`: the six most frequent non-noise topics by Q2 count, noise retained in the denominator.
- `topic_heatmap_2021Q2.png`: same topic selection, day-by-day rolling shares.
- `topic_trajectories_2021Q2.csv`: all topics including noise; start/end/peak, net change, and slope per week.
- `monthly_topic_prevalence_2021Q2.csv`: non-overlapping April/May/June counts and shares.
- `rolling_topic_windows_2021Q2.csv`: only Q2 endpoints, prior-quarter context preserved.
- `topic_assignments.csv`, `topic_labels.json`: reuse these to change temporal settings without a model refit.
- `topic_representations.json`: keywords and representative ticket IDs/titles for this run.
- `joint_embeddings.npz`: ticket-ID-aligned SBERT/UMAP vectors.
- `run_manifest.json`, `temporal_tracking_2021Q2_summary.json`: reproducibility and coverage metadata.

## Descriptive observations

- R3 (task-casting keyword cluster): monthly share 1.44% → 9.23% → 2.99%; trailing peak **16.67% on May 26**.
- R0 (duplicate-key keyword cluster): monthly share 4.32% → 4.62% → 6.72%; trailing peak **12.73% on June 28**.
- R1 (broad pickle/dtype cluster): starting trailing share 28.95%, peak 50.00% on April 15, ending 35.48%; full-Q2 least-squares slope is slightly negative despite positive endpoint net change. It merges several serialization/dtype/I/O issues and is not a validated single failure mode.

These are retrospective prevalence patterns, not confirmed incident causes, remediation evidence, or prospective detector performance. Overlapping windows are dependent.

## Q2-minus-Q1 comparisons (no further refit)

Both approved options are now in `comparisons/`: Q2 excess over aggregate Q1 prevalence, and exact day-of-quarter Q2-minus-Q1 rolling differences. They reuse these exact assignments (no new model run). The former has 91 comparable endpoints; the latter has 77 pairs after excluding 13 Q1 startup positions and unmatched Q2 day 91. See `comparisons/README.md` for figures, interpretation, and the reproducible command. The paper adds the comparison section from `q1_q2_comparison_results.tex`. The new compiled deliverable is `Paper_With_Q1_Q2_Comparisons.pdf`; the earlier Q2-only paper is preserved.

## Recompute without refitting

From the project root:

```bash
python temporal_tracking.py \
  --input paper_q2_results/topic_assignments.csv \
  --topic-labels paper_q2_results/topic_labels.json \
  --focus-quarter 2021Q2 \
  --window-days 14 --step-days 1 --min-window-tickets 5 \
  --output-dir q2_recomputed
```

To rerun the model itself, use `run_q2_trajectories.py`; see the root README and recorded package versions.
