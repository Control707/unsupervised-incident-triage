# Overcoming Template Mega Clusters and Pooling Cancellation in Unsupervised Incident Triage

[![Showcase Website](https://img.shields.io/badge/Web_Showcase-GitHub_Pages-1768ac?style=flat-square)](https://control707.github.io/unsupervised-incident-triage/)
[![Notebook](https://img.shields.io/badge/Jupyter-End--to--End_Showcase-d97706?style=flat-square)](Incident_Triage_End_to_End_Showcase.ipynb)
[![Paper PDF](https://img.shields.io/badge/Paper-Preprint_PDF-0f766e?style=flat-square)](Final_Project_Report_V2.pdf)
[![Tests](https://img.shields.io/badge/Unit_Tests-90_Passing-16a34a?style=flat-square)](tests/)

**Authors:** [Zakaria Sherif](https://www.linkedin.com/in/zakaria-sherif/) · [Nicholas Cosereanu](https://www.linkedin.com/in/nick-cosereanu-69b968325/) · [Wyatt Barker](https://www.linkedin.com/in/james-wyatt-barker/)  
**Showcase Webpage:** https://control707.github.io/unsupervised-incident-triage/

---

## Overview

Enterprise helpdesks and open-source issue trackers ingest thousands of unstructured incident tickets each quarter. Standard unsupervised topic models systematically fail on these streams due to two structural traps:

1. **The Template & Mega Cluster Trap:** Sliding-window word coherence ($C_v$) rewards models that group recurring submission boilerplate (`hello team`, `error message`, `dataset`), while default HDBSCAN Excess of Mass (EOM) merges distinct software failure modes sharing low-level stack trace tokens into a single **298-ticket mega cluster (`R1`)**.
2. **The Static Window Trap:** Coarse two-period comparisons suffer from **pooling cancellation**—when an opposing dip and surge fall inside the same observation window, they cancel out once pooled—and cannot distinguish an ongoing chronic bug plateau from a transient internal coding sprint.

We resolve both traps with a three-stage framework (**Discover → Track → Validate**) evaluated across two operational domains:
- **Open-Source SWE Stream (`github_issues_180d.csv`):** 881 Hugging Face `datasets` issues and PRs across Q1–Q2 2021 (`478` Q1 + `403` Q2).
- **Enterprise IT Service Desk (`it_incident_tickets.csv`):** 1,380 synthetic helpdesk tickets (`1,338` in Q1 2021 + `42` in 2024), including `314` orphaned `Unassigned` error incidents.

---

## Key Results

| Finding | Result | Evidence |
| :--- | :--- | :--- |
| **Break the Mega Cluster** | Bounded EOM (`M_max = 120`) splits 298-ticket `R1` into **5 distinct subsystems** (`C5`, `C7`, `C2`, `C3`, `C8`) while leaving sibling clusters untouched (`ARI = 1.000`). | [`exp6_cluster_hierarchy_mapping.csv`](final_experiments/exp6_cluster_hierarchy_mapping.csv) |
| **Coherence Mirage Exposed** | LDA achieves highest $C_v$ (`0.948` SWE / `0.905` IT) with negative 5D Silhouette (`-0.225` / `-0.060`), while Bounded EOM hits **`0.531`** (SWE) and **`98.3%` IT type / `82.3%` subcategory purity**. | [`exp1_trimodel_ablation.csv`](final_experiments/exp1_trimodel_ablation.csv), [`exp7_it_trimodel_benchmark.csv`](final_experiments/exp7_it_trimodel_benchmark.csv) |
| **Plateaus vs. Sprints** | 14-day rolling windows show `R1` elevated above Q1 on **94.5% of Q2 days** (mean `+6.29 pp`), whereas `R3` (`task_cast`) spikes in late May (`16.7%` peak on May 26) and resolves. | [`baseline_excess_2021Q2_vs_2021Q1.csv`](paper_q2_results/comparisons/baseline_excess_2021Q2_vs_2021Q1.csv) |
| **Pooling Cancellation Rescued** | Coarse Feb+Mar vs. Jan pooling finds **0/14** FDR-significant IT shifts (`IT03` cancels 99% of its `-4.43 pp` Feb dip / `+9.88 pp` Mar surge). Monthly testing reveals **4/14** shifts (`q < 0.10`). | [`exp8_it_coarse_vs_monthly_shifts.csv`](final_experiments/exp8_it_coarse_vs_monthly_shifts.csv) |
| **Internal Sprint vs. User Bugs** | Withheld GitHub roles reveal **`R3` has 0.0% external authors** (`75% MEMBER, 25% CONTRIBUTOR`), versus `33%–54%` external users in `R0`, `R1`, `R6`. | [`exp6_extrinsic_validation_r10.csv`](final_experiments/exp6_extrinsic_validation_r10.csv) |
| **Orphaned Incident Grouping** | Groups **237 / 314 (75.5%)** orphaned `Unassigned` IT incidents with similar previously handled specialist tickets (`77` residual noise for human review). | [`exp9_it_extrinsic_triage_and_routing.csv`](final_experiments/exp9_it_extrinsic_triage_and_routing.csv) |

---

## Repository Contents

- **Showcase Webpage:** [`docs/index.html`](docs/index.html) (published via GitHub Pages)
- **Executed End-to-End Notebook:** [`Incident_Triage_End_to_End_Showcase.ipynb`](Incident_Triage_End_to_End_Showcase.ipynb) (and [`NLP_Final_Experiments_180d.ipynb`](NLP_Final_Experiments_180d.ipynb))
- **Paper & Poster (7372-UR):** [`Final_Project_Report_V2.pdf`](Final_Project_Report_V2.pdf), [`final_report_v2.tex`](final_report_v2.tex), [`Incident_Triage_Poster_High_Resolution_Final.pdf`](Overcoming-Template-Mega-Clusters-and-Pooling-Cancellation-in-Unsupervised-Incident-Triage-4/Incident_Triage_Poster_High_Resolution_Final.pdf), and [`Incident_Triage_Poster_High_Resolution_Final.pptx`](Overcoming-Template-Mega-Clusters-and-Pooling-Cancellation-in-Unsupervised-Incident-Triage-4/Incident_Triage_Poster_High_Resolution_Final.pptx)
- **Core Pipeline & Modules:**
  - [`run_final_experiments.py`](run_final_experiments.py) — GitHub SWE tri-model ablations, Bounded EOM, prospective validation, dose-response perturbation
  - [`run_cross_domain_it_experiments.py`](run_cross_domain_it_experiments.py) — Enterprise IT benchmark, pooling cancellation, extrinsic queue routing
  - [`temporal_tracking.py`](temporal_tracking.py) & [`trajectory_comparisons.py`](trajectory_comparisons.py) — Leakage-free UTC rolling-window tracking & Q2-vs-Q1 comparison library
  - [`run_q2_trajectories.py`](run_q2_trajectories.py) — Focused Q2 trajectory runner
- **Saved Outputs & Tests:** [`final_experiments/`](final_experiments/), [`paper_q2_results/`](paper_q2_results/), [`poster_work/hires_figures/`](poster_work/hires_figures/), [`tests/`](tests/) (`90` unit tests)

---

## Quickstart & Reproducibility

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Run all 90 temporal & trajectory unit tests
python -m unittest discover -s tests -v

# 2. Reproduce Q2 trajectories and comparisons from saved assignments (no model refit)
python temporal_tracking.py \
  --input paper_q2_results/topic_assignments.csv \
  --topic-labels paper_q2_results/topic_labels.json \
  --focus-quarter 2021Q2 --window-days 14 --step-days 1 --min-window-tickets 5 \
  --output-dir q2_recomputed

# 3. Preview the showcase webpage locally
python -m http.server --directory docs 8080
```

---

## Citation

```bibtex
@misc{sherif2026unsupervised,
  title  = {Overcoming Template Mega Clusters and Pooling Cancellation in Unsupervised Incident Triage},
  author = {Sherif, Zakaria and Cosereanu, Nicholas and Barker, Wyatt},
  year   = {2026},
  note   = {Preprint, Kennesaw State University},
  url    = {https://github.com/Control707/unsupervised-incident-triage}
}
```
