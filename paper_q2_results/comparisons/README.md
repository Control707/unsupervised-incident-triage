# Both Q2-minus-Q1 trajectory comparisons

These outputs reuse the **exact saved 881-ticket assignments** from the documented 10-topic rerun. There was **no model refit, topic relabeling, synthetic injection, or added data** in this step.

Input assignments SHA-256: `f7bf0c1aebc8fed81bda0ce989bdfabdad31d69130d400b402a77c267d21a3dc`.

## Option 1: Q2 excess above the overall Q1 baseline

For each topic, subtract `Q1 topic count / 478` from every full, supported Q2 trailing-window share. The PNG/PDF has two panels: actual Q2 prevalence and baseline-relative excess in **percentage points**.

- 91/91 Q2 daily endpoints comparable.
- Baseline is the overall original-ticket share, not a mean of overlapping windows.
- Positive/negative values mean higher/lower share than the Q1 baseline, not an incident-count difference or significance test.

Outputs: `baseline_excess_2021Q2_vs_2021Q1.csv`, `.png`, `.pdf`.

## Option 2: Q2 minus Q1 at the same day of quarter

Subtract full, supported Q1 rolling shares from Q2 rolling shares at the **exact one-based day of quarter**. No interpolation, stretching, quarter-fraction normalization, or nearest-neighbor matching is applied.

- 77 comparable pairs: days **14-90**.
- Days 1-13 excluded because corresponding Q1 windows lack a complete 14-day history.
- Q2 day 91 remains an unmatched row with missing difference.
- Actual baseline/target dates and exclusion reasons are saved. Example: day 56 pairs **May 26** with **February 25**, not the same calendar date or release event.
- Ordinal matching does not control weekdays, holidays, releases, or other external differences.

Outputs: `aligned_difference_2021Q2_vs_2021Q1.csv`, `.png`, `.pdf`.

## Descriptive results

All IDs below refer only to the saved rerun (R-prefix in paper), not historical 14-topic IDs.

| Topic cue | Q1 share | Mean excess, 91 endpoints | Mean paired difference, 77 pairs |
| --- | ---: | ---: | ---: |
| R1 pickle/dtype | 30.33% | +6.29 pp | +5.27 pp |
| R0 duplicate keys | 1.67% | +2.92 pp | +3.34 pp |
| R3 task casting | 0.42% | +4.26 pp | +4.98 pp |
| R6 NumPy/core | 12.55% | -6.56 pp | -6.14 pp |

- R3 exceeds its aggregate Q1 baseline on **80/91 endpoints**, with peak excess **+16.25 pp on May 26**. The same day has **+16.67 pp** paired difference against February 25.
- R6 is below its overall Q1 baseline on **all 91 Q2 endpoints**.
- R0 has peak excess **+11.05 pp on June 28**, but peak paired difference **+10.00 pp on April 21 versus January 21**. The earlier curve is variable, so these are different comparison questions.
- R1 has peak excess **+19.67 pp on April 15**, versus peak paired difference **+37.50 pp** against January 15. The latter partly reflects a low value at that particular Q1 position, not a stronger causal effect.

`comparison_topics_2021Q2_vs_2021Q1.csv` includes all topics and noise. `comparison_summary_2021Q2_vs_2021Q1.json` records settings, comparability counts/exclusions, assignment/label hashes, and interpretation limits.

Means use unweighted comparable sampled endpoints, not quarterly original-ticket weights. The two options use different reference constructions and endpoint sets. Overlapping windows are dependent; neither mean is an independent significance estimate. Noise remains in every denominator; figures show the same six most frequent non-noise Q2 topics.

## Reproduce without fitting a model

From the project root:

```bash
python trajectory_comparisons.py \
  --input paper_q2_results/topic_assignments.csv \
  --topic-labels paper_q2_results/topic_labels.json \
  --baseline-quarter 2021Q1 --target-quarter 2021Q2 \
  --window-days 14 --step-days 1 --min-window-tickets 5 \
  --top-n 6 --output-dir paper_q2_results/comparisons
```

Quarter settings are configurable. A changed step can produce fewer exact ordinal matches; the code does not interpolate to hide this.
