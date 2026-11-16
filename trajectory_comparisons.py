"""Two descriptive quarter-trajectory comparisons of existing topic assignments.

No embeddings or topic models are fit.  ``compare_quarter_trajectories`` consumes
one full-input result from ``temporal_tracking.analyze_temporal_topics`` so that
trailing windows retain their original cross-quarter history.  All prevalence
denominators include noise (-1).  Matplotlib is optional and imported lazily.

Method 1 subtracts the *aggregate* baseline-quarter prevalence from each target
rolling prevalence.  Method 2 pairs sampled rolling endpoints at exactly the
same one-based day of quarter.  It does not interpolate, stretch unequal
quarters, or map a target date onto a false baseline calendar date.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from temporal_tracking import TemporalConfig, _parse_focus_quarter, analyze_temporal_topics


BASELINE_EXCESS_COLUMNS = [
    "Baseline_Quarter", "Target_Quarter", "Cluster_ID", "Topic_Label",
    "Window_Start", "Window_End", "Day_Of_Quarter", "Count", "Total_Tickets",
    "Target_Prevalence", "Baseline_Count", "Baseline_Total_Tickets",
    "Baseline_Prevalence", "Excess_Prevalence", "Excess_PP", "Is_Comparable",
    "Is_Partial_Window", "Is_Low_Support", "Exclusion_Reason",
]
ALIGNED_DIFFERENCE_COLUMNS = [
    "Baseline_Quarter", "Target_Quarter", "Cluster_ID", "Topic_Label",
    "Day_Of_Quarter", "Baseline_Window_End", "Target_Window_End",
    "Baseline_Count", "Baseline_Total_Tickets", "Target_Count", "Target_Total_Tickets",
    "Baseline_Prevalence", "Target_Prevalence", "Difference", "Difference_PP",
    "Is_Comparable", "Exclusion_Reason", "Baseline_Is_Partial_Window",
    "Target_Is_Partial_Window", "Baseline_Is_Low_Support", "Target_Is_Low_Support",
]
COMPARISON_TOPIC_COLUMNS = [
    "Cluster_ID", "Topic_Label", "Baseline_Count", "Baseline_Total_Tickets",
    "Target_Count", "Target_Total_Tickets", "Baseline_Prevalence", "Target_Prevalence",
    "Quarterly_Delta_PP", "Baseline_Comparable_Endpoints", "Mean_Excess_PP",
    "Peak_Excess_PP", "Peak_Excess_Date", "Above_Baseline_Fraction",
    "Aligned_Comparable_Pairs", "Mean_Aligned_Difference_PP",
    "Peak_Aligned_Difference_PP", "Peak_Aligned_Target_Date",
]


def _require_columns(frame: pd.DataFrame, name: str, columns: list[str]) -> None:
    if not frame.columns.is_unique:
        raise ValueError(f"{name} has duplicate column names")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _validate_analysis(analysis: Mapping) -> tuple[pd.DataFrame, pd.DataFrame, TemporalConfig]:
    if not isinstance(analysis, Mapping):
        raise TypeError("analysis must be the mapping returned by analyze_temporal_topics")
    for key in ("quarterly", "rolling", "summary"):
        if key not in analysis:
            raise ValueError(f"analysis is missing required key: {key}")
    for key in ("quarterly", "rolling"):
        if not isinstance(analysis[key], pd.DataFrame):
            raise TypeError(f"analysis[{key!r}] must be a pandas DataFrame")
    if not isinstance(analysis["summary"], Mapping) or "config" not in analysis["summary"]:
        raise ValueError("analysis summary must contain a TemporalConfig config mapping")
    try:
        config = TemporalConfig(**analysis["summary"]["config"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid analysis config: {exc}") from exc
    quarterly = analysis["quarterly"].copy(deep=True)
    rolling = analysis["rolling"].copy(deep=True)
    _require_columns(quarterly, "quarterly", [
        "Quarter", "Cluster_ID", "Topic_Label", "Count", "Total_Tickets", "Prevalence", "Is_Partial_Quarter",
    ])
    _require_columns(rolling, "rolling", [
        "Quarter", "Cluster_ID", "Topic_Label", "Window_Start", "Window_End", "Count",
        "Total_Tickets", "Prevalence", "Is_Partial_Window", "Is_Low_Support",
    ])
    if quarterly.empty or rolling.empty:
        raise ValueError("quarterly and rolling analysis tables must be nonempty")
    for frame, name, flags, group_key in (
        (quarterly, "quarterly", ["Is_Partial_Quarter"], "Quarter"),
        (rolling, "rolling", ["Is_Partial_Window", "Is_Low_Support"], "Window_End"),
    ):
        for value in frame["Quarter"].unique():
            _parse_focus_quarter(value)
        ids = pd.to_numeric(frame["Cluster_ID"], errors="coerce")
        if (ids.isna().any() or not np.isfinite(ids.to_numpy(dtype=float)).all()
                or (ids < -1).any() or (ids != np.floor(ids)).any()
                or frame["Cluster_ID"].map(lambda value: isinstance(value, (bool, np.bool_))).any()):
            raise ValueError(f"{name} Cluster_ID values must be integer IDs >= -1")
        frame["Cluster_ID"] = ids.astype("int64")
        for column in ("Count", "Total_Tickets"):
            values = pd.to_numeric(frame[column], errors="coerce")
            if (values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all()
                    or (values < 0).any() or (values != np.floor(values)).any()):
                raise ValueError(f"{name} {column} must contain finite nonnegative integers")
            frame[column] = values.astype("int64")
        if (frame["Count"] > frame["Total_Tickets"]).any():
            raise ValueError(f"{name} Count cannot exceed Total_Tickets")
        values = pd.to_numeric(frame["Prevalence"], errors="coerce")
        if (frame["Prevalence"].notna() & values.isna()).any():
            raise ValueError(f"{name} Prevalence must be numeric or missing")
        finite_values = values.dropna().to_numpy(dtype=float)
        if not np.isfinite(finite_values).all() or ((finite_values < 0) | (finite_values > 1)).any():
            raise ValueError(f"{name} Prevalence must be a finite proportion or missing")
        frame["Prevalence"] = values.astype(float)
        if frame["Topic_Label"].isna().any():
            raise ValueError(f"{name} Topic_Label cannot be missing")
        frame["Topic_Label"] = frame["Topic_Label"].astype(str)
        for flag in flags:
            if frame[flag].isna().any() or not frame[flag].map(lambda value: isinstance(value, (bool, np.bool_))).all():
                raise ValueError(f"{name} {flag} must contain boolean values")
            frame[flag] = frame[flag].astype(bool)
        if name == "rolling":
            for date_column in ("Window_Start", "Window_End"):
                dates = pd.to_datetime(frame[date_column], errors="coerce", utc=True, format="mixed")
                if dates.isna().any():
                    raise ValueError(f"rolling {date_column} must contain valid dates")
                frame[date_column] = dates.dt.tz_convert(None)
            if (frame["Window_Start"] > frame["Window_End"]).any():
                raise ValueError("rolling Window_Start cannot follow Window_End")
            if not frame["Window_End"].dt.to_period("Q").astype(str).eq(frame["Quarter"]).all():
                raise ValueError("rolling Quarter must match the UTC Window_End quarter")
        if frame.duplicated([group_key, "Cluster_ID"]).any():
            raise ValueError(f"{name} has duplicate period/topic rows")
        groups = frame.groupby(group_key)
        if (groups["Total_Tickets"].nunique() > 1).any():
            raise ValueError(f"{name} total-ticket denominators must agree across topics")
        if not groups["Count"].sum().eq(groups["Total_Tickets"].first()).all():
            raise ValueError(f"{name} topic counts must sum to all-ticket denominators, including noise")
        expected_topics = set(frame["Cluster_ID"])
        if not groups["Cluster_ID"].apply(lambda values: set(values) == expected_topics).all():
            raise ValueError(f"{name} must include every topic in every period, zero-filled")
        for flag in flags:
            if (groups[flag].nunique() > 1).any():
                raise ValueError(f"{name} {flag} must agree across topics within each period")
        observed = frame["Prevalence"].notna()
        expected = frame["Count"].div(frame["Total_Tickets"].replace(0, np.nan))
        if not np.allclose(frame.loc[observed, "Prevalence"], expected.loc[observed], rtol=1e-9, atol=1e-12):
            raise ValueError(f"{name} Prevalence must equal Count / Total_Tickets where observed")
    if set(quarterly["Cluster_ID"]) != set(rolling["Cluster_ID"]):
        raise ValueError("quarterly and rolling topic ID sets must agree")
    if (quarterly["Total_Tickets"].gt(0) & quarterly["Prevalence"].isna()).any():
        raise ValueError("populated quarterly rows must have observed prevalence")
    return quarterly, rolling, config


def _exclusion_counts(frame: pd.DataFrame, date_column: str) -> dict[str, int]:
    unique = frame.drop_duplicates(date_column)
    return {str(reason): int(count) for reason, count in unique["Exclusion_Reason"].value_counts().sort_index().items()}


def compare_quarter_trajectories(
    analysis: Mapping,
    baseline_quarter: str = "2021Q1",
    target_quarter: str = "2021Q2",
) -> dict[str, Any]:
    """Compare quarters without refitting models or altering rolling history.

    ``baseline_excess`` subtracts aggregate baseline prevalence from target
    rolling prevalence.  ``aligned_difference`` uses an exact one-based ordinal
    day-of-quarter left join, retaining unmatched target endpoints.  Comparison
    values are missing for partial startup or unsupported windows; raw
    prevalences, counts and flags remain available for audit.  Aligned exclusion
    precedence is No Baseline Endpoint, Partial Window, Low Support, Comparable.

    Means/fractions are unweighted descriptive statistics over comparable
    sampled endpoints, not independent-sample estimates or significance tests.
    """
    baseline_period = _parse_focus_quarter(baseline_quarter)
    target_period = _parse_focus_quarter(target_quarter)
    if baseline_period >= target_period:
        raise ValueError("baseline_quarter must be distinct from and earlier than target_quarter")
    quarterly, rolling, config = _validate_analysis(analysis)
    aggregates = {}
    for quarter in (baseline_quarter, target_quarter):
        selected = quarterly.loc[quarterly["Quarter"] == quarter].copy()
        if selected.empty or int(selected["Count"].sum()) == 0:
            raise ValueError(f"Quarter {quarter} must be populated with original tickets")
        aggregates[quarter] = selected.set_index("Cluster_ID")
    baseline_aggregate = aggregates[baseline_quarter]
    target_aggregate = aggregates[target_quarter]
    baseline = rolling.loc[rolling["Quarter"] == baseline_quarter].copy()
    target = rolling.loc[rolling["Quarter"] == target_quarter].copy()
    if target.empty:
        raise ValueError(f"No sampled rolling endpoints match target quarter {target_quarter}")
    target = target.sort_values(["Window_End", "Cluster_ID"], kind="stable").reset_index(drop=True)
    target["Day_Of_Quarter"] = (target["Window_End"].dt.normalize() - target_period.start_time.normalize()).dt.days + 1
    baseline["Day_Of_Quarter"] = (baseline["Window_End"].dt.normalize() - baseline_period.start_time.normalize()).dt.days + 1

    excess = target[[
        "Cluster_ID", "Topic_Label", "Window_Start", "Window_End", "Day_Of_Quarter",
        "Count", "Total_Tickets", "Prevalence", "Is_Partial_Window", "Is_Low_Support",
    ]].rename(columns={"Prevalence": "Target_Prevalence"})
    excess["Baseline_Quarter"] = baseline_quarter
    excess["Target_Quarter"] = target_quarter
    for source, output in (("Count", "Baseline_Count"), ("Total_Tickets", "Baseline_Total_Tickets"),
                           ("Prevalence", "Baseline_Prevalence")):
        excess[output] = excess["Cluster_ID"].map(baseline_aggregate[source])
    target_low = excess["Is_Low_Support"] | excess["Total_Tickets"].lt(config.min_window_tickets) | excess["Target_Prevalence"].isna()
    excess["Exclusion_Reason"] = np.select(
        [excess["Is_Partial_Window"], target_low], ["Partial Window", "Low Support"], default="Comparable",
    )
    excess["Is_Comparable"] = excess["Exclusion_Reason"].eq("Comparable")
    excess["Excess_Prevalence"] = (excess["Target_Prevalence"] - excess["Baseline_Prevalence"]).where(excess["Is_Comparable"])
    excess["Excess_PP"] = 100.0 * excess["Excess_Prevalence"]
    excess = excess.reindex(columns=BASELINE_EXCESS_COLUMNS)

    baseline_part = baseline[[
        "Cluster_ID", "Day_Of_Quarter", "Window_End", "Count", "Total_Tickets", "Prevalence",
        "Is_Partial_Window", "Is_Low_Support",
    ]].rename(columns={
        "Window_End": "Baseline_Window_End", "Count": "Baseline_Count", "Total_Tickets": "Baseline_Total_Tickets",
        "Prevalence": "Baseline_Prevalence", "Is_Partial_Window": "Baseline_Is_Partial_Window",
        "Is_Low_Support": "Baseline_Is_Low_Support",
    })
    aligned = target[[
        "Cluster_ID", "Topic_Label", "Day_Of_Quarter", "Window_End", "Count", "Total_Tickets", "Prevalence",
        "Is_Partial_Window", "Is_Low_Support",
    ]].rename(columns={
        "Window_End": "Target_Window_End", "Count": "Target_Count", "Total_Tickets": "Target_Total_Tickets",
        "Prevalence": "Target_Prevalence", "Is_Partial_Window": "Target_Is_Partial_Window",
        "Is_Low_Support": "Target_Is_Low_Support",
    }).merge(baseline_part, on=["Cluster_ID", "Day_Of_Quarter"], how="left", validate="one_to_one", sort=False)
    aligned["Baseline_Quarter"] = baseline_quarter
    aligned["Target_Quarter"] = target_quarter
    for column in ("Baseline_Count", "Baseline_Total_Tickets"):
        aligned[column] = aligned[column].astype("Int64")
    no_match = aligned["Baseline_Window_End"].isna()
    baseline_partial = aligned["Baseline_Is_Partial_Window"].fillna(False).astype(bool)
    baseline_low = aligned["Baseline_Is_Low_Support"].fillna(False).astype(bool)
    partial = baseline_partial | aligned["Target_Is_Partial_Window"]
    low = (
        baseline_low | aligned["Target_Is_Low_Support"]
        | aligned["Baseline_Prevalence"].isna() | aligned["Target_Prevalence"].isna()
        | aligned["Baseline_Total_Tickets"].lt(config.min_window_tickets).fillna(False)
        | aligned["Target_Total_Tickets"].lt(config.min_window_tickets)
    )
    aligned["Exclusion_Reason"] = np.select(
        [no_match.to_numpy(dtype=bool), partial.to_numpy(dtype=bool), low.to_numpy(dtype=bool)],
        ["No Baseline Endpoint", "Partial Window", "Low Support"], default="Comparable",
    )
    aligned["Is_Comparable"] = aligned["Exclusion_Reason"].eq("Comparable")
    aligned["Difference"] = (aligned["Target_Prevalence"] - aligned["Baseline_Prevalence"]).where(aligned["Is_Comparable"])
    aligned["Difference_PP"] = 100.0 * aligned["Difference"]
    aligned = aligned.reindex(columns=ALIGNED_DIFFERENCE_COLUMNS)

    rows = []
    for topic in sorted(target_aggregate.index):
        e = excess.loc[(excess["Cluster_ID"] == topic) & excess["Is_Comparable"]]
        a = aligned.loc[(aligned["Cluster_ID"] == topic) & aligned["Is_Comparable"]]
        e_peak = e.loc[e["Excess_PP"].idxmax()] if not e.empty else None
        a_peak = a.loc[a["Difference_PP"].idxmax()] if not a.empty else None
        base_row, target_row = baseline_aggregate.loc[topic], target_aggregate.loc[topic]
        rows.append({
            "Cluster_ID": int(topic), "Topic_Label": str(target_row["Topic_Label"]),
            "Baseline_Count": int(base_row["Count"]), "Baseline_Total_Tickets": int(base_row["Total_Tickets"]),
            "Target_Count": int(target_row["Count"]), "Target_Total_Tickets": int(target_row["Total_Tickets"]),
            "Baseline_Prevalence": float(base_row["Prevalence"]), "Target_Prevalence": float(target_row["Prevalence"]),
            "Quarterly_Delta_PP": 100.0 * float(target_row["Prevalence"] - base_row["Prevalence"]),
            "Baseline_Comparable_Endpoints": int(len(e)),
            "Mean_Excess_PP": float(e["Excess_PP"].mean()) if len(e) else math.nan,
            "Peak_Excess_PP": float(e_peak["Excess_PP"]) if e_peak is not None else math.nan,
            "Peak_Excess_Date": e_peak["Window_End"] if e_peak is not None else pd.NaT,
            "Above_Baseline_Fraction": float(e["Excess_PP"].gt(0).mean()) if len(e) else math.nan,
            "Aligned_Comparable_Pairs": int(len(a)),
            "Mean_Aligned_Difference_PP": float(a["Difference_PP"].mean()) if len(a) else math.nan,
            "Peak_Aligned_Difference_PP": float(a_peak["Difference_PP"]) if a_peak is not None else math.nan,
            "Peak_Aligned_Target_Date": a_peak["Target_Window_End"] if a_peak is not None else pd.NaT,
        })
    topic_summary = pd.DataFrame(rows, columns=COMPARISON_TOPIC_COLUMNS)
    baseline_total = int(baseline_aggregate["Count"].sum())
    target_total = int(target_aggregate["Count"].sum())
    baseline_partial_quarter = bool(baseline_aggregate["Is_Partial_Quarter"].any())
    target_partial_quarter = bool(target_aggregate["Is_Partial_Quarter"].any())
    source_summary = analysis["summary"]
    summary = {
        "baseline_quarter": baseline_quarter, "target_quarter": target_quarter,
        "analysis_basis": "retrospective_joint_space", "refit_performed": False, "config": asdict(config),
        "baseline_ticket_count": baseline_total, "target_ticket_count": target_total,
        "topic_count": int(len(topic_summary)),
        "non_noise_topic_count": int((topic_summary["Cluster_ID"] != -1).sum()),
        "baseline_excess_endpoint_count": int(excess["Window_End"].nunique()),
        "baseline_excess_comparable_endpoint_count": int(excess.loc[excess["Is_Comparable"], "Window_End"].nunique()),
        "aligned_endpoint_count": int(aligned["Target_Window_End"].nunique()),
        "aligned_comparable_pair_count": int(aligned.loc[aligned["Is_Comparable"], "Target_Window_End"].nunique()),
        "baseline_excess_exclusion_counts": _exclusion_counts(excess, "Window_End"),
        "aligned_exclusion_counts": _exclusion_counts(aligned, "Target_Window_End"),
        "baseline_is_partial_quarter": baseline_partial_quarter,
        "target_is_partial_quarter": target_partial_quarter,
        "source_coverage_start": source_summary.get("coverage_start"),
        "source_coverage_end": source_summary.get("coverage_end"),
        "baseline_quarter_start": baseline_period.start_time.date().isoformat(),
        "baseline_quarter_end": baseline_period.end_time.date().isoformat(),
        "target_quarter_start": target_period.start_time.date().isoformat(),
        "target_quarter_end": target_period.end_time.date().isoformat(),
        "alignment_policy": "exact one-based day-of-quarter; no interpolation or normalized stretching",
        "history_policy": "original full-input trailing windows retain prior-quarter context",
        "baseline_excess_policy": "target rolling prevalence minus observed aggregate baseline-quarter prevalence",
        "comparison_units": "proportion; PP columns multiply proportions by 100",
        "prevalence_denominator": "all original tickets in the relevant period/window, including noise (-1)",
        "sample_count_units": "unique sampled target endpoints, not topic rows or independent tickets",
        "summary_weighting": "unweighted descriptive means/fractions over comparable sampled endpoints",
        "exclusion_precedence": ["No Baseline Endpoint", "Partial Window", "Low Support", "Comparable"],
        "baseline_coverage_warning": (
            "Baseline quarter is flagged partial from source date coverage; aggregate prevalence describes only observed tickets, with no extrapolation."
            if baseline_partial_quarter else
            "Source date bounds span the baseline quarter; dates do not prove complete collection."
        ),
        "caveats": [
            "Retrospective joint-space comparison of one unchanged set of assigned topic IDs; no refit. "
            "Joint fitting may use future information, so these are not leakage-free online estimates.",
            "Baseline-excess uses the observed aggregate baseline, including any partial-quarter coverage; no missing tickets are extrapolated.",
            "Matched trajectories align ordinal days, not identical calendar dates or causal event time. "
            "Unequal quarter lengths and sparse sampling leave unmatched target days missing.",
            "Partial startup and low-support windows retain their raw values for audit but are excluded from comparisons.",
            "Overlapping trailing windows are dependent; unweighted means and peak/fraction summaries are descriptive, not significance tests. "
            "Rolling counts are never summed as distinct ticket counts or interpreted as an AUC of distinct incidents.",
            "All prevalence denominators retain noise (-1). Figures select non-noise topics only; tables keep every topic.",
        ],
    }
    json.dumps(summary, allow_nan=False)
    return {"baseline_excess": excess, "aligned_difference": aligned, "topic_summary": topic_summary, "summary": summary}


def _plot_comparisons(comp: Mapping, output_dir: Path, top_n: int) -> dict[str, str]:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib import colormaps, dates as mdates, ticker
    except ImportError as exc:
        raise RuntimeError("Matplotlib is required only for plots; use make_plots=False or CLI --no-plots.") from exc
    summary = comp["summary"]
    baseline, target = summary["baseline_quarter"], summary["target_quarter"]
    suffix = f"{target}_vs_{baseline}"
    ranked = comp["topic_summary"].loc[comp["topic_summary"]["Cluster_ID"] != -1].sort_values(
        ["Target_Count", "Cluster_ID"], ascending=[False, True], kind="stable",
    ).head(top_n)
    selected = ranked["Cluster_ID"].tolist()
    labels = ranked.set_index("Cluster_ID")["Topic_Label"].to_dict()
    cmap = colormaps["tab10" if len(selected) <= 10 else "tab20"]
    colors = {topic: cmap(i % cmap.N) for i, topic in enumerate(selected)}

    def label(topic: int) -> str:
        text = str(labels[topic])
        return f"{topic}: " + (text if len(text) <= 65 else text[:62] + "...")

    def figure(*args: Any, **kwargs: Any) -> Any:
        fig = Figure(*args, **kwargs)
        FigureCanvasAgg(fig)
        return fig

    paths = {}
    fig = figure(figsize=(12, 8), layout="constrained")
    axes = fig.subplots(2, 1, sharex=True)
    for topic in selected:
        data = comp["baseline_excess"].loc[comp["baseline_excess"]["Cluster_ID"] == topic]
        axes[0].plot(data["Window_End"], data["Target_Prevalence"], color=colors[topic], label=label(topic), linewidth=1.6)
        axes[1].plot(data["Window_End"], data["Excess_PP"], color=colors[topic], linewidth=1.6)
    axes[0].set_title(f"{target} rolling topic prevalence and excess above aggregate {baseline}")
    axes[0].set_ylabel("Target share of all tickets")
    axes[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel("Excess above baseline (pp)")
    axes[1].axhline(0, color="#333333", linewidth=0.9, linestyle="--")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[1].set_xlabel("Target window endpoint (UTC date); excluded comparisons left blank")
    axes[1].set_xlim(pd.Timestamp(summary["target_quarter_start"]), pd.Timestamp(summary["target_quarter_end"]) + pd.Timedelta(days=1))
    for axis in axes:
        axis.grid(alpha=0.2)
    if selected:
        axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    else:
        axes[0].text(0.5, 0.5, "No non-noise topics", transform=axes[0].transAxes, ha="center")
    fig.supxlabel(
        f"Comparable target endpoints: {summary['baseline_excess_comparable_endpoint_count']}/{summary['baseline_excess_endpoint_count']}; "
        "overlapping windows are descriptive, not independent observations", fontsize=9,
    )
    for extension, key in (("png", "baseline_excess_plot"), ("pdf", "baseline_excess_pdf")):
        path = output_dir / f"baseline_excess_{suffix}.{extension}"
        fig.savefig(path, dpi=180)
        paths[key] = str(path)
    fig.clear()

    fig = figure(figsize=(12, 5), layout="constrained")
    axis = fig.subplots()
    for topic in selected:
        data = comp["aligned_difference"].loc[comp["aligned_difference"]["Cluster_ID"] == topic]
        axis.plot(data["Day_Of_Quarter"], data["Difference_PP"], color=colors[topic], linewidth=1.7, label=label(topic))
    axis.axhline(0, color="#333333", linewidth=0.9, linestyle="--")
    axis.set_title(f"{target} minus {baseline}: matched day-of-quarter rolling prevalence")
    axis.set_xlabel("One-based day of quarter (exact ordinal-day match; not calendar-date alignment)")
    axis.set_ylabel("Target − baseline (percentage points)")
    axis.grid(alpha=0.2)
    axis.set_xlim(1, int(comp["aligned_difference"]["Day_Of_Quarter"].max()))
    counts = summary["aligned_exclusion_counts"]
    excluded = "; ".join(f"{reason}: {count}" for reason, count in counts.items() if reason != "Comparable") or "None"
    fig.supxlabel(
        f"Comparable pairs: {summary['aligned_comparable_pair_count']}; excluded target days — {excluded}. "
        "No interpolation or quarter-length stretching.", fontsize=9,
    )
    if selected:
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    else:
        axis.text(0.5, 0.5, "No non-noise topics", transform=axis.transAxes, ha="center")
    for extension, key in (("png", "aligned_difference_plot"), ("pdf", "aligned_difference_pdf")):
        path = output_dir / f"aligned_difference_{suffix}.{extension}"
        fig.savefig(path, dpi=180)
        paths[key] = str(path)
    fig.clear()
    return paths


def export_trajectory_comparisons(
    comp: Mapping,
    output_dir: str | Path = "paper_q2_results/comparisons",
    make_plots: bool = True,
    top_n: int = 6,
) -> dict[str, str]:
    """Export all-topic tables, strict metadata, and optional top non-noise plots."""
    if not isinstance(comp, Mapping):
        raise TypeError("comp must be the mapping returned by compare_quarter_trajectories")
    for key in ("baseline_excess", "aligned_difference", "topic_summary", "summary"):
        if key not in comp:
            raise ValueError(f"comp is missing required key: {key}")
    for key in ("baseline_excess", "aligned_difference", "topic_summary"):
        if not isinstance(comp[key], pd.DataFrame):
            raise TypeError(f"comp[{key!r}] must be a pandas DataFrame")
    if isinstance(top_n, (bool, np.bool_)) or not isinstance(top_n, Integral) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    baseline = comp["summary"]["baseline_quarter"]
    target = comp["summary"]["target_quarter"]
    _parse_focus_quarter(baseline)
    _parse_focus_quarter(target)
    suffix = f"{target}_vs_{baseline}"
    summary_text = json.dumps(comp["summary"], indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for source, prefix, key in (
        ("baseline_excess", "baseline_excess", "baseline_excess_csv"),
        ("aligned_difference", "aligned_difference", "aligned_difference_csv"),
        ("topic_summary", "comparison_topics", "comparison_topics_csv"),
    ):
        path = directory / f"{prefix}_{suffix}.csv"
        comp[source].to_csv(path, index=False)
        paths[key] = str(path)
    path = directory / f"comparison_summary_{suffix}.json"
    path.write_text(summary_text, encoding="utf-8")
    paths["comparison_summary_json"] = str(path)
    if make_plots:
        paths.update(_plot_comparisons(comp, directory, int(top_n)))
    return paths


def _file_provenance(path: str) -> dict[str, Any]:
    source = Path(path)
    digest = hashlib.sha256()
    byte_count = 0
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            byte_count += len(chunk)
    return {"path": str(source.resolve()), "sha256": digest.hexdigest(), "bytes": byte_count, "refit_performed": False}


def main(argv: list[str] | None = None) -> int:
    """CSV-only CLI; hashes assignment/label inputs and explicitly records no refit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV of already-assigned tickets; no model fitting")
    parser.add_argument("--topic-labels", help="Optional JSON mapping of topic IDs to labels")
    parser.add_argument("--baseline-quarter", default="2021Q1")
    parser.add_argument("--target-quarter", default="2021Q2")
    parser.add_argument("--window-days", type=int, default=14)
    parser.add_argument("--step-days", type=int, default=1)
    parser.add_argument("--min-window-tickets", type=int, default=5)
    parser.add_argument("--date-column", default="datetime")
    parser.add_argument("--topic-column", default="hdbscan_cluster")
    parser.add_argument("--output-dir", default="paper_q2_results/comparisons")
    parser.add_argument("--no-plots", action="store_true", help="Do not import matplotlib or generate figures")
    parser.add_argument("--top-n", type=int, default=6)
    args = parser.parse_args(argv)
    try:
        config = TemporalConfig(
            window_days=args.window_days, step_days=args.step_days,
            min_window_tickets=args.min_window_tickets,
        )
        assignment_provenance = _file_provenance(args.input)
        labels = None
        label_provenance = None
        if args.topic_labels:
            label_provenance = _file_provenance(args.topic_labels)
            with Path(args.topic_labels).open(encoding="utf-8") as stream:
                labels = json.load(stream)
        frame = pd.read_csv(args.input)
        analysis = analyze_temporal_topics(
            frame, topic_labels=labels, config=config, date_col=args.date_column, topic_col=args.topic_column,
        )
        comparison = compare_quarter_trajectories(analysis, args.baseline_quarter, args.target_quarter)
        comparison["summary"]["assignment_provenance"] = assignment_provenance
        if label_provenance is not None:
            comparison["summary"]["topic_labels_provenance"] = label_provenance
        paths = export_trajectory_comparisons(
            comparison, output_dir=args.output_dir, make_plots=not args.no_plots, top_n=args.top_n,
        )
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(paths, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
