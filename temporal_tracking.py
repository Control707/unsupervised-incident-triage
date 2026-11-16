"""Retrospective temporal tracking for stable, already-assigned topic IDs.

Only pandas and NumPy are needed for analysis, CSV/JSON export, and the CLI
with ``--no-plots``.  Matplotlib is imported lazily when plots are requested.
The module does not fit or re-fit any topic model: comparisons assume that
all tickets were assigned in the same joint topic space.  Consequently these
are retrospective descriptions, not leakage-free online emergence estimates.

Prevalence always uses *all* valid-date tickets, including noise (-1), as its
denominator.  Thresholds are proportions: 0.025 means 2.5 percentage points.
Missing calendar quarters are unknown, not zero-prevalence observations.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import json
import math
from numbers import Integral, Real
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


QUARTERLY_COLUMNS = [
    "Quarter", "Cluster_ID", "Topic_Label", "Count", "Total_Tickets",
    "Prevalence", "Previous_Quarter", "Delta_K", "Pct_Change", "Category",
    "Is_Partial_Quarter",
]
MONTHLY_COLUMNS = [
    "Month", "Quarter", "Cluster_ID", "Topic_Label", "Count", "Total_Tickets",
    "Prevalence", "Is_Partial_Month",
]
FOCUS_TRAJECTORY_COLUMNS = [
    "Quarter", "Cluster_ID", "Topic_Label", "Count", "Quarter_Prevalence",
    "Supported_Endpoints", "Start_Date", "End_Date", "Start_Prevalence",
    "End_Prevalence", "Net_Change", "Peak_Date", "Peak_Prevalence",
    "Mean_Rolling_Prevalence", "Prevalence_Slope_Per_Week", "Trajectory",
]
ROLLING_COLUMNS = [
    "Window_Start", "Window_End", "Quarter", "Cluster_ID", "Topic_Label",
    "Count", "Total_Tickets", "Prevalence", "Delta_K", "Category",
    "Is_Partial_Window", "Is_Low_Support",
]
TRAJECTORY_COLUMNS = [
    "Cluster_ID", "Topic_Label", "Observed_Quarters", "Active_Quarters",
    "First_Seen_Quarter", "Last_Seen_Quarter", "Peak_Quarter",
    "Peak_Prevalence", "Start_Prevalence", "End_Prevalence", "Net_Change",
    "Mean_Prevalence", "Prevalence_Slope_Per_Quarter", "Trajectory",
]


@dataclass(frozen=True)
class TemporalConfig:
    """Calendar-window settings and proportion-scale classification thresholds.

    All integer settings must be positive (booleans are not integers here).
    Thresholds must be finite and nonnegative.  ``min_window_tickets`` applies
    only to rolling windows; quarters with at least one ticket are observed.
    """

    window_days: int = 14
    step_days: int = 1
    min_window_tickets: int = 1
    surge_threshold: float = 0.025
    persistence_threshold: float = 0.035

    def __post_init__(self) -> None:
        for name in ("window_days", "step_days", "min_window_tickets"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be a positive integer")
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        for name in ("surge_threshold", "persistence_threshold"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite nonnegative number")
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
            object.__setattr__(self, name, value)


def _validate_topics(series: pd.Series) -> np.ndarray:
    """Accept integral numeric values/strings without silently rounding IDs."""
    result = []
    invalid_positions = []
    max_id = np.iinfo(np.int64).max
    for position, value in enumerate(series):
        try:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError
            if isinstance(value, Integral):
                topic = int(value)
            elif isinstance(value, Real):
                if not math.isfinite(value) or value != int(value):
                    raise ValueError
                topic = int(value)
            elif isinstance(value, str):
                number = Decimal(value.strip())
                if not number.is_finite() or number != number.to_integral_value():
                    raise ValueError
                if number < -1 or number > max_id:
                    raise ValueError
                topic = int(number)
            else:
                raise ValueError
            if topic < -1 or topic > max_id:
                raise ValueError
            result.append(topic)
        except (ValueError, TypeError, OverflowError, InvalidOperation):
            invalid_positions.append(position)
            result.append(0)
    if invalid_positions:
        examples = ", ".join(map(str, invalid_positions[:5]))
        raise ValueError(
            "Topic IDs must be finite integers >= -1 within int64 range; "
            "-1 is the only allowed negative ID (noise). "
            f"Invalid zero-based row positions: {examples}"
            + (" ..." if len(invalid_positions) > 5 else "")
        )
    return np.asarray(result, dtype=np.int64)


def _topic_label(topic: int, labels: Mapping | None) -> str:
    if topic == -1:
        return "Noise / Outliers"
    value = None if labels is None else labels.get(topic, labels.get(str(topic)))
    if value is None or not str(value).strip():
        return f"Topic {topic}"
    return str(value)


def _reaches(value: float, threshold: float) -> bool:
    # Include exact threshold boundaries despite floating-point subtraction.
    return value >= threshold or math.isclose(value, threshold, rel_tol=1e-10, abs_tol=1e-12)


def _category(
    prevalence: float,
    previous: float,
    total: int,
    is_baseline: bool,
    config: TemporalConfig,
) -> str:
    if total == 0:
        return "No Data"
    if not math.isfinite(prevalence):
        return "Insufficient History"
    if is_baseline:
        return "Initial Observation"
    if not math.isfinite(previous):
        return "Insufficient History"
    delta = prevalence - previous
    if delta > 0 and _reaches(delta, config.surge_threshold):
        return "Emerging Surge"
    if delta < 0 and _reaches(-delta, config.surge_threshold):
        return "Declining"
    if (
        previous > 0 and prevalence > 0
        and _reaches(previous, config.persistence_threshold)
        and _reaches(prevalence, config.persistence_threshold)
    ):
        return "Persistent"
    return "Stable"


def _prevalences(counts: np.ndarray, totals: np.ndarray, supported: np.ndarray) -> np.ndarray:
    values = np.full(counts.shape, np.nan, dtype=float)
    np.divide(counts, totals[:, None], out=values, where=supported[:, None])
    return values


def _previous(values: np.ndarray) -> np.ndarray:
    previous = np.full(values.shape, np.nan, dtype=float)
    previous[1:] = values[:-1]
    return previous


def _quarterly_analysis(
    tickets: pd.DataFrame,
    topics: list[int],
    labels: dict[int, str],
    quarters: pd.PeriodIndex,
    first_day: pd.Timestamp,
    last_day: pd.Timestamp,
    config: TemporalConfig,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[bool]]:
    grouped = tickets.groupby(["_quarter", "_topic"]).size().unstack(fill_value=0)
    counts = grouped.reindex(index=quarters, columns=topics, fill_value=0).to_numpy(dtype=np.int64)
    totals = counts.sum(axis=1)
    prevalence = _prevalences(counts, totals, totals > 0)
    previous = _previous(prevalence)
    delta = prevalence - previous
    pct_change = np.full(prevalence.shape, np.nan, dtype=float)
    np.divide(100.0 * delta, previous, out=pct_change, where=previous > 0)
    # A quarter is potentially partial when the observed date span does not
    # cover both calendar boundaries.  Date bounds never prove completeness.
    partial = [
        bool(first_day > quarter.start_time.normalize() or last_day < quarter.end_time.normalize())
        for quarter in quarters
    ]
    quarter_names = [str(quarter) for quarter in quarters]
    rows = []
    for i, name in enumerate(quarter_names):
        for j, topic in enumerate(topics):
            rows.append({
                "Quarter": name,
                "Cluster_ID": topic,
                "Topic_Label": labels[topic],
                "Count": int(counts[i, j]),
                "Total_Tickets": int(totals[i]),
                "Prevalence": prevalence[i, j],
                "Previous_Quarter": quarter_names[i - 1] if i else None,
                "Delta_K": delta[i, j],
                "Pct_Change": pct_change[i, j],
                "Category": _category(prevalence[i, j], previous[i, j], int(totals[i]), i == 0, config),
                "Is_Partial_Quarter": partial[i],
            })
    return pd.DataFrame(rows, columns=QUARTERLY_COLUMNS), counts, totals, partial


def _monthly_analysis(
    tickets: pd.DataFrame,
    topics: list[int],
    labels: dict[int, str],
    first_day: pd.Timestamp,
    last_day: pd.Timestamp,
) -> pd.DataFrame:
    """Nonoverlapping ticket counts, not sums of overlapping window counts."""
    months = pd.period_range(first_day.to_period("M"), last_day.to_period("M"), freq="M")
    grouped = tickets.groupby([tickets["_date"].dt.to_period("M"), "_topic"]).size().unstack(fill_value=0)
    counts = grouped.reindex(index=months, columns=topics, fill_value=0).to_numpy(dtype=np.int64)
    totals = counts.sum(axis=1)
    prevalence = _prevalences(counts, totals, totals > 0)
    rows = []
    for i, month in enumerate(months):
        partial = bool(first_day > month.start_time.normalize() or last_day < month.end_time.normalize())
        for j, topic in enumerate(topics):
            rows.append({
                "Month": str(month),
                "Quarter": str(month.asfreq("Q")),
                "Cluster_ID": topic,
                "Topic_Label": labels[topic],
                "Count": int(counts[i, j]),
                "Total_Tickets": int(totals[i]),
                "Prevalence": prevalence[i, j],
                "Is_Partial_Month": partial,
            })
    return pd.DataFrame(rows, columns=MONTHLY_COLUMNS)


def _rolling_analysis(
    tickets: pd.DataFrame,
    topics: list[int],
    labels: dict[int, str],
    first_day: pd.Timestamp,
    last_day: pd.Timestamp,
    config: TemporalConfig,
) -> pd.DataFrame:
    endpoints = pd.date_range(first_day, last_day, freq=pd.DateOffset(days=config.step_days))
    if endpoints[-1] != last_day:
        endpoints = endpoints.append(pd.DatetimeIndex([last_day]))
    starts = endpoints - pd.Timedelta(days=config.window_days - 1)
    exclusive_ends = endpoints + pd.Timedelta(days=1)
    daily = tickets.groupby(["_day", "_topic"]).size().unstack(fill_value=0)
    daily = daily.reindex(columns=topics, fill_value=0).sort_index()
    # Sparse day counts and prefix sums avoid constructing a dense
    # date-by-ticket matrix.  Bound searches implement [start, end + 1 day).
    days = daily.index.to_numpy(dtype="datetime64[us]")
    left = np.searchsorted(days, starts.to_numpy(dtype="datetime64[us]"), side="left")
    right = np.searchsorted(days, exclusive_ends.to_numpy(dtype="datetime64[us]"), side="left")
    cumulative = np.vstack([
        np.zeros((1, len(topics)), dtype=np.int64),
        np.cumsum(daily.to_numpy(dtype=np.int64), axis=0),
    ])
    counts = cumulative[right] - cumulative[left]
    totals = counts.sum(axis=1)
    low_support = totals < config.min_window_tickets
    prevalence = _prevalences(counts, totals, ~low_support)
    previous = _previous(prevalence)
    delta = prevalence - previous
    rows = []
    for i, endpoint in enumerate(endpoints):
        for j, topic in enumerate(topics):
            rows.append({
                "Window_Start": starts[i],
                "Window_End": endpoint,
                "Quarter": str(endpoint.to_period("Q")),
                "Cluster_ID": topic,
                "Topic_Label": labels[topic],
                "Count": int(counts[i, j]),
                "Total_Tickets": int(totals[i]),
                "Prevalence": prevalence[i, j],
                "Delta_K": delta[i, j],
                "Category": _category(prevalence[i, j], previous[i, j], int(totals[i]), i == 0, config),
                "Is_Partial_Window": bool(starts[i] < first_day),
                "Is_Low_Support": bool(low_support[i]),
            })
    return pd.DataFrame(rows, columns=ROLLING_COLUMNS)


def _trajectory_analysis(
    topics: list[int],
    labels: dict[int, str],
    quarters: pd.PeriodIndex,
    counts: np.ndarray,
    totals: np.ndarray,
    config: TemporalConfig,
) -> pd.DataFrame:
    observed = totals > 0
    prevalence = _prevalences(counts, totals, observed)
    ordinals = np.asarray([quarter.ordinal for quarter in quarters], dtype=float)
    rows = []
    for j, topic in enumerate(topics):
        values = prevalence[observed, j]
        active_indices = np.flatnonzero(counts[:, j] > 0)
        n_observed = int(observed.sum())
        slope = math.nan
        if n_observed >= 2:
            x = ordinals[observed]
            centered = x - x.mean()
            slope = float(np.dot(centered, values - values.mean()) / np.dot(centered, centered))
        start = float(values[0]) if n_observed else math.nan
        end = float(values[-1]) if n_observed else math.nan
        net = end - start if n_observed >= 2 else math.nan
        peak_index = int(np.nanargmax(prevalence[:, j])) if n_observed else None
        mean = float(values.mean()) if n_observed else math.nan
        if topic == -1:
            trajectory = "Noise / Outliers"
        elif n_observed == 0:
            trajectory = "No Data"
        elif n_observed < 2:
            trajectory = "Insufficient History"
        elif net > 0 and _reaches(net, config.surge_threshold):
            trajectory = "Emerging"
        elif net < 0 and _reaches(-net, config.surge_threshold):
            trajectory = "Declining"
        elif bool(observed.all()) and all(
            value > 0 and _reaches(float(value), config.persistence_threshold) for value in values
        ):
            trajectory = "Persistent"
        else:
            trajectory = "Stable"
        rows.append({
            "Cluster_ID": topic,
            "Topic_Label": labels[topic],
            "Observed_Quarters": n_observed,
            "Active_Quarters": int(len(active_indices)),
            "First_Seen_Quarter": str(quarters[active_indices[0]]) if len(active_indices) else None,
            "Last_Seen_Quarter": str(quarters[active_indices[-1]]) if len(active_indices) else None,
            "Peak_Quarter": str(quarters[peak_index]) if peak_index is not None else None,
            "Peak_Prevalence": float(prevalence[peak_index, j]) if peak_index is not None else math.nan,
            "Start_Prevalence": start,
            "End_Prevalence": end,
            "Net_Change": net,
            "Mean_Prevalence": mean,
            "Prevalence_Slope_Per_Quarter": slope,
            "Trajectory": trajectory,
        })
    return pd.DataFrame(rows, columns=TRAJECTORY_COLUMNS)


def analyze_temporal_topics(
    df: pd.DataFrame,
    topic_labels: Mapping | None = None,
    config: TemporalConfig | None = None,
    date_col: str = "datetime",
    topic_col: str = "hdbscan_cluster",
) -> dict[str, Any]:
    """Return quarterly, rolling, monthly, trajectories DataFrames and metadata.

    Dates are parsed as UTC (naive input is assumed UTC), invalid dates are
    dropped and counted, and timestamps are made timezone-naive *after* UTC
    conversion.  At least one valid date is required.  All supplied topic IDs,
    even those on invalid-date rows, must be finite integral IDs >= -1.
    Integer-valued numeric strings are accepted; booleans are rejected.

    Labels may be keyed by integer IDs or their JSON string equivalents.  The
    observed noise topic is always labeled ``Noise / Outliers``.  Topics are
    taken from valid-date rows, not from extra entries in the label mapping.

    Quarterly changes compare only the immediately preceding calendar quarter;
    rolling changes compare only the preceding sampled endpoint, never across
    an unsupported endpoint.  Pct_Change is 100 * Delta_K / prior prevalence
    and is missing when that denominator is zero.  Calendar-quarter slopes
    omit missing-data quarters but preserve the actual elapsed quarter spacing.
    Single-quarter trajectories have insufficient history and a missing slope
    and net change.  Persistent trajectories require every quarter to be
    observed and every prevalence to meet the persistence threshold.

    ``monthly`` contains nonoverlapping calendar-month ticket counts for every
    topic and every month in the input date span, including explicit missing
    months (Total_Tickets=0 and Prevalence=NaN).  Is_Partial_Month uses calendar
    boundary coverage, just like Is_Partial_Quarter.  It supplies original
    ticket totals for :func:`focus_topic_quarter` without retaining raw tickets.

    Input frames are never mutated.  Date coverage flags are inferred from
    the min/max ticket days and do not establish dataset completeness.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not df.columns.is_unique:
        raise ValueError("Input DataFrame column names must be unique")
    missing = [column for column in (date_col, topic_col) if column not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(map(str, missing)))
    if df.empty:
        raise ValueError("At least one ticket with a valid date is required; input is empty")
    if config is None:
        config = TemporalConfig()
    if not isinstance(config, TemporalConfig):
        raise TypeError("config must be a TemporalConfig instance or None")
    if topic_labels is not None and not isinstance(topic_labels, Mapping):
        raise TypeError("topic_labels must be a mapping of topic IDs to labels")
    topic_values = _validate_topics(df[topic_col])
    parsed = pd.to_datetime(df[date_col], errors="coerce", utc=True, format="mixed")
    valid = parsed.notna().to_numpy()
    dropped = int((~valid).sum())
    if not valid.any():
        raise ValueError("At least one valid date is required; all input dates are invalid")
    # Resetting positional indices avoids accidental alignment on duplicate or
    # non-default source indices; working columns are separate from user data.
    dates = parsed.iloc[np.flatnonzero(valid)].dt.tz_convert(None).reset_index(drop=True)
    tickets = pd.DataFrame({"_date": dates, "_topic": topic_values[valid]})
    tickets["_day"] = tickets["_date"].dt.normalize()
    tickets["_quarter"] = tickets["_date"].dt.to_period("Q")
    first_day = tickets["_day"].min()
    last_day = tickets["_day"].max()
    quarters = pd.period_range(first_day.to_period("Q"), last_day.to_period("Q"), freq="Q")
    topics = sorted(int(topic) for topic in tickets["_topic"].unique())
    labels = {topic: _topic_label(topic, topic_labels) for topic in topics}
    quarterly, counts, totals, partial = _quarterly_analysis(
        tickets, topics, labels, quarters, first_day, last_day, config,
    )
    rolling = _rolling_analysis(tickets, topics, labels, first_day, last_day, config)
    monthly = _monthly_analysis(tickets, topics, labels, first_day, last_day)
    trajectories = _trajectory_analysis(topics, labels, quarters, counts, totals, config)
    partial_quarters = [str(quarter) for quarter, flag in zip(quarters, partial) if flag]
    caveats = [
        "Retrospective joint-space analysis: topic IDs must be assigned in a single stable "
        "joint topic space; no topic model is re-fit here. Joint fitting may use future "
        "information, so these are not leakage-free online emergence estimates.",
        "Coverage and partial-period flags are inferred from minimum and maximum UTC "
        "ticket dates. Ticket dates do not prove complete collection, and partial-period "
        "comparisons can be biased.",
        "All valid-date tickets, including noise (-1), are included in prevalence denominators.",
        "An empty calendar quarter is missing data, not a zero-prevalence observation. "
        "Changes never bridge a missing immediate predecessor.",
        "Thresholds are descriptive proportion-scale heuristics, not significance tests. "
        "Overlapping rolling windows are dependent observations.",
    ]
    summary = {
        "analysis_basis": "retrospective_joint_space",
        "config": asdict(config),
        "date_column": date_col,
        "topic_column": topic_col,
        "total_input_tickets": int(len(df)),
        "total_valid_tickets": int(len(tickets)),
        "dropped_invalid_dates": dropped,
        "topic_count": len(topics),
        "non_noise_topic_count": sum(topic != -1 for topic in topics),
        "noise_ticket_count": int((tickets["_topic"] == -1).sum()),
        "topic_labels": {str(topic): labels[topic] for topic in topics},
        "coverage_start": first_day.date().isoformat(),
        "coverage_end": last_day.date().isoformat(),
        "coverage": {
            "start_date": first_day.date().isoformat(),
            "end_date": last_day.date().isoformat(),
            "start_timestamp_utc": tickets["_date"].min().isoformat() + "Z",
            "end_timestamp_utc": tickets["_date"].max().isoformat() + "Z",
            "calendar_days": int((last_day - first_day).days + 1),
            "partial_quarters": partial_quarters,
            "inferred_from_ticket_date_bounds": True,
            "completeness_established": False,
            "timezone": "UTC (timestamps stored timezone-naive after conversion)",
        },
        "quarter_count": int(len(quarters)),
        "observed_quarter_count": int((totals > 0).sum()),
        "missing_quarter_count": int((totals == 0).sum()),
        "quarter_counts": {str(quarter): int(total) for quarter, total in zip(quarters, totals)},
        "partial_quarter_count": len(partial_quarters),
        "rolling_endpoint_count": int(rolling["Window_End"].nunique()),
        "partial_window_count": int(rolling.drop_duplicates("Window_End")["Is_Partial_Window"].sum()),
        "low_support_window_count": int(rolling.drop_duplicates("Window_End")["Is_Low_Support"].sum()),
        "prevalence_denominator": "all valid-date tickets, including noise (-1), in the period",
        "delta_units": "proportion (0.025 = 2.5 percentage points)",
        "pct_change_units": "percent relative to preceding quarterly prevalence; null when prior is zero",
        "trajectory_rules": {
            "emerging_declining": "first-to-last observed quarter net change reaches surge_threshold",
            "persistent": "at least two quarters, no missing quarters, all prevalences >= persistence_threshold",
            "slope": "least-squares prevalence per calendar quarter; missing quarters excluded without compressing time",
            "noise": "separate Noise / Outliers trajectory",
        },
        "caveats": caveats,
    }
    # Keep this invariant explicit so accidental NumPy scalars/NaNs in future
    # metadata additions cannot silently produce non-standard JSON exports.
    json.dumps(summary, allow_nan=False)
    return {
        "quarterly": quarterly, "rolling": rolling, "monthly": monthly,
        "trajectories": trajectories, "summary": summary,
    }


def _parse_focus_quarter(quarter: str) -> pd.Period:
    if not isinstance(quarter, str) or re.fullmatch(r"[0-9]{4}Q[1-4]", quarter) is None:
        raise ValueError("quarter must use strict YYYYQ[1-4] format, for example '2021Q2'")
    if int(quarter[:4]) < 1:
        raise ValueError("quarter year must be between 0001 and 9999")
    return pd.Period(quarter, freq="Q")


def focus_topic_quarter(analysis: Mapping, quarter: str = "2021Q2") -> dict[str, Any]:
    """Describe within-quarter trajectories using already-computed full-data windows.

    Always call :func:`analyze_temporal_topics` on the full input *before*
    selecting a focus quarter.  Window endpoints are filtered here without
    recomputation, so April windows can retain their March ticket history.
    ``Delta_K`` is preserved too, including any prior-quarter predecessor.

    The added monthly table supplies nonoverlapping target-quarter ticket
    counts; overlapping rolling-window counts are never summed as ticket
    totals.  All parent-analysis topics remain present, including noise and
    topics with zero tickets in the target quarter.  Months outside the global
    input date span are not invented.  A quarter with no selected endpoints or
    no target-quarter tickets raises ValueError.

    Start/End/Peak_Date and prevalence statistics use supported target-quarter
    endpoints only.  With one endpoint the start/end/peak are that observation;
    net change and weekly slope require at least two.  With no supported
    endpoints dates are NaT and statistics NaN.  Slope is least-squares
    prevalence per actual elapsed day multiplied by seven, not a change across
    quarters.  Persistent requires every selected endpoint to be supported and
    every positive prevalence to meet the configured threshold.  Noise is
    always classified separately.  Output metadata is strict JSON-safe.
    """
    target = _parse_focus_quarter(quarter)
    if not isinstance(analysis, Mapping):
        raise TypeError("analysis must be the mapping returned by analyze_temporal_topics")
    for key in ("rolling", "monthly", "summary"):
        if key not in analysis:
            raise ValueError(f"analysis is missing required key: {key}; analyze the full input first")
    for key in ("rolling", "monthly"):
        if not isinstance(analysis[key], pd.DataFrame):
            raise TypeError(f"analysis[{key!r}] must be a pandas DataFrame")
    rolling = analysis["rolling"].loc[analysis["rolling"]["Quarter"] == quarter].copy()
    if rolling.empty:
        raise ValueError(f"No rolling window endpoints match focus quarter {quarter}")
    rolling = rolling.sort_values(["Window_End", "Cluster_ID"], kind="stable").reset_index(drop=True)
    monthly = analysis["monthly"].loc[analysis["monthly"]["Quarter"] == quarter].copy()
    monthly = monthly.sort_values(["Month", "Cluster_ID"], kind="stable").reset_index(drop=True)
    target_ticket_count = int(monthly["Count"].sum())
    if target_ticket_count == 0:
        raise ValueError(f"No target-quarter tickets are available for {quarter}")
    config = TemporalConfig(**analysis["summary"]["config"])
    quarter_counts = monthly.groupby("Cluster_ID")["Count"].sum()
    rows = []
    for topic, data in rolling.groupby("Cluster_ID", sort=True):
        supported = data.loc[data["Prevalence"].notna() & ~data["Is_Low_Support"]].copy()
        values = supported["Prevalence"].to_numpy(dtype=float)
        n_supported = len(supported)
        start_date = supported["Window_End"].iloc[0] if n_supported else pd.NaT
        end_date = supported["Window_End"].iloc[-1] if n_supported else pd.NaT
        start = float(values[0]) if n_supported else math.nan
        end = float(values[-1]) if n_supported else math.nan
        net = end - start if n_supported >= 2 else math.nan
        slope = math.nan
        if n_supported >= 2:
            dates = pd.to_datetime(supported["Window_End"])
            elapsed = (dates - dates.iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 86400.0
            centered = elapsed - elapsed.mean()
            slope = float(7.0 * np.dot(centered, values - values.mean()) / np.dot(centered, centered))
        peak_position = int(np.argmax(values)) if n_supported else None
        if int(topic) == -1:
            trajectory = "Noise / Outliers"
        elif n_supported == 0:
            trajectory = "No Data"
        elif n_supported < 2:
            trajectory = "Insufficient History"
        elif net > 0 and _reaches(net, config.surge_threshold):
            trajectory = "Emerging"
        elif net < 0 and _reaches(-net, config.surge_threshold):
            trajectory = "Declining"
        elif n_supported == len(data) and all(
            value > 0 and _reaches(float(value), config.persistence_threshold) for value in values
        ):
            trajectory = "Persistent"
        else:
            trajectory = "Stable"
        count = int(quarter_counts.get(topic, 0))
        rows.append({
            "Quarter": quarter,
            "Cluster_ID": int(topic),
            "Topic_Label": str(data["Topic_Label"].iloc[0]),
            "Count": count,
            "Quarter_Prevalence": float(count / target_ticket_count),
            "Supported_Endpoints": n_supported,
            "Start_Date": start_date,
            "End_Date": end_date,
            "Start_Prevalence": start,
            "End_Prevalence": end,
            "Net_Change": net,
            "Peak_Date": supported["Window_End"].iloc[peak_position] if peak_position is not None else pd.NaT,
            "Peak_Prevalence": float(values[peak_position]) if peak_position is not None else math.nan,
            "Mean_Rolling_Prevalence": float(values.mean()) if n_supported else math.nan,
            "Prevalence_Slope_Per_Week": slope,
            "Trajectory": trajectory,
        })
    trajectories = pd.DataFrame(rows, columns=FOCUS_TRAJECTORY_COLUMNS)
    endpoints = rolling.drop_duplicates("Window_End")
    quarter_start = target.start_time.normalize()
    quarter_end = target.end_time.normalize()
    coverage_start = max(pd.Timestamp(analysis["summary"]["coverage_start"]), quarter_start)
    coverage_end = min(pd.Timestamp(analysis["summary"]["coverage_end"]), quarter_end)
    summary = {
        "focus_quarter": quarter,
        "analysis_basis": "retrospective_joint_space",
        "config": asdict(config),
        "quarter_start": quarter_start.date().isoformat(),
        "quarter_end": quarter_end.date().isoformat(),
        "coverage_start": coverage_start.date().isoformat(),
        "coverage_end": coverage_end.date().isoformat(),
        "endpoint_start": pd.Timestamp(endpoints["Window_End"].min()).date().isoformat(),
        "endpoint_end": pd.Timestamp(endpoints["Window_End"].max()).date().isoformat(),
        "target_ticket_count": target_ticket_count,
        "topic_count": int(len(trajectories)),
        "non_noise_topic_count": int((trajectories["Cluster_ID"] != -1).sum()),
        "noise_ticket_count": int(quarter_counts.get(-1, 0)),
        "endpoint_count": int(len(endpoints)),
        "supported_endpoint_count": int((endpoints["Prevalence"].notna() & ~endpoints["Is_Low_Support"]).sum()),
        "low_support_endpoint_count": int(endpoints["Is_Low_Support"].sum()),
        "partial_window_count": int(endpoints["Is_Partial_Window"].sum()),
        "is_partial_quarter": bool(coverage_start > quarter_start or coverage_end < quarter_end),
        "month_counts": {str(month): int(data["Count"].sum()) for month, data in monthly.groupby("Month")},
        "history_policy": "trailing windows retain prior-quarter context",
        "source_coverage_start": analysis["summary"]["coverage_start"],
        "source_coverage_end": analysis["summary"]["coverage_end"],
        "source_valid_tickets": int(analysis["summary"]["total_valid_tickets"]),
        "dropped_invalid_dates": int(analysis["summary"]["dropped_invalid_dates"]),
        "prevalence_denominator": "all valid-date tickets, including noise (-1), within each window or calendar month",
        "trajectory_rules": {
            "start_end": "first and last supported target-quarter endpoints; not calendar-quarter averages",
            "net_change": "end minus start; null when fewer than two supported endpoints",
            "slope": "least-squares prevalence per actual elapsed day times seven; unsupported endpoints excluded",
            "persistent": "at least two supported endpoints, all selected endpoints supported, all positive prevalences >= persistence_threshold",
            "noise": "separate Noise / Outliers trajectory",
        },
        "caveats": list(analysis["summary"].get("caveats", [])) + [
            "Retrospective within-quarter trajectories retain prior-quarter trailing-window context; "
            "they do not estimate a quarter-to-quarter slope.",
            "Monthly and target-quarter counts are nonoverlapping original-ticket counts. "
            "Rolling counts overlap and must not be added as ticket totals.",
            "Coverage bounds are inferred from source ticket dates, not proof of complete collection.",
        ],
    }
    json.dumps(summary, allow_nan=False)
    return {"rolling": rolling, "monthly": monthly, "trajectories": trajectories, "summary": summary}


def _plot_analysis(analysis: Mapping, output_dir: Path, top_n: int) -> dict[str, str]:
    """Render headless figures without importing pyplot or changing a backend."""
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib import colormaps, dates as mdates, ticker
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required only for plots. Install matplotlib in this Python "
            "environment, use make_plots=False, or rerun the CLI with --no-plots."
        ) from exc
    quarterly = analysis["quarterly"]
    rolling = analysis["rolling"]
    selected = (
        quarterly.groupby("Cluster_ID")["Count"].sum()
        .sort_values(ascending=False, kind="stable").head(top_n).index.tolist()
    )
    label_map = quarterly.drop_duplicates("Cluster_ID").set_index("Cluster_ID")["Topic_Label"].to_dict()
    cmap = colormaps["tab10" if len(selected) <= 10 else "tab20"]
    colors = {topic: ('#777777' if topic == -1 else cmap(i % cmap.N)) for i, topic in enumerate(selected)}
    # Keep noise distinguishable even when its prevalence overlaps another topic.
    styles = {topic: ('--' if topic == -1 else '-') for topic in selected}

    def legend_label(topic: int) -> str:
        label = label_map[topic]
        if len(label) > 85:
            label = label[:82] + "..."
        return label if topic == -1 else f"{topic}: {label}"

    def make_figure(*args: Any, **kwargs: Any) -> Any:
        figure = Figure(*args, **kwargs)
        FigureCanvasAgg(figure)
        return figure

    def decorate(axis: Any, title: str, date_axis: bool = False) -> None:
        axis.set_title(title)
        axis.set_ylabel("Share of all tickets")
        axis.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
        if date_axis:
            locator = mdates.AutoDateLocator(minticks=2, maxticks=7)
            axis.xaxis.set_major_locator(locator)
            axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            axis.set_xlabel("Window endpoint (UTC date)")

    paths = {}
    quarter_names = quarterly["Quarter"].drop_duplicates().tolist()
    fig = make_figure(figsize=(13, 6), layout="constrained")
    ax = fig.subplots()
    for topic in selected:
        data = quarterly[quarterly["Cluster_ID"] == topic]
        ax.plot(range(len(data)), data["Prevalence"], marker="o", linewidth=1.8,
                color=colors[topic], linestyle=styles[topic], zorder=3 if topic == -1 else 2,
                label=legend_label(topic))
    ax.set_xticks(range(len(quarter_names)), quarter_names, rotation=45, ha="right")
    decorate(ax, "Calendar-quarter topic prevalence (retrospective)")
    ax.set_xlabel("Calendar quarter; gaps indicate no data")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    path = output_dir / "quarterly_topic_trajectories.png"
    fig.savefig(path, dpi=160)
    fig.clear()
    paths["quarterly_plot"] = str(path)

    window_days = analysis["summary"]["config"]["window_days"]
    fig = make_figure(figsize=(13, 6), layout="constrained")
    ax = fig.subplots()
    for topic in selected:
        data = rolling[rolling["Cluster_ID"] == topic]
        ax.plot(data["Window_End"], data["Prevalence"], linewidth=1.5,
                color=colors[topic], linestyle=styles[topic], zorder=3 if topic == -1 else 2,
                label=legend_label(topic))
    decorate(ax, f"Trailing {window_days}-day topic prevalence (continuous across quarters)", True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    path = output_dir / "rolling_topic_trajectories.png"
    fig.savefig(path, dpi=160)
    fig.clear()
    paths["rolling_plot"] = str(path)

    endpoint_quarters = rolling["Quarter"].drop_duplicates().tolist()
    ncols = min(3, len(endpoint_quarters))
    nrows = math.ceil(len(endpoint_quarters) / ncols)
    fig = make_figure(figsize=(5 * ncols, 3.8 * nrows + 1), layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False, sharey=True)
    for axis, quarter in zip(axes.flat, endpoint_quarters):
        for topic in selected:
            data = rolling[(rolling["Cluster_ID"] == topic) & (rolling["Quarter"] == quarter)]
            axis.plot(data["Window_End"], data["Prevalence"], linewidth=1.4,
                      color=colors[topic], linestyle=styles[topic], zorder=3 if topic == -1 else 2,
                label=legend_label(topic))
        decorate(axis, quarter, True)
    for axis in list(axes.flat)[len(endpoint_quarters):]:
        axis.set_visible(False)
    fig.suptitle(f"Trailing {window_days}-day prevalence by endpoint quarter\n"
                 "Windows remain continuous across quarter boundaries", fontsize=13)
    # Put a single compact legend on the last occupied panel, not on each facet.
    list(axes.flat)[len(endpoint_quarters) - 1].legend(fontsize=7, loc="best")
    path = output_dir / "rolling_trajectories_by_quarter.png"
    fig.savefig(path, dpi=160)
    fig.clear()
    paths["rolling_by_quarter_plot"] = str(path)
    return paths


def export_temporal_analysis(
    analysis: Mapping,
    output_dir: str | Path = "temporal_results",
    make_plots: bool = True,
    top_n: int = 10,
) -> dict[str, str]:
    """Write the three tables, strict JSON metadata, and optional PNG figures.

    Return path strings under ``quarterly_csv``, ``rolling_csv``,
    ``trajectories_csv``, and ``summary_json``.  With plots enabled, also return
    ``quarterly_plot``, ``rolling_plot``, and ``rolling_by_quarter_plot``.
    ``top_n`` limits plots only; exported tables always contain every topic.
    Matplotlib is never imported if ``make_plots=False``.
    """
    if not isinstance(analysis, Mapping):
        raise TypeError("analysis must be the mapping returned by analyze_temporal_topics")
    for key in ("quarterly", "rolling", "trajectories", "summary"):
        if key not in analysis:
            raise ValueError(f"analysis is missing required key: {key}")
    for key in ("quarterly", "rolling", "trajectories"):
        if not isinstance(analysis[key], pd.DataFrame):
            raise TypeError(f"analysis[{key!r}] must be a pandas DataFrame")
    if isinstance(top_n, (bool, np.bool_)) or not isinstance(top_n, Integral) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    summary_text = json.dumps(analysis["summary"], indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    filenames = {
        "quarterly": "quarterly_topic_prevalence.csv",
        "rolling": "rolling_topic_windows.csv",
        "trajectories": "topic_trajectories.csv",
    }
    for key, filename in filenames.items():
        path = directory / filename
        analysis[key].to_csv(path, index=False)
        paths[f"{key}_csv"] = str(path)
    path = directory / "temporal_tracking_summary.json"
    path.write_text(summary_text, encoding="utf-8")
    paths["summary_json"] = str(path)
    if make_plots:
        paths.update(_plot_analysis(analysis, directory, int(top_n)))
    return paths


def _plot_focus_quarter(focus: Mapping, output_dir: Path, top_n: int) -> dict[str, str]:
    """Plot target-quarter rolling curves/heatmap with a shared topic palette."""
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib import colormaps, dates as mdates, ticker
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required only for plots. Install matplotlib in this Python "
            "environment, use make_plots=False, or rerun the CLI with --no-plots."
        ) from exc
    summary = focus["summary"]
    quarter = summary["focus_quarter"]
    target = _parse_focus_quarter(quarter)
    quarter_start = target.start_time.normalize()
    quarter_stop = target.end_time.normalize() + pd.Timedelta(days=1)
    trajectories = focus["trajectories"]
    ranked = trajectories.loc[trajectories["Cluster_ID"] != -1].sort_values(
        ["Count", "Cluster_ID"], ascending=[False, True], kind="stable",
    ).head(top_n)
    selected = ranked["Cluster_ID"].tolist()
    label_map = ranked.set_index("Cluster_ID")["Topic_Label"].to_dict()
    cmap = colormaps["tab10" if len(selected) <= 10 else "tab20"]
    colors = {topic: cmap(i % cmap.N) for i, topic in enumerate(selected)}
    rolling = focus["rolling"]

    def label(topic: int) -> str:
        text = str(label_map[topic])
        return f"{topic}: " + (text if len(text) <= 70 else text[:67] + "...")

    def make_figure(*args: Any, **kwargs: Any) -> Any:
        figure = Figure(*args, **kwargs)
        FigureCanvasAgg(figure)
        return figure

    def decorate_dates(axis: Any) -> None:
        axis.set_xlim(quarter_start, quarter_stop)
        axis.xaxis.set_major_locator(mdates.MonthLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        axis.grid(axis="x", alpha=0.3)
        axis.set_xlabel(f"Window endpoint (UTC date, {quarter[:4]})")
        axis.axvline(quarter_start, color="#555555", linewidth=0.8)
        axis.axvline(quarter_stop, color="#555555", linewidth=0.8)

    window_days = summary["config"]["window_days"]
    fig = make_figure(figsize=(12, 5), layout="constrained")
    axis = fig.subplots()
    for topic in selected:
        data = rolling.loc[rolling["Cluster_ID"] == topic]
        axis.plot(data["Window_End"], data["Prevalence"], color=colors[topic],
                  linewidth=1.7, label=label(topic))
    axis.set_title(f"{quarter}: within-quarter topic trajectories\n"
                   f"Trailing {window_days}-day windows retain prior-quarter context")
    decorate_dates(axis)
    axis.set_ylim(bottom=0)
    axis.set_ylabel("Share of all tickets (including noise)")
    axis.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    axis.grid(axis="y", alpha=0.15)
    if selected:
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    else:
        axis.text(0.5, 0.5, "No non-noise topics in the analysis", ha="center", va="center",
                  transform=axis.transAxes)
    path_png = output_dir / f"topic_trajectories_{quarter}.png"
    path_pdf = output_dir / f"topic_trajectories_{quarter}.pdf"
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_pdf)
    fig.clear()

    # Daily cells explicitly retain unavailable/unsupported endpoints as blank,
    # rather than interpolating them or duplicating the last observed value.
    days = pd.date_range(quarter_start, quarter_stop - pd.Timedelta(days=1), freq="D")
    fig = make_figure(figsize=(12, max(3.2, 0.65 * len(selected) + 1.8)), layout="constrained")
    axis = fig.subplots()
    if selected:
        matrix = (
            rolling.loc[rolling["Cluster_ID"].isin(selected)]
            .pivot(index="Cluster_ID", columns="Window_End", values="Prevalence")
            .reindex(index=selected, columns=days).to_numpy(dtype=float)
        )
        x_edges = mdates.date2num(pd.date_range(quarter_start, quarter_stop, freq="D").to_pydatetime())
        heatmap = axis.pcolormesh(
            x_edges, np.arange(len(selected) + 1), np.ma.masked_invalid(matrix),
            cmap="YlGnBu", vmin=0, vmax=max(0.01, float(np.nanmax(matrix))) if np.isfinite(matrix).any() else 1.0,
            shading="flat",
        )
        axis.set_yticks(np.arange(len(selected)) + 0.5, [label(topic) for topic in selected])
        for tick_label, topic in zip(axis.get_yticklabels(), selected):
            tick_label.set_color(colors[topic])
        axis.set_ylim(len(selected), 0)
        colorbar = fig.colorbar(heatmap, ax=axis, pad=0.02)
        colorbar.set_label("Share of all tickets")
        colorbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    else:
        axis.set_yticks([])
        axis.text(0.5, 0.5, "No non-noise topics in the analysis", ha="center", va="center",
                  transform=axis.transAxes)
    decorate_dates(axis)
    axis.set_title(f"{quarter}: rolling topic prevalence heatmap\n"
                   "Blank cells indicate unsampled or unsupported endpoints")
    path_heatmap = output_dir / f"topic_heatmap_{quarter}.png"
    fig.savefig(path_heatmap, dpi=180)
    fig.clear()
    return {
        "focus_plot": str(path_png), "focus_plot_pdf": str(path_pdf),
        "focus_heatmap": str(path_heatmap),
    }


def export_focus_quarter(
    focus: Mapping,
    output_dir: str | Path = "temporal_results",
    make_plots: bool = True,
    top_n: int = 6,
) -> dict[str, str]:
    """Export focused tables/metadata and optional within-quarter PNG/PDF plots.

    Tables include all topics, including noise.  Figures show the ``top_n``
    non-noise topics ranked by original target-quarter ticket counts, using
    consistent colors and calendar-month ticks on explicit quarter bounds.
    The curves reflect the configured endpoint sampling; the heatmap leaves
    unsampled or unsupported days blank.  No input data or fitted labels are
    changed and matplotlib is not imported when ``make_plots=False``.
    """
    if not isinstance(focus, Mapping):
        raise TypeError("focus must be the mapping returned by focus_topic_quarter")
    for key in ("rolling", "monthly", "trajectories", "summary"):
        if key not in focus:
            raise ValueError(f"focus is missing required key: {key}")
    for key in ("rolling", "monthly", "trajectories"):
        if not isinstance(focus[key], pd.DataFrame):
            raise TypeError(f"focus[{key!r}] must be a pandas DataFrame")
    if isinstance(top_n, (bool, np.bool_)) or not isinstance(top_n, Integral) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    quarter = focus["summary"]["focus_quarter"]
    _parse_focus_quarter(quarter)
    summary_text = json.dumps(focus["summary"], indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    filenames = {
        "rolling": f"rolling_topic_windows_{quarter}.csv",
        "monthly": f"monthly_topic_prevalence_{quarter}.csv",
        "trajectories": f"topic_trajectories_{quarter}.csv",
    }
    paths = {}
    for key, filename in filenames.items():
        path = directory / filename
        focus[key].to_csv(path, index=False)
        paths[f"focus_{key}_csv"] = str(path)
    path = directory / f"temporal_tracking_{quarter}_summary.json"
    path.write_text(summary_text, encoding="utf-8")
    paths["focus_summary_json"] = str(path)
    if make_plots:
        paths.update(_plot_focus_quarter(focus, directory, int(top_n)))
    return paths


def main(argv: list[str] | None = None) -> int:
    """Analyze a CSV of existing topic assignments; optional --topic-labels is JSON.

    Example: ``python temporal_tracking.py --input assigned_tickets.csv
    --topic-labels topic_labels.json --output-dir temporal_results --no-plots``.
    """
    parser = argparse.ArgumentParser(
        description="Retrospective rolling-window and calendar-quarter tracking of assigned topics. "
                    "No embedding or topic models are fit. Dates are normalized to UTC.",
    )
    parser.add_argument("--input", required=True, help="CSV containing dates and integer topic IDs")
    parser.add_argument("--output-dir", default="temporal_results")
    parser.add_argument("--window-days", type=int, default=14)
    parser.add_argument("--step-days", type=int, default=1)
    parser.add_argument("--min-window-tickets", type=int, default=1)
    parser.add_argument("--surge-threshold", type=float, default=0.025)
    parser.add_argument("--persistence-threshold", type=float, default=0.035)
    parser.add_argument("--date-column", default="datetime")
    parser.add_argument("--topic-column", default="hdbscan_cluster")
    parser.add_argument(
        "--topic-labels", help="Optional JSON file mapping topic IDs to human-readable labels",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plots; matplotlib is not required")
    parser.add_argument("--top-n", type=int, default=10, help="Maximum plotted topics; CSVs include all topics")
    parser.add_argument(
        "--focus-quarter", help="Also export within-quarter trajectories for strict YYYYQn (e.g. 2021Q2)",
    )
    args = parser.parse_args(argv)
    try:
        config = TemporalConfig(
            window_days=args.window_days,
            step_days=args.step_days,
            min_window_tickets=args.min_window_tickets,
            surge_threshold=args.surge_threshold,
            persistence_threshold=args.persistence_threshold,
        )
        frame = pd.read_csv(args.input)
        labels = None
        if args.topic_labels:
            with Path(args.topic_labels).open(encoding="utf-8") as stream:
                labels = json.load(stream)
        analysis = analyze_temporal_topics(
            frame, topic_labels=labels, config=config,
            date_col=args.date_column, topic_col=args.topic_column,
        )
        focus = focus_topic_quarter(analysis, args.focus_quarter) if args.focus_quarter else None
        paths = export_temporal_analysis(
            analysis, output_dir=args.output_dir, make_plots=not args.no_plots, top_n=args.top_n,
        )
        if focus is not None:
            paths.update(export_focus_quarter(
                focus, output_dir=args.output_dir, make_plots=not args.no_plots, top_n=min(args.top_n, 6),
            ))
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(paths, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
