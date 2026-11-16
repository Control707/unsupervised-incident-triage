"""Regression coverage for configurable, within-quarter topic trajectories.

The focused view must reuse the full-stream rolling windows.  Filtering tickets
before computing those windows would incorrectly discard March context from
April endpoints; monthly/quarter counts, conversely, must exclude that context.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from temporal_tracking import (  # noqa: E402
    TemporalConfig,
    analyze_temporal_topics,
    export_focus_quarter,
    focus_topic_quarter,
)


MONTHLY_COLUMNS = {
    "Month", "Quarter", "Cluster_ID", "Topic_Label", "Count", "Total_Tickets",
    "Prevalence", "Is_Partial_Month",
}
FOCUS_TRAJECTORY_COLUMNS = {
    "Quarter", "Cluster_ID", "Topic_Label", "Count", "Quarter_Prevalence",
    "Supported_Endpoints", "Start_Date", "End_Date", "Start_Prevalence",
    "End_Prevalence", "Net_Change", "Peak_Date", "Peak_Prevalence",
    "Mean_Rolling_Prevalence", "Prevalence_Slope_Per_Week", "Trajectory",
}
LABELS = {0: "Authentication", 1: "Networking"}
EXPORT_NAMES = {
    "focus_rolling_csv": "rolling_topic_windows_{quarter}.csv",
    "focus_monthly_csv": "monthly_topic_prevalence_{quarter}.csv",
    "focus_trajectories_csv": "topic_trajectories_{quarter}.csv",
    "focus_summary_json": "temporal_tracking_{quarter}_summary.json",
}
PLOT_NAMES = {
    "focus_plot": "topic_trajectories_{quarter}.png",
    "focus_plot_pdf": "topic_trajectories_{quarter}.pdf",
    "focus_heatmap": "topic_heatmap_{quarter}.png",
}


def quarter_boundary_tickets() -> pd.DataFrame:
    """Unsorted duplicate-index data spanning Q1, every Q2 month, and Q3."""
    return pd.DataFrame(
        {
            "datetime": [
                "2021-06-30T23:59:59.999999999Z",
                "2021-03-30T23:59:59.999999999Z",
                "2021-04-01T23:59:59.999999999Z",
                "2021-03-31T23:59:59.999999999Z",
                "2021-04-01T00:00:00Z",
                "2021-04-30T23:59:59.999999999Z",
                "2021-05-01T00:00:00Z",
                "2021-05-31T23:59:59.999999999Z",
                "2021-06-01T00:00:00Z",
                "2021-07-01T00:00:00Z",
                "invalid-date",
            ],
            "hdbscan_cluster": [1, 0, -1, 0, 1, 0, 1, -1, 1, 0, 1],
        },
        index=[5, 5, 2, 2, 1, 1, 0, 0, 6, 6, 0],
    )


def as_utc(values) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce", format="mixed")


def day(value) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True).normalize()


def row_for(table: pd.DataFrame, **matches) -> pd.Series:
    selected = table
    for column, value in matches.items():
        selected = selected.loc[selected[column] == value]
    if len(selected) != 1:
        raise AssertionError(f"Expected exactly one row for {matches!r}; got {len(selected)}")
    return selected.iloc[0]


def ending_at(table: pd.DataFrame, date: str) -> pd.DataFrame:
    return table.loc[as_utc(table["Window_End"]) == day(date)]


def all_periods_in(quarter: str, table: pd.DataFrame, column: str) -> bool:
    periods = as_utc(table[column]).dt.tz_localize(None).dt.to_period("Q").astype(str)
    return bool((periods == quarter).all())


class MonthlyTopicPrevalenceTests(unittest.TestCase):
    def test_monthly_result_schema_and_actual_nonoverlapping_counts(self):
        frame = quarter_boundary_tickets()
        analysis = analyze_temporal_topics(frame, LABELS)
        self.assertIn("monthly", analysis)
        monthly = analysis["monthly"]
        self.assertTrue(MONTHLY_COLUMNS <= set(monthly.columns))
        self.assertFalse(monthly.duplicated(["Month", "Cluster_ID"]).any())
        self.assertEqual(monthly["Month"].drop_duplicates().tolist(), ["2021-03", "2021-04", "2021-05", "2021-06", "2021-07"])
        expected = {
            "2021-03": {-1: 0, 0: 2, 1: 0},
            "2021-04": {-1: 1, 0: 1, 1: 1},
            "2021-05": {-1: 1, 0: 0, 1: 1},
            "2021-06": {-1: 0, 0: 0, 1: 2},
            "2021-07": {-1: 0, 0: 1, 1: 0},
        }
        for month, counts in expected.items():
            total = sum(counts.values())
            for topic, count in counts.items():
                with self.subTest(month=month, topic=topic):
                    row = row_for(monthly, Month=month, Cluster_ID=topic)
                    self.assertEqual(row["Count"], count)
                    self.assertEqual(row["Total_Tickets"], total)
                    self.assertAlmostEqual(row["Prevalence"], count / total)
            group = monthly.loc[monthly["Month"] == month]
            self.assertEqual(group["Count"].sum(), total)
            self.assertAlmostEqual(group["Prevalence"].sum(), 1.0)
        self.assertEqual(monthly["Count"].sum(), analysis["summary"]["total_valid_tickets"])
        for topic, label in LABELS.items():
            self.assertEqual(set(monthly.loc[monthly["Cluster_ID"] == topic, "Topic_Label"]), {label})

    def test_missing_calendar_month_is_retained_as_unknown_not_zero_prevalence(self):
        frame = pd.DataFrame({"datetime": ["2021-04-01", "2021-06-30"], "hdbscan_cluster": [0, -1]})
        monthly = analyze_temporal_topics(frame)["monthly"]
        missing = monthly.loc[monthly["Month"] == "2021-05"]
        self.assertEqual(set(missing["Cluster_ID"]), {-1, 0})
        self.assertEqual(set(missing["Count"]), {0})
        self.assertEqual(set(missing["Total_Tickets"]), {0})
        self.assertTrue(missing["Prevalence"].isna().all())
        self.assertEqual(set(monthly["Quarter"]), {"2021Q2"})

    def test_partial_months_use_global_utc_date_coverage_not_ticket_density(self):
        monthly = analyze_temporal_topics(quarter_boundary_tickets())["monthly"]
        self.assertTrue(monthly.loc[monthly["Month"].isin(["2021-03", "2021-07"]), "Is_Partial_Month"].all())
        self.assertFalse(monthly.loc[monthly["Month"].isin(["2021-04", "2021-05", "2021-06"]), "Is_Partial_Month"].any())

    def test_monthly_assignment_uses_utc_across_year_boundary(self):
        frame = pd.DataFrame({
            "datetime": ["2022-01-01T00:30:00+01:00", "2021-12-31T23:30:00-01:00"],
            "hdbscan_cluster": [0, -1],
        })
        monthly = analyze_temporal_topics(frame)["monthly"]
        self.assertEqual(row_for(monthly, Month="2021-12", Cluster_ID=0)["Count"], 1)
        self.assertEqual(row_for(monthly, Month="2022-01", Cluster_ID=-1)["Count"], 1)
        self.assertEqual(row_for(monthly, Month="2021-12", Cluster_ID=0)["Quarter"], "2021Q4")
        self.assertEqual(row_for(monthly, Month="2022-01", Cluster_ID=-1)["Quarter"], "2022Q1")


class FocusQuarterAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.frame = quarter_boundary_tickets()
        self.analysis = analyze_temporal_topics(self.frame, LABELS, TemporalConfig(window_days=3))
        self.focus = focus_topic_quarter(self.analysis)

    def test_default_focus_schema_and_target_endpoint_filter(self):
        self.assertTrue({"rolling", "monthly", "trajectories", "summary"} <= set(self.focus))
        self.assertTrue(FOCUS_TRAJECTORY_COLUMNS <= set(self.focus["trajectories"].columns))
        self.assertEqual(set(self.focus["trajectories"]["Quarter"]), {"2021Q2"})
        self.assertEqual(set(self.focus["trajectories"]["Cluster_ID"]), {-1, 0, 1})
        self.assertTrue(all_periods_in("2021Q2", self.focus["rolling"], "Window_End"))
        self.assertEqual(set(self.focus["monthly"]["Month"]), {"2021-04", "2021-05", "2021-06"})
        self.assertEqual(as_utc(self.focus["rolling"]["Window_End"]).min(), day("2021-04-01"))
        self.assertEqual(as_utc(self.focus["rolling"]["Window_End"]).max(), day("2021-06-30"))
        self.assertEqual(self.focus["rolling"]["Window_End"].nunique(), 91)
        self.assertFalse(self.focus["trajectories"].duplicated("Cluster_ID").any())

    def test_april_rolling_windows_keep_march_history_without_q1_endpoints(self):
        first_day = ending_at(self.focus["rolling"], "2021-04-01")
        auth = row_for(first_day, Cluster_ID=0)
        self.assertEqual(day(auth["Window_Start"]), day("2021-03-30"))
        self.assertEqual(auth["Count"], 2)
        self.assertEqual(auth["Total_Tickets"], 4)
        self.assertAlmostEqual(auth["Prevalence"], 0.5)
        self.assertEqual(row_for(first_day, Cluster_ID=-1)["Count"], 1)
        self.assertEqual(row_for(first_day, Cluster_ID=1)["Count"], 1)
        self.assertTrue(all_periods_in("2021Q2", self.focus["rolling"], "Window_End"))
        global_q2 = self.analysis["rolling"].loc[self.analysis["rolling"]["Quarter"] == "2021Q2"]
        assert_frame_equal(self.focus["rolling"].reset_index(drop=True), global_q2.reset_index(drop=True))

    def test_target_counts_do_not_sum_overlapping_rolling_counts(self):
        trajectories = self.focus["trajectories"]
        expected_counts = {-1: 2, 0: 1, 1: 4}
        for topic, count in expected_counts.items():
            row = row_for(trajectories, Cluster_ID=topic)
            self.assertEqual(row["Count"], count)
            self.assertAlmostEqual(row["Quarter_Prevalence"], count / 7)
        self.assertEqual(trajectories["Count"].sum(), 7)
        self.assertAlmostEqual(trajectories["Quarter_Prevalence"].sum(), 1.0)
        self.assertEqual(self.focus["monthly"]["Count"].sum(), 7)

    def test_summary_preserves_source_config_and_is_strict_json_safe(self):
        summary = self.focus["summary"]
        self.assertEqual(summary["focus_quarter"], "2021Q2")
        self.assertEqual(summary["target_ticket_count"], 7)
        self.assertEqual(summary["topic_count"], 3)
        self.assertEqual(summary["endpoint_count"], 91)
        self.assertEqual(summary["quarter_start"], "2021-04-01")
        self.assertEqual(summary["quarter_end"], "2021-06-30")
        self.assertEqual(summary["coverage_start"], "2021-04-01")
        self.assertEqual(summary["coverage_end"], "2021-06-30")
        self.assertEqual(summary["endpoint_start"], "2021-04-01")
        self.assertEqual(summary["endpoint_end"], "2021-06-30")
        self.assertEqual(summary["config"], self.analysis["summary"]["config"])
        self.assertTrue(summary["history_policy"])
        self.assertTrue(summary["caveats"])
        unique_endpoints = self.focus["rolling"].drop_duplicates("Window_End")
        self.assertEqual(summary["supported_endpoint_count"], int((~unique_endpoints["Is_Low_Support"]).sum()))
        self.assertEqual(json.loads(json.dumps(summary, allow_nan=False)), summary)

    def test_focus_does_not_mutate_analysis_or_source_dataframe(self):
        originals = {
            key: value.copy(deep=True) if isinstance(value, pd.DataFrame) else json.loads(json.dumps(value, allow_nan=False))
            for key, value in self.analysis.items()
        }
        original_frame = self.frame.copy(deep=True)
        focus_topic_quarter(self.analysis, "2021Q2")
        for key, value in originals.items():
            if isinstance(value, pd.DataFrame):
                assert_frame_equal(self.analysis[key], value)
            else:
                self.assertEqual(self.analysis[key], value)
        assert_frame_equal(self.frame, original_frame)

    def test_unsorted_input_and_duplicate_indices_do_not_change_focus_results(self):
        reordered = self.frame.iloc[::-1].copy()
        reordered.index = [0] * len(reordered)
        result = focus_topic_quarter(analyze_temporal_topics(reordered, LABELS, TemporalConfig(window_days=3)))
        for name, columns in [("rolling", ["Window_End", "Cluster_ID"]), ("monthly", ["Month", "Cluster_ID"]), ("trajectories", ["Cluster_ID"])]:
            with self.subTest(table=name):
                assert_frame_equal(result[name].sort_values(columns).reset_index(drop=True), self.focus[name].sort_values(columns).reset_index(drop=True))

    def test_any_quarter_is_configurable_with_utc_and_year_handling(self):
        frame = pd.DataFrame({
            "datetime": [
                "2022-01-01T00:30:00+01:00",  # UTC 2021-12-31
                "2021-12-31T23:30:00-01:00",  # UTC 2022-01-01
                "2022-01-02T23:59:59.999999999Z",
            ],
            "hdbscan_cluster": [0, 1, -1],
        })
        analysis = analyze_temporal_topics(frame, config=TemporalConfig(window_days=2))
        result = focus_topic_quarter(analysis, "2022Q1")
        self.assertEqual(result["summary"]["focus_quarter"], "2022Q1")
        self.assertEqual(result["summary"]["target_ticket_count"], 2)
        self.assertEqual(result["summary"]["coverage_start"], "2022-01-01")
        self.assertEqual(result["summary"]["coverage_end"], "2022-01-02")
        self.assertTrue(all_periods_in("2022Q1", result["rolling"], "Window_End"))
        first = ending_at(result["rolling"], "2022-01-01")
        self.assertEqual(row_for(first, Cluster_ID=0)["Count"], 1)
        self.assertEqual(row_for(result["trajectories"], Cluster_ID=0)["Count"], 0)
        self.assertEqual(set(result["monthly"]["Month"]), {"2022-01"})
        self.assertTrue(result["monthly"]["Is_Partial_Month"].all())
        q4 = focus_topic_quarter(analysis, "2021Q4")
        self.assertEqual(q4["summary"]["target_ticket_count"], 1)
        self.assertTrue(all_periods_in("2021Q4", q4["rolling"], "Window_End"))

    def test_global_sampling_schedule_is_not_restarted_at_quarter_boundary(self):
        analysis = analyze_temporal_topics(self.frame, LABELS, TemporalConfig(window_days=3, step_days=13))
        result = focus_topic_quarter(analysis)
        expected = analysis["rolling"].loc[analysis["rolling"]["Quarter"] == "2021Q2"].reset_index(drop=True)
        assert_frame_equal(result["rolling"].reset_index(drop=True), expected)
        self.assertNotEqual(as_utc(result["rolling"]["Window_End"]).min(), day("2021-04-01"))


class FocusQuarterTrajectoryTests(unittest.TestCase):
    def test_within_quarter_direction_is_not_the_previous_quarter_comparison(self):
        frame = pd.DataFrame({
            "datetime": ["2021-03-31"] * 4 + ["2021-04-01"] * 4 + ["2021-04-02"] * 4,
            "hdbscan_cluster": [0] * 4 + [-1] * 4 + [0, -1, -1, -1],
        })
        analysis = analyze_temporal_topics(frame, config=TemporalConfig(window_days=1))
        between_quarters = row_for(analysis["quarterly"], Quarter="2021Q2", Cluster_ID=0)
        within_quarter = row_for(focus_topic_quarter(analysis)["trajectories"], Cluster_ID=0)
        self.assertIn("declining", between_quarters["Category"].lower())
        self.assertIn("emerging", within_quarter["Trajectory"].lower())
        self.assertAlmostEqual(within_quarter["Start_Prevalence"], 0.0)
        self.assertAlmostEqual(within_quarter["End_Prevalence"], 0.25)
        self.assertAlmostEqual(within_quarter["Net_Change"], 0.25)
        self.assertAlmostEqual(within_quarter["Prevalence_Slope_Per_Week"], 1.75)

    def test_slope_is_within_quarter_ols_per_week_using_actual_elapsed_days(self):
        frame = quarter_boundary_tickets()
        config = TemporalConfig(window_days=20, step_days=13)
        focus = focus_topic_quarter(analyze_temporal_topics(frame, LABELS, config))
        dates = pd.Series([pd.to_datetime(value, utc=True, errors="coerce") for value in frame["datetime"]])
        topics = frame["hdbscan_cluster"].reset_index(drop=True)
        for topic in (-1, 0, 1):
            with self.subTest(topic=topic):
                series = focus["rolling"].loc[focus["rolling"]["Cluster_ID"] == topic].sort_values("Window_End")
                expected_dates = []
                expected_values = []
                for _, endpoint in series.iterrows():
                    end = day(endpoint["Window_End"])
                    start = end - pd.Timedelta(days=config.window_days - 1)
                    mask = (dates >= start) & (dates < end + pd.Timedelta(days=1))
                    total = int(mask.sum())
                    if total >= config.min_window_tickets:
                        expected_dates.append(end)
                        expected_values.append(float((topics.loc[mask] == topic).sum()) / total)
                self.assertGreaterEqual(len(expected_dates), 3)
                elapsed_days = np.asarray([(date - expected_dates[0]).total_seconds() / 86400 for date in expected_dates])
                expected_slope = float(np.polyfit(elapsed_days, expected_values, 1)[0] * 7)
                row = row_for(focus["trajectories"], Cluster_ID=topic)
                self.assertEqual(row["Supported_Endpoints"], len(expected_dates))
                self.assertEqual(day(row["Start_Date"]), expected_dates[0])
                self.assertEqual(day(row["End_Date"]), expected_dates[-1])
                self.assertAlmostEqual(row["Start_Prevalence"], expected_values[0])
                self.assertAlmostEqual(row["End_Prevalence"], expected_values[-1])
                self.assertAlmostEqual(row["Net_Change"], expected_values[-1] - expected_values[0])
                self.assertAlmostEqual(row["Mean_Rolling_Prevalence"], float(np.mean(expected_values)))
                self.assertAlmostEqual(row["Prevalence_Slope_Per_Week"], expected_slope, places=12)
                peak = int(np.argmax(expected_values))
                self.assertEqual(day(row["Peak_Date"]), expected_dates[peak])
                self.assertAlmostEqual(row["Peak_Prevalence"], expected_values[peak])

    def test_single_supported_endpoint_has_no_net_change_or_slope(self):
        frame = pd.DataFrame({"datetime": ["2021-04-15", "2021-04-15T23:59:59Z"], "hdbscan_cluster": [0, -1]})
        focus = focus_topic_quarter(analyze_temporal_topics(frame))
        row = row_for(focus["trajectories"], Cluster_ID=0)
        self.assertEqual(row["Supported_Endpoints"], 1)
        self.assertEqual(day(row["Start_Date"]), day("2021-04-15"))
        self.assertEqual(day(row["End_Date"]), day("2021-04-15"))
        self.assertAlmostEqual(row["Start_Prevalence"], 0.5)
        self.assertAlmostEqual(row["End_Prevalence"], 0.5)
        self.assertTrue(pd.isna(row["Net_Change"]))
        self.assertTrue(pd.isna(row["Prevalence_Slope_Per_Week"]))
        self.assertEqual(row["Trajectory"], "Insufficient History")

    def test_no_supported_endpoints_keep_actual_counts_but_missing_trend_statistics(self):
        frame = pd.DataFrame({"datetime": ["2021-04-01", "2021-04-03"], "hdbscan_cluster": [0, -1]})
        focus = focus_topic_quarter(analyze_temporal_topics(frame, config=TemporalConfig(window_days=1, min_window_tickets=2)))
        self.assertTrue(focus["rolling"]["Is_Low_Support"].all())
        self.assertTrue(focus["rolling"]["Prevalence"].isna().all())
        self.assertEqual(focus["summary"]["target_ticket_count"], 2)
        self.assertEqual(focus["summary"]["supported_endpoint_count"], 0)
        row = row_for(focus["trajectories"], Cluster_ID=0)
        self.assertEqual(row["Count"], 1)
        self.assertAlmostEqual(row["Quarter_Prevalence"], 0.5)
        self.assertEqual(row["Supported_Endpoints"], 0)
        for column in ("Start_Date", "End_Date", "Start_Prevalence", "End_Prevalence", "Net_Change", "Peak_Date", "Peak_Prevalence", "Mean_Rolling_Prevalence", "Prevalence_Slope_Per_Week"):
            with self.subTest(column=column):
                self.assertTrue(pd.isna(row[column]))
        self.assertFalse(any(signal in row["Trajectory"].lower() for signal in ("emerging", "declining", "persistent")))
        json.dumps(focus["summary"], allow_nan=False)

    def test_support_gaps_do_not_compress_time_or_claim_persistence(self):
        frame = pd.DataFrame({
            "datetime": ["2021-04-01", "2021-04-04", "2021-04-11"],
            "hdbscan_cluster": [0, 0, 0],
        })
        focus = focus_topic_quarter(analyze_temporal_topics(frame, config=TemporalConfig(window_days=1)))
        row = row_for(focus["trajectories"], Cluster_ID=0)
        self.assertEqual(row["Supported_Endpoints"], 3)
        self.assertAlmostEqual(row["Prevalence_Slope_Per_Week"], 0.0)
        self.assertAlmostEqual(row["Net_Change"], 0.0)
        self.assertNotEqual(row["Trajectory"], "Persistent")
        self.assertTrue(ending_at(focus["rolling"], "2021-04-04")["Delta_K"].isna().all())

    def test_constant_supported_series_can_be_persistent_and_noise_stays_explicit(self):
        frame = pd.DataFrame({
            "datetime": ["2021-04-01", "2021-04-01", "2021-04-02", "2021-04-02"],
            "hdbscan_cluster": [0, -1, 0, -1],
        })
        focus = focus_topic_quarter(analyze_temporal_topics(frame, config=TemporalConfig(window_days=1)))
        row = row_for(focus["trajectories"], Cluster_ID=0)
        self.assertEqual(row["Trajectory"], "Persistent")
        self.assertAlmostEqual(row["Net_Change"], 0.0)
        self.assertAlmostEqual(row["Prevalence_Slope_Per_Week"], 0.0)
        self.assertEqual(row_for(focus["trajectories"], Cluster_ID=-1)["Trajectory"], "Noise / Outliers")

    def test_all_noise_focus_is_valid_and_uses_all_ticket_denominator(self):
        frame = pd.DataFrame({"datetime": ["2021-04-01", "2021-06-30"], "hdbscan_cluster": [-1, -1]})
        focus = focus_topic_quarter(analyze_temporal_topics(frame))
        self.assertEqual(set(focus["trajectories"]["Cluster_ID"]), {-1})
        row = focus["trajectories"].iloc[0]
        self.assertEqual(row["Count"], 2)
        self.assertEqual(row["Quarter_Prevalence"], 1.0)
        self.assertEqual(row["Trajectory"], "Noise / Outliers")
        self.assertTrue((focus["monthly"].loc[focus["monthly"]["Total_Tickets"] > 0, "Prevalence"] == 1.0).all())


class FocusQuarterValidationTests(unittest.TestCase):
    def setUp(self):
        self.analysis = analyze_temporal_topics(quarter_boundary_tickets(), config=TemporalConfig(step_days=10))

    def test_invalid_quarter_syntax_is_rejected(self):
        for quarter in ("Q2", "2021-Q2", "2021Q0", "2021Q5", "2021Q2-extra", "", None, 20212):
            with self.subTest(quarter=quarter):
                with self.assertRaises((ValueError, TypeError)):
                    focus_topic_quarter(self.analysis, quarter)

    def test_quarter_outside_source_coverage_is_rejected(self):
        for quarter in ("2020Q2", "2021Q4", "2022Q2"):
            with self.subTest(quarter=quarter):
                with self.assertRaises(ValueError):
                    focus_topic_quarter(self.analysis, quarter)

    def test_missing_target_quarter_cannot_be_invented_from_trailing_history(self):
        frame = pd.DataFrame({"datetime": ["2021-03-31", "2021-07-01"], "hdbscan_cluster": [0, 1]})
        analysis = analyze_temporal_topics(frame, config=TemporalConfig(window_days=14))
        april = ending_at(analysis["rolling"], "2021-04-01")
        self.assertEqual(set(april["Total_Tickets"]), {1})
        with self.assertRaises(ValueError):
            focus_topic_quarter(analysis, "2021Q2")


class FocusQuarterExportAndCliTests(unittest.TestCase):
    def setUp(self):
        self.analysis = analyze_temporal_topics(quarter_boundary_tickets(), LABELS, TemporalConfig(window_days=3))
        self.focus = focus_topic_quarter(self.analysis)

    def test_export_named_tables_and_strict_json_without_mutating_focus(self):
        originals = {name: self.focus[name].copy(deep=True) for name in ("rolling", "monthly", "trajectories")}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "focus results"
            paths = export_focus_quarter(self.focus, output, make_plots=False, top_n=1)
            self.assertEqual(set(paths), set(EXPORT_NAMES))
            tables = {"focus_rolling_csv": "rolling", "focus_monthly_csv": "monthly", "focus_trajectories_csv": "trajectories"}
            for key, name_template in EXPORT_NAMES.items():
                path = Path(paths[key])
                self.assertEqual(path.resolve(), (output / name_template.format(quarter="2021Q2")).resolve())
                self.assertTrue(path.is_file())
                if key in tables:
                    table = pd.read_csv(path)
                    self.assertEqual(len(table), len(self.focus[tables[key]]))
                    self.assertEqual(table.columns.tolist(), self.focus[tables[key]].columns.tolist())
                    self.assertEqual(set(table["Cluster_ID"]), {-1, 0, 1}, "top_n must not truncate CSVs")
                    self.assertFalse(any(name.startswith("Unnamed:") for name in table.columns))
            with Path(paths["focus_summary_json"]).open(encoding="utf-8") as handle:
                def reject_constant(value):
                    raise AssertionError(f"Non-standard JSON value {value}")
                self.assertEqual(json.load(handle, parse_constant=reject_constant), self.focus["summary"])
            self.assertEqual({path.name for path in output.iterdir()}, {name.format(quarter="2021Q2") for name in EXPORT_NAMES.values()})
        for name, original in originals.items():
            assert_frame_equal(self.focus[name], original)

    def test_nondefault_quarter_export_names_follow_requested_quarter(self):
        focus = focus_topic_quarter(self.analysis, "2021Q3")
        with tempfile.TemporaryDirectory() as temporary:
            paths = export_focus_quarter(focus, temporary, make_plots=False)
            for key, template in EXPORT_NAMES.items():
                self.assertEqual(Path(paths[key]).name, template.format(quarter="2021Q3"))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "optional matplotlib dependency is not available")
    def test_plot_export_contains_png_pdf_and_heatmap(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = export_focus_quarter(self.focus, temporary, make_plots=True, top_n=2)
            self.assertEqual(set(paths), set(EXPORT_NAMES) | set(PLOT_NAMES))
            for key, template in PLOT_NAMES.items():
                path = Path(paths[key])
                self.assertEqual(path.name, template.format(quarter="2021Q2"))
                self.assertGreater(path.stat().st_size, 0)
                signature = path.read_bytes()[:8]
                self.assertTrue(signature.startswith(b"%PDF-") if path.suffix == ".pdf" else signature == b"\x89PNG\r\n\x1a\n")

    def test_cli_focus_adds_target_outputs_without_removing_generic_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "tickets with spaces.csv"
            labels = Path(temporary) / "topic labels.json"
            output = Path(temporary) / "focus with spaces"
            quarter_boundary_tickets().to_csv(source, index=False)
            labels.write_text(json.dumps(LABELS), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "temporal_tracking.py"),
                 "--input", str(source), "--output-dir", str(output),
                 "--window-days", "3", "--focus-quarter", "2021Q2",
                 "--topic-labels", str(labels), "--no-plots"],
                cwd=temporary, capture_output=True, text=True, timeout=60,
                env={**os.environ, "MPLBACKEND": "Agg"},
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            for filename in ("rolling_topic_windows.csv", "quarterly_topic_prevalence.csv", "topic_trajectories.csv", "temporal_tracking_summary.json"):
                self.assertTrue((output / filename).is_file(), filename)
            for template in EXPORT_NAMES.values():
                self.assertTrue((output / template.format(quarter="2021Q2")).is_file())
            rolling = pd.read_csv(output / "rolling_topic_windows_2021Q2.csv")
            self.assertEqual(set(rolling["Quarter"]), {"2021Q2"})
            self.assertEqual(row_for(ending_at(rolling, "2021-04-01"), Cluster_ID=0)["Count"], 2)
            trajectories = pd.read_csv(output / "topic_trajectories_2021Q2.csv")
            self.assertEqual(row_for(trajectories, Cluster_ID=0)["Count"], 1)
            self.assertEqual(row_for(trajectories, Cluster_ID=0)["Topic_Label"], LABELS[0])
            with (output / "temporal_tracking_2021Q2_summary.json").open(encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["focus_quarter"], "2021Q2")
            self.assertEqual(summary["target_ticket_count"], 7)
            self.assertEqual(summary["config"]["window_days"], 3)
            self.assertFalse(list(output.glob("*.png")))
            self.assertFalse(list(output.glob("*.pdf")))

    def test_cli_invalid_focus_quarter_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "tickets.csv"
            quarter_boundary_tickets().to_csv(source, index=False)
            completed = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "temporal_tracking.py"),
                 "--input", str(source), "--output-dir", str(Path(temporary) / "results"),
                 "--focus-quarter", "2021Q5", "--no-plots"],
                cwd=temporary, capture_output=True, text=True, timeout=60,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertTrue(completed.stderr.strip())


if __name__ == "__main__":
    unittest.main()
