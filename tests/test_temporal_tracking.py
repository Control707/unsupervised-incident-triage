"""Regression tests for standalone, retrospective calendar topic tracking.

Run with ``python -m unittest discover -s tests -p 'test_temporal_tracking.py'``.
The fixtures deliberately keep noise in every prevalence denominator and cross
both year and quarter boundaries without requiring any NLP/model dependencies.
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
    export_temporal_analysis,
)


QUARTERLY_COLUMNS = {
    "Quarter", "Cluster_ID", "Topic_Label", "Count", "Total_Tickets",
    "Prevalence", "Previous_Quarter", "Delta_K", "Pct_Change", "Category",
    "Is_Partial_Quarter",
}
ROLLING_COLUMNS = {
    "Window_Start", "Window_End", "Quarter", "Cluster_ID", "Topic_Label",
    "Count", "Total_Tickets", "Prevalence", "Delta_K", "Category",
    "Is_Partial_Window", "Is_Low_Support",
}
TRAJECTORY_COLUMNS = {
    "Cluster_ID", "Topic_Label", "Observed_Quarters", "Active_Quarters",
    "First_Seen_Quarter", "Last_Seen_Quarter", "Peak_Quarter",
    "Peak_Prevalence", "Start_Prevalence", "End_Prevalence", "Net_Change",
    "Mean_Prevalence", "Prevalence_Slope_Per_Quarter", "Trajectory",
}
LABELS = {0: "Authentication", 1: "Networking"}


def boundary_tickets() -> pd.DataFrame:
    """Six valid tickets, a bad date, unsorted rows, and duplicate index labels."""
    return pd.DataFrame(
        {
            "datetime": [
                "2024-01-03T23:59:59.999999999Z",
                "2023-12-31T23:59:59.999999999Z",
                "2024-01-01T23:59:59.999999999Z",
                "not-a-date",
                "2023-12-30T23:59:59.999999999Z",
                "2024-01-02T23:59:59.999999999Z",
                "2024-01-01T00:00:00Z",
            ],
            "hdbscan_cluster": [1, -1, 0, 1, 0, -1, 1],
            "unused_text": ["ticket"] * 7,
        },
        index=[8, 8, 2, 10, 2, 1, 1],
    )


def gap_tickets() -> pd.DataFrame:
    """A completely unobserved quarter separates two fully bounded quarters."""
    return pd.DataFrame(
        {
            "datetime": [
                "2023-10-01", "2023-12-31", "2024-04-01",
                "2024-06-30", "2024-06-30T23:59:59.999999999Z",
            ],
            "hdbscan_cluster": [0, -1, 1, 1, -1],
        }
    )


def normalized_days(values) -> pd.Series:
    """Independently normalize each scalar to avoid pandas mixed-format inference."""
    return pd.Series([pd.to_datetime(value, utc=True, errors="coerce") for value in values])


def one_row(table: pd.DataFrame, **matches) -> pd.Series:
    selected = table
    for column, value in matches.items():
        selected = selected.loc[selected[column] == value]
    if len(selected) != 1:
        raise AssertionError(f"Expected one row matching {matches!r}, got {len(selected)}")
    return selected.iloc[0]


def day(value) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True).normalize()


def rows_for_end(table: pd.DataFrame, end: str) -> pd.DataFrame:
    endings = pd.to_datetime(table["Window_End"], utc=True)
    return table.loc[endings == day(end)]


class TemporalAnalysisContractTests(unittest.TestCase):
    def setUp(self):
        self.frame = boundary_tickets()
        self.config = TemporalConfig(window_days=3, step_days=2)
        self.analysis = analyze_temporal_topics(self.frame, LABELS, self.config)

    def test_default_config_matches_public_contract(self):
        config = TemporalConfig()
        self.assertEqual(config.window_days, 14)
        self.assertEqual(config.step_days, 1)
        self.assertEqual(config.min_window_tickets, 1)
        self.assertAlmostEqual(config.surge_threshold, 0.025)
        self.assertAlmostEqual(config.persistence_threshold, 0.035)

    def test_result_schema_and_unique_observation_keys(self):
        self.assertTrue({"quarterly", "rolling", "trajectories", "summary"} <= set(self.analysis))
        for name, required, keys in [
            ("quarterly", QUARTERLY_COLUMNS, ["Quarter", "Cluster_ID"]),
            ("rolling", ROLLING_COLUMNS, ["Window_End", "Cluster_ID"]),
            ("trajectories", TRAJECTORY_COLUMNS, ["Cluster_ID"]),
        ]:
            with self.subTest(table=name):
                table = self.analysis[name]
                self.assertIsInstance(table, pd.DataFrame)
                self.assertTrue(required <= set(table.columns), required - set(table.columns))
                self.assertFalse(table.duplicated(keys).any())
                self.assertEqual(set(table["Cluster_ID"]), {-1, 0, 1})
        self.assertIsInstance(self.analysis["summary"], dict)

    def test_quarter_prevalence_uses_all_tickets_and_zero_fills_topics(self):
        quarterly = self.analysis["quarterly"]
        self.assertEqual(set(quarterly["Quarter"]), {"2023Q4", "2024Q1"})
        self.assertEqual(len(quarterly), 6)
        expected = {
            ("2023Q4", -1): (1, 2, 0.5),
            ("2023Q4", 0): (1, 2, 0.5),
            ("2023Q4", 1): (0, 2, 0.0),
            ("2024Q1", -1): (1, 4, 0.25),
            ("2024Q1", 0): (1, 4, 0.25),
            ("2024Q1", 1): (2, 4, 0.5),
        }
        for (quarter, topic), (count, total, prevalence) in expected.items():
            with self.subTest(quarter=quarter, topic=topic):
                row = one_row(quarterly, Quarter=quarter, Cluster_ID=topic)
                self.assertEqual(row["Count"], count)
                self.assertEqual(row["Total_Tickets"], total)
                self.assertAlmostEqual(row["Prevalence"], prevalence)
        for _, group in quarterly.groupby("Quarter"):
            self.assertEqual(group["Count"].sum(), group["Total_Tickets"].iloc[0])
            self.assertAlmostEqual(group["Prevalence"].sum(), 1.0)

    def test_quarter_comparisons_cross_year_boundary_without_infinite_growth(self):
        quarterly = self.analysis["quarterly"]
        first = quarterly.loc[quarterly["Quarter"] == "2023Q4"]
        self.assertTrue(first["Delta_K"].isna().all())
        self.assertTrue(first["Pct_Change"].isna().all())
        declining = one_row(quarterly, Quarter="2024Q1", Cluster_ID=0)
        emerging = one_row(quarterly, Quarter="2024Q1", Cluster_ID=1)
        self.assertEqual(declining["Previous_Quarter"], "2023Q4")
        self.assertAlmostEqual(declining["Delta_K"], -0.25)
        self.assertAlmostEqual(declining["Pct_Change"], -50.0)
        self.assertAlmostEqual(emerging["Delta_K"], 0.5)
        self.assertIn("declining", declining["Category"].lower())
        self.assertIn("emerging", emerging["Category"].lower())
        self.assertFalse(np.isinf(quarterly["Pct_Change"].to_numpy(dtype=float)).any())

    def test_explicit_noise_and_topic_label_mapping_are_preserved(self):
        for name in ("quarterly", "rolling", "trajectories"):
            table = self.analysis[name]
            for cluster_id, label in LABELS.items():
                self.assertEqual(set(table.loc[table["Cluster_ID"] == cluster_id, "Topic_Label"]), {label})
            noise_labels = table.loc[table["Cluster_ID"] == -1, "Topic_Label"]
            self.assertFalse(noise_labels.empty)
            self.assertTrue(noise_labels.map(lambda value: isinstance(value, str) and bool(value.strip())).all())

    def test_invalid_dates_are_dropped_and_summary_is_strict_json(self):
        summary = self.analysis["summary"]
        self.assertEqual(summary["total_input_tickets"], 7)
        self.assertEqual(summary["total_valid_tickets"], 6)
        self.assertEqual(summary["dropped_invalid_dates"], 1)
        self.assertEqual(summary["quarter_count"], 2)
        self.assertEqual(summary["observed_quarter_count"], 2)
        self.assertEqual(summary["missing_quarter_count"], 0)
        self.assertEqual(summary["topic_count"], 3)
        self.assertEqual(summary["quarter_counts"], {"2023Q4": 2, "2024Q1": 4})
        self.assertEqual(summary["coverage_start"], "2023-12-30")
        self.assertEqual(summary["coverage_end"], "2024-01-03")
        self.assertEqual(summary["config"]["window_days"], 3)
        self.assertEqual(summary["config"]["step_days"], 2)
        self.assertEqual(summary["analysis_basis"], "retrospective_joint_space")
        self.assertTrue(summary["caveats"])
        encoded = json.dumps(summary, allow_nan=False)
        self.assertEqual(json.loads(encoded), summary)

    def test_analysis_does_not_mutate_input_even_with_duplicate_indices(self):
        original = boundary_tickets()
        assert_frame_equal(self.frame, original)
        analyze_temporal_topics(self.frame, LABELS, self.config)
        assert_frame_equal(self.frame, original)

    def test_input_order_and_duplicate_index_do_not_change_results(self):
        reordered = self.frame.iloc[::-1].copy()
        reordered.index = [0] * len(reordered)
        actual = analyze_temporal_topics(reordered, LABELS, self.config)
        sort_keys = {
            "quarterly": ["Quarter", "Cluster_ID"],
            "rolling": ["Window_End", "Cluster_ID"],
            "trajectories": ["Cluster_ID"],
        }
        for table, keys in sort_keys.items():
            with self.subTest(table=table):
                expected_table = self.analysis[table].sort_values(keys).reset_index(drop=True)
                actual_table = actual[table].sort_values(keys).reset_index(drop=True)
                assert_frame_equal(actual_table, expected_table)

    def test_custom_column_names_and_default_topic_labels(self):
        renamed = self.frame.rename(columns={"datetime": "occurred_at", "hdbscan_cluster": "topic"})
        result = analyze_temporal_topics(renamed, date_col="occurred_at", topic_col="topic")
        self.assertEqual(result["summary"]["total_valid_tickets"], 6)
        self.assertEqual(set(result["quarterly"]["Cluster_ID"]), {-1, 0, 1})
        self.assertFalse(result["quarterly"]["Topic_Label"].isna().any())

    def test_label_mapping_does_not_create_unobserved_topics(self):
        result = analyze_temporal_topics(self.frame, {**LABELS, 777: "Never observed"})
        for table in ("quarterly", "rolling", "trajectories"):
            self.assertEqual(set(result[table]["Cluster_ID"]), {-1, 0, 1})


class RollingWindowTests(unittest.TestCase):
    def test_exact_inclusive_days_for_multiple_window_and_step_sizes(self):
        frame = boundary_tickets()
        timestamps = normalized_days(frame["datetime"].tolist())
        valid = timestamps.notna()
        valid_times = timestamps.loc[valid]
        topics = frame["hdbscan_cluster"].reset_index(drop=True).loc[valid]
        first_day = valid_times.min().normalize()
        last_day = valid_times.max().normalize()
        for window_days, step_days in [(1, 1), (2, 2), (3, 2), (14, 9)]:
            with self.subTest(window_days=window_days, step_days=step_days):
                result = analyze_temporal_topics(frame, config=TemporalConfig(window_days=window_days, step_days=step_days))
                rolling = result["rolling"]
                expected_ends = list(pd.date_range(first_day, last_day, freq=f"{step_days}D"))
                if expected_ends[-1] != last_day:
                    expected_ends.append(last_day)
                actual_ends = pd.to_datetime(rolling["Window_End"], utc=True).drop_duplicates().tolist()
                self.assertEqual(actual_ends, expected_ends)
                for end in expected_ends:
                    start = end - pd.Timedelta(days=window_days - 1)
                    in_window = (valid_times >= start) & (valid_times < end + pd.Timedelta(days=1))
                    group = rows_for_end(rolling, str(end))
                    self.assertEqual(set(group["Cluster_ID"]), {-1, 0, 1})
                    for _, row in group.iterrows():
                        expected_count = int((topics.loc[in_window] == row["Cluster_ID"]).sum())
                        self.assertEqual(day(row["Window_Start"]), start)
                        self.assertEqual(row["Count"], expected_count)
                        self.assertEqual(row["Total_Tickets"], int(in_window.sum()))
                        self.assertEqual(row["Quarter"], f"{end.year}Q{end.quarter}")
                        self.assertEqual(bool(row["Is_Partial_Window"]), start < first_day)
                        if in_window.any():
                            self.assertAlmostEqual(row["Prevalence"], expected_count / int(in_window.sum()))
                        else:
                            self.assertTrue(pd.isna(row["Prevalence"]))

    def test_end_of_day_tickets_count_but_future_dates_do_not_leak(self):
        result = analyze_temporal_topics(boundary_tickets(), config=TemporalConfig(window_days=3, step_days=2))
        jan1 = rows_for_end(result["rolling"], "2024-01-01")
        self.assertEqual(set(jan1["Total_Tickets"]), {4})
        self.assertEqual(one_row(jan1, Cluster_ID=0)["Count"], 2)
        self.assertEqual(one_row(jan1, Cluster_ID=1)["Count"], 1)
        self.assertEqual(one_row(jan1, Cluster_ID=-1)["Count"], 1)
        jan3 = rows_for_end(result["rolling"], "2024-01-03")
        self.assertEqual(one_row(jan3, Cluster_ID=1)["Count"], 2)

    def test_rolling_deltas_compare_previous_sampled_endpoint(self):
        rolling = analyze_temporal_topics(boundary_tickets(), config=TemporalConfig(window_days=3, step_days=2))["rolling"]
        first = rows_for_end(rolling, "2023-12-30")
        self.assertTrue(first["Delta_K"].isna().all())
        middle = rows_for_end(rolling, "2024-01-01")
        final = rows_for_end(rolling, "2024-01-03")
        self.assertAlmostEqual(one_row(middle, Cluster_ID=0)["Delta_K"], -0.5)
        self.assertAlmostEqual(one_row(middle, Cluster_ID=1)["Delta_K"], 0.25)
        self.assertAlmostEqual(one_row(final, Cluster_ID=0)["Delta_K"], -0.25)
        self.assertAlmostEqual(one_row(final, Cluster_ID=1)["Delta_K"], 0.25)

    def test_empty_windows_have_nan_prevalence_and_do_not_bridge_baseline(self):
        frame = pd.DataFrame({"datetime": ["2024-01-01", "2024-01-05"], "hdbscan_cluster": [0, 1]})
        rolling = analyze_temporal_topics(frame, config=TemporalConfig(window_days=1))["rolling"]
        for end in ("2024-01-02", "2024-01-03", "2024-01-04"):
            group = rows_for_end(rolling, end)
            self.assertEqual(set(group["Count"]), {0})
            self.assertEqual(set(group["Total_Tickets"]), {0})
            self.assertTrue(group["Prevalence"].isna().all())
            self.assertTrue(group["Delta_K"].isna().all())
            self.assertTrue(group["Is_Low_Support"].all())
        after_gap = rows_for_end(rolling, "2024-01-05")
        self.assertTrue(after_gap["Delta_K"].isna().all())

    def test_minimum_support_is_reported_and_blocks_spurious_classification(self):
        rolling = analyze_temporal_topics(boundary_tickets(), config=TemporalConfig(window_days=3, step_days=2, min_window_tickets=5))["rolling"]
        self.assertTrue(rolling["Is_Low_Support"].all())
        for category in rolling["Category"]:
            self.assertFalse(any(signal in category.lower() for signal in ("emerging", "declining", "persistent")))
        self.assertTrue(rolling["Prevalence"].isna().all())
        self.assertTrue(rolling["Delta_K"].isna().all())
        supported = analyze_temporal_topics(boundary_tickets(), config=TemporalConfig(window_days=3, step_days=2, min_window_tickets=4))["rolling"]
        self.assertTrue(rows_for_end(supported, "2023-12-30")["Is_Low_Support"].all())
        self.assertFalse(rows_for_end(supported, "2024-01-01")["Is_Low_Support"].any())
        self.assertTrue(rows_for_end(supported, "2024-01-01")["Delta_K"].isna().all())

    def test_threshold_configuration_changes_surge_and_persistence_classification(self):
        frame = pd.DataFrame({
            "datetime": ["2024-01-01"] * 4 + ["2024-01-02"] * 4,
            "hdbscan_cluster": [0, -1, -1, -1, 0, 0, -1, -1],
        })
        low = analyze_temporal_topics(frame, config=TemporalConfig(window_days=1, surge_threshold=0.2, persistence_threshold=0.2))["rolling"]
        high = analyze_temporal_topics(frame, config=TemporalConfig(window_days=1, surge_threshold=0.3, persistence_threshold=0.2))["rolling"]
        stable = analyze_temporal_topics(frame, config=TemporalConfig(window_days=1, surge_threshold=0.3, persistence_threshold=0.6))["rolling"]
        self.assertIn("emerging", one_row(rows_for_end(low, "2024-01-02"), Cluster_ID=0)["Category"].lower())
        self.assertEqual(one_row(rows_for_end(high, "2024-01-02"), Cluster_ID=0)["Category"], "Persistent")
        self.assertEqual(one_row(rows_for_end(stable, "2024-01-02"), Cluster_ID=0)["Category"], "Stable")


class QuarterlyCoverageAndTrajectoryTests(unittest.TestCase):
    def test_missing_calendar_quarter_is_retained_without_fake_comparison(self):
        analysis = analyze_temporal_topics(gap_tickets(), config=TemporalConfig(step_days=40))
        quarterly = analysis["quarterly"]
        self.assertEqual(quarterly["Quarter"].drop_duplicates().tolist(), ["2023Q4", "2024Q1", "2024Q2"])
        missing = quarterly.loc[quarterly["Quarter"] == "2024Q1"]
        self.assertEqual(len(missing), 3)
        self.assertEqual(set(missing["Count"]), {0})
        self.assertEqual(set(missing["Total_Tickets"]), {0})
        self.assertTrue(missing["Prevalence"].isna().all())
        self.assertTrue(missing["Delta_K"].isna().all())
        self.assertTrue(missing["Pct_Change"].isna().all())
        later = quarterly.loc[quarterly["Quarter"] == "2024Q2"]
        self.assertTrue(later["Delta_K"].isna().all())
        self.assertTrue(later["Pct_Change"].isna().all())
        self.assertEqual(analysis["summary"]["missing_quarter_count"], 1)
        self.assertEqual(analysis["summary"]["observed_quarter_count"], 2)
        self.assertEqual(analysis["summary"]["quarter_counts"]["2024Q1"], 0)

    def test_partial_quarters_use_utc_calendar_coverage_boundaries(self):
        partial = analyze_temporal_topics(boundary_tickets())["quarterly"]
        self.assertTrue(partial["Is_Partial_Quarter"].all())
        full = analyze_temporal_topics(gap_tickets())["quarterly"]
        self.assertFalse(full.loc[full["Quarter"].isin(["2023Q4", "2024Q2"]), "Is_Partial_Quarter"].any())

    def test_offset_dates_are_converted_to_utc_before_quarter_and_day_bucketing(self):
        frame = pd.DataFrame({
            "datetime": ["2024-01-01T00:30:00+01:00", "2023-12-31T23:30:00-01:00"],
            "hdbscan_cluster": [0, -1],
        })
        analysis = analyze_temporal_topics(frame, config=TemporalConfig(window_days=1))
        self.assertEqual(one_row(analysis["quarterly"], Quarter="2023Q4", Cluster_ID=0)["Count"], 1)
        self.assertEqual(one_row(analysis["quarterly"], Quarter="2024Q1", Cluster_ID=-1)["Count"], 1)
        self.assertEqual(one_row(rows_for_end(analysis["rolling"], "2023-12-31"), Cluster_ID=0)["Count"], 1)
        self.assertEqual(one_row(rows_for_end(analysis["rolling"], "2024-01-01"), Cluster_ID=-1)["Count"], 1)

    def test_trajectory_uses_elapsed_quarters_and_ignores_missing_prevalence(self):
        trajectories = analyze_temporal_topics(gap_tickets(), LABELS)["trajectories"]
        old = one_row(trajectories, Cluster_ID=0)
        new = one_row(trajectories, Cluster_ID=1)
        for _, row in trajectories.iterrows():
            self.assertEqual(row["Observed_Quarters"], 2)
        self.assertEqual(old["Active_Quarters"], 1)
        self.assertEqual(old["First_Seen_Quarter"], "2023Q4")
        self.assertEqual(old["Last_Seen_Quarter"], "2023Q4")
        self.assertEqual(old["Peak_Quarter"], "2023Q4")
        self.assertAlmostEqual(old["Peak_Prevalence"], 0.5)
        self.assertAlmostEqual(old["Start_Prevalence"], 0.5)
        self.assertAlmostEqual(old["End_Prevalence"], 0.0)
        self.assertAlmostEqual(old["Net_Change"], -0.5)
        self.assertAlmostEqual(old["Mean_Prevalence"], 0.25)
        self.assertAlmostEqual(old["Prevalence_Slope_Per_Quarter"], -0.25)
        self.assertEqual(new["First_Seen_Quarter"], "2024Q2")
        self.assertEqual(new["Last_Seen_Quarter"], "2024Q2")
        self.assertAlmostEqual(new["Start_Prevalence"], 0.0)
        self.assertAlmostEqual(new["End_Prevalence"], 2 / 3)
        self.assertAlmostEqual(new["Mean_Prevalence"], 1 / 3)
        self.assertAlmostEqual(new["Prevalence_Slope_Per_Quarter"], 1 / 3)
        self.assertIn("declining", old["Trajectory"].lower())
        self.assertIn("emerging", new["Trajectory"].lower())

    def test_single_quarter_trajectory_does_not_claim_a_trend(self):
        frame = pd.DataFrame({"datetime": ["2024-04-05", "2024-04-06"], "hdbscan_cluster": [0, -1]})
        trajectories = analyze_temporal_topics(frame)["trajectories"]
        row = one_row(trajectories, Cluster_ID=0)
        self.assertEqual(row["Observed_Quarters"], 1)
        self.assertTrue(pd.isna(row["Prevalence_Slope_Per_Quarter"]))
        self.assertEqual(row["Trajectory"], "Insufficient History")

    def test_gap_prevents_a_persistent_trajectory(self):
        frame = pd.DataFrame({"datetime": ["2023-10-01", "2024-06-30"], "hdbscan_cluster": [0, 0]})
        row = one_row(analyze_temporal_topics(frame)["trajectories"], Cluster_ID=0)
        self.assertNotEqual(row["Trajectory"], "Persistent")

    def test_observed_zero_prevalence_prevents_a_persistent_trajectory(self):
        frame = pd.DataFrame({
            "datetime": ["2023-10-01", "2024-01-01", "2024-04-01"],
            "hdbscan_cluster": [0, -1, 0],
        })
        row = one_row(analyze_temporal_topics(frame)["trajectories"], Cluster_ID=0)
        self.assertEqual(row["Observed_Quarters"], 3)
        self.assertEqual(row["Active_Quarters"], 2)
        self.assertNotEqual(row["Trajectory"], "Persistent")

    def test_all_noise_is_a_valid_explicit_time_series(self):
        frame = pd.DataFrame({"datetime": ["2024-07-01", "2024-09-30T23:59:59Z"], "hdbscan_cluster": [-1, -1]})
        analysis = analyze_temporal_topics(frame)
        for name in ("quarterly", "rolling", "trajectories"):
            self.assertEqual(set(analysis[name]["Cluster_ID"]), {-1})
        quarterly = analysis["quarterly"].iloc[0]
        self.assertEqual(quarterly["Count"], 2)
        self.assertEqual(quarterly["Total_Tickets"], 2)
        self.assertEqual(quarterly["Prevalence"], 1.0)
        self.assertFalse(quarterly["Is_Partial_Quarter"])
        self.assertEqual(analysis["summary"]["total_valid_tickets"], 2)
        self.assertEqual(analysis["summary"]["topic_count"], 1)
        self.assertEqual(analysis["trajectories"].iloc[0]["Trajectory"], "Noise / Outliers")
        json.dumps(analysis["summary"], allow_nan=False)

    def test_noise_is_not_invented_when_no_noise_tickets_exist(self):
        frame = pd.DataFrame({"datetime": ["2024-01-01", "2024-01-02"], "hdbscan_cluster": [0, 1]})
        analysis = analyze_temporal_topics(frame)
        for name in ("quarterly", "rolling", "trajectories"):
            self.assertEqual(set(analysis[name]["Cluster_ID"]), {0, 1})


class TemporalValidationTests(unittest.TestCase):
    def test_empty_and_all_invalid_dates_are_rejected(self):
        frames = [
            pd.DataFrame(columns=["datetime", "hdbscan_cluster"]),
            pd.DataFrame({"datetime": [None, "not-a-date", "2024-02-30"], "hdbscan_cluster": [0, 1, -1]}),
        ]
        for frame in frames:
            with self.subTest(rows=len(frame)):
                with self.assertRaises(ValueError):
                    analyze_temporal_topics(frame)

    def test_missing_required_columns_are_rejected(self):
        for frame in [pd.DataFrame({"datetime": ["2024-01-01"]}), pd.DataFrame({"hdbscan_cluster": [0]})]:
            with self.subTest(columns=frame.columns.tolist()):
                with self.assertRaises((ValueError, KeyError)):
                    analyze_temporal_topics(frame)

    def test_invalid_cluster_identifiers_are_rejected(self):
        for value in [1.5, None, np.nan, np.inf, -np.inf, "topic-a", True, -2]:
            with self.subTest(cluster_id=repr(value)):
                frame = pd.DataFrame({"datetime": ["2024-01-01"], "hdbscan_cluster": [value]})
                with self.assertRaises((ValueError, TypeError)):
                    analyze_temporal_topics(frame)

    def test_bad_date_does_not_hide_invalid_cluster_identifier(self):
        frame = pd.DataFrame({"datetime": ["2024-01-01", "invalid"], "hdbscan_cluster": [0, 1.5]})
        with self.assertRaises((ValueError, TypeError)):
            analyze_temporal_topics(frame)

    def test_integral_numeric_cluster_representations_are_accepted(self):
        frame = pd.DataFrame({"datetime": ["2024-01-01"] * 3, "hdbscan_cluster": ["0", 1.0, "-1"]})
        quarterly = analyze_temporal_topics(frame)["quarterly"]
        self.assertEqual(set(quarterly["Cluster_ID"]), {-1, 0, 1})
        self.assertEqual(set(quarterly["Count"]), {1})
        self.assertEqual(set(quarterly["Total_Tickets"]), {3})

    def test_config_rejects_nonpositive_or_noninteger_day_and_support_counts(self):
        frame = boundary_tickets()
        for field in ("window_days", "step_days", "min_window_tickets"):
            for value in (0, -1, 1.5, True):
                with self.subTest(field=field, value=value):
                    with self.assertRaises((ValueError, TypeError)):
                        config = TemporalConfig(**{field: value})
                        analyze_temporal_topics(frame, config=config)

    def test_config_rejects_nonfinite_or_negative_thresholds(self):
        for field in ("surge_threshold", "persistence_threshold"):
            for value in (-0.1, np.nan, np.inf, -np.inf):
                with self.subTest(field=field, value=value):
                    with self.assertRaises((ValueError, TypeError)):
                        config = TemporalConfig(**{field: value})
                        analyze_temporal_topics(boundary_tickets(), config=config)


class TemporalExportAndCliTests(unittest.TestCase):
    def setUp(self):
        self.analysis = analyze_temporal_topics(boundary_tickets(), LABELS, TemporalConfig(window_days=3, step_days=2))

    def test_export_writes_named_csvs_and_strict_json_without_plots(self):
        expected = {
            "rolling_csv": ("rolling_topic_windows.csv", "rolling"),
            "quarterly_csv": ("quarterly_topic_prevalence.csv", "quarterly"),
            "trajectories_csv": ("topic_trajectories.csv", "trajectories"),
        }
        originals = {name: self.analysis[name].copy(deep=True) for name in ("quarterly", "rolling", "trajectories")}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "temporal_results"
            paths = export_temporal_analysis(self.analysis, output, make_plots=False)
            self.assertEqual(set(paths), {*expected, "summary_json"})
            for key, (filename, table) in expected.items():
                with self.subTest(export=key):
                    path = Path(paths[key])
                    self.assertEqual(path.resolve(), (output / filename).resolve())
                    self.assertTrue(path.is_file())
                    csv = pd.read_csv(path)
                    self.assertEqual(csv.columns.tolist(), self.analysis[table].columns.tolist())
                    self.assertEqual(len(csv), len(self.analysis[table]))
                    self.assertFalse(any(column.startswith("Unnamed:") for column in csv.columns))
            summary_path = Path(paths["summary_json"])
            self.assertEqual(summary_path.name, "temporal_tracking_summary.json")
            with summary_path.open(encoding="utf-8") as handle:
                def reject_constant(value):
                    raise AssertionError(f"Non-standard JSON number {value}")
                summary = json.load(handle, parse_constant=reject_constant)
            self.assertEqual(summary, self.analysis["summary"])
            self.assertEqual({path.name for path in output.iterdir()}, {item[0] for item in expected.values()} | {"temporal_tracking_summary.json"})
        for name, original in originals.items():
            assert_frame_equal(self.analysis[name], original)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "optional matplotlib dependency is not available")
    def test_optional_plot_export_returns_three_nonempty_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = export_temporal_analysis(self.analysis, temporary, make_plots=True, top_n=2)
            plot_keys = {"rolling_plot", "quarterly_plot", "rolling_by_quarter_plot"}
            self.assertTrue(plot_keys <= set(paths))
            for key in plot_keys:
                with self.subTest(plot=key):
                    path = Path(paths[key])
                    self.assertTrue(path.is_file())
                    self.assertGreater(path.stat().st_size, 0)

    def test_cli_custom_columns_and_no_plots_match_library(self):
        frame = boundary_tickets().rename(columns={"datetime": "occurred_at", "hdbscan_cluster": "topic"})
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "tickets with spaces.csv"
            output = Path(temporary) / "results with spaces"
            labels_path = Path(temporary) / "topic labels.json"
            labels_path.write_text(json.dumps(LABELS), encoding="utf-8")
            frame.to_csv(source, index=False)
            completed = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "temporal_tracking.py"),
                 "--input", str(source), "--output-dir", str(output),
                 "--window-days", "3", "--step-days", "2", "--no-plots",
                 "--date-column", "occurred_at", "--topic-column", "topic",
                 "--topic-labels", str(labels_path)],
                cwd=temporary, capture_output=True, text=True, timeout=60,
                env={**os.environ, "MPLBACKEND": "Agg"},
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            for filename in ("rolling_topic_windows.csv", "quarterly_topic_prevalence.csv", "topic_trajectories.csv", "temporal_tracking_summary.json"):
                self.assertTrue((output / filename).is_file(), filename)
            with (output / "temporal_tracking_summary.json").open(encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["total_valid_tickets"], 6)
            self.assertEqual(summary["dropped_invalid_dates"], 1)
            self.assertEqual(summary["config"]["window_days"], 3)
            self.assertEqual(summary["config"]["step_days"], 2)
            actual = pd.read_csv(output / "quarterly_topic_prevalence.csv")
            expected = self.analysis["quarterly"]
            for field in ("Quarter", "Cluster_ID", "Topic_Label", "Count", "Total_Tickets", "Prevalence"):
                self.assertEqual(actual[field].tolist(), expected[field].tolist())
            self.assertFalse(list(output.glob("*.png")))

    def test_cli_invalid_window_returns_nonzero_without_success_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "tickets.csv"
            output = Path(temporary) / "results"
            boundary_tickets().to_csv(source, index=False)
            completed = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "temporal_tracking.py"),
                 "--input", str(source), "--output-dir", str(output),
                 "--window-days", "0", "--no-plots"],
                cwd=temporary, capture_output=True, text=True, timeout=60,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse((output / "temporal_tracking_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
