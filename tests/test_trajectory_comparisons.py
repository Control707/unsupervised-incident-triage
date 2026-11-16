"""Both quarter-comparison views must subtract shares in a fixed topic space.

Overall-baseline excess and day-aligned trajectory differences answer distinct
questions.  These tests independently verify their denominators, percentage-
point units, eligibility masks, and exact calendar-day alignment.
"""

from __future__ import annotations

import hashlib
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from temporal_tracking import TemporalConfig, analyze_temporal_topics  # noqa: E402
from trajectory_comparisons import (  # noqa: E402
    compare_quarter_trajectories,
    export_trajectory_comparisons,
)


LABELS = {0: "Authentication", 1: "Networking"}
EXCESS_COLUMNS = {
    "Baseline_Quarter", "Target_Quarter", "Cluster_ID", "Topic_Label",
    "Window_Start", "Window_End", "Day_Of_Quarter", "Count", "Total_Tickets",
    "Target_Prevalence", "Baseline_Count", "Baseline_Total_Tickets",
    "Baseline_Prevalence", "Excess_Prevalence", "Excess_PP", "Is_Comparable",
}
ALIGNED_COLUMNS = {
    "Baseline_Quarter", "Target_Quarter", "Cluster_ID", "Topic_Label",
    "Day_Of_Quarter", "Baseline_Window_End", "Target_Window_End",
    "Baseline_Count", "Baseline_Total_Tickets", "Target_Count",
    "Target_Total_Tickets", "Baseline_Prevalence", "Target_Prevalence",
    "Difference", "Difference_PP", "Is_Comparable", "Exclusion_Reason",
}
TOPIC_COLUMNS = {
    "Cluster_ID", "Topic_Label", "Baseline_Count", "Baseline_Total_Tickets",
    "Target_Count", "Target_Total_Tickets", "Baseline_Prevalence", "Target_Prevalence",
}
BASE_FILES = {
    "baseline_excess_csv": "baseline_excess_{target}_vs_{baseline}.csv",
    "aligned_difference_csv": "aligned_difference_{target}_vs_{baseline}.csv",
    "comparison_topics_csv": "comparison_topics_{target}_vs_{baseline}.csv",
    "comparison_summary_json": "comparison_summary_{target}_vs_{baseline}.json",
}
PLOT_FILES = {
    "baseline_excess_plot": "baseline_excess_{target}_vs_{baseline}.png",
    "baseline_excess_pdf": "baseline_excess_{target}_vs_{baseline}.pdf",
    "aligned_difference_plot": "aligned_difference_{target}_vs_{baseline}.png",
    "aligned_difference_pdf": "aligned_difference_{target}_vs_{baseline}.pdf",
}


def dense_tickets() -> pd.DataFrame:
    """Four tickets daily Jan-Jun 2021, changing mix and explicit daily noise."""
    rows = []
    for date in pd.date_range("2021-01-01", "2021-06-30"):
        if date.quarter == 1:
            topics = [0, 0, 1, -1]
        elif date.day <= 15:
            topics = [0, 1, 1, -1]
        else:
            topics = [1, 1, -1, -1]
        for offset, topic in enumerate(topics):
            rows.append((date + pd.Timedelta(hours=23, minutes=59, seconds=59, microseconds=offset), topic))
    frame = pd.DataFrame(rows, columns=["datetime", "hdbscan_cluster"])
    frame = frame.iloc[::-1].copy()
    frame.index = [index // 3 for index in range(len(frame))]
    return frame


def row_for(table: pd.DataFrame, **matches) -> pd.Series:
    selected = table
    for column, value in matches.items():
        selected = selected.loc[selected[column] == value]
    if len(selected) != 1:
        raise AssertionError(f"Expected one row matching {matches!r}, got {len(selected)}")
    return selected.iloc[0]


def day(value) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True).normalize()


def strict_load(path: Path):
    def reject_constant(value):
        raise AssertionError(f"Non-standard JSON constant: {value}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant)


class TrajectoryComparisonArithmeticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = dense_tickets()
        cls.analysis = analyze_temporal_topics(cls.frame, LABELS, TemporalConfig(window_days=14))
        cls.comparison = compare_quarter_trajectories(cls.analysis)

    def test_result_schemas_and_all_target_rows_preserved(self):
        self.assertTrue({"baseline_excess", "aligned_difference", "topic_summary", "summary"} <= set(self.comparison))
        for name, required in [("baseline_excess", EXCESS_COLUMNS), ("aligned_difference", ALIGNED_COLUMNS), ("topic_summary", TOPIC_COLUMNS)]:
            with self.subTest(table=name):
                table = self.comparison[name]
                self.assertIsInstance(table, pd.DataFrame)
                self.assertTrue(required <= set(table.columns), required - set(table.columns))
                self.assertEqual(set(table["Cluster_ID"]), {-1, 0, 1})
        for name in ("baseline_excess", "aligned_difference"):
            table = self.comparison[name]
            self.assertEqual(len(table), 91 * 3)
            self.assertFalse(table.duplicated(["Day_Of_Quarter", "Cluster_ID"]).any())
            self.assertEqual(set(table["Day_Of_Quarter"]), set(range(1, 92)))
            self.assertEqual(set(table["Baseline_Quarter"]), {"2021Q1"})
            self.assertEqual(set(table["Target_Quarter"]), {"2021Q2"})

    def test_baseline_excess_uses_overall_baseline_quarter_all_ticket_share(self):
        excess = self.comparison["baseline_excess"]
        baseline = self.frame.loc[self.frame["datetime"].dt.quarter == 1]
        target = self.frame.loc[self.frame["datetime"].dt.quarter == 2]
        for topic in (-1, 0, 1):
            expected_count = int((baseline["hdbscan_cluster"] == topic).sum())
            prevalence = expected_count / len(baseline)
            series = excess.loc[excess["Cluster_ID"] == topic]
            self.assertEqual(set(series["Baseline_Count"]), {expected_count})
            self.assertEqual(set(series["Baseline_Total_Tickets"]), {len(baseline)})
            np.testing.assert_allclose(series["Baseline_Prevalence"], prevalence)
            for _, row in series.iterrows():
                start = pd.to_datetime(row["Window_Start"])
                end_exclusive = pd.to_datetime(row["Window_End"]) + pd.Timedelta(days=1)
                raw_window = self.frame.loc[(self.frame["datetime"] >= start) & (self.frame["datetime"] < end_exclusive)]
                expected_window_count = int((raw_window["hdbscan_cluster"] == topic).sum())
                expected_share = expected_window_count / len(raw_window)
                self.assertEqual(row["Count"], expected_window_count)
                self.assertEqual(row["Total_Tickets"], len(raw_window))
                self.assertAlmostEqual(row["Target_Prevalence"], expected_share)
                self.assertTrue(row["Is_Comparable"])
                self.assertAlmostEqual(row["Excess_Prevalence"], expected_share - prevalence)
                self.assertAlmostEqual(row["Excess_PP"], 100 * (expected_share - prevalence))
        self.assertEqual(len(target), 364)
        self.assertEqual(set(excess["Baseline_Total_Tickets"]), {360})

    def test_exact_day_alignment_excludes_partial_start_and_unmatched_target_day91(self):
        aligned = self.comparison["aligned_difference"]
        comparable = aligned.loc[aligned["Is_Comparable"]]
        self.assertEqual(set(comparable["Day_Of_Quarter"]), set(range(14, 91)))
        self.assertEqual(comparable["Day_Of_Quarter"].nunique(), 77)
        startup = aligned.loc[aligned["Day_Of_Quarter"] <= 13]
        self.assertFalse(startup["Is_Comparable"].any())
        self.assertEqual(set(startup["Exclusion_Reason"]), {"Partial Window"})
        self.assertTrue(startup["Difference"].isna().all())
        self.assertTrue(startup["Difference_PP"].isna().all())
        last = aligned.loc[aligned["Day_Of_Quarter"] == 91]
        self.assertEqual(set(last["Exclusion_Reason"]), {"No Baseline Endpoint"})
        self.assertFalse(last["Is_Comparable"].any())
        self.assertTrue(last["Baseline_Window_End"].isna().all())
        self.assertTrue(last["Difference"].isna().all())
        self.assertTrue(last["Difference_PP"].isna().all())
        for _, row in comparable.iterrows():
            ordinal = int(row["Day_Of_Quarter"])
            self.assertEqual(day(row["Baseline_Window_End"]), day("2021-01-01") + pd.Timedelta(days=ordinal - 1))
            self.assertEqual(day(row["Target_Window_End"]), day("2021-04-01") + pd.Timedelta(days=ordinal - 1))

    def test_aligned_raw_denominators_and_percentage_point_not_percent_change(self):
        aligned = self.comparison["aligned_difference"]
        for _, row in aligned.loc[aligned["Is_Comparable"]].iterrows():
            topic = row["Cluster_ID"]
            shares = []
            for prefix in ("Baseline", "Target"):
                end = pd.to_datetime(row[f"{prefix}_Window_End"])
                start = end - pd.Timedelta(days=13)
                raw = self.frame.loc[(self.frame["datetime"] >= start) & (self.frame["datetime"] < end + pd.Timedelta(days=1))]
                count = int((raw["hdbscan_cluster"] == topic).sum())
                self.assertEqual(row[f"{prefix}_Count"], count)
                self.assertEqual(row[f"{prefix}_Total_Tickets"], len(raw))
                shares.append(count / len(raw))
                self.assertAlmostEqual(row[f"{prefix}_Prevalence"], shares[-1])
            expected = shares[1] - shares[0]
            self.assertAlmostEqual(row["Difference"], expected)
            self.assertAlmostEqual(row["Difference_PP"], 100 * expected)
        example = row_for(aligned, Day_Of_Quarter=15, Cluster_ID=0)
        self.assertAlmostEqual(example["Difference"], -0.25)
        self.assertAlmostEqual(example["Difference_PP"], -25.0)  # Not -50% relative change.

    def test_noise_denominators_add_to_one_and_subtracted_shares_add_to_zero(self):
        for table_name, share, difference in [
            ("baseline_excess", "Target_Prevalence", "Excess_Prevalence"),
            ("aligned_difference", "Target_Prevalence", "Difference"),
        ]:
            table = self.comparison[table_name]
            for _, group in table.loc[table["Is_Comparable"]].groupby("Day_Of_Quarter"):
                self.assertAlmostEqual(group[share].sum(), 1.0)
                self.assertAlmostEqual(group["Baseline_Prevalence"].sum(), 1.0)
                self.assertAlmostEqual(group[difference].sum(), 0.0)
                self.assertEqual(set(group["Cluster_ID"]), {-1, 0, 1})

    def test_topic_summary_uses_true_quarter_counts_not_sum_of_rolling_windows(self):
        topics = self.comparison["topic_summary"]
        baseline = self.frame.loc[self.frame["datetime"].dt.quarter == 1]
        target = self.frame.loc[self.frame["datetime"].dt.quarter == 2]
        for topic in (-1, 0, 1):
            row = row_for(topics, Cluster_ID=topic)
            baseline_count = int((baseline["hdbscan_cluster"] == topic).sum())
            target_count = int((target["hdbscan_cluster"] == topic).sum())
            self.assertEqual(row["Baseline_Count"], baseline_count)
            self.assertEqual(row["Target_Count"], target_count)
            self.assertEqual(row["Baseline_Total_Tickets"], len(baseline))
            self.assertEqual(row["Target_Total_Tickets"], len(target))
            self.assertAlmostEqual(row["Baseline_Prevalence"], baseline_count / len(baseline))
            self.assertAlmostEqual(row["Target_Prevalence"], target_count / len(target))
            self.assertAlmostEqual(row["Quarterly_Delta_PP"], 100 * (target_count / len(target) - baseline_count / len(baseline)))
            excess = self.comparison["baseline_excess"]
            excess = excess.loc[(excess["Cluster_ID"] == topic) & excess["Is_Comparable"]].sort_values("Window_End")
            aligned = self.comparison["aligned_difference"]
            aligned = aligned.loc[(aligned["Cluster_ID"] == topic) & aligned["Is_Comparable"]].sort_values("Target_Window_End")
            self.assertEqual(row["Baseline_Comparable_Endpoints"], len(excess))
            self.assertEqual(row["Aligned_Comparable_Pairs"], len(aligned))
            self.assertAlmostEqual(row["Mean_Excess_PP"], excess["Excess_PP"].mean())
            self.assertAlmostEqual(row["Peak_Excess_PP"], excess["Excess_PP"].max())
            self.assertAlmostEqual(row["Above_Baseline_Fraction"], (excess["Excess_PP"] > 0).mean())
            self.assertAlmostEqual(row["Mean_Aligned_Difference_PP"], aligned["Difference_PP"].mean())
            self.assertAlmostEqual(row["Peak_Aligned_Difference_PP"], aligned["Difference_PP"].max())
            excess_peak = excess.loc[excess["Excess_PP"].idxmax()]
            aligned_peak = aligned.loc[aligned["Difference_PP"].idxmax()]
            self.assertEqual(day(row["Peak_Excess_Date"]), day(excess_peak["Window_End"]))
            self.assertEqual(day(row["Peak_Aligned_Target_Date"]), day(aligned_peak["Target_Window_End"]))
        self.assertEqual(topics["Baseline_Count"].sum(), 360)
        self.assertEqual(topics["Target_Count"].sum(), 364)
        self.assertFalse(any("p_value" in column.lower() or "significan" in column.lower() for column in topics.columns))

    def test_summary_counts_unique_endpoints_not_number_of_topics_and_is_json_safe(self):
        summary = self.comparison["summary"]
        self.assertEqual(summary["baseline_quarter"], "2021Q1")
        self.assertEqual(summary["target_quarter"], "2021Q2")
        self.assertEqual(summary["baseline_ticket_count"], 360)
        self.assertEqual(summary["target_ticket_count"], 364)
        self.assertEqual(summary["baseline_excess_endpoint_count"], 91)
        self.assertEqual(summary["baseline_excess_comparable_endpoint_count"], 91)
        self.assertEqual(summary["aligned_endpoint_count"], 91)
        self.assertEqual(summary["aligned_comparable_pair_count"], 77)
        self.assertEqual(summary["aligned_exclusion_counts"]["Partial Window"], 13)
        self.assertEqual(summary["aligned_exclusion_counts"]["No Baseline Endpoint"], 1)
        self.assertEqual(summary["config"], self.analysis["summary"]["config"])
        self.assertTrue(summary["alignment_policy"])
        self.assertTrue(summary["caveats"])
        self.assertEqual(json.loads(json.dumps(summary, allow_nan=False)), summary)

    def test_original_analysis_and_input_are_not_mutated(self):
        original_frame = self.frame.copy(deep=True)
        originals = {
            key: value.copy(deep=True) if isinstance(value, pd.DataFrame) else json.loads(json.dumps(value, allow_nan=False))
            for key, value in self.analysis.items()
        }
        compare_quarter_trajectories(self.analysis)
        assert_frame_equal(self.frame, original_frame)
        for key, original in originals.items():
            if isinstance(original, pd.DataFrame):
                assert_frame_equal(self.analysis[key], original)
            else:
                self.assertEqual(self.analysis[key], original)


class TrajectoryComparisonEligibilityTests(unittest.TestCase):
    def test_sparse_steps_must_not_nearest_match_or_stretch_quarter_fractions(self):
        analysis = analyze_temporal_topics(dense_tickets(), config=TemporalConfig(window_days=14, step_days=7))
        result = compare_quarter_trajectories(analysis)
        target = analysis["rolling"].loc[analysis["rolling"]["Quarter"] == "2021Q2"]
        self.assertEqual(len(result["aligned_difference"]), len(target))
        self.assertFalse(result["aligned_difference"]["Is_Comparable"].any())
        self.assertEqual(set(result["aligned_difference"]["Exclusion_Reason"]), {"No Baseline Endpoint"})
        self.assertTrue(result["aligned_difference"]["Difference"].isna().all())
        self.assertTrue(result["baseline_excess"]["Is_Comparable"].all())

    def test_partial_target_start_masks_excess_and_aligned_difference(self):
        frame = pd.DataFrame({"datetime": ["2021-03-31", "2021-04-01", "2021-04-03"], "hdbscan_cluster": [0, 1, -1]})
        result = compare_quarter_trajectories(analyze_temporal_topics(frame, config=TemporalConfig(window_days=3)))
        first = result["baseline_excess"].loc[result["baseline_excess"]["Day_Of_Quarter"] == 1]
        self.assertFalse(first["Is_Comparable"].any())
        self.assertTrue(first["Excess_Prevalence"].isna().all())
        self.assertTrue(first["Excess_PP"].isna().all())
        self.assertEqual(set(first["Exclusion_Reason"]), {"Partial Window"})
        aligned = result["aligned_difference"]
        self.assertEqual(set(aligned["Exclusion_Reason"]), {"No Baseline Endpoint"})

    def test_target_low_support_masks_both_views(self):
        frame = dense_tickets()
        dates = frame["datetime"].dt.normalize()
        frame = frame.loc[(dates != pd.Timestamp("2021-04-20")) | (frame["hdbscan_cluster"] == -1)]
        result = compare_quarter_trajectories(analyze_temporal_topics(frame, config=TemporalConfig(window_days=1, min_window_tickets=3)))
        excess = result["baseline_excess"].loc[result["baseline_excess"]["Day_Of_Quarter"] == 20]
        aligned = result["aligned_difference"].loc[result["aligned_difference"]["Day_Of_Quarter"] == 20]
        for table, diff in [(excess, "Excess_Prevalence"), (aligned, "Difference")]:
            self.assertEqual(set(table["Exclusion_Reason"]), {"Low Support"})
            self.assertFalse(table["Is_Comparable"].any())
            self.assertTrue(table[diff].isna().all())

    def test_partial_exclusion_takes_precedence_over_low_support(self):
        frame = dense_tickets()
        result = compare_quarter_trajectories(analyze_temporal_topics(frame, config=TemporalConfig(window_days=14, min_window_tickets=100)))
        aligned = result["aligned_difference"]
        self.assertEqual(set(aligned.loc[aligned["Day_Of_Quarter"] <= 13, "Exclusion_Reason"]), {"Partial Window"})
        self.assertEqual(set(aligned.loc[aligned["Day_Of_Quarter"] == 14, "Exclusion_Reason"]), {"Low Support"})
        self.assertEqual(set(aligned.loc[aligned["Day_Of_Quarter"] == 91, "Exclusion_Reason"]), {"No Baseline Endpoint"})
        self.assertFalse(aligned["Is_Comparable"].any())
        self.assertTrue(aligned["Difference"].isna().all())

    def test_baseline_low_support_excludes_only_aligned_option(self):
        frame = dense_tickets()
        dates = frame["datetime"].dt.normalize()
        frame = frame.loc[(dates != pd.Timestamp("2021-01-20")) | (frame["hdbscan_cluster"] == -1)]
        result = compare_quarter_trajectories(analyze_temporal_topics(frame, config=TemporalConfig(window_days=1, min_window_tickets=3)))
        excess = result["baseline_excess"].loc[result["baseline_excess"]["Day_Of_Quarter"] == 20]
        aligned = result["aligned_difference"].loc[result["aligned_difference"]["Day_Of_Quarter"] == 20]
        self.assertTrue(excess["Is_Comparable"].all())
        self.assertEqual(set(aligned["Exclusion_Reason"]), {"Low Support"})
        self.assertTrue(aligned["Difference"].isna().all())
        self.assertFalse(aligned["Is_Comparable"].any())

    def test_no_baseline_sampled_endpoint_keeps_target_rows_with_missing_pairs(self):
        frame = pd.DataFrame({"datetime": ["2020-12-31", "2021-01-01", "2021-04-10", "2021-07-01"], "hdbscan_cluster": [0, 0, 1, -1]})
        analysis = analyze_temporal_topics(frame, config=TemporalConfig(window_days=1, step_days=100))
        # Global sampling visits Dec31, Apr10 and the final Jul1: Q1 contains
        # original tickets but no endpoint at all, so pairs must remain absent.
        self.assertFalse((analysis["rolling"]["Quarter"] == "2021Q1").any())
        result = compare_quarter_trajectories(analysis)
        self.assertFalse(result["aligned_difference"]["Is_Comparable"].any())
        self.assertEqual(set(result["aligned_difference"]["Exclusion_Reason"]), {"No Baseline Endpoint"})

    def test_year_boundary_quarters_and_utc_day_numbering_are_supported(self):
        frame = pd.DataFrame({
            "datetime": ["2021-10-01", "2021-12-31T23:30:00-01:00", "2022-01-02", "2022-03-31"],
            "hdbscan_cluster": [0, 1, -1, 1],
        })
        result = compare_quarter_trajectories(analyze_temporal_topics(frame, config=TemporalConfig(window_days=1)), "2021Q4", "2022Q1")
        first = row_for(result["baseline_excess"], Cluster_ID=1, Day_Of_Quarter=1)
        self.assertEqual(day(first["Window_End"]), day("2022-01-01"))
        self.assertEqual(first["Count"], 1)
        self.assertEqual(first["Baseline_Count"], 0)
        self.assertAlmostEqual(first["Excess_PP"], 100.0)
        self.assertEqual(result["summary"]["baseline_ticket_count"], 1)
        self.assertEqual(result["summary"]["target_ticket_count"], 3)

    def test_all_noise_is_a_valid_zero_difference_comparison(self):
        frame = dense_tickets()
        frame["hdbscan_cluster"] = -1
        result = compare_quarter_trajectories(analyze_temporal_topics(frame))
        for name, diff in [("baseline_excess", "Excess_Prevalence"), ("aligned_difference", "Difference")]:
            table = result[name]
            self.assertEqual(set(table["Cluster_ID"]), {-1})
            self.assertTrue((table.loc[table["Is_Comparable"], diff] == 0).all())
        json.dumps(result["summary"], allow_nan=False)

    def test_quarters_must_be_valid_distinct_earlier_and_populated(self):
        analysis = analyze_temporal_topics(dense_tickets())
        for baseline, target in [("2021Q1", "2021Q1"), ("2021Q2", "2021Q1"), ("Q1", "2021Q2"), ("2021Q1", "2021Q5"), ("2020Q4", "2021Q2"), ("2021Q1", "2021Q3")]:
            with self.subTest(baseline=baseline, target=target):
                with self.assertRaises((ValueError, TypeError)):
                    compare_quarter_trajectories(analysis, baseline, target)
        gap = analyze_temporal_topics(pd.DataFrame({"datetime": ["2021-01-01", "2021-07-01"], "hdbscan_cluster": [0, 1]}))
        with self.assertRaises(ValueError):
            compare_quarter_trajectories(gap, "2021Q1", "2021Q2")
        with self.assertRaises(ValueError):
            compare_quarter_trajectories(gap, "2021Q2", "2021Q3")


class TrajectoryComparisonExportAndCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comparison = compare_quarter_trajectories(analyze_temporal_topics(dense_tickets(), LABELS))

    def test_export_writes_all_topics_and_strict_json_without_plots_or_mutation(self):
        originals = {name: self.comparison[name].copy(deep=True) for name in ("baseline_excess", "aligned_difference", "topic_summary")}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "comparisons"
            paths = export_trajectory_comparisons(self.comparison, output, make_plots=False, top_n=1)
            self.assertEqual(set(paths), set(BASE_FILES))
            tables = {"baseline_excess_csv": "baseline_excess", "aligned_difference_csv": "aligned_difference", "comparison_topics_csv": "topic_summary"}
            for key, template in BASE_FILES.items():
                path = Path(paths[key])
                self.assertEqual(path.resolve(), (output / template.format(target="2021Q2", baseline="2021Q1")).resolve())
                self.assertTrue(path.is_file())
                if key in tables:
                    exported = pd.read_csv(path)
                    self.assertEqual(exported.columns.tolist(), self.comparison[tables[key]].columns.tolist())
                    self.assertEqual(len(exported), len(self.comparison[tables[key]]))
                    self.assertEqual(set(exported["Cluster_ID"]), {-1, 0, 1})
                    self.assertFalse(any(column.startswith("Unnamed:") for column in exported.columns))
            self.assertEqual(strict_load(Path(paths["comparison_summary_json"])), self.comparison["summary"])
            self.assertEqual({path.name for path in output.iterdir()}, {template.format(target="2021Q2", baseline="2021Q1") for template in BASE_FILES.values()})
        for name, original in originals.items():
            assert_frame_equal(self.comparison[name], original)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "optional matplotlib dependency is not available")
    def test_optional_plots_export_two_pngs_and_two_pdfs(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = export_trajectory_comparisons(self.comparison, temporary, make_plots=True, top_n=2)
            self.assertEqual(set(paths), set(BASE_FILES) | set(PLOT_FILES))
            for key, template in PLOT_FILES.items():
                path = Path(paths[key])
                self.assertEqual(path.name, template.format(target="2021Q2", baseline="2021Q1"))
                self.assertGreater(path.stat().st_size, 0)
                signature = path.read_bytes()[:8]
                self.assertTrue(signature.startswith(b"%PDF-") if path.suffix == ".pdf" else signature == b"\x89PNG\r\n\x1a\n")

    def test_cli_reuses_exact_assignments_and_records_input_hashes(self):
        frame = dense_tickets()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "assigned tickets.csv"
            labels = Path(temporary) / "topic labels.json"
            output = Path(temporary) / "comparison results"
            frame.to_csv(source, index=False)
            labels.write_text(json.dumps(LABELS), encoding="utf-8")
            source_bytes, labels_bytes = source.read_bytes(), labels.read_bytes()
            completed = subprocess.run(
                [sys.executable, str(ROOT / "trajectory_comparisons.py"),
                 "--input", str(source), "--topic-labels", str(labels),
                 "--baseline-quarter", "2021Q1", "--target-quarter", "2021Q2",
                 "--window-days", "14", "--step-days", "1", "--min-window-tickets", "1",
                 "--output-dir", str(output), "--no-plots"],
                cwd=temporary, capture_output=True, text=True, timeout=60,
                env={**os.environ, "MPLBACKEND": "Agg"},
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertEqual(labels.read_bytes(), labels_bytes)
            for template in BASE_FILES.values():
                self.assertTrue((output / template.format(target="2021Q2", baseline="2021Q1")).is_file())
            summary = strict_load(output / "comparison_summary_2021Q2_vs_2021Q1.json")
            self.assertFalse(summary["refit_performed"])
            self.assertEqual(summary["aligned_comparable_pair_count"], 77)
            self.assertEqual(summary["config"]["window_days"], 14)
            self.assertEqual(summary["config"]["min_window_tickets"], 1)
            for key, data in [("assignment_provenance", source_bytes), ("topic_labels_provenance", labels_bytes)]:
                self.assertEqual(summary[key]["sha256"], hashlib.sha256(data).hexdigest())
                self.assertEqual(summary[key]["bytes"], len(data))
            self.assertFalse(summary["assignment_provenance"]["refit_performed"])
            exported = pd.read_csv(output / "aligned_difference_2021Q2_vs_2021Q1.csv")
            self.assertEqual(row_for(exported, Day_Of_Quarter=15, Cluster_ID=0)["Topic_Label"], LABELS[0])
            self.assertAlmostEqual(row_for(exported, Day_Of_Quarter=15, Cluster_ID=0)["Difference_PP"], -25.0)
            self.assertFalse(list(output.glob("*.png")))
            self.assertFalse(list(output.glob("*.pdf")))

    def test_export_names_follow_nondefault_baseline_and_target(self):
        frame = pd.DataFrame({"datetime": ["2021-10-01", "2022-01-01", "2022-03-31"], "hdbscan_cluster": [0, -1, 1]})
        comparison = compare_quarter_trajectories(analyze_temporal_topics(frame), "2021Q4", "2022Q1")
        with tempfile.TemporaryDirectory() as temporary:
            paths = export_trajectory_comparisons(comparison, temporary, make_plots=False)
            for key, template in BASE_FILES.items():
                self.assertEqual(Path(paths[key]).name, template.format(target="2022Q1", baseline="2021Q4"))

    def test_cli_rejects_reversed_quarters(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "assigned.csv"
            dense_tickets().to_csv(source, index=False)
            completed = subprocess.run(
                [sys.executable, str(ROOT / "trajectory_comparisons.py"),
                 "--input", str(source), "--baseline-quarter", "2021Q2", "--target-quarter", "2021Q1",
                 "--output-dir", str(Path(temporary) / "results"), "--no-plots"],
                cwd=temporary, capture_output=True, text=True, timeout=60,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertTrue(completed.stderr.strip())


if __name__ == "__main__":
    unittest.main()
