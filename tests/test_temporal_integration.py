"""Smoke tests for the notebook and benchmark integration, without NLP downloads."""

import ast
import contextlib
import io
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from temporal_tracking import (
    TemporalConfig, analyze_temporal_topics, export_temporal_analysis,
    focus_topic_quarter, export_focus_quarter,
)


def notebook_cells():
    notebook = json.loads((ROOT / 'NLP_Final_Experiments_180d.ipynb').read_text())
    return [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']


class TemporalIntegrationTests(unittest.TestCase):
    def test_all_notebook_cells_and_benchmark_parse(self):
        ast.parse((ROOT / 'experiment_quarterly_with_lda.py').read_text())
        for index, source in enumerate(notebook_cells()):
            with self.subTest(cell=index):
                ast.parse('\n'.join(line for line in source.splitlines() if not line.startswith('!')))

    def test_notebook_acquisition_is_year_aware_and_has_exclusive_end(self):
        source = next(cell for cell in notebook_cells() if cell.startswith('# 2.'))
        source = source.replace("ANALYSIS_START = '2021-01-01'", "ANALYSIS_START = '2021-10-01'")
        source = source.replace("ANALYSIS_END_EXCLUSIVE = '2021-07-01'", "ANALYSIS_END_EXCLUSIVE = '2022-04-01'")
        dates = pd.to_datetime(['2021-09-30', '2021-10-01', '2021-12-31', '2022-01-01', '2022-03-31', '2022-04-01'], utc=True)
        raw = pd.DataFrame({
            'created_at': [int(date.timestamp() * 1000) for date in dates],
            'title': ['synthetic title'] * 6, 'body': ['synthetic body'] * 6,
        })
        proxy = SimpleNamespace(read_parquet=lambda _: raw.copy(), to_datetime=pd.to_datetime)
        namespace = {'pd': proxy, 'os': SimpleNamespace(path=SimpleNamespace(exists=lambda _: False))}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, 'notebook-acquisition', 'exec'), namespace)
        self.assertEqual(namespace['df']['quarter'].tolist(), ['2021Q4', '2021Q4', '2022Q1', '2022Q1'])
        self.assertEqual(namespace['BASELINE_QUARTER'], '2021Q4')
        self.assertEqual(namespace['OBSERVATION_QUARTER'], '2022Q1')

    def test_script_temporal_section_exports_reusable_assignments(self):
        source = (ROOT / 'experiment_quarterly_with_lda.py').read_text()
        section = source[source.index('print("\\n=== Step 7B:'):source.index('print("\\n=== Step 8:')]
        frame = pd.DataFrame({
            'id': [1, 2, 3], 'datetime': ['2021-03-31', '2021-04-01', '2021-04-02'],
            'quarter': ['2021Q1', '2021Q2', '2021Q2'], 'hdbscan_cluster': [0, -1, 1],
        })
        with tempfile.TemporaryDirectory() as temporary:
            namespace = {
                'df': frame, 'hdb_top_words': {np.int64(0): ['legacy'], np.int64(1): ['emerging']},
                'temporal_config': TemporalConfig(window_days=14),
                'args': SimpleNamespace(temporal_output_dir=temporary, no_temporal_plots=True, focus_quarter='2021Q2'),
                'analyze_temporal_topics': analyze_temporal_topics,
                'focus_topic_quarter': focus_topic_quarter, 'export_focus_quarter': export_focus_quarter,
                'export_temporal_analysis': export_temporal_analysis, 'Path': Path, 'json': json,
            }
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(section, 'benchmark-temporal-section', 'exec'), namespace)
            output = Path(temporary)
            assignments = pd.read_csv(output / 'topic_assignments.csv')
            self.assertEqual(assignments['hdbscan_cluster'].tolist(), [0, -1, 1])
            labels = json.loads((output / 'topic_labels.json').read_text())
            self.assertEqual(labels, {'0': 'legacy', '1': 'emerging'})
            self.assertEqual(namespace['temporal_analysis']['summary']['quarter_counts'], {'2021Q1': 1, '2021Q2': 2})
            self.assertTrue((output / 'topic_trajectories.csv').is_file())
            self.assertTrue((output / 'topic_trajectories_2021Q2.csv').is_file())
            self.assertEqual(namespace['quarter_focus']['summary']['focus_quarter'], '2021Q2')

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'optional matplotlib missing')
    def test_notebook_comparison_section_reuses_shared_analysis(self):
        source = next(cell for cell in notebook_cells() if cell.startswith('# 8C.'))
        source = source.split('from IPython.display import Image, display')[0]
        frame = pd.DataFrame({
            'datetime': ['2021-01-01', '2021-03-31', '2021-04-01', '2021-06-30'],
            'hdbscan_cluster': [0, -1, 0, 1],
        })
        analysis = analyze_temporal_topics(frame, config=TemporalConfig(window_days=14))
        namespace = {
            'temporal_analysis': analysis, 'FOCUS_QUARTER': '2021Q2',
            'display': lambda _: None, 'json': json,
        }
        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(source, 'notebook-comparison-section', 'exec'), namespace)
                output = Path('temporal_results/comparisons')
                self.assertEqual(len(list(output.glob('*.png'))), 2)
                self.assertEqual(len(list(output.glob('*.pdf'))), 2)
                self.assertEqual(len(list(output.glob('*.csv'))), 3)
                self.assertEqual(len(list(output.glob('*.json'))), 1)
                self.assertEqual(namespace['trajectory_comparison']['summary']['baseline_quarter'], '2021Q1')
                self.assertEqual(namespace['trajectory_comparison']['summary']['target_quarter'], '2021Q2')
            finally:
                os.chdir(previous_directory)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'optional matplotlib missing')
    def test_notebook_temporal_section_runs_and_exports(self):
        source = next(cell for cell in notebook_cells() if cell.startswith('# 8B.'))
        # The final display block uses IPython; analysis/export do not require it.
        source = source.split('from IPython.display import Image, display')[0]
        frame = pd.DataFrame({
            'datetime': ['2021-03-31', '2021-04-01'], 'quarter': ['2021Q1', '2021Q2'],
            'hdbscan_cluster': [0, -1],
        })
        namespace = {'df': frame, 'DESCRIPTIVE_LABELS': {0: 'synthetic'}, 'json': json, 'display': lambda _: None}
        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(source, 'notebook-temporal-section', 'exec'), namespace)
                output = Path('temporal_results')
                self.assertTrue((output / 'topic_assignments.csv').is_file())
                self.assertTrue((output / 'topic_labels.json').is_file())
                self.assertEqual(len(list(output.glob('*.png'))), 5)
                self.assertTrue((output / 'topic_trajectories_2021Q2.pdf').is_file())
                self.assertTrue((output / 'monthly_topic_prevalence_2021Q2.csv').is_file())
            finally:
                os.chdir(previous_directory)


if __name__ == '__main__':
    unittest.main()
