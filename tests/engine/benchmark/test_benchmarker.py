"""
Tests for BaseBenchmarker utility methods.

Focuses on pure functions that can be tested without running full pipelines:
- compute_mean_std_metric_tables
- merge_tile_and_raster_summaries
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from canopyrs.engine.benchmark.base.base_benchmarker import BaseBenchmarker


class TestComputeMeanStdMetricTables:
    """Tests for compute_mean_std_metric_tables static method."""

    @pytest.fixture
    def sample_metrics_df_1(self):
        """First run metrics."""
        return pd.DataFrame({
            'location': ['dataset_a', 'dataset_a'],
            'product_name': ['raster_1', 'average_over_rasters'],
            'precision': [0.80, 0.82],
            'recall': [0.70, 0.72],
            'f1': [0.75, 0.77],
            'num_truths': [100, 100],
            'num_images': [5, 5],
        })

    @pytest.fixture
    def sample_metrics_df_2(self):
        """Second run metrics (different values, same structure)."""
        return pd.DataFrame({
            'location': ['dataset_a', 'dataset_a'],
            'product_name': ['raster_1', 'average_over_rasters'],
            'precision': [0.84, 0.86],
            'recall': [0.74, 0.76],
            'f1': [0.79, 0.81],
            'num_truths': [100, 100],
            'num_images': [5, 5],
        })

    @pytest.fixture
    def sample_metrics_df_3(self):
        """Third run metrics."""
        return pd.DataFrame({
            'location': ['dataset_a', 'dataset_a'],
            'product_name': ['raster_1', 'average_over_rasters'],
            'precision': [0.82, 0.84],
            'recall': [0.72, 0.74],
            'f1': [0.77, 0.79],
            'num_truths': [100, 100],
            'num_images': [5, 5],
        })

    def test_computes_mean_correctly(self, sample_metrics_df_1, sample_metrics_df_2, tmp_path):
        """Mean is computed correctly across runs."""
        output_csv = tmp_path / "summary.csv"

        result = BaseBenchmarker.compute_mean_std_metric_tables(
            inputs=[sample_metrics_df_1, sample_metrics_df_2],
            output_csv=output_csv
        )

        # First row: precision = (0.80 + 0.84) / 2 = 0.82
        assert pytest.approx(result.loc[0, 'precision_mean']) == 0.82
        # recall = (0.70 + 0.74) / 2 = 0.72
        assert pytest.approx(result.loc[0, 'recall_mean']) == 0.72

    def test_computes_std_correctly(self, sample_metrics_df_1, sample_metrics_df_2, tmp_path):
        """Standard deviation is computed correctly."""
        output_csv = tmp_path / "summary.csv"

        result = BaseBenchmarker.compute_mean_std_metric_tables(
            inputs=[sample_metrics_df_1, sample_metrics_df_2],
            output_csv=output_csv
        )

        # std of [0.80, 0.84] with ddof=0 = 0.02
        assert pytest.approx(result.loc[0, 'precision_std']) == 0.02

    def test_preserves_identity_columns(self, sample_metrics_df_1, sample_metrics_df_2, tmp_path):
        """Location and product_name are preserved."""
        output_csv = tmp_path / "summary.csv"

        result = BaseBenchmarker.compute_mean_std_metric_tables(
            inputs=[sample_metrics_df_1, sample_metrics_df_2],
            output_csv=output_csv
        )

        assert result.loc[0, 'location'] == 'dataset_a'
        assert result.loc[0, 'product_name'] == 'raster_1'
        assert result.loc[1, 'product_name'] == 'average_over_rasters'

    def test_passthrough_columns_not_averaged(self, sample_metrics_df_1, sample_metrics_df_2, tmp_path):
        """Passthrough columns (num_truths, num_images) are passed through, not averaged."""
        output_csv = tmp_path / "summary.csv"

        result = BaseBenchmarker.compute_mean_std_metric_tables(
            inputs=[sample_metrics_df_1, sample_metrics_df_2],
            output_csv=output_csv
        )

        # num_truths should be passed through from first df, not averaged
        assert result.loc[0, 'num_truths'] == 100
        assert result.loc[0, 'num_images'] == 5

    def test_three_runs_mean_std(
        self, sample_metrics_df_1, sample_metrics_df_2, sample_metrics_df_3, tmp_path
    ):
        """Mean/std work correctly with 3 inputs."""
        output_csv = tmp_path / "summary.csv"

        result = BaseBenchmarker.compute_mean_std_metric_tables(
            inputs=[sample_metrics_df_1, sample_metrics_df_2, sample_metrics_df_3],
            output_csv=output_csv
        )

        # First row precision: [0.80, 0.84, 0.82] -> mean = 0.82
        assert pytest.approx(result.loc[0, 'precision_mean']) == 0.82

        # std of [0.80, 0.84, 0.82] with ddof=0
        expected_std = np.std([0.80, 0.84, 0.82], ddof=0)
        assert pytest.approx(result.loc[0, 'precision_std']) == expected_std

    def test_saves_to_csv(self, sample_metrics_df_1, sample_metrics_df_2, tmp_path):
        """Result is saved to CSV file."""
        output_csv = tmp_path / "summary.csv"

        BaseBenchmarker.compute_mean_std_metric_tables(
            inputs=[sample_metrics_df_1, sample_metrics_df_2],
            output_csv=output_csv
        )

        assert output_csv.exists()
        loaded = pd.read_csv(output_csv)
        assert len(loaded) == 2
        assert 'precision_mean' in loaded.columns

    def test_accepts_csv_paths(self, sample_metrics_df_1, sample_metrics_df_2, tmp_path):
        """Can accept CSV file paths instead of DataFrames."""
        csv_1 = tmp_path / "metrics_1.csv"
        csv_2 = tmp_path / "metrics_2.csv"
        sample_metrics_df_1.to_csv(csv_1, index=False)
        sample_metrics_df_2.to_csv(csv_2, index=False)

        output_csv = tmp_path / "summary.csv"

        result = BaseBenchmarker.compute_mean_std_metric_tables(
            inputs=[csv_1, csv_2],
            output_csv=output_csv
        )

        assert pytest.approx(result.loc[0, 'precision_mean']) == 0.82

    def test_raises_on_row_count_mismatch(self, sample_metrics_df_1, tmp_path):
        """Raises error if inputs have different row counts."""
        df_different_rows = pd.DataFrame({
            'location': ['dataset_a'],
            'product_name': ['raster_1'],
            'precision': [0.85],
        })

        output_csv = tmp_path / "summary.csv"

        with pytest.raises(ValueError, match="same number of rows"):
            BaseBenchmarker.compute_mean_std_metric_tables(
                inputs=[sample_metrics_df_1, df_different_rows],
                output_csv=output_csv
            )

    def test_raises_on_missing_required_columns(self, tmp_path):
        """Raises error if required columns are missing."""
        df_missing_cols = pd.DataFrame({
            'precision': [0.80],
            'recall': [0.70],
        })

        output_csv = tmp_path / "summary.csv"

        with pytest.raises(ValueError, match="missing required columns"):
            BaseBenchmarker.compute_mean_std_metric_tables(
                inputs=[df_missing_cols],
                output_csv=output_csv
            )

    def test_handles_list_columns(self, tmp_path):
        """List columns (like f1_per_iou) are averaged element-wise."""
        df1 = pd.DataFrame({
            'location': ['dataset_a'],
            'product_name': ['raster_1'],
            'precision': [0.80],
            'f1_per_iou': [[0.70, 0.75, 0.80]],
        })
        df2 = pd.DataFrame({
            'location': ['dataset_a'],
            'product_name': ['raster_1'],
            'precision': [0.84],
            'f1_per_iou': [[0.74, 0.79, 0.84]],
        })

        output_csv = tmp_path / "summary.csv"

        result = BaseBenchmarker.compute_mean_std_metric_tables(
            inputs=[df1, df2],
            output_csv=output_csv
        )

        # Mean of [0.70, 0.74], [0.75, 0.79], [0.80, 0.84]
        expected_mean = [0.72, 0.77, 0.82]
        assert 'f1_per_iou_mean' in result.columns
        actual_mean = result.loc[0, 'f1_per_iou_mean']
        for i, (exp, act) in enumerate(zip(expected_mean, actual_mean)):
            assert pytest.approx(act, rel=0.01) == exp


class TestMergeTileAndRasterSummaries:
    """Tests for merge_tile_and_raster_summaries static method."""

    @pytest.fixture
    def tile_summary_df(self):
        """Sample tile-level summary."""
        return pd.DataFrame({
            'location': ['dataset_a', 'dataset_a'],
            'product_name': ['raster_1', 'average_over_rasters'],
            'AP_mean': [0.45, 0.47],
            'AP_std': [0.02, 0.01],
            'AR_mean': [0.50, 0.52],
            'AR_std': [0.03, 0.02],
            'num_images': [10, 10],
        })

    @pytest.fixture
    def raster_summary_df(self):
        """Sample raster-level summary."""
        return pd.DataFrame({
            'location': ['dataset_a', 'dataset_a'],
            'product_name': ['raster_1', 'average_over_rasters'],
            'precision_mean': [0.80, 0.82],
            'precision_std': [0.02, 0.01],
            'recall_mean': [0.70, 0.72],
            'recall_std': [0.03, 0.02],
            'f1_mean': [0.75, 0.77],
            'f1_std': [0.02, 0.01],
            'num_truths': [100, 100],
        })

    def test_merges_on_identity_columns(self, tile_summary_df, raster_summary_df, tmp_path):
        """Merge happens on location and product_name."""
        output_csv = tmp_path / "merged.csv"

        result = BaseBenchmarker.merge_tile_and_raster_summaries(
            tile_csv=tile_summary_df,
            raster_csv=raster_summary_df,
            output_csv=output_csv
        )

        assert len(result) == 2
        assert 'location' in result.columns
        assert 'product_name' in result.columns

    def test_contains_all_metrics(self, tile_summary_df, raster_summary_df, tmp_path):
        """Merged result contains metrics from both sources."""
        output_csv = tmp_path / "merged.csv"

        result = BaseBenchmarker.merge_tile_and_raster_summaries(
            tile_csv=tile_summary_df,
            raster_csv=raster_summary_df,
            output_csv=output_csv
        )

        # Tile metrics
        assert 'AP_mean' in result.columns
        assert 'AR_mean' in result.columns

        # Raster metrics
        assert 'precision_mean' in result.columns
        assert 'recall_mean' in result.columns
        assert 'f1_mean' in result.columns

    def test_applies_prefixes(self, tile_summary_df, raster_summary_df, tmp_path):
        """Prefixes are applied to metric columns."""
        output_csv = tmp_path / "merged.csv"

        result = BaseBenchmarker.merge_tile_and_raster_summaries(
            tile_csv=tile_summary_df,
            raster_csv=raster_summary_df,
            output_csv=output_csv,
            tile_prefix="tile",
            raster_prefix="raster"
        )

        # Tile metrics should be prefixed
        assert 'tile_AP_mean' in result.columns
        assert 'tile_AR_mean' in result.columns

        # Raster metrics should be prefixed
        assert 'raster_precision_mean' in result.columns
        assert 'raster_recall_mean' in result.columns

        # Identity columns should NOT be prefixed
        assert 'location' in result.columns
        assert 'product_name' in result.columns

    def test_identity_columns_not_prefixed(self, tile_summary_df, raster_summary_df, tmp_path):
        """Identity columns (num_images, num_truths) are not prefixed."""
        output_csv = tmp_path / "merged.csv"

        result = BaseBenchmarker.merge_tile_and_raster_summaries(
            tile_csv=tile_summary_df,
            raster_csv=raster_summary_df,
            output_csv=output_csv,
            tile_prefix="tile",
            raster_prefix="raster"
        )

        # These should remain unprefixed
        assert 'num_images' in result.columns
        assert 'num_truths' in result.columns

    def test_saves_to_csv(self, tile_summary_df, raster_summary_df, tmp_path):
        """Result is saved to CSV when output_csv is provided."""
        output_csv = tmp_path / "merged.csv"

        BaseBenchmarker.merge_tile_and_raster_summaries(
            tile_csv=tile_summary_df,
            raster_csv=raster_summary_df,
            output_csv=output_csv
        )

        assert output_csv.exists()
        loaded = pd.read_csv(output_csv)
        assert len(loaded) == 2

    def test_no_output_csv(self, tile_summary_df, raster_summary_df):
        """Returns DataFrame without saving when output_csv is None."""
        result = BaseBenchmarker.merge_tile_and_raster_summaries(
            tile_csv=tile_summary_df,
            raster_csv=raster_summary_df,
            output_csv=None
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_accepts_csv_paths(self, tile_summary_df, raster_summary_df, tmp_path):
        """Can accept CSV file paths instead of DataFrames."""
        tile_csv = tmp_path / "tile.csv"
        raster_csv = tmp_path / "raster.csv"
        tile_summary_df.to_csv(tile_csv, index=False)
        raster_summary_df.to_csv(raster_csv, index=False)

        result = BaseBenchmarker.merge_tile_and_raster_summaries(
            tile_csv=tile_csv,
            raster_csv=raster_csv,
            output_csv=None
        )

        assert len(result) == 2
        assert 'AP_mean' in result.columns
        assert 'precision_mean' in result.columns


