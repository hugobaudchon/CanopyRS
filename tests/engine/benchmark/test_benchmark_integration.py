"""
Integration tests for benchmark system using real datasets.

These tests download the SelvaMask dataset (~3.3GB) and run benchmarks
with real configs. They are marked as @pytest.mark.slow and can be skipped.

The data is cached in ~/.cache/canopyrs_test_data/benchmarks/ so subsequent
runs are faster.
"""

import shutil
import tempfile

import pytest
import pandas as pd
from pathlib import Path

from canopyrs.engine.config_parsers import (
    SegmenterConfig, AggregatorConfig,
)
from canopyrs.engine.config_parsers.base import get_config_path


# Cache directory for benchmark test data
TEST_DATA_CACHE = Path.home() / ".cache" / "canopyrs_test_data" / "benchmarks"


@pytest.fixture
def short_tmp():
    """Short temp directory to avoid Windows MAX_PATH (260 char) issues.

    Pytest's tmp_path can be very long (~85 chars before the test even starts),
    and benchmark NMS search creates deeply nested subdirectories.
    """
    d = Path(tempfile.mkdtemp(prefix="crs_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)

# RF1 50:95 IoU thresholds
RF1_50_95 = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


@pytest.fixture(scope="session")
def selvamask_dataset():
    """
    Download and cache SelvaMask dataset for benchmark integration tests.

    Downloads valid and test folds (not train) to keep tests reasonable.
    Session-scoped so download happens only once.
    """
    from canopyrs.data.detection.preprocessed_datasets import DATASET_REGISTRY

    dataset_root = TEST_DATA_CACHE / "datasets"
    dataset_root.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded by looking for a location folder
    # SelvaMask has 'selvamask' location
    expected_location = dataset_root / "selvamask"
    if (expected_location.exists()
        and any(expected_location.glob("**/test/*.tif"))
        and any(expected_location.glob("**/valid/*.tif"))):
        print(f"SelvaMask dataset already cached at {dataset_root}")
        return dataset_root

    # Download SelvaMask
    try:
        print("Downloading SelvaMask dataset (valid and test folds only)...")
        DatasetClass = DATASET_REGISTRY.get('SelvaMask')
        if DatasetClass is None:
            pytest.skip("SelvaMask not found in DATASET_REGISTRY")

        ds = DatasetClass()
        ds.download_and_extract(
            root_output_path=str(dataset_root),
            folds=['valid', 'test']  # Download valid and test folds, skip train
        )

        print("Verifying dataset integrity...")
        ds.verify_dataset(
            root_output_path=str(dataset_root),
            folds=['valid', 'test']
        )

        print(f"SelvaMask dataset downloaded and verified at {dataset_root}")
        return dataset_root
    except Exception as e:
        pytest.skip(f"Could not download SelvaMask dataset: {e}")


@pytest.fixture(scope="session")
def maskrcnn_segmenter_config():
    """Mask R-CNN segmenter config (trained on SelvaMask)."""
    return SegmenterConfig.from_yaml(
        get_config_path("segmenters/maskrcnn_r50_multi_selvamask.yaml")
    )


# =============================================================================
# Dataset Smoke Tests
# =============================================================================

@pytest.mark.slow
class TestSelvaMaskDataset:
    """Verify SelvaMask dataset is available and well-structured."""

    def test_selvamask_dataset_available(self, selvamask_dataset):
        """Verify SelvaMask dataset is downloaded and structured correctly."""
        location_dir = selvamask_dataset / "selvamask"
        assert location_dir.exists(), f"Expected location dir not found: {location_dir}"

        raster_dirs = [d for d in location_dir.iterdir() if d.is_dir()]
        assert len(raster_dirs) > 0, "No raster folders found in SelvaMask"

        has_test_data = any(
            list(d.glob("tiles/test/*.tif")) for d in raster_dirs
        )
        assert has_test_data, "No test tiles found in SelvaMask dataset"


# =============================================================================
# NMS Grid Search Tests
# =============================================================================

@pytest.mark.slow
class TestNMSGridSearch:
    """Integration test: NMS parameter search using Mask R-CNN on SelvaMask valid set."""

    def test_find_optimal_nms_params(
        self, selvamask_dataset, maskrcnn_segmenter_config, short_tmp
    ):
        """
        Run a small NMS grid search (5x5) on SelvaMask valid set using
        the SegmenterBenchmarker with Mask R-CNN.

        Asserts the returned AggregatorConfig contains values that were
        actually in the search grid.
        """
        from canopyrs.engine.benchmark.segmenter.benchmark import SegmenterBenchmarker

        # Small 5x5 grid
        nms_iou_thresholds = [0.01, 0.05, 0.1, 0.2, 0.3]
        nms_score_thresholds = [0.2, 0.35, 0.5, 0.65, 0.8]

        base_aggregator_config = AggregatorConfig(
            nms_algorithm='ioa-disambiguate',
            scores_weights={'detector_score': 0.0, 'segmenter_score': 1.0},
            scores_weighting_method='weighted_geometric_mean',
            edge_band_buffer_percentage=0.05,
        )

        output_dir = short_tmp / "nms"
        benchmarker = SegmenterBenchmarker(
            output_folder=str(output_dir),
            fold_name='valid',
            raw_data_root=str(selvamask_dataset),
            eval_iou_threshold=RF1_50_95,
        )

        optimal_config = benchmarker.find_optimal_nms_iou_threshold(
            segmenter_config=maskrcnn_segmenter_config,
            base_aggregator_config=base_aggregator_config,
            dataset_names=['SelvaMask'],
            nms_iou_thresholds=nms_iou_thresholds,
            nms_score_thresholds=nms_score_thresholds,
            eval_at_ground_resolution=0.045,
            n_workers=8,
        )

        # Returned config must be an AggregatorConfig
        assert isinstance(optimal_config, AggregatorConfig)

        # The chosen thresholds must be values from the search grid
        assert optimal_config.nms_threshold in nms_iou_thresholds, (
            f"Optimal nms_threshold {optimal_config.nms_threshold} not in grid {nms_iou_thresholds}"
        )
        assert optimal_config.score_threshold in nms_score_thresholds, (
            f"Optimal score_threshold {optimal_config.score_threshold} not in grid {nms_score_thresholds}"
        )

        # Base config properties should be preserved
        assert optimal_config.nms_algorithm == 'ioa-disambiguate'

        # CSV results file should have been saved
        csv_path = output_dir / "valid" / "NMS_search" / "optimal_nms_iou_threshold_search.csv"
        assert csv_path.exists(), f"Grid search CSV not found at {csv_path}"

        results_df = pd.read_csv(csv_path)
        assert len(results_df) > 0, "Grid search CSV is empty"
        assert 'nms_iou_threshold' in results_df.columns
        assert 'nms_score_threshold' in results_df.columns
        assert 'f1' in results_df.columns


# =============================================================================
# Benchmarking Tests
# =============================================================================

@pytest.mark.slow
class TestSegmenterBenchmark:
    """Integration test: full benchmark with Mask R-CNN using ioa-disambiguate and RF1 50:95."""

    def test_benchmark_maskrcnn(
        self, selvamask_dataset, maskrcnn_segmenter_config, short_tmp
    ):
        """
        Run a full benchmark on SelvaMask test set with Mask R-CNN,
        using ioa-disambiguate NMS and RF1 50:95 evaluation.
        """
        from canopyrs.engine.benchmark.segmenter.benchmark import SegmenterBenchmarker

        aggregator_config = AggregatorConfig(
            nms_algorithm='ioa-disambiguate',
            score_threshold=0.5,
            nms_threshold=0.5,
            scores_weights={'detector_score': 0.0, 'segmenter_score': 1.0},
            scores_weighting_method='weighted_geometric_mean',
            edge_band_buffer_percentage=0.05,
        )

        output_dir = short_tmp / "bench"
        benchmarker = SegmenterBenchmarker(
            output_folder=str(output_dir),
            fold_name='test',
            raw_data_root=str(selvamask_dataset),
            eval_iou_threshold=RF1_50_95,
        )

        tile_metrics_df, raster_metrics_df = benchmarker.benchmark(
            segmenter_config=maskrcnn_segmenter_config,
            aggregator_config=aggregator_config,
            dataset_names=['SelvaMask'],
        )

        # Tile-level metrics
        tile_csv = output_dir / "test" / "tile_level_metrics.csv"
        assert tile_csv.exists(), f"Tile metrics CSV not found at {tile_csv}"
        assert isinstance(tile_metrics_df, pd.DataFrame)
        assert len(tile_metrics_df) > 0

        # Raster-level metrics
        raster_csv = output_dir / "test" / "raster_level_metrics.csv"
        assert raster_csv.exists(), f"Raster metrics CSV not found at {raster_csv}"
        assert isinstance(raster_metrics_df, pd.DataFrame)
        assert len(raster_metrics_df) > 0

        # RF1 50:95 metrics should be present
        assert 'f1' in raster_metrics_df.columns, "Missing f1 (averaged RF1) column"
        assert 'precision' in raster_metrics_df.columns
        assert 'recall' in raster_metrics_df.columns

        # f1 values should be between 0 and 1
        f1_values = raster_metrics_df['f1'].dropna()
        assert (f1_values >= 0).all() and (f1_values <= 1).all(), "f1 values out of [0, 1] range"
