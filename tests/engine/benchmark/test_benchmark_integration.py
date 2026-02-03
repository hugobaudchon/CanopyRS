"""
Integration tests for benchmark system using real datasets.

These tests download the SelvaMask dataset (~3.3GB) and run benchmarks
with real configs. They are marked as @pytest.mark.slow and can be skipped.

The data is cached in ~/.cache/canopyrs_test_data/benchmarks/ so subsequent
runs are faster.
"""

import pytest
from pathlib import Path


# Cache directory for benchmark test data
TEST_DATA_CACHE = Path.home() / ".cache" / "canopyrs_test_data" / "benchmarks"


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


@pytest.mark.slow
class TestDetectorBenchmark:
    """Integration tests for detector benchmarking."""

    def test_selvamask_dataset_available(self, selvamask_dataset):
        """Verify SelvaMask dataset is downloaded and structured correctly."""
        # SelvaMask structure: selvamask/ contains raster folders directly
        location_dir = selvamask_dataset / "selvamask"
        assert location_dir.exists(), f"Expected location dir not found: {location_dir}"

        # Check that there are some raster folders with test tiles
        raster_dirs = [d for d in location_dir.iterdir() if d.is_dir()]
        assert len(raster_dirs) > 0, "No raster folders found in SelvaMask"

        # Check that at least one raster has test fold data
        has_test_data = False
        for raster_dir in raster_dirs:
            test_tiles = list(raster_dir.glob("tiles/test/*.tif"))
            if test_tiles:
                has_test_data = True
                break

        assert has_test_data, "No test tiles found in SelvaMask dataset"
        print(f"SelvaMask dataset verified: {len(raster_dirs)} rasters available")

    def test_benchmark_with_segmentation_config(self, selvamask_dataset, tmp_path):
        """
        Test benchmarking with default_segmentation_multi_NQOS_best_L config.

        This is a smoke test that verifies the benchmark runs without errors.
        We only run on one raster to keep the test fast.
        """
        from canopyrs.engine.benchmark.detector.benchmark import DetectorBenchmarker
        from canopyrs.engine.config_parsers import DetectorConfig, AggregatorConfig, PipelineConfig

        # Load the pipeline config to extract detector component config
        project_root = Path(__file__).parent.parent.parent.parent
        pipeline_config_path = project_root / "canopyrs" / "config" / "default_segmentation_multi_NQOS_best_L" / "pipeline.yaml"

        if not pipeline_config_path.exists():
            pytest.skip(f"Config file not found: {pipeline_config_path}")

        # Load pipeline config and extract detector config reference
        pipeline_config = PipelineConfig.from_yaml(str(pipeline_config_path))

        # Find the detector component in the pipeline
        detector_config_ref = None
        for comp_name, comp_config in pipeline_config.components_configs:
            if comp_name == 'detector':
                # If it's a string reference, resolve it
                if isinstance(comp_config, str):
                    detector_config_path = project_root / "canopyrs" / "config" / f"{comp_config}.yaml"
                    if not detector_config_path.exists():
                        pytest.skip(f"Detector config not found: {detector_config_path}")
                    detector_config = DetectorConfig.from_yaml(str(detector_config_path))
                else:
                    # It's already a config object
                    detector_config = comp_config
                break

        if detector_config is None:
            pytest.skip("No detector component found in pipeline config")

        # Create output directory
        output_dir = tmp_path / "benchmark_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create benchmarker
        benchmarker = DetectorBenchmarker(
            output_folder=str(output_dir),
            fold_name='test',
            raw_data_root=str(selvamask_dataset),
            eval_iou_threshold=0.75,  # Use RF1_75 metric
        )

        # Create aggregator config (using values from the pipeline config)
        aggregator_config = AggregatorConfig(
            nms_threshold=0.7,
            score_threshold=0.3,
            nms_algorithm='iou',
        )

        # Run benchmark on SelvaMask only
        # Note: This will run the full pipeline including tilerizer, detector, aggregator
        # and may take several minutes depending on hardware
        try:
            benchmarker.benchmark(
                detector_config=detector_config,
                aggregator_config=aggregator_config,
                dataset_names=['SelvaMask'],
            )

            # Verify output files were created
            # Check for tile-level metrics
            tile_metrics = output_dir / "test" / "tile_level_metrics.csv"
            assert tile_metrics.exists(), f"Tile metrics CSV not found at {tile_metrics}"

            # Check for raster-level metrics (if aggregation was run)
            raster_metrics = output_dir / "test" / "raster_level_metrics.csv"
            assert raster_metrics.exists(), f"Raster metrics CSV not found at {raster_metrics}"

            print(f"Benchmark completed successfully. Results in {output_dir}")

        except Exception as e:
            # If this is a CUDA/GPU error, skip the test
            if "CUDA" in str(e) or "GPU" in str(e) or "out of memory" in str(e).lower():
                pytest.skip(f"GPU/CUDA error (expected in CPU-only environment): {e}")
            else:
                raise


@pytest.mark.slow
class TestBenchmarkSmoke:
    """Lightweight smoke tests for benchmark system."""

    def test_benchmarker_initialization(self, selvamask_dataset, tmp_path):
        """Test that benchmarker can be initialized with valid parameters."""
        from canopyrs.engine.benchmark.detector.benchmark import DetectorBenchmarker

        output_dir = tmp_path / "bench_init"
        output_dir.mkdir(parents=True, exist_ok=True)

        benchmarker = DetectorBenchmarker(
            output_folder=str(output_dir),
            fold_name='test',
            raw_data_root=str(selvamask_dataset),
            eval_iou_threshold=0.75,
        )

        # Check basic attributes
        assert benchmarker.fold_name == 'test'
        assert benchmarker.output_folder == output_dir
        assert benchmarker.raw_data_root == selvamask_dataset

    def test_config_loading(self):
        """Test that the segmentation pipeline config can be loaded."""
        from canopyrs.engine.config_parsers import PipelineConfig

        project_root = Path(__file__).parent.parent.parent.parent
        config_path = project_root / "canopyrs" / "config" / "default_segmentation_multi_NQOS_best_L" / "pipeline.yaml"

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        pipeline_config = PipelineConfig.from_yaml(str(config_path))

        # Verify config has expected attributes
        assert hasattr(pipeline_config, 'components_configs')
        assert len(pipeline_config.components_configs) > 0

        # Check that it has a detector component
        component_names = [name for name, _ in pipeline_config.components_configs]
        assert 'detector' in component_names, "Pipeline should have a detector component"
        print(f"Pipeline config loaded successfully with {len(component_names)} components: {component_names}")
