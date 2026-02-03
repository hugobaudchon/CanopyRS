"""
Integration tests for Pipeline using real data.

These tests require downloading the BCI50ha dataset (~500MB).
They are marked with @pytest.mark.slow and can be skipped with:
    pytest -m "not slow"

The data is cached in ~/.cache/canopyrs_test_data/ so subsequent runs are fast.
"""

import pytest
from pathlib import Path


@pytest.mark.slow
class TestPipelineWithBCI50ha:
    """Integration tests using BCI50ha dataset."""

    def test_bci50ha_data_downloaded(self, bci50ha_raw_data):
        """Verify BCI50ha raw data is available."""
        assert bci50ha_raw_data.exists()

        # Check expected files exist
        raster = bci50ha_raw_data / "BCI_50ha_2020_08_01_crownmap_raw.tif"
        labels = bci50ha_raw_data / "BCI_50ha_2020_08_01_crownmap_improved.gpkg"

        assert raster.exists(), f"Raster not found: {raster}"
        assert labels.exists(), f"Labels not found: {labels}"

    def test_bci50ha_tiles_created(self, bci50ha_tiles):
        """Verify BCI50ha tiles are created."""
        assert bci50ha_tiles.exists()

        # Check tiles exist
        tiles = list(bci50ha_tiles.glob("**/*.tif"))
        assert len(tiles) > 0, "No tiles found"

    def test_pipeline_tilerizer_only(self, bci50ha_raw_data, tmp_path):
        """Test pipeline with just tilerizer component."""
        from canopyrs.engine.pipeline import Pipeline
        from canopyrs.engine.config_parsers import PipelineConfig, InferIOConfig, TilerizerConfig

        raster_path = bci50ha_raw_data / "BCI_50ha_2020_08_01_crownmap_raw.tif"

        io_config = InferIOConfig(
            input_imagery=str(raster_path),
            tiles_path=None,
            output_folder=str(tmp_path / "output"),
        )

        tilerizer_config = TilerizerConfig(
            tile_size=1777,
            overlap=0.25,
            tile_type="tile",
        )

        pipeline_config = PipelineConfig(
            components_configs=[("tilerizer", tilerizer_config)]
        )

        pipeline = Pipeline.from_config(io_config, pipeline_config)

        # Verify flow chart works
        pipeline._print_flow_chart()

        # Run pipeline
        pipeline()

        # Verify tiles were created
        tiles_path = Path(pipeline.data_state.tiles_path)
        assert tiles_path.exists()
        tiles = list(tiles_path.glob("**/*.tif"))
        assert len(tiles) > 0, "No tiles created"

    def test_pipeline_validation_catches_missing_dependency(self, bci50ha_tiles, tmp_path):
        """Test that pipeline validation catches missing component dependencies."""
        from canopyrs.engine.pipeline import Pipeline
        from canopyrs.engine.config_parsers import PipelineConfig, InferIOConfig, AggregatorConfig
        from canopyrs.engine.components.base import ComponentValidationError

        # Try to create a pipeline with aggregator but no detector/segmenter
        # This should fail validation because aggregator needs INFER_GDF
        io_config = InferIOConfig(
            input_imagery=None,
            tiles_path=str(bci50ha_tiles / "BCI_50ha_2020_08_01_crownmap" / "test"),
            output_folder=str(tmp_path / "output"),
        )

        aggregator_config = AggregatorConfig(
            nms_threshold=0.3,
            score_threshold=0.5,
        )

        pipeline_config = PipelineConfig(
            components_configs=[("aggregator", aggregator_config)]
        )

        # Pipeline creation should fail because aggregator needs INFER_GDF
        # which is not produced by any previous component
        with pytest.raises(ComponentValidationError):
            Pipeline.from_config(io_config, pipeline_config)


class TestPipelineWithBCI50haSmall:
    """Unit tests using small BCI50ha labels (25MB download, fast)."""

    def test_bci50ha_labels_available(self, bci50ha_small_labels):
        """Verify BCI50ha labels can be loaded."""
        import geopandas as gpd

        labels_file = bci50ha_small_labels / "BCI_50ha_2020_08_01_crownmap_improved.gpkg"
        assert labels_file.exists(), f"Labels file not found: {labels_file}"

        # Load and verify basic properties
        gdf = gpd.read_file(labels_file)
        assert len(gdf) > 0, "Labels GeoDataFrame is empty"
        assert 'geometry' in gdf.columns, "Missing geometry column"
        print(f"Loaded {len(gdf)} labels from BCI50ha dataset")


@pytest.mark.slow
class TestPipelineWithSyntheticData:
    """Integration tests using synthetic raster data (faster than BCI50ha)."""

    def test_synthetic_raster_created(self, synthetic_raster):
        """Verify synthetic raster fixture works."""
        import rasterio

        assert synthetic_raster.exists()

        with rasterio.open(synthetic_raster) as src:
            assert src.width == 256
            assert src.height == 256
            assert src.count == 3

    def test_synthetic_labels_created(self, synthetic_labels):
        """Verify synthetic labels fixture works."""
        import geopandas as gpd

        assert synthetic_labels.exists()

        gdf = gpd.read_file(synthetic_labels)
        assert len(gdf) == 4
        assert 'geometry' in gdf.columns

    def test_tilerizer_with_synthetic_data(self, synthetic_raster, tmp_path):
        """Test tilerizer component with synthetic raster."""
        from canopyrs.engine.pipeline import Pipeline
        from canopyrs.engine.config_parsers import PipelineConfig, InferIOConfig, TilerizerConfig

        io_config = InferIOConfig(
            input_imagery=str(synthetic_raster),
            tiles_path=None,
            output_folder=str(tmp_path / "output"),
        )

        tilerizer_config = TilerizerConfig(
            tile_size=128,  # Small tiles for 256x256 image
            overlap=0.25,
            tile_type="tile",
        )

        pipeline_config = PipelineConfig(
            components_configs=[("tilerizer", tilerizer_config)]
        )

        pipeline = Pipeline.from_config(io_config, pipeline_config)
        pipeline()

        # Verify tiles were created
        tiles_path = Path(pipeline.data_state.tiles_path)
        assert tiles_path.exists()
        tiles = list(tiles_path.glob("**/*.tif"))
        assert len(tiles) > 0
