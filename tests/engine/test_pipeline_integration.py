"""
Integration tests for Pipeline using the test raster (assets/).

These tests use a real orthomosaic crop included in the repo (~24MB).
They are marked with @pytest.mark.slow and can be skipped with:
    pytest -m "not slow"
"""

import pytest
from pathlib import Path

from canopyrs.engine.config_parsers import PipelineConfig, InferIOConfig
from canopyrs.engine.config_parsers.base import get_config_path
from canopyrs.engine.pipeline import Pipeline


@pytest.mark.slow
class TestPipelineIntegration:
    """Integration tests using the test raster asset."""

    def test_raster_available(self, test_raster):
        """Verify test raster is available."""
        import rasterio

        assert test_raster.exists()

        with rasterio.open(test_raster) as src:
            assert src.count >= 3, "Test raster must have at least 3 bands"
            assert src.width > 0
            assert src.height > 0

    def test_tiles_created(self, test_raster_tiles):
        """Verify test raster tiles are created."""
        assert test_raster_tiles.exists()

        tiles = list(test_raster_tiles.glob("**/*.tif"))
        assert len(tiles) > 0, "No tiles found"

    def test_pipeline_tilerizer_only(self, test_raster, tmp_path):
        """Test pipeline with just tilerizer component."""
        from canopyrs.engine.config_parsers import TilerizerConfig

        io_config = InferIOConfig(
            input_imagery=str(test_raster),
            tiles_path=None,
            output_folder=str(tmp_path / "output"),
        )

        tilerizer_config = TilerizerConfig(
            tile_size=512,
            tile_overlap=0.25,
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
        assert len(tiles) > 0, "No tiles created"

    def test_pipeline_validation_catches_missing_dependency(self, test_raster_tiles, tmp_path):
        """Test that pipeline validation catches missing component dependencies."""
        from canopyrs.engine.config_parsers import AggregatorConfig
        from canopyrs.engine.components.base import ComponentValidationError

        # Try to create a pipeline with aggregator but no detector/segmenter
        # This should fail validation because aggregator needs INFER_GDF
        io_config = InferIOConfig(
            input_imagery=None,
            tiles_path=str(test_raster_tiles),
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
        with pytest.raises(ComponentValidationError):
            Pipeline.from_config(io_config, pipeline_config)


@pytest.mark.slow
class TestPipelineWithSyntheticData:
    """Integration tests using synthetic raster data (fastest)."""

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
        from canopyrs.engine.config_parsers import TilerizerConfig

        io_config = InferIOConfig(
            input_imagery=str(synthetic_raster),
            tiles_path=None,
            output_folder=str(tmp_path / "output"),
        )

        tilerizer_config = TilerizerConfig(
            tile_size=128,  # Small tiles for 256x256 image
            tile_overlap=0.25,
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


# =============================================================================
# Model Inference Integration Tests
# =============================================================================

# Preset pipeline config names (resolved by PipelineConfig.from_yaml via get_config_path)
DINO_SAM3_CONFIG = "preset_seg_multi_NQOS_selvamask_SAM3_FT_quality.yaml"
MASKRCNN_CONFIG = "maskrcnn_test.yaml"
FASTERRCNN_CONFIG = "fasterrcnn_test.yaml"


def _run_pipeline_from_preset(preset_name: str, test_raster, tmp_path):
    """Run a pipeline from a preset YAML config on the test raster."""
    pipeline_config = PipelineConfig.from_yaml(get_config_path(preset_name))

    io_config = InferIOConfig(
        input_imagery=str(test_raster),
        tiles_path=None,
        output_folder=str(tmp_path / "output"),
    )

    pipeline = Pipeline.from_config(io_config, pipeline_config)
    pipeline()
    return pipeline


@pytest.mark.slow
class TestDinoSam3Pipeline:
    """Integration test: DINO FT on SelvaMask + SAM3 FT (detector + segmenter)."""

    def test_dino_sam3_inference(self, test_raster, tmp_path):
        """Full DINO + SAM3 pipeline produces segmentation outputs."""
        pipeline = _run_pipeline_from_preset(DINO_SAM3_CONFIG, test_raster, tmp_path)

        # Pipeline should have completed with an infer_gdf
        gdf = pipeline.data_state.infer_gdf
        assert gdf is not None, "Pipeline did not produce an infer_gdf"
        assert len(gdf) > 0, "Pipeline produced zero detections"

        # Should have segmenter-produced geometry (polygons, not just bboxes)
        assert "geometry" in gdf.columns

        # Check output files exist (detector COCO, aggregator gpkg)
        output_path = Path(pipeline.output_path)
        assert output_path.exists()
        coco_files = list(output_path.glob("**/*.json"))
        gpkg_files = list(output_path.glob("**/*.gpkg"))
        assert len(coco_files) > 0, "No COCO JSON files produced"
        assert len(gpkg_files) > 0, "No GeoPackage files produced"


@pytest.mark.slow
class TestMaskRCNNPipeline:
    """Integration test: Mask R-CNN pipeline (end-to-end segmenter)."""

    def test_maskrcnn_inference(self, test_raster, tmp_path):
        """Mask R-CNN pipeline produces segmentation outputs."""
        pipeline = _run_pipeline_from_preset(MASKRCNN_CONFIG, test_raster, tmp_path)

        gdf = pipeline.data_state.infer_gdf
        assert gdf is not None, "Pipeline did not produce an infer_gdf"
        assert len(gdf) > 0, "Pipeline produced zero detections"
        assert "geometry" in gdf.columns

        output_path = Path(pipeline.output_path)
        gpkg_files = list(output_path.glob("**/*.gpkg"))
        assert len(gpkg_files) > 0, "No GeoPackage files produced"


@pytest.mark.slow
class TestFasterRCNNPipeline:
    """Integration test: Faster R-CNN pipeline (detection only)."""

    def test_fasterrcnn_inference(self, test_raster, tmp_path):
        """Faster R-CNN pipeline produces detection outputs."""
        pipeline = _run_pipeline_from_preset(FASTERRCNN_CONFIG, test_raster, tmp_path)

        gdf = pipeline.data_state.infer_gdf
        assert gdf is not None, "Pipeline did not produce an infer_gdf"
        assert len(gdf) > 0, "Pipeline produced zero detections"

        output_path = Path(pipeline.output_path)
        coco_files = list(output_path.glob("**/*.json"))
        assert len(coco_files) > 0, "No COCO JSON files produced"
