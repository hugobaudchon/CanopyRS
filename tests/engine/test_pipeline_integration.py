"""Integration tests for the Pipeline.

The fast tier drives a real tilerizer over a synthetic raster (CPU only) and checks persistence /
reload. The @slow tier runs real models on GPU: a full pipeline over the bundled test raster, and the
tiles-folder seeding path.
"""

from pathlib import Path

import pytest

from canopyrs.engine import store
from canopyrs.engine.constants import Col
from canopyrs.engine.data import Objects, Tiles
from canopyrs.engine.pipeline import Pipeline
from canopyrs.engine.config_parsers import PipelineConfig, TilerizerConfig
from canopyrs.engine.config_parsers.base import get_config_path


# =============================================================================
# Fast tier (CPU only — real geodataset tilerizing, no models)
# =============================================================================

def test_tilerizer_pipeline_produces_tiles_and_reloads(synthetic_raster, tmp_path):
    """A tilerizer-only pipeline over a raster produces Tiles, writes a run record, and reloads."""
    run_dir = tmp_path / "run"
    steps = [('tilerizer', TilerizerConfig(tile_type='tile', tile_size=128, tile_overlap=0.0))]

    pipe = Pipeline.from_config(steps, sources=str(synthetic_raster), output_dir=str(run_dir))
    pipe.run(verbose=False)

    tiles = pipe.latest(Tiles)
    assert tiles is not None and len(tiles) > 0
    assert store.read_run_record(run_dir) is not None

    reloaded = Pipeline.from_dir(run_dir)
    assert reloaded.latest(Tiles) is not None
    assert len(reloaded.latest(Tiles)) == len(tiles)


def test_resume_skips_completed_prefix(synthetic_raster, tmp_path):
    """Re-running with resume=True over a completed tilerizer skips it (no error, tiles still present)."""
    run_dir = tmp_path / "run"
    steps = [('tilerizer', TilerizerConfig(tile_type='tile', tile_size=128, tile_overlap=0.0))]

    Pipeline.from_config(steps, sources=str(synthetic_raster), output_dir=str(run_dir)).run(verbose=False)
    resumed = Pipeline.from_config(steps, sources=str(synthetic_raster), output_dir=str(run_dir))
    resumed.run(resume=True, verbose=False)

    assert resumed.latest(Tiles) is not None and len(resumed.latest(Tiles)) > 0


# =============================================================================
# Slow tier (GPU + model weights)
# =============================================================================

DETECTOR_PRESET = 'preset_det_single_S_fasterrcnn_r50.yaml'


@pytest.mark.slow
def test_full_detector_pipeline_on_test_raster(test_raster, tmp_path):
    """Run the single-scale Faster R-CNN preset over the bundled orthomosaic crop: it produces Objects,
    writes a reloadable run, and (when the preset aggregates) a georeferenced final GPKG."""
    config = PipelineConfig.from_yaml(get_config_path(DETECTOR_PRESET))
    run_dir = tmp_path / "run"

    pipe = Pipeline.from_config(config.components_configs, sources=str(test_raster), output_dir=str(run_dir))
    pipe.run(verbose=False, strict_rgb_validation=False)

    assert pipe.latest(Tiles) is not None and len(pipe.latest(Tiles)) > 0
    assert pipe.latest(Objects) is not None

    reloaded = Pipeline.from_dir(run_dir)
    assert reloaded.latest(Objects) is not None

    if any(c.name == 'aggregator' for c in pipe.components):
        assert (run_dir / "final.gpkg").exists()


@pytest.mark.slow
def test_detector_seeded_from_tiles_dir(test_raster, tmp_path):
    """The tiles-folder seeding path: seed a detector directly from a folder of pre-cut GeoTIFF tiles
    (leading tilerizer dropped), exercising Tiles.from_image_dir end-to-end."""
    # First cut real tiles to disk with a tilerizer run.
    tiles_run = tmp_path / "tiles_run"
    tiler_steps = [('tilerizer', TilerizerConfig(tile_type='tile', tile_size=512, tile_overlap=0.0,
                                                 save_tiles_to_disk=True))]
    tiler = Pipeline.from_config(tiler_steps, sources=str(test_raster), output_dir=str(tiles_run))
    tiler.run(verbose=False, strict_rgb_validation=False)
    # The tilerizer's own tile_path column points at the saved tiles — use its folder directly.
    tiles_dir = Path(tiler.latest(Tiles).df[Col.PATH].iloc[0]).parent

    # Now seed a detector-only pipeline straight from that tiles folder (no raster, no tilerizer).
    config = PipelineConfig.from_yaml(get_config_path(DETECTOR_PRESET))
    detector_steps = [step for step in config.components_configs if step[0] == 'detector']
    if not detector_steps:
        pytest.skip(f"{DETECTOR_PRESET} has no detector step")

    det_run = tmp_path / "det_run"
    pipe = Pipeline.from_config(detector_steps, tiles=str(tiles_dir), output_dir=str(det_run))
    pipe.run(verbose=False)

    assert pipe.latest(Tiles) is not None and len(pipe.latest(Tiles)) > 0
    assert pipe.latest(Objects) is not None   # boxes (possibly zero on some tiles), but the table exists
