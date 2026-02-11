"""
Shared fixtures for CanopyRS tests.

These fixtures provide reusable test data across all test modules.
"""

import pytest
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, box
from pathlib import Path
from unittest.mock import MagicMock

from canopyrs.engine.constants import Col, StateKey
from canopyrs.engine.data_state import DataState


# =============================================================================
# GeoDataFrame Fixtures
# =============================================================================

@pytest.fixture
def sample_polygon():
    """A simple polygon for testing."""
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])


@pytest.fixture
def sample_gdf_minimal(sample_polygon):
    """Minimal GeoDataFrame with just geometry and object_id."""
    return gpd.GeoDataFrame({
        Col.GEOMETRY: [sample_polygon],
        Col.OBJECT_ID: [1],
    }, crs="EPSG:4326")


@pytest.fixture
def sample_gdf_with_tile_path(sample_polygon):
    """GeoDataFrame with geometry, object_id, and tile_path."""
    return gpd.GeoDataFrame({
        Col.GEOMETRY: [sample_polygon, sample_polygon],
        Col.OBJECT_ID: [1, 2],
        Col.TILE_PATH: ["/path/to/tile1.tif", "/path/to/tile2.tif"],
    }, crs="EPSG:4326")


@pytest.fixture
def sample_gdf_with_detector_score(sample_gdf_with_tile_path):
    """GeoDataFrame with detector score column."""
    gdf = sample_gdf_with_tile_path.copy()
    gdf[Col.DETECTOR_SCORE] = [0.95, 0.87]
    gdf[Col.DETECTOR_CLASS] = [0, 1]
    return gdf


@pytest.fixture
def sample_gdf_with_segmenter_score(sample_gdf_with_tile_path):
    """GeoDataFrame with segmenter score column."""
    gdf = sample_gdf_with_tile_path.copy()
    gdf[Col.SEGMENTER_SCORE] = [0.92, 0.88]
    gdf[Col.SEGMENTER_CLASS] = [0, 1]
    return gdf


@pytest.fixture
def sample_gdf_full(sample_polygon):
    """Complete GeoDataFrame with all common columns."""
    return gpd.GeoDataFrame({
        Col.GEOMETRY: [sample_polygon, sample_polygon, sample_polygon],
        Col.OBJECT_ID: [1, 2, 3],
        Col.TILE_PATH: ["/path/tile1.tif", "/path/tile2.tif", "/path/tile3.tif"],
        Col.DETECTOR_SCORE: [0.95, 0.87, 0.91],
        Col.DETECTOR_CLASS: [0, 1, 0],
        Col.SEGMENTER_SCORE: [0.92, 0.88, 0.90],
        Col.SEGMENTER_CLASS: [0, 1, 0],
    }, crs="EPSG:4326")


# =============================================================================
# DataState Fixtures
# =============================================================================

@pytest.fixture
def empty_data_state():
    """Empty DataState for testing initial conditions."""
    return DataState()


@pytest.fixture
def data_state_with_imagery(tmp_path):
    """DataState with imagery path set."""
    return DataState(
        imagery_path=str(tmp_path / "test_image.tif"),
        parent_output_path=str(tmp_path / "output"),
        product_name="test_image",
    )


@pytest.fixture
def data_state_with_tiles(tmp_path):
    """DataState with tiles path set."""
    return DataState(
        tiles_path=str(tmp_path / "tiles"),
        parent_output_path=str(tmp_path / "output"),
        product_name="tiled_input",
    )


@pytest.fixture
def data_state_with_gdf(sample_gdf_with_tile_path, tmp_path):
    """DataState with infer_gdf populated."""
    state = DataState(
        imagery_path=str(tmp_path / "test_image.tif"),
        parent_output_path=str(tmp_path / "output"),
        product_name="test_image",
    )
    state.infer_gdf = sample_gdf_with_tile_path
    state.infer_gdf_columns_to_pass = {Col.DETECTOR_SCORE}
    return state


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def temp_output_path(tmp_path):
    """Temporary output directory for test artifacts."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def temp_tiles_path(tmp_path):
    """Temporary tiles directory."""
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir(exist_ok=True)
    return tiles_dir


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_model():
    """Mock model for testing components without loading actual models."""
    model = MagicMock()
    model.infer.return_value = (
        ["/path/tile1.tif", "/path/tile2.tif"],  # tiles_paths
        [[0.9, 0.1], [0.2, 0.8]],  # scores
        [0, 1],  # predictions
    )
    return model


# =============================================================================
# Test Raster Fixtures (local asset, no download)
# =============================================================================

# Path to the test raster included in the repo
TEST_RASTER_PATH = Path(__file__).parent.parent / "assets" / "20240130_zf2tower_m3m_rgb_test_crop.tif"

# Cache directory for generated test artifacts (tiles, etc.)
TEST_DATA_CACHE = Path.home() / ".cache" / "canopyrs_test_data"


@pytest.fixture(scope="session")
def test_raster():
    """
    Path to the test raster (included in repo under assets/).

    This is a small (~24MB) real orthomosaic crop with actual trees,
    suitable for end-to-end pipeline testing without any downloads.
    """
    if not TEST_RASTER_PATH.exists():
        pytest.skip(f"Test raster not found: {TEST_RASTER_PATH}")
    return TEST_RASTER_PATH


@pytest.fixture(scope="session")
def test_raster_tiles(test_raster):
    """
    Tiled version of the test raster for pipeline tests.

    Tiles are cached between sessions to avoid re-tilerizing.
    """
    from canopyrs.engine.components.tilerizer import TilerizerComponent
    from canopyrs.engine.config_parsers.tilerizer import TilerizerConfig

    cache_path = TEST_DATA_CACHE / "test_raster_tiles"
    tiles_marker = cache_path / "_tiles_done"

    # Check if already tilerized
    if tiles_marker.exists():
        # Find the tiles directory created by the tilerizer
        tile_files = list(cache_path.glob("**/*.tif"))
        if tile_files:
            return cache_path

    cache_path.mkdir(parents=True, exist_ok=True)

    config = TilerizerConfig(
        tile_size=512,
        tile_overlap=0.25,
        tile_type="tile",
    )

    result = TilerizerComponent.run_standalone(
        config=config,
        imagery_path=str(test_raster),
        output_path=str(cache_path),
    )

    # Mark as done for cache check
    tiles_marker.touch()

    return Path(result.tiles_path)


# =============================================================================
# Synthetic Raster Fixture (fast, no download)
# =============================================================================

@pytest.fixture
def synthetic_raster(tmp_path):
    """
    Create a small synthetic raster for fast unit tests.

    256x256 RGB image with random data.
    """
    import numpy as np

    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        pytest.skip("rasterio not installed")

    raster_path = tmp_path / "synthetic_raster.tif"

    # Create random RGB data
    data = np.random.randint(0, 255, (3, 256, 256), dtype=np.uint8)

    # Simple transform: 1 unit per pixel, origin at (0, 0)
    transform = from_bounds(0, 0, 256, 256, 256, 256)

    with rasterio.open(
        raster_path, 'w',
        driver='GTiff',
        height=256,
        width=256,
        count=3,
        dtype='uint8',
        crs='EPSG:32618',
        transform=transform
    ) as dst:
        dst.write(data)

    return raster_path


@pytest.fixture
def synthetic_labels(tmp_path):
    """
    Create synthetic polygon labels matching synthetic_raster.

    Returns path to a GeoPackage with sample tree crown polygons.
    """
    labels = gpd.GeoDataFrame({
        'geometry': [
            box(10, 10, 30, 30),
            box(50, 50, 80, 80),
            box(100, 100, 140, 140),
            box(180, 180, 220, 220),
        ],
        'class': [0, 0, 0, 0],
    }, crs="EPSG:32618")

    labels_path = tmp_path / "synthetic_labels.gpkg"
    labels.to_file(labels_path, driver="GPKG")

    return labels_path
