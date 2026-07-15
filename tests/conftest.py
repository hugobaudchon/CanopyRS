"""
Shared fixtures for CanopyRS tests.

Pipeline-agnostic assets (rasters, labels) plus small relational-table builders
(Imagery / Objects) used by the engine unit tests.
"""

import pytest
import geopandas as gpd
from shapely.geometry import Polygon, box
from pathlib import Path

from canopyrs.engine.constants import Col, GeomKind, ImageKind
from canopyrs.engine.data import Imagery, Objects
from canopyrs.engine.utils import init_spawn_method

# The segmenter's mask post-processing (and the benchmark grid-search) use multiprocessing; with a
# live CUDA context the default 'fork' start method deadlocks. Force 'spawn' once, at collection start
# — before any model/CUDA init — mirroring infer.py and the standalone smoke scripts. Harmless for the
# fast (no-multiprocessing) tests.
init_spawn_method()


# =============================================================================
# Geometry / path fixtures
# =============================================================================

@pytest.fixture
def sample_polygon():
    """A simple polygon for testing."""
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])


@pytest.fixture
def temp_output_path(tmp_path):
    """Temporary output directory for test artifacts."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# =============================================================================
# tile metadata helper + relational-table builders
# =============================================================================

def make_tile_metadata(*, width=64, height=64, gsd=1.0, x0=0.0, y0=0.0, crs="EPSG:32618"):
    """A serializable METADATA dict for an image whose top-left is (x0, y0) in CRS units,
    ``gsd`` units per pixel (north-up). Matches the shape produced by ``tilemeta.window_meta``."""
    return {
        "transform": [gsd, 0.0, x0, 0.0, -gsd, y0],
        "crs": crs,
        "width": width,
        "height": height,
        "dtype": "uint8",
        "count": 3,
        "nodata": None,
    }


@pytest.fixture
def sources_seed(tmp_path):
    """A single-raster kind='source' Imagery seed (path need not exist for wiring/persistence tests)."""
    return Imagery.from_paths(str(tmp_path / "product.tif"))


@pytest.fixture
def tiles_seed(sources_seed):
    """Two grid tiles over the seed source — children of the source (windows, no files on disk)."""
    metadata = [make_tile_metadata(x0=0.0, y0=64.0), make_tile_metadata(x0=64.0, y0=64.0)]
    return Imagery.build(kind=ImageKind.TILE,
                         parent_id=sources_seed.df[Col.IMAGE_ID].iloc[0],
                         metadata=metadata, parent=sources_seed)


@pytest.fixture
def objects_seed(tiles_seed):
    """Two detector-style box Objects in tile-pixel coords (crs=None), one per tile, with scores."""
    image_ids = list(tiles_seed.df[Col.IMAGE_ID])
    return Objects.build(
        geometry=[box(1, 1, 10, 10), box(5, 5, 20, 20)],
        geom_kind=GeomKind.BOX,
        image_id=image_ids,
        imagery=tiles_seed,
        **{Col.DETECTOR_SCORE: [0.9, 0.8], Col.DETECTOR_CLASS: [0, 1]},
    )


# =============================================================================
# Test raster fixtures (local asset, no download)
# =============================================================================

TEST_RASTER_PATH = Path(__file__).parent.parent / "assets" / "20240130_zf2tower_m3m_rgb_test_crop.tif"


@pytest.fixture(scope="session")
def test_raster():
    """Path to the small real orthomosaic crop bundled under assets/ (for e2e pipeline tests)."""
    if not TEST_RASTER_PATH.exists():
        pytest.skip(f"Test raster not found: {TEST_RASTER_PATH}")
    return TEST_RASTER_PATH


@pytest.fixture
def synthetic_raster(tmp_path):
    """A small (256x256 RGB uint8) synthetic raster for fast unit tests."""
    import numpy as np

    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        pytest.skip("rasterio not installed")

    raster_path = tmp_path / "synthetic_raster.tif"
    data = np.random.randint(0, 255, (3, 256, 256), dtype=np.uint8)
    transform = from_bounds(0, 0, 256, 256, 256, 256)
    with rasterio.open(
        raster_path, 'w', driver='GTiff', height=256, width=256, count=3,
        dtype='uint8', crs='EPSG:32618', transform=transform,
    ) as dst:
        dst.write(data)
        dst.colorinterp = [rasterio.enums.ColorInterp.red,
                           rasterio.enums.ColorInterp.green,
                           rasterio.enums.ColorInterp.blue]
    return raster_path


@pytest.fixture
def tiles_dir(synthetic_raster, tmp_path):
    """A folder of two pre-cut georeferenced GeoTIFF tiles (for Imagery.from_tiles_dir)."""
    import rasterio
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform

    out = tmp_path / "tiles"
    out.mkdir()
    with rasterio.open(synthetic_raster) as src:
        for i, window in enumerate([Window(0, 0, 128, 128), Window(128, 0, 128, 128)]):
            data = src.read(window=window)
            meta = src.meta.copy()
            meta.update(height=128, width=128, transform=window_transform(window, src.transform))
            with rasterio.open(out / f"tile_{i}.tif", 'w', **meta) as dst:
                dst.write(data)
    return out


@pytest.fixture
def synthetic_labels(tmp_path):
    """Synthetic polygon labels (CRS) matching synthetic_raster."""
    labels = gpd.GeoDataFrame({
        'geometry': [box(10, 10, 30, 30), box(50, 50, 80, 80),
                     box(100, 100, 140, 140), box(180, 180, 220, 220)],
        'class': [0, 0, 0, 0],
    }, crs="EPSG:32618")
    labels_path = tmp_path / "synthetic_labels.gpkg"
    labels.to_file(labels_path, driver="GPKG")
    return labels_path
