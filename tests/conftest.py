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
# Test Data Download Fixtures (for integration tests)
# =============================================================================

# Cache directory for downloaded test data
TEST_DATA_CACHE = Path.home() / ".cache" / "canopyrs_test_data"


@pytest.fixture(scope="session")
def bci50ha_small_labels():
    """
    Download just the BCI50ha improved labels (25MB) for unit tests.

    This is much smaller than the full dataset and doesn't require
    downloading multi-GB raster files.
    """
    import requests
    import zipfile

    cache_path = TEST_DATA_CACHE / "unit_tests"
    labels_dir = cache_path / "bci50ha_labels"
    labels_file = labels_dir / "BCI_50ha_2020_08_01_crownmap_improved.gpkg"

    # Check if already downloaded
    if labels_file.exists():
        return labels_dir

    try:
        cache_path.mkdir(parents=True, exist_ok=True)

        # Download just the improved labels zip (25MB)
        zip_url = "https://ndownloader.figshare.com/files/43628040"
        zip_path = cache_path / "labels.zip"

        print("Downloading BCI50ha labels for unit tests (25MB)...")
        response = requests.get(zip_url, stream=True)
        response.raise_for_status()

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract
        print("Extracting labels...")
        labels_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(labels_dir)

        # Clean up zip
        zip_path.unlink()

        # Find and move the gpkg to the root
        gpkg_files = list(labels_dir.glob("**/*.shp"))
        if gpkg_files:
            # Convert shp to gpkg
            import geopandas as gpd
            gdf = gpd.read_file(gpkg_files[0])
            gdf.to_file(labels_file, driver="GPKG")

        return labels_dir
    except Exception as e:
        pytest.skip(f"Could not download BCI50ha labels: {e}")


@pytest.fixture(scope="session")
def bci50ha_raw_data():
    """
    Download and cache BCI50ha 2020 dataset for integration tests.

    Only downloads 2020 raster and labels (not 2022) to save time and bandwidth.
    Returns the path to the raw data, or skips if unavailable.
    This fixture is session-scoped so download happens only once.
    """
    import requests
    import zipfile
    import shutil

    cache_path = TEST_DATA_CACHE / "raw"
    dataset_path = cache_path / "panama_bci50ha"

    # Check if already downloaded
    raster_file = dataset_path / "BCI_50ha_2020_08_01_crownmap_raw.tif"
    labels_file = dataset_path / "BCI_50ha_2020_08_01_crownmap_improved.gpkg"
    if raster_file.exists() and labels_file.exists():
        return dataset_path

    # Download only 2020 data using Figshare API
    try:
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Get file metadata from Figshare API
        article_id = "24784053"
        api_url = f"https://api.figshare.com/v2/articles/{article_id}"
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        # Download only 2020 files
        files_to_download = [
            'BCI_50ha_2020_08_01_crownmap_raw.zip',
            'BCI_50ha_2020_08_01_crownmap_improved.zip',
        ]

        for file_info in data.get('files', []):
            file_name = file_info['name']
            if file_name in files_to_download:
                print(f"Downloading {file_name}...")
                download_url = file_info['download_url']
                local_file = dataset_path / file_name

                # Download with progress
                headers = {'User-Agent': 'Mozilla/5.0'}
                resp = requests.get(download_url, stream=True, headers=headers)
                resp.raise_for_status()
                with open(local_file, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Extract
                print(f"Extracting {file_name}...")
                with zipfile.ZipFile(local_file, 'r') as zf:
                    zf.extractall(dataset_path)
                local_file.unlink()

        # Find and rename raw raster
        raw_files = list(dataset_path.glob('**/BCI_50ha_2020_08_01_global.tif'))
        if raw_files:
            raw_files[0].rename(raster_file)

        # Find and convert improved shapefile to gpkg
        shp_files = list(dataset_path.glob('**/BCI_50ha_2020_08_01_crownmap_improved.shp'))
        if shp_files:
            gdf = gpd.read_file(shp_files[0])
            gdf.to_file(labels_file, driver='GPKG')

        # Clean up extracted directories
        for dir_name in ['BCI_50ha_2020_08_01_crownmap_improved', 'BCI_50ha_2020_08_01_crownmap_raw']:
            dir_path = dataset_path / dir_name
            if dir_path.exists():
                shutil.rmtree(dir_path)

        return dataset_path
    except Exception as e:
        pytest.skip(f"Could not download BCI50ha test data: {e}")


@pytest.fixture(scope="session")
def bci50ha_tiles(bci50ha_raw_data):
    """
    Create tiled version of BCI50ha 2020 data for pipeline tests.

    Only tilerizes 2020 data to match what bci50ha_raw_data downloads.
    Uses a small tile size for faster tests.
    Session-scoped so tilerization happens only once.
    """
    import json
    from geodataset.aoi import AOIFromPackageConfig
    from canopyrs.data.detection.tilerize import tilerize_with_overlap

    output_path = TEST_DATA_CACHE / "tiles"
    tiles_path = output_path / "panama_bci50ha"

    # Check if already tilerized (look for 2020 tiles specifically)
    test_tiles_dir = tiles_path / "BCI_50ha_2020_08_01_crownmap" / "test"
    if test_tiles_dir.exists() and any(test_tiles_dir.glob("*.tif")):
        return tiles_path

    try:
        output_path.mkdir(parents=True, exist_ok=True)

        # Get AOI and categories paths from BCI50ha dataset
        from canopyrs.data.detection.raw_datasets.BCI50ha.bci50ha import BCI50haDataset
        dataset = BCI50haDataset()

        # Only process 2020 data
        raster_name = "BCI_50ha_2020_08_01_crownmap"
        raster_path = bci50ha_raw_data / f"{raster_name}_raw.tif"
        labels_path = bci50ha_raw_data / f"{raster_name}_improved.gpkg"

        # Check files exist
        if not raster_path.exists() or not labels_path.exists():
            pytest.skip(f"Required files not found: {raster_path}, {labels_path}")

        # Load AOI for test fold
        # The paths in dataset.aois are absolute (created from parent_folder in bci50ha.py)
        aoi_path = dataset.aois[raster_name]["test"]
        aois_gdf = gpd.read_file(str(aoi_path))
        aois_config = AOIFromPackageConfig({"test": aois_gdf})

        # Determine label column name
        labels_gdf = gpd.read_file(labels_path)
        if 'Latin' in labels_gdf.columns:
            main_label_category_column_name = 'Latin'
        else:
            main_label_category_column_name = 'latin'

        # Get categories path (also absolute from parent_folder)
        categories_path = dataset.categories

        # Tilerize just the 2020 data
        tilerize_with_overlap(
            raster_path=raster_path,
            labels=labels_path,
            main_label_category_column_name=main_label_category_column_name,
            coco_categories_list=json.load(open(categories_path, 'rb'))['categories'],
            aois_config=aois_config,
            output_path=tiles_path,
            ground_resolution=0.1,
            scale_factor=None,
            tile_size=1777,
            tile_overlap=0.25
        )

        return tiles_path
    except Exception as e:
        pytest.skip(f"Could not tilerize BCI50ha test data: {e}")


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
