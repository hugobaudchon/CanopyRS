"""
Fixtures for benchmark testing.

Provides sample GeoDataFrames for testing geometry processing and evaluation.
"""

import pytest
import geopandas as gpd
from shapely.geometry import Polygon, box


# =============================================================================
# AOI Fixtures
# =============================================================================

@pytest.fixture
def aoi_polygon():
    """Area of interest polygon for filtering tests."""
    return box(0, 0, 100, 100)


@pytest.fixture
def aoi_gdf(aoi_polygon):
    """Area of interest as GeoDataFrame."""
    return gpd.GeoDataFrame({
        'geometry': [aoi_polygon]
    }, crs="EPSG:32618")


# =============================================================================
# Invalid Geometry Fixtures
# =============================================================================

@pytest.fixture
def gdf_with_invalid_geometry():
    """GeoDataFrame with an invalid (self-intersecting) geometry."""
    # Bowtie polygon (self-intersecting)
    invalid_poly = Polygon([(0, 0), (10, 10), (10, 0), (0, 10), (0, 0)])
    valid_poly = box(20, 20, 30, 30)

    return gpd.GeoDataFrame({
        'geometry': [invalid_poly, valid_poly],
        'id': [1, 2],
    }, crs="EPSG:32618")


# =============================================================================
# CRS Fixtures
# =============================================================================

@pytest.fixture
def gdf_geographic_crs():
    """GeoDataFrame in geographic CRS (lat/lon) for reprojection testing."""
    return gpd.GeoDataFrame({
        'geometry': [
            box(-75.5, 40.0, -75.4, 40.1),  # ~11km x ~11km area
        ],
        'id': [1],
    }, crs="EPSG:4326")


@pytest.fixture
def gdf_projected_crs():
    """GeoDataFrame in projected CRS (UTM)."""
    return gpd.GeoDataFrame({
        'geometry': [
            box(500000, 4400000, 510000, 4410000),  # 10km x 10km
        ],
        'id': [1],
    }, crs="EPSG:32618")
