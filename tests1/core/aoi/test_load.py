import warnings

import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

from canopyrs1.core.aoi.load import load_aois
from canopyrs1.core.tables.imagery import Sources

UTM_18N = "EPSG:32618"


def _file(tmp_path, name, geometries, crs=UTM_18N):
    """Write ``geometries`` to the GeoPackage ``name`` in ``tmp_path``, and return its path."""
    path = tmp_path / f"{name}.gpkg"
    gpd.GeoDataFrame(geometry=geometries, crs=crs).to_file(path)
    return path


def test_aois_from_files(tmp_path):
    aois = load_aois(
        {
            "train": _file(tmp_path, "train", [box(0, 0, 100, 50)]),
            "valid": _file(tmp_path, "valid", [box(0, 50, 100, 100)]),
        },
        crs=UTM_18N,
    )
    assert list(aois["aoi"]) == ["train", "valid"]
    assert aois.crs == UTM_18N
    assert aois.geometry[0].equals(box(0, 0, 100, 50))


def test_the_polygons_of_an_aoi_become_one_area(tmp_path):
    parts = [box(0, 0, 10, 10), box(5, 5, 20, 20), box(50, 50, 60, 60)]
    aois = load_aois({"test": _file(tmp_path, "test", parts)}, crs=UTM_18N)
    assert len(aois) == 1
    area = aois.geometry[0]
    assert isinstance(area, MultiPolygon) and len(area.geoms) == 2
    assert area.area == pytest.approx(100 + 225 - 25 + 100)


def test_an_aoi_in_another_crs_is_moved_into_the_images(tmp_path):
    in_utm = box(600000, 5040000, 600100, 5040100)
    in_lat_lon = gpd.GeoSeries([in_utm], crs=UTM_18N).to_crs("EPSG:4326")
    aois = load_aois({"infer": _file(tmp_path, "infer", in_lat_lon, crs="EPSG:4326")}, UTM_18N)
    assert aois.geometry[0].hausdorff_distance(in_utm) < 1e-3


def test_a_geodataframe(tmp_path):
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs=UTM_18N)
    assert load_aois({"infer": gdf}, crs=UTM_18N).geometry[0].equals(box(0, 0, 1, 1))


def test_an_invalid_polygon_is_repaired(tmp_path):
    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])  # crossing itself
    aois = load_aois({"train": gpd.GeoDataFrame(geometry=[bowtie], crs=UTM_18N)}, UTM_18N)
    assert aois.geometry[0].is_valid and aois.geometry[0].area == pytest.approx(50)


def test_aois_in_pixel_coordinates():
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 512, 512)])
    aois = load_aois({"train": gdf}, crs=None)
    assert aois.crs is None


BAD_NAMES = ["train_1", "", "tést", "valid fold"]


@pytest.mark.parametrize("name", BAD_NAMES)
def test_names_hold_only_letters_and_digits(name):
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs=UTM_18N)
    with pytest.raises(ValueError, match="may only hold letters and digits"):
        load_aois({name: gdf}, crs=UTM_18N)


NOT_AREAS = [
    ([], "has no polygon"),
    ([LineString([(0, 0), (1, 1)])], "should only hold polygons, not LineString"),
    ([box(0, 0, 1, 1), Point(0, 0)], "should only hold polygons, not Point"),
    ([None], "should only hold polygons, not None"),
]


@pytest.mark.parametrize("geometries, message", NOT_AREAS)
def test_aois_that_aren_t_areas(geometries, message):
    gdf = gpd.GeoDataFrame(geometry=geometries, crs=UTM_18N)
    with pytest.raises(ValueError, match=message):
        load_aois({"train": gdf}, crs=UTM_18N)


def test_a_crs_on_one_side_only():
    with_crs = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs=UTM_18N)
    without = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)])
    with pytest.raises(ValueError, match="has to be in pixel coordinates, without a CRS"):
        load_aois({"train": with_crs}, crs=None)
    with pytest.raises(ValueError, match="has to be in a CRS"):
        load_aois({"train": without}, crs=UTM_18N)


def test_overlapping_aois(tmp_path):
    def gdf(geometry):
        return gpd.GeoDataFrame(geometry=[geometry], crs=UTM_18N)

    # Sharing an edge only is fine, even when rounding makes them overlap by a sliver.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_aois({"train": gdf(box(0, 0, 10, 10)), "valid": gdf(box(10, 0, 20, 10))}, UTM_18N)
        sliver = box(10 - 1e-9, 0, 20, 10)
        load_aois({"train": gdf(box(0, 0, 10, 10)), "valid": gdf(sliver)}, UTM_18N)
    # Sharing an area, or one inside the other, isn't.
    for valid in (box(5, 0, 15, 10), box(2, 2, 4, 4)):
        with pytest.warns(UserWarning, match="'train' and 'valid' overlap, on"):
            load_aois({"train": gdf(box(0, 0, 10, 10)), "valid": gdf(valid)}, UTM_18N)


def test_an_aoi_over_the_real_crop(real_raster, tmp_path):
    sources = Sources.from_paths(real_raster)
    footprint = sources.df.geometry[0]
    minx, miny, maxx, maxy = footprint.bounds
    left_half = box(minx, miny, (minx + maxx) / 2, maxy)
    in_lat_lon = gpd.GeoSeries([left_half], crs=sources.df.crs).to_crs("EPSG:4326")
    path = _file(tmp_path, "infer", in_lat_lon, crs="EPSG:4326")
    aois = load_aois({"infer": path}, crs=sources.df.crs)
    assert aois.geometry[0].intersection(footprint).area == pytest.approx(footprint.area / 2)
