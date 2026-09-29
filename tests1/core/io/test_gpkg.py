import json
import warnings

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Point, box

from canopyrs1.core.constants import Col, GeomKind
from canopyrs1.core.io.gpkg import write_gpkg
from canopyrs1.core.tables.imagery import Sources
from canopyrs1.core.tables.objects import Objects


def _objects():
    """Three objects in UTM zone 18N: a box, a point and a two-part mask, with a score each and
    the score of every class."""
    return Objects.build(
        geometry=[
            box(0, 0, 10, 10),
            Point(5, 5),
            MultiPolygon([box(20, 20, 30, 30), box(40, 40, 50, 50)]),
        ],
        geom_kind=[GeomKind.BOX, GeomKind.POINT, GeomKind.MASK],
        crs="EPSG:32618",
        columns={
            Col.DETECTOR_SCORE: [0.9, 0.8, 0.7],
            Col.CLASSIFIER_SCORES: [[0.1, 0.9], [0.5, 0.5], None],
        },
    )


def test_objects_written_and_read_back(tmp_path):
    objects = _objects()
    path = write_gpkg(objects.df, tmp_path / "objects.gpkg")
    assert path == tmp_path / "objects.gpkg"
    back = Objects.from_file(path)
    assert back.df.crs == "EPSG:32618"
    assert list(back.df[Col.GEOM_KIND]) == [GeomKind.BOX, GeomKind.POINT, GeomKind.MASK]
    assert list(back.df[Col.DETECTOR_SCORE]) == [0.9, 0.8, 0.7]
    for got, expected in zip(back.df.geometry, objects.df.geometry):
        assert got.equals(expected)


def test_lists_dicts_and_arrays_are_written_as_json(tmp_path):
    gdf = gpd.GeoDataFrame(
        {
            "a_list": [[0.1, 0.9], None, [1, 2]],
            "a_dict": [{"oak": 0.7}, {"pine": 0.3}, None],
            # As read back from parquet, with numpy numbers inside.
            "an_array": [np.array([0.25, 0.75], dtype=np.float32), None, np.array([1, 2])],
            "a_number": [1, 2, 3],
        },
        geometry=[Point(0, 0)] * 3,
        crs="EPSG:32618",
    )
    back = gpd.read_file(write_gpkg(gdf, tmp_path / "nested.gpkg"))
    assert json.loads(back["a_list"][0]) == [0.1, 0.9] and back["a_list"].isna()[1]
    assert json.loads(back["a_dict"][0]) == {"oak": 0.7}
    assert json.loads(back["an_array"][0]) == [0.25, 0.75]
    assert json.loads(back["an_array"][2]) == [1, 2]
    assert list(back["a_number"]) == [1, 2, 3]


def test_the_geodataframe_given_is_left_as_it_was(tmp_path):
    objects = _objects()
    write_gpkg(objects.df, tmp_path / "objects.gpkg")
    assert objects.df[Col.CLASSIFIER_SCORES][0] == [0.1, 0.9]


def test_its_folder_is_created(tmp_path):
    path = write_gpkg(_objects().df, tmp_path / "run" / "export" / "objects.gpkg")
    assert path.exists()


def test_a_file_already_there_is_replaced(tmp_path):
    path = tmp_path / "objects.gpkg"
    write_gpkg(_objects().df, path)
    write_gpkg(_objects().df.iloc[:1], path)
    assert len(gpd.read_file(path)) == 1


def test_objects_in_pixel_coordinates(tmp_path):
    gdf = gpd.GeoDataFrame({"k": [1]}, geometry=[box(0, 0, 10, 10)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = write_gpkg(gdf, tmp_path / "pixels.gpkg")
    messages = [str(w.message) for w in caught]
    assert messages == [f"{path} is written without a CRS: its geometries are in pixel coordinates"]
    assert gpd.read_file(path).crs is None


def test_no_objects(tmp_path):
    empty = Objects.build(geometry=[], geom_kind=GeomKind.BOX, crs="EPSG:32618")
    path = write_gpkg(empty.df, tmp_path / "empty.gpkg")
    assert len(gpd.read_file(path)) == 0


def test_the_real_crop_s_objects(real_raster, tmp_path):
    # Objects over the real crop, in its CRS, keep their coordinates to the millimetre.
    sources = Sources.from_paths(real_raster)
    objects = Objects.from_imagery(sources)
    footprint = objects.get_geometry_in_image_coords()
    gdf = gpd.GeoDataFrame({"k": [1]}, geometry=footprint.to_list(), crs=sources.df.crs)
    back = gpd.read_file(write_gpkg(gdf, tmp_path / "real.gpkg"))
    assert back.crs == sources.df.crs
    assert back.geometry[0].hausdorff_distance(footprint[0]) < 1e-3


def test_a_geodataframe_with_another_geometry_column_name(tmp_path):
    gdf = gpd.GeoDataFrame({"k": [1]}, geometry=[Point(1, 2)], crs="EPSG:32618")
    gdf = gdf.rename_geometry("geom")
    back = Objects.from_file(write_gpkg(gdf, tmp_path / "geom.gpkg"))
    assert back.df.geometry[0].equals(Point(1, 2))


NESTED_VALUES = [[1, 2], {"a": 1}]


@pytest.mark.parametrize("value", NESTED_VALUES)
def test_a_column_holding_only_nested_values(value, tmp_path):
    gdf = gpd.GeoDataFrame({"v": [value]}, geometry=[Point(0, 0)], crs="EPSG:32618")
    back = gpd.read_file(write_gpkg(gdf, tmp_path / "only.gpkg"))
    assert json.loads(back["v"][0]) == value
