"""The SelvaMask GeoPackages, read and written again, keep every row, column and coordinate."""

import geopandas as gpd
import pytest
import shapely

from canopyrs1.core.constants import Col, GeomKind
from canopyrs1.core.io.gpkg import write_gpkg
from canopyrs1.core.tables.objects import Objects

pytestmark = pytest.mark.integration

# The columns every objects table has, which the original files don't.
TABLE_COLUMNS = [Col.OBJECT_ID, Col.GEOM_KIND, Col.PARENT_IMAGE_ID, Col.PARENT_OBJECT_ID]


def _assert_same_rows(got, expected, columns):
    """Check that the GeoDataFrames ``got`` and ``expected`` have the same CRS, the same values
    in ``columns``, and exactly the same geometries, row by row."""
    assert got.crs == expected.crs
    assert len(got) == len(expected)
    for column in columns:
        assert got[column].tolist() == expected[column].tolist(), column
    # The same bytes: every coordinate kept as it was, invalid geometries included.
    assert (shapely.to_wkb(got.geometry.values) == shapely.to_wkb(expected.geometry.values)).all()


def test_read_then_saved_again(selvamask_gpkgs, tmp_path):
    for path in selvamask_gpkgs:
        original = gpd.read_file(path)
        objects = Objects.from_file(path)
        assert (objects.df[Col.GEOM_KIND] == GeomKind.MASK).all()

        # Saved, then read back: the original rows and columns, plus the table's own columns.
        saved = gpd.read_file(write_gpkg(objects.df, tmp_path / f"once_{path.name}"))
        original_columns = [c for c in original.columns if c != "geometry"]
        _assert_same_rows(saved, original, original_columns)
        assert set(saved.columns) == set(original.columns) | set(TABLE_COLUMNS)

        # Read and saved a second time: exactly the first save.
        again = Objects.from_file(tmp_path / f"once_{path.name}")
        saved_again = gpd.read_file(write_gpkg(again.df, tmp_path / f"twice_{path.name}"))
        _assert_same_rows(saved_again, saved, [c for c in saved.columns if c != "geometry"])
