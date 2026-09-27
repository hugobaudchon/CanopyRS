"""A tile written by write_tile is the same file as geodataset's: same pixels, georef and colour
tags."""

import numpy as np
import pytest
import rasterio
from rasterio.windows import Window

from geodataset.geodata import Raster

from canopyrs1.core.geometry.georef import read_georef, window_georef
from canopyrs1.core.raster.read import read_window
from canopyrs1.core.raster.write import write_tile

WINDOWS = [(0, 0, 64, 64), (100, 30, 50, 70), (-20, -10, 64, 64)]


@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("raster", ["rgb_raster", "rgba_raster"])
def test_same_file_as_geodataset(raster, window, request, tmp_path):
    path = request.getfixturevalue(raster)
    col, row, width, height = window
    old_folder, new_folder = tmp_path / "old", tmp_path / "new"
    old_folder.mkdir()
    new_folder.mkdir()

    old_tile = Raster(str(path), temp_dir=tmp_path).create_tile_metadata(
        window=Window(col, row, width, height),
        tile_id=1,
    )
    old_tile.save(output_folder=old_folder)
    (old_path,) = old_folder.glob("*.tif")

    with rasterio.open(path) as src:
        georef = window_georef(
            read_georef(src),
            col_off=col,
            row_off=row,
            width=width,
            height=height,
        )
        new_path = write_tile(
            new_folder / "tile.tif",
            read_window(src, georef),
            georef,
            colorinterp=src.colorinterp,
        )

    with rasterio.open(old_path) as old, rasterio.open(new_path) as new:
        np.testing.assert_array_equal(new.read(), old.read())
        assert read_georef(new) == read_georef(old)
        assert new.colorinterp == old.colorinterp
