"""to_uint8 gives the same pixels as geodataset's tile writer with output_dtype='uint8'."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from geodataset.geodata import Raster

from canopyrs1.core.raster.conversion import to_uint8


def _random(dtype, rng):
    """Return random (3, 32, 32) pixels covering the usual range of ``dtype``."""
    if dtype == "float01":
        return rng.random((3, 32, 32)).astype(np.float32)
    if dtype == "float255":
        return (rng.random((3, 32, 32)) * 255).astype(np.float32)
    info = np.iinfo(dtype)
    return rng.integers(max(info.min, -1000), min(info.max, 70000), (3, 32, 32)).astype(dtype)


DTYPES = ["float01", "float255", np.uint16, np.int16, np.int32]


@pytest.mark.parametrize("dtype", DTYPES)
def test_same_pixels_as_geodataset(dtype, tmp_path):
    pixels = _random(dtype, np.random.default_rng(0))
    path = tmp_path / "raster.tif"
    dst = rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=3,
        dtype=pixels.dtype,
        crs="EPSG:32618",
        transform=from_origin(0, 32, 1, 1),
    )
    with dst:
        dst.write(pixels)
    tile = Raster(str(path), temp_dir=tmp_path).create_tile_metadata(
        Window(0, 0, 32, 32), tile_id=1
    )
    (tmp_path / "out").mkdir()
    tile.save(output_folder=tmp_path / "out", output_dtype="uint8")
    (saved,) = (tmp_path / "out").glob("*.tif")
    with rasterio.open(saved) as old:
        np.testing.assert_array_equal(to_uint8(pixels), old.read())
