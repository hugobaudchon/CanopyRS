import numpy as np
import pytest
import rasterio

from canopyrs1.core.geometry.georef import read_georef, window_georef
from canopyrs1.core.raster.read import get_alpha_band, read_window, should_skip


def _tile(empty_fraction, *, value=0, size=20, bands=3):
    """Return a (bands, size, size) uint8 tile of grey pixels whose first rows are set to
    ``value`` in every band, covering ``empty_fraction`` of the tile."""
    tile = np.full((bands, size, size), 120, np.uint8)
    tile[:, : round(empty_fraction * size)] = value
    return tile


CONDITIONS = {"ignore_black_white_alpha_tiles_threshold": 0.75}


@pytest.mark.parametrize("conditions", [None, {}])
def test_no_conditions_never_skip(conditions):
    assert not should_skip(_tile(1.0), conditions)


def test_the_fraction_of_empty_pixels():
    assert should_skip(_tile(0.8), CONDITIONS)
    assert should_skip(_tile(0.75), CONDITIONS)  # at the limit, it is skipped
    assert not should_skip(_tile(0.7), CONDITIONS)
    assert not should_skip(_tile(0.8), {"ignore_black_white_alpha_tiles_threshold": 0.85})


@pytest.mark.parametrize("threshold", [1.0, 1.5])
def test_a_threshold_of_1_or_more_never_skips(threshold):
    assert not should_skip(_tile(1.0), {"ignore_black_white_alpha_tiles_threshold": threshold})


def test_white_counts_as_empty():
    assert should_skip(_tile(0.8, value=255), CONDITIONS)
    tile = _tile(0.4, value=255)
    tile[:, 8:16] = 0  # 40% white and 40% black
    assert should_skip(tile, CONDITIONS)


def test_a_pixel_is_only_empty_in_every_band():
    tile = np.zeros((3, 20, 20), np.uint8)
    tile[1] = 50  # dark green: 0 in red and blue only
    assert not should_skip(tile, {"ignore_black_white_alpha_tiles_threshold": 0.01})


def test_transparent_pixels_count_as_empty():
    tile = _tile(0.0)
    alpha = np.full((20, 20), 255, np.uint8)
    alpha[:16] = 0  # 80% transparent
    assert should_skip(tile, CONDITIONS, alpha=alpha)
    assert not should_skip(tile, CONDITIONS)


def test_any_number_of_bands():
    assert should_skip(_tile(0.8, bands=1), CONDITIONS)
    assert should_skip(_tile(0.8, bands=5), CONDITIONS)
    assert not should_skip(_tile(0.5, bands=5), CONDITIONS)


def test_unknown_conditions_are_refused():
    with pytest.raises(ValueError, match="ignore_black_white_tiles_threshold"):
        should_skip(_tile(0.5), {"ignore_black_white_tiles_threshold": 0.75})


def test_get_alpha_band(rgb_raster, rgba_raster, real_raster):
    for path, alpha_band in [(rgb_raster, None), (rgba_raster, 4), (real_raster, 4)]:
        with rasterio.open(path) as src:
            assert get_alpha_band(src) == alpha_band


def test_tiles_of_a_raster_with_a_transparent_half(rgba_raster):
    # The left half of rgba_raster is transparent: tiles there are skipped, those on the right
    # are kept, and one straddling the middle is kept at 75% only if under half of it is empty.
    with rasterio.open(rgba_raster) as src:
        raster = read_georef(src)
        alpha_band = get_alpha_band(src)

        def skipped(col, threshold=0.75):
            georef = window_georef(raster, col_off=col, row_off=0, width=64, height=64)
            pixels = read_window(src, georef)
            alpha = pixels[alpha_band - 1]
            colours = np.delete(pixels, alpha_band - 1, axis=0)
            conditions = {"ignore_black_white_alpha_tiles_threshold": threshold}
            return should_skip(colours, conditions, alpha=alpha)

        assert skipped(0) and skipped(64)
        assert not skipped(128) and not skipped(192)
        assert not skipped(96)  # 50% transparent
        assert skipped(96, threshold=0.5)


def test_pixels_outside_the_raster_count_as_empty(rgb_raster):
    with rasterio.open(rgb_raster) as src:
        georef = window_georef(read_georef(src), col_off=-48, row_off=0, width=64, height=64)
        assert should_skip(read_window(src, georef), CONDITIONS)  # 75% outside
