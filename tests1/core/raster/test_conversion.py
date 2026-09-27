import numpy as np
import pytest

from canopyrs1.core.raster.conversion import to_uint8


def _pixels(values, dtype):
    """Return ``values`` as a (1, 1, n) array of ``dtype``."""
    return np.array(values, dtype=dtype).reshape(1, 1, -1)


def test_uint8_is_returned_as_it_is():
    pixels = _pixels([0, 7, 255], np.uint8)
    assert to_uint8(pixels) is pixels


INTEGER_CASES = [
    (np.uint16, [0, 255, 256, 511, 65535], [0, 0, 1, 1, 255]),
    (np.int16, [-5, 0, 127, 128, 256, 32767], [0, 0, 0, 1, 2, 255]),
    (np.int32, [-5, 0, 1000, 65535, 100000], [0, 0, 3, 255, 255]),
    (np.uint32, [0, 32768, 65535], [0, 127, 255]),
]


@pytest.mark.parametrize("dtype, values, expected", INTEGER_CASES)
def test_integers(dtype, values, expected):
    converted = to_uint8(_pixels(values, dtype))
    assert converted.dtype == np.uint8
    assert converted.ravel().tolist() == expected


def test_nodata_pixels_of_wide_integers_become_0():
    pixels = _pixels([65535, 12, 65535], np.int32)
    assert to_uint8(pixels, nodata=65535).ravel().tolist() == [0, 0, 0]
    assert to_uint8(pixels).ravel().tolist() == [255, 0, 255]


FLOAT_CASES = [
    ([0.0, 0.002, 0.5, 0.998, 1.0], [0, 1, 128, 254, 255]),  # 0 to 1
    ([0.0, 1.5, 128.7, 255.0, 300.0, -4.0], [0, 1, 128, 255, 255, 0]),  # 0 to 255
    ([-0.5, 0.25], [0, 64]),  # 0 to 1, below 0 clipped
    ([np.nan, 0.5], [255, 128]),  # NaN doesn't make a 0 to 1 tile look like 0 to 255
    ([np.nan, 100.0], [255, 100]),
]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("values, expected", FLOAT_CASES)
def test_floats(dtype, values, expected):
    converted = to_uint8(_pixels(values, dtype))
    assert converted.dtype == np.uint8
    assert converted.ravel().tolist() == expected


def test_the_shape_is_kept():
    pixels = np.random.default_rng(0).random((4, 8, 16)).astype(np.float32)
    assert to_uint8(pixels).shape == (4, 8, 16)


def test_other_dtypes_are_refused():
    with pytest.raises(ValueError, match="complex64"):
        to_uint8(_pixels([1 + 2j], np.complex64))
