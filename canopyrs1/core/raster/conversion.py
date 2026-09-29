"""Converting pixels of other dtypes to uint8, the dtype the models expect."""

import numpy as np


def to_uint8(pixels: np.ndarray, nodata: float | None = None) -> np.ndarray:
    """Return ``pixels`` (bands, height, width) as uint8, decided from these pixels alone:

    - uint8: returned as they are;
    - uint16: divided by 256 (the full 0-65535 range becomes 0-255);
    - int16: negative values become 0, the rest is divided by 128;
    - int32, int64 and uint32: taken as 0-65535 values, scaled to 0-255, and ``nodata`` pixels
      become 0;
    - floats: if the largest value other than NaN is above 1, the values are taken as 0-255 and
      clipped; otherwise as 0-1, and multiplied by 255 and rounded. NaN becomes 255.

    Raises a ValueError for any other dtype.
    """
    dtype = pixels.dtype
    if dtype == np.uint8:
        return pixels
    if dtype == np.uint16:
        return (pixels >> 8).astype(np.uint8)
    if dtype == np.int16:
        return (np.clip(pixels, 0, None) >> 7).astype(np.uint8)
    if dtype in (np.int32, np.int64, np.uint32):
        scaled = (np.clip(pixels, 0, 65535).astype(np.float64) / 65535 * 255).astype(np.uint8)
        if nodata is not None:
            scaled[pixels == nodata] = 0
        return scaled
    if np.issubdtype(dtype, np.floating):
        known = pixels[~np.isnan(pixels)]
        if known.size and known.max() > 1:
            return np.clip(np.nan_to_num(pixels, nan=255.0), 0, 255).astype(np.uint8)
        return np.rint(np.clip(np.nan_to_num(pixels, nan=1.0), 0, 1) * 255).astype(np.uint8)
    raise ValueError(f"Can't convert {dtype} pixels to uint8")
