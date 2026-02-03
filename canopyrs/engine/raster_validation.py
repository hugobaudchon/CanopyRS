"""
Raster validation utilities for checking RGB band properties.

Provides functions to validate that raster files meet expected requirements
for RGB imagery processing (color interpretation, dtype, value ranges).
"""

import warnings
from pathlib import Path
from typing import Optional

import rasterio
from rasterio.enums import ColorInterp


class RasterValidationError(Exception):
    """Raised when raster validation fails."""
    pass


def validate_raster_rgb_bands(
    raster_path: Path,
    strict_color_interp: bool = True
) -> None:
    """
    Validate that the first 3 bands of a raster have correct RGB properties.

    Checks:
    - At least 3 bands exist
    - First 3 bands have color interpretation R, G, B (if strict_color_interp=True)
    - First 3 bands have dtype uint8
    - First 3 bands have values in 0-255 range (samples a 512x512 window)

    Args:
        raster_path: Path to the raster file to validate
        strict_color_interp: If True, raise error for incorrect color interpretation.
                            If False, only warn if color interpretation is not R,G,B.

    Raises:
        RasterValidationError: If validation fails
    """
    try:
        with rasterio.open(raster_path) as src:
            # Check that we have at least 3 bands
            if src.count < 3:
                raise RasterValidationError(
                    f"Raster must have at least 3 bands, but has {src.count} bands. "
                    f"File: {raster_path}"
                )

            # Check color interpretation for first 3 bands
            expected_colors = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]
            color_names = ['Red', 'Green', 'Blue']

            for i in range(3):
                band_idx = i + 1  # rasterio uses 1-based indexing
                actual_color = src.colorinterp[i]
                expected_color = expected_colors[i]

                if actual_color != expected_color:
                    error_msg = (
                        f"Band {band_idx} should have color interpretation '{color_names[i]}', "
                        f"but has '{actual_color.name}'. File: {raster_path}"
                    )

                    if strict_color_interp:
                        # Strict mode: raise error but mention the flag
                        raise RasterValidationError(
                            f"{error_msg}\n"
                            f"If you are certain your raster bands are RGB-ordered, you can disable "
                            f"this check by setting strict_rgb_validation=False."
                        )
                    else:
                        # Permissive mode: only warn
                        warnings.warn(
                            f"{error_msg}\n"
                            f"Continuing anyway because strict_rgb_validation=False. "
                            f"Ensure your first 3 bands are RGB-ordered.",
                            UserWarning
                        )

            # Check dtype for first 3 bands
            for i in range(3):
                band_idx = i + 1
                dtype = src.dtypes[i]

                if dtype != 'uint8':
                    raise RasterValidationError(
                        f"Band {band_idx} should have dtype 'uint8', but has '{dtype}'. "
                        f"File: {raster_path}"
                    )

            # Check value range by reading a sample window from first 3 bands
            # Read a small window to check values (center 512x512 or full if smaller)
            window_size = min(512, src.height, src.width)
            row_off = (src.height - window_size) // 2
            col_off = (src.width - window_size) // 2
            window = rasterio.windows.Window(col_off, row_off, window_size, window_size)

            for i in range(3):
                band_idx = i + 1
                data = src.read(band_idx, window=window)

                min_val = data.min()
                max_val = data.max()

                if min_val < 0 or max_val > 255:
                    raise RasterValidationError(
                        f"Band {band_idx} values should be in range [0, 255], "
                        f"but found range [{min_val}, {max_val}]. File: {raster_path}"
                    )

    except rasterio.errors.RasterioError as e:
        raise RasterValidationError(f"Failed to open raster file: {raster_path}. Error: {e}")


def validate_input_raster_or_tiles(
    imagery_path: Optional[str] = None,
    tiles_path: Optional[str] = None,
    strict_color_interp: bool = True
) -> None:
    """
    Validate input raster or tiles at pipeline start.

    If imagery_path is provided, validates the raster.
    If only tiles_path is provided, validates the first tile.

    Args:
        imagery_path: Path to the main raster file (optional)
        tiles_path: Path to directory containing tiles (optional)
        strict_color_interp: If True, raise error for incorrect color interpretation.
                            If False, only warn if color interpretation is not R,G,B.

    Raises:
        RasterValidationError: If validation fails
    """
    if imagery_path:
        # Validate the main raster
        imagery_path = Path(imagery_path)
        validate_raster_rgb_bands(imagery_path, strict_color_interp)

    elif tiles_path:
        # Validate the first tile
        tiles_path = Path(tiles_path)

        # Find first tile (assumes tiles are .tif files)
        tile_files = list(tiles_path.glob("*.tif")) + list(tiles_path.glob("*.tiff"))

        if not tile_files:
            warnings.warn(f"No .tif/.tiff files found in tiles_path: {tiles_path}")
            return

        first_tile = sorted(tile_files)[0]
        validate_raster_rgb_bands(first_tile, strict_color_interp)
