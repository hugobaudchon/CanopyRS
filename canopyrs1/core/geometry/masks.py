"""Converting between polygons and masks: 2D arrays with one value per pixel, nonzero inside.

Polygons are in pixel coordinates, where pixel (col, row) is the square from (col, row) to
(col + 1, row + 1), as in the georef. With this convention, a mask turned into a polygon without
simplification, and back into a mask, gives exactly the same pixels.
"""

import math

import cv2
import numpy as np
from affine import Affine
from rasterio import features
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry

from canopyrs1.core.geometry.shapes import remove_small_parts, repair_polygon


def rasterize_in_bounds(
    polygon: BaseGeometry,
    height: int,
    width: int,
) -> tuple[np.ndarray, int, int]:
    """Return the part of ``polygon_to_mask(polygon, height, width)`` under the polygon's bounds,
    as ``(pixels, row, col)``: ``pixels`` is that part of the mask, and ``row`` and ``col`` are
    where its top-left pixel is in the whole mask. Rasterizing only this part is much faster for a
    small polygon in a large mask. ``pixels`` is empty if the polygon is empty or outside."""
    if polygon.is_empty:
        return np.zeros((0, 0), dtype=np.uint8), 0, 0
    minx, miny, maxx, maxy = polygon.bounds
    col0, row0 = max(math.floor(minx), 0), max(math.floor(miny), 0)
    col1, row1 = min(math.ceil(maxx), width), min(math.ceil(maxy), height)
    if col1 <= col0 or row1 <= row0:
        return np.zeros((0, 0), dtype=np.uint8), 0, 0
    pixels = features.rasterize(
        [polygon],
        out_shape=(row1 - row0, col1 - col0),
        transform=Affine.translation(col0, row0),
        dtype="uint8",
    )
    return pixels, row0, col0


def polygon_to_mask(polygon: BaseGeometry, height: int, width: int) -> np.ndarray:
    """Return a ``height`` x ``width`` uint8 mask holding 1 in each pixel whose centre is inside
    ``polygon`` (in pixel coordinates), and 0 elsewhere, including in the polygon's holes. Parts
    of the polygon outside the mask are ignored."""
    mask = np.zeros((height, width), dtype=np.uint8)
    pixels, row, col = rasterize_in_bounds(polygon, height, width)
    mask[row : row + pixels.shape[0], col : col + pixels.shape[1]] = pixels
    return mask


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Return ``mask`` (uint8, 0 or 1, with an empty border) with its holes set to 1. A hole is a
    group of empty pixels that can't reach the border moving up, down, left or right."""
    reached = mask.copy()
    cv2.floodFill(
        image=reached,
        mask=None,
        seedPoint=(0, 0),
        newVal=2,
        flags=4,
    )
    return (reached != 2).astype(np.uint8)


def mask_to_polygon(
    mask: np.ndarray,
    *,
    fill_holes: bool = True,
    simplify_tolerance: float = 0.0,
    min_part_area: float = 0,
) -> Polygon | MultiPolygon:
    """Return the outline of the nonzero pixels of ``mask`` (a 2D array of any type), in pixel
    coordinates, along the pixel edges. Pixels touching only at a corner are in different parts; one
    part is returned as a Polygon, several as a MultiPolygon. Returns an empty Polygon if the mask
    has no nonzero pixel.

    - ``fill_holes``: fill the holes, i.e. the groups of empty pixels enclosed by the mask.
    - ``simplify_tolerance``: simplify the outline, moving no point by more than this many pixels.
      0, the default, keeps every pixel step.
    - ``min_part_area``: drop the parts smaller than this many pixels. A single part is always kept.
      0, the default, keeps every part.
    """
    # view the mask as uint8
    mask = np.asarray(mask)
    if mask.dtype == bool:
        mask = mask.view(np.uint8)  # no copy
    elif mask.dtype != np.uint8:
        mask = (mask != 0).view(np.uint8)

    # crop to the pixels' bounding box
    x, y, w, h = cv2.boundingRect(np.ascontiguousarray(mask))
    if w == 0:
        return Polygon()
    # Only the bounding box of the pixels is traced, padded with one empty pixel on each side.
    crop = cv2.copyMakeBorder(
        (mask[y : y + h, x : x + w] != 0).astype(np.uint8),
        top=1,
        bottom=1,
        left=1,
        right=1,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    # trace the outline
    if fill_holes:
        crop = _fill_holes(crop)
    outlines = features.shapes(
        crop,
        mask=crop.astype(bool),
        transform=Affine.translation(x - 1, y - 1),
    )
    parts = [shape(outline) for outline, _ in outlines]
    polygon = parts[0] if len(parts) == 1 else MultiPolygon(parts)

    # clean it up
    if simplify_tolerance:
        polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)
    return repair_polygon(remove_small_parts(polygon, min_part_area))
