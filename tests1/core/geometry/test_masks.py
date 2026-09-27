from collections import deque

import numpy as np
import pytest
from affine import Affine
from rasterio import features
from shapely.geometry import MultiPolygon, Point, Polygon, box

from canopyrs1.core.geometry.masks import mask_to_polygon, polygon_to_mask

def _mask(size, pixels):
    """A size x size uint8 mask with 1 at each (row, col) in ``pixels``."""
    mask = np.zeros((size, size), np.uint8)
    for row, col in pixels:
        mask[row, col] = 1
    return mask


def _fill_holes_reference(mask):
    """Holes filled, computed independently: every empty pixel that can't reach the border moving
    up, down, left or right becomes 1."""
    h, w = mask.shape
    outside = np.zeros_like(mask, dtype=bool)
    queue = deque((r, c) for r in range(h) for c in range(w)
                  if (r in (0, h - 1) or c in (0, w - 1)) and not mask[r, c])
    for r, c in queue:
        outside[r, c] = True
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                queue.append((nr, nc))
    return (~outside).astype(np.uint8)


def _random_masks(n, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        size = int(rng.integers(1, 40))
        yield (rng.random((size, size)) < rng.uniform(0.05, 0.9)).astype(np.uint8)


# =============================================================================
# mask_to_polygon
# =============================================================================

def test_a_block_follows_the_pixel_edges():
    # 10 x 10 pixels in columns and rows 2 to 11: the outline goes from 2 to 12, area 100.
    mask = np.zeros((20, 20), np.uint8)
    mask[2:12, 2:12] = 1
    assert mask_to_polygon(mask).equals(box(2, 2, 12, 12))


@pytest.mark.parametrize("pixels, area", [
    ([(3, 3)], 1),                                              # one pixel
    ([(3, 2), (3, 3), (3, 4)], 3),                              # a row
    ([(1, 1), (2, 1), (3, 1), (3, 2), (3, 3)], 5),              # an L, one pixel wide
    ([(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3)], 7),   # a U, one pixel wide
])
def test_thin_shapes_keep_every_pixel(pixels, area):
    polygon = mask_to_polygon(_mask(8, pixels))
    assert isinstance(polygon, Polygon) and polygon.is_valid and polygon.area == area


def test_a_single_pixel_is_its_square():
    assert mask_to_polygon(_mask(8, [(3, 5)])).equals(box(5, 3, 6, 4))


def test_pixels_on_the_mask_border():
    mask = np.ones((6, 9), np.uint8)
    assert mask_to_polygon(mask).equals(box(0, 0, 9, 6))


def test_an_empty_mask():
    assert mask_to_polygon(np.zeros((5, 5), np.uint8)) == Polygon()


@pytest.mark.parametrize("dtype", [bool, np.uint8, np.int64, np.float32])
def test_any_nonzero_value_is_inside(dtype):
    mask = np.zeros((8, 8), dtype)
    mask[1:4, 2:6] = 1
    mask[5, 5] = 7 if dtype != bool else True
    assert mask_to_polygon(mask).equals(MultiPolygon([box(2, 1, 6, 4), box(5, 5, 6, 6)]))


def test_separate_groups_are_separate_parts():
    polygon = mask_to_polygon(_mask(8, [(1, 1), (1, 2), (5, 5)]))
    assert isinstance(polygon, MultiPolygon) and sorted(p.area for p in polygon.geoms) == [1, 2]


def test_pixels_touching_at_a_corner_are_separate_parts():
    polygon = mask_to_polygon(_mask(8, [(1, 1), (2, 2)]))
    assert isinstance(polygon, MultiPolygon) and len(polygon.geoms) == 2 and polygon.is_valid


def test_the_defaults_keep_every_pixel_step_and_part():
    mask = polygon_to_mask(Point(20, 20).buffer(8), 40, 40)
    mask[2, 2] = 1
    polygon = mask_to_polygon(mask)
    assert isinstance(polygon, MultiPolygon) and polygon.area == mask.sum()


def test_min_part_area():
    mask = np.zeros((20, 20), np.uint8)
    mask[1:6, 1:6] = 1                                          # 25 pixels
    mask[10, 10:13] = 1                                         # 3 pixels
    assert mask_to_polygon(mask, min_part_area=10).equals(box(1, 1, 6, 6))
    assert len(mask_to_polygon(mask, min_part_area=3).geoms) == 2
    single = _mask(8, [(2, 2)])
    assert mask_to_polygon(single, min_part_area=10).area == 1   # kept


def test_fill_holes():
    ring = np.ones((7, 7), np.uint8)
    ring[2:5, 2:5] = 0                                          # a 3 x 3 hole
    ring[3, 3] = 1                                              # with an island in it
    assert mask_to_polygon(ring).equals(box(0, 0, 7, 7))
    kept = mask_to_polygon(ring, fill_holes=False)
    assert kept.area == 49 - 8 and kept.is_valid


def test_a_hole_enclosed_by_corner_touching_pixels_is_filled():
    # A diamond of 8 pixels touching at their corners encloses a cross of 5 empty pixels.
    diamond = _mask(5, [(0, 2), (1, 1), (1, 3), (2, 0), (2, 4), (3, 1), (3, 3), (4, 2)])
    assert mask_to_polygon(diamond).area == 13
    assert _fill_holes_reference(diamond).sum() == 13


@pytest.mark.parametrize("fill_holes", [False, True])
def test_round_trip_gives_the_same_pixels(fill_holes):
    for mask in _random_masks(1000):
        polygon = mask_to_polygon(mask, fill_holes=fill_holes)
        assert polygon.is_valid
        expected = _fill_holes_reference(mask) if fill_holes else mask
        assert (polygon_to_mask(polygon, *mask.shape) == expected).all()


def test_simplify_moves_points_by_at_most_the_tolerance():
    circle = polygon_to_mask(Point(50, 50).buffer(30), 100, 100)
    exact = mask_to_polygon(circle)
    for tolerance in (0.5, 1.0, 2.0):
        simplified = mask_to_polygon(circle, simplify_tolerance=tolerance)
        assert len(simplified.exterior.coords) < len(exact.exterior.coords)
        assert simplified.hausdorff_distance(exact) <= tolerance + 1e-9


# =============================================================================
# polygon_to_mask
# =============================================================================

def test_a_box_fills_its_pixels():
    mask = polygon_to_mask(box(2, 2, 12, 12), 20, 20)
    assert mask.dtype == np.uint8 and mask.sum() == 100 and mask[2:12, 2:12].all()


def test_a_pixel_is_inside_when_its_centre_is():
    # Pixel centres are at .5: columns and rows 2 to 5 have their centres in [2.4, 5.6].
    mask = polygon_to_mask(box(2.4, 2.4, 5.6, 5.6), 8, 8)
    assert mask.sum() == 16 and mask[2:6, 2:6].all()
    assert polygon_to_mask(box(2.6, 2.6, 3.4, 3.4), 8, 8).sum() == 0     # covers no centre


def test_holes_and_parts():
    with_hole = Polygon(box(0, 0, 6, 6).exterior, holes=[box(2, 2, 4, 4).exterior])
    mask = polygon_to_mask(with_hole, 6, 6)
    assert mask.sum() == 32 and not mask[2:4, 2:4].any()
    assert polygon_to_mask(MultiPolygon([box(0, 0, 1, 1), box(3, 3, 5, 5)]), 6, 6).sum() == 5


def test_parts_outside_the_mask_are_ignored():
    assert polygon_to_mask(box(-5, -5, 3, 2), 10, 10).sum() == 6
    assert polygon_to_mask(box(8, 8, 30, 30), 10, 10).sum() == 4
    assert polygon_to_mask(box(20, 20, 30, 30), 10, 10).sum() == 0
    assert polygon_to_mask(Polygon(), 10, 10).sum() == 0


def test_matches_rasterizing_the_whole_mask():
    # Cropping to the polygon's bounds must not change the result.
    rng = np.random.default_rng(0)
    for _ in range(300):
        centre = rng.uniform(-10, 50, size=2)
        polygon = Point(*centre).buffer(rng.uniform(0.3, 20), quad_segs=int(rng.integers(1, 8)))
        full = features.rasterize([polygon], out_shape=(40, 40), transform=Affine.identity(), dtype="uint8")
        assert (polygon_to_mask(polygon, 40, 40) == full).all()
