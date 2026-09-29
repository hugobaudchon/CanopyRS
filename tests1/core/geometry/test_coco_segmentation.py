import json

import numpy as np
import pytest
from pycocotools import mask as coco_mask
from shapely.geometry import MultiPolygon, Point, Polygon, box

from canopyrs1.core.geometry.coco_segmentation import decode_segmentation, encode_segmentation
from canopyrs1.core.geometry.masks import mask_to_polygon, polygon_to_mask

WITH_HOLE = Polygon(box(2, 2, 12, 12).exterior, holes=[box(5, 5, 8, 8).exterior])
TWO_PARTS = MultiPolygon([box(1, 1, 4, 4), box(10, 10, 15, 13)])


def _uncompressed_rle(mask):
    """The RLE of ``mask`` with "counts" as a plain list of run lengths, column by column,
    starting with a run of zeros, as some COCO crowd annotations store it."""
    runs, value, length = [], 0, 0
    for pixel in mask.flatten(order="F"):
        if pixel != value:
            runs.append(length)
            value, length = pixel, 0
        length += 1
    runs.append(length)
    return {"size": list(mask.shape), "counts": runs}


# =============================================================================
# The polygons format
# =============================================================================


def test_encode_as_points():
    assert encode_segmentation(box(1, 2, 5, 6)) == [[5.0, 2.0, 5.0, 6.0, 1.0, 6.0, 1.0, 2.0]]
    assert len(encode_segmentation(TWO_PARTS)) == 2
    assert encode_segmentation(Polygon()) == []
    assert json.loads(json.dumps(encode_segmentation(TWO_PARTS))) == encode_segmentation(TWO_PARTS)


def test_points_round_trip():
    circle = Point(20, 20).buffer(8)
    assert decode_segmentation(encode_segmentation(circle)).equals_exact(circle, 1e-9)
    assert decode_segmentation(encode_segmentation(TWO_PARTS)).equals(TWO_PARTS)
    # Holes are lost.
    assert decode_segmentation(encode_segmentation(WITH_HOLE)).equals(box(2, 2, 12, 12))


def test_decode_points_repairs_and_skips_bad_parts():
    bowtie = [[0, 0, 2, 2, 2, 0, 0, 2]]  # its outline crosses itself
    repaired = decode_segmentation(bowtie)
    assert repaired.is_valid and repaired.area == pytest.approx(2)
    two_points_and_a_triangle = [[0, 0, 5, 5], [1, 1, 4, 1, 4, 4]]
    triangle = Polygon([(1, 1), (4, 1), (4, 4)])
    assert decode_segmentation(two_points_and_a_triangle).equals(triangle)
    assert decode_segmentation([]) == Polygon()


def test_decode_points_to_box_and_mask():
    segmentation = encode_segmentation(TWO_PARTS)
    assert decode_segmentation(segmentation, "box").equals(box(1, 1, 15, 13))
    mask = decode_segmentation(segmentation, "mask", height=20, width=20)
    assert (mask == polygon_to_mask(TWO_PARTS, 20, 20)).all()
    assert decode_segmentation([], "box") == Polygon()


# =============================================================================
# The RLE format
# =============================================================================


def test_encode_as_rle():
    rle = encode_segmentation(WITH_HOLE, rle=True, height=20, width=30)
    assert rle["size"] == [20, 30] and isinstance(rle["counts"], str)
    assert json.loads(json.dumps(rle)) == rle
    decoded = coco_mask.decode({**rle, "counts": rle["counts"].encode()})  # pycocotools reads it
    assert (decoded == polygon_to_mask(WITH_HOLE, 20, 30)).all()


def test_rle_round_trip_keeps_holes():
    rle = encode_segmentation(WITH_HOLE, rle=True, height=20, width=20)
    assert decode_segmentation(rle).equals(WITH_HOLE)
    assert (decode_segmentation(rle, "mask") == polygon_to_mask(WITH_HOLE, 20, 20)).all()


def test_rle_round_trip_on_random_masks():
    rng = np.random.default_rng(0)
    for _ in range(200):
        h, w = rng.integers(1, 30, size=2)
        mask = (rng.random((h, w)) < rng.uniform(0.05, 0.9)).astype(np.uint8)
        polygon = mask_to_polygon(mask, fill_holes=False)
        rle = encode_segmentation(polygon, rle=True, height=h, width=w)
        assert (decode_segmentation(rle, "mask") == mask).all()


def test_rle_box_is_the_pixels_extent():
    # 10 x 10 pixels in columns and rows 2 to 11: the box goes from 2 to 12.
    mask = np.zeros((20, 20), np.uint8)
    mask[2:12, 2:12] = 1
    rle = encode_segmentation(mask_to_polygon(mask), rle=True, height=20, width=20)
    assert decode_segmentation(rle, "box").equals(box(2, 2, 12, 12))


@pytest.mark.parametrize("counts_as", ["str", "bytes", "list"])
def test_every_way_of_storing_rle_counts(counts_as):
    mask = polygon_to_mask(TWO_PARTS, 20, 20)
    rle = encode_segmentation(TWO_PARTS, rle=True, height=20, width=20)
    if counts_as == "bytes":
        rle = {**rle, "counts": rle["counts"].encode()}
    elif counts_as == "list":
        rle = _uncompressed_rle(mask)
    assert (decode_segmentation(rle, "mask") == mask).all()
    assert decode_segmentation(rle).equals(TWO_PARTS)
    assert decode_segmentation(rle, "box").equals(box(1, 1, 15, 13))


FAST_RLE_CASES = [
    (WITH_HOLE, 20, 30),
    (TWO_PARTS, 20, 20),
    (Polygon(), 5, 5),  # empty
    (box(-10, -10, -5, -5), 8, 8),  # outside
    (box(0, 0, 8, 8), 8, 8),  # the whole image: no run of zeros at all
    (box(0, 0, 3, 8), 8, 8),  # on the left edge
    (box(5, 0, 8, 8), 8, 8),  # on the right edge: it ends with a run of ones
    (box(7, 7, 20, 20), 8, 8),  # past the bottom-right corner
    (Point(3.3, 11.7).buffer(2.6), 13, 9),  # not on pixel edges
]


@pytest.mark.parametrize("polygon, height, width", FAST_RLE_CASES)
def test_fast_rle_is_the_same_as_encoding_the_whole_mask(polygon, height, width):
    fast = encode_segmentation(polygon, rle=True, height=height, width=width)
    full = encode_segmentation(polygon, rle=True, height=height, width=width, fast=False)
    assert fast == full


def test_fast_rle_on_random_masks():
    rng = np.random.default_rng(0)
    for _ in range(50):
        h, w = rng.integers(5, 40, size=2)
        polygon = mask_to_polygon(rng.random((h, w)) > 0.6)
        fast = encode_segmentation(polygon, rle=True, height=h, width=w)
        assert fast == encode_segmentation(polygon, rle=True, height=h, width=w, fast=False)


def test_empty_rle():
    rle = encode_segmentation(Polygon(), rle=True, height=5, width=5)
    assert decode_segmentation(rle) == Polygon()
    assert decode_segmentation(rle, "box") == Polygon()
    assert decode_segmentation(rle, "mask").sum() == 0


def test_unknown_output():
    with pytest.raises(ValueError, match="'polygon', 'box' or 'mask'"):
        decode_segmentation([], "bbox")
