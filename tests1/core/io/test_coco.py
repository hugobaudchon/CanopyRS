import json

import numpy as np
import pytest
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from shapely.geometry import MultiPolygon, Point, Polygon, box

from canopyrs1.core.geometry.coco_segmentation import decode_segmentation
from canopyrs1.core.geometry.masks import polygon_to_mask
from canopyrs1.core.io.coco import encode_annotation

# =============================================================================
# encode_annotation
# =============================================================================


def _encode(polygon, **options):
    """Return the annotation of ``polygon`` in a 64 x 48 image, with ids 7 and 3 and category 1."""
    return encode_annotation(
        polygon,
        annotation_id=7,
        image_id=3,
        category_id=1,
        height=48,
        width=64,
        **options,
    )


def test_a_box():
    assert _encode(box(10, 5, 30, 25)) == {
        "id": 7,
        "image_id": 3,
        "category_id": 1,
        "segmentation": [[30.0, 5.0, 30.0, 25.0, 10.0, 25.0, 10.0, 5.0]],
        "area": 400.0,
        "bbox": [10.0, 5.0, 20.0, 20.0],
        "iscrowd": 0,
    }


def test_a_mask_in_two_parts():
    mask = MultiPolygon([box(0, 0, 10, 10), Point(40, 30).buffer(5)])
    annotation = _encode(mask)
    assert len(annotation["segmentation"]) == 2  # one list of points per part
    assert annotation["area"] == pytest.approx(mask.area)
    assert annotation["bbox"] == pytest.approx([0, 0, 45, 35])


def test_in_rle():
    mask = Point(20, 20).buffer(8)
    annotation = _encode(mask, rle=True)
    assert annotation["segmentation"]["size"] == [48, 64]
    # The same pixels as polygon_to_mask: those whose centre is inside the polygon.
    decoded = decode_segmentation(annotation["segmentation"], "mask")
    assert np.array_equal(decoded, polygon_to_mask(mask, 48, 64))
    # Area and box still come from the polygon.
    assert annotation["area"] == pytest.approx(mask.area)
    assert annotation["bbox"] == pytest.approx([12, 12, 16, 16])


def test_a_score_and_other_attributes_only_when_given():
    plain = _encode(box(0, 0, 1, 1))
    assert "score" not in plain and "other_attributes" not in plain
    assert "other_attributes" not in _encode(box(0, 0, 1, 1), other_attributes={})
    scored = _encode(box(0, 0, 1, 1), score=0.9, other_attributes={"species": "oak"})
    assert scored["score"] == 0.9 and scored["other_attributes"] == {"species": "oak"}


def test_without_a_category():
    annotation = encode_annotation(
        box(0, 0, 1, 1),
        annotation_id=1,
        image_id=1,
        category_id=None,
        height=8,
        width=8,
    )
    assert annotation["category_id"] is None


def test_numpy_values_become_plain_ones():
    # Ids and scores often come from pandas columns: the annotation must still be valid JSON.
    annotation = encode_annotation(
        box(0, 0, 1, 1),
        annotation_id=np.int64(7),
        image_id=np.int64(3),
        category_id=np.int32(1),
        height=8,
        width=8,
        score=np.float32(0.5),
    )
    assert json.loads(json.dumps(annotation)) == annotation
    assert type(annotation["id"]) is int and type(annotation["score"]) is float


NOT_POLYGONS = [Polygon(), Point(1, 1)]


@pytest.mark.parametrize("geometry", NOT_POLYGONS)
def test_what_can_t_be_an_annotation(geometry):
    with pytest.raises(ValueError, match="A COCO annotation needs a polygon"):
        _encode(geometry)


def test_pycocotools_reads_it(tmp_path):
    mask = Point(20, 20).buffer(8)
    coco = {
        "images": [{"id": 3, "width": 64, "height": 48, "file_name": "tile.tif"}],
        "annotations": [_encode(mask), {**_encode(mask, rle=True), "id": 8}],
        "categories": [{"id": 1, "name": "tree", "supercategory": None}],
    }
    path = tmp_path / "coco.json"
    path.write_text(json.dumps(coco))
    loaded = COCO(str(path))
    from_polygon, from_rle = (loaded.annToMask(loaded.anns[i]) for i in (7, 8))
    assert np.array_equal(from_rle, polygon_to_mask(mask, 48, 64))
    # pycocotools draws polygons its own way: the two masks differ only along the outline.
    assert np.abs(from_polygon.astype(int) - from_rle).sum() < 0.1 * from_rle.sum()
    assert coco_mask.area(coco_mask.encode(np.asfortranarray(from_rle))) == from_rle.sum()
