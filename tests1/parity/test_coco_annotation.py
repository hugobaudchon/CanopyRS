"""encode_annotation gives the annotations geodataset's COCOGenerator gives, in the polygons
format. (In RLE, geodataset's masks carry the half-pixel offset fixed in polygon_to_mask.)"""

import pytest
from shapely.geometry import MultiPolygon, Point, box

from geodataset.utils.utils import COCOGenerator

from canopyrs1.core.io.coco import encode_annotation

POLYGONS = [
    box(10, 5, 30, 25),
    Point(20.3, 17.8).buffer(7.1),
    MultiPolygon([box(0, 0, 10, 10), Point(40, 30).buffer(5)]),
    box(55, 40, 70, 60),  # reaching outside its 64 x 48 image
]


@pytest.mark.parametrize("polygon", POLYGONS)
def test_same_annotation_as_geodataset(polygon):
    old = COCOGenerator._generate_label_coco(
        polygon=polygon,
        polygon_id=7,
        score=0.8,
        tile_height=48,
        tile_width=64,
        tile_id=3,
        use_rle_for_labels=False,
        category_id=1,
        other_attributes_dict={"species": "oak"},
    )
    new = encode_annotation(
        polygon,
        annotation_id=7,
        image_id=3,
        category_id=1,
        height=48,
        width=64,
        score=0.8,
        other_attributes={"species": "oak"},
    )
    # Every field is the same, but the redundant is_rle_format, which is no longer written.
    assert old.pop("is_rle_format") is False
    assert new == old
