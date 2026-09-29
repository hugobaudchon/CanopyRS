"""Writing objects to COCO files, the format detection and segmentation models train and are
evaluated on."""

from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from canopyrs1.core.geometry.coco_segmentation import encode_segmentation


def encode_annotation(
    polygon: BaseGeometry,
    *,
    annotation_id: int,
    image_id: int,
    category_id: int | None,
    height: int,
    width: int,
    rle: bool = False,
    score: float | None = None,
    other_attributes: dict | None = None,
) -> dict:
    """Return the COCO annotation of ``polygon``, a box or a mask in the pixel coordinates of its
    ``height`` x ``width`` image: its segmentation (see ``encode_segmentation``; RLE if ``rle``),
    and its ``bbox`` ([x, y, width, height]) and ``area``, both from the polygon itself.
    ``score`` and ``other_attributes`` are only added when given. Raises a ValueError if
    ``polygon`` is empty, or isn't a Polygon or a MultiPolygon."""
    # check the polygon
    if not isinstance(polygon, (Polygon, MultiPolygon)) or polygon.is_empty:
        empty = "an empty " if polygon.is_empty else "a "
        raise ValueError(f"A COCO annotation needs a polygon, not {empty}{polygon.geom_type}")

    # the fields of every annotation
    minx, miny, maxx, maxy = polygon.bounds
    annotation = {
        "id": int(annotation_id),
        "image_id": int(image_id),
        "category_id": None if category_id is None else int(category_id),
        "segmentation": encode_segmentation(polygon, rle=rle, height=height, width=width),
        "area": float(polygon.area),
        "bbox": [float(minx), float(miny), float(maxx - minx), float(maxy - miny)],
        "iscrowd": 0,
    }

    # the optional ones
    if score is not None:
        annotation["score"] = float(score)
    if other_attributes:
        annotation["other_attributes"] = other_attributes
    return annotation
