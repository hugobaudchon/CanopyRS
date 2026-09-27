"""Converting a polygon to and from a COCO segmentation, in pixel coordinates.

COCO stores a segmentation in one of two formats:
- polygons: a list with one flat list [x1, y1, x2, y2, ...] per part. It can't hold holes.
- RLE: the mask's pixels, run-length encoded, with the mask's size: {"size": [height, width],
  "counts": ...}. It keeps holes.
"""

import numpy as np
from pycocotools import mask as coco_mask
from shapely.geometry import MultiPolygon, Polygon, box

from canopyrs1.core.geometry.masks import mask_to_polygon, polygon_to_mask
from canopyrs1.core.geometry.shapes import repair_polygon


def encode_segmentation(polygon, *, rle=False, height=None, width=None):
    """Return ``polygon`` (a Polygon or MultiPolygon, in pixel coordinates) as a COCO segmentation.

    - ``rle=False``: in the polygons format, one list per part. Holes are lost. An empty polygon
      gives an empty list.
    - ``rle=True``: in the RLE format, for a mask of ``height`` x ``width`` pixels, holding the
      pixels whose centre is inside the polygon (see ``polygon_to_mask``). Holes are kept.
    """
    if rle:
        encoded = coco_mask.encode(np.asfortranarray(polygon_to_mask(polygon, height, width)))
        return {"size": [int(height), int(width)], "counts": encoded["counts"].decode("ascii")}
    parts = getattr(polygon, "geoms", [polygon])
    return [np.asarray(part.exterior.coords)[:-1].ravel().tolist() for part in parts if not part.is_empty]


def _to_pycocotools_rle(segmentation):
    """Return an RLE segmentation in the form pycocotools reads: "counts" compressed, as bytes.
    COCO files store it as a string, or, for some crowd annotations, as a list of run lengths."""
    counts = segmentation["counts"]
    if isinstance(counts, list):
        return coco_mask.frPyObjects(segmentation, *segmentation["size"])
    if isinstance(counts, str):
        return {**segmentation, "counts": counts.encode("ascii")}
    return segmentation


def _points_to_polygon(segmentation):
    """Return the polygon of a segmentation in the polygons format, repaired if it isn't valid.
    Parts with fewer than 3 points are left out. Returns an empty Polygon if no part is left."""
    parts = [Polygon(np.reshape(coords, (-1, 2))) for coords in segmentation if len(coords) >= 6]
    if not parts:
        return Polygon()
    return repair_polygon(parts[0] if len(parts) == 1 else MultiPolygon(parts))


def decode_segmentation(segmentation, to="polygon", *, height=None, width=None):
    """Return a COCO segmentation, in either format, as:

    - ``to="polygon"``: a Polygon or MultiPolygon in pixel coordinates. From RLE, it follows the
      pixel edges exactly, holes included (see ``mask_to_polygon``).
    - ``to="box"``: the smallest north-up rectangle around it, as a Polygon.
    - ``to="mask"``: a uint8 mask. A segmentation in the polygons format needs ``height`` and
      ``width``; an RLE segmentation holds its own size.

    An empty segmentation gives an empty Polygon, or a mask of zeros.
    """
    if to not in ("polygon", "box", "mask"):
        raise ValueError(f"to must be 'polygon', 'box' or 'mask', not {to!r}")
    if not isinstance(segmentation, dict):
        polygon = _points_to_polygon(segmentation)
        if to == "mask":
            return polygon_to_mask(polygon, height, width)
        return box(*polygon.bounds) if to == "box" and not polygon.is_empty else polygon

    rle = _to_pycocotools_rle(segmentation)
    if to == "box":                                       # from the runs, without tracing the outline
        x, y, w, h = coco_mask.toBbox(rle)
        return box(x, y, x + w, y + h) if w > 0 else Polygon()
    mask = coco_mask.decode(rle)
    return mask if to == "mask" else mask_to_polygon(mask, fill_holes=False)
