"""Reading and writing COCO files, the format detection and segmentation models train and are
evaluated on."""

import json
import warnings
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry

from canopyrs1.core.constants import Col
from canopyrs1.core.geometry.coco_segmentation import decode_segmentation, encode_segmentation
from canopyrs1.core.geometry.shapes import infer_geom_kind
from canopyrs1.core.tables.imagery import Tiles
from canopyrs1.core.tables.objects import Objects
from canopyrs1.core.types import PathLike


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


def _coco_categories(
    names: list | None,
    categories: list[dict] | None,
    n_objects: int,
) -> tuple[list[dict], list[int]]:
    """Return the COCO categories, and each object's category id, from ``names`` (each object's
    category name, or None if the objects have none) and ``categories`` (a COCO categories list,
    or None). See ``write_coco`` for the rules."""
    # without names: one category for every object
    if names is None:
        if categories is None:
            return [{"id": 1, "name": "NoCategory", "supercategory": ""}], [1] * n_objects
        if len(categories) > 1:
            raise ValueError(
                f"Give a category column, to say which of the {len(categories)} categories each "
                f"object is in"
            )
        return categories, [categories[0]["id"]] * n_objects

    # with a categories list: each name, and each other name, of a category gives its id
    if categories is not None:
        name_to_id = {}
        for category in categories:
            missing = {"id", "name", "supercategory"} - set(category)
            if missing:
                raise ValueError(f"The category {category} has no {', '.join(sorted(missing))}")
            for name in [category["name"], *(category.get("other_names") or [])]:
                if name in name_to_id:
                    raise ValueError(f"The category name {name!r} is given twice")
                name_to_id[name] = category["id"]
        if len({category["id"] for category in categories}) < len(categories):
            raise ValueError("Two categories have the same id")
        coco_categories = categories

    # without one: a category per name, numbered from 1 in the order of the names
    else:
        found = sorted({name for name in names if not pd.isna(name)}, key=str)
        name_to_id = {name: i + 1 for i, name in enumerate(found)}
        coco_categories = [
            {"id": i, "name": name, "supercategory": ""} for name, i in name_to_id.items()
        ]

    # match each object's name
    unknown = sorted(
        {"(missing)" if pd.isna(name) else str(name) for name in names if name not in name_to_id}
    )
    if unknown:
        warnings.warn(
            f"The categories {unknown} aren't known: their objects get the category_id -1",
            UserWarning,
            stacklevel=3,
        )
    return coco_categories, [name_to_id.get(name, -1) for name in names]


def _plain(value: object) -> object:
    """Return ``value``, with a missing value (NaN, None) as None, so it is written as null."""
    return None if pd.api.types.is_scalar(value) and pd.isna(value) else value


def write_coco(
    objects: Objects,
    path: PathLike,
    *,
    description: str = "",
    categories: list[dict] | None = None,
    category_column: str | None = None,
    score_column: str | None = None,
    attribute_columns: Sequence[str] = (),
    rle: bool = False,
) -> Path:
    """Write ``objects`` (boxes and masks) to a COCO file at ``path``, creating its folder if
    needed, and return ``path``.

    - Images: every image of the objects' ``parent_imagery``, in its order, with ids from 1. Each
      must have its own file, whose name the COCO file holds: write the tiles to disk first.
    - Annotations: one per object, in its order, with ids from 1 (see ``encode_annotation``; RLE
      segmentations if ``rle``). ``score_column`` gives each one's score, and
      ``attribute_columns`` its ``other_attributes``. Columns are looked for in the objects'
      history too (see ``Objects.get_column``).
    - Categories: with a COCO ``categories`` list, each object's name in ``category_column`` is
      matched to a category's ``name`` or one of its ``other_names``. Without a list, there is a
      category per name in ``category_column``, numbered from 1 in the order of the names. A
      missing name, or one matching no category, gives the category_id -1, with a warning.
      Without a column, a list of one category is used for every object, and without a list
      either, a single "NoCategory".

    Raises a ValueError if an object has no image, if an image has no file of its own, or if
    ``categories`` isn't valid (a category without an id, name or supercategory, a name or an id
    given twice, or several categories without a column).
    """
    # put each object in its image's pixels
    geometry = objects.get_geometry_in_image_coords(pixels=True).to_list()
    imagery = objects.parent_imagery.df
    if imagery[Col.PATH].isna().any():
        raise ValueError(
            "Every image of a COCO file needs a file of its own: write the tiles to disk first"
        )

    # the images
    georefs = imagery[Col.GEOREF].to_list()
    images = [
        {"id": i + 1, "file_name": Path(file).name, "width": g["width"], "height": g["height"]}
        for i, (file, g) in enumerate(zip(imagery[Col.PATH], georefs))
    ]

    # the categories
    names = objects.get_column(category_column).to_list() if category_column else None
    coco_categories, category_ids = _coco_categories(names, categories, len(objects))

    # the annotations
    image_ids = objects.df[Col.PARENT_IMAGE_ID].astype(int).to_list()
    scores = objects.get_column(score_column).to_list() if score_column else [None] * len(objects)
    attributes = {column: objects.get_column(column).to_list() for column in attribute_columns}
    annotations = []
    for k, (polygon, image_id) in enumerate(zip(geometry, image_ids)):
        annotations.append(
            encode_annotation(
                polygon,
                annotation_id=k + 1,
                image_id=image_id + 1,
                category_id=category_ids[k],
                height=georefs[image_id]["height"],
                width=georefs[image_id]["width"],
                rle=rle,
                score=_plain(scores[k]),
                other_attributes={
                    column: _plain(values[k]) for column, values in attributes.items()
                },
            )
        )

    # write the file
    coco = {
        "info": {
            "description": description,
            "version": "1.0",
            "year": str(date.today().year),
            "date_created": str(date.today()),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": coco_categories,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(coco, ensure_ascii=False, indent=2, default=lambda item: item.tolist())
    path.write_text(text, encoding="utf-8")
    return path


def _warn_about_boxes_and_areas(
    path: Path,
    polygons: np.ndarray,
    annotations: list[dict],
) -> None:
    """Warn if the ``bbox`` and ``area`` the COCO file at ``path`` gives its ``annotations``
    aren't those of their ``polygons``, saying by how much."""
    # the file's values, and those of the polygons
    file_areas = np.array([a.get("area", np.nan) for a in annotations], dtype=float)
    file_boxes = np.array([a.get("bbox", [np.nan] * 4) for a in annotations], dtype=float)
    minx, miny, maxx, maxy = shapely.bounds(polygons).T
    boxes = np.column_stack([minx, miny, maxx - minx, maxy - miny])

    # how far apart they are
    area_gaps = np.abs(shapely.area(polygons) - file_areas) / np.maximum(file_areas, 1e-9)
    box_gaps = np.abs(boxes - file_boxes).max(axis=1)
    different = (area_gaps > 1e-6) | (box_gaps > 1e-6)
    if different.any():
        warnings.warn(
            f"In {path.name}, the bbox and area of {different.sum()} of {len(annotations)} "
            f"annotations aren't those of their segmentation: the areas differ by "
            f"{np.nanmedian(area_gaps[different]):.1%} (median), and the boxes by up to "
            f"{np.nanmax(box_gaps[different]):.1f} pixels. They are recomputed from the "
            f"segmentations.",
            UserWarning,
            stacklevel=3,
        )


def _corners(bbox: list[float]) -> tuple[float, float, float, float]:
    """Return a COCO ``bbox`` ([x, y, width, height]) as (minx, miny, maxx, maxy)."""
    x, y, width, height = bbox
    return x, y, x + width, y + height


def read_coco(path: PathLike, images_dir: PathLike | None = None) -> tuple[Tiles, Objects]:
    """Return the images and annotations of the COCO file at ``path``, as a Tiles table and an
    Objects table.

    - Tiles: one per image, in the file's order, read from its file in ``images_dir`` (the COCO
      file's folder by default; see ``Imagery.from_paths``).
    - Objects: one per annotation, in the file's order, in its image's pixels: its segmentation,
      or its ``bbox`` if it has none. Their kind is found from their geometry (see
      ``infer_geom_kind``). Each has its ``category_id``, its category's name
      (``category_name``), its ``score`` if the file gives one, and each of its
      ``other_attributes``, as columns.

    The file's ``bbox`` and ``area`` aren't kept, as ``write_coco`` computes them from the
    segmentation; if they aren't those of the segmentations, a warning says by how much.
    """
    # read the file
    path = Path(path)
    coco = json.loads(path.read_text(encoding="utf-8"))
    images_dir = Path(images_dir) if images_dir is not None else path.parent

    # the images
    tiles = Tiles.from_paths([images_dir / image["file_name"] for image in coco["images"]])
    position = {image["id"]: i for i, image in enumerate(coco["images"])}

    # the geometry of each annotation: its segmentation, or its box
    annotations = coco["annotations"]
    geometry = [
        decode_segmentation(a["segmentation"])
        if a.get("segmentation")
        else box(*_corners(a["bbox"]))
        for a in annotations
    ]
    segmented = [i for i, a in enumerate(annotations) if a.get("segmentation")]
    if segmented:
        polygons = np.array(geometry, dtype=object)[segmented]
        _warn_about_boxes_and_areas(path, polygons, [annotations[i] for i in segmented])

    # their columns
    names = {category["id"]: category["name"] for category in coco.get("categories", [])}
    columns = {
        Col.CATEGORY_ID: [a.get("category_id") for a in annotations],
        Col.CATEGORY_NAME: [names.get(a.get("category_id")) for a in annotations],
    }
    if any("score" in a for a in annotations):
        columns[Col.SCORE] = [a.get("score") for a in annotations]
    attributes = {key for a in annotations for key in (a.get("other_attributes") or {})}
    for key in sorted(attributes):
        columns[key] = [(a.get("other_attributes") or {}).get(key) for a in annotations]

    objects = Objects.build(
        geometry=geometry,
        geom_kind=infer_geom_kind(geometry),
        parent_image_id=[position[a["image_id"]] for a in annotations],
        parent_imagery=tiles,
        columns=columns,
    )
    return tiles, objects
