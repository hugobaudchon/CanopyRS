import json
import warnings

import numpy as np
import pytest
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from shapely.geometry import MultiPolygon, Point, Polygon, box

from canopyrs1.core.constants import Col, GeomKind
from canopyrs1.core.geometry.coco_segmentation import decode_segmentation
from canopyrs1.core.geometry.georef import make_georef
from canopyrs1.core.geometry.masks import polygon_to_mask
from canopyrs1.core.io.coco import encode_annotation, read_coco, write_coco
from canopyrs1.core.tables.imagery import Sources, Tiles
from canopyrs1.core.tables.objects import Objects

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


# =============================================================================
# write_coco
# =============================================================================


def _objects_on_tiles(tiles_dir, **columns):
    """Return three objects in the pixels of the two tiles of ``tiles_dir``: two boxes on the
    first tile, a round mask on the second, with ``columns``."""
    tiles = Tiles.from_image_dir(tiles_dir)
    return Objects.build(
        geometry=[box(10, 6, 30, 16), box(50, 50, 60, 70), Point(64, 64).buffer(10)],
        geom_kind=[GeomKind.BOX, GeomKind.BOX, GeomKind.MASK],
        parent_image_id=[0, 0, 1],
        parent_imagery=tiles,
        columns=columns,
    )


def _read(path):
    return json.loads(path.read_text())


def test_a_coco_file(tiles_dir, tmp_path):
    path = write_coco(_objects_on_tiles(tiles_dir), tmp_path / "out" / "coco.json", description="d")
    assert path == tmp_path / "out" / "coco.json"
    coco = _read(path)
    assert coco["info"]["description"] == "d" and coco["licenses"] == []
    assert coco["images"] == [
        {"id": 1, "file_name": "tile_0.tif", "width": 128, "height": 128},
        {"id": 2, "file_name": "tile_1.tif", "width": 128, "height": 128},
    ]
    assert [a["id"] for a in coco["annotations"]] == [1, 2, 3]
    assert [a["image_id"] for a in coco["annotations"]] == [1, 1, 2]
    assert coco["annotations"][0]["bbox"] == [10.0, 6.0, 20.0, 10.0]
    # Without categories: one for every object.
    assert coco["categories"] == [{"id": 1, "name": "NoCategory", "supercategory": ""}]
    assert {a["category_id"] for a in coco["annotations"]} == {1}


def test_pycocotools_reads_the_file(tiles_dir, tmp_path):
    for rle in (False, True):
        path = write_coco(_objects_on_tiles(tiles_dir), tmp_path / f"coco_{rle}.json", rle=rle)
        loaded = COCO(str(path))
        mask = loaded.annToMask(loaded.anns[3])
        if rle:  # exactly the pixels whose centre is inside
            assert np.array_equal(mask, polygon_to_mask(Point(64, 64).buffer(10), 128, 128))
        assert mask.sum() == pytest.approx(np.pi * 10**2, rel=0.05)


def test_objects_in_a_crs_are_moved_into_their_tile_s_pixels(tiles_dir, tmp_path):
    # tile_0.tif covers x 0 to 128 m and y 128 to 256 m, in 1 m pixels.
    tiles = Tiles.from_image_dir(tiles_dir)
    objects = Objects.build(
        geometry=[box(10, 234, 30, 250)],
        geom_kind=GeomKind.BOX,
        crs=tiles.df.crs,
        parent_image_id=0,
        parent_imagery=tiles,
    )
    annotation = _read(write_coco(objects, tmp_path / "coco.json"))["annotations"][0]
    assert annotation["bbox"] == pytest.approx([10, 6, 20, 16])


def test_images_without_objects_are_kept(tiles_dir, tmp_path):
    tiles = Tiles.from_image_dir(tiles_dir)
    objects = Objects.build(
        geometry=[box(0, 0, 5, 5)],
        geom_kind=GeomKind.BOX,
        parent_image_id=1,
        parent_imagery=tiles,
    )
    coco = _read(write_coco(objects, tmp_path / "coco.json"))
    assert [image["file_name"] for image in coco["images"]] == ["tile_0.tif", "tile_1.tif"]
    assert coco["annotations"][0]["image_id"] == 2


def test_images_need_a_file_of_their_own(tmp_path):
    georef = make_georef(
        transform=[1, 0, 0, 0, -1, 64],
        crs="EPSG:32618",
        width=64,
        height=64,
        count=3,
        dtype="uint8",
    )
    windows = Tiles.build(georef=[georef])  # no file of its own
    objects = Objects.build(
        geometry=[box(0, 0, 5, 5)],
        geom_kind=GeomKind.BOX,
        parent_image_id=0,
        parent_imagery=windows,
    )
    with pytest.raises(ValueError, match="write the tiles to disk first"):
        write_coco(objects, tmp_path / "coco.json")


def test_scores_and_attributes_found_in_the_history(tiles_dir, tmp_path):
    detections = _objects_on_tiles(tiles_dir, **{Col.DETECTOR_SCORE: [0.9, np.nan, 0.7]})
    kept = Objects.build(
        geometry=detections.df.geometry,
        geom_kind=detections.df[Col.GEOM_KIND],
        parent_image_id=detections.df[Col.PARENT_IMAGE_ID],
        parent_imagery=detections.parent_imagery,
        parent_object_id=[0, 1, 2],
        parent_objects=detections,
        columns={"height_m": [12.5, np.nan, 20.0]},
    )
    annotations = _read(
        write_coco(
            kept,
            tmp_path / "coco.json",
            score_column=Col.DETECTOR_SCORE,
            attribute_columns=[Col.DETECTOR_SCORE, "height_m"],
        )
    )["annotations"]
    assert annotations[0]["score"] == 0.9 and "score" not in annotations[1]
    assert annotations[0]["other_attributes"] == {"detector_score": 0.9, "height_m": 12.5}
    assert annotations[1]["other_attributes"] == {"detector_score": None, "height_m": None}


TREES = [
    {"id": 1, "name": "Pinaceae", "other_names": [], "supercategory": None},
    {"id": 2, "name": "Picea", "other_names": ["PIGL", "PIMA"], "supercategory": 1},
]


def test_categories_from_a_list(tiles_dir, tmp_path):
    objects = _objects_on_tiles(tiles_dir, species=["Pinaceae", "PIMA", "Picea"])
    coco = _read(
        write_coco(objects, tmp_path / "coco.json", categories=TREES, category_column="species")
    )
    assert coco["categories"] == TREES
    assert [a["category_id"] for a in coco["annotations"]] == [1, 2, 2]


def test_a_category_not_in_the_list(tiles_dir, tmp_path):
    objects = _objects_on_tiles(tiles_dir, species=["Pinaceae", "Quercus", None])
    with pytest.warns(UserWarning, match=r"\['\(missing\)', 'Quercus'\] aren't known"):
        coco = _read(
            write_coco(objects, tmp_path / "c.json", categories=TREES, category_column="species")
        )
    assert [a["category_id"] for a in coco["annotations"]] == [1, -1, -1]


def test_a_list_of_one_category_without_a_column(tiles_dir, tmp_path):
    tree = [{"id": 1, "name": "tree", "supercategory": None}]
    coco = _read(write_coco(_objects_on_tiles(tiles_dir), tmp_path / "c.json", categories=tree))
    assert coco["categories"] == tree
    assert {a["category_id"] for a in coco["annotations"]} == {1}


def test_categories_found_in_a_column(tiles_dir, tmp_path):
    # Numbered in the order of their names, the same from one run to the next.
    objects = _objects_on_tiles(tiles_dir, species=["pine", "birch", "pine"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # every name is known: no warning
        coco = _read(write_coco(objects, tmp_path / "c.json", category_column="species"))
    assert coco["categories"] == [
        {"id": 1, "name": "birch", "supercategory": ""},
        {"id": 2, "name": "pine", "supercategory": ""},
    ]
    assert [a["category_id"] for a in coco["annotations"]] == [2, 1, 2]


BAD_CATEGORIES = [
    ([{"id": 1, "name": "tree"}], "has no supercategory"),
    (
        [
            {"id": 1, "name": "a", "supercategory": ""},
            {"id": 2, "name": "b", "other_names": ["a"], "supercategory": ""},
        ],
        "'a' is given twice",
    ),
    (
        [{"id": 1, "name": "a", "supercategory": ""}, {"id": 1, "name": "b", "supercategory": ""}],
        "Two categories have the same id",
    ),
]


@pytest.mark.parametrize("categories, message", BAD_CATEGORIES)
def test_categories_that_aren_t_valid(categories, message, tiles_dir, tmp_path):
    objects = _objects_on_tiles(tiles_dir, species=["a", "b", "a"])
    with pytest.raises(ValueError, match=message):
        write_coco(objects, tmp_path / "c.json", categories=categories, category_column="species")


def test_several_categories_without_a_column(tiles_dir, tmp_path):
    with pytest.raises(ValueError, match="which of the 2 categories"):
        write_coco(_objects_on_tiles(tiles_dir), tmp_path / "c.json", categories=TREES)


def test_the_real_crop(real_raster, tmp_path):
    # One box covering the whole crop, in the crop's own file.
    sources = Sources.from_paths(real_raster)
    coco = _read(write_coco(Objects.from_imagery(sources), tmp_path / "coco.json"))
    georef = sources.df[Col.GEOREF][0]
    assert coco["images"][0]["file_name"] == real_raster.name
    assert coco["annotations"][0]["bbox"] == [0, 0, georef["width"], georef["height"]]


# =============================================================================
# read_coco
# =============================================================================


def test_read_back_what_was_written(tiles_dir, tmp_path):
    objects = _objects_on_tiles(
        tiles_dir,
        species=["Pinaceae", "PIMA", "Picea"],
        **{Col.DETECTOR_SCORE: [0.9, 0.8, 0.7], "height_m": [12.5, 8.0, 20.0]},
    )
    path = write_coco(
        objects,
        tiles_dir / "coco.json",  # next to its tiles: the default images_dir
        categories=TREES,
        category_column="species",
        score_column=Col.DETECTOR_SCORE,
        attribute_columns=["height_m"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # its boxes and areas are those of its segmentations
        tiles, back = read_coco(path)
    assert list(tiles.df[Col.PATH]) == [
        str(tiles_dir / "tile_0.tif"),
        str(tiles_dir / "tile_1.tif"),
    ]
    assert back.parent_imagery is tiles and not back.has_crs
    assert list(back.df[Col.PARENT_IMAGE_ID]) == [0, 0, 1]
    assert list(back.df[Col.GEOM_KIND]) == [GeomKind.BOX, GeomKind.BOX, GeomKind.MASK]
    for got, expected in zip(back.df.geometry, objects.df.geometry):
        assert got.hausdorff_distance(expected) < 1e-9
    assert list(back.df[Col.CATEGORY_ID]) == [1, 2, 2]
    assert list(back.df[Col.CATEGORY_NAME]) == ["Pinaceae", "Picea", "Picea"]
    assert list(back.df[Col.SCORE]) == [0.9, 0.8, 0.7]
    assert list(back.df["height_m"]) == [12.5, 8.0, 20.0]


def test_read_rle(tiles_dir, tmp_path):
    objects = _objects_on_tiles(tiles_dir)
    _, back = read_coco(write_coco(objects, tmp_path / "coco.json", rle=True), tiles_dir)
    # The mask comes back as its pixels, the boxes exactly.
    expected = polygon_to_mask(Point(64, 64).buffer(10), 128, 128)
    assert np.array_equal(polygon_to_mask(back.df.geometry[2], 128, 128), expected)
    assert back.df.geometry[0].equals(box(10, 6, 30, 16))


def test_annotations_without_a_segmentation_are_boxes(tiles_dir, tmp_path):
    coco = {
        "images": [{"id": 5, "file_name": "tile_1.tif", "width": 128, "height": 128}],
        "annotations": [
            {"id": 1, "image_id": 5, "category_id": 1, "bbox": [10, 20, 30, 40]},
            {"id": 2, "image_id": 5, "category_id": 1, "bbox": [0, 0, 5, 5], "segmentation": []},
        ],
        "categories": [{"id": 1, "name": "tree", "supercategory": None}],
    }
    path = tmp_path / "boxes.json"
    path.write_text(json.dumps(coco))
    tiles, objects = read_coco(path, tiles_dir)
    assert len(tiles) == 1 and list(objects.df[Col.PARENT_IMAGE_ID]) == [0, 0]
    assert objects.df.geometry[0].equals(box(10, 20, 40, 60))
    assert (objects.df[Col.GEOM_KIND] == GeomKind.BOX).all()
    assert Col.SCORE not in objects.df.columns  # no annotation has one


def test_images_without_annotations(tiles_dir, tmp_path):
    coco = {
        "images": [{"id": 1, "file_name": "tile_0.tif", "width": 128, "height": 128}],
        "annotations": [],
        "categories": [],
    }
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(coco))
    tiles, objects = read_coco(path, tiles_dir)
    assert len(tiles) == 1 and len(objects) == 0


def test_a_warning_when_boxes_and_areas_aren_t_those_of_the_segmentations(tiles_dir, tmp_path):
    path = write_coco(_objects_on_tiles(tiles_dir), tmp_path / "coco.json")
    coco = _read(path)
    coco["annotations"][0]["area"] *= 1.1  # 10% larger than its segmentation's
    coco["annotations"][1]["bbox"][2] += 2  # 2 pixels wider
    path.write_text(json.dumps(coco))
    with pytest.warns(UserWarning) as caught:
        read_coco(path, tiles_dir)
    assert str(caught[0].message) == (
        "In coco.json, the bbox and area of 2 of 3 annotations aren't those of their "
        "segmentation: the areas differ by 4.5% (median), and the boxes by up to 2.0 pixels. "
        "They are recomputed from the segmentations."
    )
