"""write_coco gives the COCO file geodataset's COCOGenerator gives, for the same objects on the
same tile files, in the polygons format. The differences are the ones chosen:

- images: every tile, with ids from 1 (geodataset: only the tiles with objects, ids from 0);
- no is_rle_format, and other_attributes without a copy of the score;
- annotations in the objects' order (geodataset: grouped by tile file).
"""

import json
import os
from contextlib import redirect_stdout

import geopandas as gpd
from shapely.geometry import Point, box

from geodataset.utils.utils import COCOGenerator

from canopyrs1.core.constants import Col, GeomKind
from canopyrs1.core.io.coco import write_coco
from canopyrs1.core.tables.imagery import Tiles
from canopyrs1.core.tables.objects import Objects

CATEGORIES = [
    {"id": 1, "name": "Pinaceae", "other_names": [], "supercategory": None},
    {"id": 2, "name": "Picea", "other_names": ["PIGL"], "supercategory": 1},
]


def test_same_coco_file_as_geodataset(tiles_dir, tmp_path):
    tiles = Tiles.from_image_dir(tiles_dir)
    objects = Objects.build(
        geometry=[Point(64, 64).buffer(10), box(10, 6, 30, 16), box(50, 50, 60, 70.5)],
        geom_kind=[GeomKind.MASK, GeomKind.BOX, GeomKind.BOX],
        parent_image_id=[1, 0, 0],
        parent_imagery=tiles,
        columns={
            "species": ["PIGL", "Pinaceae", "Picea"],
            Col.DETECTOR_SCORE: [0.7, 0.9, 0.8],
            "height_m": [20.0, 12.5, 8.0],
        },
    )
    new = json.loads(
        write_coco(
            objects,
            tmp_path / "new.json",
            description="parity",
            categories=CATEGORIES,
            category_column="species",
            score_column=Col.DETECTOR_SCORE,
            attribute_columns=["height_m"],
        ).read_text()
    )

    gdf = gpd.GeoDataFrame(objects.df.drop(columns=Col.GEOMETRY), geometry=objects.df.geometry)
    gdf["tile_path"] = tiles.df[Col.PATH].to_numpy()[objects.df[Col.PARENT_IMAGE_ID]]
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        COCOGenerator.from_gdf(
            description="parity",
            gdf=gdf,
            tiles_paths_column="tile_path",
            polygons_column=Col.GEOMETRY,
            scores_column=Col.DETECTOR_SCORE,
            categories_column="species",
            other_attributes_columns=["height_m"],
            output_path=tmp_path / "old.json",
            use_rle_for_labels=False,
            n_workers=1,
            coco_categories_list=CATEGORIES,
        ).generate_coco()
    old = json.loads((tmp_path / "old.json").read_text())

    assert new["info"] == old["info"] and new["categories"] == old["categories"]
    # The same images (both tiles have objects here), by file name.
    old_names = {image["id"]: image["file_name"] for image in old["images"]}
    new_names = {image["id"]: image["file_name"] for image in new["images"]}

    def without_id(images):
        return sorted(({k: v for k, v in i.items() if k != "id"} for i in images), key=str)

    assert without_id(new["images"]) == without_id(old["images"])

    # The same annotations, by image file and box, but for the chosen differences.
    def by_image_and_box(coco, names):
        return {
            (names[a["image_id"]], tuple(a["bbox"])): {
                k: v for k, v in a.items() if k not in ("id", "image_id", "is_rle_format")
            }
            for a in coco["annotations"]
        }

    old_annotations = by_image_and_box(old, old_names)
    for annotation in old_annotations.values():
        assert annotation["other_attributes"].pop("score") == annotation["score"]
    assert by_image_and_box(new, new_names) == old_annotations
