"""Unit tests for export: GPKG column widening, and COCO requiring tile images on disk."""

import json

import geopandas as gpd
import pytest
from shapely.geometry import box

from canopyrs.engine import store
from canopyrs.engine.data import Objects, Tiles
from canopyrs.engine.constants import Col, GeomKind
from canopyrs.engine.pipeline import Pipeline


def _georef_objects(tiles_seed):
    """Aggregator-style georeferenced (crs-set) box Objects with an aggregator_score, linked to tiles."""
    image_id = tiles_seed.df[Col.IMAGE_ID].iloc[0]
    return Objects.build(
        geometry=[box(0, 0, 32, 32), box(32, 32, 64, 64)],
        geom_kind=GeomKind.BOX, image_id=[image_id, image_id], imagery=tiles_seed, crs="EPSG:32618",
        **{Col.AGGREGATOR_SCORE: [0.7, 0.6]},
    )


def _produce_entry(data_type, columns=(), links=(), crs=None):
    return {"type": data_type.__name__, "file": store.FILENAME[data_type],
            "columns": list(columns), "links": list(links), "crs": crs,
            "modalities": None, "timestamps": None}


def _write_run_dir(root, sources, tiles, objects):
    store.save_seeds(root, [sources])
    store.save_table(tiles, root / "0_tilerizer")
    store.save_table(objects, root / "1_aggregator")
    record = [
        {"id": 0, "name": "tilerizer", "config_hash": "h0",
         "produces": [_produce_entry(Tiles, links=["parent"])]},
        {"id": 1, "name": "aggregator", "config_hash": "h1",
         "produces": [_produce_entry(Objects, columns=[Col.AGGREGATOR_SCORE], links=["imagery"], crs=True)]},
    ]
    (root / store.RUN_RECORD).write_text(json.dumps(record))


def test_export_gpkg_widens_columns(sources_seed, tiles_seed, tmp_path):
    _write_run_dir(tmp_path, sources_seed, tiles_seed, _georef_objects(tiles_seed))
    pipe = Pipeline.from_dir(tmp_path)

    out = pipe.export("gpkg")
    gdf = gpd.read_file(out)
    assert len(gdf) == 2
    assert Col.AGGREGATOR_SCORE in gdf.columns


def test_export_coco_pixel_anchor_requires_tiles_on_disk(sources_seed, tiles_seed, objects_seed, tmp_path):
    """Pixel-coord objects are only valid against their tile's own file, so window tiles must refuse.
    (CRS anchors are exempt: they export against the nearest materialized file — geodataset converts
    CRS geometry per file — which is what lets classified objects on window crops export per tile.)"""
    _write_run_dir(tmp_path, sources_seed, tiles_seed, objects_seed)
    pipe = Pipeline.from_dir(tmp_path)

    with pytest.raises(ValueError, match="on disk"):
        pipe.export("coco")
