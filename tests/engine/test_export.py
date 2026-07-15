"""Unit tests for v3 export: GPKG column widening, and COCO requiring tiles on disk."""

import json

import geopandas as gpd
import pytest
from shapely.geometry import box

from canopyrs.engine import store
from canopyrs.engine.data import Sources, Tiles, Objects
from canopyrs.engine.constants import Col, BOX
from canopyrs.engine.pipeline import Pipeline


def _georef_objects(tiles_seed):
    """Aggregator-style georeferenced (crs-set) box Objects with an aggregator_score, linked to tiles."""
    tile_id = tiles_seed.df[Col.TILE_ID].iloc[0]
    return Objects.build(
        geometry=[box(0, 0, 32, 32), box(32, 32, 64, 64)],
        geom_kind=BOX, tile_id=[tile_id, tile_id], tiles=tiles_seed, crs="EPSG:32618",
        **{Col.AGGREGATOR_SCORE: [0.7, 0.6]},
    )


def _write_run_dir(root, sources, tiles, objects):
    store.save_table(sources, root / "0_tilerizer")
    store.save_table(tiles, root / "0_tilerizer")
    store.save_table(objects, root / "1_aggregator")
    manifest = [
        {"id": 0, "name": "tilerizer", "config_hash": "h0", "produces": [
            {"type": "Sources", "file": store.FILENAME[Sources], "columns": [], "links": [], "crs": None},
            {"type": "Tiles", "file": store.FILENAME[Tiles], "columns": [], "links": ["sources"], "crs": None},
        ]},
        {"id": 1, "name": "aggregator", "config_hash": "h1", "produces": [
            {"type": "Objects", "file": store.FILENAME[Objects], "columns": [Col.AGGREGATOR_SCORE],
             "links": ["tiles"], "crs": True},
        ]},
    ]
    (root / store.MANIFEST).write_text(json.dumps(manifest))


def test_export_gpkg_widens_columns(sources_seed, tiles_seed, tmp_path):
    _write_run_dir(tmp_path, sources_seed, tiles_seed, _georef_objects(tiles_seed))
    pipe = Pipeline.from_dir(tmp_path)

    out = pipe.export("gpkg")
    gdf = gpd.read_file(out)
    assert len(gdf) == 2
    assert Col.AGGREGATOR_SCORE in gdf.columns


def test_export_coco_requires_tiles_on_disk(sources_seed, tiles_seed, tmp_path):
    # tiles_seed carries no tile_path (windows read on demand), so COCO export must refuse
    _write_run_dir(tmp_path, sources_seed, tiles_seed, _georef_objects(tiles_seed))
    pipe = Pipeline.from_dir(tmp_path)

    with pytest.raises(ValueError):
        pipe.export("coco")
