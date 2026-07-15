"""Unit tests for v3 persistence: config_hash, parquet round-trip, and from_dir FK relinking."""

import json

from canopyrs.engine import store
from canopyrs.engine.data import Sources, Tiles, Objects
from canopyrs.engine.constants import Col
from canopyrs.engine.pipeline import Pipeline
from canopyrs.engine.config_parsers import AggregatorConfig


def test_config_hash_stable_and_detects_drift():
    a, b = AggregatorConfig(), AggregatorConfig()
    assert store.config_hash(a) == store.config_hash(b)
    drifted = AggregatorConfig(nms_threshold=round(a.nms_threshold + 0.1, 3))
    assert store.config_hash(drifted) != store.config_hash(a)


def test_save_load_roundtrip(objects_seed, tmp_path):
    path = store.save_table(objects_seed, tmp_path / "1_detector")
    df = store.load_df(Objects, path)
    assert len(df) == len(objects_seed)
    assert Col.DETECTOR_SCORE in df.columns
    assert df.geometry.notna().all()


def _write_run_dir(root, sources, tiles, objects):
    """Lay out a two-component run on disk (0_tilerizer -> Sources+Tiles, 1_detector -> Objects)."""
    store.save_table(sources, root / "0_tilerizer")
    store.save_table(tiles, root / "0_tilerizer")
    store.save_table(objects, root / "1_detector")
    manifest = [
        {"id": 0, "name": "tilerizer", "config_hash": "h0", "produces": [
            {"type": "Sources", "file": store.FILENAME[Sources], "columns": [], "links": [], "crs": None},
            {"type": "Tiles", "file": store.FILENAME[Tiles], "columns": [], "links": ["sources"], "crs": None},
        ]},
        {"id": 1, "name": "detector", "config_hash": "h1", "produces": [
            {"type": "Objects", "file": store.FILENAME[Objects], "columns": [Col.DETECTOR_SCORE],
             "links": ["tiles"], "crs": False},
        ]},
    ]
    (root / store.MANIFEST).write_text(json.dumps(manifest))


def test_from_dir_reloads_and_relinks(sources_seed, tiles_seed, objects_seed, tmp_path):
    _write_run_dir(tmp_path, sources_seed, tiles_seed, objects_seed)

    pipe = Pipeline.from_dir(tmp_path)
    objs = pipe.latest(Objects)
    assert objs is not None and len(objs) == len(objects_seed)
    # FK relinked from the persisted tile_id column -> the reloaded Tiles
    assert objs.linked("tiles") is not None
    assert pipe.latest(Tiles).linked("sources") is not None
    # ancestry-reachable column survives the round-trip
    assert objs.column(Col.DETECTOR_SCORE).iloc[0] == 0.9


def test_from_dir_reloads_seed_tiles_and_relinks(tiles_seed, objects_seed, tmp_path):
    """A run seeded from a tiles folder (no tilerizer component): the seed Tiles are persisted under
    _seed/, so from_dir reloads them and the produced Objects relink their tiles FK."""
    store.save_seeds(tmp_path, [tiles_seed])                 # tiles were a seed, not a produced table
    store.save_table(objects_seed, tmp_path / "0_detector")
    manifest = [
        {"id": 0, "name": "detector", "config_hash": "h0", "produces": [
            {"type": "Objects", "file": store.FILENAME[Objects], "columns": [Col.DETECTOR_SCORE],
             "links": ["tiles"], "crs": False},
        ]},
    ]
    (tmp_path / store.MANIFEST).write_text(json.dumps(manifest))

    pipe = Pipeline.from_dir(tmp_path)
    assert pipe.latest(Tiles) is not None                    # seed tiles reloaded from _seed/
    objs = pipe.latest(Objects)
    assert objs is not None and objs.linked("tiles") is not None   # tiles FK relinked to the seed
