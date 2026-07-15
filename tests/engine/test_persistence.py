"""Unit tests for persistence: config_hash, parquet round-trip, seed persistence, and from_dir FK
relinking across the imagery tree."""

import json

from canopyrs.engine import store
from canopyrs.engine.data import Imagery, Objects
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


def test_save_seeds_indexes_same_type_files(sources_seed, tiles_seed, tmp_path):
    """Two Imagery seeds must not clobber each other's parquet in _seed/."""
    store.save_seeds(tmp_path, [sources_seed, tiles_seed])
    entries = store.read_seeds(tmp_path)
    assert len(entries) == 2
    assert entries[0]["file"] != entries[1]["file"]
    for entry in entries:
        assert (tmp_path / store.SEED_DIR / entry["file"]).exists()


def _produce_entry(data_type, columns=(), links=(), crs=None, kind=None):
    return {"type": data_type.__name__, "file": store.FILENAME[data_type],
            "columns": list(columns), "links": list(links), "crs": crs, "kind": kind,
            "modalities": None, "timestamps": None}


def _write_run_dir(root, sources, tiles, objects):
    """Lay out a run on disk: source seed under _seed/, 0_tilerizer -> tile Imagery,
    1_detector -> Objects."""
    store.save_seeds(root, [sources])
    store.save_table(tiles, root / "0_tilerizer")
    store.save_table(objects, root / "1_detector")
    record = [
        {"id": 0, "name": "tilerizer", "config_hash": "h0",
         "produces": [_produce_entry(Imagery, links=["parent"], kind="tile")]},
        {"id": 1, "name": "detector", "config_hash": "h1",
         "produces": [_produce_entry(Objects, columns=[Col.DETECTOR_SCORE], links=["imagery"], crs=False)]},
    ]
    (root / store.RUN_RECORD).write_text(json.dumps(record))


def test_from_dir_reloads_and_relinks(sources_seed, tiles_seed, objects_seed, tmp_path):
    _write_run_dir(tmp_path, sources_seed, tiles_seed, objects_seed)

    pipe = Pipeline.from_dir(tmp_path)
    objs = pipe.latest(Objects)
    assert objs is not None and len(objs) == len(objects_seed)
    # FKs relinked from the persisted columns: objects -> tiles, tiles -> source seed
    assert objs.linked("imagery") is not None
    tiles = pipe.latest(Imagery)
    assert tiles.parent is not None
    # ancestry-reachable column survives the round-trip
    assert objs.column(Col.DETECTOR_SCORE).iloc[0] == 0.9
    # the containment walk still resolves window tiles to the seed raster's file
    assert tiles.resolved_paths().notna().all()


def test_from_dir_reloads_seed_tiles_and_relinks(tiles_seed, objects_seed, tmp_path):
    """A run seeded from a tiles folder (no tilerizer component): the seed tiles are persisted under
    _seed/, so from_dir reloads them and the produced Objects relink their imagery FK."""
    store.save_seeds(tmp_path, [tiles_seed])                 # tiles were a seed, not a produced table
    store.save_table(objects_seed, tmp_path / "0_detector")
    record = [
        {"id": 0, "name": "detector", "config_hash": "h0",
         "produces": [_produce_entry(Objects, columns=[Col.DETECTOR_SCORE], links=["imagery"], crs=False)]},
    ]
    (tmp_path / store.RUN_RECORD).write_text(json.dumps(record))

    pipe = Pipeline.from_dir(tmp_path)
    assert pipe.latest(Imagery) is not None                  # seed tiles reloaded from _seed/
    objs = pipe.latest(Objects)
    assert objs is not None and objs.linked("imagery") is not None   # imagery FK relinked to the seed
