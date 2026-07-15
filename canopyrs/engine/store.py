"""v3 persistence + file writers.

Each produced table is saved as parquet in its component's folder (Objects via the geopandas geo-parquet
writer — it keeps geometry, a null CRS, and nested columns; Tiles/Sources via plain parquet). A run
``pipeline.json`` manifest records the component order, each config's hash (for resume drift detection),
and what each produced (type + declared columns) so a run can be reloaded or resumed, and so an export
knows which columns belong to which step. The GPKG/COCO writers here take an already-assembled
GeoDataFrame; the pipeline does the table resolution (it owns the lineage).
"""

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from canopyrs.engine.utils import generate_coco               # reuse v1's geodataset COCO writer
from canopyrs.engine.constants import Col
from canopyrs.engine.contracts import as_requirements
from canopyrs.engine.data import Sources, Tiles, Objects

MANIFEST = "pipeline.json"
SEED_DIR = "_seed"
SEED_MANIFEST = "seeds.json"
FILENAME = {Sources: "sources.parquet", Tiles: "tiles.parquet", Objects: "objects.parquet"}
TYPE_BY_NAME = {cls.__name__: cls for cls in (Sources, Tiles, Objects)}

# Latest-wins ordering used to default an export's COCO score / category column within a window.
SCORE_COLS = [Col.DETECTOR_SCORE, Col.SEGMENTER_SCORE, Col.AGGREGATOR_SCORE, Col.CLASSIFIER_SCORE]
CLASS_COLS = [Col.DETECTOR_CLASS, Col.CLASSIFIER_CLASS]


def config_hash(component_config) -> str:
    """Stable short hash of a Pydantic component config, to detect config drift on resume."""
    payload = json.dumps(component_config.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def save_table(table, directory) -> Path:
    """Write ``table`` as parquet into ``directory`` (named by its type). Returns the file path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / FILENAME[type(table)]
    table.df.to_parquet(path)
    return path


def load_df(data_type, path):
    """Read a saved table's dataframe — geo-parquet for Objects (geometry), plain parquet otherwise."""
    return gpd.read_parquet(path) if data_type is Objects else pd.read_parquet(path)


def write_manifest(root, components) -> list:
    """Write (and return) the run manifest: per component its id/name/config_hash and the types +
    declared columns/links/crs it produces (which file each lands in)."""
    entries = []
    for component in components:
        produces = []
        for need in as_requirements(component.produces):
            produces.append({
                "type": need.data_type.__name__,
                "file": FILENAME[need.data_type],
                "columns": list(need.columns),
                "links": list(need.links),
                "crs": need.crs,
            })
        entries.append({
            "id": component.component_id,
            "name": component.name,
            "config_hash": config_hash(component.config),
            "produces": produces,
        })
    (Path(root) / MANIFEST).write_text(json.dumps(entries, indent=2))
    return entries


def read_manifest(root):
    """The run manifest at ``root``, or None if absent."""
    path = Path(root) / MANIFEST
    return json.loads(path.read_text()) if path.exists() else None


def save_seeds(root, seeds) -> None:
    """Persist the run's seed tables (given at construction, not produced by a component) under
    ``_seed/``, so a ``from_dir`` reload can rebuild FK links from produced Objects back to seeded
    tables (e.g. a run seeded from a pre-cut tiles folder). No-op when there are no seeds."""
    if not seeds:
        return
    directory = Path(root) / SEED_DIR
    entries = []
    for seed in seeds:
        save_table(seed, directory)
        entries.append({"type": type(seed).__name__, "file": FILENAME[type(seed)]})
    (directory / SEED_MANIFEST).write_text(json.dumps(entries, indent=2))


def read_seeds(root):
    """The seed manifest at ``root/_seed/``, or None if the run persisted no seeds."""
    path = Path(root) / SEED_DIR / SEED_MANIFEST
    return json.loads(path.read_text()) if path.exists() else None


def write_gpkg(gdf, path) -> Path:
    """Write a GeoDataFrame as GeoPackage, JSON-stringifying any list/dict column (GPKG can't store
    nested cells)."""
    out = gdf.copy()
    for col in out.columns:
        if col != Col.GEOMETRY and out[col].map(lambda v: isinstance(v, (list, dict))).any():
            out[col] = out[col].map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(path, driver="GPKG")
    return path


def write_coco(gdf, path, *, scores_column, categories_column, other_attributes_columns,
               use_rle, categories) -> Path:
    """Write a COCO file from a per-object GeoDataFrame (a ``tile_path`` + geometry column per row).
    Geometry may be CRS (geodataset converts to tile pixels via each tile file) or already pixel
    (crs=None). Thin wrapper over v1's ``generate_coco``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return generate_coco(
        description="CanopyRS v3 export",
        gdf=gdf,
        tiles_paths_column=Col.TILE_PATH,
        polygons_column=Col.GEOMETRY,
        scores_column=scores_column,
        categories_column=categories_column,
        other_attributes_columns=set(other_attributes_columns),
        coco_output_path=path,
        use_rle_for_labels=use_rle,
        n_workers=4,
        coco_categories_list=categories,
    )
