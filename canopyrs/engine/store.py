"""Persistence + file writers.

Each produced table is saved as parquet in its component's folder (Objects via the geopandas
geo-parquet writer — it keeps geometry, a null CRS, and nested columns; Imagery via plain parquet).
A ``run.json`` **run record** at the run root — written by the pipeline, never hand-edited — records
the component order, each config's hash (for resume drift detection), and what each produced (type,
file, declared columns/links/crs/kind, plus the actual modality and timestamp sets) so a run can be
reloaded, resumed, or exported. The GPKG/COCO writers here take an already-assembled GeoDataFrame; the
pipeline picks and widens the right tables (it owns the lineage).
"""

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from canopyrs.engine.utils import generate_coco               # geodataset-backed COCO writer
from canopyrs.engine.constants import Col
from canopyrs.engine.contracts import as_requirements
from canopyrs.engine.data import Imagery, Objects

RUN_RECORD = "run.json"
SEED_DIR = "_seed"
SEED_RECORD = "seeds.json"
FILENAME = {Imagery: "imagery.parquet", Objects: "objects.parquet"}
TYPE_BY_NAME = {cls.__name__: cls for cls in (Imagery, Objects)}

# Column name used in *exported* GPKG/COCO files for the per-object image path (output convention,
# consumed by the benchmark evaluators / geodataset — not an Imagery table column).
EXPORT_TILE_PATH = "tile_path"

# Latest-wins ordering used to default an export's COCO score / category column within a window.
SCORE_COLS = [Col.DETECTOR_SCORE, Col.SEGMENTER_SCORE, Col.AGGREGATOR_SCORE, Col.CLASSIFIER_SCORE]
CLASS_COLS = [Col.DETECTOR_CLASS, Col.CLASSIFIER_CLASS]


def config_hash(component_config) -> str:
    """Stable short hash of a Pydantic component config, to detect config drift on resume."""
    payload = json.dumps(component_config.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def save_table(table, directory, filename=None) -> Path:
    """Write ``table`` as parquet into ``directory`` (named by its type unless ``filename`` is given).
    Returns the file path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (filename or FILENAME[type(table)])
    table.df.to_parquet(path)
    return path


def load_df(data_type, path):
    """Read a saved table's dataframe — geo-parquet for Objects (geometry), plain parquet otherwise."""
    return gpd.read_parquet(path) if data_type is Objects else pd.read_parquet(path)


def _table_sets(table):
    """The actual (modalities, timestamps) sets a produced table holds, as sorted JSON-safe lists
    (None when the column is absent). Recorded, not contract-checked — makes future constraints
    (e.g. ``min_timestamps=``) purely additive."""
    modalities = (sorted(set(table.df[Col.MODALITY].dropna()))
                  if Col.MODALITY in table.df.columns else None)
    timestamps = (sorted({str(t) for t in table.df[Col.TIMESTAMP].dropna()})
                  if Col.TIMESTAMP in table.df.columns else None)
    return modalities, timestamps


def write_run_record(root, components, outputs) -> list:
    """Write (and return) the run record: per component its id/name/config_hash and what it produced —
    type + file + declared columns/links/crs/kind, and the actual modality/timestamp sets. ``outputs``
    is the pipeline's per-component list of produced tables, aligned to ``components``."""
    entries = []
    for component, produced_tables in zip(components, outputs):
        produces = []
        for need in as_requirements(component.produces):
            table = next((t for t in produced_tables if type(t) is need.data_type), None)
            modalities, timestamps = _table_sets(table) if table is not None else (None, None)
            produces.append({
                "type": need.data_type.__name__,
                "file": FILENAME[need.data_type],
                "columns": list(need.columns),
                "links": list(need.links),
                "crs": need.crs,
                "kind": need.kind,
                "modalities": modalities,
                "timestamps": timestamps,
            })
        entries.append({
            "id": component.component_id,
            "name": component.name,
            "config_hash": config_hash(component.config),
            "produces": produces,
        })
    (Path(root) / RUN_RECORD).write_text(json.dumps(entries, indent=2, default=str))
    return entries


def read_run_record(root):
    """The run record at ``root``, or None if absent."""
    path = Path(root) / RUN_RECORD
    return json.loads(path.read_text()) if path.exists() else None


def save_seeds(root, seeds) -> None:
    """Persist the run's seed tables (given at construction, not produced by a component) under
    ``_seed/``, so a ``from_dir`` reload can rebuild FK links from produced tables back to seeded ones
    (e.g. a run seeded from a pre-cut tiles folder). Files are index-prefixed — several seeds may share
    a type. No-op when there are no seeds."""
    if not seeds:
        return
    directory = Path(root) / SEED_DIR
    entries = []
    for i, seed in enumerate(seeds):
        filename = f"seed_{i}_{FILENAME[type(seed)]}"
        save_table(seed, directory, filename=filename)
        entries.append({"type": type(seed).__name__, "file": filename})
    (directory / SEED_RECORD).write_text(json.dumps(entries, indent=2))


def read_seeds(root):
    """The seed record at ``root/_seed/``, or None if the run persisted no seeds."""
    path = Path(root) / SEED_DIR / SEED_RECORD
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
    (crs=None). Thin wrapper over ``generate_coco``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return generate_coco(
        description="CanopyRS export",
        gdf=gdf,
        tiles_paths_column=EXPORT_TILE_PATH,
        polygons_column=Col.GEOMETRY,
        scores_column=scores_column,
        categories_column=categories_column,
        other_attributes_columns=set(other_attributes_columns),
        coco_output_path=path,
        use_rle_for_labels=use_rle,
        n_workers=4,
        coco_categories_list=categories,
    )
