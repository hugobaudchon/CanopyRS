"""The relational data model: two typed tables threaded through the pipeline.

See ``canopyrs/engine/README.md`` for the model in five sentences.

Everything is a ``Table`` — a (Geo)DataFrame plus a primary key:
  - ``Imagery`` : georeferenced image regions — whole source scenes, grid tiles, per-object crops —
                  one self-referential tree. A row is materialized (has a ``path``) or a window into
                  its parent (``parent_id``); ``kind`` says whether it's a source scene or a
                  model-consumable tile.
  - ``Objects`` : detected/segmented things (boxes or masks), found in an image, optionally derived
                  from a previous Object.

Relations are foreign-key *columns* (``parent_id`` / ``image_id`` / ``prev_object_id``) plus a hydrated
pointer attribute set at build time (``imagery.parent``, ``objects.imagery``, ``objects.prev_objects``).
The column is the source of truth; the pointer is convenience. Two ancestries, one rule each:
  - Imagery containment (``parent``): to *read* a region, resolve it to its nearest materialized
    ancestor (``resolved_paths``) and read the region's window from that file.
  - Objects lineage (``prev_objects``): a value produced several steps back stays reachable
    (``provides`` / ``column`` / ``linked``).

For contract checks, every table renders itself into a ``Schema`` via ``schema()`` — the snapshot must
stay faithful to what the table exposes (including ancestry-reachable columns and links).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.windows import Window

from canopyrs.engine.constants import Col, GeomKind, ImageKind, Modality
from canopyrs.engine.contracts import Schema
from canopyrs.engine.tilemeta import window_meta

RGB = [1, 2, 3]

# Column of Imagery.reading_frame(): the file this row's pixels are actually read from — its own
# ``path`` when materialized, else its nearest materialized ancestor's (see ``resolved_paths``).
READ_PATH = "read_path"


def _has_usable_values(df, col) -> bool:
    """Whether ``df`` exposes usable values for ``col``: the column is present and — unless the frame
    is empty — has at least one non-null value."""
    return col in df.columns and (len(df) == 0 or df[col].notna().any())


class Table:
    """A (Geo)DataFrame plus its primary key. Related parents are passed as keyword args, validated
    (FK column present, values reference an existing parent row), and held as attributes."""

    pk: str = None
    fks: dict = {}            # attribute name -> foreign-key column in df

    def __init__(self, df, **related):
        if self.pk not in df.columns:
            raise ValueError(f"{type(self).__name__} requires a '{self.pk}' column")
        self.df = df
        for name, parent in related.items():
            self._link(name, parent)

    @classmethod
    def with_ids(cls, df, **related):
        """Construct, stamping a fresh 0..n primary key (local ids; a pointer says which table an FK
        resolves against, so ids needn't be globally unique)."""
        return cls._assemble(df, None, related)

    @classmethod
    def _assemble(cls, df, ids, related):
        """Stamp the primary key (given ids, else a fresh 0..n) and construct."""
        df[cls.pk] = list(ids) if ids is not None else range(len(df))
        return cls(df, **related)

    def _link(self, name, parent):
        fk = self.fks.get(name)
        if not fk:
            raise ValueError(f"{type(self).__name__} has no relation '{name}' (known: {list(self.fks)})")
        if fk not in self.df.columns:
            raise ValueError(f"{type(self).__name__} missing FK column '{fk}'")
        dangling = set(self.df[fk].dropna()) - set(parent.df[parent.pk])
        if dangling:
            raise ValueError(
                f"'{fk}' references missing {type(parent).__name__}.{parent.pk}: {sorted(dangling)[:3]}")
        setattr(self, name, parent)

    def provides(self, col) -> bool:
        """Whether this table exposes usable values for ``col`` (present and non-null). Objects also
        searches its ancestry."""
        return _has_usable_values(self.df, col)

    def linked(self, name):
        """The table at relation ``name`` — hydrated directly, or resolved back through the
        ``prev_objects`` ancestry (a derived object inherits its predecessor's links, so e.g. an
        aggregated box finds the grid tiles of the detection it kept, without anyone re-attaching
        it). None if no ancestor has it."""
        direct = getattr(self, name, None)
        if direct is not None:
            return direct
        prev = getattr(self, "prev_objects", None)
        return prev.linked(name) if prev is not None else None

    def has_link(self, name) -> bool:
        """Whether relation ``name`` is resolvable — directly or through the ``prev_objects`` ancestry."""
        return self.linked(name) is not None

    @property
    def crs_set(self):
        """True/False: whether this table's geometry is in CRS coords. (A plain DataFrame has no CRS.)"""
        return getattr(self.df, "crs", None) is not None

    @property
    def kind(self):
        """The table's uniform ImageKind, or None when not applicable / mixed (checks skip on None)."""
        return None

    @property
    def modalities(self):
        """The set of modalities this table holds, or None when unknown (checks skip on None)."""
        return None

    def schema(self) -> Schema:
        """This table as a ``Schema`` — the snapshot every contract check runs on: usable columns,
        resolvable links, crs / kind / modalities. Objects extends it with ancestry-reachable columns."""
        cols = {c for c in self.df.columns if self.provides(c)}
        links = {name for name in self.fks if self.has_link(name)}
        return Schema(columns=cols, links=links, crs=self.crs_set, kind=self.kind,
                      modalities=self.modalities)

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        return f"{type(self).__name__}({len(self.df)} rows)"


class Imagery(Table):
    """Georeferenced image regions, one self-referential tree: a source scene is a root with a
    ``path``; a grid tile is a child of its source (a window, or materialized to disk); a crop is a
    child of its tile. ``kind`` (uniform per table) separates whole scenes from model-consumable tiles.
    Imagery never points to Objects: the image<->object relationship is one-image-to-many-objects, so
    the foreign key lives on the Object side (``image_id``) — even a one-crop-per-object crop is
    reached as ``object.imagery``, never the reverse."""

    pk = Col.IMAGE_ID
    fks = {"parent": Col.PARENT_ID}
    parent: Optional["Imagery"] = None

    def __init__(self, df, **related):
        super().__init__(df, **related)
        self._resolved_paths = None   # memoized nearest-materialized-ancestor paths

    @property
    def kind(self):
        values = set(self.df[Col.KIND].dropna()) if Col.KIND in self.df.columns else set()
        return values.pop() if len(values) == 1 else None

    @property
    def modalities(self):
        if Col.MODALITY not in self.df.columns:
            return None
        return set(self.df[Col.MODALITY].dropna())

    def resolved_paths(self) -> pd.Series:
        """Per-row path of the nearest materialized ancestor — the row's own ``path`` when set, else
        the closest ancestor's up the ``parent`` chain. Memoized and resolved per *table* (each level
        reuses its parent's already-resolved paths), never walked per object. NaN where no ancestor is
        materialized (e.g. an unhydrated reload) — the loader errors on those rows."""
        if self._resolved_paths is None:
            paths = (self.df[Col.PATH] if Col.PATH in self.df.columns
                     else pd.Series([None] * len(self.df), index=self.df.index, dtype=object))
            if self.parent is not None and Col.PARENT_ID in self.df.columns and paths.isna().any():
                by_id = pd.Series(self.parent.resolved_paths().values,
                                  index=self.parent.df[self.parent.pk].values)
                paths = paths.fillna(self.df[Col.PARENT_ID].map(by_id))
            self._resolved_paths = paths
        return self._resolved_paths

    def reading_frame(self) -> pd.DataFrame:
        """One row per image with everything the loader needs, keyed by ``image_id``: the region's
        window (``metadata``), ``bands``, its own ``path`` (when materialized), and ``read_path`` —
        the file its pixels are read from (own path, else the nearest materialized ancestor's; the
        window is then read from that file at the region's bounds)."""
        cols = [c for c in (Col.IMAGE_ID, Col.METADATA, Col.BANDS, Col.PATH) if c in self.df.columns]
        frame = self.df[cols].copy()
        frame[READ_PATH] = self.resolved_paths().values
        return frame

    @classmethod
    def from_paths(cls, sources) -> "Imagery":
        """A kind='source' Imagery table from a raster path, a list of paths, or ``{path, modality,
        timestamp}`` descriptors (a single RGB raster is the common case). An Imagery instance passes
        through."""
        if isinstance(sources, cls):
            return sources
        if isinstance(sources, (str, Path)):
            sources = [sources]
        rows = [s if isinstance(s, dict) else {"path": s} for s in sources]
        n = len(rows)
        data = {
            Col.KIND: [ImageKind.SOURCE] * n,
            Col.PATH: [str(row["path"]) for row in rows],
            Col.PARENT_ID: [None] * n,
            Col.MODALITY: [row.get("modality", Modality.RGB) for row in rows],
            Col.TIMESTAMP: [row.get("timestamp") for row in rows],
        }
        return cls.with_ids(pd.DataFrame(data))

    @classmethod
    def from_tiles_dir(cls, path, bands=RGB) -> "Imagery":
        """Seed a kind='tile' Imagery table from a folder of pre-cut georeferenced GeoTIFF tiles (e.g.
        a geodataset tiles output). Each tile's window metadata is recovered from the file itself
        (full-raster window) and ``path`` points at it. Roots (no parent): the loader reads each file
        directly and the aggregator georeferences from ``metadata``."""
        paths = sorted(p for pattern in ("*.tif", "*.tiff") for p in Path(path).glob(pattern))
        if not paths:
            raise ValueError(f"no .tif/.tiff tiles found in {path}")
        metadata = []
        for p in paths:
            with rasterio.open(p) as src:
                metadata.append(window_meta(src, Window(0, 0, src.width, src.height)))
        return cls.build(kind=ImageKind.TILE, metadata=metadata, path=[str(p) for p in paths],
                         bands=bands)

    @classmethod
    def build(cls, *, kind, metadata, parent_id=None, path=None, bands=RGB, modality=Modality.RGB,
              timestamp=None, image_id=None, parent=None) -> "Imagery":
        """Construct flat imagery rows from per-row arrays — one (modality, timestamp) per row.
        ``kind`` / ``parent_id`` / ``modality`` / ``timestamp`` may be a scalar (broadcast) or per-row.
        ``image_id`` keeps given ids (e.g. geodataset's), else a fresh 0..n is stamped. ``parent`` is
        the hydrated Imagery table ``parent_id`` resolves against."""
        n = len(metadata)
        data = {
            Col.KIND: kind,
            Col.PARENT_ID: parent_id if parent_id is not None else [None] * n,
            Col.PATH: path if path is not None else [None] * n,
            Col.MODALITY: modality,
            Col.TIMESTAMP: timestamp,
            Col.BANDS: [bands] * n,
            Col.METADATA: list(metadata),
        }
        related = {"parent": parent} if parent is not None else {}
        return cls._assemble(pd.DataFrame(data), image_id, related)


class Objects(Table):
    """Detected/segmented things (boxes or masks). Found in an image; may derive from a previous Object."""

    pk = Col.OBJECT_ID
    fks = {"imagery": Col.IMAGE_ID, "prev_objects": Col.PREV_OBJECT_ID}
    imagery: Optional[Imagery] = None
    prev_objects: Optional["Objects"] = None

    def __init__(self, df, **related):
        super().__init__(df, **related)
        if Col.GEOM_KIND not in df.columns:
            raise ValueError(f"Objects requires a '{Col.GEOM_KIND}' column")
        bad = set(df[Col.GEOM_KIND].dropna()) - GeomKind.ALL   # column mandatory; None values = unknown, allowed
        if bad:
            raise ValueError(f"unknown {Col.GEOM_KIND} {sorted(bad)}, expected {sorted(GeomKind.ALL)}")

    def provides(self, col) -> bool:
        """``col`` is exposed here (present, non-null) or resolvable through the ``prev_objects`` ancestry."""
        if col in self.df.columns:
            return _has_usable_values(self.df, col)
        return self.prev_objects is not None and self.prev_objects.provides(col)

    def schema(self) -> Schema:
        """Like ``Table.schema`` but with ancestry-reachable columns included, so a contract check on
        the snapshot answers the same as ``provides`` would on the live chain."""
        base = super().schema()
        if self.prev_objects is not None:
            base.columns |= self.prev_objects.schema().columns
        return base

    def column(self, col) -> pd.Series:
        """Values of ``col`` for these objects, aligned to ``self.df``'s rows. Returned from this table
        if it carries the column, else followed back through the ``prev_objects`` ancestry (each row's
        ``prev_object_id`` -> the parent's ``object_id``) until an ancestor has it. Lets a late
        component (e.g. the aggregator) pull a score produced several steps back without every step
        carrying it forward. Raises ``KeyError`` if no ancestor in the chain carries it."""
        if col in self.df.columns:
            return self.df[col]
        if self.prev_objects is None:
            raise KeyError(f"'{col}' is not in these Objects nor their prev_objects ancestry")
        parent = self.prev_objects
        lookup = pd.Series(parent.column(col).values, index=parent.df[parent.pk].values)
        return self.df[Col.PREV_OBJECT_ID].map(lookup)

    @classmethod
    def from_gpkg(cls, path, imagery=None, prev_objects=None) -> "Objects":
        """Seed an Objects table from a GeoPackage previously written by the pipeline (round-trip): it
        carries geometry, ``geom_kind``, ``object_id`` and any FK columns. Pass ``imagery`` /
        ``prev_objects`` to re-link whichever FK columns the file holds (so the ancestry walk works)."""
        gdf = gpd.read_file(path)
        if Col.GEOM_KIND not in gdf.columns:
            raise ValueError(f"{path} has no '{Col.GEOM_KIND}' column; not a pipeline-written Objects gpkg")
        related = {}
        if imagery is not None and Col.IMAGE_ID in gdf.columns:
            related["imagery"] = imagery
        if prev_objects is not None and Col.PREV_OBJECT_ID in gdf.columns:
            related["prev_objects"] = prev_objects
        ids = gdf[Col.OBJECT_ID].values if Col.OBJECT_ID in gdf.columns else None
        return cls._assemble(gdf, ids, related)

    @classmethod
    def build(cls, *, geometry, geom_kind, image_id=None, prev_object_id=None, timestamp=None,
              object_id=None, crs=None, imagery=None, prev_objects=None, **columns) -> "Objects":
        """Construct objects from geometry + the usual keys, plus any extra attribute columns (scores,
        classes, ...) by Col name. ``object_id`` keeps given ids, else a fresh 0..n is stamped."""
        data = {Col.GEOMETRY: list(geometry), Col.GEOM_KIND: geom_kind, **columns}
        for col, val in ((Col.IMAGE_ID, image_id), (Col.PREV_OBJECT_ID, prev_object_id), (Col.TIMESTAMP, timestamp)):
            if val is not None:
                data[col] = val
        df = gpd.GeoDataFrame(data, geometry=Col.GEOMETRY, crs=crs)
        related = {k: v for k, v in (("imagery", imagery), ("prev_objects", prev_objects)) if v is not None}
        return cls._assemble(df, object_id, related)
