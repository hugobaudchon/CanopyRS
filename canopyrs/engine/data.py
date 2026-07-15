"""The v3 relational data model: three typed tables threaded through the pipeline.

Everything is a ``Table`` — a (Geo)DataFrame plus a primary key:
  - ``Sources``  : the input files (root; no parents).
  - ``Tiles``    : spatial footprints, each read from a Source.
  - ``Objects``  : detected/segmented things (boxes or masks), born in a Tile, optionally derived
                   from a previous Object.

Relations are foreign-key *columns* (``source_id`` / ``tile_id`` / ``prev_object_id``) plus a hydrated
pointer attribute set at build time (``tiles.sources``, ``objects.tiles``, ``objects.prev_objects``).
The column is the source of truth; the pointer is convenience. Objects can reach data produced several
steps back by walking the ``prev_objects`` ancestry — see ``provides`` / ``column`` / ``linked``.

Each table also acts as a *descriptor* (``provides`` / ``has_link`` / ``crs_set``) so a ``Need`` from
``contracts`` can check it directly; the static counterpart is a ``Schema``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.windows import Window

from canopyrs.engine.constants import Col, GEOM_KINDS
from canopyrs.engine.contracts import Schema
from canopyrs.engine.tilemeta import window_meta

RGB = [1, 2, 3]


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

    # --- descriptor interface (also implemented by Schema) — what a Need.check asks of a table -------
    def provides(self, col) -> bool:
        """Whether this table exposes usable values for ``col`` (present and non-null). Objects also
        searches its ancestry."""
        return _has_usable_values(self.df, col)

    def linked(self, name):
        """The table at relation ``name`` — hydrated directly, or resolved back through the
        ``prev_objects`` ancestry (a derived object inherits its predecessor's links, so e.g. an
        aggregated box finds the grid ``tiles`` of the detection it kept, without anyone re-attaching
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

    def schema(self) -> Schema:
        """A static snapshot — usable columns, hydrated links, CRS-ness — for the pipeline's
        ``validate()``. (Ancestry columns aren't enumerated; seeds are roots.)"""
        cols = {c for c in self.df.columns if self.provides(c)}
        links = {name for name in self.fks if self.has_link(name)}
        return Schema(columns=cols, links=links, crs=self.crs_set)

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        return f"{type(self).__name__}({len(self.df)} rows)"


class Sources(Table):
    """Input files: one row per (modality, timestamp). Root — no foreign keys."""

    pk = "source_id"

    @classmethod
    def from_paths(cls, sources) -> "Sources":
        """A Sources table from a raster path, a list of paths, or ``{path, modality, timestamp}``
        descriptors (a single RGB raster is the common case). A Sources instance passes through."""
        if isinstance(sources, cls):
            return sources
        if isinstance(sources, (str, Path)):
            sources = [sources]
        rows = [s if isinstance(s, dict) else {"path": s} for s in sources]
        data = {Col.SOURCE_PATH: [str(row["path"]) for row in rows]}
        if any("modality" in row for row in rows):
            data[Col.MODALITY] = [row.get("modality", "rgb") for row in rows]
        if any("timestamp" in row for row in rows):
            data[Col.TIMESTAMP] = [row.get("timestamp") for row in rows]
        return cls.with_ids(pd.DataFrame(data))


class Tiles(Table):
    """Spatial footprints, always read from a Source. Tiles never point to Objects: the tile↔object
    relationship is one-tile-to-many-objects, so the foreign key always lives on the Object side
    (``tile_id``) — even a one-crop-per-object tile is reached as ``object.tiles``, never the reverse."""

    pk = "tile_id"
    fks = {"sources": "source_id"}
    sources: Optional[Sources] = None

    def reading_frame(self) -> pd.DataFrame:
        """One row per tile with everything the image loader needs, keyed by ``tile_id``: the tile's
        window (``tile_metadata``), ``bands``, an optional pre-cut ``tile_path``, and — when the tiles
        are linked to their Sources — the ``source_path`` to read the window from. A tile is read from
        its ``tile_path`` if it has one, else from its source window, so a pre-cut source-less tile is
        fine."""
        cols = [c for c in (Col.TILE_ID, Col.SOURCE_ID, Col.TILE_METADATA, Col.BANDS, Col.TILE_PATH)
                if c in self.df.columns]
        frame = self.df[cols]
        if self.sources is not None:
            frame = frame.merge(self.sources.df[[Col.SOURCE_ID, Col.SOURCE_PATH]], on=Col.SOURCE_ID, how="left")
        return frame

    @classmethod
    def from_tiles_dir(cls, path, bands=RGB) -> "Tiles":
        """Seed a Tiles table from a folder of pre-cut georeferenced GeoTIFF tiles (e.g. a geodataset
        tiles output). Each tile's window metadata is recovered from the file itself (full-raster
        window), and ``tile_path`` points at it. No Source link: the loader reads each ``tile_path``
        directly and the aggregator georeferences from ``TILE_METADATA``, so a source-less tile is fine."""
        paths = sorted(p for pattern in ("*.tif", "*.tiff") for p in Path(path).glob(pattern))
        if not paths:
            raise ValueError(f"no .tif/.tiff tiles found in {path}")
        metadata, tile_paths = [], []
        for p in paths:
            with rasterio.open(p) as src:
                metadata.append(window_meta(src, Window(0, 0, src.width, src.height)))
            tile_paths.append(str(p))
        return cls.build(source_id=[None] * len(paths), tile_metadata=metadata,
                         tile_path=tile_paths, bands=bands)

    @classmethod
    def build(cls, *, source_id, tile_metadata, tile_path=None, bands=RGB, modality="rgb",
              timestamp=None, tile_id=None, sources=None) -> "Tiles":
        """Construct flat tiles from per-row arrays — one (modality, timestamp) per row. ``source_id``
        (and ``modality``/``timestamp``) may be a scalar (broadcast) or per-row. ``tile_id`` keeps given
        ids (e.g. geodataset's), else a fresh 0..n is stamped."""
        n = len(tile_metadata)
        data = {
            Col.SOURCE_ID: source_id,
            Col.MODALITY: modality,
            Col.TIMESTAMP: timestamp,
            Col.BANDS: [bands] * n,
            Col.TILE_PATH: tile_path if tile_path is not None else [None] * n,
            Col.TILE_METADATA: list(tile_metadata),
        }
        related = {"sources": sources} if sources is not None else {}
        return cls._assemble(pd.DataFrame(data), tile_id, related)


class Objects(Table):
    """Detected/segmented things (boxes or masks). Born in a Tile; may derive from a previous Object."""

    pk = "object_id"
    fks = {"tiles": "tile_id", "prev_objects": "prev_object_id"}
    tiles: Optional[Tiles] = None
    prev_objects: Optional[Objects] = None

    def __init__(self, df, **related):
        super().__init__(df, **related)
        if Col.GEOM_KIND not in df.columns:
            raise ValueError(f"Objects requires a '{Col.GEOM_KIND}' column")
        bad = set(df[Col.GEOM_KIND].dropna()) - GEOM_KINDS   # column mandatory; None values = unknown, allowed
        if bad:
            raise ValueError(f"unknown {Col.GEOM_KIND} {sorted(bad)}, expected {sorted(GEOM_KINDS)}")

    def provides(self, col) -> bool:
        """``col`` is exposed here (present, non-null) or resolvable through the ``prev_objects`` ancestry."""
        if col in self.df.columns:
            return _has_usable_values(self.df, col)
        return self.prev_objects is not None and self.prev_objects.provides(col)

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
    def from_gpkg(cls, path, tiles=None, prev_objects=None) -> "Objects":
        """Seed an Objects table from a GeoPackage previously written by the pipeline (round-trip): it
        carries geometry, ``geom_kind``, ``object_id`` and any FK columns. Pass ``tiles`` /
        ``prev_objects`` to re-link whichever FK columns the file holds (so the ancestry walk works)."""
        gdf = gpd.read_file(path)
        if Col.GEOM_KIND not in gdf.columns:
            raise ValueError(f"{path} has no '{Col.GEOM_KIND}' column; not a pipeline-written Objects gpkg")
        related = {}
        if tiles is not None and Col.TILE_ID in gdf.columns:
            related["tiles"] = tiles
        if prev_objects is not None and Col.PREV_OBJECT_ID in gdf.columns:
            related["prev_objects"] = prev_objects
        ids = gdf[Col.OBJECT_ID].values if Col.OBJECT_ID in gdf.columns else None
        return cls._assemble(gdf, ids, related)

    @classmethod
    def build(cls, *, geometry, geom_kind, tile_id=None, prev_object_id=None, timestamp=None,
              object_id=None, crs=None, tiles=None, prev_objects=None, **columns) -> "Objects":
        """Construct objects from geometry + the usual keys, plus any extra attribute columns (scores,
        classes, ...) by Col name. ``object_id`` keeps given ids, else a fresh 0..n is stamped."""
        data = {Col.GEOMETRY: list(geometry), Col.GEOM_KIND: geom_kind, **columns}
        for col, val in ((Col.TILE_ID, tile_id), (Col.PREV_OBJECT_ID, prev_object_id), (Col.TIMESTAMP, timestamp)):
            if val is not None:
                data[col] = val
        df = gpd.GeoDataFrame(data, geometry=Col.GEOMETRY, crs=crs)
        related = {k: v for k, v in (("tiles", tiles), ("prev_objects", prev_objects)) if v is not None}
        return cls._assemble(df, object_id, related)
