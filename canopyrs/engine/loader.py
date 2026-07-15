"""
Tile image loader for v3 inference.

A torch ``Dataset`` over a tiles reading frame (``Tiles.reading_frame()`` — one row per tile), turning
each row into a model-ready image one of two ways:
  - the row has a ``tile_path``    -> read that file from disk (pre-cut tile);
  - else, from its ``source_path`` -> read the tile's window (``tile_metadata`` bounds, resampled to
    the tile's GSD) straight from the source raster.

Each image travels with its ``tile_id``, so the caller maps predictions back to the originating tile
(the FK on produced objects) without any path matching.

Reads are plain rasterio windowed reads. With a tiled COG, the GDAL block cache + OS page cache make
overlapping windows read by multiple DataLoader workers cheap — so the source raster should be a COG.
"""

from torch.utils.data import DataLoader, Dataset
import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

from canopyrs.engine.constants import Col
from canopyrs.engine.tilemeta import bounds_of

RGB = [1, 2, 3]


def _clean(v):
    """A path value, or None (treats NaN / empty as missing)."""
    return v if (v is not None and v == v and v != "") else None


class TileDataset(Dataset):
    """Yields ``(tile_id, image)`` over a reading frame (``Tiles.reading_frame()``); image is
    ``[C, H, W]`` float in 0..1. Bands are per-row, so modalities differ freely."""

    def __init__(self, frame):
        # Pull plain lists so workers don't carry the whole frame / do geometry math.
        n = len(frame)
        cols = frame.columns
        metas = list(frame[Col.TILE_METADATA]) if Col.TILE_METADATA in cols else [None] * n
        self.tile_ids = list(frame[Col.TILE_ID])
        self.tile_paths = [_clean(v) for v in frame[Col.TILE_PATH]] if Col.TILE_PATH in cols else [None] * n
        self.source_paths = [_clean(v) for v in frame[Col.SOURCE_PATH]] if Col.SOURCE_PATH in cols else [None] * n
        self.bounds = [bounds_of(m) if m is not None else None for m in metas]   # window from tile metadata
        self.sizes = [(m["height"], m["width"]) if m is not None else None for m in metas]  # tile's GSD grid
        self.bands = list(frame[Col.BANDS]) if Col.BANDS in cols else [RGB] * n   # per-row band indices
        self._handles = {}  # source_path -> open dataset, lazily, once per worker

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, idx):
        bands = self.bands[idx]
        tile_path = self.tile_paths[idx]
        if tile_path is not None:
            with rasterio.open(tile_path) as src:
                tile = src.read(bands)
        else:
            tile = self._read_window(self.source_paths[idx], self.bounds[idx], self.sizes[idx], bands)
        return self.tile_ids[idx], tile.astype(np.float32) / 255.0

    def _read_window(self, source_path, bounds, size, bands):
        if source_path is None or bounds is None:
            raise ValueError("tile has neither a tile_path nor a (source_path, window) to read from")
        src = self._handles.get(source_path)
        if src is None:
            src = self._handles[source_path] = rasterio.open(source_path)  # per-worker, per-source
        window = from_bounds(*bounds, transform=src.transform)
        # out_shape resamples the native-resolution window to the tile's pixel grid, i.e. to the tile's
        # GSD (the metadata's height/width) — so a resampled tilerizer (ground_resolution) and the
        # on-disk tiles agree. Bilinear matches geodataset's interpolated tiling, not nearest.
        h, w = size
        return src.read(bands, window=window, out_shape=(len(bands), h, w),
                        boundless=True, fill_value=0, resampling=Resampling.bilinear)


def collate(batch):
    """(list[tile_id], list[image tensor]) — images stay a list; the model's forward takes one."""
    tile_ids = [tile_id for tile_id, _ in batch]
    images = [torch.from_numpy(image) for _, image in batch]
    return tile_ids, images


def tile_loader(frame, batch_size=8, num_workers=4):
    """DataLoader over a reading frame (``Tiles.reading_frame()``); yields ``(tile_ids, images)``
    batches in row order."""
    ds = TileDataset(frame)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      collate_fn=collate, persistent_workers=bool(num_workers))
