"""
Image loader for inference.

A component needs many windows of a raster. Reading each on its own is wasteful when they overlap or
sit close together, so windows are read in groups: each read covers a set of them and they are sliced
out of it.

Windows are first grouped by file, bands and resolution — one group per tiling level, so a run mixing
10cm and 3cm tiles just makes two groups. Each group is then swept top to bottom in bands, and each
band split left to right into runs of nearby windows. One run is one read, sized to exactly the windows
it holds: a sparse scene reads small rectangles, a dense one reads big ones, with no separate code path.

Where a run may be split depends on the raster. A read costs the source blocks it touches, so on a
tiled raster a narrow read is cheap and gaps are worth splitting on, while on a striped one a block
spans the full width and splitting a band would only re-read the same strips.

Rows whose ``path`` is set are whole images already: read directly, nothing to slice.
"""

import math
import warnings
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from canopyrs.engine.constants import Col, RGB_BANDS
from canopyrs.engine.tilemeta import transform_of
from canopyrs.engine.utils import worker_context

# Native megabytes one read may decode — the only size knob, overridable per run (``read_mb``). A
# worker holds two of these (it fetches the next while the model consumes the current), so peak memory
# is roughly 2 x this x workers, divided by however much the resampling shrinks it.
DEFAULT_READ_MB = 256

# __len__ counts images so the progress bar has a total. Each worker batches its own share and ends on
# a partial batch, so the batch count runs a little over. Expected.
warnings.filterwarnings("ignore", category=UserWarning, message="Length of IterableDataset")

Cut = namedtuple("Cut", "image_id row col height width")   # a window, in its source's grid pixels
Read = namedtuple("Read", "row col rows cols cuts")        # one rectangle, and what to slice from it


@dataclass
class Source:
    """Windows sharing one file, band set and resolution — that is, one tiling level.

    ``res_y`` is positive (rows increase downward), and a cut's row/col are grid pixels from the
    origin, which is the top-left corner of the windows' shared grid.
    """
    path: str
    bands: list
    res_x: float
    res_y: float
    origin_x: float
    origin_y: float
    cuts: list
    read_bytes: int

    @classmethod
    def from_frame(cls, frame, read_bytes):
        """``(direct, sources)``: whole images to read as they are, and windows grouped per source."""
        direct, groups = [], {}
        for row in frame.itertuples(index=False):
            bands = getattr(row, Col.BANDS, None)
            bands = tuple(RGB_BANDS) if bands is None else tuple(int(b) for b in bands)
            path = _clean(getattr(row, Col.PATH, None))
            if path is not None:
                direct.append((row.image_id, path, list(bands)))
                continue
            read_path = _clean(getattr(row, Col.READ_PATH, None))
            meta = getattr(row, Col.METADATA, None)
            if read_path is None or meta is None:
                raise ValueError(f"image {row.image_id} has no path of its own and no window to read")
            transform = transform_of(meta)
            key = (read_path, bands, round(transform.a, 9), round(-transform.e, 9))
            groups.setdefault(key, []).append((row.image_id, meta, transform))

        sources = []
        for (path, bands, res_x, res_y), rows in groups.items():
            origin_x, origin_y = min(t.c for _, _, t in rows), max(t.f for _, _, t in rows)
            cuts = [Cut(image_id, round((origin_y - t.f) / res_y), round((t.c - origin_x) / res_x),
                        int(meta["height"]), int(meta["width"])) for image_id, meta, t in rows]
            sources.append(cls(path, list(bands), res_x, res_y, origin_x, origin_y, cuts, read_bytes))
        return direct, sources

    def budget(self, src):
        """Grid pixels one read may cover, from the native pixels it would have to decode."""
        scale = self.res_x / src.transform.a                 # native pixels per grid pixel
        return self.read_bytes / (len(self.bands) * np.dtype(src.dtypes[0]).itemsize) / scale ** 2

    def reads(self, src, workers=1):
        """Rectangles as large as the budget allows: bands down the rows, runs across each band.

        A band is never shorter than one window, which is atomic. Splitting a band by column is only
        sound on a tiled raster, where the chunks touch different blocks; on a striped one they all
        touch the same full-width strips, so the band stays whole and may run over budget.
        """
        budget = self.budget(src)
        tallest = max(cut.height for cut in self.cuts)
        widest = max(cut.width for cut in self.cuts)
        span = max(c.col + c.width for c in self.cuts) - min(c.col for c in self.cuts)
        block_cols = src.block_shapes[0][1]
        tiled = block_cols < src.width

        # Striped bands are full width regardless, so their height is what the budget leaves after
        # that. Tiled ones can be square, which keeps them tall enough to group on a very wide raster.
        height = max(tallest, int(budget ** 0.5) if tiled else int(budget / span))
        bands = list(_split(sorted(self.cuts, key=lambda cut: cut.row),
                            lambda cut: (cut.row, cut.height), height))
        share = max(1, 2 * workers // max(1, len(bands)))    # aim past one read per worker, if we can

        reads = []
        for band in bands:
            rows = max(c.row + c.height for c in band) - min(c.row for c in band)
            width = max(c.col + c.width for c in band) - min(c.col for c in band)
            extent = max(widest, min(budget / rows, width / share)) if tiled else math.inf
            for run in _split(sorted(band, key=lambda cut: cut.col), lambda cut: (cut.col, cut.width),
                              extent, block_cols if tiled else math.inf):
                row, col = min(c.row for c in run), min(c.col for c in run)
                reads.append(Read(row, col, max(c.row + c.height for c in run) - row,
                                  max(c.col + c.width for c in run) - col, run))
        return reads

    def read(self, src, rect):
        """``rect`` resampled onto this grid. Anything outside the raster reads as zero."""
        left, top = self.origin_x + rect.col * self.res_x, self.origin_y - rect.row * self.res_y
        bounds = (left, top - rect.rows * self.res_y, left + rect.cols * self.res_x, top)
        return src.read(self.bands, window=from_bounds(*bounds, transform=src.transform),
                        out_shape=(len(self.bands), rect.rows, rect.cols),
                        boundless=True, fill_value=0, resampling=Resampling.bilinear)


class WindowDataset(IterableDataset):
    """Yields ``(image_id, image)`` for a reading frame; each worker takes a share of the reads."""

    def __init__(self, frame, read_mb=DEFAULT_READ_MB):
        self.direct, self.sources = Source.from_frame(frame, read_mb * 1024 ** 2)

    def __len__(self):
        return len(self.direct) + sum(len(source.cuts) for source in self.sources)

    def __iter__(self):
        info = get_worker_info()
        index, count = (info.id, info.num_workers) if info else (0, 1)

        for image_id, path, bands in self.direct[index::count]:
            with rasterio.open(path) as src:
                yield image_id, _to_model(src.read(bands))

        for source in self.sources:
            with rasterio.open(source.path) as src:
                reads = source.reads(src, count)
                if index == 0:
                    self._report(src, source, reads)
                yield from self._sweep(src, source, reads[index::count])

    def _sweep(self, src, source, reads):
        """Yield each read's windows, fetching the next read in a thread so it never stalls the model.

        Only that thread touches ``src`` (rasterio datasets are not thread-safe); this loop slices
        numpy, and GDAL releases the GIL, so the two really do overlap.
        """
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(source.read, src, reads[0]) if reads else None
            for position, rect in enumerate(reads):
                data = pending.result()
                pending = (pool.submit(source.read, src, reads[position + 1])
                           if position + 1 < len(reads) else None)
                for cut in rect.cuts:
                    row, col = cut.row - rect.row, cut.col - rect.col
                    yield cut.image_id, _to_model(data[:, row:row + cut.height, col:col + cut.width])

    @staticmethod
    def _report(src, source, reads):
        """One line per source: how its windows grouped, and whether a read had to go over budget."""
        biggest = max(rect.rows * rect.cols for rect in reads)
        held = biggest * len(source.bands) * np.dtype(src.dtypes[0]).itemsize
        over = "" if biggest <= source.budget(src) else ", over the read budget"
        print(f"  loader: {len(source.cuts)} windows at {source.res_x:.3g}m in {len(reads)} reads, "
              f"up to {held / 1024 ** 2:.0f}MB held{over}")


def _clean(value):
    """A path value, or None (treats NaN / empty as missing)."""
    return value if (value is not None and value == value and value != "") else None


def _split(cuts, axis, extent, gap=math.inf):
    """Cuts sorted along ``axis``, split when the group would pass ``extent`` or leave a wider gap.

    Used down the rows to make bands, then across the columns to make runs — same rule both ways.
    """
    group, edge = [], 0
    for cut in cuts:
        start, size = axis(cut)
        if group and (start + size - axis(group[0])[0] > extent or start - edge > gap):
            yield group
            group, edge = [], 0
        group.append(cut)
        edge = max(edge, start + size)
    if group:
        yield group


def _to_model(tile):
    """A model-ready copy: ``[C, H, W]`` float in 0..1. Copied because the read buffer is reused."""
    return tile.astype(np.float32) / 255.0


def collate(batch):
    """(list[image_id], list[image tensor]) — images stay a list; the model's forward takes one."""
    return [image_id for image_id, _ in batch], [torch.from_numpy(image) for _, image in batch]


def image_loader(frame, batch_size=8, num_workers=4, read_mb=DEFAULT_READ_MB):
    """DataLoader over a reading frame, yielding ``(image_ids, images)`` batches. ``read_mb`` is the
    native megabytes one read may decode (see ``DEFAULT_READ_MB``)."""
    return DataLoader(WindowDataset(frame, read_mb), batch_size=batch_size, num_workers=num_workers,
                      collate_fn=collate, persistent_workers=bool(num_workers),
                      multiprocessing_context=worker_context(num_workers))
