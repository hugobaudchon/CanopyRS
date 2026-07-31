"""
Image loader for inference.

A torch ``Dataset`` over an imagery reading frame (``Imagery.reading_frame()`` — one row per image),
turning each row into a model-ready image one of two ways:
  - the row has its own ``path``      -> read that file from disk (materialized image);
  - else, from its ``read_path``      -> read the region's window (``metadata`` bounds, resampled to
    the region's grid) straight from the nearest materialized ancestor (its tile or source raster).

Each image travels with its ``image_id``, so the caller maps predictions back to the originating image
(the FK on produced objects) without any path matching.

Reads are plain rasterio windowed reads. With a tiled COG, the GDAL block cache + OS page cache make
overlapping windows read by multiple DataLoader workers cheap — so the source raster should be a COG.
The window read assumes the region shares its ancestor's CRS (true for tiles/crops cut from it).
"""

import time

from torch.utils.data import DataLoader, Dataset
import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from tqdm import tqdm

from canopyrs.engine.constants import Col, RGB_BANDS
from canopyrs.engine.tilemeta import bounds_of



def _clean(v):
    """A path value, or None (treats NaN / empty as missing)."""
    return v if (v is not None and v == v and v != "") else None


class TileDataset(Dataset):
    """Yields ``(image_id, image)`` over a reading frame (``Imagery.reading_frame()``); image is
    ``[C, H, W]`` float in 0..1. Bands are per-row, so modalities differ freely."""

    def __init__(self, frame):
        # Pull plain lists so workers don't carry the whole frame / do geometry math.
        n = len(frame)
        cols = frame.columns
        metas = list(frame[Col.METADATA]) if Col.METADATA in cols else [None] * n
        self.image_ids = list(frame[Col.IMAGE_ID])
        self.own_paths = [_clean(v) for v in frame[Col.PATH]] if Col.PATH in cols else [None] * n
        self.read_paths = [_clean(v) for v in frame[Col.READ_PATH]] if Col.READ_PATH in cols else [None] * n
        self.bounds = [bounds_of(m) if m is not None else None for m in metas]   # window from metadata
        self.sizes = [(m["height"], m["width"]) if m is not None else None for m in metas]  # region's grid
        self.bands = list(frame[Col.BANDS]) if Col.BANDS in cols else [RGB_BANDS] * n   # per-row band indices
        self._handles = {}  # read_path -> open dataset, lazily, once per worker

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        bands = self.bands[idx]
        own_path = self.own_paths[idx]
        if own_path is not None:
            with rasterio.open(own_path) as src:
                tile = src.read(bands)
        else:
            tile = self._read_window(self.read_paths[idx], self.bounds[idx], self.sizes[idx], bands)
        return self.image_ids[idx], tile.astype(np.float32) / 255.0

    def _read_window(self, read_path, bounds, size, bands):
        if read_path is None or bounds is None:
            raise ValueError("image has neither its own path nor a (read_path, window) to read from")
        src = self._handles.get(read_path)
        if src is None:
            src = self._handles[read_path] = rasterio.open(read_path)  # per-worker, per-file
        window = from_bounds(*bounds, transform=src.transform)
        # out_shape resamples the native-resolution window to the region's pixel grid, i.e. to its
        # GSD (the metadata's height/width) — so a resampled tilerizer (ground_resolution) and the
        # on-disk tiles agree. Bilinear matches geodataset's interpolated tiling, not nearest.
        h, w = size
        return src.read(bands, window=window, out_shape=(len(bands), h, w),
                        boundless=True, fill_value=0, resampling=Resampling.bilinear)


def collate(batch):
    """(list[image_id], list[image tensor]) — images stay a list; the model's forward takes one."""
    image_ids = [image_id for image_id, _ in batch]
    images = [torch.from_numpy(image) for _, image in batch]
    return image_ids, images


def tile_loader(frame, batch_size=8, num_workers=4):
    """DataLoader over a reading frame (``Imagery.reading_frame()``); yields ``(image_ids, images)``
    batches in row order."""
    ds = TileDataset(frame)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      collate_fn=collate, persistent_workers=bool(num_workers))


class InferTimer:
    """Where an inference loop spent its time, on one line.

    Iterate with ``batches``, ``mark`` after each step of the body (and after the loop, for tail
    work), then ``report``::

        timer = InferTimer("Inferring detector...")
        for object_ids, images in timer.batches(loader):
            images = [img.to(self.device) for img in images]
            timer.mark("prep")
            results.extend(self.forward(images))
            timer.mark("gpu")
        boxes, scores, classes = detector_result_to_lists(results)
        timer.mark("post")
        timer.report()

    Every key is main-process wall time and they sum to the whole loop: the marked stages, plus
    ``data_wait`` (blocked on the loader) and ``other`` (unmarked — should stay near zero, so a
    nonzero one means a missing ``mark``). Marks sync CUDA, so async kernels are billed to the stage
    that launched them.
    """

    def __init__(self, desc):
        self.desc = desc
        self.stages = {}                              # stage -> seconds, in the order first marked
        self.data_wait = 0.0
        self.other = 0.0
        self.n_batches = 0
        self.workers = 0
        self._sync = torch.cuda.is_available()
        self._last = time.perf_counter()

    def batches(self, loader):
        """``tqdm(loader)``, charging the block on each batch to ``data_wait``."""
        self.workers = loader.num_workers
        self._last = time.perf_counter()
        for batch in tqdm(loader, desc=self.desc, leave=True):
            self.data_wait += self._split()           # blocked until this batch arrived
            yield batch                               # the caller runs the model, marking stages
            self.other += self._split()               # body time past the caller's last mark
            self.n_batches += 1
        self.data_wait += self._split()

    def mark(self, stage):
        """Charge the time since the last mark (or since the batch arrived) to ``stage``."""
        if self._sync:
            torch.cuda.synchronize()
        self.stages[stage] = self.stages.get(stage, 0.0) + self._split()

    def report(self):
        if not self.n_batches:
            return
        self.other += self._split()                   # tail work the caller didn't mark
        wall = sum(self.stages.values()) + self.data_wait + self.other
        timings = {"data_wait": self.data_wait, **self.stages, "other": self.other}
        parts = [f"{stage} {s:.1f}s ({s / wall:.0%})" for stage, s in timings.items()]
        print(f"{self.desc.rstrip('. ')} timing: {self.n_batches} batches in {wall:.1f}s "
              f"({self.workers} loader workers) | " + " | ".join(parts))

    def _split(self):
        now = time.perf_counter()
        elapsed = now - self._last
        self._last = now
        return elapsed
