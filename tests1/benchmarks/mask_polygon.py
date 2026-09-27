"""Speed and accuracy of canopyrs1's polygon <-> mask conversion, on real tree crowns.

Uses the SelvaMask crown polygons cached by the benchmark tests (~/.cache/canopyrs_test_data).
Masks cover a whole tile, as the segmenters produce them: at the tile size (1777 px) and at the
segmenter's default downscale (512 px). Mask -> polygon uses the segmenter's default settings (no
simplification, holes filled, extra parts under 50 px dropped). If geodataset is installed, its
versions are compared too.

    python -W ignore tests1/benchmarks/mask_polygon.py [number_of_crowns]
"""

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from rasterio import features
from shapely.affinity import scale
from shapely.geometry import Polygon

from canopyrs1.core.geometry.masks import mask_to_polygon, polygon_to_mask

try:
    from geodataset.utils import mask_to_polygon as old_mask_to_polygon
    from geodataset.utils import polygon_to_mask as old_polygon_to_mask
except ImportError:
    old_mask_to_polygon = old_polygon_to_mask = None

CROWNS = Path.home() / ".cache/canopyrs_test_data/benchmarks/datasets/selvamask"
TILE_SIZE = 1777


def load_crowns(n):
    """Return ``n`` random valid crown polygons (pixel coordinates in a 1777 px tile)."""
    crowns = []
    for path in sorted(glob.glob(str(CROWNS / "*/*_test.json"))):
        for annotation in json.load(open(path))["annotations"]:
            for coords in annotation["segmentation"]:
                crown = Polygon(np.array(coords).reshape(-1, 2))
                if crown.is_valid and crown.area > 0:
                    crowns.append(crown)
    if not crowns:
        sys.exit(f"No crowns found under {CROWNS}: run the benchmark tests once to download SelvaMask.")
    rng = np.random.default_rng(0)
    return [crowns[i] for i in rng.choice(len(crowns), size=min(n, len(crowns)), replace=False)]


def pixel_iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    union = (a | b).sum()
    return 1.0 if union == 0 else (a & b).sum() / union      # two empty masks agree


def timed(fn, arg):
    """Return fn(arg) and how long the call took, in ms."""
    start = time.perf_counter()
    out = fn(arg)
    return out, (time.perf_counter() - start) * 1000


def main():
    crowns = load_crowns(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
    print(f"{len(crowns)} SelvaMask crowns, median area {np.median([c.area for c in crowns]):.0f} px "
          f"in a {TILE_SIZE} px tile. Each crown goes through every variant in turn; only the call is timed.\n")

    for size in (TILE_SIZE, 512):
        polygons = [scale(c, xfact=size / TILE_SIZE, yfact=size / TILE_SIZE, origin=(0, 0)) for c in crowns]
        # Reference mask: rasterio on the whole tile (a pixel is inside if its centre is).
        reference = lambda p: features.rasterize([p], out_shape=(size, size), dtype="uint8")
        to_mask = {"canopyrs1": lambda p: polygon_to_mask(p, size, size)}
        to_polygon = {"canopyrs1": lambda m: mask_to_polygon(m, min_part_area=50)}
        if old_polygon_to_mask:
            to_mask["old geodataset"] = lambda p: old_polygon_to_mask(p, size, size)
            to_polygon["old geodataset"] = lambda m: old_mask_to_polygon(m, simplify_tolerance=0.0,
                                                                         remove_rings=True,
                                                                         remove_small_geoms=50)
        for polygon in polygons[:5]:                                        # warm-up
            ref = reference(polygon)
            [fn(polygon) for fn in to_mask.values()] + [fn(ref) for fn in to_polygon.values()]

        mask_stats = {name: np.zeros(3) for name in to_mask}                # ms, pixel ratio, IoU
        polygon_stats = {name: np.zeros(3) for name in to_polygon}
        for polygon in polygons:
            ref = reference(polygon)
            for name, fn in to_mask.items():
                mask, ms = timed(fn, polygon)
                mask_stats[name] += (ms, mask.sum() / polygon.area, pixel_iou(mask, ref))
            for name, fn in to_polygon.items():
                out, ms = timed(fn, ref)
                polygon_stats[name] += (ms, out.area / polygon.area,
                                        out.intersection(polygon).area / out.union(polygon).area)

        n = len(polygons)
        print(f"== masks of {size} x {size} px")
        print("  polygon -> mask    ms/crown   pixels / crown area   pixel IoU vs reference")
        for name, (ms, ratio, iou) in mask_stats.items():
            print(f"    {name:<16} {ms / n:7.3f}    {ratio / n:.3f}                 {iou / n:.4f}")
        print("  mask -> polygon    ms/crown   area / crown area     IoU vs crown")
        for name, (ms, ratio, iou) in polygon_stats.items():
            print(f"    {name:<16} {ms / n:7.3f}    {ratio / n:.3f}                 {iou / n:.4f}")
        print()


if __name__ == "__main__":
    main()
