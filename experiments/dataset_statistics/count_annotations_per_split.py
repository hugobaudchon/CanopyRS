import re
from pathlib import Path
from collections import Counter
import geopandas as gpd
from shapely.ops import unary_union


ROOT = "/media/hugo/Hard Disk 1/CanopyRS/data/raw"
RASTER_NAMES = [
    "20240130_zf2quad_m3m",
    "20240130_zf2tower_m3m",
    "20240130_zf2transectew_m3m",
    "20240131_zf2campirana_m3m",
    "20170810_transectotoni_mavicpro",
    "20230525_tbslake_m3e",
    "20230911_sanitower_mini2",
    "20231018_inundated_m3e",
    "20231018_pantano_m3e",
    "20231018_terrafirme_m3e",
    "20231207_asnortheast_amsunclouds_m3m",
    "20231207_asnorthnorth_pmclouds_m3m",
    "20231208_asforestnorthe2_m3m",
    "20231208_asforestsouth2_m3m",
]

SPLIT_RE = re.compile(r"_aoi_(train|valid|val|test)", re.IGNORECASE)
SPLIT_ALIAS = {"val": "valid"}            # unify naming


def split_from_fname(fname: str):
    m = SPLIT_RE.search(fname)
    if not m:
        return None
    split = m.group(1).lower()
    return SPLIT_ALIAS.get(split, split)   # map “val”→“valid”


def load_union(paths):
    """
    Return the unary union of geometries in the list of GPKG files.
    If the list is empty, or all geometries are null/empty, return None.
    """
    if not paths:
        return None

    geoms = []
    for p in paths:
        gdf = gpd.read_file(p, engine="pyogrio")
        geoms.extend(
            g
            for g in gdf.geometry
            if g is not None and not g.is_empty
        )

    if not geoms:                # nothing valid to union
        return None

    return unary_union(geoms)


root = Path(ROOT).expanduser()
grand_total_per_split = Counter()
grand_total_even_outside_split = 0

for raster in RASTER_NAMES:
    # annotation GPKG (exact labels, NOT AOI)
    ann_files = list(root.rglob(f"{raster}_labels_boxes.gpkg"))
    for ann_path in ann_files:
        # AOIs for this annotation
        parent = ann_path.parent
        stem = ann_path.stem                     # e.g. “…labels_boxes”
        aoi_candidates = sorted(parent.glob(f"{stem}_aoi_*.gpkg"))

        split_files = {"train": [], "valid": [], "test": []}
        for p in aoi_candidates:
            split = split_from_fname(p.name)
            if split:
                split_files[split].append(p)

        # build unions per split
        unions = {s: load_union(paths) for s, paths in split_files.items()}

        # iterate annotations
        ann_gdf = gpd.read_file(ann_path, engine="pyogrio")
        counts = Counter()
        for geom in ann_gdf.geometry:
            best_split, best_area = None, 0.0
            for s, u in unions.items():
                if u is None:
                    continue
                area = geom.intersection(u).area
                if area > best_area:
                    best_area = area
                    best_split = s
            if best_split:
                counts[best_split] += 1

        total = len(ann_gdf)
        grand_total_per_split.update(counts)
        grand_total_even_outside_split += total

        # report this annotation file
        print(f"\n{ann_path.name}:")
        print(f"  total annotations: {total}")
        for s in ("train", "valid", "test"):
            print(f"    {s:5}: {counts[s]}")

# global summary
print("\n=== Totals over all annotation files ===")
for s in ("train", "valid", "test"):
    print(f"{s:5}: {grand_total_per_split[s]}")
print(f"total annotations: {grand_total_even_outside_split}")
