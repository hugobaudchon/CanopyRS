#!/usr/bin/env python
"""Shift LiDAR GT crowns onto the RGB orthomosaic, one rigid XY translation per plot.

Reads the per-plot co-registration measured by `check_alignment.py`
(`output/alignment/plot_<p>/summary.json`), picks one shift per plot, and writes
translated copies of every GT layer (visible and full).

Shift per plot = median of the trusted 10 m window shifts of estimator A (RGB vs
LiDAR colour): robust to the few windows that lock onto the wrong feature, and
spatially representative. Falls back to the global A estimate when fewer than
`--min-windows` windows were trusted. Estimators B (CHM, geometry only) and C (per
crown) are reported next to it as cross-checks, never used.

Convention (check_alignment.py): a feature at LiDAR (x, y) appears in the ortho at
(x+dx, y+dy). Translating the LiDAR crowns by (+dx, +dy) puts them on the ortho.

Idempotent by construction: it always reads the UNSHIFTED layers from
`<layer>_unshifted/` (created from `<layer>/` on first run) and overwrites `<layer>/`,
so running it twice never shifts twice. Each written feature carries `shift_dx_m` /
`shift_dy_m`, and `centroid_x` / `centroid_y` are moved with the geometry.

    /home/hugobaudchon/anaconda3/envs/canopyrs/bin/python scripts/apply_plot_shifts.py \
        --alignment-dir output/alignment --layers data/lidar_masks data/lidar_masks_full
"""
import argparse
import glob
import json
import os
import re
import shutil

import geopandas as gpd
import pandas as pd


def choose_shift(s, min_windows):
    wa = s.get("windows_A", {})
    if wa.get("n", 0) >= min_windows:
        return wa["median_dx_m"], wa["median_dy_m"], f"median of {wa['n']} A windows"
    g = s["global_A_colour"]
    return g["dx_m"], g["dy_m"], f"global A (only {wa.get('n', 0)} trusted windows)"


def collect(alignment_dir, min_windows):
    rows = []
    for f in sorted(glob.glob(os.path.join(alignment_dir, "plot_*", "summary.json"))):
        s = json.load(open(f))
        if any(s.get("lidar_shift_applied_m", [0, 0])):
            continue  # a verification run, not a measurement
        dx, dy, how = choose_shift(s, min_windows)
        wa, ga, gb, c = s.get("windows_A", {}), s["global_A_colour"], s["global_B_chm"], s.get("crowns_C", {})
        rows.append(dict(
            plot=str(s["plot"]), dx_m=dx, dy_m=dy, norm_m=(dx ** 2 + dy ** 2) ** 0.5, source=how,
            windows_A_n=wa.get("n", 0), windows_A_p95_norm_m=wa.get("p95_norm_m"),
            global_A_dx=ga["dx_m"], global_A_dy=ga["dy_m"],
            global_B_dx=gb["dx_m"], global_B_dy=gb["dy_m"], global_B_distinct=gb["distinct"],
            crowns_C_n=c.get("n", 0), crowns_C_dx=c.get("median_dx_m"), crowns_C_dy=c.get("median_dy_m"),
            self_test_error_m=s["self_test"]["error_m"]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alignment-dir", default="output/alignment")
    ap.add_argument("--layers", nargs="+", default=["data/lidar_masks", "data/lidar_masks_full"])
    ap.add_argument("--min-windows", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    shifts = collect(args.alignment_dir, args.min_windows)
    out_csv = os.path.join(args.alignment_dir, "plot_shifts.csv")
    shifts.to_csv(out_csv, index=False)
    print(shifts[["plot", "dx_m", "dy_m", "norm_m", "source", "global_B_dx", "global_B_dy",
                  "crowns_C_dx", "crowns_C_dy"]].round(3).to_string(index=False))
    print(f"-> {out_csv}")
    by_plot = shifts.set_index("plot")

    for layer in args.layers:
        layer = layer.rstrip("/")
        src_dir = layer + "_unshifted"
        if not os.path.isdir(src_dir):
            if args.dry_run:
                src_dir = layer
            else:
                shutil.copytree(layer, src_dir)
                print(f"[{layer}] first run: unshifted originals copied to {src_dir}/")
        for src in sorted(glob.glob(os.path.join(src_dir, "*.gpkg"))):
            plot = re.search(r"plot_(\d+)", os.path.basename(src)).group(1)
            if plot not in by_plot.index:
                print(f"[{layer}] {os.path.basename(src)}: no measured shift for plot {plot} -- left unshifted")
                continue
            dx, dy = float(by_plot.loc[plot, "dx_m"]), float(by_plot.loc[plot, "dy_m"])
            g = gpd.read_file(src, layer="crowns")
            g["geometry"] = g.geometry.translate(dx, dy)
            for col, d in (("centroid_x", dx), ("centroid_y", dy)):
                if col in g:
                    g[col] = g[col] + d
            g["shift_dx_m"], g["shift_dy_m"] = dx, dy
            dst = os.path.join(layer, os.path.basename(src))
            if args.dry_run:
                print(f"[{layer}] would write {dst} shifted by ({dx:+.3f}, {dy:+.3f}) m")
                continue
            if os.path.exists(dst):
                os.remove(dst)
            for side in (dst + "-wal", dst + "-shm"):
                if os.path.exists(side):
                    os.remove(side)
            # Keep each feature's own type: by default a mixed Polygon/MultiPolygon layer is
            # promoted to all-MultiPolygon on write. The source layer is generic GEOMETRY.
            g.to_file(dst, layer="crowns", driver="GPKG", engine="pyogrio",
                      promote_to_multi=False, geometry_type="Unknown")
            print(f"[{layer}] {os.path.basename(dst)} shifted by ({dx:+.3f}, {dy:+.3f}) m")


if __name__ == "__main__":
    main()
