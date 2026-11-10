#!/usr/bin/env python
"""Why do neighbouring LiDAR GT crowns look like they overlap at their edges?

Diagnostic for the GT polygons written by ForestMamba/tools/extract_crown_hulls.py --gt.
That tool builds each crown's polygon independently (rasterise at 0.15 m, close,
trace, smooth with a buffer pass at 1.5 x 0.15 = 0.225 m), then `resolve_top_view`
processes crowns tallest-first and replaces each shorter crown by
`raw - union(taller raws)`. This script measures, per plot:

  1. true area overlap between GT crowns (pairwise intersection area);
  2. shared boundaries: how much of each crown's outline coincides with a neighbour's;
  3. thin ribbons: parts of a crown narrower than 2*r (default r = 0.2 m), found as
     geom - opening(geom, r). The smoothing pass alone cannot produce anything that
     thin, so a ribbon must come from the top-view clip;
  4. where those ribbons sit: whether each touches a TALLER neighbour, and how deep
     the clipped band is (area lost to the clip / outline shared with taller crowns);
  5. tiny detached parts (< `--tiny` m²) of MultiPolygons.

    /home/hugobaudchon/anaconda3/envs/canopyrs/bin/python scripts/analyse_gt_edges.py \
        --gt-dir data/lidar_masks --out-dir output/gt_edges \
        --ortho ../../rgb_ortho/20250318_asnortheast_50mm_p1_rgb.cog.tif
"""
import argparse
import glob
import json
import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union


def opening(g, r):
    return g.buffer(-r, join_style=1).buffer(r, join_style=1)


def analyse(path, r, min_ribbon, tiny, tol):
    gt = gpd.read_file(path, layer="crowns")
    gt = gt[gt.geometry.notna() & ~gt.geometry.is_empty].reset_index(drop=True)
    geoms = gt.geometry.values
    zmax = gt.z_max.to_numpy()
    ids = gt.instance_id.to_numpy()
    clipped = (gt.raw_area_m2 - gt.crown_area_m2).to_numpy() > 1e-3
    # resolve_top_view priority: larger z_max first, ties by instance_id
    rank = {k: n for n, k in enumerate(sorted(range(len(gt)), key=lambda k: (-zmax[k], ids[k])))}

    # 1 + 2: pairwise contact
    pairs = []
    for i, j in zip(*gt.sindex.query(gt.geometry.buffer(tol), predicate="intersects")):
        if i >= j:
            continue
        inter = geoms[i].intersection(geoms[j]).area
        shared = geoms[i].boundary.intersection(geoms[j].boundary.buffer(tol)).length
        taller, shorter = (i, j) if rank[i] < rank[j] else (j, i)
        pairs.append(dict(i=i, j=j, taller=taller, shorter=shorter, inter_area=inter, shared_len=shared))
    pairs = pd.DataFrame(pairs, columns=["i", "j", "taller", "shorter", "inter_area", "shared_len"])
    contact = pairs[pairs.shared_len > 0.05]

    # 3 + 4 + 5: per crown
    rows, ribbons = [], []
    for k, g in enumerate(geoms):
        taller = [geoms[m] for m in range(len(geoms)) if rank[m] < rank[k]]
        parts_rib = []
        thin = g.difference(opening(g, r))
        if not thin.is_empty:
            for p in getattr(thin, "geoms", [thin]):
                # Below min_ribbon the "ribbons" are mm-scale opening artefacts present on
                # every polygon, clipped or not (buffer arc approximation), not features.
                if p.geom_type == "Polygon" and p.area >= min_ribbon:
                    parts_rib.append(p)
                    ribbons.append(dict(instance_id=ids[k], clipped=bool(clipped[k]), area=p.area,
                                        mean_width=2 * p.area / p.length,
                                        touches_taller=bool(taller) and min(p.distance(t) for t in taller) < 0.01,
                                        geometry=p))
        parts = list(getattr(g, "geoms", [g]))
        shared = contact.loc[(contact.i == k) | (contact.j == k), "shared_len"].sum()
        shared_with_taller = contact.loc[contact.shorter == k, "shared_len"].sum()
        lost = float(gt.raw_area_m2[k] - g.area)
        rows.append(dict(
            instance_id=ids[k], z_max=zmax[k], area=g.area, raw_area=gt.raw_area_m2[k],
            clipped=bool(clipped[k]), clip_lost_m2=lost, clip_lost_frac=lost / gt.raw_area_m2[k],
            perimeter=g.length, shared_len=shared, shared_frac=shared / g.length,
            shared_with_taller_len=shared_with_taller,
            band_depth_m=lost / shared_with_taller if shared_with_taller > 0.5 else np.nan,
            ribbon_area=sum(p.area for p in parts_rib), n_ribbons=len(parts_rib),
            n_parts=len(parts), n_tiny_parts=sum(p.area < tiny for p in parts)))
    per = pd.DataFrame(rows)
    cols = ["instance_id", "clipped", "area", "mean_width", "touches_taller", "geometry"]
    rib = gpd.GeoDataFrame(ribbons if ribbons else None, columns=None if ribbons else cols,
                           geometry="geometry", crs=gt.crs)
    return gt, per, pairs, contact, rib


def summarise(plot, gt, per, pairs, contact, rib):
    c, u = per[per.clipped], per[~per.clipped]
    return dict(
        plot=plot, n_crowns=len(per),
        sum_area=float(per.area.sum()), union_area=float(gt.geometry.union_all().area),
        n_pairs_overlapping_area=int((pairs.inter_area > 1e-6).sum()),
        max_pair_overlap_m2=float(pairs.inter_area.max()) if len(pairs) else 0.0,
        n_pairs_sharing_edge=int(len(contact)),
        n_crowns_sharing_edge=int((per.shared_len > 0).sum()),
        shared_boundary_m=float(contact.shared_len.sum()),
        shared_frac_of_perimeter_median=float(per.loc[per.shared_len > 0, "shared_frac"].median())
        if (per.shared_len > 0).any() else 0.0,
        n_clipped=int(len(c)), clip_lost_m2=float(c.clip_lost_m2.sum()),
        clip_lost_frac_median=float(c.clip_lost_frac.median()) if len(c) else 0.0,
        band_depth_median_m=float(per.band_depth_m.median()),
        n_ribbons_clipped=int(c.n_ribbons.sum()), n_ribbons_unclipped=int(u.n_ribbons.sum()),
        ribbon_m2_clipped=float(c.ribbon_area.sum()), ribbon_m2_unclipped=float(u.ribbon_area.sum()),
        ribbons_touching_taller_frac=float(rib.touches_taller.mean()) if len(rib) else 0.0,
        ribbon_width_median_m=float(rib.mean_width.median()) if len(rib) else 0.0,
        n_tiny_parts_clipped=int(c.n_tiny_parts.sum()), n_tiny_parts_unclipped=int(u.n_tiny_parts.sum()),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", default="data/lidar_masks")
    ap.add_argument("--out-dir", default="output/gt_edges")
    ap.add_argument("--ortho", default=None)
    ap.add_argument("--r", type=float, default=0.2, help="opening radius: ribbons are narrower than 2r")
    ap.add_argument("--min-ribbon", type=float, default=0.01, help="ignore thin parts below this area, m²")
    ap.add_argument("--tiny", type=float, default=0.25, help="tiny part threshold, m²")
    ap.add_argument("--tol", type=float, default=1e-3, help="edge-coincidence tolerance, m")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    summaries, all_rib, all_per, data = [], [], [], {}
    for f in sorted(glob.glob(os.path.join(args.gt_dir, "*_gt_crowns_gt.gpkg"))):
        plot = re.search(r"plot_(\d+)", f).group(1)
        gt, per, pairs, contact, rib = analyse(f, args.r, args.min_ribbon, args.tiny, args.tol)
        s = summarise(plot, gt, per, pairs, contact, rib)
        summaries.append(s)
        per.insert(0, "plot", plot)
        all_per.append(per)
        rib.insert(0, "plot", plot)
        all_rib.append(rib)
        data[plot] = (gt, per, rib)
        print(json.dumps(s))

    summ = pd.DataFrame(summaries)
    summ.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    pd.concat(all_per).to_csv(os.path.join(args.out_dir, "per_crown.csv"), index=False)
    rib = gpd.GeoDataFrame(pd.concat(all_rib, ignore_index=True), geometry="geometry", crs=all_rib[0].crs)
    rib.to_file(os.path.join(args.out_dir, "ribbons.gpkg"), layer="ribbons")

    # pooled numbers
    per = pd.concat(all_per)
    pooled = dict(
        crowns=int(len(per)), clipped=int(per.clipped.sum()),
        clip_lost_m2=float(per.clip_lost_m2.sum()), gt_area_m2=float(per.area.sum()),
        band_depth_median_m=float(per.band_depth_m.median()),
        band_depth_iqr_m=[float(per.band_depth_m.quantile(.25)), float(per.band_depth_m.quantile(.75))],
        crowns_sharing_edge=int((per.shared_len > 0).sum()),
        ribbons_clipped=int(per.loc[per.clipped, "n_ribbons"].sum()),
        ribbons_unclipped=int(per.loc[~per.clipped, "n_ribbons"].sum()),
        ribbon_m2_clipped=float(per.loc[per.clipped, "ribbon_area"].sum()),
        ribbon_m2_unclipped=float(per.loc[~per.clipped, "ribbon_area"].sum()),
        ribbon_share_of_gt_area=float(per.ribbon_area.sum() / per.area.sum()),
        ribbon_width_median_m=float(rib.mean_width.median()),
        ribbons_touching_taller_frac=float(rib.touches_taller.mean()),
    )
    print("[pooled]", json.dumps(pooled))
    with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
        json.dump(dict(per_plot=summaries, pooled=pooled, params=vars(args)), fh, indent=2)

    # ------------------------------------------------------------------ figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw(ax, gt, per, rib, bbox=None, lw=0.8):
        if args.ortho:
            import rasterio
            from rasterio.windows import from_bounds
            x0, y0, x1, y1 = bbox if bbox else gt.total_bounds
            with rasterio.open(args.ortho) as src:
                win = from_bounds(x0, y0, x1, y1, transform=src.transform)
                sc = max(1, int(max(win.width, win.height) // 1400))
                img = src.read([1, 2, 3], window=win, boundless=True,
                               out_shape=(3, max(1, int(win.height // sc)), max(1, int(win.width // sc))))
            ax.imshow(np.moveaxis(img, 0, -1), extent=[x0, x1, y0, y1], alpha=0.6)
        cl = dict(zip(per.instance_id, per.clipped))
        for _, row in gt.iterrows():
            c = "tab:orange" if cl[row.instance_id] else "tab:blue"
            for p in getattr(row.geometry, "geoms", [row.geometry]):
                xs, ys = p.exterior.xy
                ax.fill(xs, ys, color=c, alpha=0.25, lw=0)
                ax.plot(xs, ys, color=c, lw=lw)
        for p in rib.geometry:
            xs, ys = p.exterior.xy
            ax.fill(xs, ys, color="red", alpha=0.9, lw=0)
        if bbox:
            ax.set_xlim(bbox[0], bbox[2])
            ax.set_ylim(bbox[1], bbox[3])
        ax.set_aspect("equal")

    plots = list(data)
    fig, axs = plt.subplots(1, len(plots), figsize=(6 * len(plots), 6.5))
    for ax, p in zip(np.atleast_1d(axs), plots):
        gt, per_p, rib_p = data[p]
        draw(ax, gt, per_p, rib_p)
        s = summ.set_index("plot").loc[p]
        ax.set_title(f"plot {p}: {int(s.n_clipped)}/{int(s.n_crowns)} crowns clipped\n"
                     f"ribbons {s.ribbon_m2_clipped:.1f} m² (clipped) vs {s.ribbon_m2_unclipped:.2f} m² (unclipped)")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Blue = never clipped (on top), orange = clipped by a taller neighbour, "
                 f"red = parts narrower than {2 * args.r:.1f} m", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "01_overview.png"), dpi=100)
    plt.close(fig)

    # zooms on the largest ribbons
    top = rib.sort_values("area", ascending=False).drop_duplicates("instance_id").head(6)
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    for ax, (_, t) in zip(axs.ravel(), top.iterrows()):
        gt, per_p, rib_p = data[t["plot"]]
        cx, cy = t.geometry.centroid.x, t.geometry.centroid.y
        h = 3.0
        draw(ax, gt, per_p, rib_p, bbox=(cx - h, cy - h, cx + h, cy + h), lw=1.5)
        ax.set_title(f"plot {t['plot']}, crown {t.instance_id}: ribbon {t.area:.2f} m², "
                     f"mean width {t.mean_width * 100:.0f} cm")
    fig.suptitle("6 m zooms on the largest ribbons (red). Orange crown = the clipped, shorter tree; "
                 "blue = the taller neighbour that clipped it", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "02_zooms.png"), dpi=100)
    plt.close(fig)

    # ribbon width distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 2 * args.r, 21)
    ax.hist(rib.loc[rib.clipped, "mean_width"] * 100, bins=bins * 100, alpha=0.7, label="in clipped crowns")
    ax.hist(rib.loc[~rib.clipped, "mean_width"] * 100, bins=bins * 100, alpha=0.7, label="in unclipped crowns")
    ax.axvline(0.15 * 100, color="k", ls="--", lw=1)
    ax.text(15.5, ax.get_ylim()[1] * 0.9, "1 raster cell (15 cm)", fontsize=9)
    ax.set_xlabel("ribbon mean width (cm)")
    ax.set_ylabel("ribbon parts")
    ax.legend()
    ax.set_title("Width of thin parts, all plots")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "03_ribbon_widths.png"), dpi=100)
    plt.close(fig)
    print(f"[done] {args.out_dir}")


if __name__ == "__main__":
    main()
