#!/usr/bin/env python
"""One-to-one matching of RGB (CanopyRS) crowns to LiDAR GT crowns by Jaccard index.

    J(g, p) = |g ∩ p| / |g ∪ p|        (= IoU, on exact vector geometry, m²)

Steps
  1. Preprocess exactly as eval_crown_overlap.py (plan.md §5): drop NULL / < 1 m² GT
     crowns, AOI = convex hull of GT, keep predictions with ≥ 40 % of their area in
     the AOI and clip them to it.
  2. Jaccard for every intersecting (GT, prediction) pair (sparse; others are 0).
  3. Optimal one-to-one assignment (Hungarian) maximising the summed Jaccard.
  4. An assigned pair counts as a match only if J ≥ tau (default 0.5).

Because GT crowns are mutually disjoint and so are predictions, any pair with
J > 0.5 is the *only* candidate for both its members, so for tau ≥ 0.5 the matching
is unique and the Hungarian step cannot change it; it only matters below 0.5.

GT is incomplete (plan.md §2): unmatched predictions are reported, never scored as
false positives.

    /home/hugobaudchon/anaconda3/envs/canopyrs/bin/python scripts/match_crowns_jaccard.py \
        --gt data/lidar_masks/asnortheast_plot_203_gt_crowns_gt.gpkg \
        --pred data/rgb_masks/20250318_asnortheast_50mm_p1_rgb_gr0p07_infer.gpkg \
        --ortho ../../rgb_ortho/20250318_asnortheast_50mm_p1_rgb.cog.tif \
        --out-dir output/jaccard/plot_203
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_crown_overlap import (  # noqa: E402
    AOI_MIN_OVERLAP, EDGE_BUFFER_M, MIN_AREA_M2, PrepStats, build_aoi, flag_edge_gt,
    load_gt, load_preds, overlap_pairs, restrict_preds,
)

THRESHOLDS = (0.25, 0.5, 0.75, 0.9)


def match(gt, preds, tau):
    pairs = overlap_pairs(gt, preds)
    J = np.zeros((len(gt), len(preds)))
    if len(pairs):
        J[pairs.i.to_numpy(), pairs.j.to_numpy()] = pairs.iou.to_numpy()
    rows, cols = linear_sum_assignment(-J) if J.size else ([], [])
    assigned = {int(r): int(c) for r, c in zip(rows, cols) if J[r, c] > 0}
    return pairs, J, assigned


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ortho", type=Path, default=None, help="optional, for the map background")
    ap.add_argument("--tau", type=float, default=0.5, help="Jaccard needed to call a pair a match")
    ap.add_argument("--min-area", type=float, default=MIN_AREA_M2)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stats = PrepStats()
    gt = load_gt(args.gt, args.min_area, stats)
    aoi = build_aoi(gt)
    preds = restrict_preds(load_preds(args.pred), aoi, AOI_MIN_OVERLAP, args.min_area, stats)
    edge = flag_edge_gt(gt, aoi, EDGE_BUFFER_M)
    pairs, J, assigned = match(gt, preds, args.tau)

    # ---- per-GT table
    gt_area = gt.geometry.area.to_numpy()
    pr_area = preds.geometry.area.to_numpy()
    rows = []
    for i in range(len(gt)):
        j = assigned.get(i)
        jac = J[i, j] if j is not None else 0.0
        n_touch = int((J[i] > 0).sum())
        rows.append(dict(
            instance_id=gt.instance_id.iloc[i], gt_area_m2=gt_area[i],
            height_m=gt.height_m.iloc[i] if "height_m" in gt else np.nan, edge_gt=bool(edge[i]),
            n_preds_touching=n_touch,
            pred_id=preds.canopyrs_object_id.iloc[j] if j is not None else None,
            pred_area_m2=pr_area[j] if j is not None else np.nan,
            jaccard=jac, best_jaccard=float(J[i].max()) if J.shape[1] else 0.0,
            matched=jac >= args.tau,
        ))
    per_gt = pd.DataFrame(rows)
    per_gt.to_csv(args.out_dir / "per_gt.csv", index=False)

    used = {j for i, j in assigned.items() if J[i, j] >= args.tau}
    unmatched = pd.DataFrame(dict(
        pred_id=preds.canopyrs_object_id, area_m2=pr_area,
        best_jaccard=J.max(axis=0) if J.shape[0] else 0.0,
        aggregator_score=preds.get("aggregator_score", pd.Series(np.nan, index=preds.index)),
    ))
    unmatched = unmatched[~unmatched.index.isin(used)]
    unmatched.to_csv(args.out_dir / "unmatched_preds.csv", index=False)

    # ---- summary
    n = len(gt)
    js = per_gt.jaccard.to_numpy()
    summary = dict(
        plot=args.gt.stem, tau=args.tau, n_gt=n, n_pred=len(preds),
        prep=stats.__dict__, n_candidate_pairs=int(len(pairs)),
        n_matched=int((js >= args.tau).sum()),
        recall=float((js >= args.tau).mean()),
        mean_jaccard_all_gt=float(js.mean()),
        mean_jaccard_matched=float(js[js >= args.tau].mean()) if (js >= args.tau).any() else 0.0,
        median_jaccard_all_gt=float(np.median(js)),
        area_weighted_jaccard=float((js * gt_area).sum() / gt_area.sum()),
        recall_at={str(t): float((js >= t).mean()) for t in THRESHOLDS},
        n_at={str(t): int((js >= t).sum()) for t in THRESHOLDS},
        n_gt_no_overlap=int((per_gt.n_preds_touching == 0).sum()),
        n_unmatched_preds=int(len(unmatched)),
        hungarian_changed_vs_best=int((per_gt.jaccard < per_gt.best_jaccard - 1e-12).sum()),
        no_edge=dict(n=int((~per_gt.edge_gt).sum()),
                     mean_jaccard=float(per_gt.jaccard[~per_gt.edge_gt].mean()),
                     recall=float(per_gt.matched[~per_gt.edge_gt].mean())),
    )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "prep"}, indent=2, default=float))

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20, 9.5), gridspec_kw=dict(width_ratios=[1.6, 1]))
    x0, y0, x1, y1 = aoi.buffer(2).bounds
    if args.ortho:
        import rasterio
        from rasterio.windows import from_bounds
        with rasterio.open(args.ortho) as src:
            win = from_bounds(x0, y0, x1, y1, transform=src.transform)
            scale = max(1, int(win.width // 1500))
            img = src.read([1, 2, 3], window=win,
                           out_shape=(3, int(win.height // scale), int(win.width // scale)))
        ax.imshow(np.moveaxis(img, 0, -1), extent=[x0, x1, y0, y1], alpha=0.85)
    norm = colors.Normalize(0, 1)
    cmap = cm.get_cmap("RdYlGn")
    for i, g in enumerate(gt.geometry):
        r = per_gt.iloc[i]
        gpd_col = cmap(norm(r.jaccard))
        for poly in getattr(g, "geoms", [g]):
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, color=gpd_col, alpha=0.45, lw=0)
            ax.plot(xs, ys, color="black", lw=1.0)
        c = g.representative_point()
        ax.annotate(f"{r.jaccard:.2f}", (c.x, c.y), ha="center", va="center", fontsize=7,
                    color="black", weight="bold")
    for j, p in enumerate(preds.geometry):
        for poly in getattr(p, "geoms", [p]):
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, color="magenta" if j in used else "white", lw=1.0,
                    ls="-" if j in used else "--")
    xs, ys = aoi.exterior.xy
    ax.plot(xs, ys, color="yellow", lw=0.8, ls=":")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_title("LiDAR GT crowns filled by assigned Jaccard (black outline, number)\n"
                 f"RGB masks: magenta = matched (J ≥ {args.tau}), white dashed = unmatched; yellow = AOI")
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.035, label="Jaccard")

    bins = np.linspace(0, 1, 21)
    ax2.hist(js, bins=bins, color="grey", edgecolor="black")
    for t in THRESHOLDS:
        ax2.axvline(t, color="red" if t == args.tau else "black", ls="--", lw=1)
        ax2.text(t, ax2.get_ylim()[1] * 0.95, f" ≥{t}: {int((js >= t).sum())}", fontsize=9)
    ax2.set_xlabel("Jaccard of assigned RGB mask (0 = no partner)")
    ax2.set_ylabel("LiDAR GT crowns")
    ax2.set_title(f"{n} GT crowns, {len(preds)} RGB masks in AOI\n"
                  f"recall@{args.tau} = {summary['recall']:.2f}, mean J (all GT) = {js.mean():.3f}")
    fig.tight_layout()
    fig.savefig(args.out_dir / "jaccard_matches.png", dpi=110)
    print(f"[done] {args.out_dir}")


if __name__ == "__main__":
    main()
