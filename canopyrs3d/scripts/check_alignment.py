#!/usr/bin/env python
"""Measure RGB-orthomosaic <-> LiDAR co-registration for one plot (plan.md §15.5).

Rasterises the plot's point cloud onto a metric grid (top-of-canopy height and the
colour of the top point), resamples the orthomosaic onto the same grid, and
estimates the displacement between the two with masked normalised
cross-correlation (Padfield 2012):

  A  RGB ortho  <-> LiDAR-colour raster   (near same modality)
  B  RGB ortho  <-> CHM                   (independent of the LiDAR colourisation)
  C  per GT crown, A restricted to that crown, regressed on crown height
     (relief displacement: offset proportional to height)

Sign convention for every reported shift: (dx, dy) in metres such that a feature
at LiDAR position (x, y) appears in the ortho at (x + dx, y + dy).

Run with the `geodataset` conda env (it has laspy + lazrs):

    /home/hugobaudchon/anaconda3/envs/geodataset/bin/python scripts/check_alignment.py \
        --plot 203 \
        --las-dir "../../../annotated_plots_AS/203" \
        --ortho "../../rgb_ortho/20250318_asnortheast_50mm_p1_rgb.cog.tif" \
        --gt data/lidar_masks/asnortheast_plot_203_gt_crowns_gt.gpkg \
        --out-dir output/alignment/plot_203
"""
import argparse
import glob
import json
import os

import geopandas as gpd
import laspy
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds
from scipy import ndimage as ndi
from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------- rasters

def make_grid(las_files, res, pad):
    mins = np.min([laspy.open(f).header.mins for f in las_files], axis=0)
    maxs = np.max([laspy.open(f).header.maxs for f in las_files], axis=0)
    x0 = np.floor((mins[0] - pad) / res) * res
    y1 = np.ceil((maxs[1] + pad) / res) * res
    w = int(np.ceil((maxs[0] + pad - x0) / res))
    h = int(np.ceil((y1 - (mins[1] - pad)) / res))
    return from_origin(x0, y1, res, res), (h, w)


def rasterise_lidar(las_files, transform, shape, zclip=(-3.0, 60.0)):
    """Top-of-canopy height, colour of the top point, and point count per cell."""
    h, w = shape
    res = transform.a
    x0, y1 = transform.c, transform.f
    zbuf = np.full(h * w, -np.inf, dtype=np.float32)
    rgb = np.zeros((h * w, 3), dtype=np.float32)
    count = np.zeros(h * w, dtype=np.int64)
    n_total = n_clipped = 0
    for f in las_files:
        las = laspy.read(f)
        x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z, dtype=np.float32)
        keep = (z >= zclip[0]) & (z <= zclip[1])
        n_total += len(z)
        n_clipped += int((~keep).sum())
        x, y, z = x[keep], y[keep], z[keep]
        col = np.column_stack([las.red[keep], las.green[keep], las.blue[keep]]).astype(np.float32) / 257.0
        c = np.floor((x - x0) / res).astype(np.int64)
        r = np.floor((y1 - y) / res).astype(np.int64)
        cell = r * w + c
        count += np.bincount(cell, minlength=h * w)
        order = np.lexsort((z, cell))
        cs = cell[order]
        last = np.r_[cs[1:] != cs[:-1], True]
        sel = order[last]
        cs, zs = cell[sel], z[sel]
        upd = zs > zbuf[cs]
        zbuf[cs[upd]] = zs[upd]
        rgb[cs[upd]] = col[sel[upd]]
    occ = (count > 0).reshape(h, w)
    chm = zbuf.reshape(h, w)
    rgb = rgb.reshape(h, w, 3)

    # Footprint = occupied cells, closed and hole-filled, slightly eroded (edge cells are partial).
    foot = ndi.binary_fill_holes(ndi.binary_closing(occ, iterations=4))
    foot = ndi.binary_erosion(foot, iterations=3)

    # Fill empty cells inside the footprint from the nearest occupied cell.
    _, (ri, ci) = ndi.distance_transform_edt(~occ, return_indices=True)
    chm = np.where(occ, chm, chm[ri, ci])
    rgb = np.where(occ[..., None], rgb, rgb[ri, ci])

    # Despike: isolated returns far above their neighbourhood are noise, not canopy.
    med = ndi.median_filter(chm, size=5)
    spikes = chm > med + 3.0
    chm = np.where(spikes, med, chm)

    chm[~foot] = 0
    rgb[~foot] = 0
    stats = dict(n_points=n_total, n_z_clipped=n_clipped, n_spike_cells=int(spikes.sum()),
                 occupied_fraction_of_footprint=float(occ[foot].mean()))
    return chm, rgb, foot, count.reshape(h, w), stats


def read_ortho(path, transform, shape):
    h, w = shape
    res = transform.a
    left, top = transform.c, transform.f
    bounds = (left - 1, top - h * res - 1, left + w * res + 1, top + 1)
    with rasterio.open(path) as src:
        win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        arr = src.read([1, 2, 3, 4], window=win, boundless=True, fill_value=0).astype(np.float32)
        src_tf = src.window_transform(win)
        crs = src.crs
        native_res = src.res[0]
    out = np.zeros((4, h, w), dtype=np.float32)
    for b in range(4):
        reproject(arr[b], out[b], src_transform=src_tf, src_crs=crs,
                  dst_transform=transform, dst_crs=crs, resampling=Resampling.average)
    valid = out[3] > 250
    return np.moveaxis(out[:3], 0, -1), valid, native_res


# --------------------------------------------------------------------------- matching

def lum(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def highpass(img, mask, sigma_hp_px, sigma_lp_px=1.0):
    """Band-pass: remove illumination / radiometric trends, keep crown-scale texture."""
    img = img.astype(np.float64)
    fill = img[mask].mean() if mask.any() else 0.0
    img = np.where(mask, img, fill)
    out = ndi.gaussian_filter(img, sigma_lp_px) - ndi.gaussian_filter(img, sigma_hp_px)
    out[~mask] = 0
    return out


def _parabolic(fm, f0, fp):
    den = fm - 2 * f0 + fp
    return 0.0 if den >= 0 else 0.5 * (fm - fp) / den


def ncc_shift(ref, mov, mref, mmov, max_px, overlap_ratio=0.3, excl_px=6):
    """Displacement (drow, dcol) of `mov` content relative to `ref`, sub-pixel.

    Returns (drow, dcol, peak_ncc, distinctness), distinctness = peak minus the best
    value farther than `excl_px` from the peak (0 means an ambiguous match).
    """
    xc = cross_correlate_masked(mov, ref, mmov, mref, axes=(0, 1), mode="full",
                                overlap_ratio=overlap_ratio)
    xc = np.real(xc)
    c0 = np.array(ref.shape) - 1  # index of zero displacement in the 'full' output
    sub = xc[c0[0] - max_px:c0[0] + max_px + 1, c0[1] - max_px:c0[1] + max_px + 1].copy()
    sub[~np.isfinite(sub)] = -1
    pr, pc = np.unravel_index(np.argmax(sub), sub.shape)
    peak = float(sub[pr, pc])
    on_edge = pr in (0, sub.shape[0] - 1) or pc in (0, sub.shape[1] - 1)
    fr = _parabolic(sub[pr - 1, pc], peak, sub[pr + 1, pc]) if 0 < pr < sub.shape[0] - 1 else 0.0
    fc = _parabolic(sub[pr, pc - 1], peak, sub[pr, pc + 1]) if 0 < pc < sub.shape[1] - 1 else 0.0
    rr, cc = np.ogrid[:sub.shape[0], :sub.shape[1]]
    far = (rr - pr) ** 2 + (cc - pc) ** 2 > excl_px ** 2
    second = float(sub[far].max()) if far.any() else -1.0
    return dict(drow=pr - max_px + fr, dcol=pc - max_px + fc, ncc=peak,
                distinct=peak - second, on_edge=bool(on_edge), surface=sub)


def to_metres(d, res):
    return d["dcol"] * res, -d["drow"] * res


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", required=True)
    ap.add_argument("--las-dir", required=True)
    ap.add_argument("--ortho", required=True)
    ap.add_argument("--gt", required=True, help="GT crown polygons (gpkg, layer 'crowns')")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--res", type=float, default=0.05, help="analysis grid, m/px")
    ap.add_argument("--max-shift", type=float, default=3.0, help="global search radius, m")
    ap.add_argument("--win", type=float, default=10.0, help="window size, m")
    ap.add_argument("--step", type=float, default=5.0, help="window step, m")
    ap.add_argument("--win-max-shift", type=float, default=2.0, help="window/crown search radius, m")
    ap.add_argument("--min-distinct", type=float, default=0.05,
                    help="windows/crowns with a less distinct NCC peak are reported but not trusted")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    res = args.res
    px = lambda m: int(round(m / res))  # noqa: E731

    las_files = sorted(glob.glob(os.path.join(args.las_dir, "*.la[sz]")))
    assert las_files, f"no LAS/LAZ in {args.las_dir}"
    transform, shape = make_grid(las_files, res, pad=args.max_shift + 1.5)
    print(f"[grid] {shape[1]}x{shape[0]} px at {res} m, {len(las_files)} LAS files")

    chm, lrgb, foot, count, lstats = rasterise_lidar(las_files, transform, shape)
    print(f"[lidar] {lstats}")
    orgb, ovalid, native_res = read_ortho(args.ortho, transform, shape)
    print(f"[ortho] native {native_res:.4f} m/px, valid {ovalid.mean():.3f} of grid")

    gt = gpd.read_file(args.gt, layer="crowns")
    gt = gt[gt.geometry.notna() & ~gt.geometry.is_empty].copy()

    # Band-passed inputs. Texture scale ~ crown parts; CHM uses a wider high-pass.
    m_l = foot
    m_o = ovalid
    L_col = highpass(lum(lrgb), m_l, sigma_hp_px=px(1.0))
    O_col = highpass(lum(orgb), m_o, sigma_hp_px=px(1.0))
    L_chm = highpass(chm, m_l, sigma_hp_px=px(2.0))
    O_chm = highpass(lum(orgb), m_o, sigma_hp_px=px(2.0))

    summary = dict(plot=args.plot, res_m=res, ortho=os.path.abspath(args.ortho),
                   ortho_native_res_m=native_res, lidar=lstats,
                   convention="(dx,dy): feature at LiDAR (x,y) appears in ortho at (x+dx, y+dy)")

    # ---- A / B global
    glob_res = {}
    for name, (r, m, mr, mm) in {"A_colour": (L_col, O_col, m_l, m_o),
                                 "B_chm": (L_chm, O_chm, m_l, m_o)}.items():
        d = ncc_shift(r, m, mr, mm, px(args.max_shift), excl_px=px(0.5))
        dx, dy = to_metres(d, res)
        glob_res[name] = d
        summary[f"global_{name}"] = dict(dx_m=dx, dy_m=dy, norm_m=float(np.hypot(dx, dy)),
                                         ncc=d["ncc"], distinct=d["distinct"], at_search_edge=d["on_edge"])
        print(f"[global {name}] dx={dx:+.3f} dy={dy:+.3f} m  |d|={np.hypot(dx, dy):.3f}  "
              f"ncc={d['ncc']:.3f} distinct={d['distinct']:.3f}")

    # ---- self-test: shift the ortho by a known amount, the estimate must move by exactly that
    known = np.array([0.37, -0.52])  # (dx, dy) m
    sh = (-known[1] / res, known[0] / res)  # (rows, cols)
    O_sh = ndi.shift(O_col, sh, order=1)
    m_sh = ndi.shift(m_o.astype(float), sh, order=0) > 0.5
    d = ncc_shift(L_col, O_sh, m_l, m_sh, px(args.max_shift), excl_px=px(0.5))
    got = np.array(to_metres(d, res)) - np.array(to_metres(glob_res["A_colour"], res))
    err = float(np.hypot(*(got - known)))
    summary["self_test"] = dict(injected_m=known.tolist(), recovered_m=got.tolist(), error_m=err)
    print(f"[self-test] injected {known} recovered {got.round(3)} error {err:.3f} m")

    # ---- A / B windows
    rows = []
    W, S, M = px(args.win), px(args.step), px(args.win_max_shift)
    for r0 in range(0, shape[0] - W + 1, S):
        for c0 in range(0, shape[1] - W + 1, S):
            sl = (slice(r0, r0 + W), slice(c0, c0 + W))
            frac = foot[sl].mean()
            if frac < 0.7 or ovalid[sl].mean() < 0.99:
                continue
            xc_, yc_ = transform * (c0 + W / 2, r0 + W / 2)
            row = dict(x=xc_, y=yc_, lidar_frac=frac, mean_chm=float(chm[sl][foot[sl]].mean()),
                       p95_chm=float(np.percentile(chm[sl][foot[sl]], 95)))
            for name, (r, m) in {"A": (L_col, O_col), "B": (L_chm, O_chm)}.items():
                d = ncc_shift(r[sl], m[sl], foot[sl], ovalid[sl], M, excl_px=px(0.5))
                dx, dy = to_metres(d, res)
                row.update({f"{name}_dx": dx, f"{name}_dy": dy, f"{name}_ncc": d["ncc"],
                            f"{name}_distinct": d["distinct"], f"{name}_edge": d["on_edge"]})
            rows.append(row)
    win = pd.DataFrame(rows)
    win.to_csv(os.path.join(args.out_dir, "windows.csv"), index=False)

    def robust(df, p):
        ok = df[(df[f"{p}_distinct"] >= args.min_distinct) & ~df[f"{p}_edge"]]
        if ok.empty:
            return dict(n=0, n_total=len(df))
        n = np.hypot(ok[f"{p}_dx"], ok[f"{p}_dy"])
        return dict(n=int(len(ok)), n_total=int(len(df)),
                    median_dx_m=float(ok[f"{p}_dx"].median()), median_dy_m=float(ok[f"{p}_dy"].median()),
                    std_dx_m=float(ok[f"{p}_dx"].std()), std_dy_m=float(ok[f"{p}_dy"].std()),
                    p50_norm_m=float(np.percentile(n, 50)), p95_norm_m=float(np.percentile(n, 95)),
                    max_norm_m=float(n.max()))
    summary["windows_A"] = robust(win, "A")
    summary["windows_B"] = robust(win, "B")
    # A-vs-B agreement per window (both trusted)
    both = win[(win.A_distinct >= args.min_distinct) & (win.B_distinct >= args.min_distinct)
               & ~win.A_edge & ~win.B_edge]
    if len(both):
        summary["windows_A_vs_B_disagreement_m"] = dict(
            n=int(len(both)),
            median=float(np.median(np.hypot(both.A_dx - both.B_dx, both.A_dy - both.B_dy))))
    print(f"[windows A] {summary['windows_A']}")
    print(f"[windows B] {summary['windows_B']}")

    # ---- C per crown
    crow = []
    for _, g in gt.iterrows():
        x0, y0, x1, y1 = g.geometry.bounds
        pad = args.win_max_shift
        c0, r0 = ~transform * (x0 - pad, y1 + pad)
        c1, r1 = ~transform * (x1 + pad, y0 - pad)
        r0, c0 = max(int(r0), 0), max(int(c0), 0)
        r1, c1 = min(int(np.ceil(r1)), shape[0]), min(int(np.ceil(c1)), shape[1])
        sl = (slice(r0, r1), slice(c0, c1))
        sub_tf = transform * rasterio.Affine.translation(c0, r0)
        cm = rasterize([(g.geometry.buffer(0.5), 1)], out_shape=(r1 - r0, c1 - c0),
                       transform=sub_tf, fill=0, dtype="uint8").astype(bool) & foot[sl]
        if cm.sum() < px(1.0) ** 2:
            continue
        d = ncc_shift(L_col[sl], O_col[sl], cm, ovalid[sl], M, overlap_ratio=0.5, excl_px=px(0.5))
        dx, dy = to_metres(d, res)
        crow.append(dict(instance_id=g.instance_id, height_m=g.height_m, area_m2=g.geometry.area,
                         cx=g.geometry.centroid.x, cy=g.geometry.centroid.y,
                         dx=dx, dy=dy, norm=float(np.hypot(dx, dy)), ncc=d["ncc"],
                         distinct=d["distinct"], edge=d["on_edge"]))
    cr = pd.DataFrame(crow)
    cr.to_csv(os.path.join(args.out_dir, "crowns.csv"), index=False)
    ok = cr[(cr.distinct >= args.min_distinct) & ~cr.edge]
    cs = dict(n=int(len(ok)), n_total=int(len(cr)))
    if len(ok) >= 5:
        for k in ("dx", "dy", "norm"):
            slope, icpt = np.polyfit(ok.height_m, ok[k], 1)
            cs[f"{k}_vs_height"] = dict(slope_m_per_m=float(slope), intercept_m=float(icpt),
                                       pearson_r=float(np.corrcoef(ok.height_m, ok[k])[0, 1]))
        cs["median_dx_m"], cs["median_dy_m"] = float(ok.dx.median()), float(ok.dy.median())
        cs["p95_norm_m"] = float(np.percentile(ok.norm, 95))
        # residual after removing the plot-wide median: what a rigid shift cannot fix
        rn = np.hypot(ok.dx - ok.dx.median(), ok.dy - ok.dy.median())
        cs["residual_after_rigid_p50_m"] = float(np.percentile(rn, 50))
        cs["residual_after_rigid_p95_m"] = float(np.percentile(rn, 95))
    summary["crowns_C"] = cs
    print(f"[crowns C] {cs}")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # ------------------------------------------------------------------- figures
    left, top = transform.c, transform.f
    ext = [0, shape[1] * res, 0, shape[0] * res]
    to_local = lambda x, y: (x - left, y - (top - shape[0] * res))  # noqa: E731

    def outlines(ax, color="cyan", lw=0.6, dx=0.0, dy=0.0):
        for geom in gt.geometry:
            for poly in getattr(geom, "geoms", [geom]):
                xx, yy = poly.exterior.xy
                lx, ly = to_local(np.asarray(xx) + dx, np.asarray(yy) + dy)
                ax.plot(lx, ly, color=color, lw=lw)

    def u8(img):
        return np.clip(img / 255.0, 0, 1)

    # crop figures to the LiDAR footprint
    rr, cc = np.nonzero(foot)
    cy0, cy1 = (shape[0] - rr.max()) * res, (shape[0] - rr.min()) * res
    cx0, cx1 = cc.min() * res, cc.max() * res

    def crop(ax):
        ax.set_xlim(cx0 - 1, cx1 + 1)
        ax.set_ylim(cy0 - 1, cy1 + 1)
        ax.set_aspect("equal")

    # 1. side by side
    fig, axs = plt.subplots(1, 3, figsize=(21, 7.5))
    axs[0].imshow(u8(orgb) * ovalid[..., None], extent=ext)
    axs[0].set_title("RGB ortho (P1) + LiDAR GT crowns")
    lr = lrgb / max(np.percentile(lrgb[foot], 99.5), 1) * 255
    axs[1].imshow(u8(lr), extent=ext)
    axs[1].set_title("LiDAR point colours (top point) + GT crowns")
    im = axs[2].imshow(np.ma.masked_where(~foot, chm), extent=ext, cmap="viridis")
    plt.colorbar(im, ax=axs[2], fraction=0.04, label="height above ground (m)")
    axs[2].set_title("CHM + GT crowns")
    for ax in axs:
        outlines(ax, "red" if ax is axs[2] else "cyan")
        crop(ax)
        ax.set_xlabel("m")
    fig.suptitle(f"Plot {args.plot}: inputs on the common {res} m grid")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "01_inputs.png"), dpi=110)
    plt.close(fig)

    # 2. checkerboard RGB / LiDAR colour, 4 m tiles
    t = px(4.0)
    rr_, cc_ = np.indices(shape)
    board = ((rr_ // t + cc_ // t) % 2).astype(bool)
    # match LiDAR colour radiometry to the ortho per channel for a fair visual
    lm = np.zeros_like(lrgb)
    for b in range(3):
        a, o = lrgb[..., b][foot & ovalid], orgb[..., b][foot & ovalid]
        lm[..., b] = (lrgb[..., b] - a.mean()) / (a.std() + 1e-6) * o.std() + o.mean()
    chk = np.where(board[..., None] & foot[..., None], lm, orgb)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(u8(chk), extent=ext)
    crop(ax)
    ax.set_title(f"Plot {args.plot}: checkerboard, 4 m tiles alternating RGB ortho / LiDAR colour.\n"
                 "Misalignment shows as crowns breaking at tile edges.")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "02_checkerboard.png"), dpi=110)
    plt.close(fig)

    # 3. window displacement field
    fig, axs = plt.subplots(1, 2, figsize=(20, 9.5))
    for ax, p in zip(axs, ("A", "B")):
        ax.imshow(u8(orgb) * 0.6, extent=ext)
        good = (win[f"{p}_distinct"] >= args.min_distinct) & ~win[f"{p}_edge"]
        lx, ly = to_local(win.x.values, win.y.values)
        scale = 5.0
        q = ax.quiver(lx[good], ly[good], win[f"{p}_dx"][good] * scale, win[f"{p}_dy"][good] * scale,
                      win[f"{p}_ncc"][good], cmap="plasma", angles="xy", scale_units="xy", scale=1,
                      width=0.004, clim=(0, 1))
        ax.scatter(lx[~good], ly[~good], marker="x", c="white", s=30, label="ambiguous peak")
        plt.colorbar(q, ax=ax, fraction=0.04, label="peak NCC")
        name = "A: RGB vs LiDAR colour" if p == "A" else "B: RGB vs CHM"
        st = summary[f"windows_{p}"]
        sub = (f"median ({st.get('median_dx_m', np.nan):+.2f}, {st.get('median_dy_m', np.nan):+.2f}) m, "
               f"p95 |d| {st.get('p95_norm_m', np.nan):.2f} m, n={st['n']}/{st['n_total']}")
        ax.set_title(f"{name}  (arrows ×{scale:g})\n{sub}")
        crop(ax)
        ax.legend(loc="lower left")
    fig.suptitle(f"Plot {args.plot}: {args.win:g} m windows, step {args.step:g} m. "
                 "Arrow = where a LiDAR feature appears in the ortho.")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "03_displacement_field.png"), dpi=110)
    plt.close(fig)

    # 4. per-crown offset vs height + global NCC surfaces
    fig, axs = plt.subplots(1, 4, figsize=(24, 5.5))
    for ax, k in zip(axs[:3], ("dx", "dy", "norm")):
        ax.scatter(cr.height_m, cr[k], c="lightgrey", s=20, label="ambiguous")
        ax.scatter(ok.height_m, ok[k], c=ok.ncc, cmap="plasma", vmin=0, vmax=1, s=30, label="trusted")
        if f"{k}_vs_height" in cs:
            fit = cs[f"{k}_vs_height"]
            hh = np.linspace(cr.height_m.min(), cr.height_m.max(), 10)
            ax.plot(hh, fit["intercept_m"] + fit["slope_m_per_m"] * hh, "k--",
                    label=f"slope {fit['slope_m_per_m']:+.3f} m/m, r={fit['pearson_r']:+.2f}")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("crown height (m)")
        ax.set_ylabel(f"{k} (m)")
        ax.legend(fontsize=8)
        ax.set_title(f"C: per-crown {k}")
    s = glob_res["A_colour"]["surface"]
    r_ = args.max_shift
    axs[3].imshow(s, extent=[-r_, r_, -r_, r_], cmap="magma")
    ga = summary["global_A_colour"]
    axs[3].plot(ga["dx_m"], ga["dy_m"], "c+", ms=14)
    axs[3].set_title(f"A global NCC surface (peak {ga['ncc']:.2f})")
    axs[3].set_xlabel("dx (m)")
    axs[3].set_ylabel("dy (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "04_crowns_vs_height.png"), dpi=110)
    plt.close(fig)

    print(f"[done] outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
