# Evaluating LiDAR-vs-RGB tree crown overlap

Methodology for measuring how well RGB-derived tree crown masks reproduce the tree crowns that airborne LiDAR sees from above, at the ASNortheast site.

**LiDAR is the ground truth. RGB (CanopyRS) is the prediction.**

---

## 1. Why this document exists

We have two independent top-view segmentations of the same forest canopy:

All paths below are relative to this directory (`CanopyRS/canopyrs3d/`), which sits alongside the upstream `canopyrs/` package. **`data/` is local only and not committed** — the CanopyRS repo is public; see `CLAUDE.md`.

| | Source | Role |
|---|---|---|
| `data/lidar_masks/asnortheast_plot_{209,210}_gt_crowns_gt.gpkg` | crown polygons projected from hand-annotated airborne LiDAR (`ForestMamba/tools/extract_crown_hulls.py --gt`) | **ground truth** |
| `data/rgb_masks/20250318_asnortheast_50mm_p1_rgb_gr0p07_infer.gpkg` | CanopyRS aggregator output on a 5 cm RGB orthomosaic | **prediction** |

The question is not simply "do these agree". Crown segmentation disagreement is *structured*: one LiDAR crown is often split across several RGB masks, and one RGB mask often swallows several LiDAR crowns. So the evaluation has to

1. link instances **many-to-many**,
2. decide whether the RGB masks cover a **sufficient** part of each LiDAR crown, and
3. reduce that to an **mIoU** number.

**Where this leads.** The RGB predictions come from CanopyRS, and `canopyRS3D` is intended to be built *with* CanopyRS, fine-tuning the model on several modalities including LiDAR. This evaluation is not a one-off report card — it is the diagnostic that says where the RGB-only model breaks down and what LiDAR supervision would have to fix.

## 2. The governing constraint: the ground truth is incomplete

**Some trees were deliberately left out of the LiDAR annotation because they were too uncertain to annotate.**

An RGB mask with no LiDAR counterpart is therefore *not* evidence of a false positive. Most likely it is a real tree the annotator skipped. Everything below follows from this:

- The evaluation is **recall-oriented and GT-anchored throughout**. It measures how well RGB reproduces the crowns LiDAR *did* annotate, and never penalises RGB for finding more.
- **No precision, no F1, no panoptic PQ/RQ** as headline numbers — every false-positive term would punish unannotated trees.
- Unmatched RGB masks are counted and exported for review, never scored.

Any metric added to this repo in future has to respect this.

## 3. What was measured before designing anything

These are probe results against the real files, not assumptions. Several of them killed design complexity that would otherwise have been carried forward.

*This table records the design-time probes, run on plots 209 and 210 only, before plots 203/207/208 existed. It is kept as the rationale for the design choices; §14 holds the current results across all five plots.*

| Property | Finding |
|---|---|
| CRS | Both sides EPSG:32617 (UTM 17N, metres). No reprojection needed. |
| GT features | plot 209: 38 (37 `hull_status='ok'`, 1 `below_canopy`); plot 210: 55 (44 ok, 8 `below_canopy`, 2 `occluded`, 1 `empty_mask`) |
| GT NULL geometries | every row with `hull_status != 'ok'` has NULL geometry, and must be filtered |
| GT self-overlap | **none** — Σ areas equals the union area exactly (top-view occlusion was already resolved upstream by `resolve_top_view`) |
| Prediction self-overlap | **none** within the plots (Σ/union = 1.000) |
| Georeferencing | **aligned.** A ±3 m grid search on rigid XY shift maximising GT∩Pred area gains 0.0 % (209) and 0.4 % (210). *No alignment step is needed* **for this polygon evaluation** — but the probe is too coarse for pixel-level multi-modal training; see §15. |
| Scale mismatch | GT covers ~38×38 m per plot; the RGB file covers ~377×524 m with 4 644 polygons. An AOI restriction is mandatory. |
| Survivors after AOI filter | 35 predictions (209) and 49 (210), out of 4 644 |
| Many-to-many is real | plot 210 yields 9 `split` and 4 `merge` clusters; a single GT crown intersects up to 7 RGB polygons |
| Chaining risk | **refuted.** Connected components stay tiny — max size 2 (209) / 4 (210) at `tau_link=0.5`, and 7 even at 0.3. No GT area lands in a component larger than 8. |
| Staircase-vs-smooth bias | **refuted.** RGB polygons are pixel-quantised at 0.07 m, GT are smoothed contours. Rasterising both to a common 0.07 m grid changes mean best-IoU by **+0.09 % / +0.01 %**; simplifying predictions at 0.07 m tolerance changes it by +0.17 % / +0.01 %. *Use exact vector geometry; no snapping, no simplification.* |
| Linking criterion | **validated.** With `inter/min` at 0.5, zero edges have `IoU < 0.05`. An IoU-only criterion (≥0.25) instead collapses plot 210 from 9 splits to 1 — it would erase the very structure being measured. |
| Degenerate GT | crown areas span 0.0–64 m²; 8 crowns below 3 m² in 209, 5 in 210 |
| Baseline result | `mIoU_cluster` = 0.342 (209) / 0.470 (210); `mCov_cluster` = 0.376 / 0.537, at `tau_link=0.5` |

## 4. Notation

Per plot: GT set `G = {g_1..g_N}` (LiDAR, mutually disjoint), prediction set `P = {p_1..p_M}` (RGB, mutually disjoint), both restricted to the AOI. `a(·)` is planar area in m².

For any pair:

```
inter_ij = a(g_i ∩ p_j)
IoU_ij   = inter_ij / (a(g_i) + a(p_j) - inter_ij)
Cov_ij   = inter_ij / a(g_i)                 # coverage — how much of the LiDAR crown RGB captured
Pur_ij   = inter_ij / a(p_j)                 # purity — DIAGNOSTIC ONLY, not an error term
Ovl_ij   = inter_ij / min(a(g_i), a(p_j))    # size-asymmetry-robust overlap
```

`Pur` is deliberately not an error term: a low value may simply mean the RGB mask also covers an unannotated tree (§2).

## 5. Preprocessing

1. Read both GPKGs with geopandas. Assert identical CRS; reproject predictions if not.
2. GT: drop NULL geometries (equivalently `hull_status != 'ok'`), recording the count per status. Those are trees the LiDAR annotated but could not project to a top view, so they are correctly out of scope for a top-view comparison.
3. Both sides: `make_valid`, drop empties, drop `a < min_area` (default **1.0 m²**, configurable), recording counts. **Keep MultiPolygons intact — do not explode them.** A crown legitimately splits into parts under occlusion; exploding would inflate the instance count.
4. Build `AOI = convex_hull(unary_union(GT))` per plot. GT lies inside by construction.
5. Predictions: `sindex` prefilter against the AOI, drop any with `a(p ∩ AOI)/a(p) < 0.4` (the rule CanopyRS's `filter_min_overlap` uses), then clip survivors to the AOI. Expect 35 (209) and 49 (210) to survive.
6. Flag GT crowns within 1 m of the AOI boundary as `edge_gt`. Metrics are reported with and without them, since a truncated crown depresses IoU.

## 6. Sparse overlap matrix

Enumerate only intersecting pairs via the geopandas `sindex`, then compute `inter, IoU, Cov, Pur, Ovl` for each candidate pair — 45 pairs in plot 209, 141 in plot 210. Trivially fast at this size; the sparse form is what keeps it correct as more plots are added.

## 7. Many-to-many linking

### Edge criterion

Draw an edge between `g_i` and `p_j` iff

```
Ovl_ij ≥ tau_link   AND   IoU_ij ≥ tau_floor
        (defaults: tau_link = 0.5, tau_floor = 0.05)
```

**Why min-normalisation.** In a *split* — one big GT crown, several small RGB masks inside it — each small mask is almost entirely explained by the GT crown, so `Ovl ≈ 1` and it links even though its `IoU` is small. In a *merge* — one big RGB mask over several GT crowns — each small GT crown is almost entirely inside the mask, so it links too. A mere boundary graze between neighbours produces an intersection that is small relative to *both*, so it does not link. `Cov` alone would break the merge case; `IoU` alone breaks both, as measured (§3).

**Why `tau_floor`.** It guards the one theoretical weakness of min-normalisation: a tiny crown fully contained in a huge mask links at `Ovl = 1` no matter how mismatched the two are in scale. On this data it is a **no-op** — no edge at `Ovl ≥ 0.5` has `IoU < 0.05` in either plot (only 3 fall below 0.10, both in plot 210), and enabling it changes every metric by 0.000. It is kept as a cheap safeguard for future plots. The script logs how many edges it removes, so a silent regime change is visible.

### Clusters

Connected components of the bipartite graph (union-find). Each `C_k = (G_k, P_k)` is one many-to-many group.

| `\|G_k\|` | `\|P_k\|` | Label | Meaning |
|---|---|---|---|
| 1 | 1 | `one_to_one` | clean match |
| 1 | ≥2 | `split` | RGB over-segments the LiDAR crown |
| ≥2 | 1 | `merge` | RGB under-segments — one mask spans several crowns |
| ≥2 | ≥2 | `tangled` | ambiguous group |
| 1 | 0 | `missed` | LiDAR saw a crown, RGB produced nothing — a genuine RGB failure, counts against the score |
| 0 | 1 | `unmatched_pred` | RGB mask with no LiDAR crown — **reported, never penalised** (§2) |

Measured at `tau_link = 0.5`:

- **plot 209** → 23 `one_to_one`, 11 `missed`, 12 `unmatched_pred`, and **no splits or merges at all**. Its failure mode is pure detection, not partitioning.
- **plot 210** → 15 `one_to_one`, 9 `split`, 4 `merge`, 5 `missed`, 12 `unmatched_pred`. Genuine many-to-many structure.

### Chaining guard

Because both sides tile the canopy without gaps, a low `tau_link` could in principle chain a whole plot into one giant component. Measured, it does not (§3), so this stays a safety net rather than a core mechanism:

- log the component-size histogram and the fraction of GT area in components with `|G_k| + |P_k| > 8`, and assert that fraction is 0;
- if a future plot trips it, re-run linking *within that component only* at `tau_link + 0.1` steps up to 0.9 until it breaks up, and label whatever remains `tangled`.

## 8. Does the RGB mask cover a sufficient part of the crown?

For each GT crown `g_i`, with `P(g_i)` its linked predictions:

```
Cov_i = a( g_i ∩ ⋃_{p ∈ P(g_i)} p ) / a(g_i)
```

Report the distribution of `Cov_i` and the fraction of crowns reaching `Cov_i ≥ tau_cov` for `tau_cov ∈ {0.5, 0.75, 0.9}`. This is the direct answer to the sufficiency question.

Also report `Cov_i^best = max_j Cov_ij` (best single mask), so coverage achieved by stitching several fragments is distinguishable from coverage achieved by one good mask.

`Cov` is structurally immune to ground-truth incompleteness — it only ever divides by GT area — which makes it the most trustworthy family of numbers here.

## 9. Metrics

### Headline — cluster mIoU (many-to-many)

```
IoU_k        = a( ⋃G_k ∩ ⋃P_k ) / a( ⋃G_k ∪ ⋃P_k )
mIoU_cluster = mean over clusters with |G_k| ≥ 1 of IoU_k
```

- `missed` clusters contribute `IoU_k = 0`.
- `unmatched_pred` clusters are **excluded from the mean entirely** — that exclusion is precisely what implements §2. They are reported separately as a count, a share of AOI area, and an `aggregator_score` distribution.

This is the headline because it *credits* a correct split or merge rather than punishing it, which is exactly the many-to-many framing. Measured baseline: **0.342 (209), 0.470 (210)**.

**Residual bias, stated plainly.** In a `merge` or `tangled` cluster, one RGB mask may span both annotated crowns *and* an unannotated neighbour. That extra area enters `⋃P_k`, inflating the denominator and depressing `IoU_k`. **`mIoU_cluster` is therefore a lower bound.** This is why it is always reported paired with the companion below.

### Companion — cluster coverage (bias-free recall)

```
Cov_k        = a( ⋃G_k ∩ ⋃P_k ) / a( ⋃G_k )
mCov_cluster = mean over clusters with |G_k| ≥ 1 of Cov_k
```

Always report these two side by side. The gap between them localises the problem: high `mCov` with low `mIoU_cluster` means RGB masks systematically extend beyond the annotated crowns — which, given §2, is more likely an annotation gap than a segmentation error. Measured baseline: **0.376 (209), 0.537 (210)**.

### Secondary metrics

- **`mIoU_cluster_area` / `mCov_cluster_area`** — weighted by `a(⋃G_k)`. Crown areas span 0.4–64 m², so unweighted means are dominated by small crowns; these reflect canopy area instead.
- **`mIoU_1to1`** — optimal one-to-one assignment via `scipy.optimize.linear_sum_assignment` on cost `-IoU_ij`, run **per connected component** so it stays cheap. Mean over all `N` GT crowns, 0 for unassigned. The standard instance-segmentation number, comparable to other work — but it penalises correct splits and merges, hence secondary.
- **`mIoU_best`** — `mean_i max_j IoU_ij`. Lenient (several GT crowns may claim the same mask), but mirrors ForestMamba's `cov_per_gt` in `tools/eval_predictions.py`, which makes the 2D and 3D evaluations directly comparable.
- **`Recall@IoU`** — fraction of GT crowns with `IoU ≥ t` under the one-to-one assignment, for `t ∈ {0.25, 0.5, 0.75}`. Recall only. If a precision-style number is ever wanted, it must be labelled a lower bound.
- **`IoU_global`** = `a(⋃G ∩ ⋃P) / a(⋃G ∪ ⋃P)` and **`Cov_global`** = `a(⋃G ∩ ⋃P) / a(⋃G)` — instance-agnostic canopy overlap, separating "RGB missed canopy" from "RGB mis-partitioned canopy".

  ⚠️ **`IoU_global` is the one metric in this script that an unmatched mask can cost something.** It unions *every* prediction, so a mask over a tree the LiDAR never annotated enters the denominator and depresses it. That contradicts §2, so **`IoU_global_linked`** is reported next to it — the same ratio over masks linked to at least one GT crown, which is the GT-anchored figure to quote. Measured effect: pooled 0.653 → 0.671, and 0.449 → 0.538 on plot 209. `Cov_global` divides by GT area only and is immune.

  Note that `unmatched_pred` does **not** mean "disjoint from the ground truth" — it means no edge cleared `tau_link`. Such a mask may still clip a crown, which is why removing them can lower the ratio slightly (plot 203: 0.626 → 0.618).
- **Fragmentation** — mean predictions per GT crown (split factor), mean GT crowns per prediction (merge factor), and the cluster-type histogram from §7.

## 10. Implementation

One self-contained script, **`scripts/eval_crown_overlap.py`** (implemented), argparse CLI, no package scaffolding. Tests in `tests/test_eval_crown_overlap.py`.

Run with the `canopyrs` conda env, which already has everything needed (from this directory):

```bash
/home/hugobaudchon/anaconda3/envs/canopyrs/bin/python scripts/eval_crown_overlap.py \
    --gt-dir data/lidar_masks --pred data/rgb_masks/*.gpkg --out-dir output
```

Dependencies: geopandas 1.0.1, shapely 2.0.1, scipy, pandas, matplotlib. `rasterio` is needed only for the optional rasterisation cross-check in §3.

### Outputs → `output/`

| File | Contents |
|---|---|
| `per_gt.csv` | `plot, instance_id, area_m2, height_m, cluster_id, cluster_type, n_linked_preds, cov, cov_best, best_iou, iou_1to1, sufficient@{0.5,0.75,0.9}, edge_gt` |
| `per_cluster.csv` | `plot, cluster_id, cluster_type, n_gt, n_pred, gt_area, pred_area, inter_area, iou_k, cov_k` |
| `unmatched_preds.csv` | RGB masks with no LiDAR counterpart: `canopyrs_object_id, area, aggregator_score`. **Not an error list** — a review queue for judging how many are genuinely unannotated trees. |
| `summary.json` | every metric in §9, per plot and pooled, plus preprocessing counts (dropped by `hull_status`, by `min_area`, by AOI filter) |
| `links.gpkg` | GT and predictions with `cluster_id` / `cluster_type` attached — the single most valuable output, for visual QA in QGIS |
| `sweeps.csv` + figure | headline metric vs `tau_link`, and coverage sufficiency vs `tau_cov` |

Everything in `output/` is a regenerable artefact, not an input.

## 11. Sensitivity

`mIoU_cluster` vs `tau_link`, measured at 0.3 / 0.4 / 0.5 / 0.6 / 0.7 (full grid in `output/sweeps.csv`):

| Plot | 0.3 | 0.4 | **0.5** | 0.6 | 0.7 | slope d(mIoU)/dτ |
|---|---|---|---|---|---|---|
| 203 | 0.338 | 0.330 | **0.315** | 0.315 | 0.278 | −0.138 |
| 207 | 0.524 | 0.510 | **0.474** | 0.451 | 0.407 | −0.294 |
| 208 | 0.514 | 0.496 | **0.496** | 0.483 | 0.467 | −0.112 |
| 209 | 0.345 | 0.342 | **0.342** | 0.342 | 0.327 | −0.024 |
| 210 | 0.500 | 0.491 | **0.470** | 0.428 | 0.398 | −0.260 |

**How to read the slope.** It is always negative, and that sign carries no information: raising τ can only delete edges, and a deleted edge turns a crown into `missed` with IoU = 0. (`corr(τ, n_edges) ≈ −0.95…−0.99`, `corr(τ, n_missed) ≈ +0.9`.) The downward trend is therefore mostly bookkeeping — crowns leaving the matched population — not geometric agreement degrading.

**The magnitude is the signal: it says how much of a plot's score is a linking decision rather than a measurement.** Plot 209 at −0.024 is effectively threshold-independent, so its 0.342 is a hard number. Plot 207 at −0.294 moves 0.117 across the range, an eighth of its own value, so its 0.474 is soft and must never be quoted without the τ it came from. The ranking by |slope| tracks how much many-to-many structure each plot has: 209 has zero splits and nothing for τ to act on, while 207 and 210 have the most. Where splits exist, `corr(τ, n_split)` runs −0.38 to −0.72 — raising τ specifically dismantles split clusters.

**`mIoU_cluster` and `mCov_cluster` co-move at `corr ≈ +0.98…+1.00`** across τ on every structured plot. τ shifts cluster *membership* without changing the balance between coverage and over-extension, which means the IoU-vs-coverage gap (the over-extension evidence in §14) is invariant to the threshold choice. Had they decoupled, τ would be trading one failure mode for another and that diagnosis would be threshold-dependent. Plot 209's lower +0.78 is an artifact of correlating a near-constant series, not a real difference.

**Publish the sweep alongside the point estimate**, always.

## 12. Known limitations

- **Ground-truth incompleteness** (§2) is the governing constraint. Every metric here is recall-flavoured; no number may be read as a precision estimate.
- **Plots fail differently.** Plot 209 has zero split/merge structure and 11 missed crowns — a detection problem; plot 203 likewise (17 missed of 40). Plots 207 and 210 have real partitioning structure. Never pool without also reporting per plot, and read every figure next to its `n_gt`.
- **Uneven ground-truth attrition.** `below_canopy` removes 1 crown in plots 203/209 but 21–23 in plots 207/208, so plot 208's score rests on 21 crowns against plot 207's 51. Per-plot numbers are not equally precise.
- **Plot-boundary truncation.** Predictions are clipped to the AOI; GT is not, since it defines the AOI. The `edge_gt` flag and the with/without-edge split make the residual effect visible.
- **No score threshold applied**, by decision — CanopyRS already applied score thresholding and NMS upstream. `aggregator_score` (0.43–0.96) remains available should a sweep be wanted later.
- **Two plots, ~72 usable crowns total.** Every number carries wide error bars; report counts next to every mean.

## 13. Verification

**Unit tests** in `tests/test_eval_crown_overlap.py` (33 tests, run with `python -m pytest tests/`), on synthetic boxes with analytically known IoU: one-to-one; a 1→3 split; a 3→1 merge; a missed crown; an unmatched prediction (must be *excluded* from the mean, not scored 0); a boundary-graze pair that must **not** link; a deliberately chained strip to exercise the §7 guard; plus the preprocessing filters, coverage thresholds, and the invariants below.

**Runtime invariants** — assert, don't hope:
- GT areas sum to the GT-union area (disjointness);
- every GT crown belongs to exactly one cluster;
- Σ cluster GT areas equals total GT area;
- `mIoU_1to1 ≤ mIoU_best` (the 1-1 assignment can never beat the per-crown best);
- `iou_k ≤ cov_k` for every cluster (the union can never be smaller than the GT area).

An earlier draft of this section also asserted `mIoU_best ≤ mCov_cluster`. That does **not** hold in general and is not asserted: the two average over different populations (`mIoU_best` over `N` GT crowns, `mCov_cluster` over `K` clusters), so a plot with many merge clusters can invert them. It happens to hold on both plots here, which is why it looked like an invariant.

**End-to-end regression** against the measured baseline. For plots 209 / 210 at `tau_link = 0.5` the script must reproduce:

| Quantity | 209 | 210 |
|---|---|---|
| predictions surviving AOI filter | 35 | 49 |
| GT crowns after `min_area=1.0` | 34 | 38 |
| candidate pairs | 45 | 141 |
| linking edges | 23 | 42 |
| `mIoU_cluster` | 0.342 | 0.470 |
| `mCov_cluster` | 0.376 | 0.537 |

plus the cluster-type histograms in §7, and `tau_floor = 0.05` removing 0 edges on both plots. Disagreement means the preprocessing is wrong.

**Visual QA.** Open `links.gpkg` in QGIS and eyeball a few `split`, `merge` and `tangled` clusters, plus a sample of `unmatched_pred` masks to judge what share are genuinely unannotated trees. The linking is the part most likely to be subtly wrong, and only looking at it catches that. **This is the one verification step still outstanding** — everything above is automated and passing.

## 14. Results

> **Data regenerated 2026-11-10 — the table below is the previous data set.** Predictions are now `data/rgb_masks/20250318_asnortheast_50mm_p1_rgb_sf1p0_inferfinal.gpkg` (new CanopyRS run at 0.045 m/px, 3 502 masks site-wide) instead of `…_gr0p07_infer.gpkg` (0.07 m/px, 4 644 masks), and the GT was re-extracted with ForestMamba `extract_crown_hulls.py --gt` defaults (§16; same crown counts, outlines differ by ≤ 0.05 m² per crown). The old files are kept locally in `data/_previous_2026-11-10/`. Re-run at defaults, pooled: `mIoU_cluster` 0.414 → **0.394**, `mCov_cluster` 0.464 → 0.449, `mIoU_cluster_area` 0.589 → 0.603, `mIoU_1to1` 0.362 → 0.334, `IoU_global_linked` 0.671 → 0.690, Recall@IoU 0.75 0.03 → 0.05. Per plot `mIoU_cluster`: 203 0.296, 207 0.474, 208 0.542, 209 0.276, 210 0.424. Predictions surviving the AOI filter drop on every plot (203 28→24, 207 80→67, 208 30→29, 209 35→27, 210 49→41) while GT counts are unchanged, so the §13 end-to-end regression targets describe the previous data and no longer apply. Full detail in `output/summary.json`.

Five plots, produced by `scripts/eval_crown_overlap.py` at defaults (`tau_link=0.5`). Full detail in `output/summary.json`; every regression target in §13 reproduced exactly, and adding plots 203/207/208 left the 209/210 figures bit-identical.

| | 203 | 207 | 208 | 209 | 210 | **pooled** |
|---|---|---|---|---|---|---|
| **`mIoU_cluster`** (optimistic) | 0.315 | 0.474 | 0.496 | 0.342 | 0.470 | **0.414** |
| **`mIoU_cluster_penalised`** (floor) | 0.315 | 0.373 | 0.407 | 0.342 | 0.346 | **0.353** |
| τ-sensitivity (slope, §11) | −0.138 | −0.294 | −0.112 | −0.024 | −0.260 | — |
| `mCov_cluster` (companion) | 0.347 | 0.544 | 0.531 | 0.376 | 0.537 | 0.464 |
| `mIoU_cluster_area` | 0.555 | 0.576 | 0.648 | 0.486 | 0.593 | 0.589 |
| `mCov_cluster_area` | 0.599 | 0.631 | 0.671 | 0.510 | 0.652 | 0.631 |
| `mIoU_1to1` | 0.307 | 0.385 | 0.459 | 0.342 | 0.354 | 0.362 |
| `mIoU_best` | 0.322 | 0.405 | 0.461 | 0.352 | 0.396 | 0.382 |
| `IoU_global` (penalised, see §9) | 0.626 | 0.663 | 0.697 | 0.449 | 0.700 | 0.653 |
| `IoU_global_linked` (GT-anchored) | 0.618 | 0.693 | 0.699 | 0.538 | 0.705 | 0.671 |
| `Cov_global` | 0.658 | 0.753 | 0.726 | 0.557 | 0.774 | 0.717 |
| Recall@IoU 0.25 | 0.53 | 0.71 | 0.76 | 0.68 | 0.68 | 0.66 |
| Recall@IoU 0.5 | 0.33 | 0.39 | 0.67 | 0.38 | 0.32 | 0.39 |
| Recall@IoU 0.75 | 0.03 | 0.02 | 0.05 | 0.03 | 0.03 | 0.03 |
| Coverage ≥ 0.5 | 43 % | 71 % | 76 % | 50 % | 71 % | 61 % |
| Coverage ≥ 0.75 | 10 % | 22 % | 5 % | 9 % | 37 % | 18 % |
| GT crowns / predictions | 40 / 28 | 51 / 80 | 21 / 30 | 34 / 35 | 38 / 49 | 184 / 222 |
| unmatched predictions | 5 (27 m²) | 23 (148 m²) | 6 (60 m²) | 12 (91 m²) | 12 (116 m²) | 58 (442 m²) |

Cluster types, pooled: 95 `one_to_one`, 43 `missed`, 25 `split`, 7 `merge`, 3 `tangled`, 58 `unmatched_pred`.

**Reading these numbers.**

1. **Plots fail for different reasons; never quote the pooled number alone.** Plot 209 has *zero* splits or merges — RGB either matches a crown one-to-one or misses it outright (11 of 34). Its `mIoU_cluster` equals its `mIoU_1to1` to 14 decimal places, the signature of a pure detection problem, and its τ-slope of −0.024 confirms there is no linking structure to perturb. Plots 207 and 210 are the opposite: `mIoU_cluster` sits 0.09–0.12 above `mIoU_1to1` because the many-to-many view recovers splits a 1-1 metric would discard, and their steep τ-slopes say the same thing.
2. **Plot 203 is the weakest (0.315) and it is a detection failure**: 17 of 40 crowns `missed`, and only 28 predictions for 40 crowns — the sole plot where RGB produces *fewer* instances than LiDAR annotated.
3. **Area weighting lifts every plot by 0.10–0.24.** Failures concentrate in small crowns; by canopy area agreement is materially better than the unweighted mean suggests. Plot 203 moves most (0.315 → 0.555), so its problem is specifically small trees.
4. **`IoU_global_linked` (0.671 pooled) far exceeds `mIoU_cluster` (0.414).** RGB recovers most of the canopy *area*; it is the per-instance partitioning that disagrees. That gap is the concrete argument for LiDAR supervision. Quote the `_linked` row: the plain `IoU_global` is depressed by masks over unannotated trees (§9), most visibly on plot 209 (0.449 vs 0.538).
5. **Boundaries agree far less than presence.** Recall@IoU 0.25 is 0.66 but Recall@IoU 0.75 is 0.03 — uniformly, on every plot. RGB finds roughly the right trees and draws roughly the wrong outlines. This is the single most consistent finding across the site.
6. **Ground-truth attrition is severe and uneven.** Plot 207 keeps 51 of 85 annotated trees and plot 208 keeps 21 of 50 — mostly `below_canopy` (23 and 21 respectively), i.e. trees the LiDAR annotated that never reach the canopy top and so are invisible from above by construction. Plot 208's headline 0.496 therefore rests on 21 crowns. Read every figure next to its `n_gt`.
7. **58 unmatched RGB masks (442 m²) are excluded from every metric except `IoU_global`**, per §2 and the caveat in §9. Within GT-anchored clusters, linked masks put only 1.3–4.5 % of their area outside any annotated crown, so the documented lower-bound effect on `mIoU_cluster` is real but small. Plot 207 alone contributes 23. Reviewing what share are genuinely unannotated trees remains the highest-value next step, since it bounds how pessimistic these numbers are.

## 15. RGB ↔ LiDAR raster co-registration (planned)

Everything above compares *polygons*. Fine-tuning CanopyRS on stacked RGB + LiDAR channels needs the **rasters** to agree at pixel level, which is a much stricter requirement. This section plans how to measure and, if needed, correct that. Status: **Phase 2 implemented (`scripts/check_alignment.py`); plot 203 measured (§15.9); other plots pending.**

### 15.1 Why the §3 probe is not enough

The §3 georeferencing probe (±3 m rigid XY shift on GT∩Pred polygon area, plots 209/210) is sound for the polygon evaluation but cannot certify raster alignment:

- **Low sensitivity.** It is computed against CanopyRS predictions whose boundaries already disagree with GT (Recall@IoU 0.75 = 0.03, §14), so the objective is flat — a 0.3–1 m offset would not register.
- **Translation only.** It cannot detect rotation, scale, or spatially varying distortion.
- **Blind to relief displacement**, the most likely forest-ortho failure: tree tops displaced proportionally to height.
- **Two of twelve plots.**

### 15.2 Facts established (probed, not assumed)

| Property | Finding |
|---|---|
| Acquisition dates | LiDAR `gps_time` decodes to 2025-03-17 (plots 153/161/170/264) and 2025-03-18 (203–210, 256, 257) — same days as the `20250318_*` orthos. No phenological or growth change. |
| CRS | LAS headers and both local orthos declare EPSG:32617 (WGS 84 / UTM 17N). Equality of labels, not yet of practice (§15.3). |
| LiDAR colour | Every point carries populated `red/green/blue` (max ≈ 56 000–65 000). A LiDAR-colourised raster is a near-same-modality reference for the ortho. |
| LiDAR Z | **Height-normalised**, not elevation (≈ −2 to 26 m). Absolute Z requires the DTM (`ForestMamba/tools/dtm_reader.py`, `denormalize_height.py`). Plot 257 reaches 55 m — outlier noise to filter. |
| Local orthos | `../../rgb_ortho/` holds `asnorthnorth` (≈1.10 cm/px; contains plots 153, 161, 170, 264) and `asnorthwestcent` (≈1.04 cm/px; contains none). |
| Missing ortho | The `asnortheast` ortho behind the §14 predictions (plots 203–210, 256, 257) is **not on disk locally**. |

### 15.3 Phase 0 — Inventory

1. Locate the `asnortheast` ortho; record its GSD and pixel-grid origin.
2. Record ortho provenance: software, **DSM-based (true ortho) vs DTM-based**, RTK/PPK/GCPs. Same for LiDAR (sensor, trajectory processing). DTM-based orthorectification predicts metres of displacement on 25 m crowns and makes §15.6 the main work.
3. Verify CRS and vertical datum agree in practice (§15.5 measures it).
4. Fix one **reference grid per plot**: the ortho pixel grid, plus training-GSD derivatives (e.g. 0.07 m, the inference GSD). All LiDAR rasters snap to these; no ad-hoc resampling anywhere downstream.

### 15.4 Phase 1 — LiDAR rasters on the reference grid (`scripts/lidar_rasters.py`)

Per plot: pit-free **CHM** (max-Z, outliers filtered), **LiDAR-colour RGB** (top-return colour), **intensity**, **point density**. These double as the LiDAR input channels for fine-tuning, so they are built once, correctly.

### 15.5 Phase 2 — Measure misalignment three independent ways (`scripts/check_alignment.py`)

Each estimator runs on overlapping ~10×10 m windows, producing a **displacement field** rather than one number.

| Estimator | Signal | Strength |
|---|---|---|
| **A. RGB ↔ LiDAR-colour** | phase correlation / NCC between ortho and LiDAR-colour raster | sub-pixel; near-same modality |
| **B. RGB ↔ CHM structure** | mutual information or gradient-orientation NCC (ortho luminance/edges vs CHM edges/gaps) | independent of the LiDAR's own colourisation, which may itself be mis-projected |
| **C. Object level** | per-crown centroid offsets for linked GT↔pred pairs from `eval_crown_overlap.py`, regressed on **crown height** and **direction** | diagnoses relief displacement (offset ∝ height, radial pattern) |

Plus **manual tie-points**: 10–20 per plot (gaps, dead trees, trail edges) picked in QGIS as independent truth. A–C are trusted only where they agree with the tie-points.

Outputs → `output/alignment/`: per-window offsets (CSV + GPKG), per-plot rigid/affine fit with residuals, offset-vs-height plot, QA overlays (CHM contours on RGB, checkerboard/swipe composites).

### 15.6 Phase 3 — Decide and correct

Tolerance, fixed before looking: **p95 residual ≤ 2 px at the training GSD (≈ 0.14 m at 0.07 m)**. Wind sway (~0.1–0.3 m at crown tops) is an irreducible floor.

| Diagnosis | Correction |
|---|---|
| Constant offset per plot | translate the LiDAR rasters (never resample the RGB); store the shift in a per-plot sidecar |
| Rotation / scale / tilt | per-plot 2D affine fitted to the window offsets |
| Offset ∝ height (DTM-based ortho) | best: re-orthorectify on a LiDAR-derived DSM if raw images exist; else warp RGB with CHM × measured displacement-per-metre; failing both, document as label noise |
| Unstructured local error | smooth displacement field only where A and B agree; otherwise flag windows and exclude them from training |

Apply the same transform to the GT crown polygons so evaluation and training stay consistent, then re-run `eval_crown_overlap.py` and compare with §14 as the baseline.

### 15.7 Phase 4 — Lock it in

- **Tests**: synthetic known shift / rotation / height-proportional displacement must be recovered to ≤ 0.25 px.
- **Gate**: multi-modal tile creation refuses any plot without a passing alignment report.
- **Docs**: replace "verified to be well co-registered" in `CLAUDE.md` with the measured per-plot residuals and the date measured.

### 15.8 Open questions that change the plan

1. Where is the `asnortheast` ortho, and do the raw drone images exist (needed for re-orthorectification)?
2. DSM- or DTM-based orthorectification?

**Cheapest first step:** run estimator A on plots 153 / 161 / 170 / 264, which sit inside the local `asnorthnorth` ortho — a real displacement field before any other infrastructure is built.

### 15.9 Results — plot 203 (2026-11-05)

`scripts/check_alignment.py` on a 0.05 m grid; ortho `../../rgb_ortho/20250318_asnortheast_50mm_p1_rgb.cog.tif` (1.10 cm/px, resampled by averaging); 74.7 M points, 47 LAZ files. Outputs in `output/alignment/plot_203/`. Shift convention: a feature at LiDAR (x, y) appears in the ortho at (x+dx, y+dy).

| Measure | Result |
|---|---|
| Self-test (injected (+0.37, −0.52) m) | recovered to **0.003 m** — sign and sub-pixel logic correct |
| A, global (RGB vs LiDAR colour) | **(+0.013, −0.047) m**, \|d\| = 0.048 m; single clean NCC peak |
| A, 10 m windows (45 of 60 had a clear match) | median (+0.03, −0.04) m; std 0.08 / 0.06 m; p50 \|d\| 0.06 m, **p95 0.15 m**; 40/45 windows ≤ 0.15 m |
| B, CHM-only (10 of 60 had a clear match) | median (+0.00, −0.03) m — agrees with A; most windows give no clear match, so B is corroboration only |
| C, per crown (20 of 43 had a clear match) | median (+0.03, −0.04) m; residual after rigid p50 0.08 m |
| Height dependence | none. Theil–Sen \|d\| vs height −0.003 m/m (CI −0.027…+0.003), Spearman p = 0.57; the 6 crowns ≥ 10 m tall (up to 19 m) all have \|d\| ≤ 0.15 m |

**Verdict: aligned; no correction needed.** The constant offset is ~5 cm (1 px at 0.05 m, < 1 px at the 0.07 m training GSD). The p95 of 0.15 m sits at the §15.6 tolerance (0.14 m) but is dominated by noise in the estimator itself (cross-camera NCC peaks of only 0.2–0.4). No offset grows with tree height, so the ortho behaves like a true ortho (DSM-based); relief displacement is not a concern on this plot. The large outliers are all matching failures, not misalignment: crowns of 0.6–7 m² (ids 14, 18, 28: 0.4–2.2 m) and one grass window (mean CHM 0.4 m, 0.47 m). The checkerboard (`02_checkerboard.png`) shows crowns and dead-branch skeletons continuous across tile edges.

**Caveats.** (1) Estimator A uses the L2 camera's point colours, so it checks the colourised cloud. The geometry-only checks (B, plus the GT outlines in `01_inputs.png` sitting on the RGB crowns) agree but are weaker. (2) One plot only; repeat on 207–210, 256, 257 and on 153/161/170/264 (the `asnorthnorth` flight, 2025-03-17). (3) Manual tie-points (§15.5) not yet collected.

## 16. LiDAR GT geometry: visible vs full footprint (2026-11-10)

### 16.1 Why GT crowns look like they overlap at their edges

`scripts/analyse_gt_edges.py` (outputs in `output/gt_edges/`) measured the visible GT layer on all five plots. **There is no area overlap**: Σ crown area equals the union on every plot, and the largest pairwise intersection is 7×10⁻¹¹ m² (floating-point noise). What reads as overlap in QGIS has two causes, both from how `extract_crown_hulls.py` builds the visible layer:

1. **Shared, coincident edges.** `resolve_top_view` places crowns tallest-first and replaces each shorter crown by `raw − ⋃ taller raws`, so the shorter crown's new edge *is* the taller crown's outline, vertex for vertex. 182 of 207 crowns share part of their outline with a neighbour (median 22–55 % of the perimeter, by plot), and QGIS draws both strokes on the same line. Before the clip, the raw masks overlapped by 160 m² (≈ 4 % of GT area; 151 of 207 crowns clipped); the band removed is a median **0.18 m** deep (IQR 0.12–0.27 m), i.e. about one 0.15 m raster cell plus the 0.225 m smoothing buffer — consistent with mask dilation rather than genuine 3D interlocking.
2. **Thin slivers.** Where a shorter crown's mask reached into the gap between two taller crowns, the clip leaves a narrow wedge pinched between them: 111 parts narrower than 0.4 m in clipped crowns vs 6 in unclipped ones, 94 % touching a taller crown, median width 4.5 cm, 5.7 m² in total (0.14 % of GT area).

Neither double-counts area, so §2–§14 are unaffected. The systematic effect is that every contested edge goes to the taller tree. For training masks the slivers (1–2 px at 0.045–0.07 m) are worth removing or re-assigning to the neighbour they touch.

### 16.2 Two GT layers

ForestMamba `tools/extract_crown_hulls.py` now writes two footprints per plot (`--footprint both`, the default; see ForestMamba `readme_crowns.md`):

| Layer | Path | Built from | Overlap | Overtopped trees |
|---|---|---|---|---|
| **visible** (the GT used everywhere above) | `data/lidar_masks/asnortheast_plot_{…}_gt_crowns_gt.gpkg` | top 1 m of canopy, then clipped by taller neighbours | none | no polygon (`below_canopy`) |
| **full** | `data/lidar_masks_full/asnortheast_plot_{…}_gt_crowns_full_gt.gpkg` | every leaf point, no clip | wherever trees overlap in plan view | whole crown |

Both use the same mask and smoothing settings and carry a `footprint` column; join on `instance_id`. The full layer lives in its own directory because `eval_crown_overlap.py` globs every `*.gpkg` in `--gt-dir`.

| Plot | visible crowns | full crowns | visible area (m²) | full Σ area (m²) | full union (m²) |
|---|---|---|---|---|---|
| 203 | 43 | 44 | 643 | 782 | 650 |
| 207 | 57 | 85 | 981 | 1 579 | 993 |
| 208 | 26 | 50 | 1 133 | 1 621 | 1 140 |
| 209 | 37 | 38 | 381 | 408 | 382 |
| 210 | 44 | 55 | 962 | 1 380 | 973 |

The full layer recovers the 65 `below_canopy` / `occluded` / `empty_mask` trees the visible layer cannot show (1 on 203, 28 on 207, 24 on 208, 1 on 209, 11 on 210). Its union is within ~1 % of the visible area, so the extra Σ area is almost entirely stacked crowns: 42–59 % of the union is covered by more than one full crown on 207/208/210, 7 % on 209.

**Which to evaluate against.** The visible layer is the right GT for an RGB orthomosaic: the camera only sees the top of the canopy, and §2's recall-oriented metrics assume disjoint GT. The full layer is the target if CanopyRS3D is meant to predict whole crowns from LiDAR, and it breaks the disjointness invariant in §13, so `eval_crown_overlap.py` must not be pointed at it unchanged.
