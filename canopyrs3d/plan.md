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

| Property | Finding |
|---|---|
| CRS | Both sides EPSG:32617 (UTM 17N, metres). No reprojection needed. |
| GT features | plot 209: 38 (37 `hull_status='ok'`, 1 `below_canopy`); plot 210: 55 (44 ok, 8 `below_canopy`, 2 `occluded`, 1 `empty_mask`) |
| GT NULL geometries | every row with `hull_status != 'ok'` has NULL geometry, and must be filtered |
| GT self-overlap | **none** — Σ areas equals the union area exactly (top-view occlusion was already resolved upstream by `resolve_top_view`) |
| Prediction self-overlap | **none** within the plots (Σ/union = 1.000) |
| Georeferencing | **aligned.** A ±3 m grid search on rigid XY shift maximising GT∩Pred area gains 0.0 % (209) and 0.4 % (210). *No alignment step is needed.* |
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
- **Fragmentation** — mean predictions per GT crown (split factor), mean GT crowns per prediction (merge factor), and the cluster-type histogram from §7.

## 10. Implementation

One self-contained script, **`scripts/eval_crown_overlap.py`**, argparse CLI, no package scaffolding.

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

`mIoU_cluster` vs `tau_link`, measured at 0.3 / 0.4 / 0.5 / 0.6 / 0.7:

| Plot | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|
| 209 | 0.346 | 0.342 | 0.342 | 0.342 | 0.327 |
| 210 | 0.497 | 0.491 | 0.470 | 0.428 | 0.398 |

Plot 209 is flat; plot 210 declines steadily. The headline is threshold-sensitive exactly where many-to-many structure exists, so **the sweep must always be published alongside the point estimate** rather than the single number quoted alone.

## 12. Known limitations

- **Ground-truth incompleteness** (§2) is the governing constraint. Every metric here is recall-flavoured; no number may be read as a precision estimate.
- **The two plots fail differently.** Plot 209 has zero split/merge structure and 11 missed crowns — a detection problem. Plot 210 has real partitioning structure. Never pool without also reporting per plot.
- **Plot-boundary truncation.** Predictions are clipped to the AOI; GT is not, since it defines the AOI. The `edge_gt` flag and the with/without-edge split make the residual effect visible.
- **No score threshold applied**, by decision — CanopyRS already applied score thresholding and NMS upstream. `aggregator_score` (0.43–0.96) remains available should a sweep be wanted later.
- **Two plots, ~72 usable crowns total.** Every number carries wide error bars; report counts next to every mean.

## 13. Verification

**Unit tests** on synthetic polygons with analytically known IoU, covering: one-to-one; a 1→3 split; a 3→1 merge; a missed crown; an unmatched prediction (must be *excluded* from the mean, not scored 0); a boundary-graze pair that must **not** link; and a deliberately chained strip to exercise the §7 guard.

**Runtime invariants** — assert, don't hope:
- GT areas sum to the GT-union area (disjointness);
- every GT crown belongs to exactly one cluster;
- Σ cluster GT areas equals total GT area;
- `mIoU_1to1 ≤ mIoU_best ≤ mCov_cluster`.

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

**Visual QA.** Open `links.gpkg` in QGIS and eyeball a few `split`, `merge` and `tangled` clusters, plus a sample of `unmatched_pred` masks to judge what share are genuinely unannotated trees. The linking is the part most likely to be subtly wrong, and only looking at it catches that.
