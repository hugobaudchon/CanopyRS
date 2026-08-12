#!/usr/bin/env python3
"""Evaluate overlap between LiDAR crown instances (ground truth) and RGB crown
instances (predictions), seen from above.

Implements the methodology in ``canopyrs3d/plan.md``. Read that first — it
explains *why* each choice is made and records the probe results the defaults
are based on.

The short version:

* LiDAR is the ground truth, RGB (CanopyRS) is the prediction.
* The LiDAR annotation is **deliberately incomplete** — uncertain trees were
  skipped — so RGB masks with no LiDAR counterpart are reported but never
  penalised. Every metric here is recall-oriented and ground-truth-anchored.
  There is deliberately no precision, F1 or panoptic PQ/RQ.
* Instances are linked **many-to-many** (connected components of a bipartite
  graph), because one LiDAR crown is often split across several RGB masks and
  one RGB mask often spans several LiDAR crowns.

Usage::

    python scripts/eval_crown_overlap.py \\
        --gt-dir data/lidar_masks --pred data/rgb_masks/*.gpkg --out-dir output
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

log = logging.getLogger("eval_crown_overlap")

# --- Defaults (see plan.md §5, §7, §9) --------------------------------------

MIN_AREA_M2 = 1.0          # drop degenerate crowns/masks below this
AOI_MIN_OVERLAP = 0.4      # a prediction must have this share of its area in the AOI
EDGE_BUFFER_M = 1.0        # GT crowns this close to the AOI boundary are flagged
TAU_LINK = 0.5             # inter / min(area_gt, area_pred) threshold for an edge
TAU_FLOOR = 0.05           # IoU floor guarding extreme scale-ratio containment
MAX_CLUSTER_SIZE = 8       # chaining guard trips above this many members
TAU_COV = (0.5, 0.75, 0.9)     # "sufficient coverage" thresholds
RECALL_IOU = (0.25, 0.5, 0.75)  # Recall@IoU thresholds
TAU_LINK_SWEEP = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)

EPS = 1e-12


# =============================================================================
# Small helpers
# =============================================================================


class DSU:
    """Union-find over ``n`` nodes, for building connected components."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > EPS else 0.0


def _mean(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def _weighted_mean(values, weights) -> float:
    values, weights = np.asarray(list(values), float), np.asarray(list(weights), float)
    if values.size == 0 or weights.sum() <= EPS:
        return 0.0
    return float((values * weights).sum() / weights.sum())


def classify_cluster(n_gt: int, n_pred: int) -> str:
    """Cluster taxonomy from plan.md §7."""
    if n_gt == 0:
        return "unmatched_pred"
    if n_pred == 0:
        return "missed"
    if n_gt == 1 and n_pred == 1:
        return "one_to_one"
    if n_gt == 1:
        return "split"
    if n_pred == 1:
        return "merge"
    return "tangled"


# =============================================================================
# Preprocessing (plan.md §5)
# =============================================================================


@dataclass
class PrepStats:
    """Counts of what preprocessing removed, so nothing vanishes silently."""

    gt_total: int = 0
    gt_by_hull_status: dict = field(default_factory=dict)
    gt_null_geom: int = 0
    gt_invalid_or_empty: int = 0
    gt_below_min_area: int = 0
    gt_kept: int = 0
    pred_total: int = 0
    pred_outside_aoi: int = 0
    pred_low_aoi_overlap: int = 0
    pred_empty_after_clip: int = 0
    pred_below_min_area: int = 0
    pred_kept: int = 0


def _clean(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, int]:
    """make_valid + drop empties. MultiPolygons are kept intact on purpose: a
    crown legitimately splits into parts under occlusion (plan.md §5.3)."""
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.make_valid()
    keep = ~(gdf.geometry.is_empty | gdf.geometry.isna())
    return gdf[keep].copy(), int((~keep).sum())


def load_gt(path: Path, min_area: float, stats: PrepStats) -> gpd.GeoDataFrame:
    """Load LiDAR crowns. Rows with ``hull_status != 'ok'`` carry NULL geometry —
    they are trees the LiDAR annotated but could not project to a top view, so
    they are out of scope for a top-view comparison."""
    gdf = gpd.read_file(path)
    stats.gt_total = len(gdf)
    if "hull_status" in gdf.columns:
        stats.gt_by_hull_status = {
            str(k): int(v) for k, v in gdf["hull_status"].value_counts().items()
        }

    null_mask = gdf.geometry.isna()
    stats.gt_null_geom = int(null_mask.sum())
    gdf = gdf[~null_mask].copy()

    gdf, stats.gt_invalid_or_empty = _clean(gdf)

    small = gdf.geometry.area < min_area
    stats.gt_below_min_area = int(small.sum())
    gdf = gdf[~small].reset_index(drop=True)

    stats.gt_kept = len(gdf)
    return gdf


def load_preds(path: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


def build_aoi(gt: gpd.GeoDataFrame):
    """AOI = convex hull of the GT crown union (plan.md §5.4). GT lies inside by
    construction, so only predictions are ever clipped."""
    return gt.geometry.union_all().convex_hull


def restrict_preds(
    preds: gpd.GeoDataFrame,
    aoi,
    min_overlap: float,
    min_area: float,
    stats: PrepStats,
) -> gpd.GeoDataFrame:
    """Cut the site-wide prediction layer down to the plot, dropping masks that
    only graze the AOI rather than clipping them into slivers."""
    stats.pred_total = len(preds)

    idx = preds.sindex.query(aoi, predicate="intersects")
    preds = preds.iloc[sorted(idx)].copy()
    stats.pred_outside_aoi = stats.pred_total - len(preds)

    preds, _ = _clean(preds)

    frac = preds.geometry.intersection(aoi).area / preds.geometry.area.clip(lower=EPS)
    low = frac < min_overlap
    stats.pred_low_aoi_overlap = int(low.sum())
    preds = preds[~low].copy()

    preds["geometry"] = preds.geometry.intersection(aoi)
    preds, stats.pred_empty_after_clip = _clean(preds)

    small = preds.geometry.area < min_area
    stats.pred_below_min_area = int(small.sum())
    preds = preds[~small].reset_index(drop=True)

    stats.pred_kept = len(preds)
    return preds


def flag_edge_gt(gt: gpd.GeoDataFrame, aoi, buffer_m: float) -> np.ndarray:
    """Crowns near the AOI boundary are truncated against an untruncated
    neighbourhood, which depresses their IoU. Flag, don't drop."""
    return (gt.geometry.distance(aoi.exterior) <= buffer_m).to_numpy()


# =============================================================================
# Sparse overlap matrix (plan.md §6)
# =============================================================================


def overlap_pairs(gt: gpd.GeoDataFrame, preds: gpd.GeoDataFrame) -> pd.DataFrame:
    """Every intersecting (GT, prediction) pair with its overlap measures.

    Columns: ``i, j, inter, iou, cov, pur, ovl`` where ``i`` indexes ``gt`` and
    ``j`` indexes ``preds`` positionally.
    """
    rows = []
    if len(gt) == 0 or len(preds) == 0:
        return pd.DataFrame(columns=["i", "j", "inter", "iou", "cov", "pur", "ovl"])

    pred_geoms = preds.geometry.to_numpy()
    pred_areas = preds.geometry.area.to_numpy()

    for i, (geom, area_g) in enumerate(zip(gt.geometry, gt.geometry.area)):
        for j in preds.sindex.query(geom, predicate="intersects"):
            inter = geom.intersection(pred_geoms[j]).area
            if inter <= EPS:
                continue
            area_p = pred_areas[j]
            rows.append(
                {
                    "i": int(i),
                    "j": int(j),
                    "inter": float(inter),
                    "iou": _safe_div(inter, area_g + area_p - inter),
                    "cov": _safe_div(inter, area_g),
                    "pur": _safe_div(inter, area_p),
                    "ovl": _safe_div(inter, min(area_g, area_p)),
                }
            )
    return pd.DataFrame(rows, columns=["i", "j", "inter", "iou", "cov", "pur", "ovl"])


# =============================================================================
# Many-to-many linking (plan.md §7)
# =============================================================================


@dataclass
class Cluster:
    cluster_id: int
    gt_idx: list
    pred_idx: list

    @property
    def kind(self) -> str:
        return classify_cluster(len(self.gt_idx), len(self.pred_idx))


def _components(edges, n_gt: int, n_pred: int) -> list[Cluster]:
    dsu = DSU(n_gt + n_pred)
    for i, j in edges:
        dsu.union(i, n_gt + j)

    groups: dict[int, tuple[list, list]] = defaultdict(lambda: ([], []))
    for i in range(n_gt):
        groups[dsu.find(i)][0].append(i)
    for j in range(n_pred):
        groups[dsu.find(n_gt + j)][1].append(j)

    # Deterministic ordering: GT-anchored clusters first, by smallest GT index.
    ordered = sorted(
        groups.values(),
        key=lambda gp: (0, gp[0][0]) if gp[0] else (1, gp[1][0]),
    )
    return [Cluster(k, gt_i, pr_j) for k, (gt_i, pr_j) in enumerate(ordered)]


def link_instances(
    pairs: pd.DataFrame,
    n_gt: int,
    n_pred: int,
    tau_link: float = TAU_LINK,
    tau_floor: float = TAU_FLOOR,
    max_cluster_size: int = MAX_CLUSTER_SIZE,
) -> tuple[list[Cluster], dict]:
    """Link GT crowns to predictions and group them into clusters.

    An edge is drawn iff ``ovl >= tau_link AND iou >= tau_floor``. Min-normalised
    overlap is what makes splits *and* merges link correctly; the IoU floor
    guards the one weakness of that normalisation (a tiny crown swallowed by a
    huge mask). See plan.md §7 for the full argument.
    """
    info: dict = {"tau_link": tau_link, "tau_floor": tau_floor}

    if pairs.empty:
        info.update(n_edges=0, n_edges_removed_by_floor=0, tightened_clusters=[])
        return _components([], n_gt, n_pred), info

    passes_ovl = pairs["ovl"] >= tau_link
    passes_both = passes_ovl & (pairs["iou"] >= tau_floor)
    info["n_edges"] = int(passes_both.sum())
    info["n_edges_removed_by_floor"] = int((passes_ovl & ~passes_both).sum())

    edges = list(zip(pairs.loc[passes_both, "i"], pairs.loc[passes_both, "j"]))
    clusters = _components(edges, n_gt, n_pred)

    # Chaining guard: measured never to trip on this data, but a silent giant
    # component would quietly dominate the headline metric (plan.md §7).
    clusters, tightened = _apply_chaining_guard(
        clusters, pairs, tau_link, tau_floor, max_cluster_size
    )
    info["tightened_clusters"] = tightened

    sizes = [len(c.gt_idx) + len(c.pred_idx) for c in clusters]
    info["max_cluster_size"] = max(sizes) if sizes else 0
    info["cluster_size_histogram"] = {
        str(k): int(v) for k, v in sorted(Counter(sizes).items())
    }
    return clusters, info


def _apply_chaining_guard(
    clusters: list[Cluster],
    pairs: pd.DataFrame,
    tau_link: float,
    tau_floor: float,
    max_cluster_size: int,
) -> tuple[list[Cluster], list]:
    """Re-link oversized components at a progressively higher tau until they
    break up (or 0.9 is reached), rather than letting one blob swallow the plot."""
    tightened, result = [], []
    for cluster in clusters:
        size = len(cluster.gt_idx) + len(cluster.pred_idx)
        if size <= max_cluster_size:
            result.append(cluster)
            continue

        gt_set, pred_set = set(cluster.gt_idx), set(cluster.pred_idx)
        local = pairs[pairs["i"].isin(gt_set) & pairs["j"].isin(pred_set)]
        tau, sub = tau_link, [cluster]
        while tau < 0.9 - EPS and max(len(c.gt_idx) + len(c.pred_idx) for c in sub) > max_cluster_size:
            tau = round(min(tau + 0.1, 0.9), 10)
            keep = (local["ovl"] >= tau) & (local["iou"] >= tau_floor)
            sub = _relabel_within(
                list(zip(local.loc[keep, "i"], local.loc[keep, "j"])),
                sorted(gt_set),
                sorted(pred_set),
            )
        tightened.append(
            {"original_size": size, "final_tau": tau, "n_subclusters": len(sub)}
        )
        result.extend(sub)

    for new_id, cluster in enumerate(result):
        cluster.cluster_id = new_id
    return result, tightened


def _relabel_within(edges, gt_ids: list, pred_ids: list) -> list[Cluster]:
    """Run connected components over a subset, mapping global<->local indices."""
    gt_pos = {g: k for k, g in enumerate(gt_ids)}
    pred_pos = {p: k for k, p in enumerate(pred_ids)}
    local_edges = [(gt_pos[i], pred_pos[j]) for i, j in edges]
    return [
        Cluster(
            0,
            [gt_ids[k] for k in c.gt_idx],
            [pred_ids[k] for k in c.pred_idx],
        )
        for c in _components(local_edges, len(gt_ids), len(pred_ids))
    ]


# =============================================================================
# One-to-one assignment (plan.md §9, secondary metric)
# =============================================================================


def assign_one_to_one(pairs: pd.DataFrame, clusters: list[Cluster]) -> dict:
    """Optimal 1-1 assignment maximising total IoU, solved per connected
    component so the Hungarian matrices stay tiny. Returns ``{gt_idx: (pred_idx,
    iou)}`` for assigned crowns only."""
    if pairs.empty:
        return {}
    iou_lookup = {(int(r.i), int(r.j)): float(r.iou) for r in pairs.itertuples()}

    assignment = {}
    for cluster in clusters:
        if not cluster.gt_idx or not cluster.pred_idx:
            continue
        cost = np.zeros((len(cluster.gt_idx), len(cluster.pred_idx)))
        for a, i in enumerate(cluster.gt_idx):
            for b, j in enumerate(cluster.pred_idx):
                cost[a, b] = -iou_lookup.get((i, j), 0.0)
        for a, b in zip(*linear_sum_assignment(cost)):
            iou = -cost[a, b]
            if iou > EPS:
                assignment[cluster.gt_idx[a]] = (cluster.pred_idx[b], iou)
    return assignment


# =============================================================================
# Per-plot evaluation
# =============================================================================


@dataclass
class PlotResult:
    plot: str
    per_gt: pd.DataFrame
    per_cluster: pd.DataFrame
    unmatched_preds: pd.DataFrame
    metrics: dict
    gt: gpd.GeoDataFrame
    preds: gpd.GeoDataFrame
    prep: PrepStats
    link_info: dict
    pred_cluster_id: list = field(default_factory=list)


def evaluate_plot(
    plot: str,
    gt: gpd.GeoDataFrame,
    preds: gpd.GeoDataFrame,
    pairs: pd.DataFrame,
    edge_flags: np.ndarray,
    args,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, list[Cluster], dict]:
    clusters, link_info = link_instances(
        pairs, len(gt), len(preds), args.tau_link, args.tau_floor, args.max_cluster_size
    )
    assignment = assign_one_to_one(pairs, clusters)

    gt_geom = gt.geometry.to_numpy()
    gt_area = gt.geometry.area.to_numpy()
    pred_geom = preds.geometry.to_numpy()
    pred_area = preds.geometry.area.to_numpy()

    cluster_of_gt = {i: c.cluster_id for c in clusters for i in c.gt_idx}
    cluster_of_pred = {j: c.cluster_id for c in clusters for j in c.pred_idx}
    kind_of_cluster = {c.cluster_id: c.kind for c in clusters}
    linked_preds = {i: list(c.pred_idx) for c in clusters for i in c.gt_idx}

    # --- per-cluster ---------------------------------------------------------
    cluster_rows = []
    for c in clusters:
        g_union = gpd.GeoSeries(gt_geom[c.gt_idx]).union_all() if c.gt_idx else None
        p_union = gpd.GeoSeries(pred_geom[c.pred_idx]).union_all() if c.pred_idx else None
        g_area = float(gt_area[c.gt_idx].sum()) if c.gt_idx else 0.0
        p_area = float(pred_area[c.pred_idx].sum()) if c.pred_idx else 0.0

        if g_union is not None and p_union is not None:
            inter = g_union.intersection(p_union).area
            union = g_union.union(p_union).area
        else:
            inter = union = 0.0

        cluster_rows.append(
            {
                "plot": plot,
                "cluster_id": c.cluster_id,
                "cluster_type": c.kind,
                "n_gt": len(c.gt_idx),
                "n_pred": len(c.pred_idx),
                "gt_area": g_area,
                "pred_area": p_area,
                "inter_area": float(inter),
                "union_area": float(union) if union else g_area + p_area,
                "iou_k": _safe_div(inter, union),
                "cov_k": _safe_div(inter, g_area),
                # Cardinality agreement: 1.0 when the cluster has as many masks
                # as crowns, 1/3 for a 1-to-3 split or a 3-to-1 merge. Unioning
                # fragments would otherwise hide over-segmentation entirely --
                # three masks reconstructing one crown score IoU_k = 1.0.
                # Penalising these masks does NOT violate the rule that
                # unmatched predictions go unpunished: they demonstrably overlap
                # an annotated crown, so they are fragments, not discoveries.
                "card_k": (
                    min(len(c.gt_idx), len(c.pred_idx))
                    / max(len(c.gt_idx), len(c.pred_idx))
                    if c.gt_idx and c.pred_idx else 0.0
                ),
            }
        )
    per_cluster = pd.DataFrame(cluster_rows)
    if len(per_cluster):
        per_cluster["iou_k_penalised"] = per_cluster["iou_k"] * per_cluster["card_k"]

    # --- per-GT --------------------------------------------------------------
    best_iou = defaultdict(float)
    best_cov = defaultdict(float)
    for r in pairs.itertuples():
        best_iou[int(r.i)] = max(best_iou[int(r.i)], float(r.iou))
        best_cov[int(r.i)] = max(best_cov[int(r.i)], float(r.cov))

    gt_rows = []
    for i in range(len(gt)):
        linked = linked_preds.get(i, [])
        if linked:
            union_linked = gpd.GeoSeries(pred_geom[linked]).union_all()
            cov = _safe_div(gt_geom[i].intersection(union_linked).area, gt_area[i])
        else:
            cov = 0.0
        row = {
            "plot": plot,
            "instance_id": _attr(gt, i, "instance_id", i),
            "area_m2": float(gt_area[i]),
            "height_m": _attr(gt, i, "height_m", np.nan),
            "cluster_id": cluster_of_gt[i],
            "cluster_type": kind_of_cluster[cluster_of_gt[i]],
            "n_linked_preds": len(linked),
            "cov": cov,
            "cov_best": best_cov.get(i, 0.0),
            "best_iou": best_iou.get(i, 0.0),
            "iou_1to1": assignment.get(i, (None, 0.0))[1],
            "edge_gt": bool(edge_flags[i]),
        }
        for t in TAU_COV:
            row[f"sufficient@{t}"] = bool(cov >= t)
        gt_rows.append(row)
    per_gt = pd.DataFrame(gt_rows)

    # --- unmatched predictions (reported, never penalised) -------------------
    unmatched_idx = [
        j
        for j in range(len(preds))
        if kind_of_cluster[cluster_of_pred[j]] == "unmatched_pred"
    ]
    unmatched = pd.DataFrame(
        {
            "plot": plot,
            "canopyrs_object_id": [
                _attr(preds, j, "canopyrs_object_id", j) for j in unmatched_idx
            ],
            "area_m2": [float(pred_area[j]) for j in unmatched_idx],
            "aggregator_score": [
                _attr(preds, j, "aggregator_score", np.nan) for j in unmatched_idx
            ],
        }
    )

    linked_pred_idx = [j for j in range(len(preds)) if j not in set(unmatched_idx)]
    metrics = compute_metrics(
        per_gt, per_cluster, gt, preds, unmatched, linked_pred_idx
    )
    return per_gt, per_cluster, unmatched, metrics, clusters, link_info


def _attr(gdf: gpd.GeoDataFrame, pos: int, column: str, default):
    if column not in gdf.columns:
        return default
    value = gdf[column].iloc[pos]
    return default if pd.isna(value) else value


# =============================================================================
# Metrics (plan.md §9)
# =============================================================================


def compute_metrics(
    per_gt: pd.DataFrame,
    per_cluster: pd.DataFrame,
    gt: gpd.GeoDataFrame,
    preds: gpd.GeoDataFrame,
    unmatched: pd.DataFrame,
    linked_pred_idx: list | None = None,
) -> dict:
    anchored = per_cluster[per_cluster["n_gt"] >= 1]

    if len(gt) and len(preds):
        g_union = gt.geometry.union_all()
        p_union = preds.geometry.union_all()
        global_inter = g_union.intersection(p_union).area
        global_union = g_union.union(p_union).area
        global_gt_area = g_union.area

        # IoU_global unions *every* prediction, so masks over trees the LiDAR
        # never annotated inflate its denominator — the one place in this script
        # where an unmatched mask costs anything. IoU_global_linked repeats the
        # calculation over linked masks only and is the GT-anchored figure.
        if linked_pred_idx:
            l_union = preds.geometry.iloc[linked_pred_idx].union_all()
            linked_inter = g_union.intersection(l_union).area
            linked_union = g_union.union(l_union).area
        else:
            linked_inter, linked_union = 0.0, global_gt_area
    else:
        global_inter = global_union = 0.0
        global_gt_area = float(gt.geometry.area.sum()) if len(gt) else 0.0
        linked_inter, linked_union = 0.0, global_gt_area

    m = {
        # --- headline --------------------------------------------------------
        "mIoU_cluster": _mean(anchored["iou_k"]),
        "mCov_cluster": _mean(anchored["cov_k"]),
        # --- fragmentation-penalised (see card_k above) ----------------------
        "mIoU_cluster_penalised": _mean(anchored["iou_k_penalised"]),
        # --- area-weighted ---------------------------------------------------
        "mIoU_cluster_area": _weighted_mean(anchored["iou_k"], anchored["gt_area"]),
        "mCov_cluster_area": _weighted_mean(anchored["cov_k"], anchored["gt_area"]),
        # --- secondary, GT-averaged -----------------------------------------
        "mIoU_1to1": _mean(per_gt["iou_1to1"]),
        "mIoU_best": _mean(per_gt["best_iou"]),
        # --- instance-agnostic ----------------------------------------------
        "IoU_global": _safe_div(global_inter, global_union),
        "IoU_global_linked": _safe_div(linked_inter, linked_union),
        "Cov_global": _safe_div(global_inter, global_gt_area),
        # --- counts ----------------------------------------------------------
        "n_gt": int(len(per_gt)),
        "n_pred": int(len(preds)),
        "n_clusters_gt_anchored": int(len(anchored)),
        "n_unmatched_pred": int(len(unmatched)),
        "unmatched_pred_area_m2": float(unmatched["area_m2"].sum()) if len(unmatched) else 0.0,
    }

    for t in RECALL_IOU:
        m[f"Recall@IoU{t}"] = _mean(per_gt["iou_1to1"] >= t)
    for t in TAU_COV:
        m[f"frac_sufficient@{t}"] = _mean(per_gt[f"sufficient@{t}"])

    m["mCov_gt"] = _mean(per_gt["cov"])
    m["mCov_gt_best_single"] = _mean(per_gt["cov_best"])

    # Fragmentation
    matched = per_gt[per_gt["n_linked_preds"] > 0]
    m["split_factor"] = _mean(matched["n_linked_preds"]) if len(matched) else 0.0
    linked_clusters = anchored[anchored["n_pred"] > 0]
    m["merge_factor"] = (
        _safe_div(linked_clusters["n_gt"].sum(), linked_clusters["n_pred"].sum())
        if len(linked_clusters)
        else 0.0
    )
    m["cluster_types"] = {
        str(k): int(v) for k, v in per_cluster["cluster_type"].value_counts().items()
    }

    # Metrics excluding crowns truncated by the plot boundary
    interior = per_gt[~per_gt["edge_gt"]]
    m["n_gt_edge"] = int(per_gt["edge_gt"].sum())
    m["mIoU_1to1_interior"] = _mean(interior["iou_1to1"])
    m["mCov_gt_interior"] = _mean(interior["cov"])

    # Kept so pooled figures can be recomputed without geometry
    m["_global_inter"] = float(global_inter)
    m["_global_union"] = float(global_union)
    m["_global_gt_area"] = float(global_gt_area)
    m["_linked_inter"] = float(linked_inter)
    m["_linked_union"] = float(linked_union)
    return m


def pool_metrics(results: list[PlotResult]) -> dict:
    per_gt = pd.concat([r.per_gt for r in results], ignore_index=True)
    per_cluster = pd.concat([r.per_cluster for r in results], ignore_index=True)
    unmatched = pd.concat([r.unmatched_preds for r in results], ignore_index=True)
    anchored = per_cluster[per_cluster["n_gt"] >= 1]

    # Plots are spatially disjoint, so global overlap pools additively.
    inter = sum(r.metrics["_global_inter"] for r in results)
    union = sum(r.metrics["_global_union"] for r in results)
    gt_area = sum(r.metrics["_global_gt_area"] for r in results)
    l_inter = sum(r.metrics["_linked_inter"] for r in results)
    l_union = sum(r.metrics["_linked_union"] for r in results)

    m = {
        "mIoU_cluster": _mean(anchored["iou_k"]),
        "mCov_cluster": _mean(anchored["cov_k"]),
        "mIoU_cluster_penalised": _mean(anchored["iou_k_penalised"]),
        "mIoU_cluster_area": _weighted_mean(anchored["iou_k"], anchored["gt_area"]),
        "mCov_cluster_area": _weighted_mean(anchored["cov_k"], anchored["gt_area"]),
        "mIoU_1to1": _mean(per_gt["iou_1to1"]),
        "mIoU_best": _mean(per_gt["best_iou"]),
        "IoU_global": _safe_div(inter, union),
        "IoU_global_linked": _safe_div(l_inter, l_union),
        "Cov_global": _safe_div(inter, gt_area),
        "n_gt": int(len(per_gt)),
        "n_pred": int(sum(r.metrics["n_pred"] for r in results)),
        "n_clusters_gt_anchored": int(len(anchored)),
        "n_unmatched_pred": int(len(unmatched)),
        "unmatched_pred_area_m2": float(unmatched["area_m2"].sum()) if len(unmatched) else 0.0,
        "mCov_gt": _mean(per_gt["cov"]),
        "mCov_gt_best_single": _mean(per_gt["cov_best"]),
        "cluster_types": {
            str(k): int(v) for k, v in per_cluster["cluster_type"].value_counts().items()
        },
    }
    for t in RECALL_IOU:
        m[f"Recall@IoU{t}"] = _mean(per_gt["iou_1to1"] >= t)
    for t in TAU_COV:
        m[f"frac_sufficient@{t}"] = _mean(per_gt[f"sufficient@{t}"])
    return m


# =============================================================================
# Invariants (plan.md §13)
# =============================================================================


def check_invariants(result: PlotResult) -> list[str]:
    """Assert what must hold. Returns violations rather than raising, so one bad
    plot does not hide results for the others."""
    problems = []
    gt, per_gt, per_cluster = result.gt, result.per_gt, result.per_cluster
    m = result.metrics

    if len(gt):
        summed = float(gt.geometry.area.sum())
        union = float(gt.geometry.union_all().area)
        if abs(summed - union) > 1e-4 * max(union, 1.0):
            problems.append(
                f"GT crowns are not disjoint: Σareas={summed:.4f} vs union={union:.4f}"
            )

    if len(per_gt) != len(gt):
        problems.append(f"per_gt has {len(per_gt)} rows for {len(gt)} GT crowns")
    if per_gt["cluster_id"].isna().any():
        problems.append("some GT crowns have no cluster")

    cluster_gt_area = float(per_cluster["gt_area"].sum())
    total_gt_area = float(gt.geometry.area.sum()) if len(gt) else 0.0
    if abs(cluster_gt_area - total_gt_area) > 1e-4 * max(total_gt_area, 1.0):
        problems.append(
            f"cluster GT area {cluster_gt_area:.4f} != total GT area {total_gt_area:.4f}"
        )

    if m["mIoU_1to1"] > m["mIoU_best"] + 1e-9:
        problems.append(
            f"mIoU_1to1 ({m['mIoU_1to1']:.6f}) > mIoU_best ({m['mIoU_best']:.6f})"
        )

    # Per cluster, union >= GT area, so IoU can never exceed coverage.
    bad = per_cluster[per_cluster["iou_k"] > per_cluster["cov_k"] + 1e-9]
    if len(bad):
        problems.append(f"{len(bad)} clusters have iou_k > cov_k")

    return problems


# =============================================================================
# Sweeps (plan.md §11)
# =============================================================================


def sweep_tau_link(results: list[PlotResult], args) -> pd.DataFrame:
    """Recompute the headline metric across tau_link. The plan requires this be
    published alongside the point estimate — plot 210's mIoU moves 0.10 across
    the range, so a single number would overstate its precision."""
    rows = []
    for r in results:
        pairs = overlap_pairs(r.gt, r.preds)
        gt_geom, pred_geom = r.gt.geometry.to_numpy(), r.preds.geometry.to_numpy()
        gt_area = r.gt.geometry.area.to_numpy()

        for tau in TAU_LINK_SWEEP:
            clusters, info = link_instances(
                pairs, len(r.gt), len(r.preds), tau, args.tau_floor, args.max_cluster_size
            )
            ious, covs, types = [], [], Counter()
            for c in clusters:
                types[c.kind] += 1
                if not c.gt_idx:
                    continue
                g_union = gpd.GeoSeries(gt_geom[c.gt_idx]).union_all()
                g_area = float(gt_area[c.gt_idx].sum())
                if c.pred_idx:
                    p_union = gpd.GeoSeries(pred_geom[c.pred_idx]).union_all()
                    inter = g_union.intersection(p_union).area
                    ious.append(_safe_div(inter, g_union.union(p_union).area))
                    covs.append(_safe_div(inter, g_area))
                else:
                    ious.append(0.0)
                    covs.append(0.0)
            rows.append(
                {
                    "plot": r.plot,
                    "tau_link": tau,
                    "mIoU_cluster": _mean(ious),
                    "mCov_cluster": _mean(covs),
                    "n_edges": info["n_edges"],
                    "max_cluster_size": info["max_cluster_size"],
                    **{f"n_{k}": types[k] for k in
                       ("one_to_one", "split", "merge", "tangled", "missed", "unmatched_pred")},
                }
            )
    return pd.DataFrame(rows)


def sweep_tau_cov(results: list[PlotResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for tau in np.round(np.arange(0.0, 1.001, 0.05), 2):
            rows.append(
                {
                    "plot": r.plot,
                    "tau_cov": float(tau),
                    "frac_sufficient": _mean(r.per_gt["cov"] >= tau),
                }
            )
    return pd.DataFrame(rows)


def make_figure(link_df: pd.DataFrame, cov_df: pd.DataFrame, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib unavailable — skipping %s", path.name)
        return False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for plot, sub in link_df.groupby("plot"):
        axes[0].plot(sub["tau_link"], sub["mIoU_cluster"], marker="o", label=f"plot {plot}")
        axes[1].plot(sub["tau_link"], sub["mCov_cluster"], marker="o", label=f"plot {plot}")
    for plot, sub in cov_df.groupby("plot"):
        axes[2].plot(sub["tau_cov"], sub["frac_sufficient"], marker=".", label=f"plot {plot}")

    axes[0].set(xlabel=r"$\tau_{link}$", ylabel="mIoU_cluster", title="Headline mIoU vs linking threshold")
    axes[1].set(xlabel=r"$\tau_{link}$", ylabel="mCov_cluster", title="Cluster coverage vs linking threshold")
    axes[2].set(xlabel=r"$\tau_{cov}$", ylabel="fraction of GT crowns", title="Coverage sufficiency")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# =============================================================================
# Output
# =============================================================================


def write_links_gpkg(results: list[PlotResult], path: Path) -> None:
    """GT and predictions with cluster membership attached, for visual QA in
    QGIS. plan.md calls this the single most valuable output."""
    gt_parts, pred_parts = [], []
    for r in results:
        gt = r.gt[["geometry"]].copy()
        gt["plot"] = r.plot
        for col in ("instance_id", "cluster_id", "cluster_type", "cov", "best_iou",
                    "iou_1to1", "n_linked_preds", "edge_gt"):
            gt[col] = r.per_gt[col].to_numpy()
        gt_parts.append(gt)

        kind = dict(zip(r.per_cluster["cluster_id"], r.per_cluster["cluster_type"]))
        pred = r.preds[["geometry"]].copy()
        pred["plot"] = r.plot
        pred["canopyrs_object_id"] = [
            _attr(r.preds, j, "canopyrs_object_id", j) for j in range(len(r.preds))
        ]
        pred["aggregator_score"] = [
            _attr(r.preds, j, "aggregator_score", np.nan) for j in range(len(r.preds))
        ]
        pred["cluster_id"] = r.pred_cluster_id
        pred["cluster_type"] = [kind[c] for c in r.pred_cluster_id]
        pred_parts.append(pred)

    if path.exists():
        path.unlink()
    gpd.GeoDataFrame(pd.concat(gt_parts, ignore_index=True), crs=results[0].gt.crs).to_file(
        path, layer="gt_crowns", driver="GPKG"
    )
    gpd.GeoDataFrame(pd.concat(pred_parts, ignore_index=True), crs=results[0].preds.crs).to_file(
        path, layer="rgb_masks", driver="GPKG"
    )


def print_report(results: list[PlotResult], pooled: dict) -> None:
    def fmt(m: dict, label: str) -> None:
        print(f"\n  {label}")
        print(f"    HEADLINE  mIoU_cluster        {m['mIoU_cluster']:.3f}"
              f"      (area-weighted {m['mIoU_cluster_area']:.3f})")
        print(f"    COMPANION mCov_cluster        {m['mCov_cluster']:.3f}"
              f"      (area-weighted {m['mCov_cluster_area']:.3f})")
        print(f"    PESSIMIST mIoU_cluster_penal. {m['mIoU_cluster_penalised']:.3f}"
              f"      (splits/merges penalised)")
        print(f"              mIoU_1to1           {m['mIoU_1to1']:.3f}")
        print(f"              mIoU_best           {m['mIoU_best']:.3f}")
        print(f"              IoU_global          {m['IoU_global']:.3f}"
              f"      Cov_global {m['Cov_global']:.3f}")
        print(f"              IoU_global_linked   {m['IoU_global_linked']:.3f}"
              f"      (excludes unmatched masks)")
        recalls = "  ".join(f"@{t}={m[f'Recall@IoU{t}']:.2f}" for t in RECALL_IOU)
        print(f"              Recall              {recalls}")
        suff = "  ".join(f"@{t}={m[f'frac_sufficient@{t}']:.2f}" for t in TAU_COV)
        print(f"              Sufficient coverage {suff}")
        print(f"              clusters            {m['cluster_types']}")
        print(f"              unmatched preds     {m['n_unmatched_pred']}"
              f" ({m['unmatched_pred_area_m2']:.0f} m2) — reported, not penalised")

    print("\n" + "=" * 78)
    print("LiDAR (GT) vs RGB (pred) crown overlap")
    print("=" * 78)
    for r in results:
        p = r.prep
        print(f"\nPLOT {r.plot}")
        print(f"  GT     {p.gt_total} features -> {p.gt_kept} usable "
              f"(null geom {p.gt_null_geom}, below min area {p.gt_below_min_area})")
        print(f"  PRED   {p.pred_total} features -> {p.pred_kept} in AOI "
              f"(low AOI overlap {p.pred_low_aoi_overlap}, below min area {p.pred_below_min_area})")
        print(f"  LINK   {r.link_info['n_edges']} edges, max cluster size "
              f"{r.link_info['max_cluster_size']}, "
              f"{r.link_info['n_edges_removed_by_floor']} removed by tau_floor")
        fmt(r.metrics, f"metrics (plot {r.plot})")
    if len(results) > 1:
        print("\n" + "-" * 78)
        fmt(pooled, "POOLED")
    print()


# =============================================================================
# CLI
# =============================================================================


def parse_plot_name(path: Path) -> str:
    m = re.search(r"plot[_-]?(\w+?)_", path.stem)
    return m.group(1) if m else path.stem


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--gt-dir", type=Path, default=here / "data" / "lidar_masks",
                   help="directory of LiDAR ground-truth GeoPackages")
    p.add_argument("--gt", type=Path, nargs="*", default=None,
                   help="explicit GT files (overrides --gt-dir)")
    p.add_argument("--pred", type=Path, nargs="+", default=None,
                   help="RGB prediction GeoPackage(s)")
    p.add_argument("--out-dir", type=Path, default=here / "output")
    p.add_argument("--min-area", type=float, default=MIN_AREA_M2)
    p.add_argument("--aoi-min-overlap", type=float, default=AOI_MIN_OVERLAP)
    p.add_argument("--edge-buffer", type=float, default=EDGE_BUFFER_M)
    p.add_argument("--tau-link", type=float, default=TAU_LINK)
    p.add_argument("--tau-floor", type=float, default=TAU_FLOOR)
    p.add_argument("--max-cluster-size", type=int, default=MAX_CLUSTER_SIZE)
    p.add_argument("--no-sweeps", action="store_true", help="skip threshold sweeps and figure")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    gt_paths = sorted(args.gt) if args.gt else sorted(args.gt_dir.glob("*.gpkg"))
    if not gt_paths:
        log.error("no ground-truth GeoPackages found (looked in %s)", args.gt_dir)
        return 2

    if args.pred:
        pred_paths = sorted(args.pred)
    else:
        default_pred_dir = Path(__file__).resolve().parent.parent / "data" / "rgb_masks"
        pred_paths = sorted(default_pred_dir.glob("*.gpkg"))
    if not pred_paths:
        log.error("no prediction GeoPackages found")
        return 2

    log.info("predictions: %s", ", ".join(p.name for p in pred_paths))
    preds_all = pd.concat([load_preds(p) for p in pred_paths], ignore_index=True)
    preds_all = gpd.GeoDataFrame(preds_all, crs=gpd.read_file(pred_paths[0]).crs)
    log.info("loaded %d prediction polygons", len(preds_all))

    results: list[PlotResult] = []
    for gt_path in gt_paths:
        plot = parse_plot_name(gt_path)
        log.info("--- plot %s (%s) ---", plot, gt_path.name)

        prep = PrepStats()
        gt = load_gt(gt_path, args.min_area, prep)
        if gt.empty:
            log.warning("plot %s has no usable GT crowns — skipped", plot)
            continue

        preds = preds_all
        if preds.crs != gt.crs:
            log.info("reprojecting predictions %s -> %s", preds.crs, gt.crs)
            preds = preds.to_crs(gt.crs)

        aoi = build_aoi(gt)
        preds = restrict_preds(preds, aoi, args.aoi_min_overlap, args.min_area, prep)
        edge_flags = flag_edge_gt(gt, aoi, args.edge_buffer)

        pairs = overlap_pairs(gt, preds)
        log.info("%d GT, %d predictions, %d candidate pairs", len(gt), len(preds), len(pairs))

        per_gt, per_cluster, unmatched, metrics, clusters, link_info = evaluate_plot(
            plot, gt, preds, pairs, edge_flags, args
        )
        metrics["n_candidate_pairs"] = int(len(pairs))
        metrics["n_link_edges"] = int(link_info["n_edges"])

        result = PlotResult(plot, per_gt, per_cluster, unmatched, metrics,
                            gt, preds, prep, link_info)
        # positional cluster id per prediction, needed by write_links_gpkg
        pred_cluster = {j: c.cluster_id for c in clusters for j in c.pred_idx}
        result.pred_cluster_id = [pred_cluster[j] for j in range(len(preds))]

        for problem in check_invariants(result):
            log.error("INVARIANT VIOLATED (plot %s): %s", plot, problem)
        results.append(result)

    if not results:
        log.error("nothing evaluated")
        return 2

    pooled = pool_metrics(results)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([r.per_gt for r in results], ignore_index=True).to_csv(
        args.out_dir / "per_gt.csv", index=False)
    pd.concat([r.per_cluster for r in results], ignore_index=True).to_csv(
        args.out_dir / "per_cluster.csv", index=False)
    pd.concat([r.unmatched_preds for r in results], ignore_index=True).to_csv(
        args.out_dir / "unmatched_preds.csv", index=False)

    summary = {
        "config": {
            "min_area_m2": args.min_area,
            "aoi_min_overlap": args.aoi_min_overlap,
            "edge_buffer_m": args.edge_buffer,
            "tau_link": args.tau_link,
            "tau_floor": args.tau_floor,
            "max_cluster_size": args.max_cluster_size,
            "gt_files": [str(p) for p in gt_paths],
            "pred_files": [str(p) for p in pred_paths],
        },
        "note": (
            "The LiDAR annotation is deliberately incomplete; unmatched RGB masks "
            "are reported but never penalised. No metric here is a precision estimate."
        ),
        "plots": {
            r.plot: {
                "metrics": {k: v for k, v in r.metrics.items() if not k.startswith("_")},
                "preprocessing": vars(r.prep),
                "linking": r.link_info,
            }
            for r in results
        },
        "pooled": pooled,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    write_links_gpkg(results, args.out_dir / "links.gpkg")

    if not args.no_sweeps:
        link_df = sweep_tau_link(results, args)
        cov_df = sweep_tau_cov(results)
        link_df.to_csv(args.out_dir / "sweeps.csv", index=False)
        cov_df.to_csv(args.out_dir / "sweeps_coverage.csv", index=False)
        make_figure(link_df, cov_df, args.out_dir / "sweeps.png")

    print_report(results, pooled)
    log.info("outputs written to %s", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
