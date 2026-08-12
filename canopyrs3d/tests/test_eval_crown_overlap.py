"""Tests for scripts/eval_crown_overlap.py (plan.md §13).

Every case uses axis-aligned boxes so the expected IoU / coverage values are
analytic and written out in the assertions, rather than golden numbers copied
from a previous run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

# --- load the script as a module --------------------------------------------

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_crown_overlap.py"
_spec = importlib.util.spec_from_file_location("eval_crown_overlap", _SCRIPT)
eco = importlib.util.module_from_spec(_spec)
sys.modules["eval_crown_overlap"] = eco
_spec.loader.exec_module(eco)

CRS = "EPSG:32617"
ARGS = SimpleNamespace(
    tau_link=eco.TAU_LINK, tau_floor=eco.TAU_FLOOR, max_cluster_size=eco.MAX_CLUSTER_SIZE
)


def gdf(*geoms, **cols) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({**cols, "geometry": list(geoms)}, crs=CRS)


def run(gt: gpd.GeoDataFrame, preds: gpd.GeoDataFrame, args=ARGS):
    """Full per-plot evaluation on already-prepared inputs."""
    pairs = eco.overlap_pairs(gt, preds)
    edge_flags = np.zeros(len(gt), dtype=bool)
    per_gt, per_cluster, unmatched, metrics, clusters, info = eco.evaluate_plot(
        "test", gt, preds, pairs, edge_flags, args
    )
    return SimpleNamespace(
        pairs=pairs, per_gt=per_gt, per_cluster=per_cluster, unmatched=unmatched,
        metrics=metrics, clusters=clusters, info=info,
    )


# =============================================================================
# Pair-level arithmetic
# =============================================================================


def test_overlap_pair_measures_are_analytic():
    gt = gdf(box(0, 0, 10, 10))       # area 100
    preds = gdf(box(0, 0, 8, 10))     # area 80, fully inside
    (row,) = eco.overlap_pairs(gt, preds).itertuples()

    assert row.inter == pytest.approx(80.0)
    assert row.iou == pytest.approx(80 / 100)   # union is 100
    assert row.cov == pytest.approx(0.8)        # 80 / area(gt)
    assert row.pur == pytest.approx(1.0)        # 80 / area(pred)
    assert row.ovl == pytest.approx(1.0)        # 80 / min(100, 80)


def test_non_intersecting_pairs_are_absent():
    gt = gdf(box(0, 0, 10, 10))
    preds = gdf(box(50, 50, 60, 60))
    assert eco.overlap_pairs(gt, preds).empty


# =============================================================================
# Cluster taxonomy (plan.md §7)
# =============================================================================


@pytest.mark.parametrize(
    "n_gt,n_pred,expected",
    [
        (1, 1, "one_to_one"),
        (1, 3, "split"),
        (3, 1, "merge"),
        (2, 2, "tangled"),
        (1, 0, "missed"),
        (0, 1, "unmatched_pred"),
    ],
)
def test_classify_cluster(n_gt, n_pred, expected):
    assert eco.classify_cluster(n_gt, n_pred) == expected


def test_one_to_one_match():
    gt = gdf(box(0, 0, 10, 10))
    preds = gdf(box(0, 0, 8, 10))
    r = run(gt, preds)

    (cluster,) = r.per_cluster.itertuples()
    assert cluster.cluster_type == "one_to_one"
    assert cluster.iou_k == pytest.approx(0.8)
    assert cluster.cov_k == pytest.approx(0.8)
    assert r.metrics["mIoU_cluster"] == pytest.approx(0.8)


def test_split_is_credited_not_punished():
    """One GT crown cut into three abutting RGB masks reconstructs it exactly,
    so the cluster IoU is 1.0 even though no single mask exceeds IoU 1/3."""
    gt = gdf(box(0, 0, 9, 10))
    preds = gdf(box(0, 0, 3, 10), box(3, 0, 6, 10), box(6, 0, 9, 10))
    r = run(gt, preds)

    (cluster,) = r.per_cluster.itertuples()
    assert cluster.cluster_type == "split"
    assert cluster.n_pred == 3
    assert cluster.iou_k == pytest.approx(1.0)
    assert r.metrics["mIoU_cluster"] == pytest.approx(1.0)

    # The one-to-one view sees only a third of the crown — this gap is exactly
    # why the cluster metric is the headline and mIoU_1to1 is secondary.
    assert r.metrics["mIoU_1to1"] == pytest.approx(1 / 3)
    assert r.per_gt.loc[0, "cov"] == pytest.approx(1.0)
    assert r.per_gt.loc[0, "n_linked_preds"] == 3


def test_merge_is_credited_not_punished():
    gt = gdf(box(0, 0, 3, 10), box(3, 0, 6, 10), box(6, 0, 9, 10))
    preds = gdf(box(0, 0, 9, 10))
    r = run(gt, preds)

    (cluster,) = r.per_cluster.itertuples()
    assert cluster.cluster_type == "merge"
    assert cluster.n_gt == 3
    assert cluster.iou_k == pytest.approx(1.0)
    assert r.metrics["mIoU_cluster"] == pytest.approx(1.0)
    # Each crown is fully covered, though only one can win the 1-1 assignment.
    assert list(r.per_gt["cov"]) == pytest.approx([1.0, 1.0, 1.0])
    assert (r.per_gt["iou_1to1"] > 0).sum() == 1


def test_missed_crown_scores_zero():
    gt = gdf(box(0, 0, 10, 10), box(100, 100, 110, 110))
    preds = gdf(box(0, 0, 10, 10))
    r = run(gt, preds)

    types = set(r.per_cluster["cluster_type"])
    assert types == {"one_to_one", "missed"}
    missed = r.per_cluster[r.per_cluster["cluster_type"] == "missed"].iloc[0]
    assert missed.iou_k == 0.0 and missed.cov_k == 0.0
    # A missed crown drags the mean down: (1.0 + 0.0) / 2
    assert r.metrics["mIoU_cluster"] == pytest.approx(0.5)


# =============================================================================
# The governing constraint: incomplete ground truth (plan.md §2)
# =============================================================================


def test_unmatched_prediction_is_excluded_not_scored_zero():
    """An RGB mask with no LiDAR counterpart is most likely a tree the annotator
    deliberately skipped, so it must not enter the mean at all."""
    gt = gdf(box(0, 0, 10, 10))
    preds = gdf(box(0, 0, 10, 10), box(100, 100, 110, 110))
    r = run(gt, preds)

    assert len(r.unmatched) == 1
    assert r.metrics["n_unmatched_pred"] == 1
    # One perfect match plus one unmatched mask. Scoring the mask as 0 would
    # give 0.5; excluding it — as the plan requires — gives 1.0.
    assert r.metrics["mIoU_cluster"] == pytest.approx(1.0)
    assert r.metrics["mCov_cluster"] == pytest.approx(1.0)


def test_over_extending_mask_lowers_iou_but_not_coverage():
    """A mask spanning the crown plus an unannotated neighbour keeps coverage at
    1.0 while IoU drops — the documented lower-bound behaviour of mIoU_cluster."""
    gt = gdf(box(0, 0, 10, 10))            # area 100
    preds = gdf(box(0, 0, 20, 10))         # area 200, covers GT entirely
    r = run(gt, preds)

    (cluster,) = r.per_cluster.itertuples()
    assert cluster.cov_k == pytest.approx(1.0)
    assert cluster.iou_k == pytest.approx(0.5)   # 100 / 200
    assert r.metrics["mIoU_cluster"] < r.metrics["mCov_cluster"]


def test_iou_global_is_penalised_by_unmatched_masks_but_the_linked_variant_is_not():
    """IoU_global unions *every* prediction, so a mask over an unannotated tree
    inflates its denominator — the one metric where an unmatched mask costs
    something. IoU_global_linked is the GT-anchored counterpart."""
    gt = gdf(box(0, 0, 10, 10))                                  # 100 m2
    alone = run(gt, gdf(box(0, 0, 10, 10)))
    with_extra = run(gt, gdf(box(0, 0, 10, 10), box(50, 50, 60, 60)))  # +100 m2 elsewhere

    assert alone.metrics["IoU_global"] == pytest.approx(1.0)
    # union becomes 200 while the intersection stays 100
    assert with_extra.metrics["IoU_global"] == pytest.approx(0.5)
    # the linked variant ignores the unmatched mask entirely
    assert with_extra.metrics["IoU_global_linked"] == pytest.approx(1.0)

    # and the headline is untouched either way
    assert with_extra.metrics["mIoU_cluster"] == pytest.approx(1.0)
    assert with_extra.metrics["Cov_global"] == pytest.approx(1.0)


def test_no_precision_style_metrics_are_reported():
    gt = gdf(box(0, 0, 10, 10))
    preds = gdf(box(0, 0, 10, 10))
    keys = set(run(gt, preds).metrics)
    for forbidden in ("precision", "f1", "PQ", "RQ", "SQ"):
        assert not any(forbidden.lower() in k.lower() for k in keys), forbidden


# =============================================================================
# Linking criterion (plan.md §7)
# =============================================================================


def test_boundary_graze_does_not_link():
    """Neighbouring crowns touch; a slight overlap must not fuse them."""
    gt = gdf(box(0, 0, 10, 10))            # area 100
    preds = gdf(box(9.5, 0, 20, 10))       # inter 5 -> ovl = 5/100 = 0.05
    r = run(gt, preds)

    assert r.info["n_edges"] == 0
    assert set(r.per_cluster["cluster_type"]) == {"missed", "unmatched_pred"}


def test_min_normalisation_links_a_small_mask_inside_a_large_crown():
    """IoU alone would miss this; inter/min catches it. This is the split case
    in miniature and the whole reason for the criterion."""
    gt = gdf(box(0, 0, 10, 10))            # area 100
    preds = gdf(box(0, 0, 10, 1))          # area 10, inter 10
    pairs = eco.overlap_pairs(gt, preds)
    (row,) = pairs.itertuples()

    assert row.iou == pytest.approx(0.1)   # far below any sane IoU threshold
    assert row.ovl == pytest.approx(1.0)   # but fully contained
    clusters, info = eco.link_instances(pairs, 1, 1, tau_link=0.5, tau_floor=0.05)
    assert info["n_edges"] == 1


def test_tau_floor_rejects_extreme_scale_ratio_containment():
    """A tiny crown swallowed by a huge mask links at ovl=1.0; the IoU floor is
    what stops it. Measured to be a no-op on the real plots."""
    gt = gdf(box(0, 0, 1, 1))              # area 1
    preds = gdf(box(0, 0, 100, 100))       # area 10000, inter 1 -> iou = 1e-4
    pairs = eco.overlap_pairs(gt, preds)

    _, permissive = eco.link_instances(pairs, 1, 1, tau_link=0.5, tau_floor=0.0)
    _, guarded = eco.link_instances(pairs, 1, 1, tau_link=0.5, tau_floor=0.05)
    assert permissive["n_edges"] == 1
    assert guarded["n_edges"] == 0
    assert guarded["n_edges_removed_by_floor"] == 1


# =============================================================================
# Chaining guard (plan.md §7)
# =============================================================================


def test_chaining_guard_breaks_up_an_oversized_component():
    """Offset strips chain every instance into one component of 9. The guard
    must tighten tau until it fragments, rather than let one blob dominate."""
    gt = gdf(*[box(2 * k, 0, 2 * k + 2, 10) for k in range(5)])       # 5 crowns
    preds = gdf(*[box(2 * k + 1, 0, 2 * k + 3, 10) for k in range(4)])  # 4 masks

    pairs = eco.overlap_pairs(gt, preds)
    # every overlap is exactly half of each 20 m2 box -> ovl = 0.5
    assert set(np.round(pairs["ovl"], 6)) == {0.5}

    _, unguarded = eco.link_instances(
        pairs, 5, 4, tau_link=0.5, tau_floor=0.05, max_cluster_size=99)
    assert unguarded["max_cluster_size"] == 9

    _, guarded = eco.link_instances(
        pairs, 5, 4, tau_link=0.5, tau_floor=0.05, max_cluster_size=8)
    assert guarded["max_cluster_size"] <= 8
    assert guarded["tightened_clusters"], "guard should record that it fired"


def test_guard_is_inert_on_well_separated_instances():
    gt = gdf(box(0, 0, 10, 10), box(100, 0, 110, 10))
    preds = gdf(box(0, 0, 10, 10), box(100, 0, 110, 10))
    r = run(gt, preds)

    assert r.info["max_cluster_size"] == 2
    assert r.info["tightened_clusters"] == []


# =============================================================================
# Preprocessing (plan.md §5)
# =============================================================================


def test_aoi_restriction_drops_far_masks_and_clips_straddling_ones():
    gt = gdf(box(0, 0, 10, 10))
    preds = gdf(
        box(0, 0, 10, 10),        # inside
        box(500, 500, 510, 510),  # far away -> dropped
        box(8, 0, 40, 10),        # mostly outside -> below 0.4 overlap, dropped
    )
    stats = eco.PrepStats()
    aoi = eco.build_aoi(gt)
    kept = eco.restrict_preds(preds, aoi, min_overlap=0.4, min_area=1.0, stats=stats)

    assert len(kept) == 1
    assert stats.pred_outside_aoi == 1
    assert stats.pred_low_aoi_overlap == 1
    assert kept.geometry.iloc[0].area == pytest.approx(100.0)


def test_min_area_filter_drops_degenerate_crowns(tmp_path):
    path = tmp_path / "gt.gpkg"
    gdf(box(0, 0, 10, 10), box(20, 20, 20.5, 20.5)).to_file(path, driver="GPKG")

    stats = eco.PrepStats()
    kept = eco.load_gt(path, min_area=1.0, stats=stats)
    assert len(kept) == 1
    assert stats.gt_below_min_area == 1


def test_null_geometries_are_dropped(tmp_path):
    """Rows with hull_status != 'ok' carry NULL geometry in the real data."""
    path = tmp_path / "gt.gpkg"
    frame = gpd.GeoDataFrame(
        {"hull_status": ["ok", "below_canopy"], "geometry": [box(0, 0, 10, 10), None]},
        crs=CRS,
    )
    frame.to_file(path, driver="GPKG")

    stats = eco.PrepStats()
    kept = eco.load_gt(path, min_area=1.0, stats=stats)
    assert len(kept) == 1
    assert stats.gt_null_geom == 1
    assert stats.gt_by_hull_status == {"ok": 1, "below_canopy": 1}


def test_multipolygons_are_not_exploded():
    """An occluded crown split into two parts stays one instance."""
    from shapely.geometry import MultiPolygon

    gt = gdf(MultiPolygon([box(0, 0, 4, 10), box(6, 0, 10, 10)]))
    preds = gdf(box(0, 0, 10, 10))
    r = run(gt, preds)

    assert len(r.per_gt) == 1
    assert r.per_gt.loc[0, "cov"] == pytest.approx(1.0)


# =============================================================================
# Coverage sufficiency (plan.md §8)
# =============================================================================


def test_coverage_sufficiency_thresholds():
    gt = gdf(box(0, 0, 10, 10), box(100, 0, 110, 10))
    preds = gdf(box(0, 0, 10, 8), box(100, 0, 110, 6))  # 80% and 60% covered
    r = run(gt, preds)

    assert list(r.per_gt["cov"]) == pytest.approx([0.8, 0.6])
    assert list(r.per_gt["sufficient@0.5"]) == [True, True]
    assert list(r.per_gt["sufficient@0.75"]) == [True, False]
    assert list(r.per_gt["sufficient@0.9"]) == [False, False]
    assert r.metrics["frac_sufficient@0.5"] == pytest.approx(1.0)
    assert r.metrics["frac_sufficient@0.75"] == pytest.approx(0.5)


def test_cov_best_distinguishes_fragmented_from_single_mask_coverage():
    gt = gdf(box(0, 0, 9, 10))
    preds = gdf(box(0, 0, 3, 10), box(3, 0, 6, 10), box(6, 0, 9, 10))
    r = run(gt, preds)

    assert r.per_gt.loc[0, "cov"] == pytest.approx(1.0)        # stitched together
    assert r.per_gt.loc[0, "cov_best"] == pytest.approx(1 / 3)  # any single mask


# =============================================================================
# Invariants (plan.md §13)
# =============================================================================


def _result(gt, preds):
    r = run(gt, preds)
    return eco.PlotResult(
        "test", r.per_gt, r.per_cluster, r.unmatched, r.metrics,
        gt, preds, eco.PrepStats(), r.info,
    )


def test_invariants_hold_on_a_clean_case():
    gt = gdf(box(0, 0, 9, 10), box(20, 0, 29, 10))
    preds = gdf(box(0, 0, 8, 10), box(20, 0, 28, 10))
    assert eco.check_invariants(_result(gt, preds)) == []


def test_invariants_detect_overlapping_ground_truth():
    """GT crowns are a top-view partition; overlap means the input is wrong."""
    gt = gdf(box(0, 0, 10, 10), box(5, 0, 15, 10))
    preds = gdf(box(0, 0, 10, 10))
    problems = eco.check_invariants(_result(gt, preds))
    assert any("not disjoint" in p for p in problems)


def test_cluster_gt_area_is_conserved():
    gt = gdf(box(0, 0, 9, 10), box(20, 0, 29, 10), box(40, 0, 41, 10))
    preds = gdf(box(0, 0, 8, 10), box(20, 0, 28, 10))
    r = run(gt, preds)
    assert r.per_cluster["gt_area"].sum() == pytest.approx(gt.geometry.area.sum())


def test_one_to_one_never_exceeds_best_overlap():
    gt = gdf(box(0, 0, 9, 10), box(20, 0, 29, 10))
    preds = gdf(box(0, 0, 8, 10), box(4, 0, 9, 10), box(20, 0, 28, 10))
    m = run(gt, preds).metrics
    assert m["mIoU_1to1"] <= m["mIoU_best"] + 1e-9


def test_cluster_iou_never_exceeds_cluster_coverage():
    gt = gdf(box(0, 0, 10, 10), box(30, 0, 40, 10))
    preds = gdf(box(0, 0, 20, 10), box(30, 0, 35, 10))
    per_cluster = run(gt, preds).per_cluster
    assert (per_cluster["iou_k"] <= per_cluster["cov_k"] + 1e-9).all()


# =============================================================================
# Degenerate inputs
# =============================================================================


def test_no_predictions_at_all():
    gt = gdf(box(0, 0, 10, 10))
    preds = gpd.GeoDataFrame({"geometry": []}, crs=CRS)
    r = run(gt, preds)

    assert r.metrics["mIoU_cluster"] == 0.0
    assert r.metrics["mCov_cluster"] == 0.0
    assert set(r.per_cluster["cluster_type"]) == {"missed"}


def test_plot_name_parsing():
    assert eco.parse_plot_name(Path("asnortheast_plot_209_gt_crowns_gt.gpkg")) == "209"
    assert eco.parse_plot_name(Path("asnortheast_plot_210_gt_crowns_gt.gpkg")) == "210"
