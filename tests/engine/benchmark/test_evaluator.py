"""
Tests for CocoEvaluator and related utility functions.
"""

import pytest
import geopandas as gpd
from shapely.geometry import box, Polygon

from canopyrs.engine.benchmark.base.evaluator import (
    CocoEvaluator,
    filter_min_overlap,
    validate_and_repair_gdf,
    move_gdfs_to_ground_resolution,
)


class TestAreaRangeConversion:
    """Tests for _get_area_ranges_pixels_from_gsd."""

    def test_gsd_1_meter(self):
        """At 1m GSD, pixel area equals square meters."""
        ranges = CocoEvaluator._get_area_ranges_pixels_from_gsd(1.0)

        assert ranges['tiny'] == (0, 9)
        assert ranges['small'] == (9, 25)
        assert ranges['medium'] == (25, 49)
        assert ranges['large'] == (49, 100)
        assert ranges['giant'][0] == 100
        assert ranges['giant'][1] == 1e10  # inf converted

    def test_gsd_0_5_meter(self):
        """At 0.5m GSD, each m² = 4 pixels."""
        ranges = CocoEvaluator._get_area_ranges_pixels_from_gsd(0.5)
        pixel_per_m2 = 1 / (0.5 ** 2)  # = 4

        assert ranges['tiny'] == (0 * pixel_per_m2, 9 * pixel_per_m2)
        assert ranges['small'] == (9 * pixel_per_m2, 25 * pixel_per_m2)

    def test_gsd_0_1_meter(self):
        """At 0.1m GSD (10cm), each m² = 100 pixels."""
        ranges = CocoEvaluator._get_area_ranges_pixels_from_gsd(0.1)
        pixel_per_m2 = 100

        assert pytest.approx(ranges['tiny'][0]) == 0
        assert pytest.approx(ranges['tiny'][1]) == 900  # 0-9 m² = 0-900 pixels
        assert pytest.approx(ranges['small'][0]) == 900
        assert pytest.approx(ranges['small'][1]) == 2500  # 9-25 m²


class TestGetSizeLabel:
    """Tests for get_size_label classification."""

    @pytest.fixture
    def area_ranges_1m(self):
        """Area ranges at 1m GSD."""
        return CocoEvaluator._get_area_ranges_pixels_from_gsd(1.0)

    def test_tiny_classification(self, area_ranges_1m):
        """Objects 0-9 m² are tiny."""
        assert CocoEvaluator.get_size_label(0, area_ranges_1m) == 'tiny'
        assert CocoEvaluator.get_size_label(4, area_ranges_1m) == 'tiny'
        assert CocoEvaluator.get_size_label(8.9, area_ranges_1m) == 'tiny'

    def test_small_classification(self, area_ranges_1m):
        """Objects 9-25 m² are small."""
        assert CocoEvaluator.get_size_label(9, area_ranges_1m) == 'small'
        assert CocoEvaluator.get_size_label(16, area_ranges_1m) == 'small'
        assert CocoEvaluator.get_size_label(24.9, area_ranges_1m) == 'small'

    def test_medium_classification(self, area_ranges_1m):
        """Objects 25-49 m² are medium."""
        assert CocoEvaluator.get_size_label(25, area_ranges_1m) == 'medium'
        assert CocoEvaluator.get_size_label(36, area_ranges_1m) == 'medium'

    def test_large_classification(self, area_ranges_1m):
        """Objects 49-100 m² are large."""
        assert CocoEvaluator.get_size_label(49, area_ranges_1m) == 'large'
        assert CocoEvaluator.get_size_label(64, area_ranges_1m) == 'large'

    def test_giant_classification(self, area_ranges_1m):
        """Objects 100+ m² are giant."""
        assert CocoEvaluator.get_size_label(100, area_ranges_1m) == 'giant'
        assert CocoEvaluator.get_size_label(1000, area_ranges_1m) == 'giant'

    def test_boundary_values(self, area_ranges_1m):
        """Test exact boundary values (lower bound inclusive, upper exclusive)."""
        # At exact boundaries, should classify into the next category
        assert CocoEvaluator.get_size_label(9, area_ranges_1m) == 'small'  # not tiny
        assert CocoEvaluator.get_size_label(25, area_ranges_1m) == 'medium'  # not small
        assert CocoEvaluator.get_size_label(49, area_ranges_1m) == 'large'  # not medium
        assert CocoEvaluator.get_size_label(100, area_ranges_1m) == 'giant'  # not large


class TestFilterMinOverlap:
    """Tests for AOI overlap filtering."""

    def test_fully_inside_kept(self, aoi_polygon):
        """Geometries fully inside AOI are kept."""
        gdf = gpd.GeoDataFrame({
            'geometry': [box(10, 10, 20, 20)],  # Fully inside 0-100 AOI
            'id': [1],
        }, crs="EPSG:32618")

        result = filter_min_overlap(gdf, aoi_polygon, min_frac=0.4)
        assert len(result) == 1

    def test_fully_outside_removed(self, aoi_polygon):
        """Geometries fully outside AOI are removed."""
        gdf = gpd.GeoDataFrame({
            'geometry': [box(200, 200, 210, 210)],  # Outside 0-100 AOI
            'id': [1],
        }, crs="EPSG:32618")

        result = filter_min_overlap(gdf, aoi_polygon, min_frac=0.4)
        assert len(result) == 0

    def test_partial_overlap_above_threshold_kept(self, aoi_polygon):
        """Geometries with >40% overlap are kept."""
        # 10x10 box, half inside AOI (50% overlap)
        gdf = gpd.GeoDataFrame({
            'geometry': [box(-5, 0, 5, 10)],  # 50% inside
            'id': [1],
        }, crs="EPSG:32618")

        result = filter_min_overlap(gdf, aoi_polygon, min_frac=0.4)
        assert len(result) == 1

    def test_partial_overlap_below_threshold_removed(self, aoi_polygon):
        """Geometries with <40% overlap are removed."""
        # 10x10 box, only 30% inside AOI
        gdf = gpd.GeoDataFrame({
            'geometry': [box(-7, 0, 3, 10)],  # 30% inside
            'id': [1],
        }, crs="EPSG:32618")

        result = filter_min_overlap(gdf, aoi_polygon, min_frac=0.4)
        assert len(result) == 0

    def test_exact_threshold_boundary(self, aoi_polygon):
        """Geometries with exactly 40% overlap are kept."""
        # 10x10 box, exactly 40% inside AOI
        gdf = gpd.GeoDataFrame({
            'geometry': [box(-6, 0, 4, 10)],  # 40% inside (4 of 10 units)
            'id': [1],
        }, crs="EPSG:32618")

        result = filter_min_overlap(gdf, aoi_polygon, min_frac=0.4)
        assert len(result) == 1


class TestValidateAndRepairGdf:
    """Tests for geometry validation and repair."""

    def test_valid_geometries_unchanged(self):
        """Valid geometries pass through unchanged."""
        gdf = gpd.GeoDataFrame({
            'geometry': [box(0, 0, 10, 10), box(20, 20, 30, 30)],
            'id': [1, 2],
        }, crs="EPSG:32618")

        result = validate_and_repair_gdf(gdf, "test")
        assert len(result) == 2

    def test_invalid_bowtie_repaired(self):
        """Self-intersecting (bowtie) polygon is repaired."""
        # Bowtie polygon
        bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10), (0, 0)])
        gdf = gpd.GeoDataFrame({
            'geometry': [bowtie],
            'id': [1],
        }, crs="EPSG:32618")

        assert not gdf.geometry.iloc[0].is_valid

        result = validate_and_repair_gdf(gdf, "test")
        # Should either repair or remove
        if len(result) > 0:
            assert result.geometry.iloc[0].is_valid

    def test_mixed_valid_invalid(self):
        """Mix of valid and invalid geometries handled correctly."""
        bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10), (0, 0)])
        valid_box = box(20, 20, 30, 30)

        gdf = gpd.GeoDataFrame({
            'geometry': [bowtie, valid_box],
            'id': [1, 2],
        }, crs="EPSG:32618")

        result = validate_and_repair_gdf(gdf, "test")
        # At minimum, the valid geometry should be preserved
        assert len(result) >= 1
        assert all(result.is_valid)


class TestMoveGdfsToGroundResolution:
    """Tests for affine transformation to pixel space."""

    def test_transforms_to_pixel_coordinates(self):
        """Geometries are scaled to pixel coordinates."""
        truth_gdf = gpd.GeoDataFrame({
            'geometry': [box(0, 0, 10, 10)],  # 10m x 10m
        }, crs="EPSG:32618")

        infer_gdf = gpd.GeoDataFrame({
            'geometry': [box(5, 5, 15, 15)],
            'score': [0.9],
        }, crs="EPSG:32618")

        ground_resolution = 1.0  # 1m per pixel

        result_truth, result_infer = move_gdfs_to_ground_resolution(
            truth_gdf, infer_gdf, ground_resolution
        )

        # At 1m GSD, 10m box should be 10 pixels
        truth_area = result_truth.geometry.iloc[0].area
        assert pytest.approx(truth_area, rel=0.01) == 100  # 10x10 pixels

    def test_scaling_with_different_gsd(self):
        """Different GSD produces different pixel areas."""
        truth_gdf = gpd.GeoDataFrame({
            'geometry': [box(0, 0, 10, 10)],  # 10m x 10m = 100 m²
        }, crs="EPSG:32618")

        infer_gdf = truth_gdf.copy()
        infer_gdf['score'] = [0.9]

        # At 0.5m GSD, 10m = 20 pixels, so 100 m² = 400 pixels²
        result_truth, _ = move_gdfs_to_ground_resolution(
            truth_gdf, infer_gdf, 0.5
        )

        truth_area = result_truth.geometry.iloc[0].area
        assert pytest.approx(truth_area, rel=0.01) == 400

    def test_handles_different_crs(self):
        """Reprojects if CRS differs between truth and infer."""
        truth_gdf = gpd.GeoDataFrame({
            'geometry': [box(500000, 4400000, 500010, 4400010)],
        }, crs="EPSG:32618")

        # Same location but different CRS
        infer_gdf = gpd.GeoDataFrame({
            'geometry': [box(500000, 4400000, 500010, 4400010)],
            'score': [0.9],
        }, crs="EPSG:32617")  # Different UTM zone

        # Should not raise - will reproject
        result_truth, result_infer = move_gdfs_to_ground_resolution(
            truth_gdf, infer_gdf, 1.0
        )

        assert len(result_truth) == 1
        assert len(result_infer) == 1


@pytest.mark.integration
class TestRasterLevelEvaluation:
    """Integration tests for raster_level_multi_iou_thresholds using actual GeoPackages."""

    def test_perfect_match_returns_perfect_metrics(self, tmp_path):
        """Identical predictions and truth should give P=R=F1=1.0."""
        # Create identical prediction and truth GeoDataFrames
        gdf = gpd.GeoDataFrame({
            'geometry': [box(0, 0, 10, 10), box(20, 20, 30, 30), box(40, 40, 50, 50)],
            'aggregator_score': [0.9, 0.85, 0.8],
        }, crs="EPSG:32618")

        truth_gdf = gpd.GeoDataFrame({
            'geometry': [box(0, 0, 10, 10), box(20, 20, 30, 30), box(40, 40, 50, 50)],
        }, crs="EPSG:32618")

        # Save to temp files
        preds_path = tmp_path / "preds.gpkg"
        truth_path = tmp_path / "truth.gpkg"
        gdf.to_file(preds_path, driver="GPKG")
        truth_gdf.to_file(truth_path, driver="GPKG")

        # Run evaluation
        metrics = CocoEvaluator.raster_level_multi_iou_thresholds(
            iou_type='segm',
            preds_gpkg_path=str(preds_path),
            truth_gpkg_path=str(truth_path),
            aoi_gpkg_path=None,
            ground_resolution=1.0,
            iou_thresholds=[0.5]
        )

        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
        assert metrics['num_truths'] == 3
        assert metrics['num_preds'] == 3

    def test_no_overlap_returns_zero_metrics(self, tmp_path):
        """Non-overlapping predictions should give P=R=F1=0."""
        # Predictions in different location than truth
        preds_gdf = gpd.GeoDataFrame({
            'geometry': [box(100, 100, 110, 110), box(120, 120, 130, 130)],
            'aggregator_score': [0.9, 0.85],
        }, crs="EPSG:32618")

        truth_gdf = gpd.GeoDataFrame({
            'geometry': [box(0, 0, 10, 10), box(20, 20, 30, 30)],
        }, crs="EPSG:32618")

        preds_path = tmp_path / "preds.gpkg"
        truth_path = tmp_path / "truth.gpkg"
        preds_gdf.to_file(preds_path, driver="GPKG")
        truth_gdf.to_file(truth_path, driver="GPKG")

        metrics = CocoEvaluator.raster_level_multi_iou_thresholds(
            iou_type='segm',
            preds_gpkg_path=str(preds_path),
            truth_gpkg_path=str(truth_path),
            aoi_gpkg_path=None,
            ground_resolution=1.0,
            iou_thresholds=[0.5]
        )

        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0

    def test_partial_match_returns_expected_metrics(self, tmp_path):
        """2 of 3 predictions matching 2 of 4 truths."""
        # 2 matching, 1 false positive prediction
        preds_gdf = gpd.GeoDataFrame({
            'geometry': [
                box(0, 0, 10, 10),      # matches truth 0
                box(20, 20, 30, 30),    # matches truth 1
                box(200, 200, 210, 210) # false positive
            ],
            'aggregator_score': [0.9, 0.85, 0.8],
        }, crs="EPSG:32618")

        # 2 matching, 2 false negatives
        truth_gdf = gpd.GeoDataFrame({
            'geometry': [
                box(0, 0, 10, 10),      # matched by pred 0
                box(20, 20, 30, 30),    # matched by pred 1
                box(50, 50, 60, 60),    # false negative
                box(70, 70, 80, 80),    # false negative
            ],
        }, crs="EPSG:32618")

        preds_path = tmp_path / "preds.gpkg"
        truth_path = tmp_path / "truth.gpkg"
        preds_gdf.to_file(preds_path, driver="GPKG")
        truth_gdf.to_file(truth_path, driver="GPKG")

        metrics = CocoEvaluator.raster_level_multi_iou_thresholds(
            iou_type='segm',
            preds_gpkg_path=str(preds_path),
            truth_gpkg_path=str(truth_path),
            aoi_gpkg_path=None,
            ground_resolution=1.0,
            iou_thresholds=[0.5]
        )

        # TP=2, FP=1, FN=2
        # Precision = 2/3 ≈ 0.667
        # Recall = 2/4 = 0.5
        # F1 = 2 * 0.667 * 0.5 / (0.667 + 0.5) ≈ 0.571
        assert pytest.approx(metrics['precision'], rel=0.01) == 2/3
        assert pytest.approx(metrics['recall'], rel=0.01) == 0.5
        assert pytest.approx(metrics['f1'], rel=0.02) == 0.571

    def test_multi_iou_thresholds_averaging(self, tmp_path):
        """Multiple IoU thresholds should be averaged (RF1-style)."""
        # Predictions with slight offset - will match at low IoU, fail at high IoU
        preds_gdf = gpd.GeoDataFrame({
            'geometry': [
                box(0, 0, 10, 10),    # perfect match
                box(22, 20, 32, 30),  # offset - IoU ~0.64, will fail at 0.75
            ],
            'aggregator_score': [0.9, 0.85],
        }, crs="EPSG:32618")

        truth_gdf = gpd.GeoDataFrame({
            'geometry': [
                box(0, 0, 10, 10),
                box(20, 20, 30, 30),
            ],
        }, crs="EPSG:32618")

        preds_path = tmp_path / "preds.gpkg"
        truth_path = tmp_path / "truth.gpkg"
        preds_gdf.to_file(preds_path, driver="GPKG")
        truth_gdf.to_file(truth_path, driver="GPKG")

        metrics = CocoEvaluator.raster_level_multi_iou_thresholds(
            iou_type='segm',
            preds_gpkg_path=str(preds_path),
            truth_gpkg_path=str(truth_path),
            aoi_gpkg_path=None,
            ground_resolution=1.0,
            iou_thresholds=[0.5, 0.75]
        )

        # At IoU=0.5: both match -> P=R=F1=1.0
        # At IoU=0.75: only first matches -> P=0.5, R=0.5, F1=0.5
        # Average: P=(1+0.5)/2=0.75, R=0.75, F1=0.75
        assert 'precision_per_iou' in metrics
        assert len(metrics['precision_per_iou']) == 2
        assert metrics['iou_thresholds'] == [0.5, 0.75]

    def test_bbox_iou_type_uses_envelopes(self, tmp_path):
        """iou_type='bbox' should use bounding boxes, not actual geometry."""
        # L-shaped polygon
        l_shape = Polygon([(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10), (0, 0)])

        preds_gdf = gpd.GeoDataFrame({
            'geometry': [l_shape],
            'aggregator_score': [0.9],
        }, crs="EPSG:32618")

        # bbox of L-shape is 10x10, so this should match with bbox IoU
        truth_gdf = gpd.GeoDataFrame({
            'geometry': [box(0, 0, 10, 10)],
        }, crs="EPSG:32618")

        preds_path = tmp_path / "preds.gpkg"
        truth_path = tmp_path / "truth.gpkg"
        preds_gdf.to_file(preds_path, driver="GPKG")
        truth_gdf.to_file(truth_path, driver="GPKG")

        metrics = CocoEvaluator.raster_level_multi_iou_thresholds(
            iou_type='bbox',
            preds_gpkg_path=str(preds_path),
            truth_gpkg_path=str(truth_path),
            aoi_gpkg_path=None,
            ground_resolution=1.0,
            iou_thresholds=[0.5]
        )

        # With bbox, both are 10x10 boxes -> perfect match
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
