"""
Tests for Pipeline orchestration logic.

Uses _StubComponent (a minimal BaseComponent subclass) so that validation,
initial-state derivation, and merge logic all run through the real Pipeline code.
"""

import pytest
from unittest.mock import patch

from shapely.geometry import box
import geopandas as gpd
import pandas as pd

from canopyrs.engine.pipeline import Pipeline, PipelineValidationError
from canopyrs.engine.constants import Col, StateKey
from canopyrs.engine.data_state import DataState
from canopyrs.engine.components.base import BaseComponent, ComponentResult


class _StubComponent(BaseComponent):
    """Minimal concrete component for pipeline unit tests."""

    def __init__(self, name, requires_state=None, requires_columns=None,
                 produces_state=None, produces_columns=None):
        super().__init__(config=None)
        self.name = name
        self.requires_state = requires_state or set()
        self.requires_columns = requires_columns or set()
        self.produces_state = produces_state or set()
        self.produces_columns = produces_columns or set()

    def __call__(self, data_state):
        return ComponentResult()


@pytest.fixture
def make_pipeline(tmp_path):
    """Factory fixture: constructs Pipeline with flowchart suppressed, handles executor cleanup."""
    pipelines = []

    def _factory(components, data_state=None):
        if data_state is None:
            data_state = DataState()
        with patch.object(Pipeline, '_print_flow_chart'):
            p = Pipeline(
                components=components,
                data_state=data_state,
                output_path=str(tmp_path),
            )
        pipelines.append(p)
        return p

    yield _factory

    for p in pipelines:
        p.background_executor.shutdown(wait=False)


class TestPipelineValidation:
    """Pipeline._validate_pipeline catches real dependency errors at construction time."""

    def test_missing_state_raises(self, make_pipeline):
        """Component requiring state that nothing produces → PipelineValidationError."""
        with pytest.raises(PipelineValidationError, match="infer_gdf"):
            make_pipeline([
                _StubComponent("aggregator", requires_state={StateKey.INFER_GDF}),
            ])

    def test_missing_column_raises(self, make_pipeline):
        """Component requiring column not present in initial GDF → PipelineValidationError."""
        gdf = gpd.GeoDataFrame({
            Col.GEOMETRY: [box(0, 0, 1, 1)],
            Col.OBJECT_ID: [0],
        }, geometry=Col.GEOMETRY)

        with pytest.raises(PipelineValidationError):
            make_pipeline(
                [_StubComponent("aggregator",
                    requires_state={StateKey.INFER_GDF},
                    requires_columns={Col.DETECTOR_SCORE},
                )],
                data_state=DataState(infer_gdf=gdf),
            )

    def test_valid_chain_passes(self, make_pipeline):
        """State produced by earlier component satisfies later component → no error."""
        make_pipeline([
            _StubComponent("tilerizer",
                requires_state={StateKey.IMAGERY_PATH},
                produces_state={StateKey.TILES_PATH},
            ),
            _StubComponent("detector",
                requires_state={StateKey.TILES_PATH},
                produces_state={StateKey.INFER_GDF},
            ),
        ], data_state=DataState(imagery_path="/fake/raster.tif"))

    def test_order_matters(self, make_pipeline):
        """Detector before tilerizer fails — tiles_path not yet produced."""
        with pytest.raises(PipelineValidationError, match="tiles_path"):
            make_pipeline([
                _StubComponent("detector",
                    requires_state={StateKey.TILES_PATH},
                ),
                _StubComponent("tilerizer",
                    requires_state={StateKey.IMAGERY_PATH},
                    produces_state={StateKey.TILES_PATH},
                ),
            ], data_state=DataState(imagery_path="/fake/raster.tif"))

    def test_column_produced_by_earlier_component_satisfies_later(self, make_pipeline):
        """Column produced by detector is available to aggregator."""
        make_pipeline([
            _StubComponent("detector",
                requires_state={StateKey.TILES_PATH},
                produces_state={StateKey.INFER_GDF},
                produces_columns={Col.GEOMETRY, Col.OBJECT_ID, Col.TILE_PATH, Col.DETECTOR_SCORE},
            ),
            _StubComponent("aggregator",
                requires_state={StateKey.INFER_GDF},
                requires_columns={Col.DETECTOR_SCORE},
            ),
        ], data_state=DataState(tiles_path="/fake/tiles/"))


class TestPipelineInitialState:
    """Pipeline correctly derives initial state keys and columns from DataState."""

    def test_imagery_path(self, make_pipeline):
        p = make_pipeline([], DataState(imagery_path="/r.tif"))
        assert StateKey.IMAGERY_PATH in p._get_initial_state_keys()

    def test_tiles_path(self, make_pipeline):
        p = make_pipeline([], DataState(tiles_path="/tiles/"))
        assert StateKey.TILES_PATH in p._get_initial_state_keys()

    def test_infer_gdf_columns(self, make_pipeline):
        gdf = gpd.GeoDataFrame({
            Col.GEOMETRY: [box(0, 0, 1, 1)],
            Col.DETECTOR_SCORE: [0.9],
        }, geometry=Col.GEOMETRY)
        p = make_pipeline([], DataState(infer_gdf=gdf))

        assert Col.DETECTOR_SCORE in p._get_initial_columns()
        assert Col.GEOMETRY in p._get_initial_columns()

    def test_empty_data_state(self, make_pipeline):
        p = make_pipeline([])
        assert len(p._get_initial_state_keys()) == 0
        assert len(p._get_initial_columns()) == 0


class TestPipelineMergeLogic:
    """Pipeline._merge_result_gdf exercises all documented merge rules."""

    def test_no_existing_gdf_sets_directly_with_object_ids(self, make_pipeline):
        """Rule 1: no existing GDF → result becomes base, object_ids assigned."""
        p = make_pipeline([])

        result = gpd.GeoDataFrame({
            Col.GEOMETRY: [box(0, 0, 1, 1), box(1, 1, 2, 2)],
            Col.DETECTOR_SCORE: [0.9, 0.8],
        }, geometry=Col.GEOMETRY)

        merged = p._merge_result_gdf(result)

        assert Col.OBJECT_ID in merged.columns
        assert list(merged[Col.OBJECT_ID]) == [0, 1]
        assert len(merged) == 2

    def test_new_geometry_merges_existing_columns(self, make_pipeline):
        """Rule 2: result has geometry + object_id → new geometry, existing columns preserved."""
        existing = gpd.GeoDataFrame({
            Col.GEOMETRY: [box(0, 0, 1, 1), box(1, 1, 2, 2)],
            Col.OBJECT_ID: [0, 1],
            Col.DETECTOR_SCORE: [0.9, 0.8],
        }, geometry=Col.GEOMETRY)
        p = make_pipeline([], DataState(infer_gdf=existing))

        result = gpd.GeoDataFrame({
            Col.GEOMETRY: [box(0.1, 0.1, 0.9, 0.9), box(1.1, 1.1, 1.9, 1.9)],
            Col.OBJECT_ID: [0, 1],
            Col.SEGMENTER_SCORE: [0.85, 0.75],
        }, geometry=Col.GEOMETRY)

        merged = p._merge_result_gdf(result)

        # Geometry replaced by segmenter masks
        assert merged.geometry.iloc[0].equals(box(0.1, 0.1, 0.9, 0.9))
        # Detector score carried over from existing
        assert Col.DETECTOR_SCORE in merged.columns
        # Segmenter score present from result
        assert Col.SEGMENTER_SCORE in merged.columns

    def test_full_replacement_when_no_merge_key(self, make_pipeline):
        """Rule 3: result has geometry but no shared merge key → full replacement."""
        existing = gpd.GeoDataFrame({
            Col.GEOMETRY: [box(0, 0, 1, 1), box(1, 1, 2, 2)],
            Col.OBJECT_ID: [0, 1],
        }, geometry=Col.GEOMETRY)
        p = make_pipeline([], DataState(infer_gdf=existing))

        # Aggregator-style: merged polygon, no object_id column at all
        result = gpd.GeoDataFrame({
            Col.GEOMETRY: [box(0, 0, 2, 2)],
            Col.AGGREGATOR_SCORE: [0.95],
        }, geometry=Col.GEOMETRY)

        merged = p._merge_result_gdf(result)

        assert len(merged) == 1
        assert Col.AGGREGATOR_SCORE in merged.columns
        assert Col.OBJECT_ID in merged.columns  # freshly assigned

    def test_attributes_merge_into_existing(self, make_pipeline):
        """Rule 4: result has no geometry → attributes merge into existing GDF by object_id."""
        existing = gpd.GeoDataFrame({
            Col.GEOMETRY: [box(0, 0, 1, 1), box(1, 1, 2, 2)],
            Col.OBJECT_ID: [0, 1],
            Col.DETECTOR_SCORE: [0.9, 0.8],
        }, geometry=Col.GEOMETRY)
        p = make_pipeline([], DataState(infer_gdf=existing))

        # Classifier-style: DataFrame with new columns, no geometry
        result = pd.DataFrame({
            Col.OBJECT_ID: [0, 1],
            Col.CLASSIFIER_CLASS: [2, 5],
            Col.CLASSIFIER_SCORE: [0.95, 0.88],
        })

        merged = p._merge_result_gdf(result)

        assert Col.GEOMETRY in merged.columns  # geometry preserved from existing
        assert Col.CLASSIFIER_CLASS in merged.columns
        assert list(merged[Col.CLASSIFIER_CLASS]) == [2, 5]

    def test_no_geometry_no_merge_key_raises(self, make_pipeline):
        """No geometry + no valid merge key → ValueError."""
        existing = gpd.GeoDataFrame({
            Col.GEOMETRY: [box(0, 0, 1, 1)],
            Col.OBJECT_ID: [0],
        }, geometry=Col.GEOMETRY)
        p = make_pipeline([], DataState(infer_gdf=existing))

        with pytest.raises(ValueError, match="no valid merge key"):
            p._merge_result_gdf(pd.DataFrame({"unrelated": [1]}))

    def test_empty_gdf_replaces_existing(self, make_pipeline):
        """Empty result GDF replaces existing GDF (e.g. aggregator filters all detections)."""
        existing = gpd.GeoDataFrame({
            Col.GEOMETRY: [box(0, 0, 1, 1), box(1, 1, 2, 2)],
            Col.OBJECT_ID: [0, 1],
            Col.DETECTOR_SCORE: [0.9, 0.8],
        }, geometry=Col.GEOMETRY)
        p = make_pipeline([], DataState(infer_gdf=existing))

        empty_gdf = gpd.GeoDataFrame(columns=[Col.GEOMETRY], geometry=Col.GEOMETRY)
        component = _StubComponent("aggregator",
            requires_state={StateKey.INFER_GDF},
            produces_state={StateKey.INFER_GDF},
        )
        component.output_path = p.output_path
        component.component_id = 0
        result = ComponentResult(gdf=empty_gdf, save_coco=True, coco_scores_column="aggregator_score")

        with pytest.warns(UserWarning, match="empty GeoDataFrame"):
            p._process_result(component, result)

        assert p.data_state.infer_gdf is not None
        assert len(p.data_state.infer_gdf) == 0
