"""
ClassifierComponent with simplified architecture.

Single __call__() method returns flattened DataFrame (no geometry).
Pipeline handles merging into existing GDF and I/O.
"""

import warnings
from pathlib import Path
from typing import Set

import pandas as pd

from geodataset.dataset import InstanceSegmentationLabeledRasterCocoDataset

from canopyrs.engine.constants import Col, StateKey, INFER_AOI_NAME
from canopyrs.engine.components.base import BaseComponent, ComponentResult, validate_requirements
from canopyrs.engine.config_parsers import ClassifierConfig
from canopyrs.engine.data_state import DataState
from canopyrs.engine.models.registry import CLASSIFIER_REGISTRY
from canopyrs.engine.models.utils import collate_fn_infer_image_masks


class ClassifierComponent(BaseComponent):
    """
    Classifies objects in polygon-tiled imagery.

    Requirements:
        - tiles_path: Directory containing polygon tiles
        - infer_coco_path: COCO annotations with instance masks

    Produces:
        - Updated infer_gdf with classification results
        - Columns: classifier_score, classifier_class, classifier_scores
    """

    name = 'classifier'

    BASE_REQUIRES_STATE = {StateKey.TILES_PATH, StateKey.INFER_COCO_PATH}
    BASE_REQUIRES_COLUMNS: Set[str] = set()

    BASE_PRODUCES_STATE = {StateKey.INFER_GDF, StateKey.INFER_COCO_PATH}
    BASE_PRODUCES_COLUMNS = {Col.CLASSIFIER_SCORE, Col.CLASSIFIER_CLASS, Col.CLASSIFIER_SCORES}

    BASE_STATE_HINTS = {
        StateKey.TILES_PATH: "Classifier needs polygon tiles. Add a tilerizer with tile_type='polygon'.",
        StateKey.INFER_COCO_PATH: "Classifier needs COCO annotations from a polygon tilerizer.",
    }

    BASE_COLUMN_HINTS = {
        Col.OBJECT_ID: "Classifier needs object IDs to merge results back to infer_gdf.",
    }

    def __init__(
        self,
        config: ClassifierConfig,
        parent_output_path: str = None,
        component_id: int = None
    ):
        super().__init__(config, parent_output_path, component_id)

        # Store model class (instantiate in __call__ to avoid loading during validation)
        if config.model not in CLASSIFIER_REGISTRY:
            raise ValueError(f'Invalid classifier model: {config.model}')
        self._model_class = CLASSIFIER_REGISTRY.get(config.model)

        # Set requirements
        self.requires_state = set(self.BASE_REQUIRES_STATE)
        self.requires_columns = set(self.BASE_REQUIRES_COLUMNS)
        self.produces_state = set(self.BASE_PRODUCES_STATE)
        self.produces_columns = set(self.BASE_PRODUCES_COLUMNS)

        # Set hints
        self.state_hints = dict(self.BASE_STATE_HINTS)
        self.column_hints = dict(self.BASE_COLUMN_HINTS)

    @validate_requirements
    def __call__(self, data_state: DataState) -> ComponentResult:
        """
        Run classification on polygon tiles.

        Returns flattened DataFrame (no geometry) with classification results.
        Pipeline handles merging into existing GDF.
        """
        
        classifier = self._model_class(self.config)

        # Create dataset
        infer_ds = InstanceSegmentationLabeledRasterCocoDataset(
            root_path=[data_state.tiles_path, Path(data_state.infer_coco_path).parent],
            transform=None,
            fold=INFER_AOI_NAME,
            other_attributes_names_to_pass=[Col.OBJECT_ID]
        )

        # Run inference
        tiles_paths, class_scores, class_predictions, object_ids = classifier.infer(
            infer_ds, collate_fn_infer_image_masks
        )

        # Flatten outputs into DataFrame (no geometry - will merge into existing)
        df = pd.DataFrame({
            Col.OBJECT_ID: object_ids,
            Col.TILE_PATH: tiles_paths,  # Include for fallback merge key
            Col.CLASSIFIER_CLASS: class_predictions,
            Col.CLASSIFIER_SCORE: [
                scores[pred_idx] for scores, pred_idx in zip(class_scores, class_predictions)
            ],
            Col.CLASSIFIER_SCORES: class_scores,
        })

        # Component-specific validation: warn about unclassified items
        unclassified = df[Col.CLASSIFIER_CLASS].isnull().sum()
        if unclassified > 0:
            warnings.warn(f"{unclassified} items could not be classified.")

        print(f"ClassifierComponent: Classified {len(df) - unclassified}/{len(df)} items.")

        return ComponentResult(
            gdf=df,  # DataFrame, not GeoDataFrame - no geometry
            produced_columns={Col.CLASSIFIER_SCORE, Col.CLASSIFIER_CLASS, Col.CLASSIFIER_SCORES},
            save_gpkg=True,
            gpkg_name_suffix="notaggregated",  # classifier saves final results
            save_coco=True,
            coco_scores_column=Col.CLASSIFIER_SCORE,
            coco_categories_column=Col.CLASSIFIER_CLASS,
        )
