"""
DetectorComponent with simplified architecture.

Single __call__() method returns flattened GDF.
Pipeline handles merging, object_id assignment, and I/O.
"""

from typing import Set

import geopandas as gpd

from geodataset.dataset import UnlabeledRasterDataset

from canopyrs.engine.constants import Col, StateKey
from canopyrs.engine.components.base import BaseComponent, ComponentResult, validate_requirements
from canopyrs.engine.config_parsers import DetectorConfig
from canopyrs.engine.data_state import DataState
from canopyrs.engine.models.registry import DETECTOR_REGISTRY
from canopyrs.engine.models.utils import collate_fn_images


class DetectorComponent(BaseComponent):
    """
    Runs object detection on image tiles.

    Requirements:
        - tiles_path: Directory containing tiles to process

    Produces:
        - infer_gdf: GeoDataFrame with detected bounding boxes
        - Columns: geometry, object_id, tile_path, detector_score, detector_class
    """

    name = 'detector'

    BASE_REQUIRES_STATE = {StateKey.TILES_PATH}
    BASE_REQUIRES_COLUMNS: Set[str] = set()

    BASE_PRODUCES_STATE = {StateKey.INFER_GDF, StateKey.INFER_COCO_PATH}
    BASE_PRODUCES_COLUMNS = {Col.GEOMETRY, Col.OBJECT_ID, Col.TILE_PATH, Col.DETECTOR_SCORE, Col.DETECTOR_CLASS}

    BASE_STATE_HINTS = {
        StateKey.TILES_PATH: (
            "Detector needs tiles to process. Add a tilerizer before detector."
        ),
    }

    BASE_COLUMN_HINTS: dict = {}

    def __init__(
        self,
        config: DetectorConfig,
        parent_output_path: str = None,
        component_id: int = None
    ):
        super().__init__(config, parent_output_path, component_id)

        # Store model class (instantiate in __call__ to avoid loading during validation)
        if config.model not in DETECTOR_REGISTRY:
            raise ValueError(f'Invalid detector model: {config.model}')
        self._model_class = DETECTOR_REGISTRY.get(config.model)

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
        Run object detection on tiles.

        Returns flattened GDF. Pipeline handles merging and object_id assignment.
        """
        
        detector = self._model_class(self.config)

        # Create dataset from tiles
        infer_ds = UnlabeledRasterDataset(
            fold=None,
            root_path=data_state.tiles_path,
            transform=None
        )

        # Run inference
        tiles_paths, boxes, boxes_scores, classes = detector.infer(infer_ds, collate_fn_images)

        # Flatten outputs into GDF
        rows = []
        unique_id = 0
        for i, tile_path in enumerate(tiles_paths):
            for box_geom, score, cls_id in zip(boxes[i], boxes_scores[i], classes[i]):
                rows.append({
                    Col.GEOMETRY: box_geom,
                    Col.TILE_PATH: str(tile_path),
                    Col.DETECTOR_SCORE: score,
                    Col.DETECTOR_CLASS: cls_id,
                    Col.OBJECT_ID: unique_id,
                })
                unique_id += 1

        # Create GDF
        if not rows:
            gdf = gpd.GeoDataFrame(
                columns=list(self.produces_columns),
                crs=None
            )
        else:
            gdf = gpd.GeoDataFrame(rows, geometry=Col.GEOMETRY, crs=None)
            # Ensure geometry is valid
            gdf[Col.GEOMETRY] = gdf[Col.GEOMETRY].buffer(0)
            gdf = gdf[gdf.is_valid & ~gdf.is_empty]

        print(f"DetectorComponent: Generated {len(gdf)} detections.")

        return ComponentResult(
            gdf=gdf,
            produced_columns=self.produces_columns,
            save_gpkg=True,
            gpkg_name_suffix="notaggregated",
            save_coco=True,
            coco_scores_column=Col.DETECTOR_SCORE,
            coco_categories_column=Col.DETECTOR_CLASS,
        )
