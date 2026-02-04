"""
SegmenterComponent with simplified architecture.

Single __call__() method returns flattened GDF with new geometries (masks).
Pipeline handles merging with existing GDF (to preserve detector columns) and I/O.
"""

from pathlib import Path
from typing import Set

import geopandas as gpd

from geodataset.dataset import DetectionLabeledRasterCocoDataset, UnlabeledRasterDataset

from canopyrs.engine.constants import Col, StateKey, INFER_AOI_NAME
from canopyrs.engine.components.base import BaseComponent, ComponentResult, validate_requirements
from canopyrs.engine.config_parsers import SegmenterConfig
from canopyrs.engine.data_state import DataState
from canopyrs.engine.models.registry import SEGMENTER_REGISTRY


class SegmenterComponent(BaseComponent):
    """
    Runs instance segmentation on image tiles.

    Modes:
        - With box prompts: Requires infer_coco_path with detection boxes
        - Without box prompts: Runs segmentation on full tiles

    Requirements:
        - tiles_path: Directory containing tiles
        - infer_coco_path + infer_gdf: If model requires box prompts

    Produces:
        - infer_gdf: GeoDataFrame with segmented polygons
        - Columns: segmenter_score (+ preserves detector columns if present)
    """

    name = 'segmenter'

    BASE_REQUIRES_STATE = {StateKey.TILES_PATH}
    BASE_REQUIRES_COLUMNS: Set[str] = set()

    BASE_PRODUCES_STATE = {StateKey.INFER_GDF, StateKey.INFER_COCO_PATH}
    BASE_PRODUCES_COLUMNS = {Col.GEOMETRY, Col.OBJECT_ID, Col.TILE_PATH, Col.SEGMENTER_SCORE}

    BASE_STATE_HINTS = {
        StateKey.TILES_PATH: "Segmenter needs tiles to process. Add a tilerizer before segmenter.",
        StateKey.INFER_COCO_PATH: "This segmenter model requires box prompts from a COCO file.",
        StateKey.INFER_GDF: "This segmenter model requires a GeoDataFrame with detection boxes.",
    }

    BASE_COLUMN_HINTS = {
        Col.OBJECT_ID: "Segmenter needs object IDs to associate masks with detections.",
    }

    def __init__(
        self,
        config: SegmenterConfig,
        parent_output_path: str = None,
        component_id: int = None
    ):
        super().__init__(config, parent_output_path, component_id)

        # Get model class (without instantiating) to check requirements
        if config.model not in SEGMENTER_REGISTRY:
            raise ValueError(f'Invalid segmenter model: {config.model}')
        self._model_class = SEGMENTER_REGISTRY.get(config.model)

        # Set base requirements
        self.requires_state = set(self.BASE_REQUIRES_STATE)
        self.requires_columns = set(self.BASE_REQUIRES_COLUMNS)
        self.produces_state = set(self.BASE_PRODUCES_STATE)
        self.produces_columns = set(self.BASE_PRODUCES_COLUMNS)

        # Set hints
        self.state_hints = dict(self.BASE_STATE_HINTS)
        self.column_hints = dict(self.BASE_COLUMN_HINTS)

        # Add model-specific requirements
        if self._model_class.REQUIRES_BOX_PROMPT:
            self.requires_state.add(StateKey.INFER_COCO_PATH)
            self.state_hints[StateKey.INFER_COCO_PATH] = (
                f"The '{config.model}' segmenter requires box prompts. "
                f"Add a detector before segmenter."
            )

    @validate_requirements
    def __call__(self, data_state: DataState) -> ComponentResult:
        """
        Run instance segmentation on tiles.

        Returns flattened GDF with new geometries (masks).
        Pipeline handles merging with existing GDF (to preserve detector columns).
        """
        
        segmenter = self._model_class(self.config)

        # Create appropriate dataset
        data_paths = [data_state.tiles_path]

        if segmenter.REQUIRES_BOX_PROMPT:
            data_paths.append(Path(data_state.infer_coco_path).parent)
            dataset = DetectionLabeledRasterCocoDataset(
                fold=INFER_AOI_NAME,
                root_path=data_paths,
                box_padding_percentage=self.config.box_padding_percentage,
                transform=None,
                other_attributes_names_to_pass=[Col.OBJECT_ID]
            )
        else:
            dataset = UnlabeledRasterDataset(
                fold=None,
                root_path=data_paths,
                transform=None
            )

        # Run inference
        tiles_paths, tiles_masks_objects_ids, tiles_masks_polygons, tiles_masks_scores = \
            segmenter.infer_on_dataset(dataset)

        # Flatten outputs into GDF
        rows = []
        unique_id = 0
        for i in range(len(tiles_paths)):
            for j in range(len(tiles_masks_polygons[i])):
                row = {
                    Col.TILE_PATH: tiles_paths[i],
                    Col.GEOMETRY: tiles_masks_polygons[i][j],
                    Col.SEGMENTER_SCORE: tiles_masks_scores[i][j],
                }
                # Include object_id if available (from detector), or assign new unique id otherwise
                if tiles_masks_objects_ids is not None:
                    row[Col.OBJECT_ID] = tiles_masks_objects_ids[i][j]
                else:
                    row[Col.OBJECT_ID] = unique_id
                    unique_id += 1
                rows.append(row)

        # Create GDF with new geometries (masks)
        gdf = gpd.GeoDataFrame(rows, geometry=Col.GEOMETRY, crs=None) if rows else gpd.GeoDataFrame(
            columns=self.produces_columns,
            crs=None
        )

        print(f"SegmenterComponent: Generated {len(gdf)} masks.")

        return ComponentResult(
            gdf=gdf,
            produced_columns=self.produces_columns,
            save_gpkg=True,
            gpkg_name_suffix="notaggregated",
            save_coco=True,
            coco_scores_column=Col.SEGMENTER_SCORE,
            coco_categories_column=None,
        )
