from pathlib import Path

import geopandas as gpd

from geodataset.dataset import DetectionLabeledRasterCocoDataset, UnlabeledRasterDataset
from geodataset.utils import GeoPackageNameConvention, TileNameConvention

from canopyrs.engine.components.base import BaseComponent
from canopyrs.engine.config_parsers import SegmenterConfig
from canopyrs.engine.data_state import DataState
from canopyrs.engine.models.registry import SEGMENTER_REGISTRY
from canopyrs.engine.utils import infer_aoi_name, generate_future_coco, object_id_column_name, tile_path_column_name


class SegmenterComponent(BaseComponent):
    name = 'segmenter'

    def __init__(self, config: SegmenterConfig, parent_output_path: str, component_id: int):
        super().__init__(config, parent_output_path, component_id)
        if config.model in SEGMENTER_REGISTRY:
            self.segmenter = SEGMENTER_REGISTRY[config.model](config)
        else:
            raise ValueError(f'Invalid detector model {config.model}')

    def __call__(self, data_state: DataState) -> DataState:
        # Find the tiles (and the COCO file for box prompts if required)
        data_paths = [data_state.tiles_path]
        if self.segmenter.REQUIRES_BOX_PROMPT:
            if not data_state.infer_coco_path:
                # Maybe there is a COCO still being generated in a side process
                data_state.clean_side_processes()
            assert data_state.infer_coco_path is not None, \
                ("The selected Segmenter model requires a COCO file with boxes to prompt. Either input it,"
                 " add a tilerizer before the segmenter, add a detector before the"
                 " segmenter in the pipeline, or choose a segmenter model that doesn't require boxes prompts.")
            data_paths.append(Path(data_state.infer_coco_path).parent)
            dataset = DetectionLabeledRasterCocoDataset(
                fold=infer_aoi_name,
                root_path=data_paths,
                box_padding_percentage=self.config.box_padding_percentage,
                transform=None,
                other_attributes_names_to_pass=[object_id_column_name]
            )
        else:
            dataset = UnlabeledRasterDataset(
                fold=None,  # load all tiles,
                root_path=data_paths,
                transform=None
            )

        # Run inference
        (tiles_paths, tiles_masks_objects_ids, tiles_masks_polygons, tiles_masks_scores) = self.segmenter.infer_on_dataset(dataset)

        # Combine results into a GeoDataFrame
        results_gdf, new_columns = self.combine_as_gdf(
            data_state.infer_gdf, tiles_paths, tiles_masks_polygons, tiles_masks_scores, tiles_masks_objects_ids
        )

        pre_aggregated_gpkg_name = self.get_pre_aggregated_gpkg_name(results_gdf)

        # Save pre-aggregated GeoPackage
        results_gdf.to_file(pre_aggregated_gpkg_name, driver='GPKG')

        # Generate COCO output asynchronously and update the data state
        columns_to_pass = data_state.infer_gdf_columns_to_pass.union(new_columns)

        future_coco = generate_future_coco(
            future_key='infer_coco_path',
            executor=data_state.background_executor,
            component_name=self.name,
            component_id=self.component_id,
            description="Segmenter inference",
            gdf=results_gdf,
            tiles_paths_column=tile_path_column_name,
            polygons_column='geometry',
            scores_column='segmenter_score',
            categories_column=None,
            other_attributes_columns=columns_to_pass,
            output_path=self.output_path,
            use_rle_for_labels=False,
            n_workers=4,
            coco_categories_list=None
        )

        return self.update_data_state(data_state, results_gdf, columns_to_pass, future_coco, pre_aggregated_gpkg_name)

    def get_pre_aggregated_gpkg_name(self, infer_gdf: gpd.GeoDataFrame) -> Path:
        tiles_path = infer_gdf[tile_path_column_name].iat[0]
        product_name, scale_factor, ground_resolution, _, _, _ = TileNameConvention().parse_name(
            Path(tiles_path).name
        )
        pre_aggregated_gpkg_name = GeoPackageNameConvention.create_name(
            product_name=product_name,
            fold=f'{infer_aoi_name}notaggregated',
            scale_factor=scale_factor,
            ground_resolution=ground_resolution
        )
        return self.output_path / pre_aggregated_gpkg_name

    def update_data_state(self,
                          data_state: DataState,
                          results_gdf: gpd.GeoDataFrame,
                          columns_to_pass: set,
                          future_coco: tuple,
                          pre_aggregated_gpkg_name: Path) -> DataState:
        # Register the component folder
        data_state = self.register_outputs_base(data_state)
        data_state.update_infer_gdf(results_gdf)
        data_state.register_output_file(
            self.name,
            self.component_id,
            'pre_aggregated_gpkg',
            pre_aggregated_gpkg_name
        )

        data_state.infer_gdf_columns_to_pass = columns_to_pass
        data_state.side_processes.append(future_coco)
        return data_state

    @staticmethod
    def combine_as_gdf(infer_gdf, tiles_paths, masks, masks_scores, masks_object_ids) -> (gpd.GeoDataFrame, set):
        gdf_items = []
        for i in range(len(tiles_paths)):
            for j in range(len(masks[i])):
                pred_data = {
                    tile_path_column_name: tiles_paths[i],
                    'geometry': masks[i][j],
                    'segmenter_score': masks_scores[i][j],
                    # 'segmenter_class': masks_classes[i][j]    # currently not supported but will be in future
                }

                if masks_object_ids is not None:
                    pred_data[object_id_column_name] = masks_object_ids[i][j]

                gdf_items.append(pred_data)

        gdf = gpd.GeoDataFrame(
            data=gdf_items,
            geometry='geometry',
            crs=None
        )

        new_columns = {'segmenter_score'}   # 'segmenter_class' currently not supported but will be in future
        if masks_object_ids is None:
            # New objects, assign a unique ID to each object detected
            gdf[object_id_column_name] = range(len(gdf))
            new_columns.add(object_id_column_name)
        else:
            # Objects IDs are already assigned, need to match them with existing infer_gdf objects
            gdf.set_index(object_id_column_name, inplace=True)
            infer_gdf.set_index(object_id_column_name, inplace=True)
            # Joining the new columns to the existing infer_gdf, overwriting the existing columns
            other_columns = infer_gdf.columns.difference(gdf.columns)
            gdf = gdf.join(infer_gdf[other_columns], how='left')
            # Resetting the index to keep the object_id_column_name as a column
            gdf.reset_index(inplace=True)

        return gdf, new_columns
