from geodataset.aoi import AOIGeneratorConfig, AOIFromPackageConfig
from geodataset.tilerize import RasterPolygonTilerizer, RasterTilerizer, LabeledRasterTilerizer

from engine.components.base import BaseComponent
from engine.config_parsers.tilerizer import TilerizerConfig
from engine.data_state import DataState


class TilerizerComponent(BaseComponent):
    name = 'tilerizer'

    def __init__(self, config: TilerizerConfig, parent_output_path: str, component_id: int):
        super().__init__(config, parent_output_path, component_id)

    def run(self, data_state: DataState) -> DataState:
        aois_config = parse_tilerizer_aoi_config(self.config)
        if self.config.tile_type == 'tile':
            if data_state.results_gdf is not None:
                tilerizer = LabeledRasterTilerizer(
                    raster_path=data_state.imagery_path,
                    labels_path=None,
                    labels_gdf=data_state.results_gdf,
                    output_path=self.output_path,
                    tile_size=self.config.tile_size,
                    tile_overlap=self.config.tile_overlap,
                    aois_config=aois_config,
                    scale_factor=self.config.scale_factor,
                    ground_resolution=self.config.ground_resolution,
                    ignore_black_white_alpha_tiles_threshold=self.config.ignore_black_white_alpha_tiles_threshold,
                    ignore_tiles_without_labels=self.config.ignore_tiles_without_labels,
                    main_label_category_column_name=self.config.main_label_category_column_name,
                    other_labels_attributes_column_names=list(set((data_state.results_gdf_columns_to_pass + self.config.other_labels_attributes_column_names)))
                )

                coco_paths = tilerizer.generate_coco_dataset()
                tiles_path = tilerizer.tiles_path
            else:
                tilerizer = RasterTilerizer(
                    raster_path=data_state.imagery_path,
                    output_path=self.output_path,
                    tile_size=self.config.tile_size,
                    tile_overlap=self.config.tile_overlap,
                    aois_config=aois_config,
                    scale_factor=self.config.scale_factor,
                    ground_resolution=self.config.ground_resolution,
                    ignore_black_white_alpha_tiles_threshold=self.config.ignore_black_white_alpha_tiles_threshold
                )

                tilerizer.generate_tiles()
                coco_paths = None
                tiles_path = tilerizer.tiles_path
        elif self.config.tile_type == 'polygon':
            tilerizer = RasterPolygonTilerizer(
                raster_path=data_state.imagery_path,
                output_path=self.output_path,
                labels_path=None,
                labels_gdf=data_state.results_gdf,
                tile_size=self.config.tile_size,
                use_variable_tile_size=self.config.use_variable_tile_size,
                variable_tile_size_pixel_buffer=self.config.variable_tile_size_pixel_buffer,
                aois_config=aois_config,
                scale_factor=self.config.scale_factor,
                ground_resolution=self.config.ground_resolution,
                main_label_category_column_name=self.config.main_label_category_column_name,
                other_labels_attributes_column_names=list(set((data_state.results_gdf_columns_to_pass + self.config.other_labels_attributes_column_names).unique()))
            )
            coco_paths = tilerizer.generate_coco_dataset()
            tiles_path = tilerizer.tiles_folder_path
        else:
            raise ValueError(f"Invalid tile type: {self.config.tile_type}. Expected 'tile' or 'polygon'.")

        self.config.to_yaml(self.output_path / "tilerizer_config.yaml")

        return self.update_data_state(data_state, tiles_path, coco_paths)

    def update_data_state(self,
                         data_state: DataState,
                         tiles_path: str,
                         coco_paths: dict) -> DataState:
        data_state.tiles_path = tiles_path
        data_state.coco_paths = coco_paths

        return data_state


def parse_tilerizer_aoi_config(config: TilerizerConfig):
    if not config.aoi_config:
        aois_config = AOIGeneratorConfig(
            aoi_type="band",
            aois={'infer': {'percentage': 1.0, 'position': 1}}
        )
    elif config.aoi_config == "generate":
        aois_config = AOIGeneratorConfig(
            aoi_type=config.aoi_type,
            aois=config.aois
        )
    elif config.aoi_config == "package":
        aois_config = AOIFromPackageConfig(
            aois={aoi: path for aoi, path in config.aois.items()}
        )
    else:
        raise ValueError(f"Unsupported value for aoi_config {config.aoi_config}.")

    return aois_config
