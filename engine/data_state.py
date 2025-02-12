from dataclasses import dataclass, field

import geopandas as gpd


@dataclass
class DataState:
    imagery_path: str = None
    parent_output_path: str = None

    tiles_path: str = None
    tiles_names: list = None

    infer_coco_path: str = None
    infer_gdf: gpd.GeoDataFrame = None
    infer_gdf_columns_to_pass: list = field(default_factory=list)   # for tilerizer
    infer_gdf_columns_to_delete_on_save = []

    ground_truth_coco_path: str = None
    ground_truth_gdf: gpd.GeoDataFrame = None
    ground_truth_gdf_columns_to_pass: list = field(default_factory=list)

    side_processes = []