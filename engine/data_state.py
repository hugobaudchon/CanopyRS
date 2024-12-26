from dataclasses import dataclass, field
import geopandas as gpd


@dataclass
class DataState:
    imagery_path: str = None
    parent_output_path: str = None
    tiles_path: str = None
    coco_paths: dict = None
    results_gdf: gpd.GeoDataFrame = None
    results_gdf_columns_to_pass: list = field(default_factory=list)   # for tilerizer
    results_gdf_columns_to_delete_on_save = []   # TODO to not save embedding columns etc?? or maybe embeddings should be a separate variable in DataState
