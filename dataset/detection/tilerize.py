from pathlib import Path

import pandas as pd
import geopandas as gpd
from geodataset.aoi import AOIConfig

from geodataset.tilerize import LabeledRasterTilerizer
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union


def combine_gdfs(gdf1, gdf2):
    gdf_combined = gpd.GeoDataFrame(pd.concat([gdf1, gdf2], ignore_index=True))
    merged_polygons = unary_union(gdf_combined.geometry)
    if isinstance(merged_polygons, Polygon):
        gdf_combined = gpd.GeoDataFrame(geometry=[merged_polygons], crs=gdf_combined.crs)
    elif isinstance(merged_polygons, MultiPolygon):
        gdf_combined = gpd.GeoDataFrame(geometry=list(merged_polygons.geoms), crs=gdf_combined.crs)
    else:
        raise Exception

    return gdf_combined


def tilerize_no_overlap(raster_path: str or Path,
                        labels: str or Path or gpd.GeoDataFrame,
                        main_label_category_column_name: str or None,
                        coco_categories_list: list or None,
                        aois_config: AOIConfig,
                        output_path: str or Path):

    tilerizer = LabeledRasterTilerizer(
        raster_path=raster_path,
        labels_path=labels,
        output_path=output_path,
        tile_size=2048,
        tile_overlap=0.0,
        aois_config=aois_config,
        ground_resolution=None,
        scale_factor=1.0,       # we keep the source resolution
        use_rle_for_labels=True,
        min_intersection_ratio=0.4,
        ignore_black_white_alpha_tiles_threshold=0.8,
        ignore_tiles_without_labels=True,
        main_label_category_column_name=main_label_category_column_name,
        coco_categories_list=coco_categories_list,
        coco_n_workers=10
    )

    coco_paths = tilerizer.generate_coco_dataset()

    return coco_paths


def tilerize_with_overlap(raster_path: str or Path,
                          labels: str or Path or gpd.GeoDataFrame,
                          main_label_category_column_name: str or None,
                          coco_categories_list: list or None,
                          aois_config: AOIConfig,
                          output_path: str or Path,
                          ground_resolution: float = None,
                          scale_factor: float = None,
                          tile_size: int = 1024,
                          tile_overlap: float = 0.5):

    tilerizer = LabeledRasterTilerizer(
        raster_path=raster_path,
        labels_path=labels,
        output_path=output_path,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        aois_config=aois_config,
        ground_resolution=ground_resolution,
        scale_factor=scale_factor,
        use_rle_for_labels=True,
        min_intersection_ratio=0.4,
        ignore_black_white_alpha_tiles_threshold=0.8,
        ignore_tiles_without_labels=True,
        main_label_category_column_name=main_label_category_column_name,
        coco_categories_list=coco_categories_list if coco_categories_list is not None else [{'id': 1, 'name': 'tree', 'supercategory': ''}],
        coco_n_workers=10
    )

    coco_outputs = tilerizer.generate_coco_dataset()
    tiles_path = tilerizer.tiles_path

    return coco_outputs, tiles_path
