from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Set

import geopandas as gpd
from geodataset.aoi import AOIGeneratorConfig, AOIFromPackageConfig
from geodataset.utils import COCOGenerator, TileNameConvention, CocoNameConvention

infer_aoi_name = 'infer'
ground_truth_aoi_name = 'groundtruth'


def get_component_folder_name(component_id: int, component_name: str) -> str:
    component_folder = f"{component_id}_{component_name}"
    return component_folder


executor = ProcessPoolExecutor(max_workers=1)


def generate_future_coco(
    future_key: str,
    component_name: str,
    component_id: int,
    description: str,
    gdf: gpd.GeoDataFrame,
    tiles_paths_column: str,
    polygons_column: str,
    scores_column: str or None,
    categories_column: str or None,
    other_attributes_columns: Set[str] or None,
    output_path: Path,
    use_rle_for_labels: bool,
    n_workers: int,
    coco_categories_list: List[dict] or None
) -> tuple:
    """
    Starts a side process for generating the COCO file, this way the main process isn't blocked in the meantime.

    Parameters
    ----------
    future_key : str
        The key to be used to store the future result in the data state.
    description : str
        Description of the COCO file.
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing the data to be used for generating the COCO file.
    tiles_paths_column : str
        Name of the column containing the paths to the tiles.
    polygons_column : str
        Name of the column containing the polygons.
    scores_column : str or None
        Name of the column containing the scores.
    categories_column : str or None
        Name of the column containing the categories.
    other_attributes_columns : Set[str] or None
        List of names of the columns containing other attributes.
    output_path : Path
        Path to the output directory.
    use_rle_for_labels : bool
        Whether to use RLE encoding for the labels.
    n_workers : int
        Number of workers to use for the process.
    coco_categories_list : List[dict] or None
        List of categories to be used in the COCO file.

    Returns
    -------
    tuple
        Tuple containing the future key and the future COCO file path: (future_key, future_coco_path).
    """

    print('Starting side process for generating COCO file...')

    product_name, scale_factor, ground_resolution, _, _, aoi = TileNameConvention().parse_name(
        Path(gdf[tiles_paths_column].iloc[0]).name
    )
    coco_output_name = CocoNameConvention().create_name(
        product_name=product_name,
        fold=aoi,
        scale_factor=scale_factor,
        ground_resolution=ground_resolution
    )

    coco_output_path = output_path / coco_output_name

    future_coco_process = executor.submit(
        generate_coco,
        description=description,
        gdf=gdf,
        tiles_paths_column=tiles_paths_column,
        polygons_column=polygons_column,
        scores_column=scores_column,
        categories_column=categories_column,
        other_attributes_columns=other_attributes_columns,
        coco_output_path=coco_output_path,
        use_rle_for_labels=use_rle_for_labels,
        n_workers=n_workers,
        coco_categories_list=coco_categories_list
    )

    future_coco = (
        future_key, future_coco_process,
        {
            'component_name': component_name,
            'component_id': component_id,
            'file_type': 'coco',
            'expected_path': str(coco_output_path)  # Include the expected path directly
        }
     )

    return future_coco


def generate_coco(
    description: str,
    gdf: gpd.GeoDataFrame,
    tiles_paths_column: str,
    polygons_column: str,
    scores_column: str or None,
    categories_column: str or None,
    other_attributes_columns: Set[str] or None,
    coco_output_path: Path,
    use_rle_for_labels: bool,
    n_workers: int,
    coco_categories_list: List[dict] or None
) -> Path:

    """
    Generates a COCO file from a GeoDataFrame.

    Parameters
    ----------
    description : str
        Description of the COCO file.
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing the data to be used for generating the COCO file.
    tiles_paths_column : str
        Name of the column containing the paths to the tiles.
    polygons_column : str
        Name of the column containing the polygons.
    scores_column : str or None
        Name of the column containing the scores.
    categories_column : str or None
        Name of the column containing the categories.
    other_attributes_columns : Set[str] or None
        List of names of the columns containing other attributes.
    coco_output_path : Path
        Path to the COCO output path.
    use_rle_for_labels : bool
        Whether to use RLE encoding for the labels.
    n_workers : int
        Number of workers to use for the process.
    coco_categories_list : List[dict] or None
        List of categories to be used in the COCO file.

    Returns
    -------
    Path
        Path to the generated COCO file.
    """

    COCOGenerator.from_gdf(
        description=description,
        gdf=gdf,
        tiles_paths_column=tiles_paths_column,
        polygons_column=polygons_column,
        scores_column=scores_column,
        categories_column=categories_column,
        other_attributes_columns=list(other_attributes_columns),
        output_path=coco_output_path,
        use_rle_for_labels=use_rle_for_labels,
        n_workers=n_workers,
        coco_categories_list=coco_categories_list
    ).generate_coco()

    print('COCO file generated!')

    return coco_output_path


def parse_tilerizer_aoi_config(aoi_config: str or None,
                               aoi_type: str or None,
                               aois: dict or None):
    if not aoi_config:
        aois_config = AOIGeneratorConfig(
            aoi_type="band",
            aois={'infer': {'percentage': 1.0, 'position': 1}}
        )
    elif aoi_config == "generate":
        aois_config = AOIGeneratorConfig(
            aoi_type=aoi_type,
            aois=aois
        )
    elif aoi_config == "package":
        aois_config = AOIFromPackageConfig(
            aois={aoi: path for aoi, path in aois.items()}
        )
    else:
        raise ValueError(f"Unsupported value for aoi_config {aoi_config}.")

    return aois_config


def green_print(text: str, add_return: bool = False):
    add_return_str = '\n' if add_return else ''
    print(f'{add_return_str}\033[32m ------ {text} ------ \033[0m')


def init_spawn_method():
    """
    Initializes the spawn method for the ProcessPoolExecutor.
    """
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # The start method was already set
        pass

