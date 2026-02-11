import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Set

import geopandas as gpd
from geodataset.aoi import AOIGeneratorConfig, AOIFromPackageConfig
from geodataset.utils import COCOGenerator, TileNameConvention, CocoNameConvention

infer_aoi_name = 'infer'
object_id_column_name = 'canopyrs_object_id'
tile_path_column_name = 'tile_path'


def get_component_folder_name(component_id: int, component_name: str) -> str:
    component_folder = f"{component_id}_{component_name}"
    return component_folder


def parse_product_name(tile_path: str):
    try:
        product_name, scale_factor, ground_resolution, _, _, aoi = TileNameConvention().parse_name(
            Path(tile_path).name
        )
    except ValueError:
        # input is probably images not tiled with geodataset
        product_name = 'images'
        scale_factor = 1.0
        ground_resolution = None
        aoi = infer_aoi_name

    return product_name, scale_factor, ground_resolution, aoi


def generate_future_coco(
    future_key: str,
    executor: ProcessPoolExecutor,
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

    product_name, scale_factor, ground_resolution, _ = parse_product_name(gdf[tiles_paths_column].iloc[0])

    coco_output_name = CocoNameConvention().create_name(
        product_name=product_name,
        fold=infer_aoi_name,
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

    # Ensure paths in the dataframe are JSON-serializable (PosixPath would break json.dump).
    gdf = gdf.copy()
    gdf[tiles_paths_column] = gdf[tiles_paths_column].apply(lambda v: str(v) if isinstance(v, Path) else v)
    if other_attributes_columns:
        for col in other_attributes_columns:
            if col in gdf.columns:
                gdf[col] = gdf[col].apply(lambda v: str(v) if isinstance(v, Path) else v)

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
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError as e:
        # The start method was already set
        print(f"Error while setting multiprocessing start method: {e}")
        pass

def merge_coco_jsons(json_files: list[str or Path], output_file: str or Path):
    merged = {
        "images": [],
        "annotations": [],
        "categories": None  # assuming all files have the same categories
    }

    new_image_id = 0
    new_annotation_id = 0

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # For the first file, grab the categories
        if merged["categories"] is None and "categories" in data:
            merged["categories"] = data["categories"]

        # Create a mapping from old image ids to new image ids
        id_mapping = {}
        for image in data["images"]:
            old_id = image["id"]
            image["id"] = new_image_id
            id_mapping[old_id] = new_image_id
            merged["images"].append(image)
            new_image_id += 1

        # Update annotations: assign new annotation ids and update image_id
        for ann in data["annotations"]:
            ann["id"] = new_annotation_id
            if ann["image_id"] in id_mapping:
                ann["image_id"] = id_mapping[ann["image_id"]]
            else:
                raise ValueError(f"Annotation references missing image id: {ann['image_id']}")
            merged["annotations"].append(ann)
            new_annotation_id += 1

    # Write the merged result to the output file
    with open(output_file, "w") as f:
        json.dump(merged, f, indent=2)
