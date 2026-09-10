import json
from pathlib import Path
from typing import List, Set

import geopandas as gpd
from geodataset.aoi import AOIGeneratorConfig, AOIFromPackageConfig
from geodataset.utils import COCOGenerator


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
