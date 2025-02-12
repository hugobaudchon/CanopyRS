from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List

from geodataset.aoi import AOIGeneratorConfig, AOIFromPackageConfig
from geodataset.utils import COCOGenerator, TileNameConvention, CocoNameConvention
from shapely.geometry import Polygon


infer_aoi_name = 'infer'
ground_truth_aoi_name = 'groundtruth'


def generate_future_coco(
        future_key: str,
        description: str,
        tiles_paths: List[Path or str],
        tile_names_order_reference: List[str],
        polygons: List[List[Polygon]],
        scores: List[List[float]],
        categories: List[List[int]] or None,
        other_attributes: dict or None,
        output_path: Path,
        use_rle_for_labels: bool = True,
        n_workers: int = 2,
        coco_categories_list: List[dict] = None
):
    """
    Starts a side process for generating the COCO file, this way the main process isn't blocked in the meantime.
    """

    print('Starting side process for generating COCO file...')

    with ProcessPoolExecutor(max_workers=1) as executor:
        future_coco = executor.submit(
            generate_coco,
            description=description,
            tiles_paths=tiles_paths,
            tile_names_order_reference=tile_names_order_reference,
            polygons=polygons,
            scores=scores,
            categories=categories,
            other_attributes=other_attributes,
            output_path=output_path,
            use_rle_for_labels=use_rle_for_labels,
            n_workers=n_workers,
            coco_categories_list=coco_categories_list
        )

    return tuple([future_key, future_coco])




def generate_coco(
    description: str,
    tiles_paths: List[Path or str],
    tile_names_order_reference: List[str],
    polygons: List[List[Polygon]],
    scores: List[List[float]],
    categories: List[List[int]] or None,
    other_attributes: dict or None,
    output_path: Path,
    use_rle_for_labels: bool = True,
    n_workers: int = 2,
    coco_categories_list: List[dict] = None
):

    product_name, scale_factor, ground_resolution, _, _, aoi = TileNameConvention().parse_name(
        Path(tiles_paths[0]).name
    )
    coco_output_name = CocoNameConvention().create_name(
        product_name=product_name,
        fold=aoi,
        scale_factor=scale_factor,
        ground_resolution=ground_resolution
    )

    output_path = output_path / coco_output_name

    tiles_names = [Path(tile_path).name for tile_path in tiles_paths]

    # order all the results by the order of tile_names_order_reference
    ordered_polygons = []
    ordered_scores = []
    ordered_categories = [] if categories else None
    ordered_other_attributes = {key: [] for key in other_attributes} if other_attributes else {}
    ordered_tile_paths = []
    for tile_name in tile_names_order_reference:
        if tile_name not in tiles_names:
            tile_index = None
            tile_path = Path(tiles_paths[0]).parent / tile_name
        else:
            tile_index = tiles_names.index(tile_name)
            tile_path = tiles_paths[tile_index]

        ordered_polygons.append(polygons[tile_index] if tile_index is not None else [])
        ordered_scores.append(scores[tile_index] if tile_index is not None else [])
        if categories:
            ordered_categories.append(categories[tile_index] if tile_index is not None else [])
        if other_attributes:
            for key in other_attributes:
                ordered_other_attributes[key].append(other_attributes[key][tile_index] if tile_index is not None else [])
        ordered_tile_paths.append(tile_path)

    COCOGenerator(
        description=description,
        tiles_paths=ordered_tile_paths,
        polygons=ordered_polygons,
        scores=ordered_scores,
        categories=ordered_categories,
        other_attributes=ordered_other_attributes,
        output_path=output_path,
        use_rle_for_labels=use_rle_for_labels,
        n_workers=n_workers,
        coco_categories_list=coco_categories_list
    ).generate_coco()

    print('COCO file generated!')

    return output_path


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

def green_print(text: str):
    print(f'\n\033[32m ------ {text} ------ \033[0m')