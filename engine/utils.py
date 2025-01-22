from pathlib import Path

from geodataset.aoi import AOIGeneratorConfig, AOIFromPackageConfig
from geodataset.utils import COCOGenerator

from engine.data_state import DataState


infer_aoi_name = 'infer'
ground_truth_aoi_name = 'groundtruth'


def generate_coco(description,
                  tiles_paths,
                  tile_names_order_reference,
                  polygons,
                  scores,
                  categories,
                  other_attributes,
                  output_path,
                  use_rle_for_labels,
                  n_workers,
                  coco_categories_list):

    print('Starting side process for generating COCO...')

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

    print('Side process for generating COCO finished!')

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