from abc import ABC
from pathlib import Path
from typing import Dict, Type, Union, Iterator, Tuple, Optional

from geodataset.utils import CocoNameConvention
from huggingface_hub import hf_hub_download

from data.detection.hf_tools import HFDatasetTools


class BasePreprocessedDataset(ABC):
    dataset_name: str = None
    license: str = None
    ground_resolution: float = None
    scale_factor: float = None

    train_tile_size: int = None
    train_n_tiles: int = None
    train_n_annotations: int = None

    valid_tile_size: int = None
    valid_n_tiles: int = None
    valid_n_annotations: int = None

    test_tile_size: int = None
    test_n_tiles: int = None
    test_n_annotations: int = None

    tile_level_eval_maxDets: int = None

    locations: list[str] = None
    products_per_location: dict[str, list[str]] = None
    raster_level_eval_inputs: dict[str, dict[str, dict[str, str]]] = None

    def download_and_extract(self, root_output_path: str, folds: list[str], hf_root='CanopyRS'):
        # Download the dataset from Hugging Face, and extract images and annotations from the parquet files
        HFDatasetTools.download_and_extract_huggingface_dataset(
            hf_dataset_name=f'{hf_root}/{self.dataset_name}',
            root_output_path=root_output_path
        )

        # Download the raster level evaluation inputs (.gpkg files)
        if self.raster_level_eval_inputs is not None:
            for fold in folds:
                if fold not in self.raster_level_eval_inputs:
                    continue
                for raster in self.raster_level_eval_inputs[fold]:
                    hf_hub_download(
                        repo_id=f'{hf_root}/{self.dataset_name}',
                        repo_type="dataset",
                        revision='gpkg',
                        filename=self.raster_level_eval_inputs[fold][raster]['ground_truth_gpkg']
                    )
                    hf_hub_download(
                        repo_id=f'{hf_root}/{self.dataset_name}',
                        repo_type="dataset",
                        revision='gpkg',
                        filename=self.raster_level_eval_inputs[fold][raster]['aoi_gpkg']
                    )

        self.verify_dataset(root_output_path=root_output_path, folds=folds)

    def verify_dataset(self, root_output_path: str or Path, folds: list[str]):
        root_output_path = Path(root_output_path)

        # Verify the dataset structure
        for fold in folds:
            for location in self.locations:
                for product in self.products_per_location[location]:
                    assert (root_output_path / location / product).exists(), \
                        f"Product {product} not found in {location} under {fold} fold."

    def iter_fold(
            self,
            root_output_path: Union[str, Path],
            fold: str,
            hf_root: str = "CanopyRS"
    ) -> Iterator[Tuple[
        str,  # location
        str,  # product_name
        Path,  # tile_dir
        Optional[str],  # aoi_gpkg path (or None)
        Optional[str],  # ground_truth_gpkg path (or None)
        str,  # ground_truth_coco path
    ]]:
        """
        Loop over every (location, product) in this dataset for `fold`,
        yielding:
          - location
          - product_name
          - Path to tiles for that fold
          - local path to AOI .gpkg (downloaded via hf_hub_download), or None
          - local path to ground-truth .gpkg, or None
          - local path to ground-truth COCO .json file
        """
        root = Path(root_output_path)
        repo_id = f"{hf_root}/{self.dataset_name}"

        for loc in self.locations:
            for prod in self.products_per_location[loc]:
                if prod not in self.raster_level_eval_inputs[fold]:
                    continue
                # 1) where are the tiles on disk?
                tile_dir = root / loc / prod / "tiles" / fold

                gt_coco_path = CocoNameConvention.create_name(
                    product_name=prod,
                    fold=fold,
                    ground_resolution=self.ground_resolution
                )

                gt_coco_path = root / loc / prod / gt_coco_path

                aoi_gpkg_path: Optional[str] = None
                gt_gpkg_path: Optional[str] = None
                # 2) if we have raster inputs for this fold+product, grab them
                if (
                        self.raster_level_eval_inputs
                        and fold in self.raster_level_eval_inputs
                        and prod in self.raster_level_eval_inputs[fold]
                ):
                    info = self.raster_level_eval_inputs[fold][prod]
                    gt_gpkg_path = hf_hub_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        revision="gpkg",
                        filename=info["ground_truth_gpkg"]
                    )
                    aoi_gpkg_path = hf_hub_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        revision="gpkg",
                        filename=info["aoi_gpkg"]
                    )

                yield loc, prod, tile_dir, aoi_gpkg_path, gt_gpkg_path, gt_coco_path


DATASET_REGISTRY: Dict[str, Type[BasePreprocessedDataset]] = {}


def register_dataset(cls: Type[BasePreprocessedDataset]) -> Type[BasePreprocessedDataset]:
    name = getattr(cls, "dataset_name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must define a dataset_name")
    DATASET_REGISTRY[name] = cls
    return cls


@register_dataset
class CanopyDataset(BasePreprocessedDataset):
    dataset_name = 'canopy3'
    license = 'CC-BY-NC-4.0'
    ground_resolution = 0.045   # 4.5cm/pixel
    scale_factor = None

    train_tile_size = 3555
    train_n_tiles = 585
    train_n_annotations = 232071

    valid_tile_size = 1777
    valid_n_tiles = 387
    valid_n_annotations = 38651

    test_tile_size = 1777
    test_n_tiles = 1477
    test_n_annotations = 161188

    tile_level_eval_maxDets = 400  # at tiles of 80x80m, there are in average 100 trees per tiles, so need to set this from 100 to 400 for COCOEval to correctly assess model confidence.

    locations = [
        'brazil_zf2',
        'ecuador_tiputini',
        'panama_aguasalud'
    ]

    products_per_location = {
        'brazil_zf2': [
            '20240130_zf2quad_m3m_rgb',
            '20240130_zf2tower_m3m_rgb',
            '20240130_zf2transectew_m3m_rgb',
            '20240131_zf2campirana_m3m_rgb'
        ],
        'ecuador_tiputini': [
            '20170810_transectotoni_mavicpro_rgb',
            '20230525_tbslake_m3e_rgb',
            '20230911_sanitower_mini2_rgb',
            '20231018_inundated_m3e_rgb',
            '20231018_pantano_m3e_rgb',
            '20231018_terrafirme_m3e_rgb'
        ],
        'panama_aguasalud': [
            '20231207_asnortheast_amsunclouds_m3m_rgb',
            '20231207_asnorthnorth_pmclouds_m3m_rgb',
            '20231208_asforestnorthe2_m3m_rgb',
            '20231208_asforestsouth2_m3m_rgb'
        ]
    }

    raster_level_eval_inputs = {
        'valid': {
            '20240130_zf2quad_m3m_rgb': {
                'ground_truth_gpkg': '20240130_zf2quad_m3m_labels_boxes.gpkg',
                'aoi_gpkg': '20240130_zf2quad_m3m_labels_boxes_aoi_valid.gpkg',
            },
            '20231018_inundated_m3e_rgb': {
                'ground_truth_gpkg': '20231018_inundated_m3e_labels_boxes.gpkg',
                'aoi_gpkg': '20231018_inundated_m3e_labels_boxes_aoi_valid.gpkg',
            },
            '20231207_asnortheast_amsunclouds_m3m_rgb': {
                'ground_truth_gpkg': '20231207_asnortheast_amsunclouds_m3m_labels_boxes.gpkg',
                'aoi_gpkg': '20231207_asnortheast_amsunclouds_m3m_labels_boxes_aoi_valid.gpkg'
            },
            '20231208_asforestnorthe2_m3m_rgb':{
                'ground_truth_gpkg': '20231208_asforestnorthe2_m3m_labels_boxes.gpkg',
                'aoi_gpkg': '20231208_asforestnorthe2_m3m_labels_boxes_aoi_valid.gpkg'
            }
        },
        'test': {
            '20240130_zf2tower_m3m_rgb': {
                'ground_truth_gpkg': '20240130_zf2tower_m3m_labels_boxes.gpkg',
                'aoi_gpkg': '20240130_zf2tower_m3m_labels_boxes_aoi_test.gpkg',
            },
            '20230525_tbslake_m3e_rgb': {
                'ground_truth_gpkg': '20230525_tbslake_m3e_labels_boxes.gpkg',
                'aoi_gpkg': '20230525_tbslake_m3e_labels_boxes_aoi_test.gpkg'
            },
            '20231018_inundated_m3e_rgb': {
                'ground_truth_gpkg': '20231018_inundated_m3e_labels_boxes.gpkg',
                'aoi_gpkg': '20231018_inundated_m3e_labels_boxes_aoi_test.gpkg'
            },
            '20231207_asnortheast_amsunclouds_m3m_rgb': {
                'ground_truth_gpkg': '20231207_asnortheast_amsunclouds_m3m_labels_boxes.gpkg',
                'aoi_gpkg': '20231207_asnortheast_amsunclouds_m3m_labels_boxes_aoi_test.gpkg'
            },
            '20231208_asforestnorthe2_m3m_rgb': {
                'ground_truth_gpkg': '20231208_asforestnorthe2_m3m_labels_boxes.gpkg',
                'aoi_gpkg': '20231208_asforestnorthe2_m3m_labels_boxes_aoi_test.gpkg'
            }
        }
    }
