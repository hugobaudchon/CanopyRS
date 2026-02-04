from abc import ABC
from pathlib import Path
from typing import Dict, Type, Union, Iterator, Tuple, Optional

from geodataset.utils import CocoNameConvention
from huggingface_hub import hf_hub_download

from canopyrs.data.detection.hf_tools import HFDatasetTools


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
            root_output_path=root_output_path,
            cleanup_hf_cache=True,
            cleanup_hf_cache_temp_dir=f"{root_output_path}/hf_cache_temp"
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
        for location in self.locations:
            for product in self.products_per_location[location]:
                assert (root_output_path / location / product).exists(), \
                    f"Product {product} not found in {location}."

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

                try:
                    gt_coco_path = CocoNameConvention.create_name(
                        product_name=prod,
                        fold=fold,
                        ground_resolution=self.ground_resolution
                    )
                    with (root / loc / prod / gt_coco_path).open("rb") as f:
                        pass
                except FileNotFoundError:
                    gt_coco_path = CocoNameConvention.create_name(
                        product_name=prod,
                        fold=fold,
                        scale_factor=self.scale_factor
                    )

                gt_coco_path = root / loc / prod / gt_coco_path

                aoi_gpkg_path: Optional[str] = None
                gt_gpkg_path: Optional[str] = None
                # 2) if we have raster inputs for this fold+product, grab them
                if (
                        self.raster_level_eval_inputs
                        and fold in self.raster_level_eval_inputs
                        and prod in self.raster_level_eval_inputs[fold]
                        and "ground_truth_gpkg" in self.raster_level_eval_inputs[fold][prod]
                        and "aoi_gpkg" in self.raster_level_eval_inputs[fold][prod]
                ):
                    
                    info = self.raster_level_eval_inputs[fold][prod]
                    if Path(root / loc / prod / info["ground_truth_gpkg"]).exists():
                        gt_gpkg_path = str(root / loc / prod / info["ground_truth_gpkg"])
                    else:
                        gt_gpkg_path = hf_hub_download(
                            repo_id=repo_id,
                            repo_type="dataset",
                            revision="gpkg",
                            filename=info["ground_truth_gpkg"]
                        )
                    
                    if Path(root / loc / prod / info["aoi_gpkg"]).exists():
                        aoi_gpkg_path = str(root / loc / prod / info["aoi_gpkg"])
                    else:
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
class SelvaBoxDataset(BasePreprocessedDataset):
    dataset_name = 'SelvaBox'
    license = 'CC-BY-4.0'
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


@register_dataset
class BCI50haDataset(BasePreprocessedDataset):
    dataset_name = 'BCI50ha'
    license = 'CC-BY-4.0'
    ground_resolution = 0.045
    scale_factor = None

    train_tile_size = None
    train_n_tiles = None
    train_n_annotations = None

    valid_tile_size = None
    valid_n_tiles = None
    valid_n_annotations = None

    test_tile_size = 1777
    test_n_tiles = 2706
    test_n_annotations = 75629

    tile_level_eval_maxDets = 100

    locations = [
        'panama_bci50ha'
    ]

    products_per_location = {
        'panama_bci50ha': [
            'bci_50ha_2020_08_01_crownmap_raw',
            'bci_50ha_2022_09_29_crownmap_raw',
        ]
    }

    raster_level_eval_inputs = {
        'test': {
            'bci_50ha_2020_08_01_crownmap_raw': {
                'ground_truth_gpkg': 'BCI_50ha_2020_08_01_crownmap_improved.gpkg',
                'aoi_gpkg': '20200801_aoi_test_without_holes.gpkg'
            },
            'bci_50ha_2022_09_29_crownmap_raw': {
                'ground_truth_gpkg': 'BCI_50ha_2022_09_29_crownmap_improved.gpkg',
                'aoi_gpkg': '20220929_aoi_test_without_holes.gpkg'
            }
        }
    }


@register_dataset
class QuebecTreesDataset(BasePreprocessedDataset):
    dataset_name = 'QuebecTrees'
    license = 'CC-BY-4.0'
    ground_resolution = 0.03
    scale_factor = None

    train_tile_size = 3333
    train_n_tiles = 148
    train_n_annotations = 53289

    valid_tile_size = 1666
    valid_n_tiles = 139
    valid_n_annotations = 15155

    test_tile_size = 1666
    test_n_tiles = 168
    test_n_annotations = 22818

    tile_level_eval_maxDets = 400

    locations = [
        'quebec_trees'
    ]

    products_per_location = {
        'quebec_trees': [
            '20210902_sblz1_p4rtk_rgb',
            '20210902_sblz2_p4rtk_rgb',
            '20210902_sblz3_p4rtk_rgb',
        ]
    }

    raster_level_eval_inputs = {
        'valid': {
            '20210902_sblz1_p4rtk_rgb': {
                'ground_truth_gpkg': '20210902_sblz1_p4rtk_labels_masks.gpkg',
                'aoi_gpkg': '20210902_sblz1_p4rtk_labels_masks_aoi_valid.gpkg'
            },
        },
        'test': {
            '20210902_sblz1_p4rtk_rgb': {
                'ground_truth_gpkg': '20210902_sblz1_p4rtk_labels_masks.gpkg',
                'aoi_gpkg': '20210902_sblz1_p4rtk_labels_masks_aoi_test.gpkg'
            },
            '20210902_sblz3_p4rtk_rgb': {
                'ground_truth_gpkg': '20210902_sblz3_p4rtk_labels_masks.gpkg',
                'aoi_gpkg': '20210902_sblz3_p4rtk_labels_masks_aoi_test.gpkg'
            }
        }
    }


@register_dataset
class NeonTreeEvaluationDataset(BasePreprocessedDataset):
    dataset_name = 'NeonTreeEvaluation'
    license = 'CC-BY-4.0'
    ground_resolution = 0.1
    scale_factor = 1.0

    train_tile_size = 1200
    train_n_tiles = 912
    train_n_annotations = 61812

    valid_tile_size = 400
    valid_n_tiles = 934
    valid_n_annotations = 11580

    test_tile_size = 400
    test_n_tiles = 194
    test_n_annotations = 6634

    tile_level_eval_maxDets = 100

    locations = [
        'unitedstates_neon'
    ]

    products_per_location = {
        'unitedstates_neon': [
            '2018_bart_4_322000_4882000_image_crop',
            '2018_harv_5_733000_4698000_image_crop',
            '2018_jerc_4_742000_3451000_image_crop',
            '2018_mlbs_3_541000_4140000_image_crop',
            '2018_mlbs_3_541000_4140000_image_crop2',
            '2018_niwo_2_450000_4426000_image_crop',
            '2018_osbs_4_405000_3286000_image',
            '2018_sjer_3_258000_4106000_image',
            '2018_sjer_3_259000_4110000_image',
            '2018_teak_3_315000_4094000_image_crop',
            '2019_dela_5_423000_3601000_image_crop',
            '2019_leno_5_383000_3523000_image_crop',
            '2019_onaq_2_367000_4449000_image_crop',
            '2019_osbs_5_405000_3287000_image_crop',
            '2019_osbs_5_405000_3287000_image_crop2',
            '2019_sjer_4_251000_4103000_image',
            '2019_tool_3_403000_7617000_image',
            'NeonTreeEvaluation_Test'
        ]
    }

    raster_level_eval_inputs = {
        'valid': {
            '2018_bart_4_322000_4882000_image_crop': {},
            '2018_harv_5_733000_4698000_image_crop': {},
            '2018_jerc_4_742000_3451000_image_crop': {},
            '2018_mlbs_3_541000_4140000_image_crop': {},
            '2018_mlbs_3_541000_4140000_image_crop2': {},
            '2018_niwo_2_450000_4426000_image_crop': {},
            '2018_osbs_4_405000_3286000_image': {},
            '2018_sjer_3_258000_4106000_image': {},
            '2018_sjer_3_259000_4110000_image': {},
            '2018_teak_3_315000_4094000_image_crop': {},
            '2019_dela_5_423000_3601000_image_crop': {},
            '2019_leno_5_383000_3523000_image_crop': {},
            '2019_onaq_2_367000_4449000_image_crop': {},
            '2019_osbs_5_405000_3287000_image_crop': {},
            '2019_osbs_5_405000_3287000_image_crop2': {},
            '2019_sjer_4_251000_4103000_image': {},
            '2019_tool_3_403000_7617000_image': {}
        },
        'test': {
            'NeonTreeEvaluation_Test': {}
        }
    }


@register_dataset
class OamTcdDataset(BasePreprocessedDataset):
    dataset_name = 'OAM-TCD'
    license = 'CC-BY-4.0'
    ground_resolution = 0.1
    scale_factor = 1.0

    train_tile_size = 2048
    train_n_tiles = 3024
    train_n_annotations = 199515

    valid_tile_size = 1024
    valid_n_tiles = 4010
    valid_n_annotations = 91772

    test_tile_size = 1024
    test_n_tiles = 2527
    test_n_annotations = 56727

    tile_level_eval_maxDets = 400

    locations = [
        'global_oamtcd'
    ]

    products_per_location = {
        'global_oamtcd': [
            'oam_tcd'
        ]
    }

    raster_level_eval_inputs = {
        'valid': {
            'oam_tcd': {}
        },
        'test': {
            'oam_tcd': {}
        }
    }


@register_dataset
class Detectree2Dataset(BasePreprocessedDataset):
    dataset_name = 'Detectree2'
    license = 'CC-BY-4.0'
    ground_resolution = 0.1
    scale_factor = 1.0

    train_tile_size = None
    train_n_tiles = None
    train_n_annotations = None

    valid_tile_size = 1000
    valid_n_tiles = 331
    valid_n_annotations = 4881

    test_tile_size = 1000
    test_n_tiles = 311
    test_n_annotations = 9169

    tile_level_eval_maxDets = 100

    locations = [
        'malaysia_detectree2'
    ]

    products_per_location = {
        'malaysia_detectree2': [
            'dan_2014_rgb_project_to_chm',
            'sep_ma14_21_orthomosaic_20141023_reprojected_full_res',
        ]
    }

    raster_level_eval_inputs = {
        'valid': {
            'dan_2014_rgb_project_to_chm': {
                'ground_truth_gpkg': 'Dan_2014_RGB_project_to_CHM_labels.gpkg',
                'aoi_gpkg': 'detectree2_aoi_valid.gpkg'
            },
            'sep_ma14_21_orthomosaic_20141023_reprojected_full_res': {
                'ground_truth_gpkg': 'Sep_MA14_21_orthomosaic_20141023_reprojected_full_res_labels.gpkg',
                'aoi_gpkg': 'detectree2_aoi_valid.gpkg'
            }
        },
        'test': {
            'dan_2014_rgb_project_to_chm': {
                'ground_truth_gpkg': 'Dan_2014_RGB_project_to_CHM_labels.gpkg',
                'aoi_gpkg': 'detectree2_aoi_test.gpkg'
            },
            'sep_ma14_21_orthomosaic_20141023_reprojected_full_res': {
                'ground_truth_gpkg': 'Sep_MA14_21_orthomosaic_20141023_reprojected_full_res_labels.gpkg',
                'aoi_gpkg': 'detectree2_aoi_test.gpkg'
            }
        }
    }


@register_dataset
class BAMForestDataset(BasePreprocessedDataset):
    """BAMFORESTS converted to the same *preprocessed* layout used by NeonTreeEvaluation.

        After running the prep script, the expected structure under `raw_data_root` is:
            raw_data_root/
                bamforests/
                    BAMForest_train2023/
                        tiles/train/*.tif
                        BAMForest_train2023_coco_gr0p017_train.json
                    BAMForest_val2023/
                        tiles/valid/*.tif
                        BAMForest_val2023_coco_gr0p017_valid.json
                    BAMForest_TestSet1_2023/
                        tiles/test/*.tif
                        BAMForest_TestSet1_2023_coco_gr0p017_test.json
                    BAMForest_TestSet2_2023/
                        tiles/test/*.tif
                        BAMForest_TestSet2_2023_coco_gr0p017_test.json

    Tile filenames are expected to follow `TileNameConvention` so downstream components (e.g. aggregator)
    can parse product/ground-resolution from tile names.
    """

    dataset_name = 'BAMForest'
    license = 'See BAMFORESTS Readme (research use; contact authors for commercial use)'

    # Inferred from GeoTIFF metadata (EPSG:25832; pixel size ~0.017m)
    ground_resolution = 0.017
    scale_factor = 1.0

    train_tile_size = 2048
    train_n_tiles = 1439
    train_n_annotations = 58228

    valid_tile_size = 2048
    valid_n_tiles = 382
    valid_n_annotations = 15177

    test_tile_size = 2048
    test_n_tiles = 635
    test_n_annotations = 19040

    tile_level_eval_maxDets = 400

    locations = ['bamforests']
    products_per_location = {
        'bamforests': [
            'BAMForest_train2023',
            'BAMForest_val2023',
            'BAMForest_TestSet1_2023',
            'BAMForest_TestSet2_2023',
        ]
    }

    # Tile-only dataset (no raster-level gpkg inputs). We still list products per fold so BasePreprocessedDataset.iter_fold
    # knows which products belong to which fold (same pattern as NeonTreeEvaluation).
    raster_level_eval_inputs = {
        'train': {
            'BAMForest_train2023': {},
        },
        'valid': {
            'BAMForest_val2023': {},
        },
        'test': {
            'BAMForest_TestSet1_2023': {},
            'BAMForest_TestSet2_2023': {},
        }
    }

@register_dataset
class PanamaBCNMDataset(BasePreprocessedDataset):
    dataset_name = 'PanamaBCNM'
    license = 'CC-BY-4.0'
    ground_resolution = 0.045
    scale_factor = None

    train_tile_size = None
    train_n_tiles = None
    train_n_annotations = None

    valid_tile_size = None
    valid_n_tiles = None
    valid_n_annotations = None

    test_tile_size = 1777
    test_n_tiles = 1035
    test_n_annotations = 28624

    tile_level_eval_maxDets = 400

    locations = [
        'panama_bcnm'
    ]

    products_per_location = {
        'panama_bcnm': [
            '20241125_bcigiganteplot_m3e_rgb_cog',
            '2024_07_16_orthowhole_bci_resfull_cropped_25haplot_cog',
            '2024_07_16_orthowhole_bci_resfull_cropped_ava6haplot_cog',
            '2024_07_16_orthowhole_bci_resfull_cropped_drayton6haplot_cog',
            '2024_07_16_orthowhole_bci_resfull_cropped_pearson6haplot_cog',
            '2024_07_16_orthowhole_bci_resfull_cropped_zetek6haplot_cog',
            'bci_50ha_2024_11_12_cropped_cog'
        ]
    }

    raster_level_eval_inputs = {
        'test': {
            '20241125_bcigiganteplot_m3e_rgb_cog': {
                'ground_truth_gpkg': 'BCNM_gigante_crownmap_2025.gpkg',
                'aoi_gpkg': '20241125_bcigiganteplot_m3e_rgb_cog_aoi_gr0p045_test.gpkg'
            },
            '2024_07_16_orthowhole_bci_resfull_cropped_25haplot_cog': {
                'ground_truth_gpkg': 'BCI_25ha_crownmap_2025.gpkg',
                'aoi_gpkg': '2024_07_16_orthowhole_bci_resfull_cropped_25haplot_cog_aoi_gr0p045_test.gpkg'
            },
            '2024_07_16_orthowhole_bci_resfull_cropped_ava6haplot_cog': {
                'ground_truth_gpkg': 'BCI_ava_crownmap_2025.gpkg',
                'aoi_gpkg': '2024_07_16_orthowhole_bci_resfull_cropped_ava6haplot_cog_aoi_gr0p045_test.gpkg'
            },
            '2024_07_16_orthowhole_bci_resfull_cropped_drayton6haplot_cog': {
                'ground_truth_gpkg': 'BCI_drayton_crownmap_2025.gpkg',
                'aoi_gpkg': '2024_07_16_orthowhole_bci_resfull_cropped_drayton6haplot_cog_aoi_gr0p045_test.gpkg'
            },
            '2024_07_16_orthowhole_bci_resfull_cropped_pearson6haplot_cog': {
                'ground_truth_gpkg': 'BCI_pearson_crownmap_2025.gpkg',
                'aoi_gpkg': '2024_07_16_orthowhole_bci_resfull_cropped_pearson6haplot_cog_aoi_gr0p045_test.gpkg'
            },
            '2024_07_16_orthowhole_bci_resfull_cropped_zetek6haplot_cog': {
                'ground_truth_gpkg': 'BCI_zetek_crownmap_2025.gpkg',
                'aoi_gpkg': '2024_07_16_orthowhole_bci_resfull_cropped_zetek6haplot_cog_aoi_gr0p045_test.gpkg'
            },
            'bci_50ha_2024_11_12_cropped_cog': {
                'ground_truth_gpkg': 'BCI_50ha_crownmap_2025.gpkg',
                'aoi_gpkg': 'bci_50ha_2024_11_12_cropped_cog_aoi_gr0p045_test.gpkg'
            }
        }
    }
    
@register_dataset
class SelvaMaskDataset(BasePreprocessedDataset):
    dataset_name = 'SelvaMask'
    license = 'CC-BY-4.0'
    ground_resolution = 0.045
    scale_factor = None

    train_tile_size = None                  # TODO add values here
    train_n_tiles = None
    train_n_annotations = None

    valid_tile_size = None                  # TODO add values here
    valid_n_tiles = None
    valid_n_annotations = None

    test_tile_size = None                  # TODO add values here
    test_n_tiles = None
    test_n_annotations = None

    tile_level_eval_maxDets = 400

    locations = ['selvamask']
    products_per_location = {
        'selvamask': [
            '20240131_zf2block4_ms_m3m_rgb',
            '20240613_tbsnewsite2_m3e_rgb',
            '20241122_bcifairchildn_m3m_rgb',
        ]
    }

    raster_level_eval_inputs = {
        'valid': {
            '20240131_zf2block4_ms_m3m_rgb': {
                'ground_truth_gpkg': '20240131_zf2block4_ms_m3m_labels_masks.gpkg',
                'aoi_gpkg': '20240131_zf2block4_ms_m3m_rgb_aoi_gr0p045_valid.gpkg',
            },
            '20240613_tbsnewsite2_m3e_rgb': {
                'ground_truth_gpkg': '20240613_tbsnewsite2_m3e_labels_masks.gpkg',
                'aoi_gpkg': '20240613_tbsnewsite2_m3e_rgb_aoi_gr0p045_valid.gpkg',
            },
            '20241122_bcifairchildn_m3m_rgb': {
                'ground_truth_gpkg': '20241122_bcifairchildn_m3m_rgb_labels_masks.gpkg',
                'aoi_gpkg': '20241122_bcifairchildn_m3m_rgb_aoi_gr0p045_valid.gpkg',
            },
        },
        'test': {
            '20240131_zf2block4_ms_m3m_rgb': {
                'ground_truth_gpkg': '20240131_zf2block4_ms_m3m_labels_masks.gpkg',
                'aoi_gpkg': '20240131_zf2block4_ms_m3m_rgb_aoi_gr0p045_test.gpkg',
            },
            '20240613_tbsnewsite2_m3e_rgb': {
                'ground_truth_gpkg': '20240613_tbsnewsite2_m3e_labels_masks.gpkg',
                'aoi_gpkg': '20240613_tbsnewsite2_m3e_rgb_aoi_gr0p045_test.gpkg',
            },
            '20241122_bcifairchildn_m3m_rgb': {
                'ground_truth_gpkg': '20241122_bcifairchildn_m3m_rgb_labels_masks.gpkg',
                'aoi_gpkg': '20241122_bcifairchildn_m3m_rgb_aoi_gr0p045_test.gpkg',
            },
        },
    }
