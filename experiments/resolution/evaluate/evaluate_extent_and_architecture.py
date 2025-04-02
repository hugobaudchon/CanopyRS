import argparse
import json
from pathlib import Path

import pandas as pd
from geodataset.aoi import AOIFromPackageConfig
from geodataset.utils import CocoNameConvention, validate_and_convert_product_name, strip_all_extensions_and_path

from dataset.detection.tilerize import tilerize_with_overlap
from engine.benchmark.evaluator import CocoEvaluator
from engine.config_parsers import DetectorConfig, InferIOConfig, PipelineConfig, AggregatorConfig
from engine.pipeline import Pipeline
from experiments.resolution.evaluate.get_wandb import extract_tilerized_image_size_regex, wandb_runs_to_dataframe, extract_ground_resolution_regex
from tools.find_optimal_detector_aggregator import find_optimal_detector_aggregator

inputs = {
    'valid': {              # TODO move these as constants in the dataset/detection Dataset classes
        'brazil_zf2': [
            {
                'input_imagery': '20240130_zf2quad_m3m_rgb.cog.tif',
                'ground_truth_gpkg': '20240130_zf2quad_m3m_labels_boxes.gpkg',
                'aoi': '20240130_zf2quad_m3m_labels_boxes_aoi_valid.gpkg'
            },
        ],
        'ecuador_tiputini': [
            {
                'input_imagery': '20231018_inundated_m3e_rgb.cog.tif',
                'ground_truth_gpkg': '20231018_inundated_m3e_labels_boxes.gpkg',
                'aoi': '20231018_inundated_m3e_labels_boxes_aoi_valid.gpkg'
            },
        ],
        'panama_aguasalud': [
            {
                'input_imagery': '20231207_asnortheast_amsunclouds_m3m_rgb.cog.tif',
                'ground_truth_gpkg': '20231207_asnortheast_amsunclouds_m3m_labels_boxes.gpkg',
                'aoi': '20231207_asnortheast_amsunclouds_m3m_labels_boxes_aoi_valid.gpkg'
            },
            {
                'input_imagery': '20231208_asforestnorthe2_m3m_rgb.cog.tif',
                'ground_truth_gpkg': '20231208_asforestnorthe2_m3m_labels_boxes.gpkg',
                'aoi': '20231208_asforestnorthe2_m3m_labels_boxes_aoi_valid.gpkg'
            }
        ]
    },
    'test': {              # TODO move these as constants in the dataset/detection Dataset classes
        'brazil_zf2': [
            {
                'input_imagery': '20240130_zf2tower_m3m_rgb.cog.tif',
                'ground_truth_gpkg': '20240130_zf2tower_m3m_labels_boxes.gpkg',
                'aoi': '20240130_zf2tower_m3m_labels_boxes_aoi_test.gpkg'
            },
        ],
        'ecuador_tiputini': [
            {
                'input_imagery': '20230525_tbslake_m3e_rgb.cog.tif',
                'ground_truth_gpkg': '20230525_tbslake_m3e_labels_boxes.gpkg',
                'aoi': '20230525_tbslake_m3e_labels_boxes_aoi_test.gpkg'
            },
            {
                'input_imagery': '20231018_inundated_m3e_rgb.cog.tif',
                'ground_truth_gpkg': '20231018_inundated_m3e_labels_boxes.gpkg',
                'aoi': '20231018_inundated_m3e_labels_boxes_aoi_test.gpkg'
            }
        ],
        'panama_aguasalud': [
            {
                'input_imagery': '20231207_asnortheast_amsunclouds_m3m_rgb.cog.tif',
                'ground_truth_gpkg': '20231207_asnortheast_amsunclouds_m3m_labels_boxes.gpkg',
                'aoi': '20231207_asnortheast_amsunclouds_m3m_labels_boxes_aoi_test.gpkg'
            },
            {
                'input_imagery': '20231208_asforestnorthe2_m3m_rgb.cog.tif',
                'ground_truth_gpkg': '20231208_asforestnorthe2_m3m_labels_boxes.gpkg',
                'aoi': '20231208_asforestnorthe2_m3m_labels_boxes_aoi_test.gpkg'
            }
        ]
    }
}


tilerized_name_to_extent = {
    'tilerized_888_0p5_0p045_None': '40m',
    'tilerized_666_0p5_0p06_None': '40m',
    'tilerized_400_0p5_0p1_None': '40m',

    'tilerized_1777_0p5_0p045_None': '80m',
    'tilerized_1333_0p5_0p06_None': '80m',
    'tilerized_800_0p5_0p1_None': '80m'
}


def find_model_checkpoint_path(models_root, model_name: str):
    return f"{models_root}/{model_name}/model_best.pth"


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


def evaluate_extent_and_architecture(wandb_project: str, architecture: str, extent: str, raw_root: str, models_root: str, output_folder, temp_folder):
    print(f"Architecture: {architecture}, Extent: {extent}")

    raw_root = Path(raw_root)

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Get wandb logs
    wandb_df = wandb_runs_to_dataframe(wandb_project)

    # Filter by architecture and extent
    wandb_df['extent'] = wandb_df['valid_dataset_names'].apply(lambda x: tilerized_name_to_extent[x[0].split('/')[0]])
    wandb_df['tilerizer_tile_size'] = wandb_df['valid_dataset_names'].apply(lambda x: extract_tilerized_image_size_regex(x[0]))
    wandb_df['ground_resolution'] = wandb_df['valid_dataset_names'].apply(lambda x: extract_ground_resolution_regex(x[0]))
    wandb_df = wandb_df[(wandb_df['architecture'] == architecture) & (wandb_df['extent'] == extent)]

    # First, tilerize the raw data for each configuration of tile size and ground resolution
    tilerized_paths = {}
    for fold, fold_inputs in inputs.items():
        tilerized_paths[fold] = {}
        for (tile_size, ground_resolution) in list(wandb_df[['tilerizer_tile_size', 'ground_resolution']].drop_duplicates().itertuples(index=False, name=None)):
            tilerized_paths[fold][(tile_size, ground_resolution)] = {}
            for dataset_name, raster_list in fold_inputs.items():
                for raster in raster_list:
                    aoi_path = raw_root / dataset_name / raster['aoi']
                    raster_path = raw_root / dataset_name / raster['input_imagery']
                    labels_path = raw_root / dataset_name / raster['ground_truth_gpkg']

                    coco_outputs, tiles_path = tilerize_with_overlap(
                        raster_path=raster_path,
                        labels=labels_path,
                        main_label_category_column_name=None,
                        coco_categories_list=None,
                        aois_config=AOIFromPackageConfig({fold: aoi_path}),
                        output_path=Path(temp_folder) / 'tilerized' / f"{tile_size}_{ground_resolution}",
                        ground_resolution=ground_resolution,
                        scale_factor=None,
                        tile_size=tile_size,
                        tile_overlap=0.75
                    )

                    coco_path = coco_outputs[fold]
                    tiles_path = tiles_path / fold
                    product_name, _, _, _ = CocoNameConvention.parse_name(coco_path.name)

                    tilerized_paths[fold][(tile_size, ground_resolution)][product_name] = {
                        'labels_gpkg_path': labels_path,
                        'labels_coco_path': coco_path,
                        'tiles_path': tiles_path
                    }

                    # product_name = validate_and_convert_product_name(strip_all_extensions_and_path(raster_path))
                    # coco_name = CocoNameConvention.create_name(
                    #     product_name=product_name,
                    #     ground_resolution=ground_resolution,
                    #     fold=fold
                    # )
                    # coco_path = Path(temp_folder) / 'tilerized' / f"{tile_size}_{ground_resolution}" / product_name / coco_name
                    # tiles_path = Path(temp_folder) / 'tilerized' / f"{tile_size}_{ground_resolution}" / product_name / 'tiles' / fold
                    # tilerized_paths[fold][(tile_size, ground_resolution)][product_name] = {
                    #     'labels_gpkg_path': labels_path,
                    #     'labels_coco_path': coco_path,
                    #     'tiles_path': tiles_path
                    # }

    # Find the best model based on the bbox/AP metric
    best_model = wandb_df.sort_values('bbox/AP.max', ascending=False).iloc[0]
    best_model_config = DetectorConfig(**best_model.to_dict())
    best_model_config.checkpoint_path = find_model_checkpoint_path(models_root, best_model['run_name'])
    best_model_config.batch_size = 2

    # Find Aggregator parameters based on the best model
    # First infer the best model on the tilerized data for VALID (!!!) fold
    best_model_raster_names = []
    best_model_coco_preds = []
    best_model_gpkg_truths = []
    best_model_tiles_paths = []
    for product_name, tilerized_product_paths in tilerized_paths['valid'][(best_model['tilerizer_tile_size'], best_model['ground_resolution'])].items():
        tiles_path = tilerized_product_paths['tiles_path']

        io_config = InferIOConfig(
            input_imagery=None,
            tiles_path=str(tiles_path),
            output_folder=str(Path(output_folder) / 'best_model' / product_name),
        )

        pipeline_config = PipelineConfig(
            components_configs=[
                ("detector", best_model_config),
            ]
        )

        pipeline = Pipeline(io_config, pipeline_config)
        pipeline()

        detector_output = pipeline.data_state.get_output_file('detector', 0, 'coco')

        best_model_coco_preds.append(detector_output)
        best_model_gpkg_truths.append(tilerized_product_paths['labels_gpkg_path'])
        best_model_tiles_paths.append(tiles_path)
        best_model_raster_names.append(product_name)

    # Find the optimal detector aggregator
    aggregators_results = find_optimal_detector_aggregator(
        output_folder=f"{output_folder}/best_model/aggregator_search",
        raster_names=best_model_raster_names,
        preds_coco_jsons=best_model_coco_preds,
        truths_gdfs=best_model_gpkg_truths,
        tiles_roots=best_model_tiles_paths,
        ground_resolution=0.045,            # Evaluating all models on the same resolution, 0.045m/pixel.
        nms_iou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        min_centroid_distance_weights=[1.0],    # not using this in the end, so setting to 1.0
        min_nms_score_threshold=0.2,
        n_workers=4
    )

    aggregators_results.to_csv(f"{output_folder}/aggregator_search_results_valid_fold.csv")
    aggregators_results.to_csv(f"{output_folder}/best_model/aggregator_search_results_valid_fold.csv")

    # Find the best aggregator based on a composite metric 'F1' (but it is not really F1 as mAP and mAR are not direct analogs to precision and recall)
    aggregators_agg = aggregators_results[aggregators_results['raster_name'] == 'average_over_rasters']
    best_aggregator = aggregators_agg.sort_values('F1', ascending=False).iloc[0]
    best_aggregator_config = AggregatorConfig(
        nms_algorithm='iou',
        nms_threshold=best_aggregator['nms_iou_threshold'],
        score_threshold=best_aggregator['nms_score_threshold'],
        min_centroid_distance_weight=best_aggregator['min_centroid_distance_weight']
    )
    print("Best aggregator found: ", str(best_aggregator_config))

    # Now, evaluate all the models on the TEST (!!!) fold, using the best aggregator
    all_tile_level_metrics = []
    all_raster_level_metrics = []
    for idx, model_row in wandb_df.iterrows():
        # Create a DetectorConfig for the current model.
        detector_config = DetectorConfig(**model_row.to_dict())
        detector_config.checkpoint_path = find_model_checkpoint_path(models_root, model_row['run_name'])
        detector_config.batch_size = 2

        detector_outputs = []
        detector_truths = []
        for product_name, tilerized_product_paths in tilerized_paths['test'][(model_row['tilerizer_tile_size'], model_row['ground_resolution'])].items():
            tiles_path = tilerized_product_paths['tiles_path']

            pipeline = Pipeline(
                io_config=InferIOConfig(
                    input_imagery=None,
                    tiles_path=str(tiles_path),
                    output_folder=str(Path(output_folder) / 'test' / model_row['run_name'] / product_name),
                ),
                config=PipelineConfig(
                    components_configs=[
                        ("detector", detector_config),
                        ("aggregator", best_aggregator_config)
                    ]
                )
            )

            pipeline()

            detector_output = pipeline.data_state.get_output_file('detector', 0, 'coco')
            aggregator_output = pipeline.data_state.get_output_file('aggregator', 1, 'gpkg')

            evaluator = CocoEvaluator()

            tile_level_metrics = evaluator.tile_level(
                iou_type='bbox',
                preds_coco_path=detector_output,
                truth_coco_path=tilerized_product_paths['labels_coco_path'],
                max_dets=[1, 10, 100]
            )

            raster_level_metrics = evaluator.raster_level(
                iou_type='bbox',
                preds_gpkg_path=aggregator_output,
                truth_gpkg_path=tilerized_product_paths['labels_gpkg_path'],
                ground_resolution=0.045            # Evaluating all models on the same resolution, 0.045m/pixel.
            )

            all_tile_level_metrics.append(
                {
                    'model_name': model_row['run_name'],
                    'tilerizer_tile_size': model_row['tilerizer_tile_size'],
                    'augmentation_image_size': model_row['augmentation_image_size'],
                    'ground_resolution': model_row['ground_resolution'],
                    'seed': model_row['seed'],
                    'product_name': product_name,
                    **tile_level_metrics
                }
            )

            all_raster_level_metrics.append(
                {
                    'model_name': model_row['run_name'],
                    'tilerizer_tile_size': model_row['tilerizer_tile_size'],
                    'augmentation_image_size': model_row['augmentation_image_size'],
                    'ground_resolution': model_row['ground_resolution'],
                    'seed': model_row['seed'],
                    'product_name': product_name,
                    **raster_level_metrics
                }
            )

            detector_outputs.append(detector_output)
            detector_truths.append(tilerized_product_paths['labels_coco_path'])

        # Evaluating a the tile level for all rasters (the metric isnt linear so cant weight the individual rasters metrics, we have to merge predictions as single COCO)
        merged_tile_level_preds_path = Path(output_folder) / 'test' / model_row['run_name'] / 'merged_tile_level_preds.json'
        merged_tile_level_truth_path = Path(output_folder) / 'test' / model_row['run_name'] / 'merged_tile_level_truth.json'
        merge_coco_jsons(detector_outputs, merged_tile_level_preds_path)
        merge_coco_jsons(detector_truths, merged_tile_level_truth_path)
        merged_tile_metrics = CocoEvaluator().tile_level(
            iou_type='bbox',
            preds_coco_path=str(merged_tile_level_preds_path),
            truth_coco_path=str(merged_tile_level_truth_path),
            max_dets=[1, 10, 100]            
        )

        all_tile_level_metrics.append(
            {
                'model_name': model_row['run_name'],
                'tilerizer_tile_size': model_row['tilerizer_tile_size'],
                'augmentation_image_size': model_row['augmentation_image_size'],
                'ground_resolution': model_row['ground_resolution'],
                'seed': model_row['seed'],
                'product_name': 'all',
                **merged_tile_metrics
            }
        )

    tile_level_df = pd.DataFrame(all_tile_level_metrics)
    raster_level_df = pd.DataFrame(all_raster_level_metrics)

    # Save the results
    tile_level_df.to_csv(f"{output_folder}/test_tile_level_metrics.csv")
    raster_level_df.to_csv(f"{output_folder}/test_raster_level_metrics.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training args
    parser.add_argument("--wandb_project", type=str, help="The wandb project to use for loading models configs.", required=True)
    parser.add_argument("--architecture", type=str, help="The architecture of the model to evaluate", required=True)
    parser.add_argument("--extent", type=str, help="To only select models trained for a specific tile extent (40m or 80m...).", required=True)
    parser.add_argument("--raw_root", type=str, help="Path to the root folder of the raw datasets to use for evaluating models.", required=True)
    parser.add_argument("--models_root", type=str, help="Path to the folder where to save the results of the evaluation.", required=True)
    parser.add_argument("--output_folder", type=str, help="Path to the folder where to save the results of the evaluation.", required=True)
    parser.add_argument("--temp_folder", type=str, help="Path to the folder where to save temporary artifacts such as tiled data.", required=True)

    args = parser.parse_args()

    evaluate_extent_and_architecture(
        wandb_project=args.wandb_project,
        architecture=args.architecture,
        extent=args.extent,
        raw_root=args.raw_root,
        models_root=args.models_root,
        output_folder=args.output_folder,
        temp_folder=args.temp_folder
    )
