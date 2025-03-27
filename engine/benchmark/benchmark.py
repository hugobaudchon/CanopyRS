import json
from pathlib import Path

from geodataset.aoi import AOIFromPackageConfig

from dataset.detection.tilerize import tilerize_with_overlap
from engine.benchmark.evaluator import CocoEvaluator
from engine.config_parsers import DetectorConfig, AggregatorConfig, PipelineConfig, InferIOConfig
from engine.config_parsers.evaluator import EvaluatorConfig
from engine.pipeline import Pipeline


class DetectorBenchmarker:
    def __init__(self,
                 fold_name: str,
                 dataset_name: str,
                 raw_data_root: Path or str):
        self.fold_name = fold_name
        self.dataset_name = dataset_name
        self.raw_data_root = Path(raw_data_root)

        assert fold_name in ['test', 'valid'], f'Fold {fold_name} not supported. Supported folds are "test" and "valid".'
        self.verify_data_exists()

    def get_inputs(self):
        if self.dataset_name == 'tropics':
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
        elif self.dataset_name == 'quebec_trees':
            raise NotImplementedError()    # TODO
        elif self.dataset_name == 'panama_bci50ha':
            raise NotImplementedError()    # TODO
        elif self.dataset_name == 'global_oamtcd':
            raise NotImplementedError()    # TODO
        elif self.dataset_name == 'unitedstates_neon':
            raise NotImplementedError()    # TODO
        else:
            raise ValueError(f'Dataset {self.dataset_name} not supported.')

        return inputs[self.fold_name]

    def verify_data_exists(self):
        dataset_inputs = self.get_inputs()
        for dataset_name, raster_list in dataset_inputs.items():
            for raster in raster_list:
                for data_type, data_name in raster.items():
                    if not (self.raw_data_root / dataset_name / data_name).exists():
                        raise FileNotFoundError(f'{self.raw_data_root / dataset_name / data_name} does not exist.')

    @staticmethod
    def get_tilerizer_params(tile_extent_in_meters):
        if tile_extent_in_meters == 40:
            tile_size = 888
            ground_resolution = 0.045
        elif tile_extent_in_meters == 80:
            tile_size = 1777
            ground_resolution = 0.045
        else:
            raise ValueError(f'Tile extent {tile_extent_in_meters} not supported.')

        return tile_size, ground_resolution

    @staticmethod
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

    def benchmark(self,
                  detector_config: DetectorConfig,
                  aggregator_config: AggregatorConfig,
                  tile_extent_in_meters: int,
                  output_folder: str):

        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=False)

        detector_config.to_yaml(output_folder / 'detector_config.yaml')
        aggregator_config.to_yaml(output_folder / 'aggregator_config.yaml')

        pipeline_config = PipelineConfig(
            components_configs=[
                ("detector", detector_config),
                ("aggregator", aggregator_config)
            ]
        )

        evaluator_config_tiles = EvaluatorConfig(**{
            "type": 'instance_detection',
            "level": 'tile'
        })
        evaluator_config_raster = EvaluatorConfig(**{
            "type": 'instance_detection',
            "level": 'raster',
            "raster_eval_ground_resolution": 0.05
        })

        all_tile_level_coco_preds = {}
        all_tile_level_coco_truth = {}
        all_tile_level_metrics = {}

        all_raster_level_gpkg_preds = {}
        all_raster_level_gpkg_truth = {}

        inputs = self.get_inputs()
        for dataset_name, raster_list in inputs.items():
            for raster in raster_list:
                tile_size, ground_resolution = self.get_tilerizer_params(tile_extent_in_meters)

                print(tile_size, ground_resolution)

                aoi_name = self.fold_name
                aoi_path = self.raw_data_root / dataset_name / raster['aoi']
                raster_path = self.raw_data_root / dataset_name / raster['input_imagery']
                labels_path = self.raw_data_root / dataset_name / raster['ground_truth_gpkg']

                coco_outputs, tiles_path = tilerize_with_overlap(
                    raster_path=raster_path,
                    labels=labels_path,
                    main_label_category_column_name=None,
                    coco_categories_list=None,
                    aois_config=AOIFromPackageConfig({aoi_name: aoi_path}),
                    output_path=output_folder / dataset_name,
                    ground_resolution=ground_resolution,
                    scale_factor=None,
                    tile_size=tile_size,
                    tile_overlap=0.75
                )

                tiles_path = tiles_path / aoi_name
                coco_path = coco_outputs[aoi_name]
                pipeline_output_path = tiles_path.parent.parent

                io_config = InferIOConfig(
                    input_imagery=None,  # don't need it for pipeline since we already tilerized
                    tiles_path=str(tiles_path),
                    output_folder=str(pipeline_output_path)
                )

                pipeline = Pipeline(io_config, pipeline_config)
                pipeline()

                detector_output = pipeline.data_state.get_output_file('detector', 0, 'coco')
                aggregator_output = pipeline.data_state.get_output_file('aggregator', 1, 'gpkg')

                raster_tile_metrics = CocoEvaluator().tile_level(
                    config=evaluator_config_tiles,
                    preds_coco_path=str(detector_output),
                    truth_coco_path=str(coco_path)
                )

                all_tile_level_coco_preds[raster_path] = detector_output
                all_tile_level_coco_truth[raster_path] = coco_path
                all_tile_level_metrics[raster_path] = raster_tile_metrics
                all_raster_level_gpkg_preds[raster_path] = aggregator_output
                all_raster_level_gpkg_truth[raster_path] = labels_path

        # Evaluate at the tile level by merging all the coco files
        all_tile_level_coco_preds_list = list(all_tile_level_coco_preds.values())
        all_tile_level_coco_truth_list = list(all_tile_level_coco_truth.values())
        merged_tile_level_preds_path = output_folder / 'merged_tile_level_preds.json'
        merged_tile_level_truth_path = output_folder / 'merged_tile_level_truth.json'
        self.merge_coco_jsons(all_tile_level_coco_preds_list, merged_tile_level_preds_path)
        self.merge_coco_jsons(all_tile_level_coco_truth_list, merged_tile_level_truth_path)

        tile_metrics = CocoEvaluator().tile_level(
            config=evaluator_config_tiles,
            preds_coco_path=str(merged_tile_level_preds_path),
            truth_coco_path=str(merged_tile_level_truth_path)
        )

        print(json.dumps(tile_metrics, indent=3))
        all_tile_level_metrics = {str(k.stem): v for k, v in all_tile_level_metrics.items()}  # for saving to disk, convert Path to str of stem
        all_tile_level_metrics['all_merged'] = tile_metrics
        with open(output_folder / 'tile_level_metrics.json', 'w') as f:
            json.dump(all_tile_level_metrics, f, indent=3)

        # Evaluate at the raster level by comparing the gpkg outputs
        all_raster_level_metrics = {}
        for input_imagery in all_raster_level_gpkg_preds.keys():
            raster_metrics = CocoEvaluator().raster_level(
                config=evaluator_config_raster,
                preds_gpkg_path=str(all_raster_level_gpkg_preds[input_imagery]),
                truth_gpkg_path=str(all_raster_level_gpkg_truth[input_imagery]),
                imagery_path=str(input_imagery)
            )

            all_raster_level_metrics[input_imagery] = raster_metrics

        metric_names = [
            "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
            "AR_max", "AR_small", "AR_medium", "AR_large"
        ]
        total_truths = 0
        weighted_sum = {metric: 0.0 for metric in metric_names}

        for input_imagery, metrics in all_raster_level_metrics.items():
            truths = metrics.get("num_truths", 0)
            total_truths += truths
            for metric in metric_names:
                weighted_sum[metric] += metrics.get(metric, 0) * truths

        weighted_average = {metric: weighted_sum[metric] / total_truths for metric in metric_names}

        print("Weighted Average Metrics:")
        print(json.dumps(weighted_average, indent=3))

        all_raster_level_metrics = {str(k.stem): v for k, v in all_raster_level_metrics.items()}    # for saving to disk, convert Path to str of stem
        all_raster_level_metrics['weighted_average'] = weighted_average

        with open(output_folder / 'raster_level_metrics.json', 'w') as f:
            json.dump(all_raster_level_metrics, f, indent=3)

        return all_tile_level_metrics, all_raster_level_metrics

