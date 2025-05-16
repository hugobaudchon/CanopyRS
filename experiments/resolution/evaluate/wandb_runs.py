import re

import wandb

from engine.benchmark.detector.benchmark import DetectorBenchmarker
from engine.config_parsers import DetectorConfig, AggregatorConfig


def get_wandb_runs(wandb_project: str):
    api = wandb.Api()
    runs = api.runs(wandb_project)

    run_config_mapping = {}
    for run in runs:
        run_config_mapping[run.name] = run.config

    # for run in runs:
    #     history = run.history()  # returns a pandas DataFrame of the run's logged metrics
    #     csv_filename = f"{run.id}_logs.csv"
    #     history.to_csv(csv_filename, index=False)
    #     print(f"Saved logs for run {run.id} to {csv_filename}")

    return run_config_mapping


def extract_ground_resolution_regex(input_str):
    # The pattern \d+p\d+ matches one or more digits, followed by "p", followed by one or more digits.
    matches = re.findall(r'\d+p\d+', input_str)
    if len(matches) >= 2:
        # Return the second occurrence (index 1)
        match = matches[1]
    elif matches:
        # Fallback: if only one match exists, return it.
        match = matches[0]
    else:
        raise ValueError(f"No ground resolution found in {input_str}")

    ground_resolution = float(match.replace("p", "."))
    return ground_resolution


if __name__ == "__main__":
    dataset_raw_root = '/media/hugo/Hard Disk 1/CanopyRS/data/raw'

    # input_imagery = '/media/hugo/Hard Disk 1/XPrize/Data/raw/brazil_zf2/20240130_zf2tower_m3m_rgb.tif'
    # ground_truth_gpkg = '/media/hugo/Hard Disk 1/XPrize/Data/raw_new/brazil_zf2/20240130_zf2tower_m3m_labels_boxes.gpkg'
    # aoi_gpkg = '/home/hugo/Documents/20240130_zf2tower_m3m_labels_boxes_aoi_test.gpkg'
    input_imagery = '/media/hugo/Hard Disk 1/XPrize/Data/raw/equator/20230525_tbslake_m3e_rgb.tif'
    ground_truth_gpkg = '/media/hugo/Hard Disk 1/XPrize/Data/raw/equator/20230525_tbslake_m3e_labels_boxes.gpkg'
    aoi_gpkg = '/media/hugo/Hard Disk 1/XPrize/Data/raw/equator/valid_aoi.gpkg'

    output_folder = '/media/hugo/Hard Disk 1/CanopyRS/benchmark_TEST/test_all_tropics_final5'
    batch_size = 16

    wandb_project = "hugobaudchon_team/detector_experience_resolution_optimalHPs_40m"

    run_config_mapping = get_wandb_runs(wandb_project)

    # model_name = "faster_rcnn_detectron2_20250316_114746_135369_6357465"
    # checkpoint_path = "/home/hugo/Documents/model_best_fasterrcnn.pth"
    model_name = "dino_detrex_20250316_112035_206222_6357461"
    checkpoint_path = "/home/hugo/Documents/model_best_dino.pth"

    # run_config = run_config_mapping[model_name]
    # run_config['checkpoint_path'] = checkpoint_path
    #
    # pipeline_config = get_evaluation_pipeline(run_config, batch_size)
    # pipeline = Pipeline(
    #     io_config=InferIOConfig(
    #         input_imagery=input_imagery,
    #         output_folder=output_folder,
    #         ground_truth_gpkg=ground_truth_gpkg,
    #
    #         aoi_config='package',
    #         aoi=aoi_gpkg
    #     ),
    #     config=pipeline_config
    # )
    #
    # pipeline()


    # fold_name = 'test'
    fold_name = 'valid'

    benchmarker = DetectorBenchmarker(
        fold_name=fold_name,
        dataset_name='tropics',
        raw_data_root=dataset_raw_root
    )

    run_config = run_config_mapping[model_name]
    run_config['checkpoint_path'] = checkpoint_path

    print(run_config)

    detector_config = DetectorConfig(**run_config)
    # detector_config = DetectorConfig.from_yaml('/home/hugo/PycharmProjects/CanopyRS/config/default/detector.yaml')

    extract_ground_resolution_regex(detector_config.valid_dataset_names[0])

    benchmarker.benchmark(
        detector_config=detector_config,
        aggregator_config=AggregatorConfig(**{
            "nms_algorithm": 'iou',
            "score_threshold": 0.2,
            "nms_threshold": 0.3,
            "scores_weights": {'detector_score': 1.0},
            "min_centroid_distance_weight": 0.8
        }),
        tile_extent_in_meters=40,
        output_folder=f"{output_folder}/{fold_name}",
    )



