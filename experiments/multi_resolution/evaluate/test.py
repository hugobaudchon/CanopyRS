import argparse
import os

os.environ["HF_DATASETS_CACHE"] = "/network/scratch/h/hugo.baudchon/data/hf_dataset"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/network/scratch/h/hugo.baudchon/data/hf_hub"

import pandas as pd
import numpy as np
from pathlib import Path
from experiments.resolution.evaluate.get_wandb import wandb_runs_to_dataframe

from engine.benchmark.detector.benchmark import DetectorBenchmarker
from engine.config_parsers.aggregator import AggregatorConfig
from engine.config_parsers.detector import DetectorConfig



def evaluate(run_ids, best_run_id, models_root_path, raw_data_root, output_path, augmentation_image_size):
    dataset_names = ['SelvaBox']

    detector_config = DetectorConfig.from_yaml(f'{models_root_path}/{best_run_id}/config.yaml')
    detector_config.checkpoint_path = f'{models_root_path}/{best_run_id}/model_best.pth'
    # detector_config.augmentation_image_size = augmentation_image_size # this is what should usually be used
    detector_config.augmentation_image_size = 1777      # TODO remove !!!!!!!!!!!!!!!!!!!!!!!!!!!
    detector_config.augmentation_early_image_resize_test_only = augmentation_image_size # setting value to augmentation_early_image_resize_test_only in order to drop res to 800px but still allow img to be resized to 1024 (lower bound of [1024, 1777]), instead of evaluating at 800px directly which will impact model as it wasn't trained on that image size range.
    # TODO ALSO MAKE SURE image_size etc ARE FINE LOWER IN THE FUNCTION WHEN INFERING ON TEST SET !!

    output_path_valid_run = Path(output_path) / 'valid' / best_run_id
    output_path_valid_run.mkdir(parents=True, exist_ok=False)

    valid_benchmarker = DetectorBenchmarker(
        output_folder=output_path_valid_run,
        fold_name='valid',
        raw_data_root=raw_data_root
    )
    best_aggregator_iou, best_aggregator_score_threshold = valid_benchmarker.find_optimal_nms_iou_threshold(
        detector_config=detector_config,
        dataset_names=dataset_names,
        nms_iou_thresholds=[round(x, 1) for x in np.arange(0.05, 1.05, 0.05)],
        nms_score_thresholds=[round(x, 2) for x in np.arange(0.05, 1.05, 0.05)],
        n_workers=12
    )

    tile_level_results = []
    raster_level_results = []

    for run_id in run_ids:
        test_fold = 'test'
        output_path_run = Path(output_path) / run_id
        output_path_run.mkdir(parents=True, exist_ok=False)

        test_benchmarker = DetectorBenchmarker(
            output_folder=output_path_run,
            fold_name=test_fold,
            raw_data_root=raw_data_root
        )

        detector_config = DetectorConfig.from_yaml(f'{models_root_path}/{run_id}/config.yaml')
        detector_config.checkpoint_path = f'{models_root_path}/{run_id}/model_best.pth'
        # detector_config.augmentation_image_size = augmentation_image_size # this is what should usually be used
        detector_config.augmentation_image_size = 1777      # TODO remove !!!!!!!!!!!!!!!!!!!!!!!!!!!
        detector_config.augmentation_early_image_resize_test_only = augmentation_image_size # setting value to augmentation_early_image_resize_test_only in order to drop res to 800px but still allow img to be resized to 1024 (lower bound of [1024, 1777]), instead of evaluating at 800px directly which will impact model as it wasn't trained on that image size range.
        # TODO ALSO MAKE SURE image_size etc ARE FINE UP THERE AT START OF FUNCTION WHEN FINDING OPTIMAL NMS THRESHOLDS

        aggregator_config = AggregatorConfig(
            score_threshold=best_aggregator_score_threshold,
            nms_threshold=best_aggregator_iou,
            nms_algorithm='iou'
        )

        test_benchmarker.benchmark(
            detector_config=detector_config,
            aggregator_config=aggregator_config,
            dataset_names=dataset_names,
        )

        tlm = pd.read_csv(output_path_run / test_fold / "tile_level_metrics.csv")
        rlm = pd.read_csv(output_path_run / test_fold / "raster_level_metrics.csv")
        tlm['run_id'] = run_id
        rlm['run_id'] = run_id

        tile_level_results.append(tlm)
        raster_level_results.append(rlm)

    def summarize_and_write(dfs, level, out_fname):
        all_df = pd.concat(dfs, ignore_index=True)

        # only metrics that actually exist in this DataFrame
        metrics_to_average = [
            'AP','AP50','AP75','AP_small','AP_medium','AP_large',
            'AR','AR50','AR75','AR_small','AR_medium','AR_large',
            'F1','F1_50','F1_75','F1_small','F1_medium','F1_large',
            'precision', 'recall', 'f1'
        ]

        present = [m for m in metrics_to_average if m in all_df.columns]

        stats = (
            all_df
            .groupby('product_name')[present]
            .agg(['mean','std'])
        )
        stats.columns = [f"{level}_{metric}_{stat}" for metric, stat in stats.columns]
        stats = stats.reset_index()  # bring product_name back as a column

        meta_cols = [
            c for c in all_df.columns
            if c not in present + ['run_id']
        ]
        # drop duplicates so we have one row per product and location
        meta = (
            all_df[meta_cols]
            .drop_duplicates(subset=['product_name'])
        )

        summary = pd.merge(meta, stats, on='product_name', how='right')

        summary.to_csv(Path(output_path) / out_fname, index=False)
        return summary

    summary_tile = summarize_and_write(
        tile_level_results,
        'tile',
        'tile_level_summary.csv'
    )
    summary_raster = summarize_and_write(
        raster_level_results,
        'raster', 
        'raster_level_summary.csv'
    )

    common = [c for c in summary_tile.columns if c in summary_raster.columns]
    raster_unique = summary_raster.drop(columns=[c for c in common if c != 'product_name'])

    # 3) Merge on product_name
    combined = pd.merge(
        summary_tile,
        raster_unique,
        on='product_name',
        how='inner'
    )

    # 4) Write out your combined summary
    combined.to_csv(Path(output_path) / 'combined_level_summary.csv', index=False)


def main(augmentation_image_size):
    wandb_project = f"hugobaudchon_team/detector_experience_multi_resolution"
    wandb_df = wandb_runs_to_dataframe(wandb_project)
    print(wandb_df)
    
    raw_data_root = '/network/scratch/h/hugo.baudchon/data/extracted'
    models_root_path = '/network/scratch/h/hugo.baudchon/training/detector_experience_multi_resolution'
    output_path_root = f"/network/scratch/h/hugo.baudchon/eval/detector_experience_multi_resolution_NEW_METRIC_imgsize1777_BETTER_{str(augmentation_image_size).replace(' ', '')}"

    # wandb_project = f"hugobaudchon_team/detector_experience_multi_resolution_cropprob1p0"
    # wandb_df = wandb_runs_to_dataframe(wandb_project)
    # print(wandb_df)

    # raw_data_root = '/network/scratch/h/hugo.baudchon/data/extracted'
    # models_root_path = '/network/scratch/h/hugo.baudchon/training/detector_experience_multi_resolution_cropprob1p0'
    # output_path_root = f"/network/scratch/h/hugo.baudchon/eval/detector_experience_multi_resolution_NEW_METRIC_cropprob1p0_{str(augmentation_image_size).replace(' ', '')}"

    # wandb_project = f"hugobaudchon_team/detector_experience_resolution_optimalHPs_80m_FIXED"
    # wandb_df = wandb_runs_to_dataframe(wandb_project)
    # print(wandb_df)

    # raw_data_root = '/network/scratch/h/hugo.baudchon/data/extracted'
    # models_root_path = '/network/scratch/h/hugo.baudchon/training/detector_experience_resolution_optimalHPs_80m_FIXED'
    # output_path_root = f"/network/scratch/h/hugo.baudchon/eval/detector_experience_multi_resolution_NEW_METRIC_multi_res_models_imgsize1777_{str(augmentation_image_size).replace(' ', '')}"


    exp_name_to_run_ids = {
        '34_88m': [
            'dino_detrex_20250408_071705_415176_6540183',
            'dino_detrex_20250408_072217_176862_6540184',
            'dino_detrex_20250408_072850_655443_6540185'
        ],
        '30_100m': [
            'dino_detrex_20250409_034730_631287_6548134',
            'dino_detrex_20250406_044042_752862_6529903',
            'dino_detrex_20250409_034721_885204_6548135'
        ],
        '30_120m': [
            'dino_detrex_20250415_152900_472695_6590712',
            'dino_detrex_20250416_064931_881983_6590713',
            'dino_detrex_20250416_084416_939465_6590714'
        ]

        ##cropprob1p0
        # '34_88m': [
        #     'dino_detrex_20250527_020106_313263_6892996',
        #     'dino_detrex_20250527_020112_171031_6892997',
        #     'dino_detrex_20250527_020147_013679_6892998'
        # ],
        # '30_100m': [
        #     'dino_detrex_20250527_020117_970817_6892995',
        #     'dino_detrex_20250527_020112_128008_6892993',
        #     'dino_detrex_20250527_020112_171031_6892997'
        # ],
        # '30_120m': [
        #     'dino_detrex_20250527_020053_743732_6892990',
        #     'dino_detrex_20250527_020055_633362_6892991',
        #     'dino_detrex_20250527_020054_013458_6892992'
        # ]

        # '36_88m': [
        #     'dino_detrex_20250605_213523_281215_6965592',
        #     'dino_detrex_20250611_144112_301084_6980956',
        #     'dino_detrex_20250611_065033_701587_6980954'
        # ],
        # '30_100m': [
        #     'dino_detrex_20250605_193721_556164_6965588',
        #     'dino_detrex_20250611_133619_089670_6980966',
        #     'dino_detrex_20250611_142134_273326_6980967'
        # ],
        # '30_120m': [
        #     'dino_detrex_20250605_171813_955157_6965580',
        #     'dino_detrex_20250613_020011_672678_7001612',
        #     'dino_detrex_20250613_020038_763495_7001611'
        # ]

        # '30_160m': [
        #     'dino_detrex_20250620_013351_632169_7032634',
        #     'dino_detrex_20250620_031903_165590_7032635',
        #     'dino_detrex_20250620_042507_866167_7032636'
        # ]

        # 'singleres_800': [
        #     'dino_detrex_20250327_045309_678038_6444509',
        #     'dino_detrex_20250327_045614_365706_6444510',
        #     'dino_detrex_20250327_045614_035967_6444511'
        # ],
        # 'singleres_1333': [
        #     'dino_detrex_20250327_170256_905581_6447556',
        #     'dino_detrex_20250327_171201_279552_6447557',
        #     'dino_detrex_20250328_010939_402376_6447558'
        # ],
        # 'singleres_1777': [
        #     'dino_detrex_20250327_074446_013929_6444506',
        #     'dino_detrex_20250327_075032_754232_6444507',
        #     'dino_detrex_20250327_075428_492222_6444508'
        # ],

        #trained on img size 1777
        # 'singleres_800_1777': [
        #     'dino_detrex_20250327_074648_976249_6444512',
        #     'dino_detrex_20250327_075016_543402_6444513',
        #     'dino_detrex_20250327_081115_465882_6444514'
        # ],
        # 'singleres_1333_1777': [
        #     'dino_detrex_20250328_093914_416452_6447559',
        #     'dino_detrex_20250328_202147_319662_6447560',
        #     'dino_detrex_20250328_204903_056088_6447561'
        # ],
        # 'singleres_1777_1777': [
        #     'dino_detrex_20250327_074446_013929_6444506',
        #     'dino_detrex_20250327_075032_754232_6444507',
        #     'dino_detrex_20250327_075428_492222_6444508'
        # ],
    }

    for exp_name, run_ids in exp_name_to_run_ids.items():

        output_path = f'{output_path_root}/{exp_name}'

        df_sub = wandb_df[wandb_df['run_name'].isin(run_ids)]
        print(df_sub)
        print(df_sub['bbox/AP.max'])

        # find the index of the max AP and pull out its run_name
        best_idx  = df_sub['bbox/AP.max'].idxmax()
        best_model_id = df_sub.at[best_idx, 'run_name']

        evaluate(
            run_ids=run_ids,
            best_run_id=best_model_id,
            models_root_path=models_root_path,
            raw_data_root=raw_data_root,
            output_path=output_path,
            augmentation_image_size=augmentation_image_size
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "augmentation_image_size",
    )
    args = parser.parse_args()

    main(args.augmentation_image_size)
