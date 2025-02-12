import argparse
import logging
import multiprocessing
import os
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Importing from timm.models.layers is deprecated"
)
detrex_logger = logging.getLogger("detrex.checkpoint.c2_model_loading")
detrex_logger.disabled = True

from engine.config_parsers import InferIOConfig, PipelineConfig, DetectorConfig
from engine.config_parsers.base import get_config_path
from engine.models.detector.train_detectron2.train_detectron2 import train_detectron2_fasterrcnn
from engine.models.detector.train_detectron2.train_detrex import train_detrex
from engine.pipeline import Pipeline


def pipeline_main(args):
    config_path = get_config_path(f'{args.config}/pipeline')
    config = PipelineConfig.from_yaml(config_path)

    if args.io_config_path and (args.imagery_path or args.output_path):
        raise ValueError("Either provide an io config file or imagery path and output path.")
    elif args.io_config_path:
        io_config = InferIOConfig.from_yaml(args.io_config_path)
    elif args.imagery_path and args.output_path:
        config_args = {
            'input_imagery': args.imagery_path,
            'output_folder': args.output_path
        }
        if args.ground_truth_path:
            config_args['ground_truth_gpkg'] = args.ground_truth_path
        if args.aoi_path:
            config_args['aoi_config'] = 'package'
            config_args['aoi'] = args.aoi_path

        io_config = InferIOConfig(**config_args)
    else:
        raise ValueError("Provide either an io config file or imagery path and output path.")

    Pipeline(io_config, config)()


def train_detector_main(args):
    config = DetectorConfig.from_yaml(args.config)

    if args.dataset:
        config.data_root_path = args.dataset

    if config.model == 'faster_rcnn_detectron2':
        train_detectron2_fasterrcnn(config)
    elif config.model == 'dino_detrex':
        train_detrex(config)
    else:
        raise ValueError("Invalid model type/name.")



if __name__ == '__main__':
    if os.name == 'posix':
        multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    # Inference args
    parser.add_argument("-t", "--task", type=str, help="The task to perform.", required=True)
    parser.add_argument("-st", "--subtask", type=str, help="The subtask of the task to perform.")
    parser.add_argument("-c", "--config", type=str, default='default',
                        help="Name of a default, predefined config or path to the appropriate .yaml config file.")
    parser.add_argument("-io", "--io_config_path", type=str, help="Path to the appropriate .yaml io config file.")
    parser.add_argument("-i", "--imagery_path", type=str, help="Path to the imagery.")
    parser.add_argument("-o", "--output_path", type=str, help="Path to the output folder.")
    parser.add_argument("-gt", "--ground_truth_path", type=str, help="Path to the ground truth data.")
    parser.add_argument("-aoi", "--aoi_path", type=str, help="Path to the area of interest (AOI) geopackage.")
    
    # Training args
    parser.add_argument("-d", "--dataset", type=str, help="Path to the root folder of the dataset to use for training a model. Will override whatever is in the yaml config file.")

    args = parser.parse_args()

    task = args.task
    subtask = args.subtask

    if task == "pipeline":
        pipeline_main(args)
    elif task == "train" and subtask == "detector":
        train_detector_main(args)
    else:
        raise ValueError("Invalid task or subtask.")




