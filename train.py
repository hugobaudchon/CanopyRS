import argparse
import logging
import warnings

from engine.models.utils import set_all_seeds
from engine.utils import init_spawn_method

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Importing from timm.models.layers is deprecated"
)
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument."
)
detrex_logger = logging.getLogger("detrex.checkpoint.c2_model_loading")
detrex_logger.disabled = True

from engine.config_parsers import DetectorConfig
from engine.models.detector.train_detectron2.train_detectron2 import train_detectron2_fasterrcnn
from engine.models.detector.train_detectron2.train_detrex import train_detrex, eval_detrex


def train_detector_main(args):
    config = DetectorConfig.from_yaml(args.config)

    if args.dataset:
        config.data_root_path = args.dataset

    if config.seed:
        set_all_seeds(config.seed)

    if config.model == 'faster_rcnn_detectron2':
        train_detectron2_fasterrcnn(config)
    elif config.model == 'dino_detrex':
        train_detrex(config)
    else:
        raise ValueError("Invalid model type/name.")


def eval_detector_main(args):
    config = DetectorConfig.from_yaml(args.config)

    if args.dataset:
        config.data_root_path = args.dataset

    if config.seed:
        set_all_seeds(config.seed)

    if config.model == 'faster_rcnn_detectron2':
        raise NotImplementedError("Evaluation for FasterRCNN is not yet implemented.")
    elif config.model == 'dino_detrex':
        eval_detrex(config, fold_name=args.eval_only_fold)
    else:
        raise ValueError("Invalid model type/name.")


if __name__ == '__main__':
    init_spawn_method()
    parser = argparse.ArgumentParser()

    # Training args
    parser.add_argument("-m", "--model", type=str, help="The type of model to train (detector, segmenter, classifier...).", required=True)
    parser.add_argument("-c", "--config", type=str, default='default', help="Name of a default, predefined config or path to the appropriate .yaml config file.")
    parser.add_argument("-d", "--dataset", type=str, help="Path to the root folder of the dataset to use for training a model. Will override whatever is in the yaml config file.")

    parser.add_argument("--eval_only_fold", type=str, help="Whether to only evaluate the model. If defined, the value passed will be used as the fold name to find and load the appropriate data.")

    args = parser.parse_args()

    if args.model == "detector":
        if args.eval_only_fold:
            eval_detector_main(args)
        else:
            train_detector_main(args)




