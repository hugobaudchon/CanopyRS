import argparse
import logging
import warnings

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

from engine.config_parsers import InferIOConfig, PipelineConfig
from engine.config_parsers.base import get_config_path
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
            'output_folder': args.output_path,
            'tiles_path': args.tiles_path
        }
        if args.aoi_path:
            config_args['aoi_config'] = 'package'
            config_args['aoi'] = args.aoi_path

        io_config = InferIOConfig(**config_args)
    else:
        raise ValueError("Provide either an io config file or imagery path and output path.")

    pipeline = Pipeline(io_config, config)
    pipeline()


if __name__ == '__main__':
    init_spawn_method()
    parser = argparse.ArgumentParser()

    # Inference args
    parser.add_argument("-c", "--config", type=str, default='default', help="Name of a default, predefined config or path to the appropriate .yaml config file.")
    parser.add_argument("-io", "--io_config_path", type=str, help="Path to the appropriate .yaml io config file.")
    parser.add_argument("-i", "--imagery_path", type=str, help="Path to the imagery.")
    parser.add_argument("-o", "--output_path", type=str, help="Path to the output folder.")
    parser.add_argument("-t", "--tiles_path", type=str, help="Path to the tiles folder to infer on.")
    parser.add_argument("-aoi", "--aoi_path", type=str, help="Path to the area of interest (AOI) geopackage.")

    args = parser.parse_args()

    pipeline_main(args)




