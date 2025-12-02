import argparse
import logging
import warnings
from warnings import warn

from canopyrs.engine.utils import init_spawn_method

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Importing from timm.models.layers is deprecated"
)
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument."
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="pkg_resources is deprecated as an API.*"
)
detrex_logger = logging.getLogger("detrex.checkpoint.c2_model_loading")
detrex_logger.disabled = True

from canopyrs.engine.config_parsers import InferIOConfig, PipelineConfig
from canopyrs.engine.config_parsers.base import get_config_path
from canopyrs.engine.pipeline import Pipeline


def pipeline_main(args):
    config_path = get_config_path(f'{args.config}')
    config = PipelineConfig.from_yaml(config_path)

    if args.io_config_path and (args.imagery_path or args.output_path):
        raise ValueError("Either provide an io config file or pass imagery/tiles path and output path as arguments.")
    elif args.io_config_path:
        io_config = InferIOConfig.from_yaml(args.io_config_path)
    elif (args.imagery_path or args.tiles_path) and args.output_path:
        # Check if the first component is 'tilerizer' and remove it if tiles are provided (i.e. tilerizer is not needed).
        if args.tiles_path and args.imagery_path is None and config.components_configs[0][0] == 'tilerizer':
            warn('Removing the first component (tilerizer) from the pipeline as it is not needed, since tiles are already provided as input.')
            config.components_configs.pop(0)

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
        raise ValueError("Either provide an io config file or pass imagery/tiles path and output path as arguments.")

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




