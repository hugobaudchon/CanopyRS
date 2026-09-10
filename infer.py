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


def _io_aoi(io_config: InferIOConfig):
    """The AOI (a gpkg path) from an InferIOConfig, or None. Only package mode carries over;
    generate/whole-raster tiles the full extent (the default when no AOI)."""
    if io_config.aoi_config == 'package':
        return io_config.aoi
    if io_config.aoi_config in (None, 'generate'):
        return None
    raise ValueError(f"Unsupported aoi_config: {io_config.aoi_config}")


def pipeline_main(args):
    config = PipelineConfig.from_yaml(get_config_path(f'{args.config}'))
    steps = list(config.components_configs)

    if args.resume_from and args.initialize_from:
        raise ValueError("Pass only one of --resume_from / --initialize_from.")

    objects = None
    if args.io_config_path:
        if args.imagery_path or args.tiles_path or args.output_path:
            raise ValueError("Provide either an -io config file or imagery/tiles + output args, not both.")
        io_config = InferIOConfig.from_yaml(args.io_config_path)
        sources, tiles = io_config.input_imagery, io_config.tiles_path
        output_folder, aoi = io_config.output_folder, _io_aoi(io_config)
        objects = io_config.input_gpkg
        if io_config.input_coco:
            raise NotImplementedError("input_coco seeding is not yet supported by the pipeline.")
    else:
        sources, tiles, aoi = args.imagery_path, args.tiles_path, args.aoi_path
        if args.resume_from:
            # Resume in-place by default; an explicit -o resumes into a new folder (cross-folder).
            output_folder = args.output_path or args.resume_from
        else:
            output_folder = args.output_path
        if not output_folder:
            raise ValueError("Provide -o/--output_path (or an -io config file).")

    # Drop a leading tilerizer when tiles are provided directly (no raster to tile).
    if tiles and not sources and steps and steps[0][0] == 'tilerizer':
        warn('Removing the leading tilerizer from the pipeline, since tiles are provided as input.')
        steps = steps[1:]

    pipeline = Pipeline.from_config(
        steps,
        sources=sources,
        tiles=tiles,
        objects=objects,
        output_dir=output_folder,
        aoi=aoi,
        resume_from=args.resume_from,
        initialize_from=args.initialize_from,
        num_workers=config.num_workers,
    )
    pipeline.run()


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
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a previous run's output folder to resume (same config). "
                             "Skips already-finished components. Resumes in-place unless -o is given.")
    parser.add_argument("--initialize_from", type=str, default=None,
                        help="Path to a previous run's output folder whose outputs seed the inputs "
                             "of this (new-config) pipeline. Requires -o for the new output folder.")

    args = parser.parse_args()

    pipeline_main(args)




