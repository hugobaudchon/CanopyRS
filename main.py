import argparse

from engine.config_parsers import InferIOConfig, PipelineConfig
from engine.config_parsers.base import get_config_path
from engine.pipeline import Pipeline

def pipeline_main():
    config_path = get_config_path(f'{config_name}/pipeline')
    config = PipelineConfig.from_yaml(config_path)

    if io_config_path and (imagery_path or output_path):
        raise ValueError("Either provide an io config file or imagery path and output path.")
    elif io_config_path:
        io_config = InferIOConfig.from_config_path(io_config_path)
    elif imagery_path and output_path:
        io_config = InferIOConfig(
            input_imagery=imagery_path,
            output_folder=output_path
        )
    else:
        raise ValueError("Provide either an io config file or imagery path and output path.")

    Pipeline(io_config, config).run()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, help="The task to perform.", required=True)
    parser.add_argument("-st", "--subtask", type=str, help="The subtask of the task to perform.")
    parser.add_argument("-c", "--config", type=str, default='default',
                        help="Name of a default, predefined config or path to the appropriate .yaml config file.")
    parser.add_argument("-io", "--io_config_path", type=str, help="Path to the appropriate .yaml io config file.")
    parser.add_argument("-i", "--imagery_path", type=str, help="Path to the imagery.")
    parser.add_argument("-o", "--output_path", type=str, help="Path to the output folder.")

    args = parser.parse_args()

    task = args.task
    subtask = args.subtask
    config_name = args.config
    io_config_path = args.io_config_path
    imagery_path = args.imagery_path
    output_path = args.output_path

    if task == "pipeline":
        pipeline_main()




