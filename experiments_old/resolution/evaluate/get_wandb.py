import re

import pandas as pd
import wandb


def get_wandb_runs(wandb_project: str):
    api = wandb.Api()
    runs = api.runs(wandb_project)

    run_config_mapping = {}
    for run in runs:
        run_config_mapping[run.name] = run.config
        summary = run.summary._json_dict if hasattr(run.summary, '_json_dict') else run.summary
        run_config_mapping[run.name].update(summary)

        # ignore runs that crashed/finished before first eval
        if 'bbox/AP' in run_config_mapping[run.name]:
            run_config_mapping[run.name]['bbox/AP.max'] = run_config_mapping[run.name]['bbox/AP']['max']

    return run_config_mapping


def wandb_runs_to_dataframe(wandb_project: str) -> pd.DataFrame:
    runs_mapping = get_wandb_runs(wandb_project)
    # Create a DataFrame with run names as the index
    df = pd.DataFrame.from_dict(runs_mapping, orient='index')
    # Reset index to turn the run names into a column
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'run_name'}, inplace=True)
    return df


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


def extract_tilerized_image_size_regex(input_str):
    """
    Extracts the number following 'tilerized_' and preceding the next underscore.
    
    Example:
      input_str = "tilerized_888_0p5_0p045_None/panama_aguasalud"
      returns: 888 (as an integer)
    """
    matches = re.findall(r'(?<=tilerized_)(\d+)(?=_)', input_str)
    if matches:
        # Return the first occurrence as an integer.
        return int(matches[0])
    else:
        raise ValueError(f"No tilerized number found in {input_str}")
