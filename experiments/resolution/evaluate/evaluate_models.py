import subprocess
import time
from pathlib import Path

from experiments.resolution.evaluate.get_wandb import wandb_runs_to_dataframe


def launch_evaluate(wandb_project, extent, architecture):
    print(f"Extent: {extent}, Architecture: {architecture}")

    root_models_path = f"/home/mila/h/hugo.baudchon/scratch/training/detector_experience_resolution_optimalHPs_{extent}_FIXED"
    output_folder = f"/home/mila/h/hugo.baudchon/scratch/eval/detector_experience_resolution_optimalHPs_{extent}_FIXED/{architecture.split('.')[0].split('/')[-1]}"
    Path(output_folder).mkdir(parents=True, exist_ok=False)

    partition_arg = '--partition=long'

    cmd = ["sbatch", partition_arg]
    cmd.extend([
        'experiments/resolution/evaluate/sbatch_evaluate.sh',
        wandb_project,
        architecture,
        extent,
        root_models_path,
        output_folder
    ])

    # Submit the job
    subprocess.run(cmd)

    # Small delay to avoid overwhelming the scheduler
    time.sleep(1)


if __name__ == "__main__":
    extent = '40m'
    wandb_project = f"hugobaudchon_team/detector_experience_resolution_optimalHPs_{extent}_FIXED"
    df = wandb_runs_to_dataframe(wandb_project)

    architectures = df['architecture'].unique()

    for architecture in architectures:
        launch_evaluate(
            wandb_project,
            extent,
            architecture
        )
