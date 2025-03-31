import subprocess
import time

from experiments.resolution.evaluate.get_wandb import wandb_runs_to_dataframe


def launch_evaluate(wandb_project, extent, architecture):
    print(f"Extent: {extent}, Architecture: {architecture}")

    cmd = ["sbatch",]
    cmd.extend([
        'experiments/resolution/evaluate/sbatch_evaluate.sh',
        dataset_config["compressed"],
        config_path
    ])

    # Submit the job
    subprocess.run(cmd)

    # Small delay to avoid overwhelming the scheduler
    time.sleep(1)


if __name__ == "__main__":
    extent = '40m'
    wandb_project = f"hugobaudchon_team/detector_experience_resolution_optimalHPs_{extent}"
    df = wandb_runs_to_dataframe(wandb_project)

    architectures = df['architecture'].unique()

    for architecture in architectures:
        launch_evaluate(
            wandb_project,
            extent,
            architecture
        )
