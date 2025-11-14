#!/bin/bash
#SBATCH --output=./jobs_eval_multires/job_output_%j.txt        # Standard output (%j expands to job ID)
#SBATCH --time=16:00:00                   # Time limit (hh:mm:ss) – adjust as needed
#SBATCH --nodes=1                         # Request one node
#SBATCH --ntasks=1                        # Request one task (process)
#SBATCH --cpus-per-task=12                # Number of CPU cores per task
#SBATCH --mem=50G                        # Total memory per node
#SBATCH --gres=gpu:l40s:1                  # Request 4 A100 GPUs         #a100l for 80gb

module load cuda/12.1.1/cudnn/9.3
module load anaconda/3
source $CONDA_ACTIVATE
conda activate canopyrs
# pip freeze

AUGMENTATION_IMAGE_SIZE="$1"           # 800, 1333, 1777

if [[ -z "$AUGMENTATION_IMAGE_SIZE" ]]; then
  echo "Usage: sbatch $0 <AUGMENTATION_IMAGE_SIZE>"
  exit 1
fi

python experiments/multi_resolution/evaluate/test.py "$AUGMENTATION_IMAGE_SIZE"

