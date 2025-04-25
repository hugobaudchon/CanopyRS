#!/bin/bash
#SBATCH --output=./jobs_eval/job_output_%j.txt        # Standard output (%j expands to job ID)
#SBATCH --time=48:00:00                   # Time limit (hh:mm:ss) – adjust as needed
#SBATCH --nodes=1                         # Request one node
#SBATCH --ntasks=1                        # Request one task (process)
#SBATCH --cpus-per-task=10                # Number of CPU cores per task
#SBATCH --mem=80G                        # Total memory per node
#SBATCH --gres=gpu:l40s:1                 # Request 4 A100 GPUs         #a100l for 80gb

module load cuda/12.1.1/cudnn/9.3
module load anaconda/3
source $CONDA_ACTIVATE
conda activate canopyrs
# pip freeze

SOURCE_RAW_DATASET_FOLDER=$SCRATCH/data/raw
LOCAL_RAW_DATASET_FOLDER=$SLURM_TMPDIR/data
LOCAL_TILERIZED_DATASET_FOLDER=$SLURM_TMPDIR/tilerized
mkdir -p $LOCAL_RAW_DATASET_FOLDER
mkdir -p $LOCAL_TILERIZED_DATASET_FOLDER

#cp -r $SOURCE_RAW_DATASET_FOLDER/{brazil_zf2,ecuador_tiputini,panama_aguasalud} $LOCAL_RAW_DATASET_FOLDER/
#ls $LOCAL_RAW_DATASET_FOLDER

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR="127.0.0.1"

srun python experiments/resolution/evaluate/evaluate_extent_and_architecture.py\
 --wandb_project $1\
 --architecture $2\
 --extent $3\
 --raw_root $SOURCE_RAW_DATASET_FOLDER\
 --models_root $4\
 --output_folder $5\
 --temp_folder $LOCAL_TILERIZED_DATASET_FOLDER
