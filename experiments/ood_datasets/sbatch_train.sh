#!/bin/bash
#SBATCH --output=./jobs_ood_datasets/job_output_%j.txt        # Standard output (%j expands to job ID)
#SBATCH --time=24:00:00                   # Time limit (hh:mm:ss) – adjust as needed
#SBATCH --nodes=1                         # Request one node
#SBATCH --ntasks=1                        # Request one task (process)
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --mem=40G                        # Total memory per node
#SBATCH --gres=gpu:rtx8000:1                 # Request 4 A100 GPUs         #a100l for 80gb

module load cuda/12.1.1/cudnn/9.3
module load anaconda/3
source $CONDA_ACTIVATE
conda activate canopyrs
# pip freeze

CONFIG_PATH=$1

# The remaining arguments are the compressed dataset names
echo "Using config file: $CONFIG_PATH"

# Create a local folder for the datasets
LOCAL_DATASET_FOLDER=$SLURM_TMPDIR/data
mkdir -p $LOCAL_DATASET_FOLDER

# Loop over all dataset names and extract each one
tar -xf $SCRATCH/data/final_multires_multidataset.tar.gz -C $LOCAL_DATASET_FOLDER

ls $LOCAL_DATASET_FOLDER

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR="127.0.0.1"

# Launch the training job using the provided config and extracted datasets folder
srun python train.py \
    -m detector \
    -c $CONFIG_PATH \
    -d $LOCAL_DATASET_FOLDER
