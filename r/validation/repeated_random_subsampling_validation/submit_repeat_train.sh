#!/bin/bash
#SBATCH --job-name=rrs_train         # Job name
#SBATCH --output=/mnt/home/yuankeji/RanceLab/validation/rrs-train/rrs_gnn_train_%A_%a.out 
#SBATCH --error=/mnt/home/yuankeji/RanceLab/validation/rrs-train/rrs_gnn_train_%A_%a.err  
#SBATCH --array=1-10                 # Array job with 10 tasks (0-9)
#SBATCH --nodes=1                   # Request 1 node per task
#SBATCH --ntasks=1                  # Run 1 task per node
#SBATCH --cpus-per-task=4           # Request 4 CPUs per task
#SBATCH --mem=16G                   # Request 16GB memory per node
#SBATCH --time=72:00:00             # Set time limit to 24 hours
#SBATCH --gres=gpu:1                # Request 1 GPU per node (if needed)
#SBATCH --partition=gpu             # Use the GPU partition (adjust as needed)

source /mnt/home/yuankeji/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/scratch/yuankeji/rpy_env

BASE_OUTPUT_DIR="/mnt/scratch/yuankeji/RanceLab/reticula_new/gtex/output"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/exp_${SLURM_ARRAY_TASK_ID}"
mkdir -p ${OUTPUT_DIR}

python rrs_train.py --trial_id ${SLURM_ARRAY_TASK_ID}

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "Running on node: $SLURM_NODEID"
echo "Start time: $(date)"

# Create a directory for this trial
TRIAL_DIR="rrs_val_trial_${SLURM_ARRAY_TASK_ID}"
mkdir -p $TRIAL_DIR

echo "End time: $(date)"