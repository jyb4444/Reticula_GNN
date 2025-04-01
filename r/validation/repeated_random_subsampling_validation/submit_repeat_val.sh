#!/bin/bash
#SBATCH --job-name=rrs_val
#SBATCH --output=/mnt/home/yuankeji/RanceLab/validation/rrs-val/gnn_trial_%A_%a.out
#SBATCH --error=/mnt/home/yuankeji/RanceLab/validation/rrs-val/gnn_trial_%A_%a.err
#SBATCH --array=1-10
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=normal

source /mnt/home/yuankeji/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/scratch/yuankeji/rpy_env

# 设置工作目录
cd /mnt/home/yuankeji/RanceLab/reticula_new/reticula/validation/

python rrs_val.py --trial_id ${SLURM_ARRAY_TASK_ID}

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "Running on node: $SLURM_NODEID"
echo "Start time: $(date)"

echo "End time: $(date)"