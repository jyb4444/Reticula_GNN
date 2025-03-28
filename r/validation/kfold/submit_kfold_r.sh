#!/bin/bash
#SBATCH --job-name=gnn_kfold           
#SBATCH --output=gnn_kfold_r_%A_%a.out   
#SBATCH --error=gnn_kfold_r_%A_%a.err    
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=4              
#SBATCH --gres=gpu:2                   
#SBATCH --mem=92G                      
#SBATCH --time=7-00:00:00              
#SBATCH --array=0-9                    

export PATH=/mnt/home/yuankeji/anaconda3/bin:$PATH

source /mnt/home/yuankeji/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/scratch/yuankeji/rpy_env

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $SLURM_NODELIST"
echo "Current fold: $SLURM_ARRAY_TASK_ID"
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

cd /mnt/home/yuankeji/RanceLab/validation/

echo "Starting R data processing at $(date)"
Rscript stratified_kfold_data.r
echo "R data processing completed at $(date)"

echo "Job completed at: $(date)"