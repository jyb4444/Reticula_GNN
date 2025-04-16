#!/bin/bash
#SBATCH --job-name=stratified_kfold_train
#SBATCH --output=/mnt/home/yuankeji/RanceLab/validation/k-fold-gnn-train/gtex_kfold_gnn_%A_%a.out
#SBATCH --error=/mnt/home/yuankeji/RanceLab/validation/k-fold-gnn-train/gtex_kfold_gnn_%A_%a.err
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=1-10
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ykj2018720@gmail.com

# This is an array job that processes all 10 folds in parallel
# The SLURM_ARRAY_TASK_ID variable (0-9) determines which fold to process

source /mnt/home/yuankeji/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/scratch/yuankeji/rpy_env

BASE_DIR="/mnt/home/yuankeji/RanceLab/reticula_new/reticula"
SCRIPT_DIR="/mnt/home/yuankeji/RanceLab/validation"
DATA_DIR="${BASE_DIR}/data/gtex"
INPUT_DIR="${DATA_DIR}/input"
OUTPUT_DIR="/mnt/home/yuankeji/RanceLab/validation/k-fold-gnn-train/"
GNN_DIR="${DATA_DIR}/GNN"

R_SCRIPT="${SCRIPT_DIR}/stratified_kfold_data.R"
PYTHON_TRAIN_SCRIPT="${SCRIPT_DIR}/stratified_kfold_train.py"
# PYTHON_VALIDATION_SCRIPT="${SCRIPT_DIR}/stratified_kfold_validation.py"

FOLD=${SLURM_ARRAY_TASK_ID}
echo "Processing fold ${FOLD} (SLURM job array task ID: ${SLURM_ARRAY_TASK_ID})"

if [ ! -f "${INPUT_DIR}/node_features_train_${FOLD}.txt" ] || [ ! -f "${INPUT_DIR}/node_features_val_${FOLD}.txt" ]; then
    echo "Required input files for fold ${FOLD} not found."
    echo "Please run the R preprocessing script first or check file paths."
    exit 1
fi

if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
else
    echo "No GPU detected, using CPU"
fi

echo "Job started at $(date)"

echo "Starting GNN training for fold ${FOLD} at $(date)"
python ${PYTHON_TRAIN_SCRIPT} --fold ${FOLD} --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR}

if [ $? -ne 0 ]; then
    echo "Error: GNN training failed for fold ${FOLD}. Exiting."
    exit 1
fi

# echo "Starting GNN validation for fold ${FOLD} at $(date)"
# python ${PYTHON_VALIDATION_SCRIPT} --fold ${FOLD} --input_dir ${INPUT_DIR} --output_dir ${GNN_DIR}

# if [ $? -ne 0 ]; then
#     echo "Error: GNN validation failed for fold ${FOLD}. Exiting."
#     exit 1
# fi

echo "All processes for fold ${FOLD} completed at $(date)"

exit 0