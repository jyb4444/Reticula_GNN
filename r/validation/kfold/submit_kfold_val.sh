#!/bin/bash
#SBATCH --job-name=stratified_kfold_val
#SBATCH --output=/mnt/home/yuankeji/RanceLab/validation/k-fold-gnn-val/gtex_kfold_val_%A_%a.out
#SBATCH --error=/mnt/home/yuankeji/RanceLab/validation/k-fold-gnn-val/gtex_kfold_val_%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4  # Keep this as is - good for CPU-based processing
#SBATCH --mem=16G
#SBATCH --array=1-10
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ykj2018720@gmail.com

# This is an array job that processes all 10 folds in parallel (validation only)
# The SLURM_ARRAY_TASK_ID variable (1-10) determines which fold to process

source /mnt/home/yuankeji/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/scratch/yuankeji/rpy_env

BASE_DIR="/mnt/home/yuankeji/RanceLab/reticula_new/reticula"
SCRIPT_DIR="/mnt/home/yuankeji/RanceLab/validation"
DATA_DIR="${BASE_DIR}/data/gtex"
INPUT_DIR="${DATA_DIR}/input"
GNN_DIR="${DATA_DIR}/GNN"
EDGES_FILE="${INPUT_DIR}/edges_val_${SLURM_ARRAY_TASK_ID}.txt"  # Changed to use validation edge file
MODEL_FILE="${GNN_DIR}/trained_pytorch_model_fold_${SLURM_ARRAY_TASK_ID}.pt"

# Set OMP_NUM_THREADS for PyTorch CPU parallelism
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Path to the updated validation script
PYTHON_VALIDATION_SCRIPT="${SCRIPT_DIR}/stratified_kfold_validation.py"

FOLD=${SLURM_ARRAY_TASK_ID}
echo "Processing validation for fold ${FOLD} (SLURM job array task ID: ${SLURM_ARRAY_TASK_ID})"

# Check if the required files exist
if [ ! -f "${INPUT_DIR}/node_features_val_${FOLD}.txt" ]; then
    echo "Required validation input file not found: ${INPUT_DIR}/node_features_val_${FOLD}.txt"
    echo "Please check file paths."
    exit 1
fi

if [ ! -f "${INPUT_DIR}/graph_targets_val_${FOLD}.txt" ]; then
    echo "Required validation input file not found: ${INPUT_DIR}/graph_targets_val_${FOLD}.txt"
    echo "Please check file paths."
    exit 1
fi

if [ ! -f "${EDGES_FILE}" ]; then
    echo "Edges file not found: ${EDGES_FILE}"
    echo "Please check file paths."
    exit 1
fi

if [ ! -f "${MODEL_FILE}" ]; then
    echo "Trained model for fold ${FOLD} not found at ${MODEL_FILE}"
    echo "Please ensure training has completed successfully."
    exit 1
fi

# Remove GPU detection since we're using CPU only
echo "Running on CPU with $SLURM_CPUS_PER_TASK threads"

echo "Job started at $(date)"

# Create output directory if it doesn't exist
mkdir -p ${GNN_DIR}

# Run the validation script with the appropriate parameters
echo "Starting GNN validation for fold ${FOLD} at $(date)"
python ${PYTHON_VALIDATION_SCRIPT} \
    --fold ${FOLD} \
    --input_dir ${INPUT_DIR} \
    --edges_fn ${EDGES_FILE} \
    --model_fn ${MODEL_FILE} \
    --output_dir ${GNN_DIR} \
    --epochs 500

if [ $? -ne 0 ]; then
    echo "Error: GNN validation failed for fold ${FOLD}. Exiting."
    exit 1
fi

echo "Validation for fold ${FOLD} completed at $(date)"

exit 0