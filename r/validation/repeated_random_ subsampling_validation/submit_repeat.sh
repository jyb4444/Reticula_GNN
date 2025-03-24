#!/bin/bash
#SBATCH --job-name=GNN_RRS         
#SBATCH --output=GNN_RRS_%j.out    
#SBATCH --error=GNN_RRS_%j.err     
#SBATCH --time=7-00:00:00          
#SBATCH --nodes=1                  
#SBATCH --ntasks=1                 
#SBATCH --cpus-per-task=4          
#SBATCH --gres=gpu:2               
#SBATCH --mem=32G                  
#SBATCH --mail-type=END,FAIL       
#SBATCH --mail-user=ykj2018720@gmail.com  

source /mnt/home/yuankeji/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/scratch/yuankeji/rpy_env

OUTPUT_DIR="/mnt/scratch/yuankeji/RanceLab/reticula_new/gtex/output/"
mkdir -p ${OUTPUT_DIR}

echo "Job started at $(date)" > ${OUTPUT_DIR}/job_info.txt
echo "Running on host: $(hostname)" >> ${OUTPUT_DIR}/job_info.txt
echo "CUDA version: $(nvcc --version | grep release)" >> ${OUTPUT_DIR}/job_info.txt
echo "Python version: $(python --version)" >> ${OUTPUT_DIR}/job_info.txt
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')" >> ${OUTPUT_DIR}/job_info.txt
echo "PyTorch Geometric version: $(python -c 'import torch_geometric; print(torch_geometric.__version__)')" >> ${OUTPUT_DIR}/job_info.txt
echo "GPU information:" >> ${OUTPUT_DIR}/job_info.txt
nvidia-smi >> ${OUTPUT_DIR}/job_info.txt

echo "Starting GNN training at $(date)" >> ${OUTPUT_DIR}/job_info.txt
python repeated_random_subsampling_validation.py  

echo "Job completed at $(date)" >> ${OUTPUT_DIR}/job_info.txt

if [ -f "${OUTPUT_DIR}/final_results.txt" ]; then
    echo "Training results:" >> ${OUTPUT_DIR}/email_summary.txt
    cat ${OUTPUT_DIR}/final_results.txt >> ${OUTPUT_DIR}/email_summary.txt
    echo "See detailed logs at: ${OUTPUT_DIR}" >> ${OUTPUT_DIR}/email_summary.txt
    mail -s "GNN Training Completed" your.email@example.com < ${OUTPUT_DIR}/email_summary.txt
fi