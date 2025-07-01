#!/bin/bash
#SBATCH --job-name=upload-ckpt
#SBATCH --partition=cpuonly        # Use the CPU-only partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         # More CPUs for handling large files
#SBATCH --mem=64G                  # Much more memory for 30B model checkpoints
# No GPU allocation at all
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.errs
#SBATCH --time=100:00:00

echo "=========================================="
echo "Upload monitor started at: $(date)"
echo "Running on CPU-only partition"
echo "=========================================="

# Continuous loop every 30 minutes

cd /mnt/weka/home/shibo.hao/feng/code/MoE-Finetune/LLaMA-Factory/scripts

while true; do
    echo "=========================================="
    echo "Starting upload cycle at: $(date)"
    echo "=========================================="
    
    # Run your upload script
    export HUGGINGFACE_HUB_TOKEN=""
    bash upload_hf_full.sh
    export HUGGINGFACE_HUB_TOKEN=""
    # bash upload_hf_ste.sh
    
    echo "=========================================="
    echo "Upload cycle completed at: $(date)"
    echo "Waiting 30 minutes for next cycle..."
    echo "=========================================="
    
    # Wait 30 minutes (1800 seconds)
    sleep 1800
done