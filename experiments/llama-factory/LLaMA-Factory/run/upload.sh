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
    bash upload_hf_ste.sh
    
    echo "=========================================="
    echo "Upload cycle completed at: $(date)"
    echo "Waiting 30 minutes for next cycle..."
    echo "=========================================="
    
    # Wait 30 minutes (1800 seconds)
    sleep 1800
done