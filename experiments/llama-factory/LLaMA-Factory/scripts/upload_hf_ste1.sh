export HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN"
export HF_TOKEN="$HF_TOKEN"

#!/bin/bash

# Base configuration
# local_path="/lustrefs/users/shibo.hao/feng/code/MoE-Finetune/LLaMA-Factory/saves/qwen3-30b/"
local_path="/mnt/weka/home/shibo.hao/feng/code/MoE-Finetune/LLaMA-Factory/saves/qwen3-30b/"
method="ste"

experiment_names=(
    # "s1_sl16k_bs16_lr2e-6"
    # "s1_sl16k_bs16_lr5e-6"
    # "s1_sl16k_bs16_lr1e-5"
    # "s1_sl16k_bs16_lr3e-5"
    # "s1_sl16k_bs16_lr5e-5"
    "s1_sl16k_bs16_lr1e-5_zero3_partscale_fixed_ex_normdtch"
    # "s1_sl16k_bs16_lr1e-5_zero3_partscale_fixed_ex"
    # "s1_sl16k_bs16_lr1e-5_zero3_gate_fixed_tau"
    # "s1_sl16k_bs16_lr1e-5_zero3_gate_fixed"
    # "s1_sl16k_bs16_lr1e-5_zero3_gate"
    # "s1_sl16k_bs32_lr5e-6"
    # "s1_sl16k_bs32_lr1e-5"
    # "s1_sl16k_bs32_lr3e-5"
    # "s1_sl16k_bs32_lr5e-5"
    # "nemosci_sl16k_bs64_lr5e-6"
    # "nemosci_sl16k_bs64_lr1e-5"
    # "nemosci_sl16k_bs64_lr3e-5"
    # "nemosci_sl16k_bs64_lr5e-5"
    # "nemocode_sl16k_bs64_lr3e-5"
    # "nemocode_sl16k_bs64_lr1e-5"
    # "amthink_sl16k_bs64_lr5e-5"

    # "nemocode_sl16k_bs64_lr5e-5"
    # "nemocode_sl16k_bs64_lr5e-6"
)

# Loop through each experiment
for experiment_name in "${experiment_names[@]}"; do
    echo "========================================"
    echo "Processing experiment: $experiment_name"
    echo "========================================"
    
    # Loop through checkpoints for this experiment
    # for ckpt in 553 100 200 300 400 500 ; do
    # for ckpt in 553 ; do
    # for ckpt in 1044 261 522 783  ; do
    for ckpt in 285 57 114 171 228; do
    # for ckpt in 345 690 1035; do
    # for ckpt in 157 314 471; do
    # for ckpt in 29 58 87 116 140; do
        echo "Processing checkpoint-$ckpt for $experiment_name..."
        
        # FIXED: Use consistent path for both checking and uploading
        # checkpoint_path="${local_path}/${method}/${experiment_name}_zero3/checkpoint-$ckpt"
        checkpoint_path="${local_path}/${method}/${experiment_name}/checkpoint-$ckpt"
        
        # Check if checkpoint directory exists
        if [ ! -d "$checkpoint_path" ]; then
            echo "Warning: Directory for checkpoint-$ckpt does not exist at $checkpoint_path. Skipping..."
            continue
        fi
        
        repo_id="fengyao1909/${method}_${experiment_name}_ckpt${ckpt}"
        
        echo "Uploading to repository: $repo_id"
        echo "From path: $checkpoint_path"
        
        python upload_hf.py \
            --local_folder "$checkpoint_path" \
            --repo_id "$repo_id" \
            --repo_type "model" \
            --token ""
        
        # Check if upload was successful
        if [ $? -eq 0 ]; then
            echo "✓ Successfully uploaded checkpoint-$ckpt for $experiment_name"
            
            # SAFETY: Double-check the path exists before deletion
            if [ -d "$checkpoint_path" ]; then
                echo "you can delete $checkpoint_path now"
                # rm -rf "$checkpoint_path"
                # echo "✓ Deleted after upload: $checkpoint_path"
            else
                echo "⚠️  Warning: Path $checkpoint_path not found for deletion"
            fi
            echo "----------------------------------------"
        else
            echo "✗ Failed to upload checkpoint-$ckpt for $experiment_name"
            echo "⚠️  Checkpoint NOT deleted due to upload failure"
        fi
    done
    
    # echo "Completed experiment: $experiment_name"
    echo ""
done

echo "Scanned for 1 round"
# echo "All experiments and checkpoints upload process completed!"