#!/bin/bash

# create log dir (if not exist)
LOGS_DIR="logs"
mkdir -p $LOGS_DIR

# define config name as variable
method="full"

# list of config files to run
CONFIG_FILES=(
    "base_s1k_bs16_lr1e-5_zero3.yaml"
    # "base_s1_sl16k_bs16_lr2e-6_zero3.yaml"
    # "base_s1_sl16k_bs16_lr5e-6_zero3.yaml"
    # "base_s1_sl16k_bs16_lr1e-5_zero3.yaml"
    # "base_s1_sl16k_bs16_lr3e-5_zero3.yaml"
    # "base_s1_sl16k_bs16_lr5e-5_zero3.yaml"
    # "base_s1_sl16k_bs32_lr1e-5_zero3.yaml"
    # "base_s1_sl16k_bs32_lr3e-5_zero3.yaml"
    # "base_s1_sl16k_bs32_lr5e-5_zero3.yaml"
    # "base_s1_sl16k_bs32_lr5e-6_zero3.yaml"
)

export WANDB_API_KEY="$WANDB_API_KEY"
export WANDB_PROJECT="MoE-Finetune-Rerun"
export DISABLE_VERSION_CHECK=1

# loop through config files
for config_file in "${CONFIG_FILES[@]}"; do
    # create log name without .yaml extension
    LOG_FILE="$LOGS_DIR/train_${method}_${config_file%.*}.log"
    
    # print starting info
    echo "========================================"
    echo "start training: $(date)"
    echo "config: $config_file"
    echo "log saved at: $LOG_FILE"
    
    # define the command
    CMD="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train examples/train_full/qwen3-30b/${method}/${config_file}"
    
    # log the command
    echo "executing command: $CMD" | tee $LOG_FILE
    echo "----------------------------------------" | tee -a $LOG_FILE
    
    # execute the command and append to log file
    eval "$CMD" 2>&1 | tee -a $LOG_FILE
    
    echo "completed training for: $config_file"
    echo "========================================"
done

echo "All training jobs completed at: $(date)"
