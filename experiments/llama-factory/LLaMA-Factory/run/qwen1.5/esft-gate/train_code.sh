#!/bin/bash

# create log dir (if not exist)
LOGS_DIR="logs"
mkdir -p $LOGS_DIR

# define config name as variable
method="esft-gate"

# list of config files to run

CONFIG_FILES=(
    # "qwen1.5_esft_law_lr5e-6.yaml"
    # "qwen1.5_codealpaca_lr1e-5.yaml"
    "qwen1.5_codealpaca_lr5e-6.yaml"
    # "qwen1.5_codealpaca_lr1e-6.yaml"
    # "qwen1.5_esft_intent_lr5e-6.yaml"
    # "qwen1.5_esft_intent_lr1e-6.yaml"
    # "qwen1.5_esft_translation_lr1e-5.yaml"
    # "qwen1.5_esft_translation_lr5e-6.yaml"
    # "qwen1.5_esft_translation_lr1e-6.yaml"
    # "qwen1.5_esft_summary_lr1e-5.yaml"
    # "qwen1.5_esft_summary_lr5e-6.yaml"
    # "qwen1.5_esft_summary_lr1e-6.yaml"
    # "qwen1.5_esft_intent_lr2e-5.yaml"
)

export WANDB_API_KEY="$WANDB_API_KEY"
export WANDB_PROJECT="MoE-Finetune-qwen1.5"
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
    CMD="CUDA_VISIBLE_DEVICES=4,5,6,7 llamafactory-cli train examples/train_full/qwen1.5moe/${method}/${config_file}"
    
    # log the command
    echo "executing command: $CMD" | tee $LOG_FILE
    echo "----------------------------------------" | tee -a $LOG_FILE
    
    # execute the command and append to log file
    eval "$CMD" 2>&1 | tee -a $LOG_FILE
    
    echo "completed training for: $config_file"
    echo "========================================"
done

echo "All training jobs completed at: $(date)"
