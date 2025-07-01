#!/usr/bin/env bash
# This script exports selected LoRA merge configs using llamafactory-cli
set -e

MERGE_DIR="/mnt/weka/home/shibo.hao/feng/code/MoE-Finetune/LLaMA-Factory/examples/merge_lora"

if [ ! -d "$MERGE_DIR" ]; then
    echo "Error: directory $MERGE_DIR does not exist." >&2
    exit 1
fi

files=(
#   "qwen1.5moe_esft_ammath_lr1e-4_unfrozen.yaml"
#   "qwen1.5moe_esft_gsm_lr1e-4_unfrozen.yaml"
#   "qwen1.5moe_esft_intent_lr1e-4_unfrozen.yaml"
#   "qwen1.5moe_esft_law_lr1e-4_unfrozen.yaml"
#   "qwen1.5moe_esft_summary_lr1e-4_unfrozen.yaml"
#   "qwen1.5moe_esft_translation_lr1e-4_unfrozen.yaml"
qwen1.5moe_esft_gsm_lr2e-4_unfrozen_ckpt117.yaml
qwen1.5moe_esft_gsm_lr2e-4_unfrozen_ckpt234.yaml
qwen1.5moe_esft_gsm_lr2e-4_unfrozen_ckpt351.yaml
qwen1.5moe_esft_gsm_lr3e-4_unfrozen_ckpt117.yaml
qwen1.5moe_esft_gsm_lr3e-4_unfrozen_ckpt234.yaml
qwen1.5moe_esft_gsm_lr3e-4_unfrozen_ckpt351.yaml

#   "qwen1.5moe_esft_law_lr5e-4_unfrozen_ckpt60.yaml"


)

for fname in "${files[@]}"; do
    config_path="$MERGE_DIR/$fname"
    if [ -f "$config_path" ]; then
        echo "Exporting LoRA merge for $config_path..."
        llamafactory-cli export "$config_path"
    else
        echo "Error: file $config_path not found" >&2
        exit 1
    fi
done

echo "Selected exports completed."
