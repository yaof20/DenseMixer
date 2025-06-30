DATASET_NAME=gsm
LORA_ROUTER=unfrozen
LR=1e-4
NUM_EPOCHS=4
SEED=42
BATCH_SIZE=256
EXP_NAME="OLMOE-${DATASET_NAME}-lora-all-${LORA_ROUTER}__LR${LR}__epochs${NUM_EPOCHS}__bs${BATCH_SIZE}__seed${SEED}"
CACHE_DIR="xxx"

python open_instruct/merge_lora.py \
    --base_model_name_or_path allenai/OLMoE-1B-7B-0125 \
    --output_dir output/${DATASET_NAME}/olmoe_lora_all_merged_$LORA_ROUTER \
    --lora_model_name_or_path output/${DATASET_NAME}/olmoe-lora-all-unfrozen-4e-4 \
    --use_fast_tokenizer \
    --save_tokenizer \
    --cache_dir "$CACHE_DIR" \
    --exp_name "$EXP_NAME" \
    --lora_router "$LORA_ROUTER" \
    --push_to_hub