#!/bin/bash

# Set the CUDA device
# echo "Using CUDA device: 3"
# export CUDA_VISIBLE_DEVICES=2,4

# Define arguments
DATASET_NAME="RoxanneWsyw/ESFT-summary"
FILE="test.jsonl"
OUTPUT_DIR="../summary/expert_eval_results"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

BASE_MODEL_PATH="allenai/OLMoE-1B-7B-0125"
WORLD_SIZE=4
GPUS_PER_RANK=1
N_SAMPLE_TOKENS=300000
MAX_SEQ_LENGTH=4096
PREPROCESSING_NUM_WORKERS=4

# Run the expert evaluation script
CUDA_VISIBLE_DEVICES=0,1,2,3 python get_expert_scores.py \
    --dataset_name "$DATASET_NAME" \
    --file "$FILE" \
    --output_dir "$OUTPUT_DIR" \
    --base_model_path "$BASE_MODEL_PATH" \
    --world_size "$WORLD_SIZE" \
    --gpus_per_rank "$GPUS_PER_RANK" \
    --n_sample_tokens "$N_SAMPLE_TOKENS" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --preprocessing_num_workers "$PREPROCESSING_NUM_WORKERS" \
