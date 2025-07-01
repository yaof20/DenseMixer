#!/bin/bash

# mkdir -p logs
log_file="logs/eval_single_$(date '+%Y%m%d_%H%M%S').log"

export CUDA_VISIBLE_DEVICES=2
python eval.py \
    --eval_datasets=intent \
    --model_path=allenai/OLMoE-1B-7B-0125 \
    --output_dir=results/intent \
    --max_new_tokens=512 \
    --openai_api_key="xx" \
    --eval_batch_size=2 \
    --gpu_count 1 \