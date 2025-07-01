#!/bin/bash

model_list=("allenai/allenai/OLMoE-1B-7B-0125")

export HF_ALLOW_CODE_EVAL="1"

export CUDA_VISIBLE_DEVICES=2,3

task_log="evaluation_code_dense.log"
required_gpus=2


> "$task_log"

tasks=("mbpp" "humaneval")

for model_nm in "${model_list[@]}"; do
    echo "Evaluating model: $model_nm"

    for task in "${tasks[@]}"; do
        echo "Running task: $task"

        CUDA_VISIBLE_DEVICES=3,4 accelerate launch \
            --num_processes $required_gpus \
            --multi_gpu \
            --mixed_precision bf16 \
            -m lm_eval \
            --model hf \
            --model_args "pretrained=$model_nm" \
            --output_path "res_${task}_$(basename "$model_nm")" \
            --log_samples \
            --task "$task" \
            --batch_size 16 \
            --device "cuda" \
            --confirm_run_unsafe_code 2>&1 | tee -a "$task_log"
    done
done



