# Below is a list of the command-line arguments used in the script:
# - --task (codealpaca, gsm, intent, law, summary, translation)
# - --model (olmoe)
# - --num_gpus
# - --devices (cuda visible devices)
# - --port (deepspeed port)
# - --output_suffix (the output directory will be output/{task}/{model}_{finetuning method}{suffix})
# - --user (experiment personal prefix)
# - --per_device_train_batch_size
# - --per_device_eval_batch_size
# - --total_batch_size
# - --max_seq_length
# - --lr
# - --lr_type (learning rate scheduler type)
# - --warmup_ratio
# - --weight_decay
# - --num_train_epochs
# - --cache_dir
# - --freeze_gate (False, True)
# - --lora_router (frozen, unfrozen)

# For lora, we also have
# - --lora_rank
# - --lora_alpha
# - --lora_dropout

# For esft, we also have
# - --expert_type (gate, token)


# Frozen Router-Full
bash scripts/train/finetune/full.sh \
    --task gsm \
    --model olmoe \
    --total_batch_size 256 \
    --num_train_epochs 4 \
    --num_gpus 2 \
    --devices 0,1 \
    --port 29000 \
    --lr 1e-6 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --freeze_gate True \
    --gradient_checkpointing false