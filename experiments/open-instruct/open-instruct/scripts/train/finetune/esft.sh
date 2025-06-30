#!/bin/bash
# esft.sh
#
# Usage example:
#   bash esft.sh --task codealpaca --model_name_or_path allenai/OLMo-2-1124-7B --num_train_epochs 5 --task_extra_args "--other_option value"
#
# Description:
#   This script loads the task-specific configuration from a file and allows command-line arguments to override defaults.
#   The script assumes that the configuration files are located under the 'scripts/train/finetune/configs/' directory with names like "<task>.conf".
#   For instance, for a task called "codealpaca", the configuration file should be "scripts/train/finetune/configs/codealpaca.conf".
#

# Global defaults (overridden by config or CLI if provided)
USER=${USER:-"user"}

NUM_GPUS=${NUM_GPUS:-2}
CUDA_DEVICES=${CUDA_DEVICES:-"0,1"}
MIXED_PRECISION=${MIXED_PRECISION:-"bf16"}
DEEPSPEED_CONFIG_FILE=${DEEPSPEED_CONFIG_FILE:-"configs/ds_configs/stage3_no_offloading_accelerate.conf"}
DEEPSPEED_PORT=${DEEPSPEED_PORT:-29500}

# training and eval config
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-256}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-4}

MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}
LEARNING_RATE=${LEARNING_RATE:-2e-05}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-"linear"}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-4}

OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-""}


# # Common options for lora_attn mode
COMMON_OPTS=""


# Parse command-line arguments.
# Instead of a config_file parameter, we use --task to select the dataset config.
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --task)
            TASK="$2"
            shift 2
            ;;
        --model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --devices)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --port)
            DEEPSPEED_PORT="$2"
            shift 2
            ;;
        --output_suffix)
            OUTPUT_SUFFIX="$2"
            shift 2
            ;;
        --user)
            USER="$2"
            shift 2
            ;;
        --per_device_train_batch_size)
            PER_DEVICE_TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --per_device_eval_batch_size)
            PER_DEVICE_EVAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --total_batch_size)
            TOTAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --max_seq_length)
            MAX_SEQ_LENGTH="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --lr_type)
            LR_SCHEDULER_TYPE="$2"
            shift 2
            ;;
        --warmup_ratio)
            WARMUP_RATIO="$2"
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --num_train_epochs)
            NUM_TRAIN_EPOCHS="$2"
            shift 2
            ;;
        --cache_dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --expert_type)
            EXPERT_TYPE="$2"
            EXPERT_CONFIG="scripts/train/${MODEL_TYPE}_expert_cfgs/${TASK}/expert_configs_$2.json"
            shift 2
            ;;
        *)
            # Append any unknown arguments to COMMON_OPTS.
            COMMON_OPTS="$COMMON_OPTS $1"
            shift
            ;;
    esac
done

# Check that the --task parameter is provided.
if [[ -z "$TASK" ]]; then
    echo "Error: --task parameter is required."
    usage
fi

# Construct the config file path.
CONFIG_FILE="scripts/train/configs/${TASK}.conf"

# Load the task-specific configuration file if it exists.
if [ -f "$CONFIG_FILE" ]; then
    echo "Loading configuration file: $CONFIG_FILE"
    source "$CONFIG_FILE"
else
    echo "Error: Configuration file $CONFIG_FILE does not exist."
    exit 1
fi

if [[ -z "$MODEL_NAME_OR_PATH" ]]; then
    case $MODEL_TYPE in
        olmoe)
            MODEL_NAME_OR_PATH="allenai/OLMoE-1B-7B-0125"
            TOKENIZER_NAME="allenai/OLMoE-1B-7B-0125"
            ;;
        dense-olmo)
            MODEL_NAME_OR_PATH="allenai/OLMo-2-1124-7B"
            TOKENIZER_NAME="allenai/OLMo-2-1124-7B"
            ;;
        *)
            echo "Warning: Unknown model type provided. Please specify --model_name_or_path manually."
            exit 1
            ;;
    esac
fi

GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / $NUM_GPUS / $PER_DEVICE_TRAIN_BATCH_SIZE))

EXP_NAME="$USER-$MODEL_TYPE-esft-$EXPERT_TYPE"
OUTPUT_DIR="${OUTPUT_BASE_DIR}$MODEL_TYPE-esft-$EXPERT_TYPE"

# For the test file: if TEST_FILE is set to empty or "None", do not pass the parameter.
if [ -n "$TEST_FILE" ] && [ "$TEST_FILE" != "None" ]; then
    TEST_FILE_OPT="--test_file ${TEST_FILE}"
else
    TEST_FILE_OPT=""
fi

if [ -n "$CACHE_DIR" ] && [ "$CACHE_DIR" != "None" ]; then
    CACHE_DIR_OPT="--cache_dir ${CACHE_DIR}"
else
    CACHE_DIR_OPT=""
fi

# Verify that the required variables are set either via the configuration or command line.
REQUIRED_VARS=("MODEL_TYPE" "TASK")
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: Variable $var is not set. Please set it in the config file or pass it as an argument."
        exit 1
    fi
done

OUTPUT_DIR="${OUTPUT_DIR}${OUTPUT_SUFFIX}"

# Export the WandB API key (if needed)
export WANDB_API_KEY=${WANDB_API_KEY:-"wandb_api_key"}

# Launch training with accelerate
TRAIN_CMD="CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} accelerate launch \
    --mixed_precision ${MIXED_PRECISION} \
    --num_processes ${NUM_GPUS} \
    --use_deepspeed \
    --main_process_port ${DEEPSPEED_PORT} \
    --deepspeed_config_file ${DEEPSPEED_CONFIG_FILE} \
    --deepspeed_multinode_launcher standard \
    open_instruct/our_finetune.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --tokenizer_name ${TOKENIZER_NAME} \
    --use_slow_tokenizer False \
    --use_flash_attn \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --warmup_ratio ${WARMUP_RATIO} \
    --weight_decay ${WEIGHT_DECAY} \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --output_dir $OUTPUT_DIR \
    --use_esft True \
    --esft_expert_config_path $EXPERT_CONFIG \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --reduce_loss sum \
    --model_revision main \
    --wandb_project_name $WANDB_PROJECT_NAME \
    --dataset_name $DATASET_NAME  \
    --do_eval \
    --train_file ${TRAIN_FILE} \
    ${TEST_FILE_OPT} \
    --evaluation_strategy "epoch" \
    --metric_for_best_model "eval_loss" \
    --greater_is_better false \
    --save_total_limit 1 \
    --exp_name $EXP_NAME \
    --checkpointing_steps epoch \
    --gradient_checkpointing \
    --add_bos \
    ${CACHE_DIR_OPT} \
    ${COMMON_OPTS}"

# Echo the command for debugging
echo "================== Running Command =================="
echo "$TRAIN_CMD"
echo "====================================================="

# Execute the command
eval $TRAIN_CMD