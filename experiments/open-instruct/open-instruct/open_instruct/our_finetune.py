# !/usr/bin/env python
# coding=utf-8
# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import List, Optional, Union
from functools import partial
import random

import datasets
import deepspeed
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed
from huggingface_hub import HfApi
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    get_scheduler,
)

from open_instruct.dataset_transformation import (
    CHAT_TEMPLATES,
    TokenizerConfig,
    get_cached_dataset_tulu_sft,
)
from open_instruct.model_utils import push_folder_to_hub, save_with_accelerate
from open_instruct.utils import (
    ArgumentParserPlus,
    clean_last_n_checkpoints,
    get_last_checkpoint_path,
    get_wandb_tags,
    is_beaker_job,
    maybe_get_beaker_config,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
    upload_metadata_to_hf,
)

from accelerate import FullyShardedDataParallelPlugin


from esft import to_esft
from our_dataset_processor import get_encode_function, get_train_val_dataset

logger = get_logger(__name__)


def save_gate_parameters_only(accelerator, model, output_dir):
    """
    Save only router (gate) parameters from the model into a .pth file.
    """
    os.makedirs(output_dir, exist_ok=True)

    engine = model                             
    base = accelerator.unwrap_model(model)

    if isinstance(base, PeftModel):
        merged_base = base.merge_and_unload()

        for m_layer, t_layer in zip(
            merged_base.model.layers,
            base.model.model.layers,
        ):
            if hasattr(m_layer.mlp, "gate") and hasattr(t_layer.mlp, "gate"):
                m_layer.mlp.gate.load_state_dict(t_layer.mlp.gate.state_dict())
    else:
        merged_base = base
  
    if isinstance(engine.module, PeftModel):
        target = engine.module.get_base_model()
    else:
        target = engine.module

    target.load_state_dict(merged_base.state_dict(), strict=True)

    full_state = accelerator.get_state_dict(engine)

    if accelerator.is_main_process:
        processed_state = {}
        for k, v in full_state.items():
            if k.startswith("base_model.model.model."):
                k = "model." + k[len("base_model.model.model."):]
            elif k.startswith("base_model.model."):
                k = "model." + k[len("base_model.model."):]
            if k.startswith("model.lm_head."):
                k = "lm_head." + k[len("model.lm_head."):]
            processed_state[k] = v

        cfg = merged_base.config
        fresh = merged_base.__class__(cfg)

        missing, unexpected = fresh.load_state_dict(processed_state, strict=False)
        assert not missing and not unexpected, f"Still mis‑matched keys: {missing or unexpected}"

        gate_params = {}
        for name, param in processed_state.items():
            if "mlp.gate" not in name:
                continue
            if param.numel() == 0:
            
                print(f"[WARN] Skipping empty param: {name}, shape={param.shape}")
                continue
            gate_params[name] = param.cpu()

        torch.save(gate_params, f"{output_dir}/gate.pth")
        print(f"[INFO] Saved {len(gate_params)} gate weights to {output_dir}/gate.pth")

    accelerator.wait_for_everyone()


@dataclass
class FlatArguments:
    """
    Full arguments class for all fine-tuning jobs.
    """

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    run_name: Optional[str] = None
    """A unique name of this run"""
    cache_dir: Optional[str] = field(
        default="cache/",
        metadata={"help": "Cache dir."},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    tokenizer_revision: Optional[str] = field(
        default=None,
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    chat_template_name: str = field(
        default="tulu",
        metadata={
            "help": (
                f"The name of the chat template to use. "
                f"You can choose one of our pre-defined templates: {', '.join(CHAT_TEMPLATES.keys())}."
                f"Or, you can provide a tokenizer name or path here and we will apply its chat template."
            )
        },
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether to use flash attention in the model training"},
    )
    use_slow_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the slow tokenizer or not (which is then fast tokenizer)."},
    )
    model_revision: Optional[str] = field(
        default=None,
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. "
                "This option should only be set to `True` for repositories you trust and in which you "
                "have read the code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, "
                "then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_mixer: Optional[dict] = field(
        default=None,
        metadata={"help": "A dictionary of datasets (local or HF) to sample from."},
    )
    dataset_mixer_list: Optional[list[str]] = field(
        default=None,
        metadata={"help": "A list of datasets (local or HF) to sample from."},
    )
    dataset_mix_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The directory to save the mixed dataset to disk."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a json/jsonl file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input testing/validation data file (a json/jsonl file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated,"
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    add_bos: bool = field(
        default=False,
        metadata={
            "help": "Forcibly add bos token to the beginning of the input sequence."
            " Use only when tokenizer does not add bos token by default."
        },
    )
    clip_grad_norm: float = field(
        default=-1,
        metadata={"help": "Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead)."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW optimizer."},
    )
    logging_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Log the training loss and learning rate every logging_steps steps."},
    )
    eval_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Log the eval loss and learning rate every logging_steps steps."},
    )
    lora_rank: int = field(
        default=64,
        metadata={"help": "The rank of lora."},
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": "The alpha parameter of lora."},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate of lora modules."},
    )
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            "help": "The scheduler type to use for learning rate adjustment.",
            "choices": [
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ],
        },
    )
    num_train_epochs: int = field(
        default=2,
        metadata={"help": "Total number of training epochs to perform."},
    )
    output_dir: str = field(
        default="output/",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."},
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."},
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "If True, will use LORA (low-rank parameter-efficient training) to train the model."},
    )
    lora_router: str = field(
        default="frozen",
        metadata={"help": "select the status of router when using lora: frozen, unfrozen, ste"},
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "Use qLoRA training - initializes model in quantized form. Not compatible with deepspeed."},
    )
    freeze_gate: bool = field(
        default=False,
        metadata={"help": "whether to freeze the gate parameters when training."},
    )
    use_esft: bool = field(
        default=False,
        metadata={"help": "If True, will use ESFT to train the model."},
    )
    esft_expert_config_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "ESFT Expert config path."
            )
        },
    )
    use_8bit_optimizer: bool = field(
        default=False,
        metadata={"help": "Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed."},
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for AdamW if we apply some."},
    )
    timeout: int = field(
        default=1800,
        metadata={
            "help": "Timeout for the training process in seconds."
            "Useful if tokenization process is long. Default is 1800 seconds (30 minutes)."
        },
    )
    reduce_loss: str = field(
        default="mean",
        metadata={
            "help": "How to reduce loss over tokens. Options are 'mean' or 'sum'."
            "Using 'sum' can improve chat model performance."
        },
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "Entity to use for logging to wandb."},
    )
    wandb_project_name: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb project name."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "If the training should continue from a checkpoint folder."},
    )
    with_tracking: bool = field(
        default=False,
        metadata={"help": "Whether to enable experiment trackers for logging."},
    )
    report_to: Union[str, List[str]] = field(
        default="all",
        metadata={
            "help": "The integration(s) to report results and logs to. "
            "Can be a single string or a list of strings. "
            "Options are 'tensorboard', 'wandb', 'comet_ml', 'clearml', or 'all'. "
            "Specify multiple by listing them: e.g., ['tensorboard', 'wandb']"
        },
    )
    save_to_hub: Optional[str] = field(
        default=None,
        metadata={"help": "Save the model to the Hub under this name. E.g allenai/your-model"},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Turn on gradient checkpointing. Saves memory but slows training."},
    )
    use_liger_kernel: bool = field(
        default=False,
        metadata={"help": "Whether to use LigerKernel for training."},
    )
    max_train_steps: Optional[int] = field(
        default=None,
        metadata={"help": "If set, overrides the number of training steps. Otherwise, num_train_epochs is used."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for initialization and dataset shuffling."},
    )
    checkpointing_steps: Optional[str] = field(
        default=None,
        metadata={
            "help": "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."  # noqa
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the content of the output directory. Means that resumption will always start from scratch."
        },
    )
    keep_last_n_checkpoints: int = field(
        default=3,
        metadata={"help": "How many checkpoints to keep in the output directory. -1 for all."},
    )
    fused_optimizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use fused AdamW or not.",
        },
    )
    load_balancing_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to include a load balancing loss (for OLMoE) or not.",
        },
    )
    load_balancing_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for load balancing loss if applicable."},
    )
    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation on the validation set."},
    )
    evaluation_strategy: str = field(
        default="no",
        metadata={
            "help": "The evaluation strategy to use.",
            "choices": ["no", "steps", "epoch"],
        },
    )
    load_best_model_at_end: bool = field(
        default=False,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."},
    )
    metric_for_best_model: str = field(
        default="loss",
        metadata={"help": "The metric to use to compare two different models."},
    )
    greater_is_better: bool = field(
        default=False,
        metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."},
    )
    save_total_limit: int = field(
        default=None,
        metadata={"help": "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints."},
    )
    cache_dataset_only: bool = False
    """Immediately exit after caching the dataset"""
    try_auto_save_to_beaker: bool = True
    """Whether to try to save the model to Beaker dataset `/output` after training"""
    hf_entity: Optional[str] = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""
    try_launch_beaker_eval_jobs: bool = False
    """Whether to launch beaker evaluation jobs after training"""
    hf_metadata_dataset: Optional[str] = "allenai/tulu-3-evals"
    """What dataset to upload the metadata to. If unset, don't upload metadata"""

    def __post_init__(self):
        if self.reduce_loss not in ["mean", "sum"]:
            raise ValueError("reduce_loss must be either 'mean' or 'sum'")
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.dataset_mixer is None
            and self.dataset_mixer_list is None
        ):
            raise ValueError("Need either a dataset name, dataset mixer, or a training file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["json", "jsonl"], "`train_file` should be a json or a jsonl file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["json", "jsonl"], "`test_file` should be a json or a jsonl file."
        # if (
        #     (self.dataset_name is not None and (self.dataset_mixer is not None or self.dataset_mixer_list is not None))
        #     or (self.dataset_name is not None and self.train_file is not None)
        #     or (
        #         (self.dataset_mixer is not None or self.dataset_mixer_list is not None) and self.train_file is not None
        #     )
        #     or (self.dataset_mixer is not None and self.dataset_mixer_list is not None)
        # ):
        #     raise ValueError("Cannot provide two dataset selection mechanisms.")
        if self.try_launch_beaker_eval_jobs:
            raise ValueError("Cannot launch Beaker evaluation jobs without pushing to the Hub.")


def main(args: FlatArguments):
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment

    # args.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    args.run_name = f"{args.exp_name}__LR{args.learning_rate}_epochs{args.num_train_epochs}_seed{args.seed}__{int(time.time())}"


    if is_beaker_job():
        beaker_config = maybe_get_beaker_config()

    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))
    dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=dataloader_config,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    raw_train_dataset, raw_val_dataset = get_train_val_dataset(
        do_eval=args.do_eval,
        train_file=args.train_file,
        test_file=args.test_file,
        dataset_name=args.dataset_name
    )

    tokenizer_revision = args.model_revision if args.tokenizer_revision is None else args.tokenizer_revision
    tokenizer_name = args.tokenizer_name if args.tokenizer_name is not None else args.model_name_or_path
    if tokenizer_revision != args.model_revision:
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tokenizer_revision}` is different
                   from the model revision `{args.model_revision}`."""
        logger.warning(warning)
    tc = TokenizerConfig(
        model_name_or_path=tokenizer_name,
        revision=tokenizer_revision,
        use_fast=not args.use_slow_tokenizer,
        chat_template_name=args.chat_template_name,
        add_bos=args.add_bos,
    )
    tokenizer = tc.tokenizer


    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            revision=args.model_revision,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.cache_dir,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            revision=args.model_revision,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.cache_dir,
        )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    if args.model_name_or_path:
        if args.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            device_index = accelerator.local_process_index
            device_map = {"": device_index}  # force data-parallel training.
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                revision=args.model_revision,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                trust_remote_code=args.trust_remote_code,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
                cache_dir=args.cache_dir,
            )

            if args.use_esft:
                adapter_config = json.load(open(f"{args.esft_expert_config_path}"))
                model = to_esft(model, adapter_config)

        elif args.use_liger_kernel:
            from liger_kernel.transformers import AutoLigerKernelForCausalLM

            fused_linear_cross_entropy = args.reduce_loss == "mean"
            logger.info(f"Attempting to apply liger-kernel. {fused_linear_cross_entropy=}")

            # Supported models: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py#L948
            model = AutoLigerKernelForCausalLM.from_pretrained(
                args.model_name_or_path,
                revision=args.model_revision,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                trust_remote_code=args.trust_remote_code,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                use_flash_attention_2=True if args.use_flash_attn else False,
                # liger-kernel specific args
                fused_linear_cross_entropy=fused_linear_cross_entropy,
                cache_dir=args.cache_dir,
            )

            if args.use_esft:
                adapter_config = json.load(open(f"{args.esft_expert_config_path}"))
                model = to_esft(model, adapter_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                revision=args.model_revision,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                trust_remote_code=args.trust_remote_code,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
                cache_dir=args.cache_dir,
            )

            if args.use_esft:
                adapter_config = json.load(open(f"{args.esft_expert_config_path}"))
                to_esft(model, adapter_config)
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)
    

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
    # resize does its own gather
    if len(tokenizer) > embedding_size:
        # pad to multiple for tensor cores.
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    # update embedding size after resizing for sum loss
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]

    if args.use_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

        logger.info("Initializing LORA model...")
   
        target_modules = [
            "q_proj",
            "o_proj",
            "v_proj",
            "k_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        print("Apply LoRA to all modules")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)

        if args.lora_router == "unfrozen":
            for idx, layer in enumerate(model.model.model.layers):
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                    for param in layer.mlp.gate.parameters():
                        param.requires_grad = True
                    print(f"Unfroze router (gate) in layer {idx}")

        trainable_lora = [
            (name, param.shape) for name, param in model.named_parameters()
            if "lora_" in name and param.requires_grad
        ]

        print(f"Trainable LoRA params: {len(trainable_lora)}")

        for name, shape in trainable_lora[:5]:
            print(f"{name}: shape = {shape}")

        model.print_trainable_parameters()

    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    if args.freeze_gate:
        logger.info("start to freeze gate weights of MoE...")
        gate_params_count = 0
        
        for name, param in model.named_parameters():
            if "mlp.gate" in name:
                param.requires_grad = False
                gate_params_count += param.numel()
                logger.info(f"freeze gate weights: {name}")
        
        frozen = [
            (name, param.requires_grad)
            for name, param in model.named_parameters()
            if "mlp.gate" in name
        ]
        logger.info("gate weight status：")
        for name, req in frozen:
            logger.info(f"  {name}: requires_grad={req}")
        
        logger.info(f"have frozen all gate weights, {gate_params_count:,d} in total")

        ### ----------------Preprocessing the datasets ------------------

    train_encode_fn, val_encode_fn = get_encode_function(args.dataset_name, tokenizer, args.max_seq_length)
    
    with accelerator.main_process_first():
        train_dataset = raw_train_dataset.map(
            train_encode_fn,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[
                name for name in raw_train_dataset.column_names
                if name not in ["input_ids", "labels", "attention_mask"]
            ],
            desc="Tokenizing and reformatting training data",
        )
        train_dataset.set_format(type="pt")
        train_dataset = train_dataset.filter(lambda example: (example['labels'] != -100).any())

    # Log a few training samples
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    

    if args.do_eval:
        with accelerator.main_process_first():
            val_dataset = raw_val_dataset.map(
                val_encode_fn,
                batched=False,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                remove_columns=[
                    name for name in raw_val_dataset.column_names
                    if name not in ["input_ids", "labels", "attention_mask"]
                ],
                desc="Tokenizing and reformatting validation data",
            )
            val_dataset.set_format(type="pt")
            val_dataset = val_dataset.filter(lambda example: (example['labels'] != -100).any())

        # Log a few validation samples
        for index in random.sample(range(len(val_dataset)), min(3, len(val_dataset))):
            logger.info(f"Sample {index} of the validation set: {val_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=args.per_device_train_batch_size,
    )

    eval_dataloader = None
    if args.do_eval:
        eval_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
            batch_size=args.per_device_eval_batch_size,
        )


    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if args.use_qlora:
        from bitsandbytes.optim import AdamW

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8 if args.use_8bit_optimizer else 32,
            is_paged=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            fused=args.fused_optimizer,
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler
    # for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set.
    # In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set.
    # So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the
    # entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of
    # updates matches the num_training_steps.
    num_training_steps_for_scheduler = (
        args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )
    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.do_eval:
        eval_dataloader = accelerator.prepare(
            eval_dataloader
        )
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and str(checkpointing_steps).lower() != "epoch":
        checkpointing_steps = int(checkpointing_steps)


    # 定义评估函数
    def evaluate():
        model.eval()
        losses = []
        
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            
            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
        
        losses = torch.cat(losses)
        losses = losses[:len(val_dataset)]
        
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
            
        logger.info(f"Evaluation perplexity: {perplexity} | Evaluation Loss: {eval_loss}")
        
        return eval_loss, perplexity

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]

        # (Optional) Ai2 internal tracking
        if args.wandb_entity is None:
            args.wandb_entity = maybe_use_ai2_wandb_entity()
        if is_beaker_job():
            experiment_config.update(vars(beaker_config))
        accelerator.init_trackers(
            f"{args.wandb_project_name}",
            experiment_config,
            init_kwargs={
                "wandb": {
                    "name": args.run_name,
                    "entity": args.wandb_entity,
                    "tags": [args.exp_name] + get_wandb_tags(),
                }
            },
        )
        wandb_tracker = accelerator.get_tracker("wandb")

    # Train! 
    if args.lora_router == "unfrozen":
        print("Check whether router is trainable")
        for name, param in model.named_parameters():
            if "gate" in name and "gate_proj" not in name and param.requires_grad:
                print(name, "is trainable")
    

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    best_metric = float("inf") if not args.greater_is_better else float("-inf")
    best_model_checkpoint = None

    # Potentially load in the weights and states from a previous save
    last_checkpoint_path = get_last_checkpoint_path(args)
    if last_checkpoint_path:
        accelerator.print(f"Resumed from checkpoint: {last_checkpoint_path}")
        accelerator.load_state(last_checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        last_checkpoint_path = os.path.basename(last_checkpoint_path)
        training_difference = os.path.splitext(last_checkpoint_path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    print(f"Starting from epoch {starting_epoch} and step {completed_steps}.")
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    local_total_tokens = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    total_token_including_padding = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    start_time = time.time()

    if not args.use_esft and args.do_eval and eval_dataloader is not None:
        logger.info("Start initial evaluation...")
        model.eval()
        initial_eval_loss = 0
        initial_eval_steps = 0
        
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            
            loss = outputs.loss
            batch_size = batch["input_ids"].shape[0]
            initial_eval_loss += accelerator.gather_for_metrics(loss.repeat(batch_size))
            initial_eval_steps += 1
            
            if step >= 5:
                break
        
        try:
            avg_eval_loss = initial_eval_loss.mean().item() / initial_eval_steps
            initial_perplexity = math.exp(avg_eval_loss)
        except OverflowError:
            initial_perplexity = float("inf")
            
        logger.info(f"Initial evaluation result - Loss: {avg_eval_loss:.4f}, Perplexity: {initial_perplexity:.4f}")

        if args.output_dir is not None:
            initial_output_dir = os.path.join(args.output_dir, "initial_checkpoint")
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.makedirs(initial_output_dir, exist_ok=True)
            accelerator.wait_for_everyone()

            save_with_accelerate(
                accelerator=accelerator,
                model=model,
                tokenizer=tokenizer,
                output_dir=initial_output_dir,
                use_lora=args.use_lora
            )
            
            logger.info(f"Initial ckpt saved at {initial_output_dir}")
    

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        train_dataloader.set_epoch(epoch)
        total_loss = 0
        total_aux_loss = 0
        if last_checkpoint_path and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            local_total_tokens += batch["attention_mask"].sum()
            total_token_including_padding += batch["attention_mask"].numel()
            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
                # if args.load_balancing_loss:
                #     outputs = model(**batch, use_cache=False, output_router_logits=True)
                # else:
                #     # TODO: we have calculated the mean loss here anyway, so doubling the calculation
                #     outputs = model(**batch, use_cache=False)
                # if args.reduce_loss == "mean":
                #     loss = outputs.loss
                # else:
                #     # reduce loss is sum
                #     # this ensures that we weight all tokens in the dataset equally,
                #     # rather than weighting each overall example equally when
                #     # using high amounts of gradient accumulation.
                #     # this can result in > 5 point improvements in AlpacaEval
                #     # see https://github.com/huggingface/transformers/issues/24725 for
                #     # more discussion and details.
                #     logits = outputs.logits
                #     labels = batch["labels"]
                #     # Shift so that tokens < n predict n
                #     shift_logits = logits[..., :-1, :].contiguous()
                #     shift_labels = labels[..., 1:].contiguous()
                #     # Flatten the tokens
                #     loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
                #     shift_logits = shift_logits.view(-1, embedding_size)
                #     shift_labels = shift_labels.view(-1)
                #     # Enable model parallelism
                #     shift_labels = shift_labels.to(shift_logits.device)
                #     loss = loss_fct(shift_logits, shift_labels)
                #     if args.load_balancing_loss:
                #         aux_loss = args.load_balancing_weight * outputs.aux_loss
                #         loss += aux_loss
                # # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                if args.load_balancing_loss:
                    total_aux_loss += aux_loss.detach().float()
                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = (
                        accelerator.gather(total_loss).mean().item()
                        / args.gradient_accumulation_steps
                        / args.logging_steps
                    )
                    total_tokens = accelerator.gather(local_total_tokens).sum().item()
                    total_tokens_including_padding = accelerator.gather(total_token_including_padding).sum().item()
                    metrics_to_log = {
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "train_loss": avg_loss,
                        "total_tokens": total_tokens,
                        "per_device_tps": total_tokens / accelerator.num_processes / (time.time() - start_time),
                        "total_tokens_including_padding": total_tokens_including_padding,
                        "per_device_tps_including_padding": total_tokens_including_padding
                        / accelerator.num_processes
                        / (time.time() - start_time),
                    }
                    if args.load_balancing_loss:
                        avg_aux_loss = (
                            accelerator.gather(total_aux_loss).mean().item()
                            / args.gradient_accumulation_steps
                            / args.logging_steps
                        )
                        logger.info(
                            f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, Aux Loss: {avg_aux_loss}, TPS: {total_tokens / (time.time() - start_time)}"
                        )
                        metrics_to_log["aux_loss"] = avg_aux_loss
                    else:
                        logger.info(
                            f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, TPS: {total_tokens / (time.time() - start_time)}"
                        )
            
                    if args.with_tracking:
                        metrics_to_log["epoch"] = completed_steps/len(active_dataloader)

                        accelerator.log(
                            metrics_to_log,
                            step=completed_steps
                        )
                    total_loss = 0
                    total_aux_loss = 0

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
            
                        accelerator.wait_for_everyone()
             
                        save_with_accelerate(
                            accelerator=accelerator,
                            model=model,
                            tokenizer=tokenizer,
                            output_dir=output_dir,
                            use_lora=args.use_lora
                        )

                if completed_steps >= args.max_train_steps:
                    break

        if args.do_eval and eval_dataloader is not None and args.evaluation_strategy == "epoch":
            eval_loss, eval_perplexity = evaluate()
            
            if args.with_tracking:
                accelerator.log(
                    {
                        "eval_loss": eval_loss,
                        "eval_perplexity": eval_perplexity,
                        "epoch": epoch + 1,
                    },
                    step=completed_steps,
                )

        if checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)

            accelerator.wait_for_everyone()

            save_with_accelerate(
                accelerator=accelerator,
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                use_lora=args.use_lora
            )

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        
        save_with_accelerate(
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            use_lora=args.use_lora
        )

        if args.use_lora and args.lora_router == "unfrozen":
            save_gate_parameters_only(accelerator, model, args.output_dir)
    
    # remove all checkpoints to save space
    if accelerator.is_local_main_process:
        clean_last_n_checkpoints(args.output_dir, keep_last_n_checkpoints=0)

    if (
        args.try_auto_save_to_beaker
        and accelerator.is_main_process
        and is_beaker_job()
        and len(beaker_config.beaker_dataset_id_urls) > 0
        and args.output_dir.rstrip("/") != "/output"
    ):
        shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)

    if is_beaker_job() and accelerator.is_main_process:
        # dpo script only supports these two options right now for datasets
        if args.dataset_mixer:
            dataset_list = list(args.dataset_mixer.keys())
        elif args.dataset_mixer_list:
            dataset_list = args.dataset_mixer_list[::2]  # even indices
        elif args.dataset_name:
            dataset_list = [args.dataset_name]
        else:
            dataset_list = [args.train_file]
        # mainly just focussing here on what would be useful for the leaderboard.
        # wandb will have even more useful information.
        metadata_blob = {
            "model_name": args.exp_name,
            "model_type": "sft",
            "datasets": dataset_list,
            "base_model": args.model_name_or_path,
            "wandb_path": wandb_tracker.run.get_url(),
            "beaker_experiment": beaker_config.beaker_experiment_url,
            "beaker_datasets": beaker_config.beaker_dataset_id_urls,
        }
        # save metadata to the output directory. then it should also get pushed to HF.
        with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata_blob, f)

        # upload metadata to the dataset if set
        if args.hf_metadata_dataset:
            upload_metadata_to_hf(
                metadata_blob,
                "metadata.json",
                args.hf_metadata_dataset,
                "results/" + args.run_name,  # to match what the auto-evals name as.
            )

        if args.try_launch_beaker_eval_jobs:
            command = f"""\
            python mason.py  \
                --cluster ai2/ganymede-cirrascale ai2/ceres-cirrascale ai2/neptune-cirrascale ai2/saturn-cirrascale ai2/jupiter-cirrascale-2 \
                --priority low \
                --preemptible \
                --budget ai2/allennlp \
                --workspace ai2/tulu-2-improvements \
                --image nathanl/open_instruct_auto \
                --pure_docker_mode \
                --gpus 0 -- python scripts/wait_beaker_dataset_model_upload_then_evaluate_model.py \
                --beaker_workload_id {beaker_config.beaker_workload_id} \
                --upload_to_hf {args.hf_metadata_dataset} \
                --model_name {args.run_name} \
                --run_id {wandb_tracker.run.get_url()}
            """
            process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            print(f"Submit jobs after model training is finished - Stdout:\n{stdout.decode()}")
            print(f"Submit jobs after model training is finished - Stderr:\n{stderr.decode()}")
            print(f"Submit jobs after model training is finished - process return code: {process.returncode}")

    accelerator.wait_for_everyone()
    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    parser = ArgumentParserPlus((FlatArguments))
    args = parser.parse()
    main(args)