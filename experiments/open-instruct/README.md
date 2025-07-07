# Post-Training for OLMoE

This folder contains the code and instructions for reproducing the experiments on [OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0125) as described in the [DenseMixer blog post](https://fengyao.notion.site/moe-posttraining). All experiments are conducted using [open-instruct](https://github.com/allenai/open-instruct), the official codebase originally developed by AllenAI (AI2) for OLMoE post-training.

---

## Table of Contents

- [Post-Training for OLMoE](#post-training-for-olmoe)
  - [Table of Contents](#table-of-contents)
  - [Environment Setup](#environment-setup)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
    - [1. Frozen Router](#1-frozen-router)
    - [2. Conventional Training](#2-conventional-training)
    - [3. DenseMixer](#3-densemixer)
    - [4. ESFT Fine-tuning](#4-esft-fine-tuning)
  - [Evaluation](#evaluation)
  - [References](#references)

---

## Environment Setup

Set up your environment as follows:

```bash
conda create -n openinstruct python=3.12
conda activate openinstruct
conda install -c conda-forge cuda-nvcc=12.1 -y

cd open-instruct
bash init_env.sh
```

---

## Data Preparation

We use the datasets from Deepseek's [ESFT](https://github.com/deepseek-ai/ESFT) paper for both training and evaluation. Our pre-processed datasets are available on Hugging Face: [GSM](https://huggingface.co/datasets/RoxanneWsyw/gsm) (math reasoning), [CodeAlpaca](https://huggingface.co/datasets/RoxanneWsyw/CodeAlpaca) (code generation), [ESFT-intent](https://huggingface.co/datasets/RoxanneWsyw/ESFT-intent) (intent understanding), [ESFT-law](https://huggingface.co/datasets/RoxanneWsyw/ESFT-law) (legal reasoning), [ESFT-summary](https://huggingface.co/datasets/RoxanneWsyw/ESFT-summary) (summarization), and [ESFT-translation](https://huggingface.co/datasets/RoxanneWsyw/ESFT-translation) (translation).

For code generation evaluation, we use [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) and [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval).

---

## Training

We support the following fine-tuning methods. For each method, replace `{dataset_name}` with your target dataset (`gsm`, `codealpaca`,`intent`, `law`, `summary`, `translation`).

### 1. Frozen Router

**Full Fine-tuning & LoRA**
```bash
cd open-instruct
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
bash run/frozen_full/train_{dataset_name}.sh
bash run/frozen_lora/train_{dataset_name}.sh
```

### 2. Conventional Training

**Full Fine-tuning & LoRA**
```bash
cd open-instruct
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
bash run/conventional_full/train_{dataset_name}.sh
bash run/conventional_lora/train_{dataset_name}.sh
```

### 3. DenseMixer
To run DenseMixer, you need one additional setup
```bash
pip install densemixer
densemixer setup
```

**Full Fine-tuning & LoRA**
```bash
cd open-instruct
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
bash run/densemixer_full/train_{dataset_name}.sh
bash run/densemixer_lora/train_{dataset_name}.sh
```

A LoRA merge script is provided [here](open-instruct/scripts/train/finetune/merge_lora.sh).

### 4. ESFT Fine-tuning

To run ESFT, you need to generate expert configs first (We also provide the pre-generated configs for OLMoE [here](open-instruct/scripts/train/olmoe_expert_cfgs)):
```bash
cd esft_experts_gen/experts
bash get_expert_scores.sh
bash generate_expert_config.sh
```

ESFT has two variants: (1) **ESFT-Gate**, which selects experts by average gate score; (2) **ESFT-Token**, which selects experts by token selection ratio. You can run them as follows.

```bash
cd open-instruct
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
bash run/esft_gate/train_{dataset_name}.sh
bash run/esft_token/train_{dataset_name}.sh
```

> **Implementation Details:**  
> - Main fine-tuning code: [`open-instruct/open_instruct/our_finetune.py`](open-instruct/open_instruct/our_finetune.py)  
> - ESFT implementation: [`open-instruct/open_instruct/esft.py`](open-instruct/open_instruct/esft.py)  
> - Training scripts: [`open-instruct/scripts/train/finetune`](open-instruct/scripts/train/finetune)  
> - Dataset configurations: [`open-instruct/scripts/train/configs`](open-instruct/scripts/train/configs)

---

## Evaluation

Evaluation scripts are in the [`eval/`](eval/) directory.
See [`eval/README.md`](eval/README.md) for details and environment setup.

**Key arguments to modify for your trained model:**
- `--save_dir`
- `--model_name_or_path`
- `--tokenizer_name_or_path`

---

## References

- [OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0125)
- [DenseMixer Blog](https://fengyao.notion.site/moe-posttraining)
- [open-instruct](https://github.com/allenai/open-instruct)
- [ESFT Project & Datasets](https://github.com/deepseek-ai/ESFT)