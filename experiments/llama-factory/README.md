# Post-Training for Qwen1.5-MoE & Qwen3-MoE

This folder contains the code and instructions for reproducing the experiments on [Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B) and [Qwen3-30B-A3B-Base](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base) as described in the [DenseMixer blog post](https://fengyao.notion.site/moe-posttraining). All experiments are conducted using [llama-factory](https://github.com/hiyouga/LLaMA-Factory).

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [References](#references)

---

## Environment Setup

Refer to `installation.sh` for detailed setup instructions.

---

## Data Preparation

We use a diverse set of datasets for training and evaluation. Our pre-processed datasets are available on Hugging Face.

For `Qwen1.5-MoE-A2.7B`, we use [GSM](https://huggingface.co/datasets/RoxanneWsyw/gsm) (math reasoning), [CodeAlpaca](https://huggingface.co/datasets/RoxanneWsyw/CodeAlpaca) (code generation), [ESFT-intent](https://huggingface.co/datasets/RoxanneWsyw/ESFT-intent) (intent understanding), [ESFT-law](https://huggingface.co/datasets/RoxanneWsyw/ESFT-law) (legal reasoning), [ESFT-summary](https://huggingface.co/datasets/RoxanneWsyw/ESFT-summary) (summarization), [ESFT-translation](https://huggingface.co/datasets/RoxanneWsyw/ESFT-translation) (translation). For code generation evaluation, we use [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) and [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval).

For `Qwen3-30B-A3B-Base`, we use [s1](https://huggingface.co/datasets/autoprogrammer/s1K-1.1_lf_filtered) (math reasoning), and [nemotron-code](https://huggingface.co/datasets/autoprogrammer/nemotron_code_lf_filtered) (coding reasoning). For evaluation, we challenging math and coding benchmarks that require reasoning abilities.

---

## Training

We support the following fine-tuning methods. For each method, replace `{dataset_name}` with your target dataset (`gsm`, `codealpaca`,`esft`).

### 1. Frozen Router

**Full Fine-tuning & LoRA**
```bash
cd LLaMA-Factory
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
bash run/qwen1.5/frozen_full/train_{dataset_name}_frozen.sh
bash run/qwen1.5/frozen_lora/train_{dataset_name}.sh
```

### 2. Conventional Training

**Full Fine-tuning & LoRA**
```bash
cd LLaMA-Factory
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
bash run/qwen1.5/conventional_full/train_{dataset_name}.sh
bash run/qwen1.5/conventional_lora/train_{dataset_name}_unfrozen.sh
###qwen3-30b
bash run/qwen3/train_full-code.sh
bash run/qwen3/train_full-math.sh
```

### 3. DenseMixer

**Full Fine-tuning & LoRA**
```bash
cd LLaMA-Factory
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
bash run/qwen1.5/densemixer_full/train_{dataset_name}_densemixer.sh
bash run/qwen1.5/densemixer_lora/train_{dataset_name}_densemixer.sh
###qwen3-30b
bash run/qwen3/train_densemixer-code.sh
bash run/qwen3/train_densemixer-math.sh
```

### 4. ESFT Fine-tuning

Before running ESFT, generate expert configs:
```bash
cd esft_experts_gen/experts
bash get_expert_scores.sh
bash generate_expert_config.sh
```

**ESFT-Gate** (Selects experts by Average Gate Score)
```bash
bash run/qwen1.5/esft-gate/train_gsm.sh
bash run/qwen1.5/esft-gate/train_code.sh
bash run/qwen1.5/esft-gate/train_esft.sh
bash run/qwen1.5/esft-gate/train_amthinking_math.sh
```

**ESFT-Token** (Selects experts by Token Selection Ratio)
```bash
bash run/qwen1.5/esft-token/train_gsm.sh
bash run/qwen1.5/esft-token/train_code.sh
bash run/qwen1.5/esft-token/train_esft.sh
bash run/qwen1.5/esft-token/train_amthinking_math.sh
```

> **Implementation Details:**  
> - LLaMA-Factory framework: [`LLaMA-Factory`](LLaMA-Factory)  
> - Training scripts: [`LLaMA-Factory/run`](LLaMA-Factory/run)  
> - Configuration files: [`LLaMA-Factory/examples`](LLaMA-Factory/examples)

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

- [Qwen1.5-A2.7B on HuggingFace](https://huggingface.co/Qwen/Qwen1.5-A2.7B)
- [Qwen3-30B-A3B on HuggingFace](https://huggingface.co/Qwen/Qwen3-30B-A3B)
- [LLaMA-Factory Framework](https://github.com/hiyouga/LLaMA-Factory)
- [ESFT Paper](https://arxiv.org/abs/2407.01906)
- [s1k Paper](https://huggingface.co/papers/2501.19393)
- [Nemotron Paper](https://huggingface.co/papers/2505.00949)
