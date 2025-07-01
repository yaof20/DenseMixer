# Experiments

This folder contains two post-training codebases adapted for our experiments, as described in the [DenseMixer blog post](https://fengyao.notion.site/moe-posttraining).

## open-instruct
[open-instruct](open-instruct) provides the code and instructions for reproducing the experiments on [OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0125). It includes training on 6 datasets and evaluation on 7 tasks, which are listed within the [open-instruct](open-instruct) directory.

## llama-factory
[llama-factory](llama-factory) contains the code and instructions for reproducing the experiments on [Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B) and [Qwen3-30B-A3B-Base](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base).

- For `Qwen1.5-MoE-A2.7B`, we train on 6 datasets and evaluate on 7 tasks, as detailed in the [llama-factory](llama-factory) directory.

- For `Qwen3-30B-A3B-Base`, we focus on two datasets: [s1](https://huggingface.co/datasets/autoprogrammer/s1K-1.1_lf_filtered) (math reasoning) and [nemotron-code](https://huggingface.co/datasets/autoprogrammer/nemotron_code_lf_filtered) (coding reasoning). Evaluation is conducted on challenging math and coding benchmarks that require reasoning capabilities.