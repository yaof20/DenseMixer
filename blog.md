# DenseMixer: Improving MoE Post-Training with Precise Router Gradient
**Feng Yao$^{\star\dagger}$      Junxia Cui$^{\star}$      Ruohan Zhang$^{\star}$      Liyuan Liu$^{\dagger}$      Shibo Hao      Li Zhang      Chengyu Dong      Shuohang Wang      Yelong Shen      Jianfeng Gao      Jingbo Shang**

$^{\dagger}$: Project Lead; $^{\star}$: Core Contributors; (Work in Progress)

**UCSD,  Microsoft**

https://github.com/yaof20/DenseMixer

<aside>

### **TL;DR**

We introduce **DenseMixer —** a novel and effective MoE **post-training** technique that makes MoE easier to train and better performing. 

By trading one **extra forward pass** on inactive experts for **precise router gradient**, DenseMixer **consistently** outperforms conventional training — across different MoE scales (7B, 14B, 30B), architectures (with/without shared experts), pre-trained methods (from scratch/up-cycling), and post-training data types (instruction/long CoT data).

We provide a **plug-and-play** implementation for DenseMixer, empowering MoE training simply by `pip install densemixer`. It is fully compatible with existing libraries (e.g., transformers, llama-factory, open-instruct, verl) and can be applied with parameter-efficient methods (e.g., LoRA), introducing **no changes to inference**.

</aside>

![image.png](attachment:ab9dd963-7bb7-48b9-9ed1-ef7e588f9c3c:2e2c5cc0-7bfc-4540-a853-dfbda2d19788.png)

***Figure 1.** Performance gains of Qwen3-30B MoE after post-training using conventional method vs. DenseMixer. Results are reported with the decoding parameters: temperature = 0.6, top-p = 0.95. Additional results under other decoding configs are provided in the [Empirical Result](https://www.notion.so/DenseMixer-Improving-MoE-Post-Training-with-Precise-Router-Gradient-1eb721e3f6c48098811bd4a48971324b?pvs=21)s section.*

# Problem of MoE Training

MoE is notoriously **harder** to train compared with dense models. The only difference MoE introduces is its sparse routing mechanism — typically implemented via a **Top-K router,** which is mathematically **non-differentiable**. Such issue blocks the straight-forward back-propagation and complicates gradient computation. We elaborate on this in the [Technical Details](https://www.notion.so/DenseMixer-Improving-MoE-Post-Training-with-Precise-Router-Gradient-1eb721e3f6c48098811bd4a48971324b?pvs=21) section.

# Introducing DenseMixer

To address the non-differentiability problem, we introduce **DenseMixer** for MoE post-training, where we trade **additional compute** (on inactive experts during the forward pass) for more **precise router gradient** estimation. We delve into the details in the [Technical Details](https://www.notion.so/DenseMixer-Improving-MoE-Post-Training-with-Precise-Router-Gradient-1eb721e3f6c48098811bd4a48971324b?pvs=21) section.

DenseMixer **consistently** outperforms conventional MoE training in downstream performance across different MoE scales (7B, 14B, 30B), architectures (with/without shared experts), pre-trained methods (from scratch/up-cycling), and post-training data types (instruction/long CoT).

It is **universally applicable** to any MoE using Top-K router and back-propagation, and it can be used in a **plug-and-play** manner, compatible with existing training libraries ([transformers](https://github.com/huggingface/transformers), [llama-factory](https://github.com/hiyouga/LLaMA-Factory), [open-instruct](https://github.com/allenai/open-instruct), [verl](https://github.com/volcengine/verl)) and parameter-efficient methods (e.g., [LoRA](http://huggingface.co/docs/diffusers/training/lora)).

To shift from conventional MoE training to DenseMixer, you only need the following change:

```bash
# Your current MoE training
python your_moe_training_script.py

# Shift to DenseMixer (**no code changes needed!**)
pip install densemixer
densemixer setup

export DENSEMIXER_ENABLED=1
python your_moe_training_script.py
```

Please refer to our GitHub repo for more details. → https://github.com/yaof20/DenseMixer

# Empirical Results

We conduct experiments with MoEs of varying scales, architectures, and pre-training recipes, using both (relatively) short instruction and long reasoning datasets for training.

## Models & Datasets

We select the following three **base models** for conducting post-training experiments.

| Model Name | Active Param. | Total Param. | Active/TotalExpert Num. | SharedExpert Num. | Context
Length | Normalize TopK Prob | Training Strategy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [**OLMoE-1B-7B**](https://huggingface.co/allenai/OLMoE-1B-7B-0125) | 1B | 7B | 8 / 64 | 0 | 4k | False | trained from scratch |
| [**Qwen1.5-MoE-A2.7B**](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B) | 2.7B | 14B | 8 / 64 | 4 | 8k | False | trained from up-cycling |
| [**Qwen3-30B-A3B-Base**](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base) | 3B | 30B | 8 / 128 | 0 | 32k | True | trained from scratch |

For `OLMoE-1B-7B` and `Qwen1.5-MoE-A2.7B`, we mostly adopt the training and testing datasets from DeepseekAI’s recent ESFT paper [[Wang et al., 2024](https://arxiv.org/pdf/2407.01906)]

For `Qwen3-30B-A3B-Base`, we found that training it with the data above can hardly improve or even degrade its downstream performance. Therefore, we train it on long reasoning data and test it on challenging math/coding benchmarks that requires reasoning. 

- For the math domain, we post-train it on [a filtered subset](https://huggingface.co/datasets/autoprogrammer/s1K-1.1_lf_filtered) of the Stanford [`S1`](https://huggingface.co/datasets/simplescaling/s1K-1.1) dataset, which has around **1K training samples** with reasoning trajectories distilled from Deepseek-R1.
- For the coding domain, we post-train it on [a filtered subset](https://huggingface.co/datasets/autoprogrammer/nemotron_code_lf_filtered) of [`Llama-Nemotron-Post-Training-Dataset`](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset), which contains around **35K training samples**.

## Baselines

- **Conventional** → **Conventional or Standard MoE training**
    - This is the default and mostly adopted way to train MoE, where the non-differentiable Top-K operation is treated as having zero gradient during automatic differentiation.
- **ESFT** → **Expert-Specialized Fine-Tuning**
    - ESFT is proposed by DeepseekAI, focusing on the expert's specialty during fine-tuning.
- **Frozen Router** → **Freeze the router and only update other parts in MoE.**
    - Frozen Router is suggested by Unsloth in their [tutorial on how to finetune Qwen3-MoE](https://docs.unsloth.ai/basics/qwen3-how-to-run-and-fine-tune#qwen3-moe-models-fine-tuning).

In addition, we compare these methods in the parameter-efficient fine-tuning (PEFT) settings, where we apply LoRA to all modules except the router. As ESFT itself already serves as a PEFT method, we do not add the LoRA counterparts for it. 

# Empirical Results

For each method, we conduct a grid search on the training hyperparameters (learning rate and batch size) and report the best performance. More details are provided on Github repo.

### OLMoE-1B-7B

The results for **OLMoE-1B-7B** models are shown below.

|  | **GSM** | **MBPP** | **HumanEval** | **Intent** | **Law** | **Summary** | **Translation** | avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base Model | 15.85 | 19.80 | 10.97 | 0.20 | 5.70 | 7.40 | 11.09 | 10.14 |
| Frozen Router | 44.88 | 17.8 | 7.23 | 72.80 | 22.50 | 36.05 | 28.29 | 32.79 |
| Conventional | 45.94 | 23.4 | 18.92 | 74.60 | 22.35 | 35.99 | 26.89 | 35.44 |
| **DenseMixer** | **49.00** | **25.12** | **20.73** | **77.40** | **23.02** | **40.64** | **32.55** | **38.35** |
| Frozen Router -lora | 45.03 | 24.2 | 17.07 | 55.80 | 21.30 | 37.70 | 28.19 | 32.76 |
| Conventional -lora | 44.58 | 24.2 | 15.85 | 60.20 | 21.60 | 37.30 | 26.22 | 32.85 |
| **DenseMixer -lora** | **45.38** | **26.2** | **16.48** | **66.60** | **24.70** | **40.80** | **29.43** | **35.66** |
| ESFT - gate | 43.06 | 20.8 | 14.02 | 21.20 | 22.39 | 19.50 | 17.37 | 22.62 |
| ESFT - token | 43.82 | 19.6 | 12.80 | 20.80 | 22.60 | 17.80 | 16.67 | 22.01 |

### Qwen1.5-MoE-A2.7B

The results for **Qwen1.5-A2.7B** models are shown below.

|  | **GSM** | **MBPP** | **HumanEval** | **Intent** | **Law** | **Summary** | **Translation** | avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base Model | 38.69 | 38.84 | 32.31 | 16.83 | 18.20 | 28.29 | 16.53 | 27.10 |
| Frozen Router | 53.37 | 35.20 | 37.10 | 82.20 | 33.01 | 38.29 | 32.75 | 44.56 |
| Conventional | 53.42 | 34.60 | 36.43 | 81.80 | 29.25 | 37.80 | 33.02 | 43.76 |
| **DenseMixer** | **55.16** | **35.40** | **39.68** | **83.40** | **33.83** | **40.56** | **33.90** | **45.99** |
| Frozen Router -lora | 46.77 | 31.40 | 36.58 | 71.00 | 30.30 | 30.19 | 28.08 | 39.19 |
| Conventional -lora | 43.89 | 34.00 | 38.41 | 64.80 | 28.80 | 37.99 | 26.14 | 39.15 |
| **DenseMixer -lora** | **47.24** | **35.40** | **38.41** | **71.80** | **31.80** | **40.20** | **29.25** | **42.01** |
| ESFT - gate | 50.72 | 34.00 | 36.59 | 76.40 | 27.10 | 35.89 | 28.49 | 41.31 |
| ESFT - token | 52.76 | 35.80 | 37.20 | 76.00 | 28.20 | 33.39 | 28.86 | 41.74 |

### Qwen3-30B-A3B-Base

For **Qwen3-30B-A3B** models, we report the evaluation metrics under **three sets** of decoding hyperparameters (temperature and top_p) due to the instability of long CoT evaluation.

- Results of checkpoints post-trained with **Nemotron-Code** dataset (35K training samples)
    
    
    |  | temperature & top_p | **HumanEval
    (avg@4)** | **HumanEval+
    (avg@4)** | **MBPP
    (avg@1)** | **LiveCodeBench
    (avg@4)** | avg |
    | --- | --- | --- | --- | --- | --- | --- |
    | Base Model | 0.6  &  0.95 | 65.24 | 60.06 | 53.60 | 16.85 | 48.94 |
    |  | 0.7  &  0.8 | 62.80 | 61.12 | 55.60 | 16.48 | 49.00 |
    |  | 1.0  &  0.7 | 62.65 | 59.14 | 49.80 | 13.26 | 46.21 |
    | Conventional | 0.6  &  0.95 | 92.23 | 86.89 | 80.80 | 32.26 | 67.21 |
    |  | 0.7  &  0.8 | 91.01 | 85.37 | 76.80 | 29.39 | 65.59 |
    |  | 1.0  &  0.7 | 90.85 | 86.59 | 79.00 | 33.42 | 67.59 |
    | **DenseMixer** | **0.6  &  0.95** | **93.59** | **89.02** | **82.00** | **34.31** | **68.80** |
    |  | **0.7  &  0.8** | **91.92** | **86.89** | **80.80** | **31.89** | **67.32** |
    |  | **1.0  &  0.7** | **93.29** | **88.87** | **84.39** | **34.40** | **68.99** |
- Results of checkpoints post-trained with Stanford **S1** dataset (1K training samples)
    
    
    |  | temperature & top_p | **GPQA Diamond
    (avg@8)** | **AIME 2024
    (avg@32)** | **AIME 2025
    (avg@32)** | **Olympiad Bench
    (avg@1)** | **MATH-500 (avg@1)** | avg |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | Base Model | 0.6  &  0.95 | 38.88 | 20.63 | 7.71 | 34.81 | 72.80 | 34.97 |
    |  | 0.7  &  0.8 | 39.89 | 20.53 | 8.33 | 33.92 | 75.40 | 35.61 |
    |  | 1.0  &  0.7 | 36.36 | 18.75 | 8.75 | 31.70 | 68.00 | 36.20 |
    | Conventional | 0.6  &  0.95 | 54.80 | 61.56 | 45.63 | 57.33 | 93.40 | 62.54 |
    |  | 0.7  &  0.8 | 54.23 | 61.67 | 44.27 | 55.41 | 92.20 | 61.56 |
    |  | 1.0  &  0.7 | 56.55 | 63.65 | 46.15 | 59.11 | 93.00 | 63.69 |
    | **DenseMixer** | **0.6  &  0.95** | **58.52** | **63.85** | **45.83** | **58.51** | **93.60** | **64.06** |
    |  | **0.7  &  0.8** | **55.80** | **63.13** | **45.31** | **57.18** | **93.00** | **62.88** |
    |  | **1.0  &  0.7** | **58.14** | **62.71** | **47.50** | **57.77** | **93.80** | **63.98** |

Extensive experiments show that **DenseMixer consistently outperforms conventional MoE training methods** across models of diverse scales, architectures, and pre-trained recipes, using different training and test datasets. This holds true for both full fine-tuning and parameter-efficient fine-tuning (e.g., LoRA), and under varying generation hyper-parameters.

# Technical Details

For those curious about the math behind the magic and how it's implemented, we provide a **clear** and **step-by-step** walkthrough of the problem and our solution.

## The Root Problem: Non-Differentiable Routing

### MoE Forward Propagation

In Transformer-based MoE models, the standard feed-forward network (FFN) layer is replaced with a Mixture-of-Experts (MoE) layer, consisting of a set of $N$ parallel FFNs referred to as experts: $\{E_0(x), E_1(x), \dots, E_{N-1}(x)\}$, and a lightweight router network that dynamically select a subset of these experts to activate for each input $x$. 

The forward propagation of MoE layer involves the following steps:

1. **Compute routing weights —** Compute a score vector $\pi \in \mathbb{R}^N$ via a linear mapping and ****apply a softmax to obtain a probability distribution:
    
    $$
    \pi = \text{Router}(x; \theta) = \text{softmax}(x\cdot \theta^T)
    $$
    
    where  $\theta$  is the router parameter.
    
2. **Compute expert output —** Compute the weighted sum over the Top-K selected experts’ output with the corresponding routing weights:
    
    $$
    \begin{aligned}
    y &= \sum_{i=0}^{N-1} \pi_i \cdot \mathrm{TopK}(\pi)_i \cdot \mathrm{Expert}_i(x)
    \end{aligned}
    $$
    
    where: 
    
    $$
    \textcolor{black}{\text{TopK}}(\pi)_i =
    \begin{cases}
    1, & \text{if } \pi_i \text{ is among the top-k} \text{ values of } \pi,\\
    0, & \text{otherwise}.
    \end{cases}
    
    $$
    

### MoE Backward Propagation → the non-differentiability problem

To backpropagate from the expert output $y$ to router parameter $\theta$, we compute:

$$
\nabla_\theta y\;= \sum_{j=0}^{N-1}\frac{\partial y}{\partial \pi_j}\frac{\partial \pi_j}{\partial \theta}=\;
\sum_{j=0}^{N-1}\sum_{i=0}^{N-1}\mathrm{Expert}_i(x)\cdot\textcolor{blue}{\frac{\partial (\pi_i\cdot\mathrm{TopK}(\pi)_i )}{\partial\pi_j }}\cdot\frac{\partial \pi_j}{\partial\theta }
$$

Here:

1. The last term is straightforward to compute:
    
    $$
    \frac{\partial \pi_j}{\partial \theta} = \frac{\partial~\text{softmax}(x\cdot \theta^T)_j}{\partial\theta}
    $$
    
2. The middle term expands to: 
    
    $$
    \begin{aligned}
    \textcolor{blue}{\frac{\partial (\pi_i\cdot\mathrm{TopK}(\pi)_i )}{\partial\pi_j }} 
    & = \pi_i\cdot\frac{\partial\mathrm{TopK}(\pi)_i }{\partial\pi_j} + \mathrm{TopK}(\pi)_i \cdot\delta_{ij}
    \end{aligned}
    $$
    
    where:
    
    $$
    \delta_{ij} = \begin{cases}1, & i=j,\\0, & i\neq j.\end{cases}
    $$
    

Clearly, the Top-K mask is non-differentiable — specifically, the term $\frac{\partial\mathrm{TopK}(\pi)_i }{\partial\pi_j}$, is not defined. Therefore, the router’s gradient is not straightforward to compute.

## How Conventional Training Handles This

Conventional MoE training sidesteps this issue by treating $\mathrm{TopK}(\pi)$ as constant during backpropagation, thus neglecting the non-differentiable term $\frac{\partial\mathrm{TopK}(\pi)_i }{\partial\pi_j}$, namely

$$
\frac{\partial\mathrm{TopK}(\pi)_i }{\partial\pi_j} \;\approx\;0
$$

Thus:

$$
\textcolor{blue}{\frac{\partial (\pi_i~\mathrm{TopK}(\pi)_i )}{\partial\pi_j }} \approx \mathrm{TopK}(\pi)_i\cdot\delta_{ij}
$$

And finally,

$$
\begin{aligned}
\nabla_\theta y \approx \nabla_{Conventional}\;
& =
\sum_{j=0}^{N-1}\sum_{i=0}^{N-1}\mathrm{Expert}_i(x)\cdot\mathrm{TopK}(\pi)_i\cdot\delta_{ij}\cdot\frac{\partial \pi_j}{\partial\theta } \\
& = \sum_{j=0}^{N-1}\mathrm{Expert}_j(x)\cdot\textcolor{blue}{\mathrm{TopK}(\pi)_j}\cdot\frac{\partial \pi_j}{\partial\theta } 
\end{aligned}
$$

## DenseMixer: Trade Compute for Gradient Precision

### DenseMixer’s Solution

To make the gradient approximation more accurate, DenseMixer adopts the **straight-through estimator** (STE) [[Bengio et al., 2013](https://arxiv.org/pdf/1308.3432)]. Note that this amounts to a first-order approximation based on [[Liu et al., 2023](https://arxiv.org/pdf/2304.08612)] . Concretely,

$$
\frac{\partial\mathrm{TopK}(\pi)_i }{\partial\pi_j} \;\approx\;\delta_{ij}
$$

This means

$$
\textcolor{blue}{\frac{\partial (\pi_i~\mathrm{TopK}(\pi)_i )}{\partial\pi_j }} 
\approx
(\pi_i + \mathrm{TopK}(\pi)_i)\cdot\delta_{ij}
$$

Finally,

$$
\begin{aligned}
\nabla_\theta y \approx \nabla_{DenseMixer}\;
& =
\sum_{j=0}^{N-1}\sum_{i=0}^{N-1}\mathrm{Expert}_i(x)\cdot\left(\pi_i + \mathrm{TopK}(\pi)_i\right)\cdot\delta_{ij}\cdot\frac{\partial \pi_j}{\partial\theta }\\
& =
\sum_{j=0}^{N-1}\mathrm{Expert}_j(x)\cdot\textcolor{blue}{\left(\pi_j + \mathrm{TopK}(\pi)_j\right)}\cdot\frac{\partial \pi_j}{\partial\theta }
\end{aligned}
$$

Intuitively, STE pretends that the Top-K selection is an identity function with respect to the router logits during backpropagation. In practice, we implement this by

1. **Forward —** computing with the original hard Top-K as before;
2. **Backward —** overriding the gradient of the Top-K node to be the identity.

### The Side Effect of DenseMixer

An obvious drawback of using DenseMixer’s gradient approximation ($\nabla_{DenseMixer}$) is that it requires the outputs of all experts (i.e., $\mathrm{Expert}_j(x)$ for all $j \in [0, \text{N-1}]$) for a given input $x$, while conventional MoE training’s gradient approximation ($\nabla_{Conventional}$) only requires those of the Top-K selected experts. This means that DenseMixer requires the MoE layer to be densely activated during forward pass, which negates MoE’s sparsity advantage and introduces more compute. Therefore, DenseMixer is less practical for MoE pre-training. 

However, for MoE post-training, where the compute cost is not a critical problem, we can employ DenseMixer to trade compute for performance. This is **a little bit surprising** since the base model is pretrained in conventional way. We found that with more precise gradient, DenseMixer consistently outperforms conventional training on downstream tasks.

In practice, DenseMixer only requires one extra **forward pass** on those inactive experts, which is different than setting the Top-K value to the total experts number. We empirically show that this won’t significantly increase training time in [Efficiency Analysis](https://www.notion.so/DenseMixer-Improving-MoE-Post-Training-with-Precise-Router-Gradient-1eb721e3f6c48098811bd4a48971324b?pvs=21) section.  

## Handling More Complicated Cases

Some recently released MoE models adopt `normalized_topk_prob` implementation, which normalize the expert weights as follows.

$$

\begin{aligned}
y_{normalized} &= {\sum_{i=0}^{N-1} \frac{\pi_i \cdot \mathrm{TopK}(\pi)_i}{\textcolor{red}{\sum_{k=0}^{N-1}\pi_k \cdot\mathrm{TopK(\pi)_k} }} \cdot \mathrm{Expert}_i(x)}
\end{aligned}

$$

Such normalization makes the gradient computation even more complicated as both the nominator and denominator have the non-differentiable TopK term. 

We have tried several implementations and temporally reached a decent but not perfect solution — distinguishes the TopK and non-TopK experts’ output during backpropagation. For this part, please check our [code implementations](https://github.com/yaof20/DenseMixer/blob/702b64d2ce6aba37241fa7bae111c6cf8f122d5e/densemixer/models/qwen3_moe_custom.py#L33) for more details. 

# Efficiency Analysis

### Memory

In standard MoE training, the model weights are loaded on the GPU all the time. Therefore, in this case, DenseMixer barely incurs additional memory usage. This parity was empirically confirmed through rigorous analysis of training logs across all experiments.

### Time

Though DenseMixer requires extra compute on inactive experts, the time consumption does not grow linearly with the number of active experts. It only requires an extra **forward pass** and **does not require** *gradient computation* and *backward parameter update* on these inactive experts. The time overhead is negligible when post-training on a small amount of data.

We provide the time consumption comparison between conventional MoE training and DenseMixer under the full fine-tuning setting with ZeRO3 parallelism.

|  | Training Data - | Intent (7K)  | Law (1K) | Summary (19K) | Translation (11K) |
| --- | --- | --- | --- | --- | --- |
| Qwen1.5-MoE (14B - A2.7B) | Conventional | 22 min | 8.5 min | 1.2 h | 39 min |
| Qwen1.5-MoE (14B - A2.7B) | DenseMixer | 24 min | 9.5 min | 1.4 h | 45 min |

|  | Training Data | S1 (1K) | Nemotron-Code (35K) |
| --- | --- | --- | --- |
| Qwen3-MoE
(30B - A3B) | Conventional | 2.8 h | 21 h  |
| Qwen3-MoE
(30B - A3B) | DenseMixer | 3.6 h | 28 h |

# Conclusion and Future Work

We confirm that the non-differentiability of Top-K routing is a key obstacle to achieving more effective MoE training. By trading an extra forward pass on inactive experts, **DenseMixer** enables more precise routing gradients, consistently improving MoE post-training quality, as measured by downstream performance,  beyond conventional MoE training approaches.

In the future, we plan to extend our supervised fine-tuning (SFT) setup to reinforcement-learning (RL) training and to evaluate DenseMixer on even larger MoE models—those exceeding 100 B parameters, such as Qwen3-235B-A22B, Llama-4 (109 B), and DeepSeekV3 (675 B).

# Citation

```latex
@misc{yao2025densemixer,
  title = {DenseMixer: Solving MoE Post-Training with Precise Router Gradients},
  url = {https://fengyao.notion.site/moe-posttraining},
  author = {Yao, Feng and Cui, Junxia and Zhang, Ruohan and Liu, Liyuan and Hao, Shibo and Zhang, Li and Dong, Chengyu and Wang, Shuohang and Shen, Yelong and Gao, Jianfeng and Shang, Jingbo},
  journal = {Feng Yao's Notion},
  year = {2025},
  month = jun
}
```