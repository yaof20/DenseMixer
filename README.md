# DenseMixer

**DenseMixer** enhances Mixture of Experts (MoE) models in HuggingFace Transformers with more precise router gradients for improved post-training.

- 📖 [Blog & Theory](https://fengyao.notion.site/moe-posttraining)
- 🧪 Reproducible experiments: see [experiments](./experiments) folder

## Supported Models

- [Qwen3-MoE](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base)
- [Qwen2-MoE](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B)
- [OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0125)

## Installation

### Basic Installation

```bash
pip install densemixer
```

### Auto-Patching Setup (Recommended)

To enable DenseMixer automatically for all Python scripts (without needing to add any import):

```bash
densemixer setup
export DENSEMIXER_ENABLED=1
```

This will append the necessary auto-import logic to your `usercustomize.py` in your user site-packages (if not already present). Any Python process with `DENSEMIXER_ENABLED=1` will auto-load DenseMixer and patch transformers models.

**Note:** DenseMixer is **disabled by default** for safety. You must explicitly set `DENSEMIXER_ENABLED=1` to enable it.

To disable, either unset the environment variable or manually remove the relevant lines from your `usercustomize.py` in your user site-packages.

## Usage

### Basic Usage

```python
from transformers import Qwen3MoeForCausalLM

model = Qwen3MoeForCausalLM.from_pretrained("Qwen/Qwen3-MoE-30B-A3B")
```

When DenseMixer patches are applied, you'll see messages like:
```
INFO - densemixer - DenseMixer: Using custom forward method for Qwen3-MoE
INFO - densemixer - DenseMixer: Using custom forward method for Qwen2-MoE
INFO - densemixer - DenseMixer: Using custom forward method for OLMoE
```

## Configuration

DenseMixer can be controlled with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DENSEMIXER_ENABLED` | `0` | Master switch to enable/disable DenseMixer |
| `DENSEMIXER_QWEN3` | `1` | Enable for Qwen3-MoE models (when DENSEMIXER_ENABLED=1) |
| `DENSEMIXER_QWEN2` | `1` | Enable for Qwen2-MoE models (when DENSEMIXER_ENABLED=1) |
| `DENSEMIXER_OLMOE` | `1` | Enable for OLMoE models (when DENSEMIXER_ENABLED=1) |

### Examples

**Enable DenseMixer for all models:**
```bash
export DENSEMIXER_ENABLED=1
python your_script.py
```

**Enable only for Qwen3-MoE:**
```bash
export DENSEMIXER_ENABLED=1
export DENSEMIXER_QWEN3=1
export DENSEMIXER_QWEN2=0
export DENSEMIXER_OLMOE=0
python your_script.py
```

**Disable DenseMixer completely (default behavior):**
```bash
# No environment variables needed - DenseMixer is disabled by default
python your_script.py
```