"""
Patching mechanism for the transformers library to enhance MoE implementations.
"""

import logging
import sys
from importlib.util import find_spec

def get_densemixer_logger():
    """Get the densemixer logger with guaranteed console output"""
    logger = logging.getLogger("densemixer")
    
    # Ensure the logger has a console handler
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    
    return logger

logger = get_densemixer_logger()

def apply_patches(config):
    """
    Apply the DenseMixer patch to multiple MoE models in the transformers library.
    
    Args:
        config: Configuration object with enabled status for each model
        
    Returns:
        list: Names of models that were successfully patched
    """
    # Check if transformers is installed
    if find_spec("transformers") is None:
        logger.warning(
            "The transformers library is not installed. The DenseMixer patch will not be applied."
        )
        return []
    
    patched_models = []
    
    # Patch OLMoE if enabled
    if config.is_model_enabled("olmoe"):
        if apply_olmoe_patch():
            patched_models.append("OLMoE")
    
    # Patch Qwen2-MoE if enabled
    if config.is_model_enabled("qwen2"):
        if apply_qwen2_moe_patch():
            patched_models.append("Qwen2-MoE")

    # Patch Qwen3-MoE if enabled
    if config.is_model_enabled("qwen3"):
        if apply_qwen3_moe_patch():
            patched_models.append("Qwen3-MoE")
    
    if patched_models:
        logger.info(f"DenseMixer patches successfully applied to: {', '.join(patched_models)}")
    
    return patched_models


def apply_olmoe_patch():
    """Apply patch to OLMoE model"""
    try:
        from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
        from .models.olmoe_custom import CustomOlmoeSparseMoeBlock

        # Apply the patch by replacing the forward method
        OlmoeSparseMoeBlock.forward = CustomOlmoeSparseMoeBlock.forward
        
        logger.info("Successfully patched OLMoE")
        return True
    except ImportError as e:
        logger.warning(f"OLMoE module not found: {e}")
        return False
    except Exception as e:
        logger.error(f"Error patching OLMoE: {e}")
        return False


def apply_qwen2_moe_patch():
    """Apply patch to Qwen2-MoE model"""
    try:
        from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
        from .models.qwen2_moe_custom import CustomQwen2MoeSparseMoeBlock

        # Apply the patch by replacing the forward method
        Qwen2MoeSparseMoeBlock.forward = CustomQwen2MoeSparseMoeBlock.forward
        
        logger.info("Successfully patched Qwen2-MoE")
        return True
    except ImportError as e:
        logger.warning(f"Qwen2-MoE module not found: {e}")
        return False
    except Exception as e:
        logger.error(f"Error patching Qwen2-MoE: {e}")
        return False


def apply_qwen3_moe_patch():
    """Apply patch to Qwen3-MoE model"""
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
        from .models.qwen3_moe_custom import CustomQwen3MoeSparseMoeBlock
        
        # Apply the patch by replacing the forward method
        Qwen3MoeSparseMoeBlock.forward = CustomQwen3MoeSparseMoeBlock.forward

        logger.info("Successfully patched Qwen3MoeSparseMoeBlock")
        return True
    except ImportError as e:
        logger.warning(f"Qwen3-MoE module not found: {e}")
        return False
    except Exception as e:
        logger.error(f"Error patching Qwen3-MoE: {e}")
        return False