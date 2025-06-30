"""
DenseMixer: Enhanced Mixture of Experts implementation with optimized router.
"""
import os
import logging
import sys

__version__ = "0.1.0"

# Set up logging to ensure patching information is always visible
def setup_logging():
    """Set up logging to ensure patching information is always logged"""
    # Get the densemixer logger
    logger = logging.getLogger("densemixer")
    
    # Only set up if no handlers are already configured
    if not logger.handlers:
        # Create a console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        
        # Prevent propagation to avoid duplicate messages
        logger.propagate = False

# Set up logging immediately
setup_logging()
logger = logging.getLogger(__name__)

# Configuration system
class Config:
    def __init__(self):
        # Global switch - disabled by default for safety
        self.enabled = self._get_env_bool("DENSEMIXER_ENABLED", False)
        
        # Model-specific switches - enabled by default when DENSEMIXER_ENABLED=1
        self.models = {
            "qwen3": self._get_env_bool("DENSEMIXER_QWEN3", True),
            "olmoe": self._get_env_bool("DENSEMIXER_OLMOE", True),
            "qwen2": self._get_env_bool("DENSEMIXER_QWEN2", True)
        }
    
    def _get_env_bool(self, name, default=False):
        """Get boolean from environment variable"""
        val = os.environ.get(name, str(default).lower())
        return val in ("1", "true", "yes", "on", "t")
    
    def is_model_enabled(self, model_name):
        """Check if a specific model is enabled"""
        return self.enabled and self.models.get(model_name, False)

# Initialize configuration
config = Config()

# Apply patches if enabled
if config.enabled:
    from .patching import apply_patches
    patched_models = apply_patches(config)
    if patched_models:
        logger.info(f"DenseMixer applied to: {', '.join(patched_models)}")
    else:
        logger.info("DenseMixer enabled but no models were patched")
else:
    logger.debug("DenseMixer disabled by default. Set DENSEMIXER_ENABLED=1 to enable.")