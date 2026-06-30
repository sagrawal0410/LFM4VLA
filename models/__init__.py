import sys

from models.model_backbone import RoboVLMBackbone, deep_update, load_config
from models.robo_lfm import RoboLFM25VL

# Config uses robovlm_name "RoboLFM2.5" (not a valid Python identifier for direct import).
setattr(sys.modules[__name__], "RoboLFM2.5", RoboLFM25VL)

__all__ = [
    "RoboVLMBackbone",
    "RoboLFM25VL",
    "load_config",
    "deep_update",
]
