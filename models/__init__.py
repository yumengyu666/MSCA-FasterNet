"""MSCA-FasterNet: Lightweight Crop Pest and Disease Identification Model.

Based on improved FasterNet with Multi-Scale Channel Attention (MSCA)
and Cross-Layer Feature Fusion.
"""

from .fasternet import FasterNet, FasterNetT0, DropPath
from .msca import MSCA, MSCALight, SEOnly
from .fusion import CrossLayerFusion, TwoStageFusion
from .msca_fasternet import (
    MSCAFasterNet,
    msca_fasternet_t0,
    fasternet_t0_baseline,
    fasternet_t0_with_msca,
    fasternet_t0_with_fusion,
    fasternet_t0_full,
)

__all__ = [
    "FasterNet",
    "FasterNetT0",
    "DropPath",
    "MSCA",
    "MSCALight",
    "SEOnly",
    "CrossLayerFusion",
    "TwoStageFusion",
    "MSCAFasterNet",
    "msca_fasternet_t0",
    "fasternet_t0_baseline",
    "fasternet_t0_with_msca",
    "fasternet_t0_with_fusion",
    "fasternet_t0_full",
]
