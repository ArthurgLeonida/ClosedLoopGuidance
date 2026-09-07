"""CFG-Ctrl (CVPR 2026) reimplemented, plus candidate refinements.
See docs/Chattering_Fixes.md for evidence and limitations.

    from cfgctrl import SlidingModeGuidance, presets
    ctrl = SlidingModeGuidance(presets.paper(lam=6.0, k=0.1))         # the paper
    ctrl = SlidingModeGuidance(presets.boundary_layer_excess())       # candidate
    v_hat = ctrl.guided_velocity(v_uncond, v_cond, w=7.5)
"""
from . import controllers as presets
from .controllers import SMCConfig, SlidingModeGuidance, StepInfo, soft_threshold

__all__ = [
    "SMCConfig", "SlidingModeGuidance", "StepInfo", "soft_threshold", "presets",
]
