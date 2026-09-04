"""CFG-Ctrl (CVPR 2026) reimplementation plus control-theoretic refinements.

    from cfgctrl import SlidingModeGuidance, presets
    ctrl = SlidingModeGuidance(presets.paper(lam=6.0, k=0.1))
    v_hat = ctrl.guided_velocity(v_uncond, v_cond, w=7.5, dt=sigma_prev - sigma)
"""
from . import controllers as presets
from .controllers import SMCConfig, SlidingModeGuidance, StepInfo, soft_threshold
from .toy_flow import GaussianMixtureFlow, SampleTrace, bimodal_classes, ring_mixture

__all__ = [
    "SMCConfig", "SlidingModeGuidance", "StepInfo", "soft_threshold", "presets",
    "GaussianMixtureFlow", "SampleTrace", "ring_mixture", "bimodal_classes",
]
