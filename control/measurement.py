"""Measurement layer: what the controller reads.

Two distinct quantities, deliberately kept separate. See
`docs/ClosedLoopGuidance.md` section 3 in the thesis repository.

  demand      s_t  -- how different the diffusion model *wants* target from
                      source right now. Already implemented upstream as
                      `dc.guidance_utils.compute_edit_strength`; re-exported
                      here so the control layer has one import surface.

  achievement p_t  -- how far the optimised state has *actually* moved from
                      the source. New. This is the plant output the loop
                      closes on, and it is the thing no prior feedback-guidance
                      work measures, because it only exists when the plant is
                      an optimisation with accumulating state.

Both are bounded in [0, 1] by the triangle inequality, ||a - b|| <= ||a|| + ||b||,
with no clipping and no learned normalisation. The ratio form also cancels the
overall magnitude, which drifts with the timestep, so values are comparable
across iterations and across scenes.
"""

from __future__ import annotations

from typing import Optional

import torch

# Re-export so callers never reach into the vendored package directly.
from dc.guidance_utils import compute_edit_strength as demand  # noqa: F401


@torch.no_grad()
def edit_progress(z_t: torch.Tensor, z_src: torch.Tensor) -> float:
    """Achievement signal p_t in [0, 1].

    Args:
        z_t:    current state. Plant A: the latent being optimised.
                Plant B: the VAE latent of the currently rendered view.
        z_src:  the corresponding source latent.

    Bounded above by 1 because ||z_t - z_src|| <= ||z_t|| + ||z_src||.
    """
    num = torch.linalg.vector_norm(z_t - z_src)
    den = torch.linalg.vector_norm(z_t) + torch.linalg.vector_norm(z_src)
    if den <= 0:
        return 0.0
    return float((num / den).clamp(0.0, 1.0))


class EMAFilter:
    """First-order low-pass on the measurement.

    p_t from a single sampled view is noisy: view selection and the diffusion
    step are both stochastic. Filtering costs phase lag of roughly
    lam / (1 - lam) iterations -- about 10 at lam=0.9, about 100 at lam=0.99.
    That trade-off is a real tuning knob, not a detail.

    Plant A has no view sampling, so p_t is much less noisy there and lam can
    be smaller. The difference between the two plants is worth reporting.
    """

    def __init__(self, lam: float = 0.9):
        if not 0.0 <= lam < 1.0:
            raise ValueError(f"lam must be in [0, 1), got {lam}")
        self.lam = lam
        self._y: Optional[float] = None

    def reset(self) -> None:
        self._y = None

    def update(self, p: float) -> float:
        # First sample initialises the filter rather than being dragged
        # toward zero, which would otherwise create a spurious startup
        # transient that the integrator would then chase.
        if self._y is None:
            self._y = float(p)
        else:
            self._y = self.lam * self._y + (1.0 - self.lam) * float(p)
        return self._y

    @property
    def value(self) -> Optional[float]:
        return self._y

    @property
    def lag_iterations(self) -> float:
        """Approximate phase lag introduced by this filter, in iterations."""
        return self.lam / (1.0 - self.lam)
