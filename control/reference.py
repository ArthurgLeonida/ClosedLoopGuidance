"""Reference trajectory: where the edit should be, not how hard to push.

The central reframing. A hand-drawn curve is a poor way to say *how hard to
push*, because the same push produces different results in different scenes.
It is a perfectly good way to say *where the edit should be by now*. So the
bump-shaped schedule moves from the control input to the reference.

    r_t = p_target * phi(tau),    tau = t / T in [0, 1]

The only knob left with semantic meaning is `p_target`: how different the
finished result should be from the source. That is a property of the requested
edit, not of the algorithm.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional


def ramp(tau: float, kappa: float = 1.0) -> float:
    """phi(tau) = tau ** kappa. kappa < 1 front-loads the edit, > 1 delays it."""
    return float(min(max(tau, 0.0), 1.0) ** kappa)


def smoothstep(tau: float) -> float:
    """Classic 3t^2 - 2t^3. Zero slope at both ends, so the reference does not
    demand an instantaneous jump at t=0, which would spike the error term."""
    t = min(max(tau, 0.0), 1.0)
    return float(t * t * (3.0 - 2.0 * t))


class ReferenceTrajectory:
    """r_t over a run of `total_iterations`.

    `p_target` may be supplied directly, or left None and learned from the
    demand signal during a warm-up window (see `observe_demand`). The second
    mode is what preserves the no-per-scene-tuning property that Part 1 of the
    thesis rests on: prompt pairs demanding a large change produce a large
    mean s early, hence a high target, with one global constant `alpha`.
    """

    def __init__(
        self,
        total_iterations: int,
        p_target: Optional[float] = None,
        shape: Callable[[float], float] = smoothstep,
        alpha: float = 1.0,
        warmup_iterations: int = 200,
        p_target_bounds: tuple = (0.02, 0.60),
    ):
        self.total_iterations = int(total_iterations)
        self.shape = shape
        self.alpha = float(alpha)
        self.warmup_iterations = int(warmup_iterations)
        self.p_target_bounds = p_target_bounds

        self._p_target = p_target
        self._demand_samples: List[float] = []

    # -- warm-up estimation -------------------------------------------------

    def observe_demand(self, s: float) -> None:
        """Feed one demand sample during the warm-up window.

        Ignored once p_target is fixed, so it is safe to call unconditionally
        from the training loop.
        """
        if self._p_target is not None:
            return
        if len(self._demand_samples) < self.warmup_iterations:
            self._demand_samples.append(float(s))

    @property
    def warmed_up(self) -> bool:
        return (
            self._p_target is not None
            or len(self._demand_samples) >= self.warmup_iterations
        )

    @property
    def p_target(self) -> Optional[float]:
        if self._p_target is not None:
            return self._p_target
        if not self._demand_samples:
            return None
        if len(self._demand_samples) < self.warmup_iterations:
            return None
        mean_s = sum(self._demand_samples) / len(self._demand_samples)
        lo, hi = self.p_target_bounds
        self._p_target = float(min(max(self.alpha * mean_s, lo), hi))
        return self._p_target

    # -- the reference itself -----------------------------------------------

    def at(self, iteration: int) -> Optional[float]:
        """r_t, or None while still warming up.

        A None return is the signal to the controller to hold at u_nom: there
        is no meaningful error before the target exists, and integrating one
        would be the textbook way to wind up.
        """
        target = self.p_target
        if target is None:
            return None
        tau = iteration / max(self.total_iterations, 1)
        return float(target * self.shape(tau))
