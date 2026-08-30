"""PI controller with saturation, conditional integration and rate limiting.

Design notes that are not decoration:

* `K_p = 0` and `K_i = 0` must reproduce the open-loop baseline exactly. The
  baseline is a special case of the method, so any measured difference is
  attributable to the loop and nothing else. `test_reduces_to_baseline` in
  tests/ is the check; run it before trusting any experiment.

* Conditional integration is the anti-windup. Without it, during the early
  phase when the edit physically cannot move yet, the error stays large, the
  integrator grows without bound, and by the time the state becomes responsive
  the controller slams the input to u_max and stays there. That failure looks
  exactly like the over-editing artefacts already documented in the thesis.

* Rate limiting exists because the plant is stochastic. A large step in
  guidance between consecutive iterations injects a transient the state cannot
  absorb.

* The runaway monitor implements the guard for the sign-inversion failure
  mode: if progress falls while the input is pinned at maximum, the loop is
  pushing in a direction that is making things worse. That is structurally the
  same self-extinction loop measured in the thesis, where suppression erased
  the very variance that triggered it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PIConfig:
    u_nom: float                    # nominal input; K_p = K_i = 0 reproduces this
    u_min: float
    u_max: float
    k_p: float = 0.0
    k_i: float = 0.0
    delta_max: Optional[float] = None   # per-iteration rate limit; None disables
    # Runaway guard: trip if progress falls while saturated high for this many
    # consecutive iterations. None disables.
    runaway_patience: Optional[int] = 50


@dataclass
class ControllerState:
    u: float = 0.0
    e: float = 0.0
    integral: float = 0.0
    saturated: bool = False
    tripped: bool = False
    trip_reason: str = ""


class PIController:
    def __init__(self, cfg: PIConfig):
        if cfg.u_min > cfg.u_max:
            raise ValueError("u_min must be <= u_max")
        if not (cfg.u_min <= cfg.u_nom <= cfg.u_max):
            raise ValueError("u_nom must lie within [u_min, u_max]")
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._u_prev = self.cfg.u_nom
        self._integral = 0.0
        self._prev_sat = False
        self._high_sat_streak = 0
        self._p_at_streak_start: Optional[float] = None
        self.tripped = False
        self.trip_reason = ""
        self.history: List[ControllerState] = []

    @property
    def is_open_loop(self) -> bool:
        return self.cfg.k_p == 0.0 and self.cfg.k_i == 0.0

    def step(
        self,
        reference: Optional[float],
        measurement: Optional[float],
        raw_progress: Optional[float] = None,
    ) -> float:
        """Advance one iteration and return the guidance strength to apply.

        `reference is None` means the reference is not yet defined (warm-up).
        In that case hold at u_nom and do NOT integrate, which is the
        cleanest way to avoid winding up before the target exists.
        """
        cfg = self.cfg

        if self.tripped:
            return cfg.u_nom

        if reference is None or measurement is None:
            u = cfg.u_nom
            self._u_prev = u
            self.history.append(ControllerState(u=u, e=0.0, integral=self._integral))
            return u

        e = float(reference) - float(measurement)

        # Conditional integration: only accumulate when the previous input was
        # not saturated.
        if not self._prev_sat:
            self._integral += cfg.k_i * e

        u_raw = cfg.u_nom + cfg.k_p * e + self._integral

        # Rate limit before saturation so the limit is on the commanded value.
        if cfg.delta_max is not None:
            lo = self._u_prev - cfg.delta_max
            hi = self._u_prev + cfg.delta_max
            u_raw = min(max(u_raw, lo), hi)

        u = min(max(u_raw, cfg.u_min), cfg.u_max)
        saturated = (u != u_raw) or (u in (cfg.u_min, cfg.u_max))

        self._check_runaway(u, raw_progress if raw_progress is not None else measurement)

        self._prev_sat = saturated
        self._u_prev = u
        self.history.append(
            ControllerState(u=u, e=e, integral=self._integral, saturated=saturated,
                            tripped=self.tripped, trip_reason=self.trip_reason)
        )
        return u

    def _check_runaway(self, u: float, progress: float) -> None:
        cfg = self.cfg
        if cfg.runaway_patience is None:
            return

        at_ceiling = u >= cfg.u_max - 1e-12
        if not at_ceiling:
            self._high_sat_streak = 0
            self._p_at_streak_start = None
            return

        if self._p_at_streak_start is None:
            self._p_at_streak_start = progress
            self._high_sat_streak = 1
            return

        self._high_sat_streak += 1
        if self._high_sat_streak >= cfg.runaway_patience:
            if progress < self._p_at_streak_start:
                self.tripped = True
                self.trip_reason = (
                    f"progress fell from {self._p_at_streak_start:.4f} to {progress:.4f} "
                    f"while u was pinned at u_max={cfg.u_max} for "
                    f"{self._high_sat_streak} iterations: loop gain has inverted sign"
                )
            # Restart the window either way so a slow rise does not trip later.
            self._high_sat_streak = 0
            self._p_at_streak_start = progress
