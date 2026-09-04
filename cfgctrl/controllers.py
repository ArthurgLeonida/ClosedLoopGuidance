"""Guidance as feedback control: CFG as P-control, SMC-CFG, and refinements.

Model-agnostic implementation of the guidance law in

    CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance
    Wang, Liu, Chi, Liu, Xue, Duan. CVPR 2026 (highlight). arXiv:2603.03281

plus several control-theoretic refinements of it. Nothing here knows about
images, UNets or transformers. The only input is the semantic error tensor

    e_t = v_theta(x_t, t, c) - v_theta(x_t, t, None)          (paper Eq. 6)

and the only output is the *applied* error  e_t + delta_e_t, which the caller
turns into a velocity exactly as the paper does (Algorithm 1, line 13):

    v_hat = v_theta(x_t, t, None) + w * (e_t + delta_e_t)

`k = 0` reproduces classifier-free guidance *bit-exactly*. The baseline is a
special case of the method, the same principle as control/controller.py, so
any measured difference between arms is attributable to the correction.

Paper's law (Algorithm 1; authors' pipeline/common_cfg_ctrl.py):

    s_t      = (e_t - e_{t-1}) + lam * e_{t-1}      sliding variable (Eq. 19)
    delta_e  = -k * sign(s_t)                        switching control (Eq. 25)
    first step:  e_{t-1} := e_t   (so  s_0 = lam * e_0)
    e_{t-1} stored for the next step is the CORRECTED error e_t + delta_e
    (authors' code: `state.prev_guidance_eps = guidance_eps.detach()` after
    `guidance_eps = guidance_eps + u_sw`).

Refinements. Each is a flag; with everything off you get the paper.

    time_scaled     divide the finite difference by the flow-time step:
                    (e_t - e_{t-1}) / dt. Then lam has units 1/time and the
                    same lam defines the same surface at 20, 30 or 50 steps.
    switching       'sign' (paper) | 'sat' | 'tanh': boundary layer of
                    half-width phi (Slotine & Li, Applied Nonlinear Control,
                    ch. 7). Removes chattering. With phi = k*lam the
                    first-order behaviour is exact soft-thresholding of e
                    by k (derivation in docs/CFG-Ctrl_Review_and_Improvements.md).
    direction       'elementwise' (paper: one sign per latent element; the
                    correction has norm k*sqrt(D)) | 'unit' (one direction
                    per sample; norm k). 'unit' removes the sqrt(D) factor
                    from the paper's Assumption 2 / Theorem 1.
    reaching_q      adds -q*s, Gao's proportional reaching term.
    super_twisting  second-order sliding mode (Levant 1993): continuous
                    control  -k*|s|^{1/2}*sw(s) + z,  z <- z - k2*dt*sw(s),
                    with |z| <= z_max as anti-windup.
    adaptive        gain adaptation (Plestan, Shtessel, Bregeault, Poznyak
                    2010), one scalar per sample:
                    k <- clip(k + rate*dt*rms(s)*sign(rms(s) - band), k_min, k_max)
    relative_gain   k is a fraction of rms(e_t) rather than an absolute
                    velocity unit, so one k transfers across models whose
                    velocity scales differ (the paper needs k=0.1 for SD3.5
                    and Qwen-Image but k=0.7 for Flux).
    store_corrected keep the *applied* error as e_{t-1} (authors' code, the
                    default) or the measured one (False). Storing the
                    corrected error feeds the previous correction back into
                    the surface with coefficient (lam - 1) [unscaled] or
                    (lam - 1/dt) [time-scaled]. Once e ~ 0 that makes
                    s ~ (lam - 1) * delta_prev: for lam > 1 the sign
                    alternates every step (chatter of amplitude k), for
                    lam < 1 or with time scaling it locks a constant bias
                    of size k for the rest of the trajectory. The
                    refinement presets therefore store the measured error.
    flip_sign       apply +k*sw(s) instead of -k*sw(s). On a plant where
                    more guidance makes the *measured* error smaller at the
                    next step (the paper's own premise), the one-step loop
                    gain is negative and this is the direction the paper's
                    Lyapunov argument actually requires. Kept as a
                    diagnostic: it over-guides.
    excess_only     shrink only the extrapolation (w - 1) * e, never the
                    conditional prediction itself:
                        v_hat = v_cond + (w - 1) * (e + delta_e)
                              = v_uncond + w * (e + (w-1)/w * delta_e).
                    The paper's law modifies v_cond + (w-1) e as a whole, so
                    at w = 1 it no longer samples the conditional law (toy
                    plant: Frechet 0.16 vs 0.10 for CFG, alignment below the
                    true class law). With excess_only the law is exactly CFG
                    at w = 1 for any k. Needs `w` passed to `correct`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Literal, Optional, Union

import torch

Switching = Literal["sign", "sat", "tanh"]
Direction = Literal["elementwise", "unit"]


@dataclass
class SMCConfig:
    lam: float = 6.0
    k: float = 0.1
    time_scaled: bool = False
    switching: Switching = "sign"
    phi: float = 0.0
    direction: Direction = "elementwise"
    reaching_q: float = 0.0
    super_twisting: bool = False
    k2: float = 0.0
    z_max: float = float("inf")
    adaptive: bool = False
    adapt_rate: float = 0.0
    adapt_band: float = 0.0
    k_min: float = 0.0
    k_max: float = float("inf")
    relative_gain: bool = False
    store_corrected: bool = True
    flip_sign: bool = False
    excess_only: bool = False
    warmup_steps: int = 0

    def validate(self) -> None:
        if self.lam < 0:
            raise ValueError("lam must be >= 0")
        if self.k < 0:
            raise ValueError("k must be >= 0")
        if self.switching not in ("sign", "sat", "tanh"):
            raise ValueError(f"unknown switching function {self.switching!r}")
        if self.switching != "sign" and self.phi <= 0:
            raise ValueError("'sat' and 'tanh' switching need a boundary layer phi > 0")
        if self.direction not in ("elementwise", "unit"):
            raise ValueError(f"unknown direction {self.direction!r}")
        if self.reaching_q < 0 or self.k2 < 0 or self.adapt_rate < 0:
            raise ValueError("reaching_q, k2 and adapt_rate must be >= 0")
        if self.adaptive and not (self.k_min <= self.k <= self.k_max):
            raise ValueError("adaptive: need k_min <= k <= k_max")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")

    @property
    def is_cfg(self) -> bool:
        """True when the law can never modify e (pure classifier-free guidance)."""
        return (
            self.k == 0.0
            and self.reaching_q == 0.0
            and not self.super_twisting
            and not self.adaptive
            and self.warmup_steps == 0
        )


@dataclass
class StepInfo:
    """Batch-averaged diagnostics for one call of `correct`."""
    step: int
    dt: Optional[float]
    e_rms: float            # rms of the measured error
    s_rms: float            # rms of the sliding variable
    delta_rms: float        # rms of the correction actually applied
    chatter: float          # fraction of elements whose sign(s) flipped since last step
    deriv_matters: float    # fraction of elements where sign(s) != sign(e_prev)
    inside_layer: float     # fraction of elements with |s| < phi (0 for 'sign')
    k_eff: float            # mean effective gain
    z_rms: float = 0.0      # super-twisting integrator
    switch_activity: float = 0.0  # rms(delta_t - delta_{t-1}): 2k per element for full chatter


@dataclass
class _State:
    e_prev: Optional[torch.Tensor] = None
    sign_prev: Optional[torch.Tensor] = None
    delta_prev: Optional[torch.Tensor] = None
    z: Optional[torch.Tensor] = None
    k: Optional[torch.Tensor] = None
    step: int = 0


def _per_sample(x: torch.Tensor, fn) -> torch.Tensor:
    """Reduce over every dim but the batch, keepdim so it broadcasts back."""
    dims = tuple(range(1, x.dim()))
    return fn(x, dims)


def rms(x: torch.Tensor) -> torch.Tensor:
    return _per_sample(x, lambda t, d: t.pow(2).mean(dim=d, keepdim=True)).sqrt()


def soft_threshold(e: torch.Tensor, k: float) -> torch.Tensor:
    """sign(e) * max(|e| - k, 0): the L1 proximal operator."""
    return torch.sign(e) * torch.clamp(e.abs() - k, min=0.0)


class SlidingModeGuidance:
    """Stateful guidance law. Call `correct(e_t, dt)` once per denoising step."""

    def __init__(self, cfg: Optional[SMCConfig] = None, **overrides):
        cfg = cfg if cfg is not None else SMCConfig()
        if overrides:
            cfg = replace(cfg, **overrides)
        cfg.validate()
        self.cfg = cfg
        self.history: List[StepInfo] = []
        self._st = _State()

    # ------------------------------------------------------------------ api
    def reset(self) -> None:
        self._st = _State()
        self.history = []

    @property
    def is_cfg(self) -> bool:
        return self.cfg.is_cfg

    @property
    def step_index(self) -> int:
        return self._st.step

    @property
    def gain(self) -> Optional[torch.Tensor]:
        """Current adaptive gain per sample (None unless adaptive)."""
        return self._st.k

    def guided_velocity(
        self,
        v_uncond: torch.Tensor,
        v_cond: torch.Tensor,
        w: float,
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        """v_uncond + w * (e + delta_e), the paper's Algorithm 1 line 13."""
        return v_uncond + w * self.correct(v_cond - v_uncond, dt, w=w)

    def correct(self, e: torch.Tensor, dt: Optional[float] = None,
                w: Optional[float] = None) -> torch.Tensor:
        """Return the applied error e + delta_e for this step.

        Args:
            e:  measured semantic error, shape [B, ...]. Batch is dim 0.
            dt: flow-time step (sigma_t - sigma_{t-1} > 0). Required when
                `time_scaled`, `super_twisting` or `adaptive` is on because
                those integrate in time; ignored otherwise.
            w:  the guidance scale the caller will multiply the result by.
                Required when `excess_only` is on; ignored otherwise.
        """
        cfg, st = self.cfg, self._st
        if e.dim() < 1:
            raise ValueError("e must have a batch dimension")
        if cfg.excess_only and (w is None or w < 1.0):
            raise ValueError("excess_only needs the guidance scale w >= 1")

        # ---- warm-up: the authors return the unconditional prediction, i.e.
        # no guidance at all, for the first `warmup_steps` steps.
        if st.step < cfg.warmup_steps:
            st.step += 1
            e_r = float(rms(e.detach()).mean())
            self.history.append(StepInfo(st.step - 1, dt, e_r, 0.0, e_r, 0.0, 0.0, 0.0, 0.0))
            return torch.zeros_like(e)

        needs_dt = cfg.time_scaled or cfg.super_twisting or cfg.adaptive
        if needs_dt and (dt is None or dt <= 0):
            raise ValueError("this configuration integrates in time and needs dt > 0")
        tstep = float(dt) if (dt is not None and dt > 0) else 1.0

        e_meas = e.detach()
        if st.e_prev is None:
            st.e_prev = e_meas.clone()                     # paper: s_0 = lam * e_0

        # ---- sliding variable  s = e_dot + lam * e_prev
        de = e_meas - st.e_prev
        if cfg.time_scaled:
            de = de / tstep
        s = de + cfg.lam * st.e_prev

        # ---- switching function sw(s) in [-1, 1] (elementwise) or unit vector
        if cfg.direction == "elementwise":
            if cfg.switching == "sign":
                sw = torch.sign(s)
            elif cfg.switching == "sat":
                sw = torch.clamp(s / cfg.phi, -1.0, 1.0)
            else:
                sw = torch.tanh(s / cfg.phi)
            mag = s.abs()
        else:
            n = _per_sample(s, lambda t, d: t.pow(2).sum(dim=d, keepdim=True)).sqrt().clamp_min(1e-12)
            unit = s / n
            if cfg.switching == "sign":
                sw = unit
            elif cfg.switching == "sat":
                sw = unit * torch.clamp(n / cfg.phi, max=1.0)
            else:
                sw = unit * torch.tanh(n / cfg.phi)
            mag = n.expand_as(s)

        # ---- gain
        if cfg.adaptive:
            if st.k is None:
                st.k = torch.full_like(rms(e_meas), cfg.k)
            k_eff: Union[float, torch.Tensor] = st.k
        else:
            k_eff = cfg.k
        if cfg.relative_gain:
            k_eff = k_eff * rms(e_meas)

        # ---- correction
        if cfg.super_twisting:
            if st.z is None:
                st.z = torch.zeros_like(e_meas)
            delta = -k_eff * mag.sqrt() * sw + st.z
            st.z = torch.clamp(st.z - cfg.k2 * tstep * sw, -cfg.z_max, cfg.z_max)
        else:
            delta = -k_eff * sw
        if cfg.reaching_q > 0:
            delta = delta - cfg.reaching_q * s
        if cfg.flip_sign:
            delta = -delta
        if cfg.excess_only:
            delta = delta * ((w - 1.0) / w)                # shrink only the extrapolation

        e_app = e + delta                                  # keeps e's autograd graph if any

        # ---- adaptive-gain update (Plestan et al.), scalar per sample
        if cfg.adaptive:
            s_rms = rms(s)
            st.k = torch.clamp(
                st.k + cfg.adapt_rate * tstep * s_rms * torch.sign(s_rms - cfg.adapt_band),
                cfg.k_min, cfg.k_max,
            )

        # ---- diagnostics
        sgn = torch.sign(s)
        chatter = float((sgn * st.sign_prev < 0).float().mean()) if st.sign_prev is not None else 0.0
        deriv_matters = float((sgn * torch.sign(st.e_prev) < 0).float().mean())
        inside = float((s.abs() < cfg.phi).float().mean()) if cfg.phi > 0 else 0.0
        k_mean = float(k_eff.mean()) if torch.is_tensor(k_eff) else float(k_eff)
        delta_d = delta.detach()
        activity = float(rms(delta_d - st.delta_prev).mean()) if st.delta_prev is not None else 0.0
        self.history.append(StepInfo(
            step=st.step, dt=dt,
            e_rms=float(rms(e_meas).mean()),
            s_rms=float(rms(s).mean()),
            delta_rms=float(rms(delta_d).mean()),
            chatter=chatter, deriv_matters=deriv_matters, inside_layer=inside,
            k_eff=k_mean,
            z_rms=float(rms(st.z).mean()) if st.z is not None else 0.0,
            switch_activity=activity,
        ))

        # ---- state
        st.sign_prev = sgn
        st.delta_prev = delta_d.clone()
        st.e_prev = (e_app.detach() if cfg.store_corrected else e_meas).clone()
        st.step += 1
        return e_app


# ---------------------------------------------------------------- presets
def cfg_baseline() -> SMCConfig:
    """Plain classifier-free guidance: P-control with gain w, no correction."""
    return SMCConfig(k=0.0)


def paper(lam: float = 6.0, k: float = 0.1) -> SMCConfig:
    """SMC-CFG exactly as in Algorithm 1 and the authors' code.

    Tuned values reported in the supplementary (Sec. 7.3): lam = 6 for all
    three models; k = 0.1 for SD3.5 and Qwen-Image, k = 0.7 for Flux-dev.
    """
    return SMCConfig(lam=lam, k=k)


def lyapunov_consistent(lam: float = 6.0, k: float = 0.1) -> SMCConfig:
    """Paper's law with the switching direction its Lyapunov argument needs on
    a negative-loop-gain plant. Diagnostic only: it over-guides."""
    return SMCConfig(lam=lam, k=k, flip_sign=True)


def boundary_layer(lam: float = 6.0, k: float = 0.1, phi: Optional[float] = None,
                   time_scaled: bool = False) -> SMCConfig:
    """Paper's law with sat(s/phi) instead of sign(s), storing the measured
    error. Default phi = k*lam makes the first-order behaviour exact
    soft-thresholding of e by k."""
    return SMCConfig(lam=lam, k=k, switching="sat", store_corrected=False,
                     phi=(k * lam if phi is None else phi), time_scaled=time_scaled)


def boundary_layer_excess(lam: float = 6.0, k: float = 0.1, phi: Optional[float] = None) -> SMCConfig:
    """Boundary-layer law applied to the extrapolation only: exactly CFG at
    w = 1, soft-threshold of the (w-1) e term otherwise. Pass w to correct()."""
    return SMCConfig(lam=lam, k=k, switching="sat", store_corrected=False, excess_only=True,
                     phi=(k * lam if phi is None else phi))


def super_twisting(lam: float = 6.0, k1: float = 0.05, k2: float = 0.05,
                   phi: Optional[float] = None, z_max: float = 0.5) -> SMCConfig:
    """Second-order sliding mode (Levant): continuous correction.

    Caveat, learned the hard way: super-twisting assumes the sliding variable
    has relative degree one (s_dot = u + d). The paper's s = e_dot + lam*e
    depends on the correction *algebraically* (relative degree zero), so on
    a plant where the correction has real authority the law settles into a
    two-step cycle |s| = k1^2 (tests/test_smc_cfg.py). Kept as a documented
    negative result; on the diffusion plant the authority is tiny and the
    law degenerates to a smooth shrink like the others."""
    return SMCConfig(lam=lam, k=k1, k2=k2, super_twisting=True, time_scaled=True,
                     store_corrected=False, switching="sat",
                     phi=(k1 * lam if phi is None else phi), z_max=z_max)


def adaptive_gain(lam: float = 6.0, k0: float = 0.05, rate: float = 1.0, band: float = 1.0,
                  k_min: float = 0.0, k_max: float = 1.0, phi: Optional[float] = None) -> SMCConfig:
    """Self-tuning k: grows while rms(s) > band, shrinks inside the band."""
    return SMCConfig(lam=lam, k=k0, adaptive=True, adapt_rate=rate, adapt_band=band,
                     k_min=k_min, k_max=k_max, time_scaled=True, store_corrected=False,
                     switching="sat", phi=(k0 * lam if phi is None else phi))
