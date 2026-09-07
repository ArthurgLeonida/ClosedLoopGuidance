"""Guidance as feedback control: CFG as P-control, and SMC-CFG.

Implements the guidance law of

    CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance
    Wang, Liu, Chi, Liu, Xue, Duan. CVPR 2026. arXiv:2603.03281

model-agnostically. Nothing here knows about images, UNets or transformers.
The input is the semantic error tensor

    e_t = v_theta(x_t, t, c) - v_theta(x_t, t, None)          (paper Eq. 6)

and the output is the *applied* error e_t + delta_e_t, which the caller turns
into a velocity exactly as the paper does (Algorithm 1, line 13):

    v_hat = v_theta(x_t, t, None) + w * (e_t + delta_e_t)

`k = 0` reproduces classifier-free guidance bit-exactly, so the baseline is a
special case of the method and any measured difference is attributable to the
correction.

THE PAPER'S LAW (Algorithm 1; authors' pipeline/common_cfg_ctrl.py):

    s_t      = (e_t - e_{t-1}) + lam * e_{t-1}      sliding variable (Eq. 19)
    delta_e  = -k * sign(s_t)                       switching control (Eq. 25)
    first step:  e_{t-1} := e_t   (so  s_0 = lam * e_0)
    the CORRECTED error e_t + delta_e is stored as e_{t-1} for the next step

Tuned values (supplementary Sec. 7.3): lam = 6 for all three models;
k = 0.1 for SD3.5 and Qwen-Image, k = 0.7 for Flux-dev.

REFINEMENTS TO TEST AGAINST THE PAPER. Each is one flag; the defaults reproduce
the paper implementation. See docs/Improvement_Roadmap.md for the rationale,
limitations, and a real-model evaluation plan. Toy findings do not establish
an image-quality improvement.

    mode            'sliding' (paper) or 'proximal': memoryless soft-thresholding
                    of the current error. Proximal mode ignores lam, phi and
                    switching, requires measured memory, and has s=e only as
                    a diagnostic reference. It vanishes at e=0 regardless of
                    history and cannot reverse or amplify an error component.

    switching       'sign' (paper) or 'sat': a boundary layer of half-width
                    phi (Slotine & Li, Applied Nonlinear Control, ch. 7).
                    Smooths switching. With phi = k*lam and s = lam*e (the first
                    step, or constant measured-error memory), the law is exact
                    soft-thresholding of e by k. Historical toy: rms(s) reaches 0.010
                    instead of plateauing at 0.50, and the switching activity
                    is 0.03 of full chatter at every k against the sign law's
                    0.39 (k=0.02) to 0.83 (k=1).

    store_corrected keep the *applied* error as e_{t-1} (True, the authors'
                    code) or the measured one (False, recommended). Storing
                    the corrected error feeds the previous correction back into
                    the surface with coefficient (lam - 1). Once e ~ 0 that
                    makes s ~ (lam - 1) * delta_prev: for lam > 1 the sign
                    alternates every step (chatter of amplitude k), for lam < 1
                    it locks a constant bias of size k for the rest of the run.
                    These are small-error limits, not guarantees for a changing
                    model prediction.

    excess_only     shrink only the extrapolation (w - 1) * e, never the
                    conditional prediction itself:
                        v_hat = v_cond + (w - 1) * (e + delta_e)
                              = v_uncond + w * (e + (w-1)/w * delta_e).
                    The paper's law modifies v_cond + (w-1) e as a whole, so at
                    w = 1 it no longer samples the conditional law (measured:
                    Frechet 0.162 vs 0.102 for CFG AND alignment 0.554 vs
                    0.591 -- worse on both axes). With excess_only the law is
                    exactly CFG at w = 1 for any k. Needs `w` passed to
                    `correct`.

    relative_gain   k is a fraction of rms(e_t) rather than an absolute
                    velocity unit. With saturation, phi is also multiplied by
                    rms(e_t); this makes the law equivariant to a constant positive
                    rescaling of an error sequence. Cross-model transfer still
                    needs experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import List, Literal, Optional, Union

import torch

Switching = Literal["sign", "sat"]
CorrectionMode = Literal["sliding", "proximal"]


@dataclass
class SMCConfig:
    lam: float = 6.0
    k: float = 0.1
    switching: Switching = "sign"
    phi: float = 0.0
    store_corrected: bool = True
    relative_gain: bool = False
    excess_only: bool = False
    mode: CorrectionMode = "sliding"

    def validate(self) -> None:
        if self.mode not in ("sliding", "proximal"):
            raise ValueError(f"unknown correction mode {self.mode!r}")
        if self.mode == "proximal" and self.store_corrected:
            raise ValueError("proximal mode requires store_corrected=False; it uses the current error only")
        for name in ("lam", "k", "phi"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.switching not in ("sign", "sat"):
            raise ValueError(f"unknown switching function {self.switching!r}")
        if self.mode == "sliding" and self.switching == "sat" and self.k > 0 and self.phi <= 0:
            raise ValueError("'sat' switching needs a boundary layer phi > 0")

    @property
    def is_cfg(self) -> bool:
        """True when the law can never modify e (pure classifier-free guidance)."""
        return self.k == 0.0


@dataclass
class StepInfo:
    """Batch-averaged diagnostics for one call of `correct`."""
    step: int
    e_rms: float            # rms of the measured error
    s_rms: float            # pre-update sliding surface; current error in proximal mode
    delta_rms: float        # rms of the correction actually applied
    chatter: float          # fraction of elements whose sign(s) flipped since last step
    switch_activity: float  # rms(delta_t - delta_{t-1}): 2k per element for full chatter
    deriv_matters: float    # sign disagreement with memory; 0 (not applicable) in proximal mode
    k_eff: float            # mean effective gain


def rms(x: torch.Tensor) -> torch.Tensor:
    """Root mean square over every dim but the batch, keepdim so it broadcasts."""
    if x.ndim < 1 or x.numel() == 0:
        raise ValueError("x must have a nonempty batch dimension and samples")
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    if x.ndim == 1:
        return x.abs()  # Each batch entry is one scalar, not a shared norm.
    dims = tuple(range(1, x.dim()))
    return x.pow(2).mean(dim=dims, keepdim=True).sqrt()


def soft_threshold(e: torch.Tensor, k: float) -> torch.Tensor:
    """sign(e) * max(|e| - k, 0): the L1 proximal operator."""
    return torch.sign(e) * torch.clamp(e.abs() - k, min=0.0)


class SlidingModeGuidance:
    """Guidance with per-step diagnostics; proximal mode has no control memory.

    Call `correct(e_t)` once per denoising step. In proximal mode s=e is a
    diagnostic reference, not a sliding surface; lam/phi/switching are unused.
    """

    def __init__(self, cfg: Optional[SMCConfig] = None, **overrides):
        cfg = cfg if cfg is not None else SMCConfig()
        if overrides:
            cfg = replace(cfg, **overrides)
        cfg.validate()
        self.cfg = replace(cfg)
        self.history: List[StepInfo] = []
        self.reset()

    def reset(self) -> None:
        self._e_prev: Optional[torch.Tensor] = None
        self._sign_prev: Optional[torch.Tensor] = None
        self._delta_prev: Optional[torch.Tensor] = None
        self._step = 0
        self.history = []

    @property
    def is_cfg(self) -> bool:
        return self.cfg.is_cfg

    def guided_velocity(self, v_uncond: torch.Tensor, v_cond: torch.Tensor,
                        w: float) -> torch.Tensor:
        """v_uncond + w * (e + delta_e), the paper's Algorithm 1 line 13."""
        return v_uncond + w * self.correct(v_cond - v_uncond, w=w)

    def correct(self, e: torch.Tensor, w: Optional[float] = None) -> torch.Tensor:
        """Return the applied error e + delta_e for this step.

        Args:
            e: measured semantic error, shape [B, ...]. Batch is dim 0.
            w: the guidance scale the caller will multiply the result by.
               Required when `excess_only` is on; ignored otherwise.
        """
        cfg = self.cfg
        if e.dim() < 1 or e.numel() == 0:
            raise ValueError("e must have a nonempty batch dimension and samples")
        if not e.is_floating_point():
            raise ValueError("e must be a real floating-point tensor")
        if w is not None and not math.isfinite(w):
            raise ValueError("w must be finite")
        if cfg.excess_only and (w is None or w < 1.0):
            raise ValueError("excess_only needs the guidance scale w >= 1")

        # Accumulate half-precision surfaces and norms in float32. Keep the
        # original float32/float64 arithmetic for the paper reference tests.
        e_meas = e.detach()
        if e_meas.dtype in (torch.float16, torch.bfloat16):
            e_meas = e_meas.float()
        if self._e_prev is not None and (
            self._e_prev.shape != e_meas.shape
            or self._e_prev.device != e_meas.device
            or self._e_prev.dtype != e_meas.dtype
        ):
            raise ValueError("error shape, device or precision changed; call reset() first")
        if self._e_prev is None:
            self._e_prev = e_meas.clone()              # paper: s_0 = lam * e_0

        # ---- sliding variable  s = (e - e_prev) + lam * e_prev
        s = (e_meas if cfg.mode == "proximal" else
             (e_meas - self._e_prev) + cfg.lam * self._e_prev)

        # ---- gain and correction
        k_eff: Union[float, torch.Tensor] = cfg.k
        phi_eff: Union[float, torch.Tensor] = cfg.phi
        if cfg.relative_gain:
            scale = rms(e_meas)
            k_eff = k_eff * scale
            phi_eff = phi_eff * scale

        # Exact no-op branches also avoid 0 * inf and permit phi=0 at k=0.
        inactive = cfg.is_cfg or (cfg.excess_only and w == 1.0)
        if inactive:
            delta = torch.zeros_like(e_meas)
        elif cfg.mode == "proximal":
            # T_k(e)-e = -sign(e)*min(|e|, k). Writing the correction this way
            # avoids cancellation when k is small compared with a large e.
            # Its amplitude never exceeds the current component's magnitude,
            # so the applied error cannot reverse or grow. At zero it vanishes
            # immediately, irrespective of previous measurements/corrections.
            amount = (torch.minimum(e_meas.abs(), k_eff) if torch.is_tensor(k_eff)
                      else e_meas.abs().clamp(max=k_eff))
            delta = -torch.sign(e_meas) * amount
            if cfg.excess_only:
                delta = delta * ((w - 1.0) / w)
        else:
            if cfg.switching == "sign":
                sw = torch.sign(s)
            else:
                # A zero relative scale implies k_eff=0. A harmless denominator
                # keeps that sample finite even when its memory is nonzero.
                if torch.is_tensor(phi_eff):
                    phi_eff = torch.where(phi_eff > 0, phi_eff, torch.ones_like(phi_eff))
                sw = torch.clamp(s / phi_eff, -1.0, 1.0)
            delta = -k_eff * sw
            if cfg.excess_only:
                delta = delta * ((w - 1.0) / w)        # shrink only the extrapolation

        # Preserve the caller's dtype and the identity gradient through e.
        e_app = e if inactive else (e.to(e_meas.dtype) + delta).to(e.dtype)

        # ---- diagnostics
        sgn = torch.sign(s)
        delta_d = e_app.detach().to(e_meas.dtype) - e_meas
        self.history.append(StepInfo(
            step=self._step,
            e_rms=float(rms(e_meas).mean()),
            s_rms=float(rms(s).mean()),
            delta_rms=float(rms(delta_d).mean()),
            chatter=(float((sgn * self._sign_prev < 0).float().mean())
                     if self._sign_prev is not None else 0.0),
            switch_activity=(float(rms(delta_d - self._delta_prev).mean())
                             if self._delta_prev is not None else 0.0),
            deriv_matters=(0.0 if cfg.mode == "proximal" else
                           float((sgn * torch.sign(self._e_prev) < 0).float().mean())),
            k_eff=float(k_eff.mean()) if torch.is_tensor(k_eff) else float(k_eff),
        ))

        # ---- state
        self._sign_prev = sgn
        self._delta_prev = delta_d.clone()
        self._e_prev = (e_app.detach().to(e_meas.dtype) if cfg.store_corrected else e_meas).clone()
        self._step += 1
        return e_app


# ---------------------------------------------------------------- presets
def cfg_baseline() -> SMCConfig:
    """Plain classifier-free guidance: P-control with gain w, no correction."""
    return SMCConfig(k=0.0)


def paper(lam: float = 6.0, k: float = 0.1) -> SMCConfig:
    """SMC-CFG exactly as in Algorithm 1 and the authors' code."""
    return SMCConfig(lam=lam, k=k)


def boundary_layer(lam: float = 6.0, k: float = 0.1,
                   phi: Optional[float] = None) -> SMCConfig:
    """sat(s/phi) instead of sign(s), storing the measured error. The default
    phi = k*lam gives soft-thresholding at the first step or for constant e.
    At later steps the previous measurement also affects the correction."""
    return SMCConfig(lam=lam, k=k, switching="sat", store_corrected=False,
                     phi=(k * lam if phi is None else phi))


def boundary_layer_excess(lam: float = 6.0, k: float = 0.1,
                          phi: Optional[float] = None) -> SMCConfig:
    """Candidate law: boundary layer applied to the extrapolation only.
    Exactly CFG at w = 1. Pass w to correct()."""
    return SMCConfig(lam=lam, k=k, switching="sat", store_corrected=False,
                     excess_only=True, phi=(k * lam if phi is None else phi))


def proximal_excess(k: float = 0.1) -> SMCConfig:
    """Memoryless soft-thresholding of CFG extrapolation.

    v_hat = v_cond + (w-1)*sign(e)*max(|e|-k, 0).
    The correction is zero at e=0 and preserves the conditional endpoint at
    w=1. It attenuates components without reversing them; image-quality gains
    still require a matched-alignment comparison. No sliding surface is used.
    """
    return SMCConfig(k=k, mode="proximal", store_corrected=False, excess_only=True)


def proximal_relative_excess(k: float = 0.1) -> SMCConfig:
    """Proximal extrapolation with threshold k*rms(e), independently per sample.

    Here k is dimensionless. This rescales consistently under constant changes
    of velocity units, but does not establish improved cross-model quality.
    """
    return replace(proximal_excess(k), relative_gain=True)
