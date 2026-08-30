import math
from typing import Optional

import torch


def compute_tag_eta(eta_tag: float, t_normalized: float, adaptive_tag: bool) -> float:
    """Return the TAG eta for the current timestep."""
    if not adaptive_tag:
        return eta_tag
    return 1.0 + (eta_tag - 1.0) * t_normalized ** (1 / math.e)


def compute_ca_mask_weight(
    t_normalized: float,
    min_step_ratio: float,
    max_step_ratio: float,
    schedule_power: float,
) -> float:
    """Return the Cross-Attention mask weight for the current timestep.

    w_CA(t) = progress^(schedule_power · e): near zero at the start of the
    edit (high noise), rising to 1 by the end — CA sharpens localization only
    after the raw edit has had room to form structure.
    """
    min_t = min(max(float(min_step_ratio), 0.0), 1.0)
    max_t = min(max(float(max_step_ratio), min_t + 1e-8), 1.0)
    t = min(max(float(t_normalized), min_t), max_t)
    progress = (max_t - t) / max(max_t - min_t, 1e-8)
    exponent = max(float(schedule_power), 1e-8) * math.e
    return progress ** exponent


def apply_tag(noise_pred: torch.Tensor, latents_noisy: torch.Tensor, eta: float) -> torch.Tensor:
    """Apply tangential amplification around the noisy latent direction."""
    if eta == 1.0:
        return noise_pred

    v = latents_noisy / (latents_noisy.norm(p=2, dim=(1, 2, 3), keepdim=True) + 1e-8)
    noise_parallel = (noise_pred * v).sum(dim=(1, 2, 3), keepdim=True) * v
    noise_tangential = noise_pred - noise_parallel
    return noise_parallel + eta * noise_tangential


def apply_source_blend(
    eps_tgt: torch.Tensor,
    eps_src: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Localize DDS by falling back to the source branch outside the mask.

    Hard gate ``eps_src + M·(eps_tgt − eps_src)``: inside the mask the target
    prediction wins, outside it the source prediction wins, so the DDS delta
    is exactly zero where M = 0.
    """
    return eps_src + mask * (eps_tgt - eps_src)


def compute_stg_scale(
    *,
    base_scale: float,
    iteration: int,
    max_iteration: int,
    start_ratio: float,
    end_ratio: float,
    bump_peak_ratio: float,
    current_edit_strength: Optional[float],
) -> float:
    """Return the bump-scheduled STG scale, attenuated by current edit strength.

    The scale follows a triangular bump over training progress: zero outside
    [start_ratio, end_ratio], rising linearly to `base_scale` at the peak and
    falling back to zero before the end (the final stretch of training runs
    without STG so views can converge coherently).

    `current_edit_strength` is the ratio `||eps_tgt − eps_src|| / (||eps_tgt|| + ||eps_src||)`
    in [0, 1] — high for creative/structural edits, low for identity-preserving
    edits. The scale is multiplied by `(1 − edit_strength)` so bold edits
    attenuate STG more.
    """
    max_iteration = max(int(max_iteration), 1)
    progress = min(max(iteration / max_iteration, 0.0), 1.0)
    start_ratio = float(start_ratio)
    end_ratio = float(end_ratio)

    if progress <= start_ratio or progress >= end_ratio or end_ratio <= start_ratio:
        scale = 0.0
    else:
        peak_ratio = min(max(float(bump_peak_ratio), 0.0), 1.0)
        peak_time = start_ratio + peak_ratio * (end_ratio - start_ratio)
        if progress <= peak_time:
            span = max(peak_time - start_ratio, 1e-8)
            scale = float(base_scale) * (progress - start_ratio) / span
        else:
            span = max(end_ratio - peak_time, 1e-8)
            scale = float(base_scale) * (end_ratio - progress) / span

    if current_edit_strength is not None:
        s = min(max(float(current_edit_strength), 0.0), 1.0)
        scale = scale * (1.0 - s)

    return scale


def compute_edit_strength(eps_tgt: torch.Tensor, eps_src: torch.Tensor) -> float:
    """Return edit strength ∈ [0, 1] from the raw target/source noise predictions.

        s = ||eps_tgt − eps_src|| / (||eps_tgt|| + ||eps_src||)

    Bounded in [0, 1] by the triangle inequality. Time-invariant: numerator
    and denominator scale together with the noise-prediction magnitude at the
    current timestep, so the ratio cancels timestep effects.

    Physically: "how much the target's noise prediction diverges from the
    source's, relative to their combined magnitude." Low for identity-preserving
    edits (e.g. face / elf — target mostly agrees with source); high for
    structural edits (e.g. person → stormtrooper — target pulls the latent to a
    different rendering).

    Motivation over mask-based coverage: both the self-mask and the
    cross-attention mask are percentile-normalized in `normalize_relevance_map`,
    which forces their magnitudes into a narrow band across scenes. Neither raw
    mean nor the Herfindahl-inverse shape statistic discriminates scene types
    robustly in the direction the adaptation needs. `eps_tgt − eps_src` is the
    raw DDS delta before any mask is built — the clean scene-level signal.

    Used by the adaptive STG scale and outside-mask anchor weight as
    `multiplier = (1 − s)`: identity edits keep STG/anchor near full strength;
    structural edits attenuate both.
    """
    delta = (eps_tgt - eps_src).detach().float()
    tgt = eps_tgt.detach().float()
    src = eps_src.detach().float()
    delta_norm = delta.flatten(1).norm(dim=1)
    denom = (tgt.flatten(1).norm(dim=1) + src.flatten(1).norm(dim=1)).clamp_min(1e-8)
    return float((delta_norm / denom).mean().item())


def compute_preserve_weight(
    *,
    psi: float,
    grad_mask: Optional[torch.Tensor],
    outside_mask_anchor_weight: float,
    outside_mask_anchor_edit_strength_adaptive: bool,
    edit_strength: Optional[float] = None,
):
    """Compute DreamCatalyst preservation weight plus optional outside-mask anchor.

    `edit_strength` ∈ [0, 1] is the precomputed scalar from `compute_edit_strength`.
    When the adaptive flag is on, the anchor weight is attenuated linearly by
    `(1 − edit_strength)`: structural edits (high s) loosen the background
    anchor so it does not block the edit; identity edits keep it at full
    strength — no per-scene tuning.
    """
    preserve_weight = psi

    if grad_mask is not None and outside_mask_anchor_weight > 0:
        w_out_effective = outside_mask_anchor_weight
        if outside_mask_anchor_edit_strength_adaptive and edit_strength is not None:
            s = min(max(float(edit_strength), 0.0), 1.0)
            w_out_effective = w_out_effective * (1.0 - s)
        preserve_weight = preserve_weight + w_out_effective * (1.0 - grad_mask)

    return preserve_weight


def apply_latent_mean_anchor(
    grad: torch.Tensor,
    tgt_x0: torch.Tensor,
    src_x0: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """Pull per-channel spatial means of edited latent toward source latent."""
    if weight <= 0:
        return grad
    tgt_mean = tgt_x0.mean(dim=(2, 3), keepdim=True)
    src_mean = src_x0.mean(dim=(2, 3), keepdim=True)
    return grad + weight * (tgt_mean - src_mean).expand_as(grad)


def compute_gate_signal(
    target_ca: Optional[torch.Tensor],
    sm: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Semantic gate for the cache fusion: pixel-wise max of the CA and self masks."""
    if sm is not None and target_ca is not None:
        return torch.maximum(target_ca, sm).clamp(0.0, 1.0)
    return target_ca if target_ca is not None else sm
