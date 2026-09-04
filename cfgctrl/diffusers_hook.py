"""Attach a guidance controller to a diffusers pipeline without editing it.

The seam. Every standard diffusers text-to-image pipeline that does
classifier-free guidance runs the denoiser once on a doubled batch,

    model_input   = cat([latents, latents])
    prompt_embeds = cat([negative_prompt_embeds, prompt_embeds])   # uncond first
    out           = model(model_input, timestep, ...)
    uncond, cond  = out.chunk(2)
    noise_pred    = uncond + guidance_scale * (cond - uncond)        # P-control

The combination line lives inside the pipeline and differs per model, so
instead of patching pipelines we wrap the denoiser's `forward` and return

    cat([uncond, uncond + e_applied]),   e_applied = controller.correct(cond - uncond)

so that the pipeline's own combination yields  uncond + w * e_applied, which is
exactly the paper's Algorithm 1 line 13. One wrapper covers SD1.5/SDXL
(UNet2DConditionModel, output `.sample`) and SD3/SD3.5 (SD3Transformer2DModel,
output `.sample` or a tuple) because it only touches the returned tensor.

!! STATUS: NEVER EXECUTED ON A REAL MODEL. The algebra is unit-tested against a
!! dummy denoiser, but the batch-order convention ([uncond, cond]) and the
!! output container handling are assumptions. Verify both on the first GPU run:
!!   1. with `presets.cfg_baseline()` (k = 0) the hook short-circuits, so the
!!      image must be BIT-IDENTICAL to running without the hook at one seed;
!!   2. check which half of the chunk responds to a strongly negative prompt --
!!      getting this backwards silently inverts the guidance direction.

Flux-dev needs more than this hook: with `true_cfg_scale > 1` its pipeline runs
two separate forward passes rather than one doubled batch, so wrap at the
pipeline level there (the CFG-Ctrl authors subclass the pipeline for all models).
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .controllers import SlidingModeGuidance


class GuidanceHook:
    def __init__(self, model: torch.nn.Module, controller: SlidingModeGuidance,
                 uncond_first: bool = True, guidance_scale: Optional[float] = None):
        """
        Args:
            model:      pipe.unet or pipe.transformer.
            controller: a SlidingModeGuidance. Call `hook.reset()` per image.
            uncond_first: diffusers convention cat([negative, positive]).
            guidance_scale: the w the pipeline will apply. Only needed for
                        `excess_only` controllers; pass the same value you give
                        the pipeline.
        """
        self.model = model
        self.controller = controller
        self.uncond_first = uncond_first
        self.guidance_scale = guidance_scale
        self._orig_forward = None
        self._had_instance_forward = False

    # ------------------------------------------------------------ lifecycle
    def attach(self) -> "GuidanceHook":
        if self._orig_forward is None:
            self._had_instance_forward = "forward" in self.model.__dict__
            self._orig_forward = self.model.forward
            self.model.forward = self._forward  # type: ignore[method-assign]
        return self

    def detach(self) -> None:
        if self._orig_forward is None:
            return
        if self._had_instance_forward:
            self.model.forward = self._orig_forward  # type: ignore[method-assign]
        else:
            del self.model.forward                   # fall back to the class method
        self._orig_forward = None

    def reset(self) -> None:
        self.controller.reset()

    def __enter__(self) -> "GuidanceHook":
        return self.attach()

    def __exit__(self, *exc) -> None:
        self.detach()

    # ------------------------------------------------------------- forward
    @staticmethod
    def _extract(out: Any) -> torch.Tensor:
        if torch.is_tensor(out):
            return out
        if isinstance(out, (tuple, list)):
            return out[0]
        if hasattr(out, "sample"):
            return out.sample
        raise TypeError(f"cannot find the prediction tensor in {type(out)!r}")

    @staticmethod
    def _rebuild(out: Any, new: torch.Tensor) -> Any:
        if torch.is_tensor(out):
            return new
        if isinstance(out, tuple):
            return (new,) + tuple(out[1:])
        if isinstance(out, list):
            return [new] + list(out[1:])
        try:
            out.sample = new
            return out
        except Exception:                            # frozen dataclass
            return type(out)(sample=new)

    def _forward(self, *args, **kwargs):
        out = self._orig_forward(*args, **kwargs)
        if self.controller.is_cfg:
            return out                               # baseline arm: untouched
        pred = self._extract(out)
        if pred.shape[0] % 2 != 0:
            return out                               # no doubled batch: not CFG
        a, b = pred.chunk(2)
        uncond, cond = (a, b) if self.uncond_first else (b, a)
        e_app = self.controller.correct((cond - uncond).float(), w=self.guidance_scale)
        cond_new = uncond + e_app.to(pred.dtype)
        new = torch.cat([uncond, cond_new] if self.uncond_first else [cond_new, uncond], 0)
        return self._rebuild(out, new)


def attach_smc_cfg(pipe, controller: SlidingModeGuidance, **kwargs) -> GuidanceHook:
    """Convenience: picks pipe.transformer or pipe.unet and attaches the hook."""
    model = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
    if model is None:
        raise AttributeError("pipeline has neither .transformer nor .unet")
    return GuidanceHook(model, controller, **kwargs).attach()
