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

!! STATUS: integration verified once, on SD3.5-large in bf16 at 1024x1024
!! (latent 16x128x128), 8 steps, w = 7, via `experiments/real_model.py verify`:
!! zero-gain transparency, unconditional-first batch order, and one controller
!! call per solver step all passed. That covers those settings only. The
!! algebra is unit-tested against a dummy denoiser; re-run verify for any other
!! checkpoint, pipeline, scheduler, resolution or dtype, because it checks:
!!   1. with `presets.cfg_baseline()` (k = 0) the hook short-circuits, so the
!!      image must be BIT-IDENTICAL to running without the hook at one seed;
!!   2. check which half of the chunk responds to a strongly negative prompt --
!!      getting this backwards silently inverts the guidance direction.

Flux-dev needs more than this hook: with `true_cfg_scale > 1` its pipeline runs
two separate forward passes rather than one doubled batch, so wrap at the
pipeline level there (the CFG-Ctrl authors subclass the pipeline for all models).
"""

from __future__ import annotations

from dataclasses import is_dataclass, replace
from inspect import getattr_static
from typing import Any, Callable, Optional, Union

import torch

from .controllers import SlidingModeGuidance


class GuidanceHook:
    def __init__(self, model: torch.nn.Module, controller: SlidingModeGuidance,
                 uncond_first: bool = True,
                 guidance_scale: Optional[Union[float, Callable[[], float]]] = None,
                 cfg_enabled: Optional[Callable[[], bool]] = None):
        """
        Args:
            model:      pipe.unet or pipe.transformer.
            controller: a SlidingModeGuidance. Call `hook.reset()` per image.
            uncond_first: diffusers convention cat([negative, positive]).
            guidance_scale: the w the pipeline will apply. Only needed for
                        `excess_only` controllers; pass the same value you give
                        the pipeline, or a callable returning its current value.
            cfg_enabled: callable reporting whether this forward uses doubled
                        CFG. Without one, the caller guarantees that every
                        forward is a doubled CFG batch. An even batch size
                        alone cannot distinguish CFG from ordinary batching.
                        Prefer `attach_smc_cfg(pipe, ...)` for pipelines.
        """
        self.model = model
        self.controller = controller
        self.uncond_first = uncond_first
        self.guidance_scale = guidance_scale
        self.cfg_enabled = cfg_enabled
        self._orig_forward = None
        self._had_instance_forward = False

    # ------------------------------------------------------------ lifecycle
    def attach(self) -> "GuidanceHook":
        if self._orig_forward is None:
            if isinstance(getattr(self.model.forward, "__self__", None), GuidanceHook):
                raise RuntimeError("a GuidanceHook is already attached to this model")
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
            pred = out
        elif isinstance(out, (tuple, list)) and out:
            pred = out[0]
        else:
            pred = getattr(out, "sample", None)
        if not torch.is_tensor(pred) or pred.ndim == 0:
            raise TypeError(f"cannot find a batched prediction tensor in {type(out)!r}")
        return pred

    @staticmethod
    def _rebuild(out: Any, new: torch.Tensor) -> Any:
        if torch.is_tensor(out):
            return new
        if isinstance(out, tuple):
            if hasattr(out, "_fields"):
                return out._replace(**{out._fields[0]: new})
            return (new,) + tuple(out[1:])
        if isinstance(out, list):
            return [new] + list(out[1:])
        if is_dataclass(out):
            return replace(out, sample=new)
        out.sample = new
        return out

    def _forward(self, *args, **kwargs):
        out = self._orig_forward(*args, **kwargs)
        if self.controller.is_cfg:
            return out                               # baseline arm: untouched
        if self.cfg_enabled is not None and not self.cfg_enabled():
            return out                               # includes even non-CFG batches
        pred = self._extract(out)
        if pred.shape[0] == 0 or pred.shape[0] % 2 != 0:
            raise ValueError("active GuidanceHook requires a nonempty doubled CFG batch")
        a, b = pred.chunk(2)
        uncond, cond = (a, b) if self.uncond_first else (b, a)
        # Promote BEFORE subtracting: two finite fp16 predictions can have an
        # overflowing fp16 difference. Apply only the correction to the original
        # conditional branch, retaining its precision when the correction is zero.
        dtype = torch.float32 if pred.dtype in (torch.float16, torch.bfloat16) else pred.dtype
        error = cond.to(dtype) - uncond.to(dtype)
        w = self.guidance_scale() if callable(self.guidance_scale) else self.guidance_scale
        e_app = self.controller.correct(error, w=w)
        cond_new = (cond.to(dtype) + (e_app - error)).to(pred.dtype)
        new = torch.cat([uncond, cond_new] if self.uncond_first else [cond_new, uncond], 0)
        return self._rebuild(out, new)


def attach_smc_cfg(pipe, controller: SlidingModeGuidance, **kwargs) -> GuidanceHook:
    """Attach to a standard doubled-batch SD/SDXL/SD3 pipeline.

    Read the pipeline's CFG flag and guidance scale at each forward, since
    pipelines initialize them inside `__call__`. Other batch conventions and
    auxiliary denoiser passes require a pipeline-specific adapter. Guidance
    rescaling and SD3 skip-layer guidance must be disabled. At w <= 1,
    these pipelines skip CFG and this hook leaves the conditional output alone;
    it therefore cannot evaluate the paper's correction at that endpoint.
    """
    model = getattr(pipe, "transformer", None)
    if model is None:
        model = getattr(pipe, "unet", None)
    if model is None:
        raise AttributeError("pipeline has neither .transformer nor .unet")
    if "cfg_enabled" not in kwargs:
        # Static lookup avoids evaluating an as-yet uninitialized property.
        if getattr_static(pipe, "do_classifier_free_guidance", None) is None:
            raise ValueError("pipeline does not expose doubled-batch CFG state; "
                             "use a pipeline-specific adapter")
        def cfg_enabled():
            if getattr(pipe, "guidance_rescale", 0.0) != 0.0:
                raise ValueError("guidance rescaling requires a pipeline-specific adapter")
            if getattr(pipe, "skip_guidance_layers", None) is not None:
                raise ValueError("skip-layer guidance requires a pipeline-specific adapter")
            return bool(pipe.do_classifier_free_guidance)

        kwargs["cfg_enabled"] = cfg_enabled
    if "guidance_scale" not in kwargs:
        kwargs["guidance_scale"] = lambda: pipe.guidance_scale
    return GuidanceHook(model, controller, **kwargs).attach()
