"""Attach a guidance controller to a diffusers pipeline without editing it.

The seam. Every standard diffusers text-to-image pipeline that does
classifier-free guidance runs the denoiser once on a doubled batch,

    model_input  = cat([latents, latents])
    prompt_embeds = cat([negative_prompt_embeds, prompt_embeds])   # uncond first
    out          = model(model_input, timestep, ...)
    uncond, cond = out.chunk(2)
    noise_pred   = uncond + guidance_scale * (cond - uncond)         # P-control

The combination line is inside the pipeline and differs per model, so
instead of patching pipelines we wrap the denoiser's `forward` and return

    cat([uncond, uncond + e_applied]),   e_applied = controller.correct(cond - uncond, dt)

so that the pipeline's own combination yields  uncond + w * e_applied, which is
exactly the paper's Algorithm 1 line 13. One wrapper works for SD1.5/SDXL
(UNet2DConditionModel, output .sample) and SD3/SD3.5 (SD3Transformer2DModel,
output .sample or a tuple) because it only touches the returned tensor.

!! STATUS: WRITTEN BUT NEVER EXECUTED ON A REAL MODEL. No GPU was available.
!! The batch-order convention ([uncond, cond]) and the output container
!! handling are TODO(verify) against the pipeline you actually run. The
!! algebra itself is unit-tested with a dummy denoiser in tests/.

Known cases that need more than this hook:
  * Flux-dev is guidance-distilled: with `true_cfg_scale > 1` the diffusers
    pipeline runs *two separate* forward passes (cond, then uncond) instead
    of one doubled batch. Wrap at the pipeline level there, or subclass the
    pipeline's __call__ (the authors do the latter for all models).
  * Modular diffusers has a `guiders` API (ClassifierFreeGuidance,
    AdaptiveProjectedGuidance, ...). Subclassing its base guider is the
    native way to add SMC-CFG once you are on modular pipelines.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .controllers import SlidingModeGuidance


class GuidanceHook:
    def __init__(
        self,
        model: torch.nn.Module,
        controller: SlidingModeGuidance,
        num_train_timesteps: float = 1000.0,
        default_dt: Optional[float] = None,
        uncond_first: bool = True,
        guidance_scale: Optional[float] = None,
    ):
        """
        Args:
            model:      pipe.unet or pipe.transformer.
            controller: a SlidingModeGuidance. Reset it per image (call
                        `hook.reset()` before each pipe(...) call).
            num_train_timesteps: flow-matching pipelines pass timestep =
                        sigma * 1000; dt is |t_prev - t_now| / this value.
            default_dt: used at the first step when time scaling needs a dt
                        and no previous timestep is known (e.g. 1 / num_steps).
            uncond_first: diffusers convention cat([negative, positive]).
                        TODO(verify) for the pipeline you run.
            guidance_scale: the w the pipeline will apply; only needed for
                        `excess_only` controllers (pass the same value you
                        pass to the pipeline).
        """
        self.model = model
        self.controller = controller
        self.num_train_timesteps = float(num_train_timesteps)
        self.default_dt = default_dt
        self.uncond_first = uncond_first
        self.guidance_scale = guidance_scale
        self._orig_forward = None
        self._had_instance_forward = False
        self._t_prev: Optional[float] = None

    # ------------------------------------------------------------ lifecycle
    def attach(self) -> "GuidanceHook":
        if self._orig_forward is not None:
            return self
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
        self._t_prev = None

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
        except Exception:  # frozen dataclass
            return type(out)(sample=new)

    def _dt_from(self, timestep: Any) -> Optional[float]:
        if timestep is None:
            return self.default_dt
        t = float(torch.as_tensor(timestep).flatten()[0]) / self.num_train_timesteps
        dt = abs(self._t_prev - t) if self._t_prev is not None else self.default_dt
        self._t_prev = t
        return dt

    def _forward(self, *args, **kwargs):
        out = self._orig_forward(*args, **kwargs)
        if self.controller.is_cfg:
            return out                                # baseline arm: bit-identical to the pipeline
        pred = self._extract(out)
        if pred.shape[0] % 2 != 0:
            return out                                # no doubled batch: not CFG
        timestep = kwargs.get("timestep", args[1] if len(args) > 1 else None)
        dt = self._dt_from(timestep)
        a, b = pred.chunk(2)
        uncond, cond = (a, b) if self.uncond_first else (b, a)
        e_app = self.controller.correct((cond - uncond).float(), dt, w=self.guidance_scale).to(pred.dtype)
        cond_new = uncond + e_app
        new = torch.cat([uncond, cond_new] if self.uncond_first else [cond_new, uncond], 0)
        return self._rebuild(out, new)


def attach_smc_cfg(pipe, controller: SlidingModeGuidance, **kwargs) -> GuidanceHook:
    """Convenience: picks pipe.transformer or pipe.unet and attaches the hook."""
    model = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
    if model is None:
        raise AttributeError("pipeline has neither .transformer nor .unet")
    return GuidanceHook(model, controller, **kwargs).attach()
