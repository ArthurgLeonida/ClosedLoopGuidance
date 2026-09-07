"""Explicit CFG at the scheduler boundary for the three supported Diffusers pipelines.

Diffusers 0.35.2: SD3 has one [unconditional, conditional] batch; FLUX and
Qwen evaluate conditional then unconditional. Qwen's native norm rescaling is
replaced by the same raw CFG combination for every arm, as in the paper's
released SMC combiner. No intermediate conditional-output rounding is needed.
"""
import torch

from cfgctrl import SlidingModeGuidance

TESTED_DIFFUSERS = "0.35.2"


def prediction_tensor(output):
    prediction = output if torch.is_tensor(output) else output[0] if isinstance(output, (tuple, list)) and output else getattr(output, "sample", None)
    if not torch.is_tensor(prediction) or prediction.ndim == 0:
        raise TypeError("transformer output has no batched prediction tensor")
    return prediction


class SchedulerGuidance:
    def __init__(self, pipe, family, controller, w, steps):
        self.pipe, self.family, self.controller = pipe, family, controller
        self.w, self.expected_steps = w, steps
        self.pending, self.signals = [], []
        self.count = 0

    def __enter__(self):
        model, scheduler = self.pipe.transformer, self.pipe.scheduler
        if hasattr(model, "_cfgctrl_owner") or hasattr(scheduler, "_cfgctrl_owner"):
            raise RuntimeError("a guidance adapter is already attached")
        model._cfgctrl_owner = scheduler._cfgctrl_owner = self
        self.forward, self.step = model.forward, scheduler.step
        self.had_forward, self.had_step = "forward" in model.__dict__, "step" in scheduler.__dict__
        model.forward, scheduler.step = self._forward, self._step
        if self.controller:
            self.controller.reset()
        return self

    def __exit__(self, exc_type, exc, tb):
        model, scheduler = self.pipe.transformer, self.pipe.scheduler
        if self.had_forward:
            model.forward = self.forward
        else:
            del model.forward
        if self.had_step:
            scheduler.step = self.step
        else:
            del scheduler.step
        del model._cfgctrl_owner
        del scheduler._cfgctrl_owner
        if exc_type is None and (self.pending or self.count != self.expected_steps):
            raise RuntimeError(f"adapter saw {self.count} scheduler steps, expected {self.expected_steps}")

    def _forward(self, *args, **kwargs):
        result = self.forward(*args, **kwargs)
        prediction = prediction_tensor(result)
        t = kwargs.get("timestep")
        if t is None:
            raise RuntimeError("unsupported transformer calling convention: timestep is not named")
        self.pending.append((prediction.detach(), t.detach().clone()))
        if len(self.pending) > (1 if self.family == "sd35" or self.w == 1 else 2):
            raise RuntimeError("unexpected extra denoiser call before scheduler step")
        return result

    def _step(self, model_output, timestep, sample, *args, **kwargs):
        expected = 1 if self.family == "sd35" or self.w == 1 else 2
        if len(self.pending) != expected:
            raise RuntimeError("missing conditional/unconditional prediction; refusing a no-op comparison")
        predictions = [p for p, _ in self.pending]
        t = float(timestep)
        factor = 1.0 if self.family == "sd35" else 1000.0
        for _, recorded in self.pending:
            # bf16 time labels can round; all elements must represent this solver time.
            if not torch.allclose(recorded.float() * factor, torch.full_like(recorded.float(), t),
                                  rtol=0.01, atol=0.01):
                raise RuntimeError("denoiser and scheduler timesteps do not match")
        if self.w == 1:
            if self.controller is not None:
                raise ValueError("the conditional endpoint does not evaluate a correction arm")
            guided = predictions[0]
            correction_rms = 0.0
        else:
            if self.family == "sd35":
                if predictions[0].shape[0] != 2 * sample.shape[0]:
                    raise RuntimeError("SD3 expected an unconditional-first doubled batch")
                uncond, cond = predictions[0].chunk(2)
            else:
                cond, uncond = predictions
                if cond.shape != sample.shape or uncond.shape != cond.shape:
                    raise RuntimeError("separate CFG predictions have incompatible shapes")
                if not torch.equal(self.pending[0][1], self.pending[1][1]):
                    raise RuntimeError("CFG branches were evaluated at different times")
            dtype = torch.float64 if cond.dtype == torch.float64 else torch.float32
            uncond, cond = uncond.to(dtype), cond.to(dtype)
            baseline = uncond + self.w * (cond - uncond)
            guided = self.controller.guided_velocity(uncond, cond, self.w)
            correction_rms = float((guided.to(model_output.dtype).to(dtype) -
                                    baseline.to(model_output.dtype).to(dtype)).square().mean().sqrt())
        guided = guided.to(model_output.dtype)
        if not torch.isfinite(guided).all():
            raise RuntimeError("nonfinite guided velocity")
        step_index = self.count
        sigmas = getattr(self.pipe.scheduler, "sigmas", None)
        h = abs(float(sigmas[step_index + 1] - sigmas[step_index])) if sigmas is not None else None
        self.signals.append(dict(step=step_index, timestep=t, sigma=float(sigmas[step_index]) if sigmas is not None else None,
                                 velocity_correction_rms=correction_rms,
                                 euler_displacement_correction_rms=h * correction_rms if h is not None else None))
        self.pending.clear()
        self.count += 1
        return self.step(guided, timestep, sample, *args, **kwargs)


def load_pipeline(family, settings, device):
    import diffusers
    if diffusers.__version__ != TESTED_DIFFUSERS:
        raise RuntimeError(f"model adapters require diffusers=={TESTED_DIFFUSERS}; "
                           f"found {diffusers.__version__}. Pin it in the generation environment.")
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; install a matching PyTorch GPU build before model download")
        if settings["dtype"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("this GPU does not support bf16; select fp16 or fp32")
    cls = {"sd35": diffusers.StableDiffusion3Pipeline,
           "flux": diffusers.FluxPipeline, "qwen": diffusers.QwenImagePipeline}[family]
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[settings["dtype"]]
    pipe = cls.from_pretrained(settings["checkpoint"], revision=settings["revision"], torch_dtype=dtype)
    if not device.startswith("cuda") or settings["offload"] == "none":
        pipe.to(device)
    elif settings["offload"] == "model":
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.enable_sequential_cpu_offload(device=device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_image(pipe, family, settings, controller, prompt, seed, w):
    kwargs = dict(prompt=prompt, negative_prompt="", width=settings["width"], height=settings["height"],
                  num_inference_steps=settings["steps"],
                  generator=torch.Generator(device="cpu").manual_seed(seed))
    if family == "sd35":
        kwargs["guidance_scale"] = w
    else:
        kwargs["true_cfg_scale"] = w
        if family == "flux":
            kwargs["guidance_scale"] = settings["embedded_guidance"]
    with SchedulerGuidance(pipe, family, controller, w, settings["steps"]) as adapter, torch.inference_mode():
        image = pipe(**kwargs).images[0]
    return image, adapter.signals
