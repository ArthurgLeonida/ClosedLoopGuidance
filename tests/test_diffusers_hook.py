"""Offline regression coverage for real-pipeline integration boundaries."""

from collections import namedtuple
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from cfgctrl import SlidingModeGuidance, presets
from cfgctrl.diffusers_hook import GuidanceHook, attach_smc_cfg


@dataclass(frozen=True)
class FrozenOutput:
    sample: torch.Tensor
    extra: str


NamedOutput = namedtuple("NamedOutput", "sample extra")


@pytest.mark.parametrize("container", [
    lambda x: x,
    lambda x: (x, "extra"),
    lambda x: [x, "extra"],
    lambda x: NamedOutput(x, "extra"),
    lambda x: FrozenOutput(x, "extra"),
])
@pytest.mark.parametrize("uncond_first", [True, False])
def test_active_hook_matches_independent_multistep_law_and_preserves_container(container, uncond_first):
    class Denoiser(torch.nn.Module):
        def forward(self, prediction):
            return container(prediction)

    model = Denoiser()
    ctrl = SlidingModeGuidance(presets.paper())
    reference = SlidingModeGuidance(presets.paper())
    gen = torch.Generator().manual_seed(18)
    with GuidanceHook(model, ctrl, uncond_first=uncond_first):
        for _ in range(3):
            uncond = torch.randn((2, 4), generator=gen)
            cond = torch.randn((2, 4), generator=gen)
            prediction = torch.cat([uncond, cond] if uncond_first else [cond, uncond])
            expected = uncond + 3.0 * reference.correct(cond - uncond)
            out = model(prediction)
            assert type(out) is type(container(prediction))
            if not torch.is_tensor(out):
                assert (out[1] if isinstance(out, (list, tuple)) else out.extra) == "extra"
            a, b = GuidanceHook._extract(out).chunk(2)
            actual_uncond, actual_cond = (a, b) if uncond_first else (b, a)
            assert torch.equal(actual_uncond, uncond)
            assert torch.allclose(actual_uncond + 3.0 * (actual_cond - actual_uncond),
                                  expected, atol=2e-6)
    assert "forward" not in model.__dict__
    assert len(ctrl.history) == 3


def test_fp16_difference_is_promoted_before_subtraction_and_reconstruction():
    model = torch.nn.Identity()
    prediction = torch.tensor([[-60000.0], [60000.0]], dtype=torch.float16)
    ctrl = SlidingModeGuidance(presets.paper(k=32.0))
    with GuidanceHook(model, ctrl):
        actual = model(prediction)
    assert actual.dtype == torch.float16
    assert torch.isfinite(actual).all()
    assert actual[1, 0].item() == 59968.0
    assert ctrl.history[0].e_rms == pytest.approx(120000.0)


def test_zero_excess_correction_keeps_small_conditional_branch_exact():
    model = torch.nn.Identity()
    prediction = torch.tensor([[60000.0], [0.001]], dtype=torch.float16)
    with GuidanceHook(model, SlidingModeGuidance(presets.boundary_layer_excess()),
                      guidance_scale=1.0):
        assert torch.equal(model(prediction), prediction)


def test_hook_retains_float64_precision():
    model = torch.nn.Identity()
    prediction = torch.tensor([[0.123456789012], [0.987654321098]], dtype=torch.float64)
    with GuidanceHook(model, SlidingModeGuidance(presets.paper(k=1e-10))):
        result = model(prediction)
    expected = prediction.clone()
    expected[1] -= 1e-10
    assert torch.allclose(result, expected, atol=1e-15, rtol=0)


class Pipeline:
    def __init__(self):
        self.transformer = torch.nn.Identity()
        self.disabled = False

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and not self.disabled

    @property
    def guidance_scale(self):
        return self._guidance_scale

    def __call__(self, prediction, w):
        self._guidance_scale = w
        return self.transformer(prediction)


def test_pipeline_hook_skips_even_non_cfg_batches_and_reads_scale_each_call():
    pipe = Pipeline()  # guidance properties are uninitialized until __call__
    ctrl = SlidingModeGuidance(presets.boundary_layer_excess())
    prediction = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    with attach_smc_cfg(pipe, ctrl) as hook:
        assert pipe(prediction, 1.0) is prediction
        assert not ctrl.history
        for w in (2.0, 4.0):
            hook.reset()
            result = pipe(prediction, w)
            expected = prediction.clone()
            expected[2:] -= 0.1 * ((w - 1.0) / w)
            assert torch.allclose(result, expected)
        hook.reset()
        pipe.disabled = True  # e.g. a model using guidance embeddings
        assert pipe(prediction, 4.0) is prediction
        assert not ctrl.history


def test_pipeline_without_cfg_state_requires_adapter():
    with pytest.raises(ValueError, match="pipeline-specific adapter"):
        attach_smc_cfg(SimpleNamespace(transformer=torch.nn.Identity()),
                       SlidingModeGuidance())


@pytest.mark.parametrize("name,value", [("guidance_rescale", 0.5), ("skip_guidance_layers", [1])])
def test_auxiliary_guidance_modes_require_an_adapter(name, value):
    pipe = Pipeline()
    setattr(pipe, name, value)
    with attach_smc_cfg(pipe, SlidingModeGuidance()):
        with pytest.raises(ValueError, match="pipeline-specific adapter"):
            pipe(torch.ones(2, 2), 3.0)


def test_ambiguous_odd_batch_raises_and_hook_restores_instance_forward():
    model = torch.nn.Identity()
    forward = lambda x: x
    model.forward = forward
    with pytest.raises(ValueError, match="doubled CFG batch"):
        with GuidanceHook(model, SlidingModeGuidance()):
            model(torch.ones(3, 2))
    assert model.forward is forward


def test_double_hook_attachment_does_not_apply_correction_twice():
    model = torch.nn.Identity()
    with GuidanceHook(model, SlidingModeGuidance()):
        with pytest.raises(RuntimeError, match="already attached"):
            GuidanceHook(model, SlidingModeGuidance()).attach()


@pytest.mark.parametrize("out", [(), [], "prediction", torch.tensor(0.0), ("bad",)])
def test_invalid_output_has_clear_error(out):
    with pytest.raises(TypeError, match="batched prediction tensor"):
        GuidanceHook._extract(out)
