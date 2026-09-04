"""Regression coverage for batch independence, numerical stability and limits."""

import math

import pytest
import torch

from cfgctrl import SMCConfig, SlidingModeGuidance, presets, soft_threshold
from cfgctrl.controllers import rms


def test_scalar_samples_have_independent_relative_gains():
    e = torch.tensor([1.0, 10.0, 0.0])
    batched = SlidingModeGuidance(SMCConfig(k=0.1, relative_gain=True)).correct(e)
    separate = torch.cat([
        SlidingModeGuidance(SMCConfig(k=0.1, relative_gain=True)).correct(v.reshape(1))
        for v in e
    ])
    torch.testing.assert_close(batched, separate)
    torch.testing.assert_close(rms(e), e.abs())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_large_error_remains_finite(dtype):
    e = torch.tensor([[40000.0, -40000.0]], dtype=dtype)
    ctrl = SlidingModeGuidance(SMCConfig(k=0.1, relative_gain=True))
    for _ in range(2):
        result = ctrl.correct(e)
        assert result.dtype == dtype
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result.float(), 0.9 * e.float(), rtol=0.01, atol=1.0)
        assert all(math.isfinite(value) for value in vars(ctrl.history[-1]).values())


def test_diagnostics_do_not_report_corrections_lost_to_output_rounding():
    ctrl = SlidingModeGuidance(SMCConfig(k=0.1))
    e = torch.tensor([[40000.0]], dtype=torch.float16)
    for _ in range(2):
        assert torch.equal(ctrl.correct(e), e)
        assert ctrl.history[-1].delta_rms == 0.0
        assert ctrl.history[-1].switch_activity == 0.0


@pytest.mark.parametrize("preset", [presets.boundary_layer, presets.boundary_layer_excess])
def test_zero_gain_boundary_presets_preserve_identity_and_signed_zero(preset):
    e = torch.tensor([[0.0, -0.0, 3e38]], requires_grad=True)
    result = SlidingModeGuidance(preset(k=0.0)).correct(e, w=3.0)
    assert torch.equal(result, e)
    assert torch.equal(torch.signbit(result), torch.signbit(e))
    result.sum().backward()
    assert torch.equal(e.grad, torch.ones_like(e))


@pytest.mark.parametrize("store_corrected", [True, False])
def test_relative_saturation_scales_entire_error_sequence(store_corrected):
    cfg = SMCConfig(k=0.1, phi=2.0, switching="sat", relative_gain=True,
                    store_corrected=store_corrected)
    original, scaled = SlidingModeGuidance(cfg), SlidingModeGuidance(cfg)
    seq = [torch.tensor([[0.01, 1.0], [0.04, 0.2]]),
           torch.tensor([[0.02, 0.7], [0.02, 0.1]]), torch.zeros(2, 2)]
    for e in seq:
        torch.testing.assert_close(scaled.correct(10 * e), 10 * original.correct(e))
        assert torch.isfinite(scaled.history[-1].k_eff * torch.ones(1)).all()


def test_boundary_layer_is_not_memoryless_soft_threshold_after_error_changes():
    ctrl = SlidingModeGuidance(presets.boundary_layer())
    ctrl.correct(torch.tensor([[1.0]]))
    e = torch.tensor([[0.01]])
    # The previous strong error keeps s outside the boundary layer. Treating
    # this controller as a proximal operator at every step would miss reversal.
    torch.testing.assert_close(ctrl.correct(e), torch.tensor([[-0.09]]))
    assert torch.equal(soft_threshold(e, 0.1), torch.zeros_like(e))


@pytest.mark.parametrize("name", ["lam", "k", "phi"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_nonfinite_and_negative_config_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        SlidingModeGuidance(SMCConfig(**{name: value}))


def test_changed_batch_requires_reset_instead_of_broadcasting_memory():
    ctrl = SlidingModeGuidance()
    ctrl.correct(torch.ones(1, 4))
    with pytest.raises(ValueError, match="reset"):
        ctrl.correct(torch.ones(2, 4))
    assert len(ctrl.history) == 1
    ctrl.reset()
    assert ctrl.correct(torch.ones(2, 4)).shape == (2, 4)


@pytest.mark.parametrize("e", [torch.tensor(1.0), torch.empty(0, 4),
                               torch.empty(2, 0), torch.ones(1, 4, dtype=torch.int64)])
def test_invalid_error_tensor_rejected(e):
    with pytest.raises(ValueError):
        SlidingModeGuidance().correct(e)


def test_excess_only_rejects_nonfinite_scale():
    with pytest.raises(ValueError, match="finite"):
        SlidingModeGuidance(presets.boundary_layer_excess()).correct(torch.ones(1, 4), w=float("nan"))
