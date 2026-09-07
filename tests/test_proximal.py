"""Properties that motivate the memoryless correction, plus runner integration."""

from dataclasses import replace

import pytest
import torch

from cfgctrl import SMCConfig, SlidingModeGuidance, presets, soft_threshold
from cfgctrl import arms


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("w", [1.0, 1.5, 7.0])
def test_no_component_reversal_amplification_or_zero_error_correction(relative, w):
    cfg = replace(presets.proximal_excess(), relative_gain=relative)
    ctrl = SlidingModeGuidance(cfg)
    ctrl.correct(torch.full((2, 7), 1000.0), w=w)
    e = torch.tensor([[-1.0, -0.2, -0.01, 0.0, 0.01, 0.2, 1.0],
                      [0.5, 0.0, -0.3, 0.005, 0.0, -0.005, -1.0]])
    applied = ctrl.correct(e, w=w)
    assert (applied * e >= 0).all()
    assert (applied.abs() <= e.abs()).all()
    assert torch.equal(applied[e == 0], e[e == 0])
    assert torch.equal(ctrl.correct(torch.zeros_like(e), w=w), torch.zeros_like(e))
    assert ctrl.history[-1].delta_rms == 0.0
    assert ctrl.history[-1].s_rms == 0.0


def test_previous_errors_cannot_change_the_current_proximal_correction():
    cfg = presets.proximal_excess()
    left, right = SlidingModeGuidance(cfg), SlidingModeGuidance(cfg)
    left.correct(torch.tensor([[1e20, -1e20]]), w=3.0)
    right.correct(torch.tensor([[-0.1, 0.1]]), w=3.0)
    e = torch.tensor([[0.01, -0.01]])
    assert torch.equal(left.correct(e, w=3.0), right.correct(e, w=3.0))
    # The old measured-memory surface may use a stale opposite direction here.
    torch.testing.assert_close(left.correct(e, w=3.0), e / 3.0)


def test_paper_self_oscillation_is_removed_for_constant_small_error():
    e = torch.full((1, 16), 0.02)
    paper = SlidingModeGuidance(presets.paper())
    proximal = SlidingModeGuidance(presets.proximal_excess())
    for _ in range(10):
        paper.correct(e, w=3.0)
        proximal.correct(e, w=3.0)
    assert paper.history[-1].chatter == 1.0
    assert paper.history[-1].switch_activity == pytest.approx(0.2)
    assert proximal.history[-1].chatter == 0.0
    assert proximal.history[-1].switch_activity == 0.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_proximal_dtype_and_exact_cfg_limits(dtype):
    e = torch.tensor([[0.0, -0.0, 0.01, -2.0]], dtype=dtype, requires_grad=True)
    for cfg, w in [(presets.proximal_excess(k=0), 7.0),
                   (presets.proximal_relative_excess(k=0), 7.0),
                   (presets.proximal_excess(), 1.0)]:
        actual = SlidingModeGuidance(cfg).correct(e, w=w)
        assert actual.dtype == dtype
        assert torch.equal(actual, e)
        assert torch.equal(actual.signbit(), e.signbit())
    active = SlidingModeGuidance(presets.proximal_relative_excess()).correct(e, w=7.0)
    assert active.dtype == dtype and torch.isfinite(active).all()
    active.sum().backward()
    assert torch.equal(e.grad, torch.ones_like(e))  # same inference-only gradient convention


@pytest.mark.parametrize("relative", [False, True])
def test_guided_velocity_matches_conditional_plus_thresholded_extrapolation(relative):
    gen = torch.Generator().manual_seed(140)
    vu = torch.randn(3, 20, generator=gen, dtype=torch.float64)
    vc = vu + torch.randn(3, 20, generator=gen, dtype=torch.float64) * 0.2
    e = vc - vu
    threshold = 0.1 * e.square().mean(1, keepdim=True).sqrt() if relative else 0.1
    ctrl = SlidingModeGuidance(replace(presets.proximal_excess(), relative_gain=relative))
    torch.testing.assert_close(ctrl.guided_velocity(vu, vc, 7.0),
                               vc + 6.0 * soft_threshold(e, threshold))


def test_full_proximal_map_is_nonexpansive_at_fixed_threshold():
    cfg = replace(presets.proximal_excess(), excess_only=False)
    gen = torch.Generator().manual_seed(141)
    a = torch.randn(5, 100, generator=gen)
    b = a + 0.01 * torch.randn(5, 100, generator=gen)
    ctrl = SlidingModeGuidance(cfg)
    out_a, out_b = ctrl.correct(a), ctrl.correct(b)
    assert torch.linalg.vector_norm(out_a - out_b) <= torch.linalg.vector_norm(a - b) + 1e-6
    torch.testing.assert_close(out_a, soft_threshold(a, cfg.k))


def test_relative_threshold_is_per_sample_and_scales_with_error_sequence():
    cfg = presets.proximal_relative_excess()
    ctrl, scaled = SlidingModeGuidance(cfg), SlidingModeGuidance(cfg)
    for magnitude in (1.0, 0.1, 0.0, 0.005):
        e = magnitude * torch.tensor([[1.0, 0.02], [0.0, 0.001]])
        got = ctrl.correct(e, w=3.0)
        torch.testing.assert_close(scaled.correct(20 * e, w=3.0), 20 * got)
        separate = torch.cat([SlidingModeGuidance(cfg).correct(row[None], w=3.0) for row in e])
        torch.testing.assert_close(got, separate)




@pytest.mark.parametrize("name,relative", [("proximal", False), ("proximal_relative", True)])
def test_cli_resolves_new_arm_and_describes_its_actual_parameters(name, relative):
    _, cfg = arms.parse_arm(name + ":k=0.05", lam=6.0, k=0.1)
    assert cfg.mode == "proximal" and cfg.excess_only and not cfg.store_corrected
    assert cfg.k == 0.05 and cfg.relative_gain == relative
    description = arms.describe_arm(name, cfg)
    assert "memoryless soft threshold" in description and "lam=" not in description


@pytest.mark.parametrize("setting", ["lam=6", "phi=0.6", "switching=sat"])
def test_unused_sliding_parameters_are_not_silently_tuned_on_proximal(setting):
    with pytest.raises(ValueError, match="does not use"):
        arms.parse_arm("proximal:" + setting, lam=6.0, k=0.1)






def test_invalid_mode_and_corrected_memory_are_rejected():
    with pytest.raises(ValueError, match="unknown correction mode"):
        SlidingModeGuidance(SMCConfig(mode="unknown"))
    with pytest.raises(ValueError, match="store_corrected=False"):
        SlidingModeGuidance(SMCConfig(mode="proximal"))
