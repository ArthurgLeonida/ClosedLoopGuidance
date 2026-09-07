"""Properties that motivate the memoryless correction, plus runner integration."""

from dataclasses import asdict, replace

import pytest
import torch

from cfgctrl import SMCConfig, SlidingModeGuidance, presets, soft_threshold
from cfgctrl.diffusers_hook import GuidanceHook
from experiments import real_model, signals, toy_smc_cfg


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


def test_hook_applies_proximal_correction_after_arbitrary_history():
    model = torch.nn.Identity()
    ctrl = SlidingModeGuidance(presets.proximal_excess())
    with GuidanceHook(model, ctrl, guidance_scale=3.0):
        model(torch.tensor([[0.0, 0.0], [10.0, -10.0]]))
        vu, vc = torch.tensor([[0.2, 0.1]]), torch.tensor([[0.21, -0.1]])
        u, c = model(torch.cat([vu, vc])).chunk(2)
        torch.testing.assert_close(u + 3 * (c-u), vc + 2 * soft_threshold(vc-vu, 0.1))


@pytest.mark.parametrize("name,relative", [("proximal", False), ("proximal_relative", True)])
def test_cli_resolves_new_arm_and_describes_its_actual_parameters(name, relative):
    _, cfg = real_model.parse_arm(name + ":k=0.05", lam=6.0, k=0.1)
    assert cfg.mode == "proximal" and cfg.excess_only and not cfg.store_corrected
    assert cfg.k == 0.05 and cfg.relative_gain == relative
    description = real_model.describe_arm(name, cfg)
    assert "memoryless soft threshold" in description and "lam=" not in description


@pytest.mark.parametrize("setting", ["lam=6", "phi=0.6", "switching=sat"])
def test_unused_sliding_parameters_are_not_silently_tuned_on_proximal(setting):
    with pytest.raises(ValueError, match="does not use"):
        real_model.parse_arm("proximal:" + setting, lam=6.0, k=0.1)


def test_old_run_configs_can_resume_without_redefining_the_sliding_law():
    config = dict(model="m", dtype="fp32", steps=30, prompts=["p"], w=[3.0], seeds=[0])
    new_arm = asdict(presets.paper())
    old_arm = {k: v for k, v in new_arm.items() if k != "mode"}
    result = real_model.merge_config({**config, "arms": {"paper": old_arm}},
                                    {**config, "arms": {"paper": new_arm}})
    assert result["arms"]["paper"]["mode"] == "sliding"
    with pytest.raises(ValueError, match="different settings"):
        real_model.merge_config(result, {**config, "arms": {"paper": asdict(presets.proximal_excess())}})


def test_proximal_is_part_of_toy_pareto_study_and_has_no_plateau_prediction():
    configs = [cfg for _, cfg in toy_smc_cfg.method_table(0.1) if cfg.mode == "proximal"]
    assert len(configs) == 2
    assert {cfg.relative_gain for cfg in configs} == {False, True}
    assert all(signals.predicted_plateau(asdict(cfg), w=3.0) is None for cfg in configs)


def test_invalid_mode_and_corrected_memory_are_rejected():
    with pytest.raises(ValueError, match="unknown correction mode"):
        SlidingModeGuidance(SMCConfig(mode="unknown"))
    with pytest.raises(ValueError, match="store_corrected=False"):
        SlidingModeGuidance(SMCConfig(mode="proximal"))
