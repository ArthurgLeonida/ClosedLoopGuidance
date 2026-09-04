"""Pure-logic tests for the CFG-Ctrl reimplementation. CPU only, a few seconds.

    python -m pytest tests/test_smc_cfg.py -v

The important ones, in order:
  * test_k_zero_is_exact_cfg_for_every_variant  -- baseline is a special case
  * test_reference_implementation_of_paper_law  -- we implement what the
    authors' code does, not what we think Algorithm 1 says
  * the toy-plant tests -- the analytic velocity field really transports
    N(0, I) to the mixture, otherwise every experiment number is noise
"""

from dataclasses import dataclass

import math
import pytest
import torch

from cfgctrl import SMCConfig, SlidingModeGuidance, presets, ring_mixture, soft_threshold
from cfgctrl.diffusers_hook import GuidanceHook

torch.manual_seed(0)


def _seq(n=8, shape=(4, 3, 5, 5), seed=0):
    g = torch.Generator().manual_seed(seed)
    e0 = torch.randn(shape, generator=g)
    return [e0 * (0.9 ** i) + 0.05 * torch.randn(shape, generator=g) for i in range(n)]


# --------------------------------------------------------------------------
# Baseline is a special case
# --------------------------------------------------------------------------

def test_k_zero_is_exact_cfg_for_every_variant():
    variants = [
        dict(),
        dict(switching="sat", phi=0.5),
        dict(switching="tanh", phi=0.5),
        dict(direction="unit"),
        dict(time_scaled=True),
        dict(store_corrected=False),
        dict(relative_gain=True),
    ]
    for kw in variants:
        c = SlidingModeGuidance(SMCConfig(k=0.0, **kw))
        assert c.is_cfg
        for e in _seq():
            assert torch.equal(c.correct(e, dt=1 / 30), e), kw


# --------------------------------------------------------------------------
# Faithfulness to the paper / authors' code
# --------------------------------------------------------------------------

def test_reference_implementation_of_paper_law():
    """Replay the authors' pipeline/common_cfg_ctrl.py literally:
        s   = (e - prev) + lam * prev
        u   = -K * sign(s)
        e   = e + u
        prev = e            # the CORRECTED error is stored
    """
    lam, k = 6.0, 0.1
    c = SlidingModeGuidance(presets.paper(lam, k))
    prev = None
    for e in _seq():
        got = c.correct(e)
        if prev is None:
            prev = e.clone()
        s = (e - prev) + lam * prev
        ref = e - k * torch.sign(s)
        prev = ref.clone()
        assert torch.allclose(got, ref, atol=0, rtol=0)


def test_first_step_is_a_sign_shrink_of_e():
    """At the first step prev := e so s = lam*e and delta = -k*sign(e)."""
    k = 0.1
    c = SlidingModeGuidance(presets.paper(6.0, k))
    e = torch.randn(2, 16)
    assert torch.allclose(c.correct(e), e - k * torch.sign(e))


def test_store_corrected_false_uses_measured_error():
    lam, k = 6.0, 0.1
    c = SlidingModeGuidance(SMCConfig(lam=lam, k=k, store_corrected=False))
    e1, e2 = torch.randn(1, 8), torch.randn(1, 8)
    c.correct(e1)
    s2 = (e2 - e1) + lam * e1
    assert torch.allclose(c.correct(e2), e2 - k * torch.sign(s2))


def test_large_lambda_makes_surface_sign_equal_to_error_sign():
    """With lam = 6 and a slowly varying e, sign(s) == sign(e_prev) for every
    element: the derivative term never decides. The paper's law then reduces
    to e - k*sign(e)."""
    c = SlidingModeGuidance(presets.paper(6.0, 0.01))
    e = torch.randn(3, 64) + 0.5
    for _ in range(10):
        c.correct(e)
        e = 0.97 * e
    assert all(h.deriv_matters == 0.0 for h in c.history)


# --------------------------------------------------------------------------
# Refinements
# --------------------------------------------------------------------------

def test_boundary_layer_phi_k_lam_is_exact_soft_threshold_at_first_step():
    lam, k = 6.0, 0.1
    c = SlidingModeGuidance(presets.boundary_layer(lam, k))          # phi = k*lam
    e = torch.randn(4, 32) * 0.3                                     # many |e| < k
    assert torch.allclose(c.correct(e), soft_threshold(e, k), atol=1e-6)


def test_sat_equals_sign_outside_layer_and_is_continuous_inside():
    lam, k, phi = 6.0, 0.1, 0.6
    big = torch.tensor([[3.0, -3.0]])
    c_sign = SlidingModeGuidance(presets.paper(lam, k))
    c_sat = SlidingModeGuidance(SMCConfig(lam=lam, k=k, switching="sat", phi=phi))
    assert torch.allclose(c_sign.correct(big), c_sat.correct(big))
    # inside the layer the map e -> e_app is Lipschitz with constant 1 + k*lam/phi
    c_sat.reset()
    e_a = torch.tensor([[0.01, -0.02]])
    e_b = e_a + 1e-3
    out_a = c_sat.correct(e_a)
    c_sat.reset()
    out_b = c_sat.correct(e_b)
    assert (out_b - out_a).abs().max() <= (1 + k * lam / phi) * 1e-3 + 1e-7


def test_unit_direction_correction_has_norm_k_per_sample():
    k = 0.3
    c = SlidingModeGuidance(SMCConfig(lam=6.0, k=k, direction="unit"))
    e = torch.randn(5, 3, 7, 7)
    delta = c.correct(e) - e
    norms = delta.flatten(1).norm(dim=1)
    assert torch.allclose(norms, torch.full_like(norms, k), atol=1e-5)


def test_time_scaled_surface_converges_with_step_refinement():
    """e(t) = e0 exp(-a t), t: 0 -> 1. The true surface is s = (lam - a) e.
    The time-scaled discrete surface converges to it as steps grow; the
    paper's unscaled difference does not (its derivative term shrinks 1/N)."""
    a, lam = 3.0, 6.0
    e0 = torch.ones(1, 1)

    def surface_at_half(steps, time_scaled):
        c = SlidingModeGuidance(SMCConfig(lam=lam, k=0.0, time_scaled=time_scaled))
        dt = 1.0 / steps
        for i in range(steps + 1):
            t = i * dt
            c.correct(e0 * math.exp(-a * t), dt=dt)
        # history index nearest t = 0.5
        return c.history[steps // 2].s_rms

    e_half = math.exp(-a * 0.5)
    s_true = (lam - a) * e_half
    err_30 = abs(surface_at_half(30, True) - s_true) / s_true
    err_300 = abs(surface_at_half(300, True) - s_true) / s_true
    assert err_300 < err_30 / 5             # first-order convergence (O(dt))
    assert err_300 < 0.03                   # lam*e_{t-1} term costs lam*a*dt = 2% at 300 steps
    # unscaled: derivative term is O(dt) so the surface is ~ lam*e, independent of a
    s_unscaled = surface_at_half(300, False)
    assert abs(s_unscaled - lam * e_half) / (lam * e_half) < 0.02


def _integrator_plant(cfg, steps=400, drift=0.6, gain=1.0, e0=1.0, dt=None):
    """Scalar plant where the correction has authority over the next error:
        e_{t+1} = e_t + dt * (drift + gain * delta_t)
    Returns the controller (for diagnostics) and the |s| trace it saw."""
    dt = 1.0 / steps if dt is None else dt
    c = SlidingModeGuidance(cfg)
    e = torch.tensor([[e0]])
    for _ in range(steps):
        e_app = c.correct(e, dt=dt)
        delta = e_app - e
        e = e + dt * (drift + gain * delta)
    return c


def test_super_twisting_limit_cycles_on_a_relative_degree_zero_surface():
    """s = e_dot + lam*e depends ALGEBRAICALLY on the correction (the correction
    changes e_dot at once), so s has relative degree zero. Super-twisting is
    designed for s_dot = u + d (relative degree one). On this plant it settles
    into the 2-cycle |s_{t+1}| = k1*sqrt(|s_t|)  ->  |s| = k1^2, alternating
    sign every step. This is a property of the paper's surface, not a bug."""
    lam, k1 = 2.0, 1.5
    sta_cfg = SMCConfig(lam=lam, k=k1, k2=3.0, super_twisting=True, time_scaled=True,
                        switching="sat", phi=0.05, z_max=5.0, store_corrected=False)
    c = _integrator_plant(sta_cfg)
    tail = c.history[int(0.8 * len(c.history)):]
    s_tail = sum(h.s_rms for h in tail) / len(tail)
    assert abs(s_tail - k1 ** 2) / k1 ** 2 < 0.15
    assert sum(h.chatter for h in tail) / len(tail) > 0.95          # flips every step


def test_sta_integrator_is_clamped_by_z_max():
    cfg = SMCConfig(lam=6.0, k=0.1, k2=10.0, super_twisting=True, time_scaled=True,
                    switching="sat", phi=0.6, z_max=0.25)
    c = SlidingModeGuidance(cfg)
    e = torch.ones(1, 4)
    for _ in range(200):
        c.correct(e, dt=0.1)
    assert c.history[-1].z_rms <= 0.25 + 1e-6


def test_adaptive_gain_grows_outside_band_and_shrinks_inside():
    cfg = SMCConfig(lam=6.0, k=0.1, adaptive=True, adapt_rate=1.0, adapt_band=0.5,
                    k_min=0.02, k_max=1.0, time_scaled=True, switching="sat", phi=0.6,
                    store_corrected=False)
    c = SlidingModeGuidance(cfg)
    big = torch.full((1, 8), 2.0)                       # rms(s) = 12 > band
    ks = []
    for _ in range(50):
        c.correct(big, dt=0.1)
        ks.append(float(c.gain.mean()))
    assert all(b >= a for a, b in zip(ks, ks[1:])) and ks[-1] == pytest.approx(1.0)
    small = torch.full((1, 8), 0.25 / 6.0)              # rms(s) = 0.25 < band: inside, not negligible
    for _ in range(400):
        c.correct(small, dt=0.1)
    assert float(c.gain.mean()) == pytest.approx(0.02)   # Plestan's law decays at rate ~ rms(s)


def test_relative_gain_is_scale_free():
    """Scaling e by 10 scales the correction by 10 when k is relative."""
    c1 = SlidingModeGuidance(SMCConfig(lam=6.0, k=0.05, relative_gain=True))
    c2 = SlidingModeGuidance(SMCConfig(lam=6.0, k=0.05, relative_gain=True))
    e = torch.randn(2, 16)
    d1 = c1.correct(e) - e
    d2 = c2.correct(10 * e) - 10 * e
    assert torch.allclose(d2, 10 * d1, atol=1e-5)


def test_excess_only_is_exact_cfg_at_w_one_and_scales_the_correction():
    """The paper's law alters v_cond + (w-1)e as a whole, so at w = 1 it no
    longer samples the conditional law. Shrinking only the extrapolation
    (w-1)e gives exactly CFG at w = 1 and (w-1)/w of the correction otherwise."""
    lam, k = 6.0, 0.1
    e = torch.randn(3, 16)
    c1 = SlidingModeGuidance(presets.boundary_layer_excess(lam, k))
    assert torch.equal(c1.correct(e, w=1.0), e)
    c3 = SlidingModeGuidance(presets.boundary_layer_excess(lam, k))
    ref = SlidingModeGuidance(presets.boundary_layer(lam, k))
    delta_full = ref.correct(e) - e
    assert torch.allclose(c3.correct(e, w=3.0), e + (2.0 / 3.0) * delta_full, atol=1e-6)
    with pytest.raises(ValueError):
        SlidingModeGuidance(presets.boundary_layer_excess(lam, k)).correct(e)   # w missing


def test_warmup_returns_zero_guidance_then_resumes():
    c = SlidingModeGuidance(SMCConfig(lam=6.0, k=0.1, warmup_steps=2))
    e = torch.randn(1, 4)
    assert torch.equal(c.correct(e), torch.zeros_like(e))
    assert torch.equal(c.correct(e), torch.zeros_like(e))
    assert torch.allclose(c.correct(e), e - 0.1 * torch.sign(e))


def test_configuration_validation():
    with pytest.raises(ValueError):
        SlidingModeGuidance(SMCConfig(switching="sat", phi=0.0))
    with pytest.raises(ValueError):
        SlidingModeGuidance(SMCConfig(time_scaled=True)).correct(torch.zeros(1, 2))
    with pytest.raises(ValueError):
        SlidingModeGuidance(SMCConfig(k=-1.0))


# --------------------------------------------------------------------------
# Toy plant: the field must actually be the flow-matching field
# --------------------------------------------------------------------------

def test_single_gaussian_flow_transports_noise_to_target():
    from cfgctrl import GaussianMixtureFlow
    mu, sd = torch.tensor([[2.0, -1.0, 0.5]]), torch.tensor([0.7])
    plant = GaussianMixtureFlow(mu, sd)
    x, _ = plant.sample(n=6000, cond=0, w=1.0, steps=300, seed=3)
    assert torch.allclose(x.mean(0), mu[0], atol=0.06)
    assert torch.allclose(x.std(0), sd.expand(3), atol=0.04)


def test_unguided_conditional_sampling_recovers_class_and_bayes_rate():
    plant = ring_mixture(k=8, radius=4.0, std=0.75)
    x, _ = plant.sample(n=6000, cond=0, w=1.0, steps=200, seed=0)
    assert plant.frechet_distance(x, 0) < 0.12
    truth = plant.true_class_samples(6000, 0, seed=5)
    bayes = plant.class_accuracy(truth, 0)
    assert 0.5 < bayes < 1.0                     # classes overlap: guidance has a job
    assert abs(plant.class_accuracy(x, 0) - bayes) < 0.03


def test_cfg_trades_fidelity_for_alignment():
    plant = ring_mixture()
    x1, _ = plant.sample(n=3000, cond=0, w=1.0, steps=30, seed=0)
    x5, _ = plant.sample(n=3000, cond=0, w=5.0, steps=30, seed=0)
    assert plant.class_accuracy(x5, 0) > plant.class_accuracy(x1, 0)
    assert plant.frechet_distance(x5, 0) > plant.frechet_distance(x1, 0)


def test_frechet_distance_of_true_samples_is_small():
    plant = ring_mixture()
    assert plant.frechet_distance(plant.true_class_samples(6000, 0), 0) < 0.08


def test_error_jacobian_matches_autograd():
    plant = ring_mixture(k=4, dim=2)
    x = torch.randn(3, 2) * 2
    J_fd = plant.error_jacobian(x, 0.5, cond=0)
    for b in range(3):
        J_ad = torch.autograd.functional.jacobian(lambda z: plant.error(z[None], 0.5, 0)[0], x[b])
        assert torch.allclose(J_fd[b], J_ad, atol=1e-3)


# --------------------------------------------------------------------------
# The diffusers seam, with a dummy denoiser
# --------------------------------------------------------------------------

class _Dummy(torch.nn.Module):
    def __init__(self, container="tensor"):
        super().__init__()
        self.container = container

    def forward(self, x, timestep=None, **kw):
        out = torch.tanh(x) * 2 + 0.1 * (x ** 2)
        if self.container == "tensor":
            return out
        if self.container == "tuple":
            return (out, "extra")

        @dataclass
        class Out:
            sample: torch.Tensor
        return Out(sample=out)


@pytest.mark.parametrize("container", ["tensor", "tuple", "sample"])
def test_hook_makes_pipeline_cfg_formula_equal_paper_law(container):
    torch.manual_seed(1)
    w = 7.5
    model = _Dummy(container)
    lat = torch.randn(2, 4, 8, 8)
    x = torch.cat([lat, lat])                                # [uncond, cond] doubled batch
    raw = GuidanceHook._extract(model(x, timestep=torch.tensor([1000.0])))
    unc_ref, cond_ref = raw.chunk(2)
    ref_ctrl = SlidingModeGuidance(presets.paper())
    expected = unc_ref + w * ref_ctrl.correct(cond_ref - unc_ref)

    hook = GuidanceHook(model, SlidingModeGuidance(presets.paper()), default_dt=1 / 30).attach()
    try:
        out = GuidanceHook._extract(model(x, timestep=torch.tensor([1000.0])))
    finally:
        hook.detach()
    unc, cond = out.chunk(2)
    got = unc + w * (cond - unc)                             # what every pipeline computes
    assert torch.allclose(got, expected, atol=1e-6)
    assert torch.equal(unc, unc_ref)                         # unconditional branch untouched


def test_hook_is_transparent_when_k_is_zero_and_detaches_cleanly():
    model = _Dummy()
    x = torch.randn(4, 3)
    ref = model(x, timestep=torch.tensor([500.0]))
    hook = GuidanceHook(model, SlidingModeGuidance(presets.cfg_baseline())).attach()
    assert torch.equal(model(x, timestep=torch.tensor([500.0])), ref)
    hook.detach()
    assert model.forward.__func__ is _Dummy.forward


def test_hook_derives_dt_from_consecutive_timesteps():
    model = _Dummy()
    ctrl = SlidingModeGuidance(SMCConfig(lam=6.0, k=0.1, time_scaled=True))
    hook = GuidanceHook(model, ctrl, num_train_timesteps=1000.0, default_dt=1 / 30).attach()
    x = torch.randn(2, 3)
    model(x, timestep=torch.tensor([1000.0]))
    model(x, timestep=torch.tensor([966.6667]))
    hook.detach()
    assert ctrl.history[0].dt == pytest.approx(1 / 30)
    assert ctrl.history[1].dt == pytest.approx(0.0333333, abs=1e-6)
