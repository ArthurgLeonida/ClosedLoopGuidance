"""Pure-logic tests for the CFG-Ctrl reimplementation. CPU only, a few seconds.

    python -m pytest tests/ -q

The ones that carry the most weight, in order:
  * test_k_zero_is_exact_cfg_for_every_variant  -- the baseline is a special
    case of the method, so any measured difference is due to the correction
  * test_reference_implementation_of_paper_law  -- we implement what the
    authors' code does, not what we think Algorithm 1 says
  * test_corrected_memory_alternates_once_the_error_is_small -- the chattering
    mechanism, reproduced from first principles
  * the toy-plant tests -- the analytic velocity field really does transport
    N(0, I) to the mixture, otherwise every experiment number is noise
"""

from dataclasses import dataclass

import pytest
import torch

from cfgctrl import SMCConfig, SlidingModeGuidance, presets, ring_mixture, soft_threshold
from cfgctrl.diffusers_hook import GuidanceHook

torch.manual_seed(0)


def _seq(n=8, shape=(4, 3, 5, 5), seed=0):
    """A decaying, slightly noisy error sequence, like a real denoising run."""
    g = torch.Generator().manual_seed(seed)
    e0 = torch.randn(shape, generator=g)
    return [e0 * (0.9 ** i) + 0.05 * torch.randn(shape, generator=g) for i in range(n)]


# --------------------------------------------------------------------------
# The baseline is a special case of the method
# --------------------------------------------------------------------------

def test_k_zero_is_exact_cfg_for_every_variant():
    for kw in [dict(),
               dict(switching="sat", phi=0.5),
               dict(store_corrected=False),
               dict(relative_gain=True),
               dict(excess_only=True)]:
        c = SlidingModeGuidance(SMCConfig(k=0.0, **kw))
        assert c.is_cfg
        for e in _seq():
            assert torch.equal(c.correct(e, w=3.0), e), kw


# --------------------------------------------------------------------------
# Faithfulness to the paper / the authors' code
# --------------------------------------------------------------------------

def test_reference_implementation_of_paper_law():
    """Replay the authors' pipeline/common_cfg_ctrl.py literally:
        s    = (e - prev) + lam * prev
        u    = -k * sign(s)
        e    = e + u
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
    """At the first step prev := e, so s = lam*e and delta = -k*sign(e)."""
    k = 0.1
    c = SlidingModeGuidance(presets.paper(6.0, k))
    e = torch.randn(2, 16)
    assert torch.allclose(c.correct(e), e - k * torch.sign(e))


def test_large_lambda_makes_surface_sign_equal_to_error_sign():
    """With lam = 6 and a slowly varying e, sign(s) == sign(e_prev) for every
    element: the derivative term never decides, so the paper's law reduces to
    the memoryless sign-shrink e - k*sign(e)."""
    c = SlidingModeGuidance(presets.paper(6.0, 0.01))
    e = torch.randn(3, 64) + 0.5
    for _ in range(10):
        c.correct(e)
        e = 0.97 * e
    assert all(h.deriv_matters == 0.0 for h in c.history)


def test_corrected_memory_alternates_once_the_error_is_small():
    """The chattering mechanism. Storing the CORRECTED error puts the previous
    correction back into the surface with weight (lam - 1). Once |e| < k the
    sign of s is set by that term alone, so the correction flips every step and
    rms(s) never reaches zero. Storing the MEASURED error removes it."""
    lam, k = 6.0, 0.1
    e = torch.full((1, 8), 0.02)                      # |e| < k

    c_paper = SlidingModeGuidance(SMCConfig(lam=lam, k=k, store_corrected=True))
    deltas = [float((c_paper.correct(e) - e).flatten()[0]) for _ in range(8)]
    assert all(a * b < 0 for a, b in zip(deltas, deltas[1:])), deltas
    assert all(h.chatter == 1.0 for h in c_paper.history[1:])
    assert c_paper.history[-1].s_rms == pytest.approx((lam - 1) * k, rel=0.3)

    c_fixed = SlidingModeGuidance(SMCConfig(lam=lam, k=k, store_corrected=False))
    deltas = [float((c_fixed.correct(e) - e).flatten()[0]) for _ in range(8)]
    assert all(d == pytest.approx(-k) for d in deltas)
    assert all(h.chatter == 0.0 for h in c_fixed.history)


def test_store_corrected_false_uses_measured_error():
    lam, k = 6.0, 0.1
    c = SlidingModeGuidance(SMCConfig(lam=lam, k=k, store_corrected=False))
    e1, e2 = torch.randn(1, 8), torch.randn(1, 8)
    c.correct(e1)
    s2 = (e2 - e1) + lam * e1
    assert torch.allclose(c.correct(e2), e2 - k * torch.sign(s2))


# --------------------------------------------------------------------------
# The refinements
# --------------------------------------------------------------------------

def test_boundary_layer_phi_k_lam_is_exact_soft_threshold_at_first_step():
    """phi = k*lam turns the law into the L1 proximal operator: small elements
    are zeroed rather than sign-flipped, large ones shrink by k."""
    lam, k = 6.0, 0.1
    c = SlidingModeGuidance(presets.boundary_layer(lam, k))
    e = torch.randn(4, 32) * 0.3                                # many |e| < k
    assert torch.allclose(c.correct(e), soft_threshold(e, k), atol=1e-6)


def test_sat_equals_sign_outside_layer_and_is_continuous_inside():
    lam, k, phi = 6.0, 0.1, 0.6
    big = torch.tensor([[3.0, -3.0]])
    c_sign = SlidingModeGuidance(presets.paper(lam, k))
    c_sat = SlidingModeGuidance(SMCConfig(lam=lam, k=k, switching="sat", phi=phi))
    assert torch.allclose(c_sign.correct(big), c_sat.correct(big))
    # inside the layer, e -> e_app is Lipschitz with constant 1 + k*lam/phi
    e_a = torch.tensor([[0.01, -0.02]])
    c_sat.reset()
    out_a = c_sat.correct(e_a)
    c_sat.reset()
    out_b = c_sat.correct(e_a + 1e-3)
    assert (out_b - out_a).abs().max() <= (1 + k * lam / phi) * 1e-3 + 1e-7


def test_excess_only_is_exact_cfg_at_w_one_and_scales_the_correction():
    """The paper's law alters v_cond + (w-1)e as a whole, so at w = 1 it no
    longer samples the conditional law. Shrinking only the extrapolation
    (w-1)e gives exactly CFG at w = 1 and (w-1)/w of the correction otherwise."""
    lam, k = 6.0, 0.1
    e = torch.randn(3, 16)
    assert torch.equal(SlidingModeGuidance(presets.boundary_layer_excess(lam, k)).correct(e, w=1.0), e)

    delta_full = SlidingModeGuidance(presets.boundary_layer(lam, k)).correct(e) - e
    got = SlidingModeGuidance(presets.boundary_layer_excess(lam, k)).correct(e, w=3.0)
    assert torch.allclose(got, e + (2.0 / 3.0) * delta_full, atol=1e-6)

    with pytest.raises(ValueError):                             # w is required
        SlidingModeGuidance(presets.boundary_layer_excess(lam, k)).correct(e)


def test_relative_gain_is_scale_free():
    """Scaling e by 10 scales the correction by 10, so one k transfers across
    models whose velocity scales differ."""
    e = torch.randn(2, 16)
    cfg = SMCConfig(lam=6.0, k=0.05, relative_gain=True)
    d1 = SlidingModeGuidance(cfg).correct(e) - e
    d2 = SlidingModeGuidance(cfg).correct(10 * e) - 10 * e
    assert torch.allclose(d2, 10 * d1, atol=1e-5)


def test_configuration_validation():
    with pytest.raises(ValueError):
        SlidingModeGuidance(SMCConfig(switching="sat", phi=0.0))
    with pytest.raises(ValueError):
        SlidingModeGuidance(SMCConfig(k=-1.0))
    with pytest.raises(ValueError):
        SlidingModeGuidance(SMCConfig(switching="tanh"))


# --------------------------------------------------------------------------
# The toy plant must really be a flow-matching plant
# --------------------------------------------------------------------------

def test_single_gaussian_flow_transports_noise_to_target():
    from cfgctrl import GaussianMixtureFlow
    mu, sd = torch.tensor([[2.0, -1.0, 0.5]]), torch.tensor([0.7])
    plant = GaussianMixtureFlow(mu, sd)
    x, _ = plant.sample(n=6000, cond=0, w=1.0, steps=300, seed=3)
    assert torch.allclose(x.mean(0), mu[0], atol=0.06)
    assert torch.allclose(x.std(0), sd.expand(3), atol=0.04)


def test_unguided_sampling_recovers_the_class_and_its_bayes_rate():
    plant = ring_mixture(k=8, radius=4.0, std=1.5)
    x, _ = plant.sample(n=6000, cond=0, w=1.0, steps=200, seed=0)
    assert plant.frechet_distance(x, 0) < 0.15
    truth = plant.true_class_samples(6000, 0, seed=5)
    bayes = plant.class_accuracy(truth, 0)
    assert 0.5 < bayes < 1.0            # classes overlap, so guidance has a job
    assert abs(plant.class_accuracy(x, 0) - bayes) < 0.03


def test_cfg_trades_fidelity_for_alignment():
    """The whole reason a better guidance law could matter."""
    plant = ring_mixture()
    x1, _ = plant.sample(n=3000, cond=0, w=1.0, steps=30, seed=0)
    x5, _ = plant.sample(n=3000, cond=0, w=5.0, steps=30, seed=0)
    assert plant.class_confidence(x5, 0) > plant.class_confidence(x1, 0)
    assert plant.frechet_distance(x5, 0) > plant.frechet_distance(x1, 0)


def test_frechet_distance_of_true_samples_is_small():
    plant = ring_mixture()
    assert plant.frechet_distance(plant.true_class_samples(6000, 0), 0) < 0.1


def test_error_jacobian_matches_autograd():
    """E4 rests on this Jacobian, so check the finite differences against
    autograd. The tolerance is set by float32 round-off in the central
    difference (~eps_machine * |e| / h), not by the truncation error."""
    plant = ring_mixture(k=4, dim=2)
    x = torch.randn(3, 2) * 2
    J_fd = plant.error_jacobian(x, 0.5, cond=0)
    for b in range(3):
        J_ad = torch.autograd.functional.jacobian(lambda z: plant.error(z[None], 0.5, 0)[0], x[b])
        assert torch.allclose(J_fd[b], J_ad, atol=5e-3, rtol=1e-2)


# --------------------------------------------------------------------------
# The diffusers seam, against a dummy denoiser
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
def test_hook_makes_the_pipeline_cfg_formula_equal_the_paper_law(container):
    """The pipeline's own `uncond + w*(cond - uncond)` must come out equal to
    Algorithm 1 line 13 once the hook has rewritten the conditional branch."""
    torch.manual_seed(1)
    w = 7.5
    model = _Dummy(container)
    lat = torch.randn(2, 4, 8, 8)
    x = torch.cat([lat, lat])                        # [uncond, cond] doubled batch

    raw = GuidanceHook._extract(model(x, timestep=torch.tensor([1000.0])))
    unc_ref, cond_ref = raw.chunk(2)
    expected = unc_ref + w * SlidingModeGuidance(presets.paper()).correct(cond_ref - unc_ref)

    hook = GuidanceHook(model, SlidingModeGuidance(presets.paper())).attach()
    try:
        out = GuidanceHook._extract(model(x, timestep=torch.tensor([1000.0])))
    finally:
        hook.detach()
    unc, cond = out.chunk(2)
    assert torch.allclose(unc + w * (cond - unc), expected, atol=1e-6)
    assert torch.equal(unc, unc_ref)                 # unconditional branch untouched


def test_hook_is_transparent_when_k_is_zero_and_detaches_cleanly():
    model = _Dummy()
    x = torch.randn(4, 3)
    ref = model(x, timestep=torch.tensor([500.0]))
    hook = GuidanceHook(model, SlidingModeGuidance(presets.cfg_baseline())).attach()
    assert torch.equal(model(x, timestep=torch.tensor([500.0])), ref)
    hook.detach()
    assert model.forward.__func__ is _Dummy.forward
