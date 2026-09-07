"""Numerical and invariant tests for the core CFG-Ctrl guidance laws."""


import pytest
import torch

from cfgctrl import SMCConfig, SlidingModeGuidance, presets, soft_threshold

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
