"""Pure-logic tests for the control layer. No GPU, no diffusion model.

These are the only things in this repository you can run on a laptop, and they
cover the properties the whole experiment rests on. Run them before trusting
any result:

    python -m pytest tests/ -v
"""

import pytest

from control import EMAFilter, PIConfig, PIController, ReferenceTrajectory, smoothstep


# --------------------------------------------------------------------------
# The property everything depends on
# --------------------------------------------------------------------------

def test_reduces_to_baseline_exactly():
    """K_p = K_i = 0 must return u_nom on every call, whatever the error.

    This is what makes the baseline a special case of the method, so that any
    measured difference between arms is attributable to the loop and nothing
    else. If this ever fails, every downstream number is meaningless.
    """
    c = PIController(PIConfig(u_nom=3.5, u_min=0.0, u_max=7.0, k_p=0.0, k_i=0.0))
    for r, y in [(0.0, 0.0), (0.5, 0.1), (0.1, 0.9), (1.0, 0.0)]:
        assert c.step(reference=r, measurement=y) == 3.5
    assert c.is_open_loop


def test_warmup_holds_at_nominal_and_does_not_integrate():
    """Before the reference exists there is no meaningful error. Holding at
    u_nom and refusing to integrate is what stops warm-up windup."""
    c = PIController(PIConfig(u_nom=3.5, u_min=0.0, u_max=7.0, k_p=1.0, k_i=0.5))
    for _ in range(100):
        assert c.step(reference=None, measurement=None) == 3.5
    assert c._integral == 0.0


# --------------------------------------------------------------------------
# Saturation, anti-windup, rate limit
# --------------------------------------------------------------------------

def test_saturation_respected():
    c = PIController(PIConfig(u_nom=3.5, u_min=0.0, u_max=7.0, k_p=100.0))
    assert c.step(reference=1.0, measurement=0.0) == 7.0     # huge +error
    c.reset()
    assert c.step(reference=0.0, measurement=1.0) == 0.0     # huge -error


def test_anti_windup_stops_integration_while_saturated():
    """With a persistent error the integrator must not grow without bound."""
    c = PIController(PIConfig(u_nom=3.5, u_min=0.0, u_max=7.0,
                              k_p=0.0, k_i=1.0, delta_max=None))
    for _ in range(200):
        c.step(reference=1.0, measurement=0.0)
    # Integral is allowed to reach the value that first saturates the output,
    # but must not keep climbing afterwards.
    assert c._integral <= (7.0 - 3.5) + 1.0 + 1e-9, (
        f"integral wound up to {c._integral}; anti-windup is not working"
    )


def test_rate_limit_bounds_the_step():
    c = PIController(PIConfig(u_nom=3.5, u_min=0.0, u_max=7.0,
                              k_p=100.0, delta_max=0.1))
    u1 = c.step(reference=1.0, measurement=0.0)
    assert abs(u1 - 3.5) <= 0.1 + 1e-9
    u2 = c.step(reference=1.0, measurement=0.0)
    assert abs(u2 - u1) <= 0.1 + 1e-9


# --------------------------------------------------------------------------
# The runaway guard: the sign-inversion failure mode
# --------------------------------------------------------------------------

def test_runaway_trips_when_progress_falls_at_ceiling():
    """If progress falls while the input is pinned high, the loop gain has
    inverted and the controller must stop. Structurally the same failure as
    the self-extinction result in the thesis."""
    c = PIController(PIConfig(u_nom=3.5, u_min=0.0, u_max=7.0,
                              k_p=100.0, runaway_patience=10))
    p = 0.5
    for _ in range(30):
        c.step(reference=1.0, measurement=0.0, raw_progress=p)
        p -= 0.01                       # progress going the wrong way
        if c.tripped:
            break
    assert c.tripped, "runaway guard failed to trip on falling progress"
    assert "inverted" in c.trip_reason


def test_runaway_does_not_trip_when_progress_rises():
    c = PIController(PIConfig(u_nom=3.5, u_min=0.0, u_max=7.0,
                              k_p=100.0, runaway_patience=10))
    p = 0.1
    for _ in range(100):
        c.step(reference=1.0, measurement=0.0, raw_progress=p)
        p += 0.001
    assert not c.tripped


# --------------------------------------------------------------------------
# Measurement and reference
# --------------------------------------------------------------------------

def test_ema_initialises_on_first_sample():
    """Initialising on the first sample avoids a spurious startup transient
    that the integrator would otherwise chase."""
    f = EMAFilter(lam=0.9)
    assert f.update(0.4) == pytest.approx(0.4)
    assert f.update(0.4) == pytest.approx(0.4)


def test_ema_lag_matches_formula():
    assert EMAFilter(lam=0.9).lag_iterations == pytest.approx(9.0)
    assert EMAFilter(lam=0.99).lag_iterations == pytest.approx(99.0)


def test_reference_is_none_until_warmed_up():
    ref = ReferenceTrajectory(total_iterations=1000, warmup_iterations=50, alpha=2.0)
    assert ref.at(10) is None
    for _ in range(50):
        ref.observe_demand(0.10)
    assert ref.p_target == pytest.approx(0.20)      # alpha * mean(s)
    assert ref.at(0) == pytest.approx(0.0)
    assert ref.at(1000) == pytest.approx(0.20)


def test_reference_target_is_clamped():
    ref = ReferenceTrajectory(total_iterations=100, warmup_iterations=5,
                              alpha=100.0, p_target_bounds=(0.02, 0.60))
    for _ in range(5):
        ref.observe_demand(0.9)
    assert ref.p_target == pytest.approx(0.60)


def test_smoothstep_endpoints_and_monotonicity():
    assert smoothstep(0.0) == pytest.approx(0.0)
    assert smoothstep(1.0) == pytest.approx(1.0)
    vals = [smoothstep(i / 50) for i in range(51)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
