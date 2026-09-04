"""Regression checks for the analytic plant and its numerical endpoints."""

import math

import pytest
import torch

from cfgctrl import GaussianMixtureFlow, SMCConfig, SlidingModeGuidance, ring_mixture


@pytest.fixture(autouse=True, scope="module")
def _small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_velocity_is_accurate_at_tiny_sigma_and_exact_at_data_endpoint():
    plant = GaussianMixtureFlow([[2.0, -1.0]], [0.7])
    x = torch.tensor([[1.5, -0.25], [-3.0, 5.0]])
    assert torch.allclose(plant.velocity(x, 1e-9, 0), -x, atol=1e-6, rtol=1e-6)
    assert torch.equal(plant.velocity(x, 0), -x)
    assert torch.equal(plant.error(x, 0, 0), torch.zeros_like(x))


@pytest.mark.parametrize("sigma", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("cond", [None, 1])
def test_direct_velocity_matches_posterior_identity_away_from_zero(sigma, cond):
    plant = GaussianMixtureFlow([[-2.0, 0.3], [1.0, 3.0]], [0.5, 1.7], [0.2, 0.8])
    x = torch.tensor([[0.1, -0.5], [-1.0, 2.5]], dtype=torch.float64)
    expected = (x - plant.posterior_x0(x, sigma, cond)) / sigma
    assert torch.allclose(plant.velocity(x, sigma, cond), expected, atol=1e-12, rtol=1e-12)


def test_double_precision_jacobian_matches_autograd():
    plant = ring_mixture(k=4)
    x = torch.tensor([[0.3, -0.7], [-2.0, 1.0]], dtype=torch.float64)
    jacobian = plant.error_jacobian(x, 0.4, 0, eps=1e-5)
    assert jacobian.dtype == x.dtype
    for b in range(x.shape[0]):
        expected = torch.autograd.functional.jacobian(lambda z: plant.error(z[None], 0.4, 0)[0], x[b])
        assert torch.allclose(jacobian[b], expected, atol=1e-9, rtol=1e-8)


def test_zero_weight_components_do_not_poison_conditioning():
    plant = GaussianMixtureFlow([[0.0], [3.0]], [1.0, 0.7], [1.0, 0.0])
    x = torch.tensor([[0.25], [2.0]])
    conditioned = GaussianMixtureFlow([[3.0]], [0.7])
    assert torch.equal(plant.velocity(x, 0.5, 1), conditioned.velocity(x, 0.5, 0))
    assert torch.isfinite(plant.velocity(x, 0.5)).all()
    assert torch.equal(plant.class_log_posterior(x).exp()[:, 1], torch.zeros(2))


@pytest.mark.parametrize("means,stds,weights", [
    ([], [], None),
    ([[]], [1.0], None),
    ([[float("nan")]], [1.0], None),
    ([[0.0]], [0.0], None),
    ([[0.0]], [-1.0], None),
    ([[0.0]], [float("inf")], None),
    ([[0.0]], [1.0], [0.0]),
    ([[0.0]], [1.0], [-1.0]),
    ([[0.0]], [1.0], [float("nan")]),
    ([[0.0]], [1.0], [0.5, 0.5]),
])
def test_invalid_mixture_parameters_fail_early(means, stds, weights):
    with pytest.raises(ValueError):
        GaussianMixtureFlow(means, stds, weights)


@pytest.mark.parametrize("sigma", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_sigma_is_rejected_instead_of_clamped(sigma):
    with pytest.raises(ValueError, match="sigma"):
        ring_mixture().velocity(torch.zeros(2, 2), sigma)


@pytest.mark.parametrize("kwargs", [
    {"n": 0}, {"n": 1.5}, {"steps": 0}, {"steps": -1},
    {"cond": -1}, {"cond": 8}, {"cond": None}, {"w": float("nan")},
])
def test_sampling_rejects_invalid_arguments(kwargs):
    arguments = dict(n=4, cond=0, w=1.0, steps=3)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        ring_mixture().sample(**arguments)


def test_frechet_reports_w2_and_resolves_near_matching_moments():
    plant = GaussianMixtureFlow(torch.tensor([[0.0]], dtype=torch.float64), [1.0])
    x = torch.tensor([[-1 / math.sqrt(2)], [1 / math.sqrt(2)]], dtype=torch.float64)
    assert plant.frechet_distance(x + 2.0, 0) == pytest.approx(2.0, abs=1e-12)
    assert plant.frechet_distance(x + 1e-7, 0) == pytest.approx(1e-7, abs=1e-14)
    with pytest.raises(ValueError, match="B >= 2"):
        plant.frechet_distance(x[:1], 0)


def test_repeated_sampling_resets_controller_and_preserves_global_rng():
    plant = ring_mixture(k=3)
    controller = SlidingModeGuidance(SMCConfig())
    initial_rng = torch.get_rng_state().clone()
    first, first_trace = plant.sample(8, 0, 2.0, controller, steps=4, seed=123)
    second, second_trace = plant.sample(8, 0, 2.0, controller, steps=4, seed=123)
    assert torch.equal(first, second)
    assert first_trace.steps == second_trace.steps
    assert len(controller.history) == 4
    assert torch.equal(torch.get_rng_state(), initial_rng)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_plant_fields_sampling_and_metrics_support_cuda():
    plant = GaussianMixtureFlow(torch.tensor([[1.0, -1.0]], device="cuda"), [0.7])
    x, _ = plant.sample(8, 0, 1.0, steps=3)
    assert x.is_cuda
    assert plant.error_jacobian(x, 0.5, 0).is_cuda
    assert math.isfinite(plant.frechet_distance(x, 0))
