"""Analytic flow-matching plant: a Gaussian mixture with closed-form velocity.

Why this exists. SD3.5 / Flux need a GPU and have no ground truth: you cannot
say whether a guided sample is "right", only whether a reward model likes it.
A Gaussian mixture has an exact conditional velocity field, an exact
unconditional one, and an exact target distribution, so semantic errors, their
Jacobians and discrete surface-energy changes can be measured on a laptop.
These diagnostics alone do not test all assumptions of the paper's theorem.

Convention (identical to diffusers' FlowMatchEulerDiscreteScheduler):

    x_sigma = (1 - sigma) * x0 + sigma * eps,   sigma in (0, 1],  eps ~ N(0, I)
    v(x, sigma) = E[eps - x0 | x_sigma = x]     the model's velocity
    Euler step:  x_{sigma'} = x + (sigma' - sigma) * v_hat,  sigma' < sigma

For a mixture sum_j pi_j N(mu_j, s_j^2 I):

    x_sigma | j ~ N(m_j, c_j I),  m_j = (1-sigma) mu_j,  c_j = (1-sigma)^2 s_j^2 + sigma^2
    E[x0 | x, j] = mu_j + (1-sigma) s_j^2 / c_j * (x - m_j)
    r_j(x)  = softmax_j( log pi_j - D/2 log c_j - |x - m_j|^2 / (2 c_j) )
    E[x0 | x] = sum_j r_j E[x0 | x, j]
    v(x, sigma) = (x - E[x0 | x]) / sigma

The conditional field for class c restricts the sum to the components of class
c; the unconditional field uses all components. The semantic error
e = v_c - v_uncond is then (E[x0|x, uncond] - E[x0|x, c]) / sigma.

Metrics with ground truth:
    frechet_distance  Gaussian moment Wasserstein-2 distance (sqrt of FID)
    class_confidence  mean posterior p(class | x) at sigma = 0 (a toy CLIP score)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import math
from numbers import Integral
import torch

from .controllers import SlidingModeGuidance, StepInfo


@dataclass
class SampleTrace:
    sigmas: List[float] = field(default_factory=list)
    steps: List[StepInfo] = field(default_factory=list)

    def series(self, name: str) -> List[float]:
        return [getattr(s, name) for s in self.steps]


class GaussianMixtureFlow:
    def __init__(self, means: torch.Tensor, stds: torch.Tensor,
                 weights: Optional[torch.Tensor] = None):
        means = torch.as_tensor(means)
        if not means.is_floating_point():
            means = means.to(torch.float32)
        # Float16 is too imprecise for posterior probabilities and Jacobians.
        if means.dtype not in (torch.float32, torch.float64):
            means = means.to(torch.float32)
        stds = torch.as_tensor(stds, dtype=means.dtype, device=means.device)
        if means.dim() != 2:
            raise ValueError("means must be [K, D]")
        K, D = means.shape
        if K == 0 or D == 0 or not torch.isfinite(means).all():
            raise ValueError("means must have nonempty dimensions and finite values")
        if stds.shape != (K,):
            raise ValueError("stds must be [K] (isotropic per component)")
        if not torch.isfinite(stds).all() or (stds <= 0).any():
            raise ValueError("stds must be finite and strictly positive")
        weights = (torch.ones(K, dtype=means.dtype, device=means.device) if weights is None
                   else torch.as_tensor(weights, dtype=means.dtype, device=means.device))
        if (weights.shape != (K,) or not torch.isfinite(weights).all()
                or (weights < 0).any() or not (weights > 0).any()):
            raise ValueError("weights must be finite, nonnegative [K] with positive total mass")
        self.means, self.stds = means.clone(), stds.clone()
        # Scaling first also handles finite weights whose sum would overflow.
        weights = weights / weights.max()
        self.weights = weights / weights.sum()
        self.K, self.D = K, D
        self.n_classes = K                      # one Gaussian per class

    # ----------------------------------------------------------- fields
    def _validate_x(self, x: torch.Tensor, min_samples: int = 1) -> None:
        if (not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != self.D
                or x.shape[0] < min_samples or not x.is_floating_point()):
            raise ValueError(f"x must be a floating tensor [B, {self.D}] with B >= {min_samples}")
        if not torch.isfinite(x).all():
            raise ValueError("x must contain only finite values")

    def _validate_cond(self, cond: Optional[int]) -> None:
        if cond is not None and (isinstance(cond, bool) or not isinstance(cond, Integral)
                                 or not 0 <= cond < self.K):
            raise ValueError(f"cond must be an integer in [0, {self.K})")

    @staticmethod
    def _validate_sigma(sigma: float) -> float:
        sigma = float(sigma)
        if not math.isfinite(sigma) or not 0 <= sigma <= 1:
            raise ValueError("sigma must be finite and in [0, 1]")
        return sigma

    @staticmethod
    def _positive_int(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    def _components(self, x: torch.Tensor, sigma: float, cond: Optional[int]):
        """Component posterior weights and moments in the query tensor's dtype/device."""
        self._validate_x(x)
        self._validate_cond(cond)
        sigma = self._validate_sigma(sigma)
        mu, sd, pi = (v.to(x) for v in (self.means, self.stds, self.weights))
        if cond is not None:
            mu, sd = mu[cond:cond + 1], sd[cond:cond + 1]
            # Conditioning specifies the component, even if its prior mass is zero.
            pi = torch.ones(1, dtype=x.dtype, device=x.device)
        one_minus = 1.0 - sigma
        m = one_minus * mu
        c = one_minus ** 2 * sd ** 2 + sigma ** 2
        diff = x[:, None, :] - m[None, :, :]
        logr = (torch.log(pi)[None] - 0.5 * self.D * torch.log(c)[None]
                - 0.5 * diff.pow(2).sum(-1) / c[None])
        r = torch.softmax(logr, dim=1)
        return mu, sd, c, diff, r

    def posterior_x0(self, x: torch.Tensor, sigma: float, cond: Optional[int] = None
                     ) -> torch.Tensor:
        """E[x0 | x_sigma = x] under the (class-restricted) mixture."""
        mu, sd, c, diff, r = self._components(x, sigma, cond)
        one_minus = 1.0 - sigma
        gain = one_minus * sd ** 2 / c                         # [k]
        x0_j = mu[None] + gain[None, :, None] * diff           # [B, k, D]
        return (r[..., None] * x0_j).sum(1)

    def velocity(self, x: torch.Tensor, sigma: float, cond: Optional[int] = None) -> torch.Tensor:
        """E[eps - x0 | x_sigma=x], evaluated without division by sigma.

        Conditioning the joint Gaussian gives this expression directly. It
        stays accurate near sigma=0 where (x - E[x0|x]) / sigma loses precision.
        The continuous endpoint is v(x, 0) = -x for every component.
        """
        sigma = self._validate_sigma(sigma)
        mu, sd, c, diff, r = self._components(x, sigma, cond)
        if sigma == 0:
            return -x
        gain = (sigma - (1.0 - sigma) * sd ** 2) / c
        v_j = -mu[None] + gain[None, :, None] * diff
        return (r[..., None] * v_j).sum(1)

    def error(self, x: torch.Tensor, sigma: float, cond: int) -> torch.Tensor:
        """Semantic error e = v_c - v_uncond (paper Eq. 6)."""
        return self.velocity(x, sigma, cond) - self.velocity(x, sigma, None)

    def error_jacobian(self, x: torch.Tensor, sigma: float, cond: int,
                       eps: float = 1e-3) -> torch.Tensor:
        """Finite-difference Jacobian J = d e / d x, shape [B, D, D].

        This spatial Jacobian is not by itself the input gain of the sliding
        surface: a discrete sampler also contributes its signed step size,
        next evaluation point, and the controller's memory update.
        """
        self._validate_x(x)
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and strictly positive")
        B, D = x.shape
        J = x.new_zeros(B, D, D)
        for i in range(D):
            dx = x.new_zeros(D)
            dx[i] = eps
            J[:, :, i] = (self.error(x + dx, sigma, cond) - self.error(x - dx, sigma, cond)) / (2 * eps)
        return J

    # --------------------------------------------------------- sampling
    @staticmethod
    def sigma_schedule(steps: int) -> torch.Tensor:
        """sigma_0 = 1 > ... > sigma_steps = 0."""
        GaussianMixtureFlow._positive_int(steps, "steps")
        return torch.linspace(1.0, 0.0, steps + 1)

    def sample(self, n: int, cond: int, w: float,
               controller: Optional[SlidingModeGuidance] = None,
               steps: int = 30, seed: int = 0) -> Tuple[torch.Tensor, SampleTrace]:
        """Guided Euler sampling. Returns final samples [n, D] and a trace."""
        self._positive_int(n, "n")
        self._validate_cond(cond)
        if cond is None:
            raise ValueError("cond must identify a class")
        if not math.isfinite(w):
            raise ValueError("w must be finite")
        sig = self.sigma_schedule(steps)
        g = torch.Generator(device=self.means.device).manual_seed(seed)
        x = torch.randn(n, self.D, generator=g, device=self.means.device, dtype=self.means.dtype)
        if controller is not None:
            controller.reset()
        trace = SampleTrace()
        for i in range(steps):
            s_now, s_next = float(sig[i]), float(sig[i + 1])
            v_u = self.velocity(x, s_now, None)
            e = self.velocity(x, s_now, cond) - v_u
            e_app = controller.correct(e, w=w) if controller is not None else e
            x = x + (s_next - s_now) * (v_u + w * e_app)
            trace.sigmas.append(s_now)
            if controller is not None:
                trace.steps.append(controller.history[-1])
        return x, trace

    def true_class_samples(self, n: int, cond: int, seed: int = 1) -> torch.Tensor:
        self._positive_int(n, "n")
        self._validate_cond(cond)
        if cond is None:
            raise ValueError("cond must identify a class")
        g = torch.Generator(device=self.means.device).manual_seed(seed)
        return self.means[cond] + self.stds[cond] * torch.randn(
            n, self.D, generator=g, device=self.means.device, dtype=self.means.dtype)

    # ---------------------------------------------------------- metrics
    def class_log_posterior(self, x: torch.Tensor) -> torch.Tensor:
        """log p(class | x) at sigma = 0, shape [B, n_classes]."""
        self._validate_x(x)
        means, stds, weights = (v.to(x) for v in (self.means, self.stds, self.weights))
        diff = x[:, None, :] - means[None]
        ll = (torch.log(weights)[None]
              - self.D * torch.log(stds)[None]
              - 0.5 * diff.pow(2).sum(-1) / stds[None] ** 2)
        return ll - torch.logsumexp(ll, dim=1, keepdim=True)

    def class_accuracy(self, x: torch.Tensor, cond: int) -> float:
        """Fraction of samples whose posterior argmax is the requested class."""
        self._validate_cond(cond)
        if cond is None:
            raise ValueError("cond must identify a class")
        return float((self.class_log_posterior(x).argmax(1) == cond).float().mean())

    def class_confidence(self, x: torch.Tensor, cond: int) -> float:
        """Mean posterior p(cond | x). The continuous analogue of a CLIP score:
        it keeps rising as samples move into the class core, long after the
        argmax accuracy has saturated at 1."""
        self._validate_cond(cond)
        if cond is None:
            raise ValueError("cond must identify a class")
        return float(self.class_log_posterior(x)[:, cond].exp().mean())

    @staticmethod
    def _sqrtm_psd(a: torch.Tensor) -> torch.Tensor:
        vals, vecs = torch.linalg.eigh(a)
        return (vecs * vals.clamp_min(0).sqrt()[None]) @ vecs.T

    def frechet_distance(self, x: torch.Tensor, cond: int) -> float:
        """Wasserstein-2 distance between Gaussian sample and target moments.

        The return value is the square root of the FID expression, preserving
        this repository's historical scale. Moment matching cannot detect
        non-Gaussian errors in the sample law. Requires at least two samples.
        """
        self._validate_x(x, min_samples=2)
        self._validate_cond(cond)
        if cond is None:
            raise ValueError("cond must identify a class")
        x = x.to(torch.float64)
        m1 = x.mean(0)
        xc = x - m1
        S1 = xc.T @ xc / (x.shape[0] - 1)
        m2, sd = self.means[cond].to(x), self.stds[cond].to(x)
        # S2 = sd**2 I, so its covariance contribution reduces to this sum.
        # The nonnegative form avoids subtracting nearly equal large traces.
        eig = torch.linalg.eigvalsh(S1).clamp_min(0)
        d2 = (m1 - m2).pow(2).sum() + (eig.sqrt() - sd).pow(2).sum()
        return float(d2.sqrt())


def ring_mixture(k: int = 8, radius: float = 4.0, std: float = 1.5, dim: int = 2,
                 seed: int = 0) -> GaussianMixtureFlow:
    """k equally weighted Gaussians, one per class. dim=2: on a circle.
    dim>2: random directions at the given radius. The std is chosen so that
    neighbouring classes overlap and guidance has something to do."""
    GaussianMixtureFlow._positive_int(k, "k")
    GaussianMixtureFlow._positive_int(dim, "dim")
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    if not math.isfinite(std) or std <= 0:
        raise ValueError("std must be finite and strictly positive")
    if dim == 2:
        ang = torch.arange(k) * 2 * math.pi / k
        means = radius * torch.stack([ang.cos(), ang.sin()], 1)
    else:
        g = torch.Generator().manual_seed(seed)
        means = torch.randn(k, dim, generator=g)
        means = radius * means / means.norm(dim=1, keepdim=True)
    return GaussianMixtureFlow(means, torch.full((k,), std))
