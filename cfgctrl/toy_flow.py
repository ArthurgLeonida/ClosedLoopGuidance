"""Analytic flow-matching plant: a Gaussian mixture with closed-form velocity.

Why this exists. SD3.5 / Flux need a GPU and have no ground truth: you cannot
say whether a guided sample is "right", only whether a reward model likes it.
A Gaussian mixture has an exact conditional velocity field, an exact
unconditional one, and an exact target distribution, so every quantity the
CFG-Ctrl paper reasons about (e, s, the Jacobian of e, the reaching condition)
can be *measured* rather than assumed, on a laptop, in seconds.

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
    frechet_distance  Frechet distance to the true class law   (the FID formula)
    class_confidence  mean posterior p(class | x) at sigma = 0 (a toy CLIP score)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import math
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
        means = torch.as_tensor(means, dtype=torch.float32)
        stds = torch.as_tensor(stds, dtype=torch.float32)
        if means.dim() != 2:
            raise ValueError("means must be [K, D]")
        K, D = means.shape
        if stds.shape != (K,):
            raise ValueError("stds must be [K] (isotropic per component)")
        weights = torch.ones(K) / K if weights is None else torch.as_tensor(weights, dtype=torch.float32)
        self.means, self.stds = means, stds
        self.weights = weights / weights.sum()
        self.K, self.D = K, D
        self.n_classes = K                      # one Gaussian per class

    # ----------------------------------------------------------- fields
    def posterior_x0(self, x: torch.Tensor, sigma: float, cond: Optional[int] = None
                     ) -> torch.Tensor:
        """E[x0 | x_sigma = x] under the (class-restricted) mixture."""
        comps = torch.arange(self.K) if cond is None else torch.tensor([cond])
        mu, sd, pi = self.means[comps], self.stds[comps], self.weights[comps]
        one_minus = 1.0 - sigma
        m = one_minus * mu                                     # [k, D]
        c = one_minus ** 2 * sd ** 2 + sigma ** 2              # [k]
        diff = x[:, None, :] - m[None, :, :]                   # [B, k, D]
        logr = (torch.log(pi)[None] - 0.5 * self.D * torch.log(c)[None]
                - 0.5 * diff.pow(2).sum(-1) / c[None])
        r = torch.softmax(logr, dim=1)
        gain = one_minus * sd ** 2 / c                         # [k]
        x0_j = mu[None] + gain[None, :, None] * diff           # [B, k, D]
        return (r[..., None] * x0_j).sum(1)

    def velocity(self, x: torch.Tensor, sigma: float, cond: Optional[int] = None) -> torch.Tensor:
        sigma = max(float(sigma), 1e-6)
        return (x - self.posterior_x0(x, sigma, cond)) / sigma

    def error(self, x: torch.Tensor, sigma: float, cond: int) -> torch.Tensor:
        """Semantic error e = v_c - v_uncond (paper Eq. 6)."""
        return self.velocity(x, sigma, cond) - self.velocity(x, sigma, None)

    def error_jacobian(self, x: torch.Tensor, sigma: float, cond: int,
                       eps: float = 1e-3) -> torch.Tensor:
        """Finite-difference Jacobian J = d e / d x, shape [B, D, D].

        The paper's effective gain matrix is Gamma = w * J (Table 4), and its
        Assumption 2 needs Gamma ~ w*I. This is what lets you check that.
        """
        B, D = x.shape
        J = torch.zeros(B, D, D)
        for i in range(D):
            dx = torch.zeros(D)
            dx[i] = eps
            J[:, :, i] = (self.error(x + dx, sigma, cond) - self.error(x - dx, sigma, cond)) / (2 * eps)
        return J

    # --------------------------------------------------------- sampling
    @staticmethod
    def sigma_schedule(steps: int) -> torch.Tensor:
        """sigma_0 = 1 > ... > sigma_steps = 0."""
        return torch.linspace(1.0, 0.0, steps + 1)

    def sample(self, n: int, cond: int, w: float,
               controller: Optional[SlidingModeGuidance] = None,
               steps: int = 30, seed: int = 0) -> Tuple[torch.Tensor, SampleTrace]:
        """Guided Euler sampling. Returns final samples [n, D] and a trace."""
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, self.D, generator=g)
        if controller is not None:
            controller.reset()
        sig = self.sigma_schedule(steps)
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
        g = torch.Generator().manual_seed(seed)
        return self.means[cond] + self.stds[cond] * torch.randn(n, self.D, generator=g)

    # ---------------------------------------------------------- metrics
    def class_log_posterior(self, x: torch.Tensor) -> torch.Tensor:
        """log p(class | x) at sigma = 0, shape [B, n_classes]."""
        diff = x[:, None, :] - self.means[None]
        ll = (torch.log(self.weights)[None]
              - self.D * torch.log(self.stds)[None]
              - 0.5 * diff.pow(2).sum(-1) / self.stds[None] ** 2)
        return ll - torch.logsumexp(ll, dim=1, keepdim=True)

    def class_accuracy(self, x: torch.Tensor, cond: int) -> float:
        """Fraction of samples whose posterior argmax is the requested class."""
        return float((self.class_log_posterior(x).argmax(1) == cond).float().mean())

    def class_confidence(self, x: torch.Tensor, cond: int) -> float:
        """Mean posterior p(cond | x). The continuous analogue of a CLIP score:
        it keeps rising as samples move into the class core, long after the
        argmax accuracy has saturated at 1."""
        return float(self.class_log_posterior(x)[:, cond].exp().mean())

    @staticmethod
    def _sqrtm_psd(a: torch.Tensor) -> torch.Tensor:
        vals, vecs = torch.linalg.eigh(a)
        return (vecs * vals.clamp_min(0).sqrt()[None]) @ vecs.T

    def frechet_distance(self, x: torch.Tensor, cond: int) -> float:
        """Frechet (2-Wasserstein between Gaussians) distance between the sample
        moments and the true class moments. Identical formula to FID."""
        m1 = x.mean(0)
        xc = x - m1
        S1 = xc.T @ xc / (x.shape[0] - 1)
        m2 = self.means[cond]
        S2 = self.stds[cond] ** 2 * torch.eye(self.D)
        r2 = self._sqrtm_psd(S2)
        cross = self._sqrtm_psd(r2 @ S1 @ r2)
        d2 = (m1 - m2).pow(2).sum() + torch.trace(S1) + torch.trace(S2) - 2 * torch.trace(cross)
        return float(d2.clamp_min(0).sqrt())


def ring_mixture(k: int = 8, radius: float = 4.0, std: float = 1.5, dim: int = 2,
                 seed: int = 0) -> GaussianMixtureFlow:
    """k equally weighted Gaussians, one per class. dim=2: on a circle.
    dim>2: random directions at the given radius. The std is chosen so that
    neighbouring classes overlap and guidance has something to do."""
    if dim == 2:
        ang = torch.arange(k) * 2 * math.pi / k
        means = radius * torch.stack([ang.cos(), ang.sin()], 1)
    else:
        g = torch.Generator().manual_seed(seed)
        means = torch.randn(k, dim, generator=g)
        means = radius * means / means.norm(dim=1, keepdim=True)
    return GaussianMixtureFlow(means, torch.full((k,), std))
