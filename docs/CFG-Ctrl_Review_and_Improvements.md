# CFG-Ctrl / SMC-CFG: what it really does, and how to improve it

**A control-engineering review of** *CFG-Ctrl: Control-Based Classifier-Free
Diffusion Guidance* (Wang, Liu, Chi, Liu, Xue, Duan; CVPR 2026 highlight;
arXiv 2603.03281), **written for someone who studied control and wants the
concepts back.** Every concept the paper uses is re-derived in a grey
"refresher" block the first time it appears, then applied to the paper.

The code that goes with this document lives in `cfgctrl/` (see the README).
Everything quantitative below comes from `experiments/toy_smc_cfg.py`, which
runs on a CPU in a few minutes on a plant with ground truth. Nothing here was
run on SD3.5 or Flux: there is no GPU on this machine. Where a claim is only a
hypothesis about real models it is marked as such.

---

## 0. Summary

**What the paper does.** It writes the flow-matching sampler as a controlled
ODE, shows that classifier-free guidance (CFG) is a proportional controller
with gain $w$ acting on the "semantic error" $e = v_\theta(x,t,c) -
v_\theta(x,t,\varnothing)$, and replaces the fixed gain by a sliding-mode
correction: $s = \dot e + \lambda e$, $\Delta e = -k\,\mathrm{sign}(s)$, applied
velocity $\hat v = v_\varnothing + w\,(e + \Delta e)$. It proves finite-time
convergence of $s$ under two assumptions, and reports better FID / CLIP /
human-preference numbers than CFG on SD3.5, Flux-dev and Qwen-Image at the
default guidance scale of each model, plus much better robustness at large $w$.

**What I found by implementing it and measuring it on an analytic plant.**

1. **The discrete sliding surface is essentially $\lambda e$.** With
   $\lambda = 6$ and 30 steps, $s_t = e_t + (\lambda - 1)\,\hat e_{t-1}$ and
   the derivative term decides the sign of $s$ in about 1–2 % of the elements.
   SMC-CFG is, to first order, a **per-element sign-shrink of the guidance
   vector**: $e \leftarrow e - k\,\mathrm{sign}(e)$. That also explains why
   the paper's own $\lambda$ ablation is flat.
2. **It chatters by construction and the chatter is self-inflicted.** The
   authors store the *corrected* error as memory, so once $|e_i| < k$ the
   surface becomes $s \approx (\lambda - 1)\,\Delta e_{\text{prev}}$ and the
   sign alternates every step: the residual $\|s\|$ never goes to zero (it
   sits at $(\lambda-1)k$), and 100 % of the elements flip sign in the last
   ten steps. With $\lambda < 1$ (the authors' README default is 0.05) the same
   mechanism *locks in a constant bias* of size $k$ instead.
3. **The feedback loop the proof needs does not exist within a sampling
   run.** The one-step gain from the correction to the next *measured* error
   is $-\Delta\sigma\, w\, \partial e/\partial x$; measured along real
   trajectories it is **negative** (more guidance makes the measured error
   smaller, the paper's own premise) and **small** (median per-step authority
   0.005–0.05, peaks near 0.3 for a few samples mid-trajectory, against
   $k = 0.1$). Assumption 2 ($\Gamma \approx +wI$) never holds; the reaching
   condition holds for 3–16 % of the (sample, step) pairs. The law works as
   **open-loop shaping of the guidance direction**, not as a sliding mode, and
   the switching direction the Lyapunov argument actually requires over-guides.
4. **The sliding variable has relative degree zero** with respect to the
   correction (the correction changes $\dot e$ instantly, so $s$ depends on it
   algebraically). Textbook reaching-law analysis, and higher-order sliding
   mode, assume relative degree one. A super-twisting controller on this $s$
   settles into a two-step limit cycle of amplitude $k_1^2$ (unit-tested).
5. **Where the method helps, it helps as an L1-type shrinkage of guidance**,
   which lowers the effective guidance where the two branches nearly agree.
   On the toy plant, at matched alignment, it is *strictly dominated* by CFG
   at $w = 1$ (worse fidelity **and** worse alignment), roughly 1–9 % worse
   for $w \le 2$, and 2–8 % better for $w \ge 3$ in 32-D (0–2 % in 2-D),
   where the error vector is anisotropic. So most of the gain at a fixed $w$
   is "a slightly smaller effective $w$"; a smaller part is genuine and grows
   with the sparsity of $e$. The paper's Table 2 compares at one $w$ per model, which cannot
   separate the two.
6. **At $w = 1$ the law is not the conditional sampler.** The correction is
   applied to $v_c = v_\varnothing + e$ itself, so with $k > 0$ the $w = 1$
   sampler is biased toward the unconditional law (toy: fidelity 0.16 vs
   0.10 for CFG, alignment below the true class law). Shrinking only the
   extrapolation $(w - 1)\,e$ removes the bias exactly.

**What to do about it (ranked; each is developed in §4).**

1. Replace $\mathrm{sign}(s)$ by $\mathrm{sat}(s/\phi)$ with $\phi = k\lambda$.
   That is a classical boundary layer *and* it turns the law into exact
   soft-thresholding of $e$ by $k$: identical results at the paper's $k$,
   $\|s\| \to 0$ instead of a plateau, and a flat 0.03 of full chatter at
   every $k$ instead of 0.39 rising to 0.83. Zero cost.
2. Shrink the **extrapolation** $(w-1)\,e$, not the conditional prediction:
   $\hat v = v_c + (w-1)(e + \Delta e)$. Exactly CFG at $w = 1$ for any $k$,
   which the paper's law is not. On the toy it is the only law that never
   falls more than 1.2 % below the CFG Pareto curve anywhere, and it keeps the
   whole high-$w$ gain (E1).
3. Store the **measured** error as memory, not the corrected one.
4. Make $k$ **scale-free** (a fraction of $\mathrm{rms}(e)$) so one value
   transfers across models (the paper needs 0.1 for SD3.5/Qwen and 0.7 for
   Flux).
5. Reframe the law as what it is (adaptive shrinkage) and then design it
   properly: a **noise-adaptive threshold** (median absolute deviation of $e$)
   is the estimation-theoretic version of the same idea.
6. **Evaluate on Pareto curves over $w$ and across seeds**, log $\|e\|$,
   $\|s\|$ and switching activity, and include the "null model" (CFG plus a
   memoryless soft-threshold) as a baseline.

Measured and discarded: a time-scaled sliding surface, super-twisting,
adaptive gain, unit-vector switching and the flipped (Lyapunov-consistent)
switching direction. None improved on the plain shrink on the Pareto curve, so
none is in the implementation; the argument for each is kept where it matters
in §3 and §4, and the code that measured them is in this repository's git
history.

---

## 1. The paper in control language

### 1.1 The plant

Flow matching (the model class behind SD3, Flux and Qwen-Image) learns a
velocity field $v_\theta(x, \sigma, c)$ such that the ODE

$$\frac{dx}{d\sigma} = v_\theta(x, \sigma, c), \qquad x_{\sigma=1} \sim \mathcal N(0, I)$$

integrated from $\sigma = 1$ (pure noise) down to $\sigma = 0$ produces a
sample from $p(x \mid c)$. In practice you take 20–50 Euler steps
$x_{\sigma'} = x_\sigma + (\sigma' - \sigma)\,\hat v$, with $\sigma' < \sigma$.
The model is queried twice per step: with the prompt ($c$) and without
($\varnothing$).

> **Refresher: plant, state, input, measurement.** A *plant* is the thing you
> want to influence; its *state* is what you need to know to predict its
> future; the *input* is what you can change; the *measurement* is what you
> can observe. Here the plant is the ODE integrator wrapped around the
> network; the state is the latent $x_\sigma$; the input is an additive
> velocity $u$; and the measurement is $e = v_c - v_\varnothing$, which the
> network gives you for free at every step. Note the unusual feature: the
> network is *both* the plant dynamics and the sensor.

The paper writes this as the control-affine system $\dot x = v_\theta(x,t) +
G\,u_t$ with $G = I$ (Eq. 8–9): you may add any velocity you like.

### 1.2 CFG is a proportional controller

Classifier-free guidance uses the velocity

$$\hat v = v_\varnothing + w\,(v_c - v_\varnothing) = v_\varnothing + w\,e, \qquad w \ge 1.$$

Writing $u_t = K_t\,\Pi_t(e_t)$ (Eq. 10: a *gain schedule* $K_t$ and a
*direction operator* $\Pi_t$), CFG is $K_t = w$, $\Pi_t = I$. That is the whole
"CFG is P-control" statement (Eq. 11–13).

> **Refresher: P-control and why fixed gain hurts.** A proportional
> controller applies an input proportional to the error, $u = K e$. Bigger
> $K$ means a faster, more decisive response and a smaller steady-state error,
> but also two costs: (i) everything in the error signal is amplified by $K$,
> including measurement noise and model error, and (ii) with any lag in the
> loop, large $K$ overshoots and eventually oscillates. In images the
> overshoot is *oversaturation*: colours pushed past plausible, textures
> over-sharpened, structure warped. The paper's Fig. 1 (left) is exactly the
> phase portrait of an under-damped P-loop.

### 1.3 The existing variants as controllers (Table 1 of the paper)

| method | what it changes | control name |
|---|---|---|
| weight schedulers ($w(t)$) | $K_t$ varies with $t$ | gain scheduling |
| APG | splits $e$ into components parallel / orthogonal to $v_c$ and down-weights the parallel one | projection-based feedback (structured $\Pi_t$) |
| CFG-Zero$^\star$ | projects onto $v_\varnothing$ and rescales, plus zero-guidance warm-up | projection-based feedback |
| Rectified-CFG++ | uses the error at a *predicted* half-step state | model predictive control (one-step lookahead) |
| SMC-CFG (this paper) | switches the error by $\pm k$ according to $\mathrm{sign}(\dot e + \lambda e)$ | sliding mode control |

> **Refresher: gain scheduling.** Choosing the gain as a known function of an
> operating variable (here the noise level $t$). It is open-loop adaptation:
> the schedule does not look at how the loop is doing.
>
> **Refresher: MPC.** Model predictive control predicts where the state will
> be over a short horizon under a candidate input, optimises the input, applies
> the first piece, and repeats. Rectified-CFG++ is the degenerate case of a
> one-step prediction with no optimisation, but the idea is the same:
> anticipate rather than react.

### 1.4 The reference model: how should the error evolve?

Section 3.2 observes that $e$ shrinks as denoising progresses, and *defines*
the ideal behaviour as first-order exponential decay:

$$\dot e = -\lambda_0\, e \quad\Longrightarrow\quad e(t) = e(T)\,e^{-\lambda_0 t}.$$

> **Refresher: reference model.** Instead of saying "make the error zero", you
> say "make the error follow this trajectory". The reference model
> $\dot e = -\lambda e$ has time constant $1/\lambda$: the error should fall by
> a factor $e \approx 2.72$ every $1/\lambda$ time units. Choosing $\lambda$ is
> choosing how fast you demand the error to disappear. In a sampling run that
> lasts one unit of flow time (from $\sigma = 1$ to $0$), $\lambda = 6$ asks
> for a decay by $e^{-6} \approx 0.25\,\%$ over the run, which is aggressive but
> not absurd. Keep this number in mind for §3.5.

---

## 2. SMC-CFG step by step

### 2.1 The sliding variable

$$s(t) = \dot e(t) + \lambda\, e(t). \tag{Eq. 19}$$

If $s \equiv 0$ then $e$ obeys the reference model exactly. The set
$\{s = 0\}$ is the *sliding surface* (the paper calls it the semantic sliding
manifold).

> **Refresher: sliding mode control in one paragraph.** Pick a function $s$
> of the state such that "$s = 0$" is the behaviour you want (here: error
> decaying like the reference model). Design a *discontinuous* input that
> always pushes $s$ toward zero: $u = -k\,\mathrm{sign}(s)$. Two phases follow.
> In the *reaching phase* $s$ goes to zero in finite time; in the *sliding
> phase* the state stays on $s = 0$ and evolves according to the reduced,
> lower-order dynamics you designed (here: $\dot e = -\lambda e$) regardless
> of whatever bounded disturbance the plant throws at you. That last property,
> insensitivity to *matched* disturbances (those entering through the same
> channel as the input), is why SMC is called robust. The price is
> *chattering*: in any real (sampled, lagged) system the state cannot stay on
> the surface, it crosses it back and forth and the input switches at the
> sampling rate.

### 2.2 The switching law and the Lyapunov argument

$$\Delta e = -k\,\mathrm{sign}(s), \qquad \hat v = v_\varnothing + w\,(e + \Delta e). \tag{Eq. 25, Alg. 1}$$

Note the peculiarity: the "input" is applied *to the error signal itself*,
which is then multiplied by $w$. The velocity actually added to the flow is
$w\,\Delta e = -wk\,\mathrm{sign}(s)$.

The paper takes $V = \tfrac12 \|s\|^2$, assumes $\dot s = \Phi + \Gamma\,\Delta e$
with $\|\Phi\| \le \delta$ and $\Gamma = wI + \Delta\Gamma$,
$\|\Delta\Gamma\| \le \rho$, and gets

$$\dot V = s^\top \Phi - k\, s^\top \Gamma\,\mathrm{sign}(s) \le \|s\|\,\big(\delta - k(w - \rho\sqrt D)\big) = -\eta\|s\|,$$

so that $\|s(t)\| \le \|s(0)\| - \eta t$ and $s$ reaches zero within
$\|s(0)\|/\eta$ (Eq. 26–28, 40–44; Theorem 1 needs
$k > \delta/(w - \rho\sqrt D) + \epsilon$).

> **Refresher: Lyapunov functions and finite-time convergence.** A Lyapunov
> function is an "energy" $V \ge 0$ that is zero exactly at the target and
> that you can show decreases along every trajectory. If $\dot V \le -cV$ you
> get exponential convergence (never exactly reaching zero); if $\dot V \le
> -\eta\sqrt{V}$, as here, you get *finite-time* convergence: the energy hits
> zero at a computable time. The $\mathrm{sign}$ function is what buys the
> $\sqrt V$ rate: its magnitude does not shrink as $s \to 0$. That is also
> exactly why it chatters.
>
> **The two things every SMC proof needs.** (a) The disturbance must be
> *bounded* ($\|\Phi\| \le \delta$) so that a finite $k$ can dominate it. (b)
> The input must have *authority over $\dot s$ with a known sign*:
> $s^\top \Gamma\,\mathrm{sign}(s) > 0$, i.e. pushing "against" $s$ actually
> reduces it. Assumption 2 of the paper is (b) in the strong form
> "$\Gamma$ is close to $+wI$". §3.3 tests it.

### 2.3 What the code does (Algorithm 1, and `pipeline/common_cfg_ctrl.py`)

In discrete steps, with $e(t+1)$ meaning the *previous* (noisier) step:

```
e   = v_cond - v_uncond
if prev is None: prev = e                  # first step: s = lam * e
s   = (e - prev) + lam * prev              # finite difference, NOT divided by dt
u   = -K * sign(s)                         # elementwise over the whole latent
e   = e + u
prev = e                                   # the CORRECTED error is stored
v_hat = v_uncond + w * e
```

Hyper-parameters (supplementary §7.3): grid $\lambda \in \{2,\dots,8\}$,
$k \in \{0.01,\dots,0.8\}$; chosen $\lambda = 6$ for all three models,
$k = 0.1$ (SD3.5, Qwen-Image) and $k = 0.7$ (Flux-dev). The GitHub README's
defaults are $\lambda = 0.05$, $k = 0.3$; the SD3 example script uses
$\lambda = 5$, $k = 0.2$. These are not small differences (§3.2). All
experiments use 30 sampling steps, so $\Delta\sigma \approx 1/30$.

### 2.4 One element, by hand

Take $\lambda = 6$, $k = 0.1$.

* Early, a strong component: $\hat e_{\text{prev}} = 0.30$, $e_t = 0.27$ (it is
  decaying 10 % per step). $s = (0.27 - 0.30) + 6 \cdot 0.30 = 1.77 > 0$, so
  $\Delta e = -0.1$ and the applied error is $0.17$: this element's guidance
  is cut by 37 %. Notice that the derivative term ($-0.03$) is irrelevant next
  to $\lambda \hat e_{\text{prev}} = 1.8$.
* Late, a weak component: $\hat e_{\text{prev}} = 0.06$, $e_t = 0.05$.
  $s = -0.01 + 0.36 = 0.35 > 0$, $\Delta e = -0.1$, applied error $-0.05$:
  the **sign flipped**; this element now guides the wrong way with magnitude
  $0.05$. The stored memory is $-0.05$. Next step, measured $e \approx 0.045$:
  $s = (0.045 + 0.05) + 6 \cdot (-0.05) = -0.205 < 0$, $\Delta e = +0.1$,
  applied error $0.145$. The applied guidance for this element now alternates
  $-0.05, +0.145, \dots$: chatter of amplitude $k$ around a value that should
  be $\approx 0$.
* Same weak component with a boundary layer $\phi = k\lambda = 0.6$ (§4.1):
  $|s| = 0.35 < \phi$, so $\Delta e = -k \cdot s/\phi = -0.058$ and the applied
  error is $-0.008 \approx 0$. That is soft-thresholding: everything below $k$
  in magnitude is zeroed, everything above is shrunk by $k$.

---

## 3. What the mathematics actually does: a critique with measurements

The measurements below come from an analytic plant (`cfgctrl/toy_flow.py`):
a Gaussian mixture in $D$ dimensions, for which the flow-matching velocity
field, its conditional version, the Jacobian $\partial e/\partial x$ and the
true class distribution are all closed-form. "Fidelity" is the Fréchet
distance between the sample moments and the true class law (the FID formula
with exact statistics); "alignment" is the posterior probability
$p(c \mid x)$ of the requested class (a continuous CLIP-like score). The plant
is far simpler than a text-to-image model; use it for the structural
statements (signs, magnitudes, what a law reduces to), not for absolute
quality claims.

### 3.1 The surface is $\lambda e$; the law is a sign-shrink

$$s_t = (e_t - \hat e_{t-1}) + \lambda \hat e_{t-1} = e_t + (\lambda - 1)\,\hat e_{t-1}.$$

For consecutive steps $e_t \approx \hat e_{t-1}$, so $s_t \approx \lambda e_t$
unless an element changes by more than $\lambda$ times its own value in one
step. Measured on the toy plant at $\lambda = 6$: the derivative term decides
$\mathrm{sign}(s)$ for about **1 % of the elements** averaged over a run (E2,
E2). So

$$e_{\text{applied}} \approx e - k\,\mathrm{sign}(e),$$

a per-element sign-shrink of the guidance vector, and the "sliding mode" adds
nothing visible over the memoryless map. This is consistent with the paper's
own ablation, where $\lambda = 3, 4, 5, 6$ give FID 26.19, 26.01, 25.95, 26.14
(Table 3): $\lambda$ barely matters because $s \approx \lambda e$ for any
$\lambda \gg 1$, and only the sign of $s$ is used.

### 3.2 Chattering, and the corrected-error memory

Once $|e_i| < k$, the sign-shrink flips the element's sign (§2.4). With the
corrected error stored as memory, the surface becomes
$s_t \approx e_t + (\lambda - 1)\,\Delta e_{t-1}$, which for small $e$ is
dominated by the previous correction: $\mathrm{sign}(s_t) = \mathrm{sign}(\Delta e_{t-1})$, so
$\Delta e_t = -\Delta e_{t-1}$. The correction alternates every step. Measured
(E2): in the last ten steps **100 % of the elements flip** and the residual
$\mathrm{rms}(s)$ plateaus at $(\lambda - 1)k = 0.5$ instead of going to zero.
The sliding phase is never reached; what looks like "convergence onto the
manifold" in Fig. 1 is the chattering band.

Two corollaries.

* **Store the measured error and the plateau disappears.** With measured
  memory the surface has no self-reference; with the boundary layer of §4.1
  $\mathrm{rms}(s)$ falls to $\sim 10^{-3}$ (E2).
* **With $\lambda < 1$ the same loop locks a bias.** The coefficient
  $(\lambda - 1)$ becomes negative, $\mathrm{sign}(s_t) = -\mathrm{sign}(\Delta e_{t-1})$,
  so $\Delta e_t = \Delta e_{t-1}$: whatever the correction was when $e$
  became small is applied, unchanged, for the rest of the run. The GitHub
  README's default $\lambda = 0.05$ is in this regime. The same happens with
  a time-scaled surface, where the coefficient is
  $(\lambda - 1/\Delta\sigma) < 0$.

> **Refresher: chattering and the quasi-sliding-mode band.** In continuous
> time an ideal relay switches infinitely fast on the surface. Any sampled
> implementation switches at most once per step, so the state overshoots the
> surface by up to one step's worth of control authority and comes back:
> a zig-zag of amplitude proportional to $k$ (Gao, Wang & Homaifa 1995 call
> the resulting strip the quasi-sliding-mode band). Chatter is harmless when
> the plant low-pass-filters it (a motor's inertia) and harmful when it does
> not (here: every element of the velocity field jitters by $\pm wk$, and the
> network sees that jitter at the next step).

### 3.3 The loop gain has the wrong sign and is negligible

The correction $\Delta e_t$ changes the next latent by
$-\Delta\sigma\, w\, \Delta e_t$ (Euler step), so it changes the next *measured*
error by

$$\Delta e^{\text{meas}}_{t+1} \approx -\Delta\sigma\, w\, J\,\Delta e_t, \qquad J = \frac{\partial e}{\partial x}.$$

This is the paper's $\Gamma$ (Table 4: $\Gamma = w\nabla_x(v_c - v_\varnothing)$)
with the sampling direction and the step size made explicit. The paper needs
the symmetric part of $\Gamma$ to be positive and close to $wI$. Measured
along trajectories with finite differences (E4):

* the eigenvalues of $\mathrm{sym}(-J)$ are **negative** (down to about $-2$
  around $\sigma \approx 0.8$, maximum $+0.04$) and go to zero late;
* **Assumption 2 holds for 0 % of the (sample, step) pairs**;
* the reaching condition $s^\top(-J)\,\mathrm{sign}(s) > 0$ holds for **3 %
  (2-D) to 16 % (8-D)** of them, i.e. the paper's switching direction
  *increases* $\|s\|$ most of the time;
* the per-step loop gain $\Delta\sigma\, w\, \|J\|$ has median 0.05 (2-D) and
  0.006 (8-D), with peaks near 0.3 for a few samples mid-trajectory, against
  $k = 0.1$.

The sign is not an artefact of the toy. If guidance does its job, pushing
harder toward the conditional manifold makes $v_c$ and $v_\varnothing$ agree
*sooner*, so the measured error at the next step is *smaller*: $\partial e_{t+1}
/ \partial \Delta e_t$ is negative along $e$. That is precisely the paper's own
motivation in §3.2. So the Lyapunov argument, taken literally, requires the
switching direction $+k\,\mathrm{sign}(s)$ (push harder when the error decays
too slowly). Measured, that direction does what you would expect: it
over-guides, with a higher Fréchet distance at every $w$ and a Pareto position
above CFG's curve in 32-D. It is not in the implementation.

The magnitude is the second half of the story. With a median authority of
$10^{-2}$ per step against a correction of $0.1$, the correction's effect on
the *measured* error is typically one to two orders of magnitude smaller than
the correction itself; it approaches $k$ only for a few samples around
$\sigma \approx 0.8$, exactly where the eigenvalues are most negative, i.e.
where the loop is most strongly *anti*-stabilising in the paper's sense.
Over 30 steps the loop cannot move $s$ anywhere. What actually
matters is the feed-through: the applied velocity is $v_\varnothing + w(e +
\Delta e)$, and $\Delta e$ acts as **open-loop shaping of the guidance
direction**, computed from a measurement but not closing a loop in any
sense that a Lyapunov function would describe.

> **Refresher: loop gain and why it decides everything.** In a feedback loop
> the *loop gain* is the product of all gains around the loop: how much a
> change in the input comes back as a change in the measured error one trip
> later. If it is $\ll 1$ per step, feedback is a metaphor: the controller's
> output barely influences what it will measure next. If it is $O(1)$, the
> loop can regulate, and also oscillate. In diffusion sampling the trip
> includes a factor $\Delta\sigma \approx 1/30$ and a Jacobian of the
> network, so the product is of order $10^{-2}$. Contrast a plant that
> *integrates* its input — score distillation, where a persistent state is
> optimised over thousands of iterations — where the accumulated effect of the
> input on the measurement is of order one and the control vocabulary
> (integral action, anti-windup, the sensitivity function) starts to earn its
> keep. Sampling is not that plant.

### 3.4 Relative degree

$s = \dot e + \lambda e$ and the correction changes $\dot e$ instantaneously
($\dot e = J\dot x = J(v_\varnothing + w(e + \Delta e))$). So $s$ depends on
$\Delta e$ *algebraically*: the relative degree of $s$ with respect to the
input is zero. The paper's Eq. 23, $\dot s = \Phi_s + \Gamma_s \Delta e$,
assumes relative degree one (the input drives the *derivative* of $s$).
Differentiating the true $s$ would produce $\Delta \dot e$, the derivative of a
discontinuous signal.

> **Refresher: relative degree.** The number of times you must differentiate
> the output before the input appears. Classical SMC wants $s$ of relative
> degree one: then $u = -k\,\mathrm{sign}(s)$ drives $\dot s$ and finite-time
> reaching follows. With relative degree zero the input sets $s$ directly;
> the right tool is then an *equivalent control* $u_{eq}$ that solves
> $s = 0$ algebraically, plus (in discrete time) a reaching law that
> prescribes $s_{k+1}$ from $s_k$ (Gao et al.). Higher-order sliding mode
> (super-twisting, Levant 1993) needs relative degree one and a Lipschitz
> disturbance derivative; applied to a relative-degree-zero $s$ it produces
> the two-cycle $|s_{t+1}| = k_1\sqrt{|s_t|} \Rightarrow |s| = k_1^2$, which is
> exactly what was observed when it was tried here, so super-twisting is not
> in the implementation.

### 3.5 Units: $\lambda$ has none, and the reference is unreachable

The finite difference $e_t - \hat e_{t-1} \approx \Delta\sigma\,\dot e$, so

$$s_t \approx \Delta\sigma\Big(\dot e + \tfrac{\lambda}{\Delta\sigma}\, e\Big) = \tfrac{1}{30}\,(\dot e + 180\, e).$$

The surface actually encoded is $\dot e = -180\,e$: a time constant of
$1/180$ of the run, about a sixth of one step. No trajectory can follow that,
which is another way of saying the sliding phase is unreachable and the law
is a sign-shrink. It also means the paper's $\lambda = 6$ and the README's
$\lambda = 0.05$ are not "the same surface at different speeds" but two
different regimes (§3.2), and that $\lambda$ does not transfer across step
count in any principled way — though it happens not to matter, because the
derivative term is irrelevant either way.

### 3.6 Dimension: the $\sqrt D$ in Assumption 2

The correction is applied element-wise, so its norm is $k\sqrt D$. For an SD3
latent at $1024^2$ ($16 \times 128 \times 128 = 262\,144$ elements),
$\sqrt D = 512$ and $k = 0.1$ gives a correction of norm $51$, comparable to
$\|e\|$ itself. Theorem 1 then needs $w > \rho\sqrt D$: for $w = 7.5$ the
Jacobian's anisotropic part must be below $0.015$ in spectral norm, relative
to an isotropic part that should be $w$. Nothing about a transformer's
Jacobian suggests that. A per-sample unit-vector switching law,
$\Delta e = -k\, s/\|s\|$, removes the $\sqrt D$ from the analysis, but it is
a much gentler law — one scalar shrink per image instead of one per element —
and it loses the per-element attenuation that makes §4.1 work. Theoretical
tidiness only; it is not in the implementation.

### 3.7 So why does it improve images?

The honest answer is a hypothesis, because the plant that shows FID gains is
not the one I can run. Two mechanisms are consistent with everything above.

* **Adaptive guidance attenuation.** $e - k\,\mathrm{sign}(e)$ equals
  $e\,(1 - k/|e_i|)$ per element: a **spatially adaptive guidance scale**
  $w_{\text{eff},i} = w\,(1 - k/|e_i|)$, close to $w$ where the two branches
  disagree strongly (semantic content) and much smaller where they nearly
  agree (background, texture, noise). That is a sensible thing to do and it
  reduces oversaturation. It is also roughly what CFG at a smaller $w$ does,
  which is why a fair comparison must sweep $w$ (E1's Pareto view). Where the
  toy plant lands (E1): at $k = 0.1$ the paper's law sits on the CFG Pareto
  curve in 2-D for $w \ge 3$ and below it in 32-D, but *above* it at small $w$,
  and it is strictly dominated at $w = 1$ (§4.3).
* **Denoising the guidance direction.** Small components of $e$ are dominated
  by the network's own estimation error. Soft-thresholding is the classical
  estimator for a sparse signal in Gaussian noise (Donoho & Johnstone); the
  sign-shrink is its noisy cousin (it does not zero the small components, it
  flips them). If this is the mechanism, the boundary-layer version (§4.1)
  should strictly dominate on real models, and the threshold should track the
  noise level of $e$ (§4.5). On the toy plant, whose $e$ has no estimation
  noise, the two versions are indistinguishable at $k = 0.1$; at large $k$
  they diverge (E3): the sign law chatters and keeps pushing, the
  soft-threshold attenuates. The 32-D result above, where most coordinates
  of $e$ are small and the shrink helps more, is the sparsity mechanism
  showing through even without noise.

### 3.8 Gaps in the evaluation

* One $w$ per model in Table 2. A method that lowers the effective guidance
  will look better on FID at fixed $w$ whether or not it moves the Pareto
  front. Fig. 6 sweeps $w$ for Flux and does show a front shift; that is the
  figure to reproduce and extend, on several models, with error bars.
* No seed variance anywhere, on any metric, for any method.
* No internal signals: $\|e_t\|$, $\|s_t\|$, and the switching activity would
  have shown §3.1–3.2 immediately.
* $\lambda$ ablated over $\{3,\dots,6\}$ only, where it cannot matter.

---

## 4. Improvements

Each entry: the idea, the equation, why it helps, what to expect, cost.
"Toy" refers to the experiment in §5. All of them are implemented as flags of
`cfgctrl.SMCConfig`; the paper is the all-flags-off configuration.

### 4.1 Boundary layer = soft-threshold (do this first)

$$\Delta e = -k\,\mathrm{sat}(s/\phi), \qquad \mathrm{sat}(z) = \max(-1, \min(1, z)), \qquad \phi = k\lambda.$$

*Why.* This is the textbook chattering fix (Slotine & Li, *Applied Nonlinear
Control*, §7.1): inside a layer of half-width $\phi$ around the surface the
relay becomes a proportional term with gain $k/\phi$, so the input is
continuous and the state settles in the layer instead of crossing it. Here it
does more. Since $s \approx \lambda e$, inside the layer
$\Delta e = -k\lambda e/\phi = -e$ when $\phi = k\lambda$, and

$$e_{\text{applied}} = \mathrm{sign}(e)\,\max(|e| - k, 0),$$

the soft-threshold (L1 proximal) operator, exactly. Small components are
zeroed instead of flipped; large ones are shrunk by $k$ as before.

*Measured.* At $k = 0.1$ the boundary layer and the sign law are
**indistinguishable on the Pareto curve** — the difference is 1 % or less,
against a seed standard deviation of about 1 % (E1, E3), because the sign
law's chatter is $\pm k$ alternating and the Euler integration averages most
of it out. What the boundary layer buys is not a better operating point but a
better-behaved controller, and there the difference is not subtle:
$\mathrm{rms}(s)$ falls to 0.010 instead of plateauing at 0.50 (E2), and the
switching activity is 0.03 of full chatter at *every* $k$ against 0.39 at
$k = 0.02$ rising to 0.83 at $k = 1$ (E3). Those are the properties that
matter once the correction is fed back to a network that will see it again at
the next step. On
real models this is the version to test first because it removes the only
mechanism by which the law can *add* noise. Cost: none. Preset:
`presets.boundary_layer(lam, k)`.

### 4.2 Store the measured error

`store_corrected=False`. Removes the self-referential loop of §3.2 in both its
forms (alternation and lock-in). With the sign law alone this changes little
(the sign-shrink still flips small elements), with the boundary layer it is
what lets $s$ actually reach zero. Cost: none.

### 4.3 Shrink the extrapolation, not the conditional prediction

$$\hat v = v_c + (w - 1)\,(e + \Delta e) = v_\varnothing + w\Big(e + \tfrac{w-1}{w}\,\Delta e\Big).$$

*Why.* CFG is $v_c + (w-1)\,e$: the conditional prediction plus an
extrapolation. Everything wrong with large $w$ lives in the extrapolation;
the conditional prediction itself is the best estimate the network has. The
paper's law corrects the sum, so at $w = 1$ it corrects $v_c$ and the sampler
is no longer the conditional flow: on the toy, fidelity 0.16 vs 0.10 and mean
$p(c \mid x)$ 0.55 against 0.59 for the true class law (E1, $w = 1$ column).
Applying the shrink to $(w-1)\,e$ only makes the law exactly CFG at $w = 1$
for any $k$, keeps the baseline as a special case in a second, independent
way ($k = 0$ *or* $w = 1$), and scales the correction with the amount of
extrapolation, which is what needs taming. Cost: none; the controller needs
to know $w$. Flag: `excess_only=True`; preset `presets.boundary_layer_excess`.

*Measured (E1).* Exactly CFG at $w = 1$; within 1 % of the CFG curve for
$w \le 2$ where the paper's law is 2–10 % worse; 1–3 % (2-D) and 6–12 %
(32-D) better than CFG for $w \ge 3$, i.e. the same high-$w$ gain as the
paper's law. The only law here that is never worse than CFG.

### 4.4 A scale-free gain

$$k_t = \kappa\, \mathrm{rms}(e_t), \qquad \kappa \approx 0.05\text{–}0.1.$$

*Why.* The paper needs $k = 0.1$ on two models and $0.7$ on the third, which
says $k$ is measured in velocity units that differ per model. A fraction of
the current $\mathrm{rms}(e)$ is dimensionless and transfers (E5: the
absolute $k$ gives a different relative effect on a plant with twice the
scale; the relative $k$ gives the same). It also makes the correction vanish
with $e$ late in the run, which is desirable. Cost: one norm per step. Flag: `relative_gain=True`.

### 4.5 Noise-adaptive threshold (the estimation view)

If the mechanism is denoising (§3.7), estimate the noise level of $e$ and
threshold at it:

$$\hat\sigma_t = \frac{\mathrm{median}\,|e_t - \mathrm{median}(e_t)|}{0.6745}, \qquad k_t = c\,\hat\sigma_t \;\;(\text{or } \hat\sigma_t\sqrt{2\ln D},\ \text{Donoho's universal threshold}).$$

*Why.* The median absolute deviation is a robust noise estimate that ignores
the (sparse, large) semantic components. This replaces a hand-tuned $k$ by a
per-step, per-image quantity with a clear meaning, and it directly targets
the stated goal (remove what is noise, keep what is semantic). Not tested on
the toy, whose $e$ is noiseless by construction — adding synthetic noise to
$e$ there turned out to be uninformative, because noise on this plant acts
like extra diffusion and the moment-matching metric rewards it. Testing this
needs a real model and a metric that sees high-frequency artefacts (FID or
LPIPS on decoded images) rather than latent moments. Cost: one median per step,
cheap next to a transformer forward pass. Not implemented; a few lines in
`correct()`.

### 4.6 Decide what the controller is for, then design for it

The paper's control objective ("make $e$ decay like $\dot e = -\lambda e$")
and the effect that produces good images ("attenuate guidance where the
branches agree") are different objectives, and §3.3 shows they call for
*opposite* switching directions. Two coherent options:

* **Keep the effect, drop the SMC story.** Present the method as adaptive
  guidance shrinkage; design the threshold (4.1, 4.4, 4.5) and combine with
  a direction operator (APG's projection removing the component parallel to
  $v_c$) since shrinkage and projection address different failure modes
  (noise vs oversaturation along the conditional direction).
* **Keep the SMC story, fix the plant.** A real sliding mode needs a loop
  gain of order one and relative degree one. Diffusion sampling has neither
  (§3.3–3.4). A plant that integrates its input over a long horizon, such as
  score distillation, would.

### 4.7 Evaluation protocol

1. **Pareto curves over $w$** (fidelity vs alignment), per model, with seed
   error bars. A law helps only if its curve lies below CFG's.
2. **The null model**: CFG plus a memoryless soft-threshold of $e$ by $k$.
   If SMC-CFG does not beat it, the sliding-mode machinery is not doing
   anything (§3.1 predicts it does not).
3. **Internal signals**: $\mathrm{rms}(e_t)$, $\mathrm{rms}(s_t)$, the
   switching activity $\mathrm{rms}(\Delta e_t - \Delta e_{t-1})$ and the
   "derivative-decides" fraction, per step. They are free and they falsify
   or confirm §3 on the real model in one run.
4. **Step-count transfer** (20 / 30 / 50 steps at fixed $\lambda, k$) and
   **scale transfer** (fixed $k$ across models vs relative $k$).
5. **Seed variance** of every metric. (This is your axis.)

### 4.8 A relative-degree-correct redesign (higher effort)

If a real discrete-time sliding mode is wanted on the sampling plant, use a
reaching law with an online gain estimate. Per element $i$, the one-step
model is $s^{i}_{t+1} = c^{i}_t + g^{i}_t\,\Delta e^{i}_t$ with $g$ small and
negative. Estimate $g^{i}$ by recursive least squares from consecutive
$(\Delta e, s)$ pairs, then apply Gao's reaching law
$s_{t+1} = (1 - q)\,s_t - \epsilon\,\mathrm{sign}(s_t)$ solved for $\Delta e_t$
(an equivalent control plus switching). Expect the required $\Delta e$ to be
enormous (dividing by a gain of $10^{-2}$) and therefore clipped: this is the
quantitative form of "the loop cannot be closed in 30 steps". Worth doing once
to make the point, not as a method.

---

## 5. Evidence from the analytic plant

Setup: 8 Gaussian classes on a ring of radius 4 in $D = 2$ (std 1.5,
neighbouring classes overlap, Bayes accuracy of the true class law well below
1) and in $D = 32$ (std 2.0); 30 Euler steps like the paper; $\lambda = 6$,
$k = 0.1$; 3000 samples per run, 3 seeds (error bars are seed standard
deviations). Figures are in `results/toy/` (regenerate with
`python experiments/toy_smc_cfg.py`). Methods keep their colour across all
figures.

### E1. Fidelity and alignment against the guidance scale (`e1_pareto_*.png`)

**ring2d** — the true class law scores mean $p(c\mid x)$ = 0.582 at Fréchet 0 (Bayes accuracy 0.694).

Fréchet distance (lower is better):

| law | w=1.0 | w=1.25 | w=1.5 | w=2.0 | w=3.0 | w=5.0 | w=7.5 | w=10.0 |
|---|---|---|---|---|---|---|---|---|
| CFG (P-control) | 0.10 | 0.63 | 1.09 | 1.83 | 2.93 | 4.53 | 6.08 | 7.37 |
| SMC-CFG, paper (sign) | 0.16 | 0.46 | 0.90 | 1.61 | 2.66 | 4.18 | 5.63 | 6.85 |
| SMC + boundary layer (sat) | 0.17 | 0.45 | 0.88 | 1.58 | 2.60 | 4.08 | 5.50 | 6.68 |
| paper (sign), extrapolation only | 0.10 | 0.59 | 1.02 | 1.71 | 2.73 | 4.23 | 5.68 | 6.89 |
| boundary layer, extrapolation only | 0.10 | 0.59 | 1.02 | 1.70 | 2.71 | 4.17 | 5.58 | 6.75 |

Mean $p(c\mid x)$ (higher is better):

| law | w=1.0 | w=1.25 | w=1.5 | w=2.0 | w=3.0 | w=5.0 | w=7.5 | w=10.0 |
|---|---|---|---|---|---|---|---|---|
| CFG (P-control) | 0.591 | 0.691 | 0.765 | 0.854 | 0.927 | 0.971 | 0.987 | 0.994 |
| SMC-CFG, paper (sign) | 0.554 | 0.652 | 0.727 | 0.825 | 0.910 | 0.963 | 0.984 | 0.991 |
| SMC + boundary layer (sat) | 0.552 | 0.649 | 0.724 | 0.820 | 0.905 | 0.960 | 0.981 | 0.990 |
| paper (sign), extrapolation only | 0.591 | 0.682 | 0.751 | 0.839 | 0.915 | 0.964 | 0.984 | 0.992 |
| boundary layer, extrapolation only | 0.591 | 0.683 | 0.752 | 0.838 | 0.914 | 0.963 | 0.983 | 0.991 |

Fréchet **relative to the CFG curve at matched alignment** (below 1 = genuinely better, 1.0 = just a smaller effective $w$):

| law | w=1.0 | w=1.25 | w=1.5 | w=2.0 | w=3.0 | w=5.0 | w=7.5 | w=10.0 |
|---|---|---|---|---|---|---|---|---|
| SMC-CFG, paper (sign) | -- | 1.090 | 1.054 | 1.015 | 0.996 | 0.984 | 0.987 | 0.992 |
| SMC + boundary layer (sat) | -- | 1.101 | 1.059 | 1.019 | 1.001 | 0.990 | 0.999 | 1.011 |
| paper (sign), extrapolation only | 1.000 | 1.010 | 1.012 | 1.003 | 0.994 | 0.985 | 0.987 | 0.992 |
| boundary layer, extrapolation only | 1.000 | 1.009 | 1.011 | 1.003 | 0.994 | 0.987 | 0.993 | 1.004 |

**ring32d** — the true class law scores mean $p(c\mid x)$ = 0.591 at Fréchet 0 (Bayes accuracy 0.704).

Fréchet distance (lower is better):

| law | w=1.0 | w=1.25 | w=1.5 | w=2.0 | w=3.0 | w=5.0 | w=7.5 | w=10.0 |
|---|---|---|---|---|---|---|---|---|
| CFG (P-control) | 0.82 | 1.13 | 1.64 | 2.58 | 4.02 | 6.09 | 8.05 | 9.68 |
| SMC-CFG, paper (sign) | 1.00 | 0.87 | 1.13 | 1.92 | 3.26 | 5.16 | 6.91 | 8.35 |
| SMC + boundary layer (sat) | 1.05 | 0.89 | 1.09 | 1.80 | 3.06 | 4.86 | 6.51 | 7.86 |
| paper (sign), extrapolation only | 0.82 | 1.04 | 1.42 | 2.20 | 3.47 | 5.30 | 7.03 | 8.45 |
| boundary layer, extrapolation only | 0.82 | 1.04 | 1.43 | 2.19 | 3.39 | 5.11 | 6.73 | 8.05 |

Mean $p(c\mid x)$ (higher is better):

| law | w=1.0 | w=1.25 | w=1.5 | w=2.0 | w=3.0 | w=5.0 | w=7.5 | w=10.0 |
|---|---|---|---|---|---|---|---|---|
| CFG (P-control) | 0.598 | 0.721 | 0.815 | 0.923 | 0.984 | 0.998 | 1.000 | 1.000 |
| SMC-CFG, paper (sign) | 0.501 | 0.612 | 0.711 | 0.852 | 0.962 | 0.995 | 0.999 | 1.000 |
| SMC + boundary layer (sat) | 0.489 | 0.596 | 0.693 | 0.834 | 0.951 | 0.992 | 0.998 | 1.000 |
| paper (sign), extrapolation only | 0.598 | 0.695 | 0.778 | 0.888 | 0.970 | 0.996 | 0.999 | 1.000 |
| boundary layer, extrapolation only | 0.598 | 0.698 | 0.779 | 0.887 | 0.967 | 0.994 | 0.999 | 1.000 |

Fréchet **relative to the CFG curve at matched alignment** (below 1 = genuinely better, 1.0 = just a smaller effective $w$):

| law | w=1.0 | w=1.25 | w=1.5 | w=2.0 | w=3.0 | w=5.0 | w=7.5 | w=10.0 |
|---|---|---|---|---|---|---|---|---|
| SMC-CFG, paper (sign) | -- | 1.024 | 1.024 | 0.981 | 0.933 | 0.917 | 0.947 | 0.990 |
| SMC + boundary layer (sat) | -- | -- | 1.027 | 1.003 | 0.943 | 0.921 | 0.987 | 0.996 |
| paper (sign), extrapolation only | 1.000 | 0.969 | 0.987 | 0.970 | 0.941 | 0.925 | 0.947 | 0.983 |
| boundary layer, extrapolation only | 1.000 | 0.971 | 0.987 | 0.967 | 0.936 | 0.919 | 0.962 | 1.008 |


Reading. In 2-D every shrink variant traces the *same* Pareto curve as CFG for
$w \ge 3$ (ratios 0.98–1.01, against a seed standard deviation near 1 %): the
paper's law at $w = 3$ is simply CFG at $w \approx 2.8$. Below $w = 2$ the
paper's law and its boundary-layer version sit 1–10 % *above* the curve, and at
$w = 1$ they are **strictly dominated** — worse fidelity *and* worse alignment
than CFG (Fréchet 0.162 and 0.173 vs 0.102; mean $p$ 0.554 and 0.552 vs 0.591)
— because they shrink the conditional prediction itself. That is the defect
§4.3 fixes.

In 32-D the picture changes: the shrink variants sit 6–8 % below the CFG curve
at $w = 3$–$5$. The error vector there is anisotropic, most of its coordinates
are small, and per-element shrinkage removes them while keeping the few large
ones — the sparsity mechanism of §3.7, visible even without measurement noise.
The sign and boundary-layer rows stay within about 1 % of each other
throughout, so no ordering between them is claimed.

The extrapolation-only rows are the recommendation: exactly CFG at $w = 1$ by
construction, never more than 1.2 % off the curve anywhere, and they keep the
whole high-$w$ gain (32-D: 0.969, 0.987, 0.967, 0.936, 0.919 at
$w = 1.25$ … $5$).

One caveat on every ratio in these tables: they interpolate the CFG curve at
matched alignment, and above $w \approx 5$ that curve is nearly vertical (mean
$p$ within $10^{-3}$ of 1), so the ratio there is sensitive to the $w$ grid.
Only $w \in [1, 5]$ supports quantitative claims.

### E2. Internal signals of one run, $w = 5$ (`e2_signals.png`)

| law | rms(s) first | rms(s) last | chatter, last 10 | switching activity, last 10 | derivative decides |
|---|---|---|---|---|---|
| SMC-CFG, paper (sign) | 16.97 | 0.5003 | 1.00 | 0.2000 | 2.1 % |
| paper law, measured-error memory | 16.97 | 0.0151 | 0.00 | 0.0000 | 0.3 % |
| SMC + boundary layer (sat) | 16.97 | 0.0099 | 0.00 | 0.0029 | 0.3 % |


Reading. The paper's law never brings $s$ to zero: it plateaus at 0.50, which
is $(\lambda - 1)k = 5 \times 0.1$ exactly as §3.2 predicts, with **every**
element flipping sign in the last ten steps and a switching activity of 0.20,
which is $2k$ — full chatter. Swapping the memory to the measured error alone
collapses the plateau to 0.015 and the chatter to zero; the boundary layer
takes $\mathrm{rms}(s)$ to 0.010. And the derivative term decides the sign of
$s$ for only 2.1 % of elements (0.3 % once the memory is fixed), which is the
measurement behind "the surface is essentially $\lambda e$".

### E3. The chattering regime: sign vs boundary layer as $k$ grows (`e3_k_sweep.png`)

CFG ($k = 0$) scores Fréchet 4.530, mean $p$ 0.971.

| law | metric | k=0.02 | k=0.05 | k=0.1 | k=0.2 | k=0.5 | k=1.0 |
|---|---|---|---|---|---|---|---|
| SMC-CFG, paper (sign) | Fréchet | 4.430 | 4.322 | 4.178 | 3.950 | 3.509 | 3.087 |
| SMC-CFG, paper (sign) | mean $p(c\mid x)$ | 0.969 | 0.967 | 0.963 | 0.957 | 0.940 | 0.923 |
| SMC-CFG, paper (sign) | switching activity / 2k | 0.389 | 0.486 | 0.552 | 0.621 | 0.737 | 0.831 |
| SMC + boundary layer (sat) | Fréchet | 4.414 | 4.276 | 4.079 | 3.740 | 2.932 | 1.941 |
| SMC + boundary layer (sat) | mean $p(c\mid x)$ | 0.969 | 0.965 | 0.960 | 0.946 | 0.883 | 0.704 |
| SMC + boundary layer (sat) | switching activity / 2k | 0.029 | 0.029 | 0.029 | 0.028 | 0.026 | 0.024 |


Reading. The sign law's switching activity climbs from 0.39 to 0.83 of full
chatter as $k$ grows; the boundary layer's sits at 0.024–0.029 throughout, a
13× to 29× reduction, and it is flat in $k$ because the correction inside the
layer is proportional rather than bang-bang.

The Fréchet column falls with $k$ for *both* laws, but that is not a free win:
the alignment column falls with it, so this is movement down the Pareto curve,
not off it. At $k = 1$ the soft-threshold has zeroed most of $e$ and mean
$p(c \mid x)$ has dropped from 0.971 to 0.704 — it is barely guiding any more.
The unambiguous result here is the switching-activity row.

### E4. The loop gain and the paper's Assumption 2 (`e4_loop_gain.png`)

| plant | Assumption 2 holds | reaching condition holds (paper's sign) | median per-step gain $\Delta\sigma\,w\,\lVert J\rVert$ | max | eigenvalues of sym(-∂e/∂x) |
|---|---|---|---|---|---|
| ring2d | 0.0 % | 3.3 % | 0.0453 | 0.288 | [-1.73, 0.04] |
| ring8d | 0.0 % | 16.2 % | 0.0057 | 0.361 | [-2.16, 0.04] |


Reading. Assumption 2 never holds, on either plant. The paper's switching
direction satisfies the reaching condition for 3 % (2-D) to 16 % (8-D) of the
(sample, step) pairs, so the flipped direction is the one that would reduce
$\lVert s \rVert$ for the large majority — and the eigenvalues of
$\mathrm{sym}(-\partial e/\partial x)$ confirm it, running from $-2.16$ to
$+0.04$ where Assumption 2 wants them near $+1$. Meanwhile the correction's
authority over the next measured error has median 0.045 (2-D) and 0.006 (8-D)
against $k = 0.1$, reaching $k$ only for a few samples near
$\sigma \approx 0.8$. There is no loop here for a Lyapunov argument to
describe.

### E5. Scale transfer (`e5_transfer.csv`)

| plant | law | Fréchet | Fréchet / CFG |
|---|---|---|---|
| radius 4 | CFG (P-control) | 4.530 ± 0.019 | 1.000 |
| radius 4 | SMC paper, absolute k | 4.178 ± 0.019 | 0.922 |
| radius 4 | SMC sat, relative k (fraction of rms e) | 4.259 ± 0.019 | 0.940 |
| radius 8 (2x scale) | CFG (P-control) | 9.463 ± 0.038 | 1.000 |
| radius 8 (2x scale) | SMC paper, absolute k | 9.010 ± 0.038 | 0.952 |
| radius 8 (2x scale) | SMC sat, relative k (fraction of rms e) | 8.880 ± 0.038 | 0.938 |


Reading. Doubling the plant's velocity scale changes what an absolute $k$ does
— its effect relative to CFG drifts from 0.922 to 0.952 — while a $k$ expressed
as a fraction of $\mathrm{rms}(e)$ holds (0.940, 0.938). Two plants and one
axis, so this is suggestive rather than conclusive, but it is the mechanism
that would explain why the paper needs $k = 0.1$ for SD3.5 and Qwen-Image and
$k = 0.7$ for Flux.

## 6. Glossary (one paragraph each)

**P / PI / PID.** Input proportional to the error (P); plus the integral of
the error, which removes steady-state offset but can wind up when the input
saturates (I); plus the derivative, which anticipates and damps but
amplifies noise (D). CFG is P with gain $w$, and nothing in the sampler
integrates, so there is no I term to speak of.

**Gain scheduling.** Pre-computed gain as a function of an operating point.
Weight schedulers $w(t)$.

**Reference model.** A desired trajectory for the error, e.g.
$\dot e = -\lambda e$. The controller's job becomes tracking the model rather
than reaching zero.

**Sliding variable / surface.** A scalar or vector function $s$ of the state
whose zero set encodes the desired behaviour; the surface is $\{s = 0\}$.
Motion confined to it obeys reduced-order dynamics chosen by the designer.

**Reaching phase / sliding phase.** Getting to the surface (finite time,
driven by the switching term) / staying on it (the disturbance-rejecting
part).

**Reaching condition.** $s^\top \dot s < 0$, or the stronger
$s^\top \dot s \le -\eta\|s\|$ that gives finite time. Requires the input to
have authority over $\dot s$ with a known sign.

**Equivalent control.** The (continuous) input that would keep $s = 0$
exactly if the plant were known; the switching term only has to cover the
uncertainty around it.

**Matched vs unmatched disturbance.** A disturbance entering through the same
channel as the input can be cancelled by switching (matched); one entering
elsewhere cannot.

**Chattering.** Finite-frequency switching around the surface in any sampled
or lagged implementation; amplitude proportional to the switching gain and
the step.

**Boundary layer / quasi-sliding mode.** Replace $\mathrm{sign}$ by a
saturation of width $\phi$: continuous input, state settles within $\pm\phi$
of the surface (Slotine & Li). In discrete time the state lives in a band
(Gao et al.) whose width you can compute from the reaching law.

**Relative degree.** How many derivatives of the output are needed before the
input appears. Classical SMC and super-twisting need relative degree one in
$s$.

**Higher-order sliding mode / super-twisting.** $u = -k_1|s|^{1/2}\mathrm{sign}(s) + z$,
$\dot z = -k_2\,\mathrm{sign}(s)$: continuous input, finite-time convergence of
$s$ and $\dot s$, no chattering, for relative-degree-one $s$ with a Lipschitz
disturbance.

**Adaptive-gain SMC.** $k$ grows while $|s|$ is outside a band and shrinks
inside, so the gain finds the disturbance bound by itself (Plestan et al.).

**Lyapunov function.** A positive definite "energy" that decreases along
trajectories; its decrease rate gives the convergence type (exponential for
$\dot V \le -cV$, finite-time for $\dot V \le -\eta\sqrt V$).

**Loop gain.** Product of gains around the loop; below 1 per step, feedback
does little; near 1, it regulates and may oscillate.

**Sensitivity function.** $S = 1/(1 + L)$ with $L$ the loop gain; the factor
by which feedback attenuates disturbances and reference errors. Your
variance prediction $\mathrm{Var}(p) = \mathrm{Var}(d)/(1 + gK_p)^2$ is
$|S|^2$.

**Anti-windup.** Stopping the integrator while the actuator is saturated so
it does not accumulate an error it cannot act on.

**Cascade control.** A fast inner loop on a local variable inside a slow outer
loop on the variable of interest; requires time-scale separation.

**MPC.** Optimise the input over a predicted horizon, apply the first move,
repeat. Rectified-CFG++ is a one-step, no-optimisation instance.

**Soft-thresholding.** $\mathrm{sign}(e)\max(|e| - k, 0)$: the estimator that
minimises squared error plus an L1 penalty; zeroes small coefficients, shrinks
large ones by $k$. SMC-CFG with a boundary layer $\phi = k\lambda$ is this
operator applied to the guidance vector.

---

## 7. References

* Wang, Liu, Chi, Liu, Xue, Duan. *CFG-Ctrl: Control-Based Classifier-Free
  Diffusion Guidance.* CVPR 2026. arXiv:2603.03281. Code:
  github.com/hanyang-21/CFG-Ctrl (`pipeline/common_cfg_ctrl.py`).
* Ho & Salimans. *Classifier-Free Diffusion Guidance.* 2022.
* Sadat, Hilliges, Weber. *Eliminating Oversaturation and Artifacts of High
  Guidance Scales in Diffusion Models* (APG). ICLR 2025.
* Fan et al. *CFG-Zero\*.* 2025. Saini, Gupta, Bovik. *Rectified-CFG++.* 2025.
* Bradley & Nakkiran. *Classifier-Free Guidance is a Predictor-Corrector.*
  2024 (what CFG actually samples from; why fidelity degrades with $w$).
* Utkin. *Sliding Modes in Control and Optimization.* Springer 1992.
* Slotine & Li. *Applied Nonlinear Control.* Prentice Hall 1991, ch. 7
  (boundary layer).
* Edwards & Spurgeon. *Sliding Mode Control: Theory and Applications.* 1998.
* Shtessel, Edwards, Fridman, Levant. *Sliding Mode Control and Observation.*
  Birkhäuser 2014 (higher-order SMC, adaptive gains).
* Gao, Wang, Homaifa. *Discrete-time variable structure control systems.*
  IEEE Trans. Industrial Electronics 42(2), 1995 (reaching law,
  quasi-sliding-mode band).
* Levant. *Sliding order and sliding accuracy in sliding mode control.* Int.
  J. Control 58(6), 1993 (super-twisting).
* Plestan, Shtessel, Brégeault, Poznyak. *New methodologies for adaptive
  sliding mode control.* Int. J. Control 83(9), 2010.
* Åström & Hägglund. *PID Controllers: Theory, Design and Tuning.* ISA 1995
  (anti-windup, gain scheduling).
* Donoho & Johnstone. *Ideal spatial adaptation by wavelet shrinkage.*
  Biometrika 1994 (soft-thresholding, universal threshold).
