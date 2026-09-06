# CFG-Ctrl / SMC-CFG: what it really does, and how to improve it

**Review update, 2026-09-06.** See [why the final surface is not zero](Signal_Interpretation.md)
for the COCO results. The historical tables and the user's SD3.5 measurements
are retained below. Earlier claims that measured memory guarantees convergence,
that low derivative sign disagreement proves $s=\lambda e$, or that a small
toy Jacobian implies no feedback were too strong. The
[roadmap](Improvement_Roadmap.md) states current research hypotheses and limits.

**A control-engineering review of** *CFG-Ctrl: Control-Based Classifier-Free
Diffusion Guidance* (Wang, Liu, Chi, Liu, Xue, Duan; CVPR 2026 highlight;
arXiv 2603.03281), **written for someone who studied control and wants the
concepts back.** Every concept the paper uses is re-derived in a grey
"refresher" block the first time it appears, then applied to the paper.

The code that goes with this document lives in `cfgctrl/` (see the README).
The evidence tables contain historical toy runs and, in §3.2, the user's
SD3.5 verification measurements. The new COCO analysis uses the supplied
summary; the raw COCO run is not present in this local checkout. Structural
diagnostics do not establish image-quality superiority of the refinements.

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

**What the code and diagnostics support.**

1. The surface is $s_n=e_n+(\lambda-1)m_{n-1}$. Low strict sign disagreement
   with stored memory does not establish $s_n\approx\lambda e_n$.
2. Corrected memory and fixed sign switching can sustain alternation near
   $(\lambda-1)k$ when successive measured errors are sufficiently small.
   This is an alternating orbit, not a fixed point of the controller state.
3. Measured memory removes the explicit previous-correction term, but still
   uses both current and previous measured errors. Neither it nor saturation
   guarantees zero surface at the last denoiser call.
4. Feedback exists through the latent and future predictions. The old E4
   diagnostic did not test the continuous theorem; see corrected §3.3.
5. Extrapolation-only correction preserves ordinary CFG at $w=1$ and weakens the
   correction at $w>1$, requiring a retuned paper baseline.
6. Relative saturation must scale both gain and boundary width by RMS error.
   That supports consistent units, not proven cross-model transfer.

**Candidates to compare.** Test individual refinements and their combination,
ordinary CFG with tuned scales, and actual memoryless soft-thresholding.
Compare paired fidelity/alignment tradeoffs with uncertainty. The
[roadmap](Improvement_Roadmap.md) distinguishes implemented options from
unimplemented hypotheses; none is established as universally superior.

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
  error is $-0.008 \approx 0$. It is not exactly soft-thresholding: the current
  surface still contains memory, and this component remains negative.

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

### 3.1 What the derivative statistic establishes

The surface is $s_n=e_n+(\lambda-1)m_{n-1}$. If $m_{n-1}\approx e_n$,
then $s_n\approx\lambda e_n$. The logged statistic does not test that premise:
it counts strict sign disagreement with stored memory, excluding zeros.

For example, $m_{n-1}=1$, $e_n=-0.1$, $\lambda=6$ gives $s_n=4.9$:
agreement with memory but opposition to the current error. Corrected memory
can also leave a large surface when the measured error is small.

### 3.2 Chattering, and the corrected-error memory

Once $|e_i| < k$, the sign-shrink flips the element's sign (§2.4). With the
corrected error stored as memory, the surface becomes
$s_t \approx e_t + (\lambda - 1)\,\Delta e_{t-1}$, which for small $e$ is
dominated by the previous correction: $\mathrm{sign}(s_t) = \mathrm{sign}(\Delta e_{t-1})$, so
$\Delta e_t = -\Delta e_{t-1}$. The correction alternates every step. Measured
(E2): in the last ten steps **100 % of the elements flip** and the residual
$\mathrm{rms}(s)$ plateaus at $(\lambda - 1)k = 0.5$ instead of going to zero.
The toy trajectory has a residual band. Figure 1 in the paper is explicitly
schematic and cannot be read as measured convergence or chatter.

Two corollaries.

* **Store the measured error and the plateau disappears.** With measured
  memory the surface has no self-reference; with the boundary layer of §4.1
  $\mathrm{rms}(s)$ falls to about $0.010$ in E2. The trained-model observation
  below shows a smaller reduction, about 4.5-fold; neither guarantees zero.
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

#### Confirmed on SD3.5-large

The mechanism above was derived on a Gaussian mixture. It was then measured on a
real model with `experiments/real_model.py verify`: SD3.5-large, bf16,
1024x1024 (latent $16\times128\times128$, $D = 262\,144$), 30 steps, $w = 7$,
$\lambda = 6$, $k = 0.1$, one prompt, one seed.

| | $\mathrm{rms}(e)$ | $\mathrm{rms}(s)$ | chatter (last 5) | derivative decides |
|---|---|---|---|---|
| paper, corrected memory | 0.0997 -> 0.0094 | 0.5983 -> **0.5168** | **0.991** | 1.9 % |
| `store_corrected=False` | 0.0997 -> 0.0093 | 0.5983 -> **0.1139** | **0.267** | 2.0 % |

Two predictions of this section hold quantitatively.

* **The first step initializes $\hat e_{-1} := e_0$, so $s_0 = \lambda e_0$.**
  Predicted $6 \times 0.0997 = 0.5982$; measured $0.5983$.
* **Corrected memory pins $\lVert s\rVert$ at $(\lambda-1)k$** once
  $\lvert e\rvert \ll k$. Predicted $5 \times 0.1 = 0.500$; measured $0.5168$ at
  30 steps and $0.5250$ at 8 steps. Both are consistent with the small-error
  alternating amplitude, not general step-count invariance. Over the run $\mathrm{rms}(e)$ fell
  $10.6\times$ while $\mathrm{rms}(s)$ fell $1.2\times$: the surface stops
  tracking the error and sits on the correction it made last step.

Swapping to measured memory changes only the controller's own state, and the
plateau goes: $\mathrm{rms}(s)$ is $4.5\times$ lower, chatter $3.7\times$ lower,
and $\mathrm{rms}(s)$ falls $5.3\times$ over the run. Its last value still
contains the previous measured error: $s_n=e_n+5e_{n-1}$. The roughly 2%
derivative statistic does not establish $s_n\approx6e_n$. The
[COCO analysis](Signal_Interpretation.md) shows how to check the memory term.

Three honest limits on this measurement. Chatter drops to 0.267, not to the
toy's 0.00: about a quarter of surface components change sign. This statistic
alone does not identify the measured error's sign-flip rate.
The final $\mathrm{rms}(e)$ is the same to 1.1 % across the two arms, which is
a coarse summary that cannot identify local feedback gain. And **none of this is an image-quality result**: it measures the
controller's internal behaviour, not what the sampler produces.

One further observation that the toy could not have produced, because it
depends on the trained model's velocity scale. The elementwise correction has
norm $k\sqrt D = 51.2$, while the measured error has norm 51.0 at the first
step and 3.6 at the last. At the paper's $k$ the correction is the same size as
the entire error at the start and $14\times$ larger than it at the end, so late
in sampling the applied error is dominated by the $\pm k$ sign pattern rather
than by the semantic error. That it does not visibly wreck the output is
plausibly because the pattern alternates and consecutive Euler steps cancel
much of it. It is a concrete argument for the scale-free gain of §4.4.

### 3.3 The actual discrete feedback response

Feedback exists through changes in latent state and future model predictions.
For an Euler step with $h=\sigma_n-\sigma_{n+1}>0$,

$$x_{n+1}=x_n-h[v_u+w(e_n+\Delta e_n)],\qquad
B_n=\frac{\partial e_{n+1}}{\partial\Delta e_n}=-hwJ_{n+1}.$$

The Jacobian is evaluated at the resulting state and **next** noise level,
holding the current state fixed during the intervention. Corrected memory adds
$(\lambda-1)I$ to the next-surface sensitivity; measured memory does not.

The old E4 inspected $-J_n$ at the current state and mislabeled its outputs
as assumption/reaching-condition tests. The updated experiment measures the
next response and compares surface energy against a zero-current-correction
counterfactual. These are local diagnostics, not continuous theorem checks.

Small per-step response does not eliminate accumulated feedback. A gain norm
multiplies the correction and should not be compared directly with $k$ as if
both were competing error amplitudes. Endpoint error norms do not identify
the local gain.

### 3.4 Continuous and discrete surfaces require different analyses

Writing $\dot e=a+wJ\Delta e$, with $a$ collecting correction-independent terms,
gives $s=a+wJ\Delta e+\lambda e$. The surface already contains the input
algebraically; differentiation generally introduces $wJ\Delta\dot e$ and other
terms. Relating this to $\dot s=\Phi+\Gamma\Delta e$ needs justification.

A singular-value bound alone does not fix the control direction: $\Gamma=-I$
is nonsingular but reverses $s^\top\Gamma\operatorname{sign}(s)$. The paper's
supplementary directional assumption is stronger. These questions do not
disprove its measured image-quality improvements.

**Refresher.** Relative degree counts derivatives before an input appears.
A reaching law must match the actual input/output dynamics; adding higher-order
control without identifying those dynamics is not a stability argument.

### 3.5 Lambda in the discrete recurrence

The implemented difference is not divided by elapsed time. With measured
memory, setting $s_n=0$ gives $e_n=(1-\lambda)e_{n-1}$. At $\lambda=6$
this is $e_n=-5e_{n-1}$, not a decaying reference except at zero.
The surface can still become small as both measurements shrink or cross zero
at an isolated step.

A residual $e_n-\exp(-\lambda h_n)e_{n-1}$ would encode exponential decay in
elapsed progress $h_n>0$. It is another algorithm, requiring justified control
authority and retuning. Lowering lambda merely to reduce the displayed surface
does not establish better control or images.

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
  is worth comparing on real models, and the threshold might track the
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
"Toy" refers to the historical experiment in §5. Sections 4.1–4.4 describe
existing flags of `cfgctrl.SMCConfig`; later sections include proposals.
The default configuration reproduces the published discrete law.

### 4.1 Boundary layer = soft-threshold (do this first)

$$\Delta e = -k\,\mathrm{sat}(s/\phi), \qquad \mathrm{sat}(z) = \max(-1, \min(1, z)), \qquad \phi = k\lambda.$$

*Why.* This is the textbook chattering fix (Slotine & Li, *Applied Nonlinear
Control*, §7.1): inside a layer of half-width $\phi$ around the surface the
relay becomes a proportional term with gain $k/\phi$, so the input is
continuous. Whether the state remains in a layer depends on the plant and
sampling; smoothing alone does not establish it. If $s \approx \lambda e$, inside the layer
$\Delta e = -k\lambda e/\phi = -e$ when $\phi = k\lambda$, and

$$e_{\text{applied}} = \mathrm{sign}(e)\,\max(|e| - k, 0),$$

the soft-threshold operator only when $s=\lambda e$, such as initialization
or unchanged measured error with measured memory. In general it is an
approximation; stale memory can still reverse small components.

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
real models this is a candidate to test for lower switching activity, without
claiming it removes every source of noise. No extra denoiser call is needed. Preset:
`presets.boundary_layer(lam, k)`.

### 4.2 Store the measured error

`store_corrected=False`. Removes the self-referential loop of §3.2 in both its
forms (alternation and lock-in). With the sign law alone this changes little
(small components can still reverse). Neither option guarantees zero:
$s_n=e_n+(\lambda-1)e_{n-1}$ still contains two measurements. No additional
denoiser call is needed.

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
paper's law. These historical toy estimates do not establish universal dominance.

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

### 4.6 Choose and test the objective

Reducing prediction discrepancy is not identical to improving image quality
or semantic alignment. Compare selective attenuation as an operator, with
memoryless shrinkage and ordinary CFG at tuned scales. For explicit error
tracking, derive a suitable discrete residual and measure its response.

Neither the toy nor the logs establish universally opposite switching
directions, or that diffusion sampling cannot support feedback control.
See the [roadmap](Improvement_Roadmap.md).

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

### 4.8 A discrete redesign is a separate research experiment

An inverse-gain reaching law needs an identified local response, including
its sign, memory term and scheduler step. Weak or singular directions can
require excessive corrections; clipping changes the assumed dynamics.
Observational correlations alone do not identify causal authority.
Use controlled perturbations and compute-matched comparisons before adding
adaptive or higher-order control.

---

## 5. Historical evidence from the analytic plant

The metric called Fréchet here is Gaussian moment Wasserstein-2 distance in toy
state space: the square root of the squared FID expression, not image FID.
The tables retain historical values. E4's invalid theorem interpretation is
withdrawn. E5 predates relative boundary-width scaling and needs rerunning for
current comparisons; changing target data with fixed noise is not a pure
change of units.

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
measurement of sign agreement with stored memory. It does not establish
$s_n\approx\lambda e_n$, especially when memory contains the last correction.

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

### E4. Discrete next-step response

The former “Assumption 2 holds” and “reaching condition holds” counts were based
on an inappropriate proxy and are withdrawn. The current experiment reports
next-error sensitivity $-hwJ_{n+1}$, next-surface sensitivity
$-hwJ_{n+1}+(\lambda-1)I$, and surface-energy changes relative to a counterfactual
without the current correction. It excludes the terminal update, which has no
subsequent controller evaluation. None of these is a continuous theorem test.
See §3.3 and the current CSV fields.

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

**Loop gain.** Local response around a feedback loop. Its magnitude, sign,
time variation and accumulation matter; small response does not imply open-loop.

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
large ones by $k$. The boundary-layer controller matches this map only
when $s=\lambda e$; memory generally changes the operator.

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
