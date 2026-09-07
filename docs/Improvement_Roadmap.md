# Improving CFG-Ctrl: proposals, rationale, and evidence needed

The first implementation priority is now **memoryless soft-thresholding of CFG extrapolation**, followed by an RMS-relative threshold. Both are available as experiment arms. The [ranked chattering review](Chattering_Fixes.md) explains the mechanism, guarantees, limitations, and fresh CPU comparison. The existing smooth correction with measured-error memory remains a useful comparator. **Better images than the original paper remain a hypothesis**; the new toy results do not establish a consistent winner.

This roadmap accompanies the [detailed review](CFG-Ctrl_Review_and_Improvements.md). It distinguishes existing options from proposed research and explains what would support or falsify each proposal. Sources were checked on 2026-09-04 against the [local paper, arXiv v2](CFG-Ctrl.pdf), the [online paper](https://arxiv.org/html/2603.03281v2), and the [authors' implementation](https://github.com/THU-SI/CFG-Ctrl/blob/main/pipeline/common_cfg_ctrl.py).

## 1. The comparison to preserve

For conditional and unconditional velocities, define $e_n=v_c-v_u$. The published discrete method is

$$
s_n=e_n+(\lambda-1)\widehat e_{n-1},\qquad
\delta_n=-k\,\operatorname{sign}(s_n),\qquad
\widehat v_n=v_u+w(e_n+\delta_n),
$$

with corrected memory $\widehat e_n=e_n+\delta_n$ and first-step initialization from the measured error. The existing `presets.paper()` is the reproduction baseline. Preserve it when testing modifications.

Use the paper's settings as one baseline: $\lambda=6$, $k=0.1$ for SD3.5/Qwen and $k=0.7$ for Flux. The authors' README defaults and example settings differ, so record the exact code revision and values. The paper already uses a disjoint tuning set, reports compositional benchmarks, and includes a Flux guidance-scale sweep; a fair extension should reproduce and broaden these experiments. [Paper §§7.3, 8.1, 9.1](https://arxiv.org/html/2603.03281v2), [author README](https://github.com/THU-SI/CFG-Ctrl#parameters).

## 2. Improvements already available in this repository

These options live in [controllers.py](../cfgctrl/controllers.py). Their compatibility and numerical behavior can be tested without a large model; their visual benefits cannot.

| Priority and option | Why it could improve on the published recurrence | What it does not guarantee |
|---|---|---|
| 1. `store_corrected=False` | Removes the previous correction from the next surface calculation. With small measured error and $\lambda>1$, corrected memory can sustain alternating corrections of magnitude $k$. **Measured on SD3.5-large** (30 steps, $w=7$, one prompt/seed): $\mathrm{rms}(s)$ 0.5168 to 0.1139 and chatter 0.991 to 0.267, with the measured error unchanged to 1.1 %. | Measured errors can still oscillate, and useful temporal behavior might be lost. Residual chatter counts sign changes in the surface containing two measurements; it does not directly measure the current model error's sign changes. Lower chatter is not an image-quality result. |
| 2. `switching="sat"` | Replaces the discontinuous sign switch with $\operatorname{clip}(s/\phi,-1,1)$, reducing abrupt changes near zero. | Smooth correction does not prove convergence or better image quality. |
| 3. `excess_only=True` | Preserves the ordinary conditional prediction at $w=1$ and attenuates only additional guidance. | It changes the correction strength at every $w>1$; retuning is necessary. |
| 4. `relative_gain=True` | Expresses correction size relative to each sample's measured velocity discrepancy. Scaling both gain and saturation width avoids dependence on arbitrary velocity units. | Matching units does not remove differences in model geometry, prompt difficulty, or error reliability. |

### Preserve the conditional endpoint

The extrapolation-only form is

$$
\widehat v=v_c+(w-1)(e+\delta),\qquad w\geq1.
$$

At $w=1$, this is $v_c$ for any correction. The published form is instead $v_c+\delta$ at that endpoint. If the conditional model is already adequate, preserving it prevents the correction from introducing an unnecessary bias. When that model benefits from correction even without CFG, endpoint preservation could forgo that benefit; this is a design choice to evaluate.

For the same raw $\delta$, this option applies $(w-1)/w$ of the published velocity correction. A favorable result at a fixed $k$ can therefore be just weaker intervention. Compare against a retuned paper baseline as well.

### Smooth switching is not generally soft-thresholding

With $\phi=k\lambda$ and $s=\lambda e$, saturation gives exactly

$$
e-k\operatorname{clip}\left(\frac{e}{k},-1,1\right)
=\operatorname{sign}(e)\max(|e|-k,0).
$$

The equality holds at initialization, and when consecutive measured errors agree. With memory, generally $s\ne\lambda e$, so this is only an approximation. For example, $e_{n-1}=1$, $e_n=-0.1$, $\lambda=6$, and $k=0.1$ give $s_n=4.9$ and corrected error $-0.2$: a stale surface can increase the magnitude of the current error. Exact soft-thresholding would give zero. Lower switching activity alone is therefore insufficient evidence that this option denoises guidance.

### Make the entire smooth law scale consistently

For per-sample $r_n=\operatorname{rms}(e_n)$, use

$$
k_n=\alpha r_n,\qquad \phi_n=\beta r_n,\qquad
\delta_n=-\alpha r_n\operatorname{clip}\left(\frac{s_n}{\beta r_n},-1,1\right).
$$

Handle $r_n=0$ with zero correction. If an entire error sequence is multiplied by a fixed positive constant, its correction scales by the same constant in exact arithmetic. Scaling only $k_n$ while keeping $\phi$ absolute does not have this property. The audited implementation scales both for relative saturation. This supports transfer across units; transfer across trained models remains unmeasured. It is not invariance to arbitrary time-varying rescaling, because the surface contains memory.

All four changes require no extra denoiser evaluations. They still require tensor operations and diagnostics can cause CPU/GPU synchronization, so benchmark actual latency.

## 3. Follow-up experiments

### A. Compare the memoryless soft-threshold baseline

**Status:** implemented as `presets.proximal_excess()` and `presets.proximal_relative_excess()`, exposed by the CLI as `proximal` and `proximal_relative`. Both participate in toy E1/E2. See [the implementation and measured comparison](Chattering_Fixes.md).

Use

$$
T_k(e)=\operatorname{sign}(e)\max(|e|-k,0),\qquad
\widehat v=v_c+(w-1)T_k(e).
$$

Unlike the history-dependent surface, this map cannot reverse an individual guidance component or increase its magnitude. For fixed $k$, it is also a nonexpansive map: perturbations to its input are not amplified in Euclidean norm. These statements concern the guidance operator, not stability of the full sampler. Adaptive thresholds need a separate analysis.

**Why it could work better:** it removes a discontinuity and stale-memory effects while retaining selective attenuation of weak components. If the apparent advantage of SMC-CFG is mostly attenuation, this simpler baseline may match it with fewer parameters.

**How it could fail:** small coordinates can encode important detail. Coordinatewise shrinkage also depends on the latent basis; it need not preserve semantic directions. Rotate an otherwise identical isotropic toy problem, and compare sparse and dense error patterns. Reject the proposed mechanism if gains disappear under these controls or if only alignment decreases.

### B. Test a guidance interval or a bounded correction budget

**Status:** proposed, not implemented here.

Compare constant correction against a smooth noise-level window and against clipping the additional latent displacement to a chosen budget. For example, with step length $h>0$, cap $\|h(w-1)\delta\|_2$. This directly limits the perturbation delivered to the integrator rather than interpreting an unverified sliding-surface bound as a numerical guarantee.

**Why it could work better:** the useful correction may vary through denoising. Restricting guidance to part of the trajectory improved quality in other diffusion settings, which motivates testing it here; it does not establish the right interval for these flow models. [Kynkäänniemi et al., NeurIPS 2024](https://arxiv.org/abs/2404.07724).

**How it could fail:** early intervention may be needed to establish composition, and late intervention may preserve detail. A tight budget can remove the benefit entirely. Compare the same schedule applied to ordinary CFG, and tune budgets on held-out prompts. Retain a constant-correction arm to isolate the contribution.

### C. Compare structured attenuation with coordinatewise shrinkage

**Status:** proposed, not implemented here.

Test a projection-based method such as APG as its own baseline before composing it with SMC-CFG. APG separates the guidance component parallel to the conditional prediction from its orthogonal component and also uses rescaling/momentum. Its published evidence motivates testing directional structure rather than treating every latent coordinate equally. [Sadat, Hilliges, Weber, ICLR 2025](https://arxiv.org/abs/2410.02416).

**Why it could work better:** reducing a problematic direction while preserving the remaining signal may retain alignment better than shrinking all weak coordinates. **How it could fail:** that direction may carry useful prompt information, and the relevant geometry may differ across models. Equal tuning budgets and an unmodified APG arm are essential; otherwise a combined method has an unfair advantage.

### D. Investigate uncertainty-aware thresholds only after those controls

**Status:** research hypothesis.

A larger threshold is justified when a component is unreliable, not merely small. Measure whether temporal innovations or another cheap reliability proxy predict harmful guidance. Compare them against per-sample RMS before introducing channelwise or spatial thresholds.

Median absolute deviation of $e$ is a measure of its spread; without a signal/noise model it is **not** a calibrated estimate of network error. Classical sparse Gaussian-noise threshold formulas cannot simply be imported into correlated, structured latent predictions. Extra denoiser evaluations used to estimate uncertainty must count against the compute budget.

## 4. Repair the theoretical question before adding more control complexity

The sampler does contain feedback: a correction changes $x$, and $x$ changes the next measured $e$. A small local sensitivity does not make that loop open-loop, nor rule out accumulated effects over many steps.

For the continuous equations, let $J=\partial e/\partial x$ and let $a$ collect terms independent of the correction. Then

$$
\dot e=a+wJ\delta,\qquad s=a+wJ\delta+\lambda e.
$$

Thus $s$ already contains the input algebraically. Differentiating it generally introduces $wJ\dot\delta$ and other terms. The transfer from this system to a model of the form $\dot s=\Phi+\Gamma\delta$ needs additional justification. A singular-value lower bound alone also does not establish the needed control direction: $\Gamma=-I$ is nonsingular but reverses the sign of $s^T\Gamma\operatorname{sign}(s)$. The supplementary directional assumption is stronger. These are limits on transferring the continuous proof to the discrete algorithm, not evidence that the paper's measured image improvements are false.

For an Euler step in decreasing noise level,

$$
x_{n+1}=x_n-h[v_u+w(e_n+\delta_n)],\qquad
\frac{\partial e_{n+1}}{\partial\delta_n}=-hwJ_{n+1}.
$$

Here $J_{n+1}$ is evaluated at the resulting state and next noise level, holding the current state fixed during the intervention. The next discrete surface additionally contains direct memory dependence: corrected memory contributes $(\lambda-1)I$. A current-state $-J$ proxy is neither the complete next-step surface response nor the continuous proof's $\Gamma$.

Use controlled finite perturbations to check this discrete response on the toy plant, then selected trained-model trajectories. If testing a time-scaled surface, use elapsed denoising progress $h_n$ consistently, for example $s_n=(e_n-e_{n-1})/h_n+\lambda e_n$. This is a separate algorithm, with new parameter units and potential derivative amplification, not a guaranteed improvement. Avoid adding super-twisting or adaptive gains until the input-output dynamics and objective are established.

## 5. An experiment that can justify “better than the paper”

1. **Verify integration first.** Check zero-gain equivalence with the unmodified pipeline, conditional/unconditional ordering, reset between samples, dtype behavior, and one controller update per intended solver step. The current hook targets doubled-batch CFG; the $w=1$ conditional endpoint needs a direct sampler check because many pipelines disable branch doubling there. A dummy-denoiser test does not establish real-pipeline compatibility.

2. **Freeze tuning and test sets.** Use paired prompts and initial noise across methods. Tune each method's $w$ and its own parameters with equal budgets on a disjoint set. Fix checkpoint revision, scheduler, resolution, negative prompt, embedded guidance where applicable, precision, and number of denoiser evaluations. Repeat across seeds and more than one model before claiming transfer.

3. **Measure a quality/alignment tradeoff.** Compare CFG, published SMC-CFG, individual refinements, their combination, and memoryless shrinkage across guidance scales. Report fidelity/diversity and compositional alignment together. Use the same evaluation implementation and sample count. The toy's Gaussian moment distance and class posterior are not image FID and CLIP; even the distance versus squared-distance convention changes percentage comparisons.

4. **Quantify uncertainty.** Report paired differences for promptwise metrics with intervals obtained by resampling prompts and accounting for repeated seeds. Recompute dataset metrics under the appropriate resampling scheme; FID is not a per-image score. Match alignment only within the overlap of measured curves. Densify nearly saturated regions and avoid extrapolated or unstable percentage improvements.

5. **Separate mechanism from outcome.** Log measured error, surface, applied correction, switching activity, component reversals, and additional latent displacement. Low surface norm and low chatter do not prove semantic quality. Vary step counts, noise schedules, coordinate rotations, and mixture geometry in the toy experiments. Include runtime, peak memory, and model-evaluation counts for practical comparisons.

Call a variant better only when a prespecified held-out comparison shows better quality at comparable alignment and compute, with uncertainty small enough to support the difference and no material diversity regression. If it merely reproduces CFG at a lower effective scale, that is useful behavior to document, but it does not demonstrate an improved tradeoff. The immediate deliverable is a sound candidate and a reproducible test of it; the image-quality conclusion belongs to that experiment.
