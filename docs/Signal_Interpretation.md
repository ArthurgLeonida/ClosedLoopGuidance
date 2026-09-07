# Why the last sliding surface is not zero

Reviewed 2026-09-06 against the current controller and `experiments/signals.py`.
The numerical table discussed here was supplied by the user from `results/coco_test`.
That run directory is absent from the local checkout, so the explanations below
combine its reported aggregates with the verified recurrence. They are not a
replay of the COCO trajectories.

**The paper arm is consistent with a memory-induced alternating residual near
0.5. The excess arm removes that mechanism, but its surface still contains the
previous measured error. Neither arm guarantees a zero surface at the last
denoiser call.** Earlier wording suggesting that measured memory or saturation
makes the surface reach zero was too strong.

## 1. The paper arm: why 0.509–0.515 is expected here

The controller computes the surface before applying the current correction:

$$
s_n=e_n+(\lambda-1)m_{n-1},\qquad
\delta_n=-k\,\operatorname{sign}(s_n).
$$

For the paper arm, `store_corrected=True` means
$m_{n-1}=e_{n-1}+\delta_{n-1}$, so

$$s_n=e_n+5e_{n-1}+5\delta_{n-1}$$

at $\lambda=6$. If both measured-error terms are small relative to the memory
correction, then $s_n\approx5\delta_{n-1}$. Since the sign law applies magnitude
$k=0.1$ to each nonzero surface component, the limiting RMS amplitude is

$$\operatorname{rms}(s_n)\approx5k=0.5.$$

Its sign reverses the next correction:
$\delta_n\approx-\delta_{n-1}$. This is an alternating orbit, **not a fixed
point of the full state**. It need not start from exactly zero error/memory,
and $|e|<k$ alone is not a sufficient condition for every trajectory.

Your measured 0.5088–0.5154 is about 2–3% above the limiting amplitude. Nonzero
measured errors and their correlation with the correction can account for
departures from 0.5. Chatter of 0.993–0.997 and normalized activity close to one
are consistent with almost every component reversing its full correction.

The approximate independence from $w$ is also expected: the controller stores
an error correction of magnitude $k$, before multiplication by $w$. The
guidance scale still changes the latent trajectory and subsequent measurements.

## 2. The excess arm: why an error near 0.01 can give a surface near 0.1

The default `excess` preset combines saturation, measured-error memory and
extrapolation-only correction. Its surface is

$$s_n=e_n+5e_{n-1}.$$

There is no previous-correction term. However, removing that term does not
remove the previous **measurement**, and that measurement has coefficient 5.

The current table prints only the first and last error norms. It omits the
penultimate norm and the angle between successive errors. In particular,

$$\operatorname{rms}(s_n)\ne6\operatorname{rms}(e_n)$$

in general. Equality would require the relevant vectors to agree between steps,
not merely a small fraction of sign changes.

An illustrative scalar example reproduces the magnitude in your first row:

$$e_{n-1}=0.0178,\quad e_n=0.0103
\quad\Longrightarrow\quad s_n=0.0103+5(0.0178)=0.0993.$$

This is a possible explanation, not a measured penultimate error from your run.
From the triangle inequality, the pasted averages alone imply these intervals
for the mean penultimate RMS, assuming the default measured-memory configuration
and the same trajectories at both endpoints:

| Scale | Reported last RMS error | Reported last RMS surface | Compatible mean penultimate RMS |
|---|---:|---:|---:|
| 1.5 | 0.0103 | 0.0993 | 0.0178–0.0219 |
| 7.0 | 0.0159 | 0.1143 | 0.0197–0.0260 |

The new report obtains the actual penultimate values from existing CSV rows,
paired by arm, scale, prompt and seed. For each pair it checks the norm range

$$
|5R_{n-1}-R_n|\le R_s\le5R_{n-1}+R_n,\qquad
R_j=\operatorname{rms}(e_j).
$$

These are bounds, not an exact norm reconstruction: scalar RMS logs do not
retain directional correlations. CSV rounding also limits their precision.

At initialization, the surface RMS is $6\times0.1851=1.1106$. The excess
arm's reported final surface is therefore about **90–91% lower** than its
initial value. It is not a constant 0.5 residual like the paper arm. A last
value of 0.1 by itself does not show that the surface has stopped decreasing.

## 3. Saturation does not project the surface onto zero

With the default $\phi=k\lambda=0.6$, the excess correction is

$$
\delta_n=-\frac{w-1}{w}\,0.1\,
\operatorname{clip}\!\left(\frac{s_n}{0.6},-1,1\right).
$$

This smooths the correction for components inside the layer. It does not
replace the measured error or overwrite the current surface with zero.
The corrected velocity affects the next latent and, through the model, future
measurements. A convergence claim would require analyzing that whole loop.

RMS surface values around 0.1 are below the boundary half-width 0.6, but an RMS
bound does not mean every component lies inside the boundary.

The logged `s_rms` is measured during the last denoiser forward call. The
scheduler still performs its final latent update afterwards. It is not a
fresh measurement on the final latent or decoded image. The standard
[SD3 pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py)
runs the denoiser, combines guidance, then applies the scheduler update.

There is also a discrete-time distinction. For measured memory, forcing
$s_n=0$ at consecutive steps with $\lambda=6$ would require

$$e_n=-5e_{n-1}.$$

Except for the zero solution, that is alternating growth, not exponential
decay. Thus the implemented finite-difference surface cannot simply inherit
the interpretation of the continuous equation $\dot e+\lambda e=0$.
This does not prevent the surface becoming small as both measurements shrink,
or crossing zero at an isolated step. The
[paper's supplement §6.3.4](https://arxiv.org/html/2603.03281v2#S6.SS3.SSS4)
itself distinguishes the continuous design argument from discrete chattering
and a heuristic stability corridor.

## 4. What the other columns mean

**Chatter is a sign statistic.** The excess arm's 0.257–0.270 means around a
quarter of surface components change sign in the tail. Smooth corrections can
cross zero with tiny amplitude. This differs from the paper's nearly complete
reversal of a fixed-magnitude correction. The statistic alone does not identify
how often the current measured error changes sign.

**The previous switch column used the same nominal $2k$ for every arm.**
Excess correction is weakened by $(w-1)/w$. Dividing your excess values by
that factor gives:

| Scale | Old switch / nominal $2k$ | Switch / twice the arm's applied gain |
|---|---:|---:|
| 1.5 | 0.029 | 0.087 |
| 2.0 | 0.043 | 0.086 |
| 3.0 | 0.058 | 0.087 |
| 4.5 | 0.069 | 0.089 |
| 7.0 | 0.077 | 0.090 |

Much of the apparent growth across scales is the explicit correction scaling.
The revised console uses the applied fixed gain. The CSV retains the old
`switch_activity_norm` field and adds `switch_activity_applied_norm`.
Relative-gain arms are left without a fixed-gain normalization.

For a fixed-amplitude sign law without zero components, each sample/step obeys
normalized activity $=\sqrt{\text{chatter}}$. Averaging those quantities does
not generally give $\operatorname{mean}(A)=\sqrt{\operatorname{mean}(C)}$.
Rounding and zero crossings also matter.

**A low derivative percentage compares against stored memory.** It counts
strict sign disagreement between $s_n$ and $m_{n-1}$ over the whole run.
In the paper arm, that memory includes the previous correction. A small
percentage therefore does not prove $s_n\approx\lambda e_n$, or that the
current measured error and correction point in opposite directions.

Finally, `±` is a sample standard deviation across final CSV records,
typically prompt–seed pairs. It is not a confidence interval for the mean.

## 5. What to check next

Run the updated summarizer on the existing run; no new model evaluations are
needed:

~~~bash
python experiments/signals.py --run results/coco_test
~~~

The additional table shows the actual penultimate error, paired count, and
lower/upper surface-norm bounds. The plot also shows the approximate
$\lambda\operatorname{rms}(e_n)$ reference so departures caused by memory are
visible. Plateau predictions are limited to the fixed-gain, corrected-memory
sign regime and include excess scaling when applicable.

If a trajectory's surface lies outside the bounds beyond logging precision,
check its saved arm parameters, step numbering and CSV provenance. Duplicate
endpoint rows are now rejected rather than silently counted twice.

For subsequent model runs, log the scheduler noise levels and actual correction
delivered after low-precision branch rounding. Those distinguish timing,
memory and quantization effects. Compare image quality, prompt alignment and
diversity with paired prompts/seeds; the current table cannot rank image quality.

If the goal is specifically to track exponential decay, a candidate discrete
residual is $e_n-\exp(-\lambda h_n)e_{n-1}$ with elapsed progress $h_n>0$.
That is a separate algorithm requiring a justified control response, retuning
and evaluation. Lowering $\lambda$ merely to reduce the displayed surface
would not establish better control or better images. See the
[improvement roadmap](Improvement_Roadmap.md) for the broader research plan.

The follow-up [ranked chattering fixes](Chattering_Fixes.md) implement a simpler
alternative: memoryless soft-thresholding of the current CFG extrapolation,
with absolute or RMS-relative thresholds. Its logged `s=e` is a current-error
reference, so its surface norm must not be compared directly with either
sliding arm's norm. The follow-up includes fresh toy evidence and the limits
of claiming an image-quality improvement.
