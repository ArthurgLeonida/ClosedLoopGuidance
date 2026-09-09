# Chattering: diagnosis and ranked improvements

The ±0.5 surface cycle is a clear defect in the implemented paper controller. It is not yet established as the largest cause of image-quality loss. The generation/evaluation pipeline now makes that distinction testable on the paper's benchmark families.

## Why the surface stays near ±0.5

With corrected-error memory, the recurrence is

$$
s_n=e_n+(\lambda-1)(e_{n-1}+\delta_{n-1}),\qquad
\delta_n=-k\,\operatorname{sign}(s_n).
$$

When the measured discrepancy becomes small after an earlier nonzero correction,

$$s_n\approx(\lambda-1)\delta_{n-1}.$$

At λ=6 and k=0.1, surface components can alternate near ±0.5 and corrections near ∓0.1. This is a two-step cycle, not a nonzero fixed point. Your surface RMS of 0.509–0.515 and almost complete sign switching fit this mechanism. RMS itself is unsigned.

The sampler receives a velocity change `wδ`. For an Euler step of magnitude `h`, the corresponding displacement change has magnitude `h|wδ|`. At w=7 the paper correction can alternate near ±0.7 in velocity units. Changing predictions and unequal step lengths prevent assuming perfect cancellation.

The smooth `excess` arm instead uses measured memory, so `s=e+5e_previous`. Its final surface RMS near 0.1 can therefore exceed the current error RMS without the fixed-amplitude correction cycle. Neither surface is measured on the final decoded image: the last observation precedes the last scheduler update.

## Ranked solutions

This ranks expected practical value and confidence in the mechanism, not demonstrated image-quality gains.

| Rank | Solution | Why it could improve on the paper | Status |
|---|---|---|---|
| 1 | Memoryless soft-thresholding of CFG extrapolation | Removes feedback of the previous correction and finite sign jumps at zero; cannot reverse or amplify a guidance component | Implemented as `proximal`; high confidence in these properties |
| 2 | Threshold relative to each sample's current RMS error | Adapts to weaker late guidance and different velocity units | Implemented as `proximal_relative`; quality and threshold transfer need experiments |
| 3 | Noise-dependent correction interval or latent-displacement budget | Controls when and how strongly the correction moves the actual sample | Proposed; requires tuning and scheduler-aware experiments |
| 4 | Discrete tracking design using the denoiser/sampler response | Could connect the intended error-decay law to the neural model's actual response | Research direction; highest complexity and lowest implementation confidence |

Lowering k reduces the cycle amplitude but retains its cause. Lowering λ just to make the displayed surface smaller changes the diagnostic and control law simultaneously. Adding integral or higher-order state before modeling the discrete response is a weaker first step.

Guidance restricted to part of the sampling interval has helped other diffusion models, motivating rank 3; it does not determine a suitable interval here. [Kynkäänniemi et al., NeurIPS 2024](https://arxiv.org/abs/2404.07724).

## Implemented law

For `e=v_cond-v_uncond`:

$$
T_\tau(e)=\operatorname{sign}(e)\max(|e|-\tau,0),\qquad
\widehat v=v_{\rm cond}+(w-1)T_\tau(e).
$$

`proximal` uses τ=k. `proximal_relative` uses τ=k·rms(e), independently per sample. In the relative arm k is dimensionless: k=0.1 at RMS error 0.01 means τ=0.001, whereas absolute k=0.1 means τ=0.1.

Both variants have zero correction at zero current error regardless of history, preserve the conditional endpoint at w=1, recover CFG at k=0 and require no extra denoiser calls. Constant measured error produces constant correction. Changes in the model's measured error can still produce sign changes.

For fixed τ, soft-thresholding is the nonexpansive proximal operator of an ℓ1 penalty. This guarantee concerns that operator, not the complete guided sampler or the adaptive-threshold variant. [Parikh and Boyd, Proximal Algorithms](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf).

Why it might help: weak unwanted guidance coordinates are attenuated without overshoot or stale correction memory. Why it might fail: weak coordinates can encode useful details, shrinkage depends on the latent basis, and weaker extrapolation can reduce prompt adherence. There is no established semantic denoising guarantee.

## Evaluate the improvement

Start with the [small research profiles](Research_Workflow.md) while changing the controller. They retain paired images and full controller traces while reducing the default development run to 256 images. Reserve the full [paper configuration](../configs/paper.json) for later benchmark evaluation.

Use named gains such as `small=proximal:k=0.05` and `relative02=proximal_relative:k=0.2` for ablations. Tune absolute and relative gains separately on development prompts; compare against retuned paper gains and CFG scales, then validate on held-out prompts.

`diagnostics.csv` reports the final measured error and surface RMS, late switching, correction magnitude and the actual velocity perturbation after dtype conversion. Proximal's `surface_reference=current_error` means `s=e` is a diagnostic reference, not a sliding surface. The per-image JSON keeps the complete traces.

The CPU tests establish the recurrence, component bounds, scaling and adapter behavior. The old toy results did not show that either proximal variant consistently beat the paper controller; they are not image-benchmark evidence. Promote an image-quality default only after the [declared benchmark protocol](Benchmark_Protocol.md) supports it.
