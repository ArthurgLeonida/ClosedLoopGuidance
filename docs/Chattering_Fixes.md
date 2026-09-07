# Chattering: diagnosis, ranked fixes, and measured results

Reviewed and implemented on 2026-09-06.

**Persistent switching is the clearest controller defect in the supplied results. It has not been established as the largest cause of image-quality loss.** The priority is to stop injecting a fixed correction after the measured discrepancy becomes small, then evaluate whether that improves the fidelity/alignment tradeoff. Making the displayed surface small is insufficient.

## Why the paper arm oscillates near 0.5

The implemented paper recurrence, with corrected memory, is

$$
s_n=e_n+(\lambda-1)(e_{n-1}+\delta_{n-1}),\qquad
\delta_n=-k\,\operatorname{sign}(s_n).
$$

Once measured errors become small compared with a previous nonzero correction,

$$
s_n\approx(\lambda-1)\delta_{n-1}.
$$

For $\lambda=6$ and $k=0.1$, this predicts alternating surface components near $\pm0.5$ and alternating error corrections near $\mp0.1$. It is a two-step cycle, not convergence to a nonzero fixed point. Your reported surface RMS of 0.509–0.515 and almost 100% chatter are consistent with that mechanism. An RMS is unsigned; the chatter statistic supplies evidence of the sign reversals.

The quantity delivered to the sampler is different:

$$
\Delta v_n=w\delta_n,\qquad
|\Delta x_n|=h_n|w\delta_n|
$$

componentwise for an Euler step of magnitude $h_n$. At $w=7$, the velocity perturbation can alternate near $\pm0.7$. Latent displacement also depends on the scheduler. Unequal step lengths and changing model predictions prevent assuming that successive corrections cancel exactly. Zero error from initialization would produce zero correction; this cycle requires an existing perturbation or earlier nonzero error.

The existing `excess` arm already removes corrected-error memory and smooths the switch. Its surface still contains the previous **measurement**:

$$
s_n=e_n+5e_{n-1}.
$$

Consequently its final RMS near 0.1 is not evidence of the same fixed-amplitude cycle. See [the interpretation of your table](Signal_Interpretation.md) for the paired-error bounds and corrected switch normalization. The last row is measured before the last scheduler update.

## Solutions ranked by expected practical benefit

This is a priority ranking for this repository, not an experimentally established ranking of image quality.

| Rank | Change | Why it could improve on the original paper | Confidence and status |
|---|---|---|---|
| 1 | Memoryless soft-thresholding, applied only to CFG extrapolation | Removes feedback of the previous correction, stale measurement directions, and finite sign jumps at zero. Preserves the ordinary conditional prediction at $w=1$. Uses one threshold instead of a sliding-surface parameter and saturation width. | High confidence in these properties; image-quality benefit unproven. **Implemented as `proximal`.** |
| 2 | Make that threshold relative to each sample's current RMS error | An absolute 0.1 can overwhelm weak late guidance. A relative threshold follows its scale and gives consistent behavior under a change of velocity units. | High confidence in scaling and zero-error behavior; threshold selection and model transfer need experiments. **Implemented as `proximal_relative`.** |
| 3 | Schedule correction by noise level, or bound the additional latent displacement | Targets the intervention actually delivered to the sampler. Allows different behavior when establishing composition and refining details. | Moderate confidence as an experiment; no chosen interval/budget is justified here. Proposed. |
| 4 | Redesign a discrete tracking controller using the sampler's actual response | Would address the gap between a desired error-decay law and the response of neural velocity differences to latent perturbations. | Potentially valuable, but lowest implementation confidence and highest complexity. Research proposal. |

A limited guidance interval has improved diffusion quality in other settings, motivating rank 3, but those experiments do not establish an appropriate correction schedule for this flow-model implementation. [Kynkäänniemi et al., NeurIPS 2024](https://arxiv.org/abs/2404.07724).

Lowering $k$ reduces the paper cycle's amplitude, but does not remove its cause. Lowering $\lambda$ just to lower RMS($s$) changes the diagnostic as well as the controller. Adding an integral term or a higher-order sliding controller would introduce more state before the discrete response is justified. These are weaker first choices than removing the demonstrated mechanism.

## What was implemented

For the current discrepancy $e=v_c-v_u$, use the coordinatewise soft threshold

$$
T_\tau(e)=\operatorname{sign}(e)\max(|e|-\tau,0),\qquad
\widehat v=v_c+(w-1)T_\tau(e).
$$

The implementation computes its correction as

$$
\delta_{\mathrm{base}}=-\operatorname{sign}(e)\min(|e|,\tau),\qquad
\delta=\frac{w-1}{w}\delta_{\mathrm{base}},
$$

so the existing hook can continue forming $v_u+w(e+\delta)$. This also avoids subtracting two large, nearly equal errors to obtain a small correction.

For `proximal`, $\tau=k$. For `proximal_relative`, $\tau=k\,\mathrm{rms}(e)$, calculated independently for each sample across its non-batch dimensions. In the relative arm, `k=0.1` is dimensionless: at RMS error 0.01, its threshold is 0.001, rather than the absolute arm's 0.1. Equal numeric gains are **not equal intervention strengths**.

Both variants:

- Produce exactly zero correction at zero measured error, regardless of previous values.
- Cannot reverse or increase the magnitude of a guidance component.
- Preserve plain CFG at `k=0` and at `w=1`.
- Produce constant correction for a constant measured error, rather than sustaining an internal alternating cycle.
- Require no additional denoiser evaluations.

For a fixed threshold, $T_\tau$ is the proximal operator of the $\ell_1$ penalty and is nonexpansive: input perturbations are not amplified in Euclidean norm. This describes the threshold operator, not the complete guided sampler; multiplying by $w-1$ and propagating through the model are separate operations. The fixed-threshold claim also should not be transferred automatically to the adaptive-threshold variant. [Parikh and Boyd, *Proximal Algorithms*](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf).

Why quality might improve: if weak coordinates are mostly unwanted guidance, the new law attenuates them without overshooting or continuing a correction based on stale information. Why it might fail: weak coordinates can contain important detail, coordinatewise shrinkage depends on the latent basis, and reducing extrapolation can lower prompt adherence. There is no established semantic denoising guarantee.

The controller and hook retain their existing inference-only gradient convention and output dtype. The new CLI presets reject explicitly supplied `lam`, `phi`, and `switching` settings because those parameters do not participate in this law. The original paper and smooth `excess` arms remain available for comparison.

For diagnostics, proximal mode logs `s=e` as its current-error reference. It has no sliding surface or temporal derivative. The signal report labels this explicitly, leaves the derivative and plateau predictions unavailable, and adds `surface_reference` to its summary CSV. Raw `StepInfo.deriv_matters` retains numeric zero for compatibility; it means “not applicable” for this mode.

## What the fresh CPU experiments show

Validation: **241 tests passed; one CUDA-only test was skipped because CUDA was unavailable.** Coverage includes the published recurrence, zero-error behavior after arbitrary history, constant-error oscillation, component bounds, per-sample scaling, low-precision output, hook algebra, CLI configuration, diagnostics, and compatibility with older run metadata. The five-arm model grid also passed its dry-run configuration check.

Reproduce with:

~~~bash
python experiments/toy_smc_cfg.py --only e1 e2 --out results/chattering_fixes
~~~

E1 uses 3,000 samples per seed, seeds 0/1/2, 30 steps, and eight guidance scales in both 2D and 32D. E2 uses 3,000 samples, seed 0, 30 steps, and $w=5$. Both use nominal `k=0.1`; sliding arms use $\lambda=6$. Results are in [summary.json](../results/chattering_fixes/summary.json), with CSVs and figures alongside it.

E2 late diagnostics, averaged over the last ten evaluations:

| Arm | Sign-flip fraction | RMS change in applied error correction | RMS applied error correction |
|---|---:|---:|---:|
| Paper | 0.99997 | 0.199996 | 0.100000 |
| Existing smooth `excess` | 0 | 0.002103 | 0.007206 |
| `proximal` | 0 | 0.001714 | 0.005605 |
| `proximal_relative` | 0 | 0.000142 | 0.000453 |

These are error-correction units, before multiplication by $w$ and scheduler step length. The relative threshold makes a weaker late intervention. These results do not establish the same reductions on COCO, and naturally changing measured errors can still change the sign of either new arm.

For a quality check, E1 compares Gaussian moment Wasserstein-2 distance at matched class confidence using interpolation of the CFG curve. At $w=3$, the ratio against CFG is:

| Arm | 2D ratio | 32D ratio |
|---|---:|---:|
| Paper | 0.9954 | 0.9327 |
| Existing smooth `excess` | 0.9943 | 0.9362 |
| `proximal` | 0.9942 | 0.9401 |
| `proximal_relative` | 0.9894 | 0.9713 |

Lower is better; 1 means no improvement over interpolated CFG at matching confidence. **Neither new arm consistently beats the paper or the existing smooth arm.** These ratios use seed-mean metrics and coarse interpolation, without confidence intervals for the ratios. Small differences should not be treated as statistically established. This toy measures distribution moments and class confidence, not image FID or text alignment.

The strongest evidence for applying rank 1 is therefore its removal of a demonstrated failure mode, not a claim of superior generated images. Rank 2 is an available ablation with desirable scaling properties.

## Trained-model comparison to run next

First verify each new arm on the target GPU environment. `verify` exercises the first active arm, so use separate invocations:

~~~bash
python experiments/real_model.py verify --arms proximal
python experiments/real_model.py verify --arms proximal_relative
python experiments/real_model.py grid --arms cfg paper excess proximal proximal_relative --w 1.5 3 7 --seeds 0 1 2 --out results/chattering_comparison
python experiments/signals.py --run results/chattering_comparison
~~~

The grid above uses the runner's small built-in prompt set. For a COCO comparison, supply the same `--prompts` file, model, precision, steps, and seeds used in the original experiment. Use a separate output directory. Add `--dry-run` to inspect the matrix before loading a model.

Tune absolute and relative thresholds separately on held-out prompts, including retuned paper gains and a CFG scale sweep. Named arms such as `"prox005=proximal:k=0.05"` and `"rel02=proximal_relative:k=0.2"` support this without code changes. Compare fidelity/diversity at matched text alignment; better fidelity at the same nominal guidance scale can simply mean weaker guidance.

Keep the paired prompts and seeds when scoring, inspect composition and small details, and record both raw correction magnitude and the displacement delivered by the scheduler. Promote a new default only when this comparison supports it. This session validated the implementation and ran the CPU toy; it did not rerun the unavailable local `results/coco_test` dataset or perform trained-model image evaluation.
