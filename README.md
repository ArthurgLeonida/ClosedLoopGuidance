# CFG-Ctrl, reimplemented and measured

A control-engineering study of **CFG-Ctrl: Control-Based Classifier-Free Diffusion
Guidance**, Wang et al., CVPR 2026. [Paper](https://arxiv.org/abs/2603.03281),
[authors' code](https://github.com/THU-SI/CFG-Ctrl).

This repository reproduces the published discrete controller, studies it on an
analytic Gaussian-mixture flow, and provides a hook for doubled-batch CFG
pipelines. CPU tests and toy experiments validate the local implementation.
**The proposed variants have not been shown to produce better images than the
paper on trained models.**

Start with [possible improvements and why they could help](docs/Improvement_Roadmap.md).
The [implementation review](docs/CFG-Ctrl_Review_and_Improvements.md) explains
the controller, the toy evidence, and limits of the control-theory interpretation.

## The controller

For measured error `e = v_cond - v_uncond`, the published recurrence is:

~~~text
prev = e                                # initialization only
s = (e - prev) + lam * prev
delta = -k * sign(s)
v_hat = v_uncond + w * (e + delta)
prev = e + delta                        # corrected memory
~~~

The default parameters reproduce the paper's SD3.5/Qwen settings:
`lam=6`, `k=0.1`. The float32 reference test replays the authors' recurrence.
`k=0` is an exact no-op on the error.

The implemented refinements are independent options:

| Option | Motivation | Limit |
|---|---|---|
| `store_corrected=False` | Remove the previous correction from the next surface calculation. | Model measurements can still oscillate. |
| `switching="sat"` | Smooth abrupt switching near zero. | Exact soft-thresholding only when `s=lam*e`; memory can still reverse small components. |
| `excess_only=True` | Correct only the extrapolation beyond conditional prediction; preserve CFG at `w=1`. | Its weaker correction needs comparison with a retuned paper baseline. |
| `relative_gain=True` | Scale gain and saturation width by each sample's RMS error. | Consistent units do not guarantee cross-model transfer. |

The toy diagnostics show a specific memory-induced switching mechanism. They
do not certify or refute the paper's continuous theorem on neural models.
The toy fidelity metric is Gaussian moment Wasserstein-2 distance
(square root of the squared FID expression), not image FID.

## Setup and CPU checks

`cfgctrl/` needs PyTorch. Tests also need pytest; plots need matplotlib.
Choose a Python environment, then install the CPU track:

~~~bash
python -m pip install torch matplotlib pytest
python -m pytest tests/ -q
python experiments/toy_smc_cfg.py --quick --out results/quick
python experiments/toy_smc_cfg.py --out results/toy
~~~

For a virtual environment:

~~~bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
~~~

For the full dependency set, use `python -m pip install -r requirements.txt`,
or `conda env create -f environment.yml` followed by `conda activate clg`.
Select and verify a suitable PyTorch GPU build using [VLAB.md](VLAB.md)
before running a trained model.

If an existing pytest cache is inaccessible, use
`python -m pytest tests/ -q -p no:cacheprovider`.
Outputs are ignored by Git. Use separate output directories for comparisons;
partial toy runs retain per-experiment configuration metadata.

## Use

~~~python
from cfgctrl import SlidingModeGuidance, presets

ctrl = SlidingModeGuidance(presets.paper(lam=6.0, k=0.1))
# Alternative candidate:
ctrl = SlidingModeGuidance(presets.boundary_layer_excess())
# Or plain CFG:
ctrl = SlidingModeGuidance(presets.cfg_baseline())

ctrl.reset()  # once per independent sampling trajectory
v_hat = ctrl.guided_velocity(v_uncond, v_cond, w=7.5)
~~~

Errors have shape `[batch, ...]`, including one scalar per sample as `[batch]`.
Controller state must be reset before changing batch shape, device, or working
precision. Half-precision surfaces and RMS calculations use float32.
The controller preserves the input error's identity gradient and output dtype;
it is not a differentiable control-law training implementation.

## Layout

~~~text
cfgctrl/controllers.py      guidance recurrence, variants, diagnostics
cfgctrl/toy_flow.py         analytic Gaussian-mixture velocity and metrics
cfgctrl/diffusers_hook.py   adapter for active doubled-batch CFG
experiments/toy_smc_cfg.py  five CPU studies, CSV/JSON/figures
experiments/real_model.py   GPU integration verification and image grid
tests/                     reference, numerical, experiment and adapter regressions
docs/                      corrected review and improvement roadmap
VLAB.md                    GPU setup and verification
~~~

Read the controller and its reference test first, then the review and toy plant.

## Real-model verification

The adapter's algebra is covered by dummy-denoiser tests. Its behavior with a
trained checkpoint still needs verification on the target environment.

~~~bash
python experiments/real_model.py verify
python experiments/real_model.py grid --w 1.5 2.0 3.0 4.5 7.0 --seeds 0 1 2
~~~

Which guidance laws run is a command-line argument, not a code edit:
`--arms cfg paper` for the published law against its baseline,
`--arms cfg "flux=paper:k=0.7" "excess:k=0.3"` to name and parameterize each
arm. Add `--dry-run` to resolve the matrix without loading a model, and
`--resume` to continue a job that hit a time limit. See [VLAB.md](VLAB.md).

Run `verify` before a grid. The grid requires active doubled-batch CFG and
scales strictly greater than one. Pipelines commonly skip the unconditional
branch at `w=1`; testing the paper law there requires a direct sampler that
explicitly evaluates both predictions.

Flux-style separate forward passes are unsupported by this adapter.
Do not interpret a no-op hook as an experimental comparison.
See [VLAB.md](VLAB.md) for model loading, integration checks, and measurement.
