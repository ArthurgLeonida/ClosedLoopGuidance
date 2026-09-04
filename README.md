# CFG-Ctrl, reimplemented and measured

A control-engineering study of

> **CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance**
> Wang, Liu, Chi, Liu, Xue, Duan. CVPR 2026. [arXiv:2603.03281](https://arxiv.org/abs/2603.03281)

The paper writes the flow-matching sampler as a controlled ODE, shows that
classifier-free guidance is a **proportional controller** with gain `w` acting
on the semantic error `e = v(x,t,c) - v(x,t,∅)`, and replaces the fixed gain
with a **sliding-mode** correction:

```
s       = ė + λe                    sliding surface     (Eq. 19)
Δe      = -k · sign(s)              switching control   (Eq. 25)
v̂       = v_uncond + w · (e + Δe)   applied velocity    (Alg. 1)
```

This repository reimplements that law, then measures it on a plant where the
answer is known. What the measurements found, and what to do about it, is in
[docs/CFG-Ctrl_Review_and_Improvements.md](docs/CFG-Ctrl_Review_and_Improvements.md),
which doubles as a control-theory refresher: every concept the paper uses is
re-derived where it first appears.

---

## The short version

- **The sliding surface is essentially `λe`.** With `λ = 6` over 30 steps the
  derivative term decides the sign of `s` in about 1–2 % of elements, so the
  law reduces to a per-element **sign-shrink** `e ← e - k·sign(e)`. This is
  also why the paper's own `λ` ablation is flat.
- **It chatters, and the chatter is self-inflicted.** The authors store the
  *corrected* error as memory. Once `|e| < k` the surface is dominated by the
  previous correction, so every element flips sign each step and `rms(s)`
  plateaus at `(λ-1)k` instead of reaching zero.
- **The feedback loop the proof needs does not exist.** The measured one-step
  gain from the correction to the next error is *negative* and small
  (median `dt·w·‖J‖` of 0.006–0.045 against `k = 0.1`). Assumption 2 holds 0 %
  of the time; the reaching condition holds 3–16 %. The law works as
  **open-loop shaping of the guidance direction**, not as a sliding mode.
- **Where it helps, it helps as L1 shrinkage of guidance** — which is partly
  just a smaller effective `w`. Comparing at one fixed `w`, as the paper's
  Table 2 does, cannot separate the two.

Three refinements survived measurement and are implemented:

| refinement | what it does | measured effect |
|---|---|---|
| `switching="sat"` | boundary layer of half-width `φ = kλ`; makes the law exact **soft-thresholding** | `rms(s)` → 0.010 instead of a 0.50 plateau; switching activity 0.03 of full chatter at every `k`, vs 0.39–0.83 |
| `store_corrected=False` | remember the measured error, not the corrected one | removes the alternation entirely |
| `excess_only=True` | shrink only the extrapolation `(w-1)e` | **exactly CFG at `w = 1`**, which the paper's law is not (it is strictly dominated there) |

---

## Layout

```
cfgctrl/
    controllers.py    the guidance law + the three refinements   (k = 0 → plain CFG)
    toy_flow.py       analytic Gaussian-mixture flow plant, closed-form everything
    diffusers_hook.py the seam for real models (NEVER RUN ON ONE — see below)
experiments/
    toy_smc_cfg.py    the five experiments — CPU, finished
    real_model.py     drives a diffusers pipeline — GPU, never executed
tests/                20 tests, the executable specification
docs/                 the review + control-theory refresher
VLAB.md               setup and run instructions for the GPU lab
```

`cfgctrl/` depends only on `torch`. The controller never sees an image, a
model or a scheduler — just the error tensor `e` — which is what lets the same
code run against the toy plant and against SD3.5.

---

## Setup

```bash
conda env create -f environment.yml && conda activate clg     # or:
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

python -m pytest tests/ -q                   # 20 tests, ~5 s, CPU
python experiments/toy_smc_cfg.py --quick    # ~20 s
python experiments/toy_smc_cfg.py            # ~45 s, writes results/toy/
```

Everything above is CPU-only. For the GPU track see [VLAB.md](VLAB.md).

## Use

```python
from cfgctrl import SlidingModeGuidance, presets

ctrl = SlidingModeGuidance(presets.paper(lam=6.0, k=0.1))     # the paper, Algorithm 1
ctrl = SlidingModeGuidance(presets.boundary_layer_excess())   # the recommended law
ctrl = SlidingModeGuidance(presets.cfg_baseline())            # k = 0, plain CFG

v_hat = ctrl.guided_velocity(v_uncond, v_cond, w=7.5)
```

`k = 0` reproduces classifier-free guidance **bit-exactly**, so the baseline is
a special case of the method and any measured difference is attributable to the
correction. `tests/test_smc_cfg.py::test_k_zero_is_exact_cfg_for_every_variant`
is the check; if it ever fails, no number in the repository means anything.

---

## Reading order

About 1,300 lines of Python, and the dependency graph is a line: everything
imports `controllers.py`, and `controllers.py` imports only `torch`. Read it in
one sitting, in this order.

**1. `cfgctrl/controllers.py` — start here (245 lines, ~30 min).**
The whole method. Read the module docstring first: it states the paper's law in
five lines and then each surviving refinement with the measurement that
justified it. Then read exactly two things:

- `SMCConfig` — seven fields. Three are the paper (`lam`, `k`, `switching`),
  four are refinements. There is no hidden state anywhere else.
- `SlidingModeGuidance.correct()` — ~40 lines and the only method that matters.
  It has five beats, in order: **measure** `e` → **build the surface**
  `s = (e - e_prev) + λ·e_prev` → **switch** (`sign` or `sat`) → **scale by the
  gain** → **apply and remember**. Everything the review document argues about
  is visible in those five beats.

The one thing to hold onto: the controller never sees an image, a model, a
scheduler or a timestep. Its entire input is the tensor `e`. That is why the
same code drives the toy plant and SD3.5.

**2. `tests/test_smc_cfg.py` (288 lines).** The executable specification, and
the fastest way to check you understood step 1. Read three tests in particular:
`test_reference_implementation_of_paper_law` (replays the authors' code line by
line and asserts bit-equality — this is what makes the critique fair),
`test_corrected_memory_alternates_once_the_error_is_small` (derives the
chattering mechanism from first principles in ten lines), and
`test_k_zero_is_exact_cfg_for_every_variant` (why any measured difference is
attributable to the correction and nothing else).

**3. `cfgctrl/toy_flow.py` (190 lines).** Read the module docstring, which
derives the closed-form velocity field for a Gaussian mixture. That derivation
is the entire reason the numbers can be trusted: the conditional field, the
unconditional field, the Jacobian of `e` and the target distribution are all
exact, so the paper's assumptions can be *measured* instead of assumed. Then
read `sample()` — a guided Euler loop in 15 lines, the same shape as a
diffusers sampler.

**4. `docs/CFG-Ctrl_Review_and_Improvements.md` (947 lines).** The argument.
§1–2 restate the paper in control language with a refresher block for every
concept where it first appears; §3 is the critique with measurements; §4 the
improvements; §5 the evidence tables. If you read only one section, read §3.

**5. `experiments/toy_smc_cfg.py` (451 lines).** How the numbers in §5 were
produced. `exp_loop_gain()` is the one worth studying: it *measures* the loop
gain that the paper's Theorem 1 assumes, and finds it has the wrong sign.

**6. `cfgctrl/diffusers_hook.py` and `experiments/real_model.py`** — only when
you are ready to run on a GPU. See [VLAB.md](VLAB.md).

A good self-test after step 1: on paper, work out what the law does to a single
element with `λ = 6`, `k = 0.1` when `e = 0.05` — and then what it does to the
same element on the next step. §2.4 of the review has the worked answer, and it
is where the chattering result comes from.

---

## Running against a real model

`cfgctrl/diffusers_hook.py` wraps the denoiser's `forward` so a pipeline's own
CFG line produces the corrected velocity, with no pipeline patching. **Neither
it nor `experiments/real_model.py` has ever been executed against a real
model.** The algebra is unit-tested against a dummy denoiser; the interface to a
real pipeline is an assumption. So the driver script checks itself first:

```bash
python experiments/real_model.py verify        # ~1 min — run this before anything else
python experiments/real_model.py grid --w 1.0 1.5 2.0 3.0 4.5 7.0 --seeds 0 1 2
```

`verify` confirms that `k = 0` reproduces the pipeline bit-identically, works
out which half of the doubled batch is the conditional branch (getting this
backwards silently inverts guidance while still producing plausible images),
and prints the first real measurement of `rms(e)`, `rms(s)` and the chatter
index. `grid` then sweeps arm × scale × prompt × seed, writing images per
(arm, scale) and the per-step signals to `signals.csv`.

Metrics are deliberately not computed: point your own FID / CLIP tooling at the
image directories, then plot fidelity against alignment — one curve per arm,
one point per `w`. That Pareto view is the point (§4.7 of the review); a single
fixed `w`, as in the paper's Table 2, cannot distinguish a better law from a
smaller effective guidance scale.

Direct use of the hook, if you want it in your own loop:

```python
from cfgctrl import SlidingModeGuidance, presets
from cfgctrl.diffusers_hook import attach_smc_cfg

hook = attach_smc_cfg(pipe, SlidingModeGuidance(presets.paper()), guidance_scale=7.5)
hook.reset()                                  # once per image
image = pipe("a photo of a cat", guidance_scale=7.5).images[0]
hook.detach()
```

Flux-dev needs more: with `true_cfg_scale > 1` its pipeline runs two separate
forward passes instead of one doubled batch, which `verify` will detect and
report.

Full lab instructions: [VLAB.md](VLAB.md).
