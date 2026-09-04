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
    toy_smc_cfg.py    the five experiments
tests/                20 tests, the executable specification
docs/                 the review + control-theory refresher
```

`cfgctrl/` depends only on `torch`. The controller never sees an image, a
model or a scheduler — just the error tensor `e` — which is what lets the same
code run against the toy plant and against SD3.5.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest tests/ -q                   # 20 tests, ~5 s, CPU
python experiments/toy_smc_cfg.py --quick    # ~20 s
python experiments/toy_smc_cfg.py            # ~5 min, writes results/toy/
```

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

1. `tests/test_smc_cfg.py` — the executable spec, and the shortest way in.
2. `cfgctrl/controllers.py` — the whole law. Only `correct()` matters:
   measure `e` → build `s` → pick a switching function → scale by a gain → apply.
3. `cfgctrl/toy_flow.py` — read the module docstring, which derives the
   closed-form velocity field. That derivation is why the numbers can be trusted.
4. `experiments/toy_smc_cfg.py` — `exp_loop_gain()` is the interesting one: it
   *measures* the loop gain the paper assumes.
5. `docs/CFG-Ctrl_Review_and_Improvements.md` — the write-up.

---

## Running against a real model

`cfgctrl/diffusers_hook.py` wraps the denoiser's `forward` so a pipeline's own
CFG line produces the corrected velocity, with no pipeline patching. **It has
never been executed against a real model.** The algebra is unit-tested against
a dummy denoiser; the interface is not. Before trusting any output:

1. Run with `presets.cfg_baseline()`. The hook short-circuits, so the image must
   be **bit-identical** to running without the hook at the same seed.
2. Confirm the batch order. `uncond_first=True` assumes diffusers'
   `cat([negative, positive])`. Check which half of the chunk responds to a
   strongly negative prompt — backwards silently inverts the guidance direction.
3. Only then enable `presets.paper()`.

```python
from cfgctrl import SlidingModeGuidance, presets
from cfgctrl.diffusers_hook import attach_smc_cfg

hook = attach_smc_cfg(pipe, SlidingModeGuidance(presets.paper()), guidance_scale=7.5)
hook.reset()                                  # once per image
image = pipe("a photo of a cat", guidance_scale=7.5).images[0]
hook.detach()
```

Flux-dev needs more: with `true_cfg_scale > 1` its pipeline runs two separate
forward passes instead of one doubled batch.

The evaluation worth running on a GPU is the **Pareto sweep over `w`** — the
paper's Table 2 comparison done at more than one guidance scale — with seed
error bars and the per-step signals (`rms(e)`, `rms(s)`, switching activity)
logged. That script does not exist yet.
