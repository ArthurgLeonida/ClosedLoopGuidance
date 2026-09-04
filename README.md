# Closed-Loop Guidance in Score Distillation

Feedback control of guidance strength when the plant is an **optimisation**
rather than a sampling trajectory.

The guidance primitives are inherited from my undergraduate thesis and vendored
unchanged (see [PROVENANCE.md](PROVENANCE.md)). **The control layer is new.**
That line is where the contribution starts.

---

## What this is asking

The control view of diffusion guidance is already established for *sampling*:
[CFG-Ctrl](https://arxiv.org/abs/2603.03281) (CVPR 2026) shows that vanilla
classifier-free guidance is a fixed-gain proportional controller and proposes a
sliding-mode replacement; other work does closed-loop or state-dependent
guidance during denoising.

All of it operates on a ~50-step trajectory that produces one image. Score
distillation is a different plant:

|  | sampling | score distillation |
|---|---|---|
| horizon | ~50 steps | 200–3000 iterations |
| state | transient | accumulates, cannot be reset |
| input | bounded, rarely saturates | saturates |
| per-step quality oracle | available | not available |

Integral action and anti-windup only *mean* anything in the second column.
So the question is what changes when you close the loop there, and the claim is
evaluated on an axis the sampling literature does not report: **variance across
seeds**.

The prediction, from the sensitivity function of a linearised loop:

```
open loop:    Var(p) = Var(d)
closed loop:  Var(p) = Var(d) / (1 + g*K_p)^2
```

Seed-to-seed variance of the final metrics should fall roughly as
`1/(1+g*K_p)^2` as `K_p` rises from zero, then rise again once
measurement-noise amplification dominates. That curve is the paper. Either the
sweep produces it or it does not.

Full design document: `docs/ClosedLoopGuidance.md` in the thesis repository.

---

## Two plants, one controller

```
Plant A   DDS on an image latent     ~200-500 iters   cheap    n = 20+ seeds
Plant B   DDS on a NeRF              3000 iters       costly   n = 3-5 seeds
          same controller, same gains, no retuning
```

Plant A is **not** 2D sampling. There is no denoising loop; it optimises a
persistent latent with the delta-denoising gradient. That is deliberate: 2D
sampling is the crowded setting, 2D score distillation is not.

---

## Layout

```
dc/            vendored guidance, byte-identical, DO NOT EDIT
control/       NEW: measurement, reference trajectory, PI controller
cfgctrl/       NEW: CFG-Ctrl (SMC-CFG, CVPR 2026) reimplemented + refinements,
               analytic flow-matching toy plant, diffusers seam
plants/        Plant A driver; ControlledDC injects u by subclassing
bench/         PIE-Bench data + evaluation (not committed)
experiments/   run scripts (week1_plant_id.py needs a GPU; toy_smc_cfg.py does not)
tests/         pure-logic tests, runnable without a GPU
results/       run outputs (gitignored)
docs/          design docs, the CFG-Ctrl paper, and the review/improvement notes
```

The control input enters through a single seam:
`DC._get_current_stg_scale()` — one method returning one float.
`plants.plant_a_latent.ControlledDC` overrides it. Nothing under `dc/` is
modified, which is what keeps the two plants provably identical.

### `cfgctrl/`: the sampling-side baseline, reimplemented and audited

[CFG-Ctrl](https://arxiv.org/abs/2603.03281) is the closest published work, so it
is reimplemented here from the paper and the authors' code, model-agnostically:
the controller only ever sees the semantic error `e = v_cond - v_uncond` and
returns the corrected error, and `k = 0` reproduces plain CFG bit-exactly.

```python
from cfgctrl import SlidingModeGuidance, presets
ctrl = SlidingModeGuidance(presets.paper(lam=6.0, k=0.1))          # Algorithm 1
ctrl = SlidingModeGuidance(presets.boundary_layer(lam=6.0, k=0.1))  # chatter-free refinement
ctrl = SlidingModeGuidance(presets.boundary_layer_excess(lam=6.0, k=0.1))  # + exactly CFG at w=1
v_hat = ctrl.guided_velocity(v_uncond, v_cond, w=7.5, dt=sigma_prev - sigma)
```

Because no GPU is available here, the law is *measured* on an analytic
Gaussian-mixture flow-matching plant (`cfgctrl/toy_flow.py`) where the
conditional and unconditional velocity fields, the Jacobian of `e`, and the
target distribution are all exact:

```bash
python -m pytest tests/ -q                       # 38 tests, ~30 s, CPU
python experiments/toy_smc_cfg.py --quick        # a few minutes, results/toy/
python experiments/toy_smc_cfg.py                # ~25 min, the numbers in the docs
```

What it found, and what to do about it, is written up in
`docs/CFG-Ctrl_Review_and_Improvements.md` (also a control-theory refresher).
Short version: the paper's law reduces to a per-element sign-shrink of the
guidance vector, its Lyapunov argument does not describe the loop it runs in,
and a boundary layer (soft-threshold) plus measured-error memory fixes the two
concrete defects at zero cost.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # or conda
pip install -r requirements.txt

# Sanity: the only thing runnable without a GPU
python -m pytest tests/ -v
```

`tests/test_controller.py::test_reduces_to_baseline_exactly` is the important
one. `K_p = K_i = 0` must return `u_nom` on every call, so the baseline is a
special case of the method. If that fails, every downstream number is
meaningless.

### Data

PIE-Bench, from [PnPInversion](https://cure-lab.github.io/PnPInversion/)
(ICLR 2024): 700 images, 10 editing types, each with a **source prompt, target
prompt, editing instruction, edit subjects and an editing mask**. The
source/target pair is what DDS consumes, the instruction is what
InstructPix2Pix consumes, and the mask gives background-preservation metrics
comparable to the 3D pipeline's. Place it under `bench/pie_bench/` and use
their evaluation script so numbers sit next to published tables.

---

## Status

Week 0. `dc/` is copied and verified byte-identical. `control/` is written and
unit-tested. `plants/plant_a_latent.py` is **written but never executed** —
every `TODO(verify)` in it is an assumption about the vendored API that must be
checked on the first GPU run.

`cfgctrl/` (added 2026-09-03) is unit-tested and its toy-plant experiments have
run to completion on CPU. Its diffusers hook has **not** been executed against a
real model. Nothing involving the score-distillation plant has produced a
result yet.
