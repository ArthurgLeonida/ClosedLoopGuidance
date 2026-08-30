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
plants/        Plant A driver; ControlledDC injects u by subclassing
bench/         PIE-Bench data + evaluation (not committed)
experiments/   run scripts
tests/         pure-logic tests, runnable without a GPU
results/       run outputs (gitignored)
```

The control input enters through a single seam:
`DC._get_current_stg_scale()` — one method returning one float.
`plants.plant_a_latent.ControlledDC` overrides it. Nothing under `dc/` is
modified, which is what keeps the two plants provably identical.

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

Nothing here has produced a result yet.
