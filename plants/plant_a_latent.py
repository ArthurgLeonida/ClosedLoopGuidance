"""Plant A: score distillation on a single image latent.

This is NOT 2D sampling. There is no 50-step denoising loop here. The plant is
an optimisation over a persistent latent, exactly like the 3D pipeline, just
with a cheaper parameter space. That distinction is the whole reason this
project is not a reimplementation of CFG-Ctrl / FBG.

    Plant A:  DDS on an image latent   ~200-500 iters   cheap    n = 20+ seeds
    Plant B:  DDS on a NeRF            3000 iters       costly   n = 3-5 seeds

The guidance maths is NOT reimplemented. `dc/` is a byte-identical copy of the
thesis implementation (see PROVENANCE.md), and the control input is injected by
subclassing rather than by editing it. The same subclass pattern works inside
the nerfstudio pipeline for Plant B, so both plants provably share one
implementation of TAG, STG and the edit-strength signal.

!! STATUS: WRITTEN BUT NEVER EXECUTED. No GPU was available when this was
!! drafted. Everything marked TODO(verify) is an assumption about the vendored
!! API that must be checked against dc/dc.py on the first run. Treat a clean
!! run as unproven until the baseline check in experiments/week1_plant_id.py
!! passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from dc.dc import DC, DCConfig
from control import EMAFilter, PIController, ReferenceTrajectory, edit_progress


class ControlledDC(DC):
    """DC with the open-loop STG schedule replaced by an external value.

    `dc.guidance_utils.compute_stg_scale` implements the hand-designed bump.
    `DC._get_current_stg_scale` is the single seam where it enters the guidance
    path: one method, returning one float. Overriding it here is the entire
    control-input change, and it leaves dc/ byte-identical.

    Set `u_override = None` to fall straight back to the inherited schedule,
    which is how arm A1 (the thesis baseline) is run through the same code
    path as everything else.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.u_override: Optional[float] = None

    def _get_current_stg_scale(self, current_edit_strength=None, iteration=None) -> float:
        if self.u_override is None:
            return super()._get_current_stg_scale(
                current_edit_strength=current_edit_strength, iteration=iteration
            )
        return float(self.u_override)


@dataclass
class RunConfig:
    iterations: int = 300
    lr: float = 0.01
    seed: int = 0
    # Arm selection. Exactly one of these should be active:
    #   fixed_u   -> A0, constant guidance (used for Week 1 plant identification)
    #   None      -> A1, the inherited hand-designed schedule
    #   controller-> A2/A3, closed loop
    fixed_u: Optional[float] = None
    use_controller: bool = False
    log_every: int = 1


@dataclass
class RunTrace:
    """Everything Week 1 needs, recorded per iteration."""
    p: List[float] = field(default_factory=list)          # achievement, raw
    p_filtered: List[float] = field(default_factory=list) # after the EMA
    s: List[float] = field(default_factory=list)          # demand, from DC
    u: List[float] = field(default_factory=list)          # applied guidance
    r: List[Optional[float]] = field(default_factory=list)
    e: List[Optional[float]] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)
    tripped: bool = False
    trip_reason: str = ""

    @property
    def p_final(self) -> float:
        return self.p[-1] if self.p else float("nan")

    @property
    def tracking_error(self) -> float:
        """mean |e| over iterations where a reference existed."""
        vals = [abs(v) for v in self.e if v is not None]
        return sum(vals) / len(vals) if vals else float("nan")


def run_plant_a(
    dc: ControlledDC,
    src_x0: torch.Tensor,
    src_encoded: torch.Tensor,
    src_prompt: str,
    tgt_prompt: str,
    cfg: RunConfig,
    controller: Optional[PIController] = None,
    reference: Optional[ReferenceTrajectory] = None,
    ema: Optional[EMAFilter] = None,
) -> tuple[torch.Tensor, RunTrace]:
    """Optimise a latent under DDS and return (final latent, trace).

    Args:
        src_x0:       source latent, from dc.encode_src_image(image_tensor).
                      TODO(verify): confirm which of encode_image /
                      encode_src_image dc/dc.py expects for each argument.
        src_encoded:  IP2P image conditioning for the source.
                      TODO(verify): confirm how dc_pipeline.py builds this and
                      mirror it exactly. Getting this wrong silently changes
                      the plant.
    """
    torch.manual_seed(cfg.seed)

    if cfg.use_controller and controller is None:
        raise ValueError("use_controller=True requires a controller")

    tgt_x0 = src_x0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([tgt_x0], lr=cfg.lr)

    if ema is None:
        ema = EMAFilter(lam=0.9)
    trace = RunTrace()

    for it in range(cfg.iterations):
        # --- decide the control input for THIS iteration --------------------
        if cfg.fixed_u is not None:
            dc.u_override = cfg.fixed_u                       # arm A0
        elif cfg.use_controller:
            r_t = reference.at(it) if reference is not None else None
            y_t = ema.value
            u = controller.step(reference=r_t, measurement=y_t,
                                raw_progress=trace.p[-1] if trace.p else None)
            dc.u_override = u                                 # arms A2 / A3
        else:
            dc.u_override = None                              # arm A1, inherited

        # --- plant step ------------------------------------------------------
        out = dc(
            tgt_x0,
            src_x0,
            src_encoded,
            tgt_prompt=tgt_prompt,
            src_prompt=src_prompt,
            return_dict=True,
            step=it,
        )
        loss = out["loss"]

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # --- measure ---------------------------------------------------------
        p_t = edit_progress(tgt_x0.detach(), src_x0)
        y_t = ema.update(p_t)
        s_t = float(out.get("edit_strength", float("nan")))

        if reference is not None:
            reference.observe_demand(s_t)

        r_t = reference.at(it) if reference is not None else None
        trace.p.append(p_t)
        trace.p_filtered.append(y_t)
        trace.s.append(s_t)
        trace.u.append(float(out.get("stg_scale", float("nan"))))
        trace.r.append(r_t)
        trace.e.append((r_t - y_t) if r_t is not None else None)
        trace.loss.append(float(loss.detach()))

        if controller is not None and controller.tripped:
            trace.tripped = True
            trace.trip_reason = controller.trip_reason
            break

    return tgt_x0.detach(), trace
