"""Week 1: plant identification. Run this BEFORE writing any control code.

You cannot design a controller for a plant you have not characterised. This
sweep holds the guidance strength FIXED (arm A0) and measures how the plant
responds. It answers three questions, in order of how badly you need them:

  1. PLANT GAIN g.  Slope of final p against u. Appears directly in the
     variance prediction Var(d)/(1+g*K_p)^2, so you cannot choose K_p without
     it. Everything in Weeks 2-3 depends on this number.

  2. MONOTONICITY.  *** THE GO/NO-GO. *** Is final p monotone increasing in u?
     If pushing harder ever REDUCES measured progress, there is a
     sign-inversion region, the loop can run away, and the failure mode is
     real rather than hypothetical. It is the same structure as the
     self-extinction result in the thesis, where suppression erased the very
     variance that triggered it (3 of 3 runs against a 1-in-3 base rate).
     If this fails across the whole range, STOP. That is itself a reportable
     finding about score distillation.

  3. BASELINE VARIANCE.  Var(p_final) at each u with n=3. This is the number
     the closed loop has to beat. Measuring it now means the headline claim
     has a reference point from week one.

Cost: 20 pairs x 5 levels x 3 seeds = 300 runs. Cheap.

!! This script is WRITTEN BUT NEVER EXECUTED. The dc/ loading and PIE-Bench
!! reading below are marked TODO and must be filled in against the vendored
!! API and the downloaded benchmark.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import List

import torch

from plants.plant_a_latent import ControlledDC, RunConfig, run_plant_a


def build_dc(device: str) -> ControlledDC:
    """TODO: construct DCConfig + ControlledDC exactly as the thesis pipeline
    does. Mirror nerfstudio/3d_editing/dc_nerf/pipelines/dc_pipeline.py so the
    two plants share the configuration, not just the code.

    Pin and record: guidance_scale (7.5), image_guidance_scale (1.5),
    num_inference_steps, min/max_step_ratio, eta_tag, adaptive_tag,
    stg_enabled. Any of these differing between plants breaks the transfer
    claim as surely as a code difference would.
    """
    raise NotImplementedError("see docstring; mirror dc_pipeline.py")


def load_subset(path: Path) -> List[dict]:
    """TODO: read bench/subset_50.json -> list of dicts with keys:
    image_path, src_prompt, tgt_prompt, instruction, mask_path, edit_type.
    Take the first `--n-pairs` of them for this sweep.
    """
    with open(path) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=Path, default=Path("bench/subset_50.json"))
    ap.add_argument("--out", type=Path, default=Path("results/week1_plant_id.csv"))
    ap.add_argument("--n-pairs", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--u-levels", type=float, nargs="+",
                    default=[0.0, 1.75, 3.5, 5.25, 7.0],
                    help="span the usable range; 3.5 is the thesis nominal")
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pairs = load_subset(args.subset)[: args.n_pairs]
    dc = build_dc(args.device)

    total = len(pairs) * len(args.u_levels) * len(args.seeds)
    print(f"{total} runs: {len(pairs)} pairs x {len(args.u_levels)} levels "
          f"x {len(args.seeds)} seeds")

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_id", "edit_type", "u", "seed",
                    "p_final", "s_mean", "loss_final", "tripped"])

        for i, (pair, u, seed) in enumerate(
            itertools.product(pairs, args.u_levels, args.seeds), start=1
        ):
            # TODO: encode the source image -> src_x0, src_encoded, mirroring
            # dc_pipeline.py. Getting src_encoded wrong silently changes the
            # plant, so verify it against a known-good 3D run first.
            src_x0, src_encoded = None, None
            raise NotImplementedError("wire up image loading and encoding")

            _, trace = run_plant_a(
                dc=dc,
                src_x0=src_x0,
                src_encoded=src_encoded,
                src_prompt=pair["src_prompt"],
                tgt_prompt=pair["tgt_prompt"],
                cfg=RunConfig(iterations=args.iterations, seed=seed, fixed_u=u),
            )

            s_mean = sum(trace.s) / len(trace.s) if trace.s else float("nan")
            w.writerow([pair.get("image_path"), pair.get("edit_type"), u, seed,
                        trace.p_final, s_mean, trace.loss[-1], trace.tripped])
            fh.flush()
            print(f"[{i}/{total}] u={u} seed={seed} p_final={trace.p_final:.4f}")

    print(f"\nwrote {args.out}")
    print("\nNext, from this CSV:")
    print("  1. plot mean p_final vs u        -> slope is g")
    print("  2. check monotonicity per pair   -> GO/NO-GO for the whole project")
    print("  3. std of p_final per (pair, u)  -> the variance the loop must beat")


if __name__ == "__main__":
    main()
