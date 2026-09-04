"""Drive a real diffusers pipeline through cfgctrl. Needs a GPU.

!! WRITTEN BUT NEVER EXECUTED: there is no GPU on the machine where this was
!! written, and `diffusers` is not even installed there. The algebra it relies
!! on is unit-tested against a dummy denoiser (tests/test_smc_cfg.py), but the
!! interface to a real pipeline is an assumption. That is exactly why `verify`
!! exists and why you should run it first. Treat a clean `verify` as the point
!! where this file stops being a guess.

Two modes.

    verify   Three checks on the hook itself, no experiment, ~1 minute. Run
             this before trusting any image:
               1. TRANSPARENCY. With k = 0 the hook short-circuits, so the
                  image must be bit-identical to running with no hook at all
                  at the same seed. Failure means the hook is not being called
                  where you think, or the pipeline is not deterministic.
               2. BATCH ORDER. Runs one step with two different prompts and
                  the same seed, and sees which half of the doubled batch
                  changes. That half is the conditional branch. If it is not
                  the half `uncond_first` assumes, every guidance direction
                  afterwards is inverted -- silently, with plausible-looking
                  images.
               3. SHAPE / DTYPE / STEP COUNT actually observed by the hook.

    grid     Generate images across (arm x guidance scale x prompt x seed) and
             log the controller's internal signals per step. No metrics are
             computed here on purpose: the images land in one directory per
             (arm, w) so you can point any FID / CLIP / ImageReward tool at
             them afterwards, and the per-step CSV is the more interesting
             output anyway -- it is what confirms or refutes section 3 of
             docs/CFG-Ctrl_Review_and_Improvements.md on a real model.

Usage:

    python experiments/real_model.py verify
    python experiments/real_model.py grid --w 1.0 2.0 3.0 5.0 7.5 --seeds 0 1 2
    python experiments/real_model.py grid --model stabilityai/stable-diffusion-3.5-medium
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfgctrl import SMCConfig, SlidingModeGuidance, presets          # noqa: E402
from cfgctrl.diffusers_hook import GuidanceHook                      # noqa: E402

DEFAULT_PROMPTS = [
    "a blue backpack and a brown cow",
    "a man in a revolutionary-era costume talks on a cellphone",
    "a poster with large centered text that reads: WELCOME TO THE FOREST",
    "a deer made of shimmering starlight grazing beside a silver river",
    "a small kitten and a big dog sat side by side",
    "a red tomato and a yellow pepper on a wooden table",
]


def arms(k: float, lam: float) -> Dict[str, SMCConfig]:
    """The three laws worth comparing. `cfg` must be first: it is the baseline
    every other row is measured against."""
    return {
        "cfg": presets.cfg_baseline(),
        "paper": presets.paper(lam, k),
        "excess": presets.boundary_layer_excess(lam, k),
    }


def load_pipe(model: str, dtype: str, device: str):
    from diffusers import DiffusionPipeline

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    print(f"loading {model} ({dtype}) ...", flush=True)
    pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch_dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def denoiser(pipe):
    m = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
    if m is None:
        raise AttributeError("pipeline has neither .transformer nor .unet")
    return m


class _Probe:
    """Record the raw halves of the denoiser output, changing nothing."""

    def __init__(self, model):
        self.model, self.records, self._orig, self._had = model, [], None, False

    def __enter__(self) -> "_Probe":
        self._had = "forward" in self.model.__dict__
        self._orig = self.model.forward

        def fwd(*a, **kw):
            out = self._orig(*a, **kw)
            pred = GuidanceHook._extract(out)
            if pred.shape[0] % 2 == 0:
                first, second = pred.detach().float().cpu().chunk(2)
                self.records.append((first, second, tuple(pred.shape), str(pred.dtype)))
            return out

        self.model.forward = fwd
        return self

    def __exit__(self, *exc) -> None:
        if self._had:
            self.model.forward = self._orig
        else:
            del self.model.forward


def generate(pipe, prompt: str, w: float, steps: int, seed: int, device: str):
    g = torch.Generator(device=device).manual_seed(seed)
    return pipe(prompt, guidance_scale=w, num_inference_steps=steps, generator=g).images[0]


# ------------------------------------------------------------------ verify
def cmd_verify(args) -> int:
    import numpy as np

    pipe = load_pipe(args.model, args.dtype, args.device)
    model = denoiser(pipe)
    w, steps = args.check_w, args.check_steps
    failures: List[str] = []

    # -- 1. transparency ----------------------------------------------------
    print("\n[1/3] transparency: k = 0 must reproduce the pipeline exactly")
    ref = np.asarray(generate(pipe, DEFAULT_PROMPTS[0], w, steps, 0, args.device))
    hook = GuidanceHook(model, SlidingModeGuidance(presets.cfg_baseline()),
                        guidance_scale=w).attach()
    try:
        got = np.asarray(generate(pipe, DEFAULT_PROMPTS[0], w, steps, 0, args.device))
    finally:
        hook.detach()
    if np.array_equal(ref, got):
        print("      PASS  bit-identical")
    else:
        d = np.abs(ref.astype(int) - got.astype(int))
        failures.append(f"transparency: images differ (max {d.max()}, mean {d.mean():.4f})")
        print(f"      FAIL  max |diff| = {d.max()}, mean = {d.mean():.4f}")
        print("            the pipeline may be non-deterministic, or the hook is")
        print("            attached somewhere the pipeline does not call.")

    # -- 2. batch order -----------------------------------------------------
    print("\n[2/3] batch order: which half of the doubled batch is conditional?")
    caps = []
    for prompt in (DEFAULT_PROMPTS[0], DEFAULT_PROMPTS[3]):
        with _Probe(model) as probe:
            generate(pipe, prompt, w, 1, 0, args.device)
        if not probe.records:
            failures.append("batch order: the denoiser never saw an even batch -- "
                            "this pipeline does not do doubled-batch CFG, so the hook "
                            "cannot work as written (Flux with true_cfg_scale is like this)")
            print("      FAIL  no doubled batch observed; see the note in diffusers_hook.py")
            caps = []
            break
        caps.append(probe.records[0])
    if caps:
        d_first = float((caps[0][0] - caps[1][0]).abs().mean())
        d_second = float((caps[0][1] - caps[1][1]).abs().mean())
        cond_is_second = d_second > d_first
        print(f"      prompt sensitivity: first half {d_first:.6f}, second half {d_second:.6f}")
        print(f"      -> conditional branch is the {'second' if cond_is_second else 'first'} half"
              f"  =>  uncond_first={cond_is_second}")
        if cond_is_second:
            print("      PASS  matches the default uncond_first=True")
        else:
            failures.append("batch order: conditional is the FIRST half; "
                            "pass uncond_first=False to GuidanceHook")
            print("      FAIL  pass uncond_first=False, or guidance is inverted")
        if min(d_first, d_second) > 0.5 * max(d_first, d_second):
            print("      WARN  the two halves respond almost equally; this test is "
                  "inconclusive, inspect manually")

    # -- 3. what the hook actually sees -------------------------------------
    print("\n[3/3] what the hook sees over a full run")
    ctrl = SlidingModeGuidance(presets.paper(args.lam, args.k))
    hook = GuidanceHook(model, ctrl, guidance_scale=w).attach()
    try:
        generate(pipe, DEFAULT_PROMPTS[0], w, steps, 0, args.device)
    finally:
        hook.detach()
    if not ctrl.history:
        failures.append("the controller was never called during a full run")
        print("      FAIL  controller.history is empty")
    else:
        h = ctrl.history
        print(f"      steps seen        {len(h)}  (requested {steps})")
        print(f"      latent shape      {caps[0][2] if caps else 'n/a'}  dtype {caps[0][3] if caps else 'n/a'}")
        print(f"      rms(e)            {h[0].e_rms:.4f} -> {h[-1].e_rms:.4f}")
        print(f"      rms(s)            {h[0].s_rms:.4f} -> {h[-1].s_rms:.4f}")
        print(f"      chatter (last 5)  {sum(x.chatter for x in h[-5:]) / 5:.3f}")
        print(f"      deriv decides     {sum(x.deriv_matters for x in h) / len(h) * 100:.1f} %")
        if len(h) != steps:
            print(f"      WARN  {len(h)} controller calls for {steps} requested steps")
        print("\n      Compare against the toy plant (docs section 3): a mean "
              "derivative-decides\n      index near 1-2 % and a late chatter near 1.0 would "
              "reproduce the finding\n      that the paper's law is a sign-shrink that "
              "chatters.")

    print("\n" + "=" * 70)
    if failures:
        print("VERIFY FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("VERIFY PASSED -- the hook is safe to use on this pipeline.")
    return 0


# -------------------------------------------------------------------- grid
def cmd_grid(args) -> int:
    pipe = load_pipe(args.model, args.dtype, args.device)
    model = denoiser(pipe)
    prompts = ([p.strip() for p in Path(args.prompts).read_text(encoding="utf-8").splitlines() if p.strip()]
               if args.prompts else DEFAULT_PROMPTS)
    table = arms(args.k, args.lam)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    total = len(table) * len(args.w) * len(prompts) * len(args.seeds)
    print(f"{total} images: {len(table)} arms x {len(args.w)} scales x "
          f"{len(prompts)} prompts x {len(args.seeds)} seeds")
    (out / "config.json").write_text(json.dumps({
        "model": args.model, "dtype": args.dtype, "steps": args.steps, "k": args.k,
        "lam": args.lam, "w": args.w, "seeds": args.seeds, "arms": list(table),
        "prompts": prompts,
    }, indent=1), encoding="utf-8")

    sig_path = out / "signals.csv"
    with open(sig_path, "w", newline="") as fh:
        wri = csv.writer(fh)
        wri.writerow(["arm", "w", "prompt_id", "seed", "step", "e_rms", "s_rms",
                      "delta_rms", "chatter", "switch_activity", "deriv_matters", "k_eff"])
        done, t0 = 0, time.time()
        for arm, cfg in table.items():
            ctrl = SlidingModeGuidance(cfg)
            hook = GuidanceHook(model, ctrl).attach()
            try:
                for w in args.w:
                    hook.guidance_scale = w                  # excess_only needs the real w
                    d = out / arm / f"w{w}"
                    d.mkdir(parents=True, exist_ok=True)
                    for pid, prompt in enumerate(prompts):
                        for seed in args.seeds:
                            hook.reset()
                            img = generate(pipe, prompt, w, args.steps, seed, args.device)
                            img.save(d / f"p{pid:02d}_s{seed}.png")
                            for st in ctrl.history:
                                wri.writerow([arm, w, pid, seed, st.step,
                                              f"{st.e_rms:.6f}", f"{st.s_rms:.6f}",
                                              f"{st.delta_rms:.6f}", f"{st.chatter:.4f}",
                                              f"{st.switch_activity:.6f}",
                                              f"{st.deriv_matters:.4f}", f"{st.k_eff:.6f}"])
                            done += 1
                            if done % 10 == 0 or done == total:
                                el = time.time() - t0
                                print(f"  [{done}/{total}] {el:.0f}s elapsed, "
                                      f"~{el / done * (total - done):.0f}s left", flush=True)
                            fh.flush()
            finally:
                hook.detach()

    print(f"\nwrote {out}/  (images per arm/scale, signals in {sig_path.name})")
    print("\nNext: point a metric tool at the image directories, e.g. CLIP score for")
    print("alignment and FID against your reference set for fidelity, then plot")
    print("fidelity against alignment -- one curve per arm, one point per w. A law")
    print("only helps if its curve lies below CFG's (docs section 4.7).")
    return 0


# -------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["verify", "grid"])
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=30, help="the paper uses 30")
    ap.add_argument("--k", type=float, default=0.1, help="0.1 for SD3.5/Qwen, 0.7 for Flux")
    ap.add_argument("--lam", type=float, default=6.0)
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--w", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0, 4.5, 7.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--prompts", default=None, help="file with one prompt per line")
    ap.add_argument("--check-w", type=float, default=7.0, help="guidance scale used by verify")
    ap.add_argument("--check-steps", type=int, default=8, help="steps used by verify")
    args = ap.parse_args()
    return cmd_verify(args) if args.mode == "verify" else cmd_grid(args)


if __name__ == "__main__":
    sys.exit(main())
