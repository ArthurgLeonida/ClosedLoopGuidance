"""Drive a real diffusers pipeline through cfgctrl. Needs a GPU.

!! WRITTEN BUT NEVER EXECUTED: there is no GPU on the machine where this was
!! written. Offline tests cover dummy pipeline integration; real weights,
!! scheduler behavior and image quality still require a GPU run. Run `verify`
!! first; a pass covers the checked settings, not every pipeline feature.

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
    python experiments/real_model.py grid --w 1.5 2.0 3.0 5.0 7.5 --seeds 0 1 2
    python experiments/real_model.py grid --model stabilityai/stable-diffusion-3.5-medium

Only w > 1 is supported: standard SD pipelines omit the unconditional branch
at w <= 1, so a denoiser hook cannot evaluate the paper's endpoint correction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfgctrl import SMCConfig, SlidingModeGuidance, presets          # noqa: E402
from cfgctrl.diffusers_hook import GuidanceHook, attach_smc_cfg      # noqa: E402

DEFAULT_PROMPTS = [
    "a blue backpack and a brown cow",
    "a man in a revolutionary-era costume talks on a cellphone",
    "a poster with large centered text that reads: WELCOME TO THE FOREST",
    "a deer made of shimmering starlight grazing beside a silver river",
    "a small kitten and a big dog sat side by side",
    "a red tomato and a yellow pepper on a wooden table",
]


# --------------------------------------------------------------- arm specs
# An "arm" is one guidance law to compare. Arms are given on the command line
# as strings so a new comparison needs no code change:
#
#     paper                      the published law at the run's --lam/--k
#     paper:k=0.7                the published law at Flux's gain
#     excess                     boundary layer on the extrapolation only
#     flux=paper:k=0.7           the same, but written to results/flux/
#     bl:switching=sign,phi=0    any SMCConfig field, comma separated
#
# Grammar: [name=]preset[:field=value,...]. `lam` and `k` are applied before
# the preset builds its derived values, so `excess:k=0.7` gets the matching
# boundary layer phi = k*lam rather than one left over from the default k.
PRESETS = {
    "cfg": presets.cfg_baseline,
    "paper": presets.paper,
    "boundary_layer": presets.boundary_layer,
    "excess": presets.boundary_layer_excess,
}
_ALIASES = {"cfg_baseline": "cfg", "bl": "boundary_layer", "sat": "boundary_layer",
            "boundary_layer_excess": "excess"}
# The baseline first: every other arm is read relative to it.
DEFAULT_ARMS = ["cfg", "paper", "excess"]
_FLOAT_FIELDS = ("lam", "k", "phi")
_BOOL_FIELDS = ("store_corrected", "relative_gain", "excess_only")
_BOOLS = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}


def _coerce(field: str, raw: str):
    if field in _FLOAT_FIELDS:
        try:
            return float(raw)
        except ValueError:
            raise ValueError(f"{field}={raw!r} is not a number") from None
    if field == "switching":
        if raw not in ("sign", "sat"):
            raise ValueError(f"switching={raw!r} must be 'sign' or 'sat'")
        return raw
    if field in _BOOL_FIELDS:
        value = _BOOLS.get(raw.lower())
        if value is None:
            raise ValueError(f"{field}={raw!r} must be true or false")
        return value
    known = ", ".join(_FLOAT_FIELDS + ("switching",) + _BOOL_FIELDS)
    raise ValueError(f"unknown controller field {field!r}; known fields: {known}")


def parse_arm(spec: str, lam: float, k: float) -> Tuple[str, SMCConfig]:
    """Turn one `[name=]preset[:field=value,...]` string into (name, config)."""
    head, _, override_text = spec.partition(":")
    first, eq, second = head.partition("=")
    # "name=preset" splits in two; a bare "preset" leaves everything in `first`.
    name, preset = (first.strip(), second.strip()) if eq else ("", first.strip())
    preset = _ALIASES.get(preset, preset)
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r} in arm {spec!r}; "
                         f"choose from {', '.join(sorted(PRESETS))}")
    name = name or preset
    if "/" in name or "\\" in name:
        raise ValueError(f"arm name {name!r} cannot contain a path separator")

    overrides: Dict[str, object] = {}
    for item in override_text.split(","):
        item = item.strip()
        if not item:
            continue
        field, eq, raw = item.partition("=")
        if not eq:
            raise ValueError(f"override {item!r} in arm {spec!r} must be field=value")
        field = field.strip()
        if field in overrides:
            raise ValueError(f"arm {spec!r} sets {field!r} twice")
        overrides[field] = _coerce(field, raw.strip())

    # Build the preset at the effective lam/k so its derived phi matches, then
    # lay every explicit override on top.
    lam_eff = float(overrides.get("lam", lam))
    k_eff = float(overrides.get("k", k))
    cfg = PRESETS[preset]() if preset == "cfg" else PRESETS[preset](lam_eff, k_eff)
    cfg = replace(cfg, **overrides)
    cfg.validate()
    return name, cfg


def parse_arms(specs, lam: float, k: float) -> Dict[str, SMCConfig]:
    """Resolve every arm spec, rejecting duplicate names (they share a
    directory and would overwrite one another's images)."""
    table: Dict[str, SMCConfig] = {}
    for spec in specs:
        name, cfg = parse_arm(spec, lam, k)
        if name in table:
            raise ValueError(f"two arms are both named {name!r}; "
                             f"give one an explicit name, e.g. myname={spec}")
        table[name] = cfg
    if not table:
        raise ValueError("at least one arm is required")
    return table


def describe_arm(name: str, cfg: SMCConfig) -> str:
    if cfg.is_cfg:
        return f"{name:<12} plain CFG (k = 0, the baseline every other arm is measured against)"
    bits = [f"lam={cfg.lam:g}", f"k={cfg.k:g}", f"switching={cfg.switching}"]
    if cfg.switching == "sat":
        bits.append(f"phi={cfg.phi:g}")
    if not cfg.store_corrected:
        bits.append("measured-error memory")
    if cfg.excess_only:
        bits.append("extrapolation only")
    if cfg.relative_gain:
        bits.append("relative gain")
    return f"{name:<12} " + "  ".join(bits)


def preflight(args) -> None:
    """Fail before downloading tens of gigabytes.

    A checkpoint download is the slow, expensive part of a first run, and the
    two things that most often stop that run are visible beforehand: a torch
    build the driver cannot run, and a gated repository.
    """
    device = str(getattr(args, "device", "cuda"))
    print(f"torch {torch.__version__}, built for CUDA {torch.version.cuda}", flush=True)
    if not device.startswith("cuda"):
        return
    if torch.cuda.is_available():
        print(f"device {torch.cuda.get_device_name()}, "
              f"bf16 {torch.cuda.is_bf16_supported()}", flush=True)
        return
    raise RuntimeError(
        "torch cannot use CUDA, so this run would download the checkpoint and then "
        f"fail.\n  This torch is built for CUDA {torch.version.cuda}.\n"
        "  Compare it with the driver's supported version from `nvidia-smi`. A wheel "
        "built for a NEWER CUDA than the driver supports is reported as the driver "
        "being 'too old', which is the usual cause.\n"
        "  Fix by installing a torch build the driver supports, e.g. for a CUDA 12.8 "
        "driver:\n"
        "    pip install --force-reinstall torch "
        "--index-url https://download.pytorch.org/whl/cu128\n"
        "  Or pass --device cpu for a correctness smoke test (far too slow for a grid)."
    )


def _explain_load_failure(exc: BaseException, model: str) -> Optional[RuntimeError]:
    """Translate an opaque download failure into the action that fixes it."""
    text, name = str(exc), type(exc).__name__
    if "Gated" in name or "restricted" in text or "401" in text or "gated" in text:
        return RuntimeError(
            f"cannot access {model}: the repository is gated.\n"
            "  1. Open its page on huggingface.co and accept the licence with the "
            "same account you will authenticate as.\n"
            "  2. Authenticate in this environment: `huggingface-cli login` (older "
            "versions) or `hf auth login`, or export HF_TOKEN=<a read token>.\n"
            "  3. Confirm with: python -c \"from huggingface_hub import whoami; "
            "print(whoami()['name'])\"\n"
            "  Or pass --model with a checkpoint you already have access to."
        )
    if "Repository Not Found" in text or "404" in text:
        return RuntimeError(f"no such checkpoint: {model!r}. Check --model for a typo.")
    return None


def load_pipe(model: str, dtype: str, device: str):
    from diffusers import DiffusionPipeline

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    print(f"loading {model} ({dtype}) ...", flush=True)
    try:
        pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch_dtype)
    except Exception as exc:                       # noqa: BLE001 - re-raised below
        explained = _explain_load_failure(exc, model)
        if explained is None:
            raise
        raise explained from exc
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def denoiser(pipe):
    m = getattr(pipe, "transformer", None)
    if m is None:
        m = getattr(pipe, "unet", None)
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
            # verify generates one image, so its doubled batch is exactly two.
            if pred.shape[0] == 2:
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


def _validate_scales(scales, steps: int) -> None:
    if steps <= 0:
        raise ValueError("the number of inference steps must be positive")
    if not scales or any(not math.isfinite(w) or w <= 1 for w in scales):
        raise ValueError("this doubled-batch CFG runner requires finite guidance scales w > 1; "
                         "at w <= 1 standard SD pipelines omit the unconditional branch "
                         "and cannot evaluate the paper's correction")


# ------------------------------------------------------------------ verify
def cmd_verify(args) -> int:
    import numpy as np

    _validate_scales([args.check_w], args.check_steps)
    table = parse_arms(getattr(args, "arms", None) or DEFAULT_ARMS, args.lam, args.k)
    active = [(name, cfg) for name, cfg in table.items() if not cfg.is_cfg]
    if not active:
        raise ValueError("verify needs an arm with k > 0 to exercise the controller, "
                         "but every requested arm is plain CFG")
    arm_name, cfg = active[0]
    print(f"checking arm: {describe_arm(arm_name, cfg)}")
    preflight(args)
    pipe = load_pipe(args.model, args.dtype, args.device)
    model = denoiser(pipe)
    w, steps = args.check_w, args.check_steps
    failures: List[str] = []

    # -- 1. transparency ----------------------------------------------------
    print("\n[1/3] transparency: k = 0 must reproduce the pipeline exactly")
    ref = np.asarray(generate(pipe, DEFAULT_PROMPTS[0], w, steps, 0, args.device))
    hook = attach_smc_cfg(pipe, SlidingModeGuidance(presets.cfg_baseline()))
    try:
        got = np.asarray(generate(pipe, DEFAULT_PROMPTS[0], w, steps, 0, args.device))
    finally:
        hook.detach()
    if not np.isfinite(ref).all() or not np.isfinite(got).all():
        failures.append("transparency: pipeline images contain nonfinite values")
        print("      FAIL  nonfinite image values")
    elif np.array_equal(ref, got):
        print("      PASS  bit-identical")
    else:
        d = np.abs(ref.astype(np.float64) - got.astype(np.float64))
        failures.append(f"transparency: images differ (max {d.max()}, mean {d.mean():.4f})")
        print(f"      FAIL  max |diff| = {d.max()}, mean = {d.mean():.4f}")
        print("            the pipeline may be non-deterministic, or the hook is")
        print("            attached somewhere the pipeline does not call.")

    # -- 2. batch order -----------------------------------------------------
    print("\n[2/3] batch order: which half of the doubled batch is conditional?")
    caps = []
    batch_valid = False
    for prompt in (DEFAULT_PROMPTS[0], DEFAULT_PROMPTS[3]):
        with _Probe(model) as probe:
            generate(pipe, prompt, w, 1, 0, args.device)
        if not probe.records:
            failures.append("batch order: the denoiser never saw a batch of two -- "
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
        if (not math.isfinite(d_first) or not math.isfinite(d_second)
                or max(d_first, d_second) == 0
                or min(d_first, d_second) > 0.5 * max(d_first, d_second)):
            failures.append("batch order: prompt sensitivity is nonfinite, absent, or "
                            "similar in both halves; the test is inconclusive")
            print("      FAIL  cannot establish the conditional branch")
        elif cond_is_second:
            batch_valid = True
            print("      PASS  matches the default uncond_first=True")
        else:
            failures.append("batch order: conditional is the FIRST half; "
                            "pass uncond_first=False to GuidanceHook")
            print("      FAIL  pass uncond_first=False, or guidance is inverted")

    # -- 3. what the hook actually sees -------------------------------------
    print("\n[3/3] what the hook sees over a full run")
    ctrl = SlidingModeGuidance(cfg)
    if batch_valid:
        with attach_smc_cfg(pipe, ctrl):
            generate(pipe, DEFAULT_PROMPTS[0], w, steps, 0, args.device)
    if not batch_valid:
        print("      SKIP  doubled-batch order could not be verified")
    elif not ctrl.history:
        failures.append("the controller was never called during a full run")
        print("      FAIL  controller.history is empty")
    else:
        h = ctrl.history
        print(f"      steps seen        {len(h)}  (requested {steps})")
        print(f"      latent shape      {caps[0][2] if caps else 'n/a'}  dtype {caps[0][3] if caps else 'n/a'}")
        print(f"      rms(e)            {h[0].e_rms:.4f} -> {h[-1].e_rms:.4f}")
        print(f"      rms(s)            {h[0].s_rms:.4f} -> {h[-1].s_rms:.4f}")
        tail = h[-5:]
        print(f"      chatter (last {len(tail)})  {sum(x.chatter for x in tail) / len(tail):.3f}")
        print(f"      deriv decides     {sum(x.deriv_matters for x in h) / len(h) * 100:.1f} %")
        if len(h) != steps:
            print(f"      WARN  {len(h)} controller calls for {steps} requested steps")
        if any(not math.isfinite(value) for st in h for value in vars(st).values()):
            failures.append("controller diagnostics contain nonfinite values")
            print("      FAIL  nonfinite controller diagnostics")
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
    print("VERIFY PASSED -- checked batch order, transparency and controller activity for these settings.")
    return 0


# -------------------------------------------------------------------- grid
def cmd_grid(args) -> int:
    _validate_scales(args.w, args.steps)
    prompts = ([p.strip() for p in Path(args.prompts).read_text(encoding="utf-8").splitlines() if p.strip()]
               if args.prompts else DEFAULT_PROMPTS)
    if not prompts:
        raise ValueError("the prompt file must contain at least one nonempty prompt")
    table = parse_arms(getattr(args, "arms", None) or DEFAULT_ARMS, args.lam, args.k)
    resume = getattr(args, "resume", False)
    out = Path(args.out)
    total = len(table) * len(args.w) * len(prompts) * len(args.seeds)

    print(f"{total} images: {len(table)} arms x {len(args.w)} scales x "
          f"{len(prompts)} prompts x {len(args.seeds)} seeds")
    for name, cfg in table.items():
        print("  " + describe_arm(name, cfg))
    print(f"  scales  {args.w}\n  seeds   {args.seeds}\n  steps   {args.steps}"
          f"\n  model   {args.model} ({args.dtype})\n  out     {out}")
    if getattr(args, "dry_run", False):
        print("\n--dry-run: nothing generated, no model loaded.")
        return 0

    preflight(args)
    pipe = load_pipe(args.model, args.dtype, args.device)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "model": args.model, "dtype": args.dtype, "steps": args.steps, "k": args.k,
        "lam": args.lam, "w": args.w, "seeds": args.seeds, "prompts": prompts,
        # the resolved controller settings, not just the names, so a result
        # directory says exactly which law produced it
        "arms": {name: asdict(cfg) for name, cfg in table.items()},
    }, indent=1), encoding="utf-8")

    sig_path = out / "signals.csv"
    append = resume and sig_path.exists()
    with open(sig_path, "a" if append else "w", newline="") as fh:
        wri = csv.writer(fh)
        if not append:
            wri.writerow(["arm", "w", "prompt_id", "seed", "step", "e_rms", "s_rms",
                          "delta_rms", "chatter", "switch_activity", "deriv_matters", "k_eff"])
        done, skipped, t0 = 0, 0, time.time()
        for arm, cfg in table.items():
            ctrl = SlidingModeGuidance(cfg)
            hook = attach_smc_cfg(pipe, ctrl)
            try:
                for w in args.w:
                    d = out / arm / f"w{w}"
                    d.mkdir(parents=True, exist_ok=True)
                    for pid, prompt in enumerate(prompts):
                        for seed in args.seeds:
                            path = d / f"p{pid:02d}_s{seed}.png"
                            if resume and path.exists():
                                done += 1
                                skipped += 1
                                continue
                            hook.reset()
                            img = generate(pipe, prompt, w, args.steps, seed, args.device)
                            if not ctrl.is_cfg and not ctrl.history:
                                raise RuntimeError("the active controller was never called; "
                                                   "this pipeline does not support this CFG hook")
                            img.save(path)
                            for st in ctrl.history:
                                wri.writerow([arm, w, pid, seed, st.step,
                                              f"{st.e_rms:.6f}", f"{st.s_rms:.6f}",
                                              f"{st.delta_rms:.6f}", f"{st.chatter:.4f}",
                                              f"{st.switch_activity:.6f}",
                                              f"{st.deriv_matters:.4f}", f"{st.k_eff:.6f}"])
                            done += 1
                            if done % 10 == 0 or done == total:
                                el = time.time() - t0
                                rate = el / max(done - skipped, 1)
                                print(f"  [{done}/{total}] {el:.0f}s elapsed, "
                                      f"~{rate * (total - done):.0f}s left", flush=True)
                            fh.flush()
            finally:
                hook.detach()
    if skipped:
        print(f"  resumed: {skipped} image(s) already present were skipped")

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
    ap.add_argument("--k", type=float, default=0.1,
                    help="default gain for arms that do not set their own; SD3.5: 0.1, Flux: 0.7")
    ap.add_argument("--lam", type=float, default=6.0,
                    help="default sliding-surface slope for arms that do not set their own")
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, metavar="SPEC",
                    help="guidance laws to run, as [name=]preset[:field=value,...]. "
                         f"Presets: {', '.join(sorted(PRESETS))} (aliases: "
                         f"{', '.join(sorted(_ALIASES))}). Fields: "
                         f"{', '.join(_FLOAT_FIELDS + ('switching',) + _BOOL_FIELDS)}. "
                         "Examples: 'paper' for the published law alone; 'paper:k=0.7' "
                         "for Flux's gain; 'flux=paper:k=0.7' to name its output "
                         f"directory. Default: {' '.join(DEFAULT_ARMS)}")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved plan and exit without loading the model")
    ap.add_argument("--resume", action="store_true",
                    help="skip images that already exist and append to signals.csv")
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--w", type=float, nargs="+", default=[1.5, 2.0, 3.0, 4.5, 7.0],
                    help="guidance scales, all > 1 for doubled-batch CFG")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--prompts", default=None, help="file with one prompt per line")
    ap.add_argument("--check-w", type=float, default=7.0, help="guidance scale used by verify")
    ap.add_argument("--check-steps", type=int, default=8, help="steps used by verify")
    args = ap.parse_args()
    try:
        return cmd_verify(args) if args.mode == "verify" else cmd_grid(args)
    except ValueError as exc:
        ap.error(str(exc))                 # a bad argument: show usage
    except RuntimeError as exc:
        # An environment or integration problem. The message already says what
        # to do, so print it plainly instead of a traceback.
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
