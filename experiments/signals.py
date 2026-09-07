"""Summarise the controller diagnostics a real-model grid wrote.

`signals.csv` has one row per (arm, scale, prompt, seed, denoising step), so a
full sweep is hundreds of thousands of rows. This reduces it to the handful of
numbers section 3 of docs/CFG-Ctrl_Review_and_Improvements.md actually argues
about, with the spread across prompts, and draws the per-step trajectories.

    python experiments/signals.py --run results/coco_test

What it reports, per (arm, scale):

  rms(s) final    the sliding variable at the last step. For the paper's law,
                  which stores the CORRECTED error, this is predicted to sit at
                  approximately (lam-1)*k in the small-error alternating regime.
                  An arm storing the MEASURED error instead has
                  s_n = e_n + (lam-1)*e_(n-1); both measurements matter.
                  Proximal mode instead logs s=e, without a sliding surface.
                  This is the last denoiser call, before its scheduler update,
                  not a measurement at the final decoded image.
  chatter         fraction of latent elements whose sign(s) flipped since the
                  previous step, averaged over the last few steps. Near 1.0
                  means essentially every element reverses every step.
  switch activity rms(delta_t - delta_{t-1}) / (2*k*factor), where factor is
                  (w-1)/w for excess-only correction, otherwise 1. This uses
                  the arm's applied gain; relative-gain arms have no fixed norm.
  derivative      fraction of strict sign disagreements between s and the
                  stored memory. With corrected memory, this does not compare
                  s with the current measured error.

Only arms with k > 0 appear: a k = 0 arm never calls the controller, by design,
so it writes no rows. That is why `cfg` is absent rather than empty.

Everything here describes the controller's internal behaviour. It says nothing
about image quality; for that see evaluate.py and pareto.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

Key = Tuple[str, float]


class Running:
    """Mean and sample standard deviation without holding the rows."""

    __slots__ = ("n", "total", "sq")

    def __init__(self) -> None:
        self.n = 0
        self.total = 0.0
        self.sq = 0.0

    def add(self, x: float) -> None:
        if math.isfinite(x):
            self.n += 1
            self.total += x
            self.sq += x * x

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else float("nan")

    @property
    def sd(self) -> float:
        if self.n < 2:
            return float("nan")
        var = (self.sq - self.n * self.mean ** 2) / (self.n - 1)
        return math.sqrt(max(var, 0.0))


def applied_gain(arm_cfg: Dict, w: Optional[float] = None) -> Optional[float]:
    """Fixed error-correction amplitude, accounting for excess-only scaling."""
    if arm_cfg.get("relative_gain", False):
        return None
    k = float(arm_cfg.get("k", 0.0))
    if not math.isfinite(k) or k <= 0:
        return None
    if arm_cfg.get("excess_only", False):
        if w is None or not math.isfinite(w) or w <= 1:
            return None
        k *= (w - 1) / w
    return k


def predicted_plateau(arm_cfg: Dict, w: Optional[float] = None) -> Optional[float]:
    """Small-error alternating surface amplitude, not a fixed point or proof.

    Feeding the previous correction back into the surface can sustain a cycle.
    This approximation requires corrected memory, sign switching, a fixed
    positive gain and lam > 1. Smooth/relative laws need separate analysis.
    """
    if (arm_cfg.get("mode", "sliding") != "sliding"
            or not arm_cfg.get("store_corrected", True)
            or arm_cfg.get("switching", "sign") != "sign"):
        return None
    lam, k = float(arm_cfg.get("lam", 0.0)), applied_gain(arm_cfg, w)
    return (lam - 1.0) * k if k is not None and math.isfinite(lam) and lam > 1 else None


def summarise(run: Path, tail: int = 5):
    if tail <= 0:
        raise ValueError("tail must be positive")
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    arms: Dict[str, Dict] = cfg.get("arms", {})
    steps = int(cfg.get("steps", 0))
    if steps <= 0:
        raise ValueError("config.json must specify a positive number of steps")
    path = run / "signals.csv"
    if not path.is_file():
        raise ValueError(f"{path} not found")

    final_s, late_chat, deriv, switch, e_first, e_final = (defaultdict(Running) for _ in range(6))
    per_step: Dict[Tuple[str, float, int], Dict[str, Running]] = defaultdict(
        lambda: {"s_rms": Running(), "chatter": Running(), "e_rms": Running()})
    # Keep only the last two measurements per trajectory. Pair by identity,
    # not CSV adjacency, so interleaved/resumed runs do not mix prompts/seeds.
    endpoints = defaultdict(dict)
    rows = 0
    max_step = 0

    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                arm, w, step = row["arm"], float(row["w"]), int(row["step"])
            except (KeyError, TypeError, ValueError):
                continue
            rows += 1
            max_step = max(max_step, step)
            key: Key = (arm, w)
            s = float(row["s_rms"])
            bucket = per_step[(arm, w, step)]
            bucket["s_rms"].add(s)
            bucket["chatter"].add(float(row["chatter"]))
            bucket["e_rms"].add(float(row["e_rms"]))
            deriv[key].add(float(row["deriv_matters"]))
            if step == 0:
                e_first[key].add(float(row["e_rms"]))
            last = steps - 1 if steps else None
            if step in (last - 1, last):
                trajectory = (arm, w, row["prompt_id"], row["seed"])
                if step in endpoints[trajectory]:
                    raise ValueError(f"duplicate endpoint row for {trajectory}, step {step}")
                endpoints[trajectory][step] = (float(row["e_rms"]), s)
            if last is not None and step == last:
                final_s[key].add(s)
                e_final[key].add(float(row["e_rms"]))
            if last is not None and step > last - tail:
                late_chat[key].add(float(row["chatter"]))
                # Same window as chatter on purpose: they describe one
                # phenomenon. For a fixed-amplitude sign law without zeros,
                # the identity is per sample/step, before averaging sqrt(chatter).
                switch[key].add(float(row["switch_activity"]))

    if not rows:
        raise ValueError(f"{path} has no usable rows")
    if steps and max_step != steps - 1:
        print(f"WARNING: config says {steps} steps but the highest step seen is "
              f"{max_step}; the 'final' column may not be the last step.\n")
    measured = defaultdict(lambda: {name: Running() for name in
                                   ("e_prev", "s", "lower", "upper")})
    for (arm, w, _, _), values in endpoints.items():
        cfgs = arms.get(arm, {})
        if (cfgs.get("mode", "sliding") != "sliding" or cfgs.get("store_corrected", True)
                or not all(s in values for s in (steps - 2, steps - 1))):
            continue
        prev = values[steps - 2][0]
        current, surface = values[steps - 1]
        coefficient = abs(float(cfgs.get("lam", 0.0)) - 1)
        bucket = measured[(arm, w)]
        for name, value in (("e_prev", prev), ("s", surface),
                            ("lower", abs(coefficient * prev - current)),
                            ("upper", coefficient * prev + current)):
            bucket[name].add(value)
    return dict(arms=arms, steps=steps, rows=rows, final_s=final_s, late_chat=late_chat,
                deriv=deriv, switch=switch, e_first=e_first, e_final=e_final,
                per_step=per_step, measured_memory=measured)


def report(res: Dict, tail: int) -> None:
    arms, keys = res["arms"], sorted(res["final_s"])
    arm_width = max([10] + [len(arm) + 1 for arm, _ in keys])
    print(f"{res['rows']} rows, {res['steps']} steps\n")
    print(f"{'arm':<{arm_width}}{'w':>6}{'rms(e)':>17}{'rms(s) last':>16}{'predicted':>11}"
          f"{'ratio':>8}{'chatter':>9}{'switch':>9}{'deriv':>8}")
    for key in keys:
        arm, w = key
        cfgs = arms.get(arm, {})
        gain = applied_gain(cfgs, w)
        pred = predicted_plateau(cfgs, w)
        f = res["final_s"][key]
        e0, e1 = res["e_first"][key].mean, res["e_final"][key].mean
        norm = f"{res['switch'][key].mean / (2 * gain):.3f}" if gain else "--"
        pred_txt = ("s=e" if cfgs.get("mode") == "proximal" else f"{pred:.3f}" if pred else
                    "tracks e" if not cfgs.get("store_corrected", True) else "n/a")
        deriv_txt = ("--" if cfgs.get("mode") == "proximal" else
                     f"{res['deriv'][key].mean * 100:.1f}%")
        ratio = f"{f.mean / pred:.2f}" if pred else "--"
        print(f"{arm:<{arm_width}}{w:>6.2f}{e0:>8.4f}->{e1:<8.4f}"
              f"{f.mean:>9.4f} ±{f.sd:.3f}{pred_txt:>11}{ratio:>8}"
              f"{res['late_chat'][key].mean:>9.3f}{norm:>9}"
              f"{deriv_txt:>8}")

    print("\n  'last' is the last denoiser evaluation, before its scheduler update.")
    print("  'predicted' is a small-error alternating amplitude for corrected-memory")
    print("  sign switching with a fixed gain. A ratio near 1 is consistent with it.")
    print("  'tracks e' means s_n = e_n + (lam-1)*e_(n-1), not s_n = lam*e_n")
    print("  or a guarantee of zero at the last step.")
    print(f"  chatter and switch both average the last {tail} steps; switch is")
    print("  normalised by twice the applied fixed gain (including (w-1)/w for excess).")
    print("  For fixed sign switching without zeros, per-sample switch = sqrt(chatter).")
    print("  Averages need not obey that equality. Relative-gain normalisation is omitted.")
    print("  'deriv' counts strict sign disagreement with stored memory over the whole")
    print("  run; low values do not imply agreement with the current measured error.")
    print("  Proximal arms use s=e for logging, with no sliding surface or derivative;")
    print("  their surface norms are not comparable to the sliding arms' norms.")
    print(f"  These are controller diagnostics only: they say nothing about image")
    print(f"  quality. Use evaluate.py and pareto.py for that.")
    if res["measured_memory"]:
        print("\n  Measured-memory check, paired last two calls (mean RMS bounds):")
        print(f"  {'arm':<10}{'w':>6}{'pairs':>8}{'rms(e_prev)':>14}{'s lower':>12}{'observed s':>12}{'s upper':>12}")
        for (arm, w), b in sorted(res["measured_memory"].items()):
            print(f"  {arm:<10}{w:>6.2f}{b['s'].n:>8}{b['e_prev'].mean:>14.4f}"
                  f"{b['lower'].mean:>12.4f}{b['s'].mean:>12.4f}{b['upper'].mean:>12.4f}")
        print("  Bounds: abs(|lam-1|*rms(e_prev)-rms(e)) <= rms(s)")
        print("          <= |lam-1|*rms(e_prev)+rms(e), averaged over paired trajectories.")
        print("  Bounds constrain the norm; scalar RMS logs cannot reconstruct direction.")


def write_csv(run: Path, res: Dict) -> Path:
    out = run / "signals_summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wri = csv.writer(fh)
        wri.writerow(["arm", "w", "n_final", "e_rms_first", "e_rms_final", "s_rms_final",
                      "s_rms_final_sd", "predicted_plateau", "ratio", "chatter_late",
                      "switch_activity_norm", "deriv_matters", "switch_activity_applied_norm",
                      "n_paired", "e_rms_prev_final", "s_rms_paired_final",
                      "s_rms_lower_bound", "s_rms_upper_bound", "surface_reference"])
        for key in sorted(res["final_s"]):
            arm, w = key
            cfgs = res["arms"].get(arm, {})
            k = float(cfgs.get("k", 0.0))
            pred = predicted_plateau(cfgs, w)
            gain = applied_gain(cfgs, w)
            b = res["measured_memory"].get(key)
            f = res["final_s"][key]
            wri.writerow([arm, w, f.n, f"{res['e_first'][key].mean:.6f}",
                          f"{res['e_final'][key].mean:.6f}", f"{f.mean:.6f}",
                          f"{f.sd:.6f}", "" if pred is None else f"{pred:.6f}",
                          "" if pred is None else f"{f.mean / pred:.6f}",
                          f"{res['late_chat'][key].mean:.6f}",
                          f"{res['switch'][key].mean / (2 * k):.6f}" if k else "",
                          f"{res['deriv'][key].mean:.6f}" if cfgs.get("mode") != "proximal" else "",
                          f"{res['switch'][key].mean / (2 * gain):.6f}" if gain else "",
                          b["s"].n if b else 0,
                          *([f"{b[name].mean:.6f}" for name in ("e_prev", "s", "lower", "upper")]
                            if b else [""] * 4),
                          "current_error" if cfgs.get("mode") == "proximal" else "sliding_surface"])
    return out


def plot(run: Path, res: Dict) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the figure")
        return None

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"]
    arms = sorted({a for a, _, _ in res["per_step"]})
    scales = sorted({w for _, w, _ in res["per_step"]})
    if not arms:
        return None
    fig, axes = plt.subplots(2, len(arms), figsize=(5.5 * len(arms), 7), squeeze=False)
    for col, arm in enumerate(arms):
        for i, w in enumerate(scales):
            pred = predicted_plateau(res["arms"].get(arm, {}), w)
            steps = sorted(s for a, ww, s in res["per_step"] if a == arm and ww == w)
            if not steps:
                continue
            for row, metric in enumerate(("s_rms", "chatter")):
                axes[row][col].plot(
                    steps, [res["per_step"][(arm, w, s)][metric].mean for s in steps],
                    color=colors[i % len(colors)], linewidth=1.6, label=f"w={w:g}")
            if pred:
                axes[0][col].axhline(pred, color=colors[i % len(colors)], linestyle="--",
                                    linewidth=0.9, label=f"limit w={w:g}: {pred:.2f}")
            elif (res["arms"].get(arm, {}).get("mode", "sliding") == "sliding"
                  and not res["arms"].get(arm, {}).get("store_corrected", True)):
                lam = float(res["arms"][arm].get("lam", 0.0))
                axes[0][col].plot(steps, [lam * res["per_step"][(arm, w, s)]["e_rms"].mean
                                        for s in steps], color=colors[i % len(colors)],
                                 linestyle=":", linewidth=1.0, label=f"lam*rms(e), w={w:g} (approx.)")
        reference = "current error (s=e)" if res["arms"].get(arm, {}).get("mode") == "proximal" else "sliding variable"
        axes[0][col].set_title(f"{arm}: {reference}", loc="left", fontsize=10)
        axes[1][col].set_title(f"{arm}: chatter", loc="left", fontsize=10)
        for row in (0, 1):
            axes[row][col].set_xlabel("denoising step")
            axes[row][col].grid(True, color="#e6e5e1", linewidth=0.8)
            axes[row][col].set_axisbelow(True)
        axes[0][col].set_ylabel("rms(s)")
        axes[1][col].set_ylabel("fraction of elements flipping")
        axes[1][col].set_ylim(-0.05, 1.05)
        axes[0][col].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out = run / "signals_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="a grid output directory")
    ap.add_argument("--tail", type=int, default=5,
                    help="steps at the end to average chatter over")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    try:
        res = summarise(args.run, args.tail)
    except ValueError as exc:
        ap.error(str(exc))
    report(res, args.tail)
    print(f"\nwrote {write_csv(args.run, res)}")
    if not args.no_plot:
        made = plot(args.run, res)
        if made:
            print(f"wrote {made}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
