"""Summarise the controller diagnostics a real-model grid wrote.

`signals.csv` has one row per (arm, scale, prompt, seed, denoising step), so a
full sweep is hundreds of thousands of rows. This reduces it to the handful of
numbers section 3 of docs/CFG-Ctrl_Review_and_Improvements.md actually argues
about, with the spread across prompts, and draws the per-step trajectories.

    python experiments/signals.py --run results/coco_test

What it reports, per (arm, scale):

  rms(s) final    the sliding variable at the last step. For the paper's law,
                  which stores the CORRECTED error, this is predicted to sit at
                  (lam-1)*k regardless of how small the measured error gets:
                  once |e| < k the surface is dominated by the previous
                  correction. An arm storing the MEASURED error has no such
                  fixed point and should instead track the error down.
  chatter         fraction of latent elements whose sign(s) flipped since the
                  previous step, averaged over the last few steps. Near 1.0
                  means essentially every element reverses every step.
  switch activity rms(delta_t - delta_{t-1}) / 2k. 1.0 is full bang-bang
                  reversal; a boundary layer should sit far below it.
  derivative      fraction of elements where sign(s) != sign(e_prev), i.e. how
                  often the derivative term in the sliding surface decides
                  anything. Small means the surface is effectively lam*e.

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


def predicted_plateau(arm_cfg: Dict) -> Optional[float]:
    """(lam-1)*k, but only where the recurrence actually has that fixed point.

    It comes from feeding the previous correction back into the surface, so it
    exists only when the corrected error is stored. With measured-error memory
    the surface tracks e instead and has no plateau to predict.
    """
    if not arm_cfg.get("store_corrected", True):
        return None
    lam, k = float(arm_cfg.get("lam", 0.0)), float(arm_cfg.get("k", 0.0))
    return (lam - 1.0) * k if k > 0 and lam > 1 else None


def summarise(run: Path, tail: int = 5):
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    arms: Dict[str, Dict] = cfg.get("arms", {})
    steps = int(cfg.get("steps", 0))
    path = run / "signals.csv"
    if not path.is_file():
        raise ValueError(f"{path} not found")

    final_s, late_chat, deriv, switch, e_first, e_final = (defaultdict(Running) for _ in range(6))
    per_step: Dict[Tuple[str, float, int], Dict[str, Running]] = defaultdict(
        lambda: {"s_rms": Running(), "chatter": Running()})
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
            deriv[key].add(float(row["deriv_matters"]))
            if step == 0:
                e_first[key].add(float(row["e_rms"]))
            last = steps - 1 if steps else None
            if last is not None and step == last:
                final_s[key].add(s)
                e_final[key].add(float(row["e_rms"]))
            if last is not None and step > last - tail:
                late_chat[key].add(float(row["chatter"]))
                # Same window as chatter on purpose: they describe one
                # phenomenon, and for a sign law switch should equal
                # sqrt(chatter), which is only checkable if the windows match.
                switch[key].add(float(row["switch_activity"]))

    if not rows:
        raise ValueError(f"{path} has no usable rows")
    if steps and max_step != steps - 1:
        print(f"WARNING: config says {steps} steps but the highest step seen is "
              f"{max_step}; the 'final' column may not be the last step.\n")
    return dict(arms=arms, steps=steps, rows=rows, final_s=final_s, late_chat=late_chat,
                deriv=deriv, switch=switch, e_first=e_first, e_final=e_final,
                per_step=per_step)


def report(res: Dict, tail: int) -> None:
    arms, keys = res["arms"], sorted(res["final_s"])
    print(f"{res['rows']} rows, {res['steps']} steps\n")
    print(f"{'arm':<10}{'w':>6}{'rms(e)':>17}{'rms(s) final':>16}{'predicted':>11}"
          f"{'ratio':>8}{'chatter':>9}{'switch':>9}{'deriv':>8}")
    for key in keys:
        arm, w = key
        cfgs = arms.get(arm, {})
        k = float(cfgs.get("k", 0.0)) or float("nan")
        pred = predicted_plateau(cfgs)
        f = res["final_s"][key]
        e0, e1 = res["e_first"][key].mean, res["e_final"][key].mean
        norm = res["switch"][key].mean / (2 * k) if k == k and k else float("nan")
        pred_txt = f"{pred:.3f}" if pred else "tracks e"
        ratio = f"{f.mean / pred:.2f}" if pred else "--"
        print(f"{arm:<10}{w:>6.2f}{e0:>8.4f}->{e1:<8.4f}"
              f"{f.mean:>9.4f} ±{f.sd:.3f}{pred_txt:>11}{ratio:>8}"
              f"{res['late_chat'][key].mean:>9.3f}{norm:>9.3f}"
              f"{res['deriv'][key].mean * 100:>7.1f}%")

    print(f"\n  rms(e) is the measured error, first step -> last.")
    print(f"  'predicted' is (lam-1)*k, the fixed point that corrected-error memory")
    print(f"  creates once |e| < k. 'ratio' near 1.00 confirms it. Arms storing the")
    print(f"  measured error have no such fixed point and should track e down instead.")
    print(f"  chatter and switch both average the last {tail} steps; switch is")
    print(f"  normalised by 2k, so 1.0 is full reversal every step. For a sign law")
    print(f"  switch should equal sqrt(chatter); a mismatch means the correction is")
    print(f"  not bang-bang. 'deriv' is averaged over the whole run.")
    print(f"  These are controller diagnostics only: they say nothing about image")
    print(f"  quality. Use evaluate.py and pareto.py for that.")


def write_csv(run: Path, res: Dict) -> Path:
    out = run / "signals_summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wri = csv.writer(fh)
        wri.writerow(["arm", "w", "n_final", "e_rms_first", "e_rms_final", "s_rms_final",
                      "s_rms_final_sd", "predicted_plateau", "ratio", "chatter_late",
                      "switch_activity_norm", "deriv_matters"])
        for key in sorted(res["final_s"]):
            arm, w = key
            cfgs = res["arms"].get(arm, {})
            k = float(cfgs.get("k", 0.0))
            pred = predicted_plateau(cfgs)
            f = res["final_s"][key]
            wri.writerow([arm, w, f.n, f"{res['e_first'][key].mean:.6f}",
                          f"{res['e_final'][key].mean:.6f}", f"{f.mean:.6f}",
                          f"{f.sd:.6f}", "" if pred is None else f"{pred:.6f}",
                          "" if pred is None else f"{f.mean / pred:.6f}",
                          f"{res['late_chat'][key].mean:.6f}",
                          f"{res['switch'][key].mean / (2 * k):.6f}" if k else "",
                          f"{res['deriv'][key].mean:.6f}"])
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
        pred = predicted_plateau(res["arms"].get(arm, {}))
        for i, w in enumerate(scales):
            steps = sorted(s for a, ww, s in res["per_step"] if a == arm and ww == w)
            if not steps:
                continue
            for row, metric in enumerate(("s_rms", "chatter")):
                axes[row][col].plot(
                    steps, [res["per_step"][(arm, w, s)][metric].mean for s in steps],
                    color=colors[i % len(colors)], linewidth=1.6, label=f"w={w:g}")
        if pred:
            axes[0][col].axhline(pred, color="#52514e", linestyle="--", linewidth=1.2,
                                 label=f"(lam-1)k = {pred:.2f}")
        axes[0][col].set_title(f"{arm}: sliding variable", loc="left", fontsize=10)
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
