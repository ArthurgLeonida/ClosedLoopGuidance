"""CFG vs SMC-CFG vs the surviving refinements, on the analytic plant.

Runs on a laptop CPU in a few minutes. Everything the CFG-Ctrl paper argues
about is *measured* here on a plant with ground truth:

  E1  pareto     fidelity (Frechet) and alignment (mean p(class|x)) vs the
                 guidance scale w, per method, with seed error bars. The
                 Pareto view is the honest way to compare guidance laws: a law
                 that merely lowers the effective w moves *along* the CFG
                 curve, not off it.
  E2  signals    per-step |e|, |s|, chatter, switching activity and the
                 "derivative matters" index. Does s reach zero? Does the
                 derivative term in the sliding surface ever decide the sign?
                 Includes the paper's law with measured-error memory, to
                 isolate the effect of storing the corrected error.
  E3  k_sweep    the chattering regime: sign vs boundary layer as k grows.
  E4  loop_gain  finite-difference Jacobian of e along real trajectories: is
                 Gamma ~ w*I as Theorem 1 assumes? what is the actual per-step
                 authority dt*w*|J| compared with k?
  E5  transfer   the same absolute k on a plant with twice the velocity scale,
                 vs a relative (rms-normalised) k.

Outputs: results/toy/*.csv, *.png and summary.json (the numbers quoted in
docs/CFG-Ctrl_Review_and_Improvements.md).

    python experiments/toy_smc_cfg.py            # ~10 min
    python experiments/toy_smc_cfg.py --quick    # ~2 min
    python experiments/toy_smc_cfg.py --only e2  # one experiment
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cfgctrl import GaussianMixtureFlow, SMCConfig, SlidingModeGuidance, ring_mixture  # noqa: E402

LAM = 6.0

# Fixed colour slot per method, kept across every figure. Markers are the
# secondary encoding so the figures survive greyscale printing.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "X", "v"]
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def method_table(k: float, lam: float = LAM) -> List[Tuple[str, SMCConfig]]:
    """Fixed order == fixed colour slot. Index 0 is always the CFG baseline."""
    sat = dict(switching="sat", phi=k * lam, store_corrected=False)
    return [
        ("CFG (P-control)", SMCConfig(k=0.0)),
        ("SMC-CFG, paper (sign)", SMCConfig(lam=lam, k=k)),
        ("SMC + boundary layer (sat)", SMCConfig(lam=lam, k=k, **sat)),
        ("paper (sign), extrapolation only", SMCConfig(lam=lam, k=k, excess_only=True)),
        ("boundary layer, extrapolation only", SMCConfig(lam=lam, k=k, excess_only=True, **sat)),
    ]


# ------------------------------------------------------------------ helpers
def run_one(plant: GaussianMixtureFlow, w: float, cfg: Optional[SMCConfig], steps: int,
            n: int, seed: int, cond: int = 0):
    ctrl = SlidingModeGuidance(cfg) if cfg is not None else None
    x, trace = plant.sample(n, cond, w, ctrl, steps=steps, seed=seed)
    row = {
        "frechet": plant.frechet_distance(x, cond),
        "confidence": plant.class_confidence(x, cond),
        "accuracy": plant.class_accuracy(x, cond),
    }
    if trace.steps:
        for key in ("s_rms", "chatter", "switch_activity", "deriv_matters", "delta_rms"):
            row[f"{key}_mean"] = sum(trace.series(key)) / len(trace.steps)
        row["s_rms_final"] = trace.steps[-1].s_rms
    return row, x, trace


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        wri = csv.DictWriter(fh, fieldnames=keys)
        wri.writeheader()
        wri.writerows(rows)


def mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    return m, math.sqrt(sum((x - m) ** 2 for x in vals) / max(len(vals) - 1, 1))


def agg(rows: List[Dict], key: str, **where) -> Tuple[float, float]:
    return mean_std([r[key] for r in rows if all(r.get(f) == v for f, v in where.items())])


def pareto_ratio(rows: List[Dict], method: str, ws: Sequence[float]) -> Dict[str, float]:
    """Frechet relative to the CFG curve at MATCHED alignment. Below 1 = better.

    This is the only fair comparison: a law that just lowers the effective w
    slides along the CFG curve and scores 1.0. Note the ratio is meaningless
    where the curve is vertical (alignment saturated near 1), so read it only
    in the region where mean p(class|x) is still moving.
    """
    pts = sorted((agg(rows, "confidence", method="CFG (P-control)", w=w)[0],
                  agg(rows, "frechet", method="CFG (P-control)", w=w)[0]) for w in ws)
    out = {}
    for w in ws:
        c, f = agg(rows, "confidence", method=method, w=w)[0], agg(rows, "frechet", method=method, w=w)[0]
        base = None
        for (c0, f0), (c1, f1) in zip(pts, pts[1:]):
            if c0 <= c <= c1:
                base = f0 + (c - c0) / max(c1 - c0, 1e-12) * (f1 - f0)
                break
        out[str(w)] = f / base if base and base > 0 else float("nan")
    return out


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=10, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=TEXT2, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=8)


def line(ax, xs, ys, idx: int, label: str, yerr=None) -> None:
    ax.errorbar(xs, ys, yerr=yerr, color=COLORS[idx], marker=MARKERS[idx], markersize=5,
                linewidth=1.8, capsize=2, label=label,
                markeredgecolor=SURFACE, markeredgewidth=0.8)


def savefig(fig, path: Path) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------- experiments
def exp_pareto(out: Path, plant: GaussianMixtureFlow, tag: str, ws: Sequence[float],
               k: float, steps: int, n: int, seeds: Sequence[int], summary: Dict) -> None:
    methods = method_table(k)
    rows = []
    for name, cfg in methods:
        for w in ws:
            for seed in seeds:
                r, _, _ = run_one(plant, w, cfg, steps, n, seed)
                rows.append(dict(method=name, w=w, seed=seed, **r))
    write_csv(out / f"e1_pareto_{tag}.csv", rows)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for i, (name, _) in enumerate(methods):
        fr = [agg(rows, "frechet", method=name, w=w) for w in ws]
        cf = [agg(rows, "confidence", method=name, w=w) for w in ws]
        line(axes[0], ws, [m for m, _ in fr], i, name, yerr=[s for _, s in fr])
        line(axes[1], ws, [m for m, _ in cf], i, name, yerr=[s for _, s in cf])
        axes[2].plot([m for m, _ in cf], [m for m, _ in fr], color=COLORS[i], marker=MARKERS[i],
                     markersize=5, linewidth=1.8, label=name,
                     markeredgecolor=SURFACE, markeredgewidth=0.8)
    truth = plant.true_class_samples(n, 0, seed=77)
    t_conf = plant.class_confidence(truth, 0)
    axes[1].axhline(t_conf, color=TEXT2, linewidth=1.0, linestyle=":",
                    label=f"true class law ({t_conf:.2f})")
    style(axes[0], f"Fidelity vs guidance scale ({tag}, k={k})", "guidance scale w",
          "Frechet distance to true class (lower is better)")
    style(axes[1], "Alignment vs guidance scale", "guidance scale w",
          "mean p(class | x)  (higher is better)")
    style(axes[2], "Pareto view: one point per w, lower-right is better",
          "mean p(class | x)", "Frechet distance")
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[1].legend(fontsize=6.5, frameon=False, labelcolor=TEXT2, loc="lower right")
    savefig(fig, out / f"e1_pareto_{tag}.png")

    summary[f"e1_{tag}"] = {
        "true_confidence": t_conf,
        "bayes_accuracy": plant.class_accuracy(truth, 0),
        **{name: {
            "frechet": {str(w): agg(rows, "frechet", method=name, w=w) for w in ws},
            "confidence": {str(w): agg(rows, "confidence", method=name, w=w) for w in ws},
            "pareto_ratio_vs_cfg": pareto_ratio(rows, name, ws),
        } for name, _ in methods},
    }


def exp_signals(out: Path, plant: GaussianMixtureFlow, w: float, k: float, steps: int,
                n: int, summary: Dict) -> None:
    mt = method_table(k)
    methods = [
        (mt[1][0], mt[1][1], 1),
        ("paper law, measured-error memory", SMCConfig(lam=LAM, k=k, store_corrected=False), 5),
        (mt[2][0], mt[2][1], 2),
    ]
    keys = ("e_rms", "s_rms", "chatter", "switch_activity", "deriv_matters")
    fig, axes = plt.subplots(1, len(keys), figsize=(3.8 * len(keys), 3.6))
    rows = []
    for name, cfg, idx in methods:
        r, _, trace = run_one(plant, w, cfg, steps, n, 0)
        for key, ax in zip(keys, axes):
            ax.plot(trace.sigmas, trace.series(key), color=COLORS[idx], marker=MARKERS[idx],
                    markersize=3.5, linewidth=1.6, label=name,
                    markeredgecolor=SURFACE, markeredgewidth=0.6)
        for t, st in enumerate(trace.steps):
            rows.append(dict(method=name, step=t, sigma=trace.sigmas[t],
                             **{kk: getattr(st, kk) for kk in keys}))
        summary.setdefault("e2", {})[name] = {
            "s_rms_first": trace.steps[0].s_rms, "s_rms_final": trace.steps[-1].s_rms,
            "chatter_last10": sum(trace.series("chatter")[-10:]) / 10,
            "switch_activity_last10": sum(trace.series("switch_activity")[-10:]) / 10,
            "deriv_matters_mean": r["deriv_matters_mean"],
            "frechet": r["frechet"], "confidence": r["confidence"],
        }
    write_csv(out / "e2_signals.csv", rows)
    style(axes[0], f"Measured error |e| (w={w}, k={k})", "sigma (1 = noise, 0 = data)", "rms(e)")
    style(axes[1], "Sliding variable |s|: does it reach 0?", "sigma", "rms(s)")
    style(axes[2], "Chatter index: sign(s) flips per element", "sigma", "fraction flipped")
    style(axes[3], "Switching activity rms(delta_t - delta_t-1)", "sigma", "rms (2k = full chatter)")
    style(axes[4], "Derivative-matters index: sign(s) != sign(e_prev)", "sigma", "fraction")
    for ax in axes:
        ax.invert_xaxis()
    axes[1].legend(fontsize=6.5, frameon=False, labelcolor=TEXT2)
    savefig(fig, out / "e2_signals.png")


def exp_k_sweep(out: Path, plant: GaussianMixtureFlow, w: float, ks: Sequence[float],
                steps: int, n: int, seeds: Sequence[int], summary: Dict) -> None:
    rows = []
    for k in ks:
        for idx in (1, 2):
            name, cfg = method_table(k)[idx]
            for seed in seeds:
                r, _, _ = run_one(plant, w, cfg, steps, n, seed)
                rows.append(dict(method=name, k=k, seed=seed, **r))
    base = [run_one(plant, w, SMCConfig(k=0.0), steps, n, s)[0] for s in seeds]
    write_csv(out / "e3_k_sweep.csv", rows)

    names = {idx: method_table(0.1)[idx][0] for idx in (1, 2)}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for idx, name in names.items():
        fr = [agg(rows, "frechet", method=name, k=k) for k in ks]
        cf = [agg(rows, "confidence", method=name, k=k) for k in ks]
        sa = [agg(rows, "switch_activity_mean", method=name, k=k)[0] / (2 * k) for k in ks]
        line(axes[0], ks, [m for m, _ in fr], idx, name, yerr=[s for _, s in fr])
        line(axes[1], ks, [m for m, _ in cf], idx, name, yerr=[s for _, s in cf])
        line(axes[2], ks, sa, idx, name)
    bf, bc = mean_std([b["frechet"] for b in base]), mean_std([b["confidence"] for b in base])
    axes[0].axhline(bf[0], color=COLORS[0], linewidth=1.4, linestyle="--", label="CFG (k=0)")
    axes[1].axhline(bc[0], color=COLORS[0], linewidth=1.4, linestyle="--", label="CFG (k=0)")
    style(axes[0], f"Fidelity vs switching gain k (w={w})", "k", "Frechet distance")
    style(axes[1], "Alignment vs k", "k", "mean p(class | x)")
    style(axes[2], "Normalised switching activity", "k",
          "rms(delta_t - delta_t-1) / 2k   (1 = full chatter)")
    for ax in axes:
        ax.set_xscale("log")
    axes[0].legend(fontsize=6.5, frameon=False, labelcolor=TEXT2)
    savefig(fig, out / "e3_k_sweep.png")
    summary["e3"] = {
        "cfg": {"frechet": bf, "confidence": bc},
        **{name: {str(k): {
            "frechet": agg(rows, "frechet", method=name, k=k),
            "confidence": agg(rows, "confidence", method=name, k=k),
            "switch_activity_norm": agg(rows, "switch_activity_mean", method=name, k=k)[0] / (2 * k),
        } for k in ks} for name in names.values()},
    }


def exp_loop_gain(out: Path, plants: Dict[str, GaussianMixtureFlow], w: float, k: float,
                  steps: int, n: int, summary: Dict) -> None:
    """Along a paper-SMC trajectory, measure J = de/dx.

    The Euler step moves x by -dt*w*(e + delta), so the correction changes the
    NEXT measured error by -dt*w*J*delta. Define M = -J: the paper's reaching
    condition needs s^T M sign(s) > 0, and its Assumption 2 wants M ~ I to
    within a 1/sqrt(D) relative deviation.
    """
    rows = []
    fig, axes = plt.subplots(1, len(plants), figsize=(5.2 * len(plants), 3.8), squeeze=False)
    for p_i, (tag, plant) in enumerate(plants.items()):
        D = plant.D
        ctrl = SlidingModeGuidance(SMCConfig(lam=LAM, k=k))
        g = torch.Generator().manual_seed(0)
        x = torch.randn(n, D, generator=g)
        sig = plant.sigma_schedule(steps)
        e_prev = None
        emin, emax, emean, sigmas = [], [], [], []
        for i in range(steps):
            s_now, s_next = float(sig[i]), float(sig[i + 1])
            dt = s_now - s_next
            v_u = plant.velocity(x, s_now, None)
            e = plant.velocity(x, s_now, 0) - v_u
            J = plant.error_jacobian(x, s_now, 0)
            M = -J
            eig = torch.linalg.eigvalsh(0.5 * (M + M.transpose(1, 2)))          # [n, D]
            e_prev = e if e_prev is None else e_prev
            s = (e - e_prev) + LAM * e_prev
            sg = torch.sign(s)
            reach = torch.einsum("bi,bij,bj->b", s, M, sg)
            dev = torch.linalg.matrix_norm(M - torch.eye(D)[None], ord=2)
            gain = dt * w * torch.linalg.matrix_norm(J, ord=2)
            rows.append(dict(plant=tag, step=i, sigma=s_now,
                             eig_min=float(eig.min()), eig_mean=float(eig.mean()),
                             eig_max=float(eig.max()),
                             frac_reaching_ok_paper_sign=float((reach > 0).float().mean()),
                             frac_assumption2=float((dev < 1 / math.sqrt(D)).float().mean()),
                             loop_gain_median=float(gain.median()),
                             loop_gain_max=float(gain.max()), k=k))
            emin.append(float(eig.min())); emax.append(float(eig.max()))
            emean.append(float(eig.mean())); sigmas.append(s_now)
            e_app = ctrl.correct(e)
            e_prev = e_app
            x = x + (s_next - s_now) * (v_u + w * e_app)

        ax = axes[0][p_i]
        ax.fill_between(sigmas, emin, emax, color=COLORS[4], alpha=0.18, linewidth=0,
                        label="min..max over samples")
        ax.plot(sigmas, emean, color=COLORS[4], linewidth=1.8, label="mean eigenvalue of sym(-de/dx)")
        ax.plot(sigmas, [1.0] * len(sigmas), color=COLORS[0], linewidth=1.4, linestyle="--",
                label="Assumption 2 wants ~ +1 (identity)")
        ax.axhline(0.0, color=TEXT2, linewidth=0.8)
        style(ax, f"One-step loop gain along trajectories ({tag}, D={D})", "sigma",
              "eigenvalues of sym(-de/dx)")
        ax.invert_xaxis()
        ax.set_yscale("symlog", linthresh=1.0)
        ax.legend(fontsize=6.5, frameon=False, labelcolor=TEXT2)

        inner = [r for r in rows if r["plant"] == tag][1:]      # skip sigma=1 where J=0
        summary.setdefault("e4", {})[tag] = {
            "frac_assumption2_mean": sum(r["frac_assumption2"] for r in inner) / len(inner),
            "frac_reaching_ok_paper_sign_mean":
                sum(r["frac_reaching_ok_paper_sign"] for r in inner) / len(inner),
            "loop_gain_median_over_traj":
                sorted(r["loop_gain_median"] for r in inner)[len(inner) // 2],
            "loop_gain_max_over_traj": max(r["loop_gain_max"] for r in inner),
            "eig_min_overall": min(r["eig_min"] for r in inner),
            "eig_max_overall": max(r["eig_max"] for r in inner), "k": k,
        }
    write_csv(out / "e4_loop_gain.csv", rows)
    savefig(fig, out / "e4_loop_gain.png")


def exp_transfer(out: Path, w: float, k: float, steps: int, n: int,
                 seeds: Sequence[int], summary: Dict) -> None:
    plants = {"radius 4": ring_mixture(radius=4.0, std=1.5),
              "radius 8 (2x scale)": ring_mixture(radius=8.0, std=3.0)}
    configs = [
        ("CFG (P-control)", SMCConfig(k=0.0)),
        ("SMC paper, absolute k", SMCConfig(lam=LAM, k=k)),
        ("SMC sat, relative k (fraction of rms e)",
         SMCConfig(lam=LAM, k=k, relative_gain=True, switching="sat",
                   phi=k * LAM, store_corrected=False)),
    ]
    rows = []
    for tag, plant in plants.items():
        for name, cfg in configs:
            for seed in seeds:
                r, _, _ = run_one(plant, w, cfg, steps, n, seed)
                rows.append(dict(plant=tag, method=name, seed=seed, **r))
    write_csv(out / "e5_transfer.csv", rows)
    summary["e5"] = {tag: {name: {
        "frechet": agg(rows, "frechet", plant=tag, method=name),
        "confidence": agg(rows, "confidence", plant=tag, method=name),
        "frechet_rel_to_cfg": agg(rows, "frechet", plant=tag, method=name)[0]
        / max(agg(rows, "frechet", plant=tag, method="CFG (P-control)")[0], 1e-9),
    } for name, _ in configs} for tag in plants}


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/toy"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--k", type=float, default=0.1, help="paper's k for SD3.5 / Qwen-Image")
    ap.add_argument("--steps", type=int, default=30, help="the paper uses 30 sampling steps")
    ap.add_argument("--only", nargs="*", default=None, help="subset of e1..e5")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    n = 1000 if args.quick else 3000
    seeds = [0, 1] if args.quick else [0, 1, 2]
    ws = [1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0]
    want = set(args.only) if args.only else {f"e{i}" for i in range(1, 6)}
    summary: Dict = {"config": dict(n=n, seeds=seeds, ws=ws, k=args.k,
                                    steps=args.steps, lam=LAM)}
    t0 = time.time()

    # Classes overlap (Bayes accuracy ~ 0.7) so guidance has a job and the
    # alignment metric is not saturated at w = 1.5 already.
    ring2 = ring_mixture(k=8, radius=4.0, std=1.5, dim=2)
    ring32 = ring_mixture(k=8, radius=4.0, std=2.0, dim=32)

    def log(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    if "e1" in want:
        log("E1 Pareto sweep, ring2d")
        exp_pareto(args.out, ring2, "ring2d", ws, args.k, args.steps, n, seeds, summary)
        log("E1 Pareto sweep, ring32d")
        exp_pareto(args.out, ring32, "ring32d", ws, args.k, args.steps, n, seeds, summary)
    if "e2" in want:
        log("E2 internal signals")
        exp_signals(args.out, ring2, 5.0, args.k, args.steps, n, summary)
    if "e3" in want:
        log("E3 k sweep")
        exp_k_sweep(args.out, ring2, 5.0, [0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
                    args.steps, n, seeds, summary)
    if "e4" in want:
        log("E4 loop gain / Assumption 2")
        exp_loop_gain(args.out, {"ring2d": ring2, "ring8d": ring_mixture(k=8, radius=4.0, std=1.5, dim=8)},
                      5.0, args.k, args.steps, 64 if args.quick else 256, summary)
    if "e5" in want:
        log("E5 scale transfer")
        exp_transfer(args.out, 5.0, args.k, args.steps, n, seeds, summary)

    path = args.out / "summary.json"
    if path.exists():                      # merge, so `--only eN` does not erase the rest
        with open(path) as fh:
            old = json.load(fh)
        old.update(summary)
        summary = old
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    log(f"done; wrote {args.out}/")


if __name__ == "__main__":
    main()
