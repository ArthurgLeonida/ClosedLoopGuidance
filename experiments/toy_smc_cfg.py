"""CFG vs SMC-CFG vs control-theoretic refinements on the analytic plant.

Runs on a laptop CPU in a few minutes. Everything the CFG-Ctrl paper argues
about is *measured* here on a plant with ground truth:

  E1  w_sweep        fidelity (Frechet) and alignment (class accuracy) vs the
                     guidance scale w, per method, with seed error bars. This is
                     the paper's Fig. 6 with a ground-truth metric. Also the
                     Pareto view (Frechet vs accuracy), the honest way to
                     compare guidance laws: a law that merely lowers the
                     effective w moves *along* the CFG curve, not off it.
  E2  diagnostics    per-step |e|, |s|, chatter, switching activity and the
                     "derivative matters" index: does s reach zero? does the
                     derivative term in the sliding surface ever decide?
                     Includes the paper's law with measured-error memory to
                     isolate the effect of storing the corrected error.
  E3  step_transfer  same lam, k at 10..100 sampling steps.
  E4  k_sweep        the chattering regime: sign vs boundary layer as k grows.
  E5  noise          measurement noise on e (a stand-in for network error) at
                     w = 1 and w = 3.
  E6  assumption2    finite-difference Jacobian of e along real trajectories:
                     sign and size of the one-step loop gain vs the paper's
                     Assumption 2 (Gamma ~ w*I), and how often the reaching
                     condition holds for the paper's switching direction.
  E7  scale_transfer the same absolute k on a plant with twice the velocity
                     scale, vs a relative (rms-normalised) k.
  E8  bimodal        a class with an inner and an outer mode: mode collapse.
  E9  clouds         2-D sample clouds for the write-up.

Outputs: results/toy/*.csv, *.png and summary.json (numbers quoted in
docs/CFG-Ctrl_Review_and_Improvements.md).

    python experiments/toy_smc_cfg.py            # full, ~5 min
    python experiments/toy_smc_cfg.py --quick    # ~1.5 min
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

from cfgctrl import GaussianMixtureFlow, SMCConfig, SlidingModeGuidance, bimodal_classes, ring_mixture  # noqa: E402

LAM = 6.0

# Categorical slots in fixed order (dataviz palette); a method keeps its colour
# in every figure. Markers are the secondary encoding.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


# ------------------------------------------------------------------ methods
def method_table(k: float, lam: float = LAM, adapt_rate: float = 0.1) -> List[Tuple[str, SMCConfig]]:
    """Fixed order == fixed colour slot. Index 0 is always the CFG baseline.
    Refinements store the MEASURED error (see controllers.py docstring)."""
    sat = dict(switching="sat", phi=k * lam, store_corrected=False)
    return [
        ("CFG (P-control)", SMCConfig(k=0.0)),
        ("SMC-CFG, paper (sign)", SMCConfig(lam=lam, k=k)),
        ("SMC + boundary layer (sat)", SMCConfig(lam=lam, k=k, **sat)),
        ("SMC + sat + time-scaled", SMCConfig(lam=lam, k=k, time_scaled=True, **sat)),
        ("Super-twisting (2nd order)", SMCConfig(lam=lam, k=0.4 * k, k2=k, z_max=k, super_twisting=True,
                                                time_scaled=True, **sat)),
        ("Adaptive gain", SMCConfig(lam=lam, k=0.5 * k, adaptive=True, adapt_rate=adapt_rate, adapt_band=1.0,
                                    k_min=0.0, k_max=4 * k, time_scaled=True, **sat)),
        ("SMC, Lyapunov-consistent sign", SMCConfig(lam=lam, k=k, flip_sign=True)),
    ]


# ------------------------------------------------------------------ helpers
def run_one(plant: GaussianMixtureFlow, cond: int, w: float, cfg: Optional[SMCConfig], steps: int, n: int,
            seed: int, noise: float = 0.0, shift: float = 1.0, energy: bool = False):
    ctrl = SlidingModeGuidance(cfg) if cfg is not None else None
    x, trace = plant.sample(n, cond, w, ctrl, steps=steps, seed=seed, shift=shift, error_noise_std=noise)
    row = {
        "frechet": plant.frechet_distance(x, cond),
        "accuracy": plant.class_accuracy(x, cond),
        "confidence": plant.class_confidence(x, cond),
    }
    if energy:
        row["energy"] = plant.energy_distance(x, cond, seed=seed + 100)
    if trace.steps:
        row["s_rms_final"] = trace.steps[-1].s_rms
        row["s_rms_mean"] = sum(trace.series("s_rms")) / len(trace.steps)
        row["chatter_mean"] = sum(trace.series("chatter")) / len(trace.steps)
        row["switch_activity_mean"] = sum(trace.series("switch_activity")) / len(trace.steps)
        row["deriv_matters_mean"] = sum(trace.series("deriv_matters")) / len(trace.steps)
        row["delta_rms_mean"] = sum(trace.series("delta_rms")) / len(trace.steps)
        row["k_eff_final"] = trace.steps[-1].k_eff
    return row, x, trace


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        wri = csv.DictWriter(fh, fieldnames=keys)
        wri.writeheader()
        wri.writerows(rows)


def mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    vals = [v for v in vals if v == v]           # drop NaN
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / max(len(vals) - 1, 1)
    return m, math.sqrt(v)


def agg(rows: List[Dict], key: str, **where) -> Tuple[float, float]:
    return mean_std([r[key] for r in rows if all(r.get(f) == v for f, v in where.items())])


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


def line(ax, xs, ys, idx: int, label: str, yerr=None, **kw) -> None:
    ax.errorbar(xs, ys, yerr=yerr, color=COLORS[idx], marker=MARKERS[idx], markersize=5,
                linewidth=1.8, capsize=2, label=label, markeredgecolor=SURFACE, markeredgewidth=0.8, **kw)


def savefig(fig, path: Path) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------- experiments
def exp_w_sweep(out: Path, plant: GaussianMixtureFlow, tag: str, ws: Sequence[float], k: float,
                steps: int, n: int, seeds: Sequence[int], summary: Dict) -> None:
    rows = []
    methods = method_table(k)
    for name, cfg in methods:
        for w in ws:
            for seed in seeds:
                r, _, _ = run_one(plant, 0, w, cfg, steps, n, seed)
                rows.append(dict(plant=tag, method=name, w=w, seed=seed, **r))
    write_csv(out / f"e1_w_sweep_{tag}.csv", rows)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for i, (name, _) in enumerate(methods):
        fr = [agg(rows, "frechet", method=name, w=w) for w in ws]
        cf = [agg(rows, "confidence", method=name, w=w) for w in ws]
        line(axes[0], ws, [m for m, _ in fr], i, name, yerr=[s for _, s in fr])
        line(axes[1], ws, [m for m, _ in cf], i, name, yerr=[s for _, s in cf])
        axes[2].plot([m for m, _ in cf], [m for m, _ in fr], color=COLORS[i], marker=MARKERS[i],
                     markersize=5, linewidth=1.8, label=name, markeredgecolor=SURFACE, markeredgewidth=0.8)
    truth = plant.true_class_samples(n, 0, seed=77)
    bayes = plant.class_accuracy(truth, 0)
    truth_conf = plant.class_confidence(truth, 0)
    axes[1].axhline(truth_conf, color=TEXT2, linewidth=1.0, linestyle=":", label=f"true class law ({truth_conf:.2f})")
    style(axes[0], f"Fidelity vs guidance scale ({tag}, k={k})", "guidance scale w", "Frechet distance to true class (lower is better)")
    style(axes[1], "Alignment vs guidance scale", "guidance scale w", "mean p(class | x)  (higher is better)")
    style(axes[2], "Pareto view: one point per w, lower-right is better", "mean p(class | x)", "Frechet distance")
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[1].legend(fontsize=6.5, frameon=False, labelcolor=TEXT2, loc="lower right")
    savefig(fig, out / f"e1_w_sweep_{tag}.png")

    summary[f"e1_{tag}"] = {
        "bayes_accuracy": bayes,
        "true_confidence": truth_conf,
        **{name: {str(w): {"frechet": agg(rows, "frechet", method=name, w=w),
                           "accuracy": agg(rows, "accuracy", method=name, w=w),
                           "confidence": agg(rows, "confidence", method=name, w=w)} for w in ws}
           for name, _ in methods},
    }


def exp_diagnostics(out: Path, plant: GaussianMixtureFlow, w: float, k: float, steps: int, n: int,
                    summary: Dict) -> None:
    mt = method_table(k)
    methods = [
        (mt[1][0], mt[1][1], 1),
        ("paper law, measured-error memory", SMCConfig(lam=LAM, k=k, store_corrected=False), 7),
        (mt[2][0], mt[2][1], 2),
        (mt[3][0], mt[3][1], 3),
    ]
    keys = ("e_rms", "s_rms", "chatter", "switch_activity", "deriv_matters")
    fig, axes = plt.subplots(1, len(keys), figsize=(3.8 * len(keys), 3.6))
    rows = []
    for name, cfg, idx in methods:
        r, _, trace = run_one(plant, 0, w, cfg, steps, n, 0)
        sig = trace.sigmas
        for key, ax in zip(keys, axes):
            ax.plot(sig, trace.series(key), color=COLORS[idx], marker=MARKERS[idx], markersize=3.5,
                    linewidth=1.6, label=name, markeredgecolor=SURFACE, markeredgewidth=0.6)
        for t, st in enumerate(trace.steps):
            rows.append(dict(method=name, step=t, sigma=sig[t], **{kk: getattr(st, kk) for kk in keys},
                             delta_rms=st.delta_rms, inside_layer=st.inside_layer, k_eff=st.k_eff))
        summary.setdefault("e2", {})[name] = {
            "s_rms_first": trace.steps[0].s_rms, "s_rms_final": trace.steps[-1].s_rms,
            "chatter_mean": r["chatter_mean"], "deriv_matters_mean": r["deriv_matters_mean"],
            "chatter_last10": sum(trace.series("chatter")[-10:]) / 10,
            "switch_activity_last10": sum(trace.series("switch_activity")[-10:]) / 10,
            "delta_rms_last10": sum(trace.series("delta_rms")[-10:]) / 10,
            "frechet": r["frechet"], "accuracy": r["accuracy"],
        }
    write_csv(out / "e2_diagnostics.csv", rows)
    style(axes[0], f"Measured error |e| (w={w}, k={k})", "sigma (1 = noise, 0 = data)", "rms(e)")
    style(axes[1], "Sliding variable |s|: does it reach 0?", "sigma", "rms(s)")
    style(axes[2], "Chatter index: sign(s) flips per element", "sigma", "fraction flipped")
    style(axes[3], "Switching activity rms(delta_t - delta_t-1)", "sigma", "rms (2k = full chatter)")
    style(axes[4], "Derivative-matters index: sign(s) != sign(e_prev)", "sigma", "fraction")
    for ax in axes:
        ax.invert_xaxis()
    axes[1].legend(fontsize=6.5, frameon=False, labelcolor=TEXT2)
    savefig(fig, out / "e2_diagnostics.png")


def exp_step_transfer(out: Path, plant: GaussianMixtureFlow, w: float, k: float, step_list: Sequence[int],
                      n: int, seeds: Sequence[int], summary: Dict) -> None:
    idxs = (0, 1, 2, 3)
    methods = [method_table(k)[i] for i in idxs]
    rows = []
    for (name, cfg), idx in zip(methods, idxs):
        for steps in step_list:
            for seed in seeds:
                r, _, _ = run_one(plant, 0, w, cfg, steps, n, seed)
                rows.append(dict(method=name, steps=steps, seed=seed, **r))
    write_csv(out / "e3_step_transfer.csv", rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    cfg_fr = {s: agg(rows, "frechet", method=methods[0][0], steps=s)[0] for s in step_list}
    for (name, _), idx in zip(methods, idxs):
        fr = [agg(rows, "frechet", method=name, steps=s) for s in step_list]
        rel = [fr_i[0] / cfg_fr[s] for fr_i, s in zip(fr, step_list)]
        dr = [agg(rows, "deriv_matters_mean", method=name, steps=s) for s in step_list]
        line(axes[0], step_list, [m for m, _ in fr], idx, name, yerr=[s for _, s in fr])
        line(axes[1], step_list, rel, idx, name)
        if idx != 0:
            line(axes[2], step_list, [m for m, _ in dr], idx, name)
    style(axes[0], f"Same lam, k at different step counts (w={w})", "sampling steps", "Frechet distance")
    style(axes[1], "Effect relative to CFG at the same step count", "sampling steps", "Frechet / Frechet(CFG)")
    style(axes[2], "How often the derivative term decides sign(s)", "sampling steps", "mean derivative-matters index")
    for ax in axes:
        ax.set_xscale("log")
    axes[0].legend(fontsize=6.5, frameon=False, labelcolor=TEXT2)
    savefig(fig, out / "e3_step_transfer.png")
    summary["e3"] = {name: {str(s): {"frechet": agg(rows, "frechet", method=name, steps=s),
                                     "rel_to_cfg": agg(rows, "frechet", method=name, steps=s)[0] / cfg_fr[s],
                                     "deriv_matters": agg(rows, "deriv_matters_mean", method=name, steps=s)}
                            for s in step_list} for name, _ in methods}


def exp_k_sweep(out: Path, plant: GaussianMixtureFlow, w: float, ks: Sequence[float], steps: int, n: int,
                seeds: Sequence[int], summary: Dict) -> None:
    rows = []
    for k in ks:
        for idx in (1, 2):
            name, cfg = method_table(k)[idx]
            for seed in seeds:
                r, _, _ = run_one(plant, 0, w, cfg, steps, n, seed)
                rows.append(dict(method=name, k=k, seed=seed, **r))
    base = [run_one(plant, 0, w, SMCConfig(k=0.0), steps, n, seed)[0] for seed in seeds]
    write_csv(out / "e4_k_sweep.csv", rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    names = {idx: method_table(0.1)[idx][0] for idx in (1, 2)}
    for idx, name in names.items():
        fr = [agg(rows, "frechet", method=name, k=k) for k in ks]
        ac = [agg(rows, "accuracy", method=name, k=k) for k in ks]
        sa = [agg(rows, "switch_activity_mean", method=name, k=k)[0] / (2 * k) for k in ks]
        line(axes[0], ks, [m for m, _ in fr], idx, name, yerr=[s for _, s in fr])
        line(axes[1], ks, [m for m, _ in ac], idx, name, yerr=[s for _, s in ac])
        line(axes[2], ks, sa, idx, name)
    bf, ba = mean_std([b["frechet"] for b in base]), mean_std([b["accuracy"] for b in base])
    axes[0].axhline(bf[0], color=COLORS[0], linewidth=1.4, linestyle="--", label="CFG (k=0)")
    axes[1].axhline(ba[0], color=COLORS[0], linewidth=1.4, linestyle="--", label="CFG (k=0)")
    style(axes[0], f"Fidelity vs switching gain k (w={w})", "k", "Frechet distance")
    style(axes[1], "Alignment vs k", "k", "class accuracy")
    style(axes[2], "Normalised switching activity", "k", "rms(delta_t - delta_t-1) / 2k  (1 = full chatter)")
    for ax in axes:
        ax.set_xscale("log")
    axes[0].legend(fontsize=6.5, frameon=False, labelcolor=TEXT2)
    savefig(fig, out / "e4_k_sweep.png")
    summary["e4"] = {
        "cfg": {"frechet": bf, "accuracy": ba},
        **{name: {str(k): {"frechet": agg(rows, "frechet", method=name, k=k),
                           "accuracy": agg(rows, "accuracy", method=name, k=k),
                           "switch_activity_norm": agg(rows, "switch_activity_mean", method=name, k=k)[0] / (2 * k)}
                  for k in ks} for name in names.values()},
    }


def exp_noise(out: Path, plant: GaussianMixtureFlow, ws: Sequence[float], k: float, noises: Sequence[float],
              steps: int, n: int, seeds: Sequence[int], summary: Dict) -> None:
    methods = method_table(k)[:3]
    rows = []
    for w in ws:
        for name, cfg in methods:
            for nz in noises:
                for seed in seeds:
                    r, _, _ = run_one(plant, 0, w, cfg, steps, n, seed, noise=nz)
                    rows.append(dict(method=name, w=w, noise=nz, seed=seed, **r))
    write_csv(out / "e5_noise.csv", rows)
    fig, axes = plt.subplots(len(ws), 2, figsize=(9, 3.6 * len(ws)), squeeze=False)
    for r_i, w in enumerate(ws):
        for i, (name, _) in enumerate(methods):
            fr = [agg(rows, "frechet", method=name, w=w, noise=nz) for nz in noises]
            ac = [agg(rows, "accuracy", method=name, w=w, noise=nz) for nz in noises]
            line(axes[r_i][0], noises, [m for m, _ in fr], i, name, yerr=[s for _, s in fr])
            line(axes[r_i][1], noises, [m for m, _ in ac], i, name, yerr=[s for _, s in ac])
        style(axes[r_i][0], f"Noise on the measured error e (w={w}, k={k})", "std of noise added to e", "Frechet distance")
        style(axes[r_i][1], f"Alignment under measurement noise (w={w})", "std of noise added to e", "class accuracy")
    axes[0][0].legend(fontsize=6.5, frameon=False, labelcolor=TEXT2)
    savefig(fig, out / "e5_noise.png")
    summary["e5"] = {str(w): {name: {str(nz): {"frechet": agg(rows, "frechet", method=name, w=w, noise=nz),
                                               "accuracy": agg(rows, "accuracy", method=name, w=w, noise=nz)}
                                     for nz in noises} for name, _ in methods} for w in ws}


def exp_assumption2(out: Path, plants: Dict[str, GaussianMixtureFlow], w: float, k: float, steps: int,
                    n: int, summary: Dict) -> None:
    """Along a paper-SMC trajectory, measure J = de/dx.

    The Euler step moves x by -dt*w*(e + delta), so the correction changes the
    NEXT measured error by  -dt*w*J*delta. Define M = -J: the paper's
    reaching condition needs  s^T M sign(s) > 0  (Assumption 2 even wants
    -dt*w*J ~ +w*I, i.e. M ~ I up to a 1/sqrt(D) relative deviation).
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
        eig_min, eig_max, eig_mean, sigmas = [], [], [], []
        for i in range(steps):
            s_now, s_next = float(sig[i]), float(sig[i + 1])
            dt = s_now - s_next
            v_u = plant.velocity(x, s_now, None)
            e = plant.velocity(x, s_now, 0) - v_u
            J = plant.error_jacobian(x, s_now, 0)
            M = -J
            symm = 0.5 * (M + M.transpose(1, 2))
            eig = torch.linalg.eigvalsh(symm)                          # [n, D]
            e_prev = e if e_prev is None else e_prev
            s = (e - e_prev) + LAM * e_prev
            sg = torch.sign(s)
            reach = torch.einsum("bi,bij,bj->b", s, M, sg)            # > 0: paper's -k sign(s) reduces |s|
            dev = torch.linalg.matrix_norm(M - torch.eye(D)[None], ord=2)
            loop_gain = dt * w * torch.linalg.matrix_norm(J, ord=2)    # per-step authority of delta over e
            rows.append(dict(plant=tag, step=i, sigma=s_now,
                             eig_min=float(eig.min()), eig_mean=float(eig.mean()), eig_max=float(eig.max()),
                             frac_reaching_ok_paper_sign=float((reach > 0).float().mean()),
                             frac_reaching_ok_flipped_sign=float((reach < 0).float().mean()),
                             frac_assumption2=float((dev < 1 / math.sqrt(D)).float().mean()),
                             dev_from_identity_median=float(dev.median()),
                             loop_gain_per_step_median=float(loop_gain.median()),
                             loop_gain_per_step_max=float(loop_gain.max()),
                             k=k))
            eig_min.append(float(eig.min())); eig_max.append(float(eig.max())); eig_mean.append(float(eig.mean()))
            sigmas.append(s_now)
            e_app = ctrl.correct(e, dt)
            e_prev = e_app
            x = x + (s_next - s_now) * (v_u + w * e_app)
        ax = axes[0][p_i]
        ax.fill_between(sigmas, eig_min, eig_max, color=COLORS[6], alpha=0.18, linewidth=0, label="min..max over samples")
        ax.plot(sigmas, eig_mean, color=COLORS[6], linewidth=1.8, label="mean eigenvalue of sym(-de/dx)")
        ax.plot(sigmas, [1.0] * len(sigmas), color=COLORS[0], linewidth=1.4, linestyle="--", label="Assumption 2 wants ~ +1 (identity)")
        ax.axhline(0.0, color=TEXT2, linewidth=0.8)
        style(ax, f"One-step loop gain along trajectories ({tag}, D={D})", "sigma", "eigenvalues of sym(-de/dx)")
        ax.invert_xaxis()
        ax.set_yscale("symlog", linthresh=1.0)
        ax.legend(fontsize=6.5, frameon=False, labelcolor=TEXT2)
        pr = [r for r in rows if r["plant"] == tag]
        inner = pr[1:]                                                   # skip sigma = 1 where J = 0 exactly
        summary.setdefault("e6", {})[tag] = {
            "frac_assumption2_mean": sum(r["frac_assumption2"] for r in inner) / len(inner),
            "frac_reaching_ok_paper_sign_mean": sum(r["frac_reaching_ok_paper_sign"] for r in inner) / len(inner),
            "frac_reaching_ok_flipped_sign_mean": sum(r["frac_reaching_ok_flipped_sign"] for r in inner) / len(inner),
            "loop_gain_per_step_median_over_traj": sorted(r["loop_gain_per_step_median"] for r in inner)[len(inner) // 2],
            "loop_gain_per_step_max_over_traj": max(r["loop_gain_per_step_max"] for r in inner),
            "k": k,
            "eig_min_overall": min(r["eig_min"] for r in inner),
            "eig_max_overall": max(r["eig_max"] for r in inner),
            "eig_mean_at_sigma_0.8": [r["eig_mean"] for r in inner if abs(r["sigma"] - 0.8) < 0.02][:1],
        }
    write_csv(out / "e6_assumption2.csv", rows)
    savefig(fig, out / "e6_assumption2.png")


def exp_scale_transfer(out: Path, w: float, k: float, steps: int, n: int, seeds: Sequence[int], summary: Dict) -> None:
    plants = {"radius 4": ring_mixture(radius=4.0, std=1.5), "radius 8 (2x scale)": ring_mixture(radius=8.0, std=3.0)}
    configs = [
        ("CFG (P-control)", SMCConfig(k=0.0), 0),
        ("SMC paper, absolute k", SMCConfig(lam=LAM, k=k), 1),
        ("SMC sat, relative k (fraction of rms e)",
         SMCConfig(lam=LAM, k=k, relative_gain=True, switching="sat", phi=k * LAM, store_corrected=False), 2),
    ]
    rows = []
    for tag, plant in plants.items():
        for name, cfg, idx in configs:
            for seed in seeds:
                r, _, _ = run_one(plant, 0, w, cfg, steps, n, seed)
                rows.append(dict(plant=tag, method=name, seed=seed, **r))
    write_csv(out / "e7_scale_transfer.csv", rows)
    summary["e7"] = {tag: {name: {
        "frechet": agg(rows, "frechet", plant=tag, method=name),
        "accuracy": agg(rows, "accuracy", plant=tag, method=name),
        "delta_rms_mean": agg(rows, "delta_rms_mean", plant=tag, method=name),
        "frechet_rel_to_cfg": agg(rows, "frechet", plant=tag, method=name)[0]
        / max(agg(rows, "frechet", plant=tag, method="CFG (P-control)")[0], 1e-9),
    } for name, _, _ in configs} for tag in plants}


def exp_bimodal(out: Path, ws: Sequence[float], k: float, steps: int, n: int, seeds: Sequence[int], summary: Dict) -> None:
    plant = bimodal_classes()
    nc = plant.n_classes
    methods = method_table(k)[:3]
    rows = []
    for name, cfg in methods:
        for w in ws:
            for seed in seeds:
                r, x, _ = run_one(plant, 0, w, cfg, steps, n, seed, energy=True)
                lp = plant.class_log_posterior(x)
                in_class = lp.argmax(1) == 0
                diff = x[in_class][:, None, :] - plant.means[None, [0, nc]]     # outer, inner
                share_inner = float((diff.pow(2).sum(-1).argmin(1) == 1).float().mean()) if in_class.any() else float("nan")
                rows.append(dict(method=name, w=w, seed=seed, share_inner=share_inner, **r))
    write_csv(out / "e8_bimodal.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for i, (name, _) in enumerate(methods):
        en = [agg(rows, "energy", method=name, w=w) for w in ws]
        sh = [agg(rows, "share_inner", method=name, w=w) for w in ws]
        line(axes[0], ws, [m for m, _ in en], i, name, yerr=[s for _, s in en])
        line(axes[1], ws, [m for m, _ in sh], i, name, yerr=[s for _, s in sh])
    axes[1].axhline(0.5, color=TEXT2, linewidth=1.0, linestyle=":", label="true share (0.5)")
    style(axes[0], f"Two-mode class: energy distance to truth (k={k})", "guidance scale w", "energy distance (lower is better)")
    style(axes[1], "Mode collapse: share of samples in the inner mode", "guidance scale w", "inner-mode share (truth = 0.5)")
    for ax in axes:
        ax.set_xscale("log")
    axes[1].legend(fontsize=6.5, frameon=False, labelcolor=TEXT2)
    savefig(fig, out / "e8_bimodal.png")
    summary["e8"] = {name: {str(w): {"energy": agg(rows, "energy", method=name, w=w),
                                     "share_inner": agg(rows, "share_inner", method=name, w=w)}
                            for w in ws} for name, _ in methods}


def exp_excess_only(out: Path, plants: Dict[str, GaussianMixtureFlow], ws: Sequence[float], k: float, steps: int,
                    n: int, seeds: Sequence[int], summary: Dict) -> None:
    """The paper corrects v_cond + (w-1)e as a whole, so at w = 1 it is not the
    conditional sampler. `excess_only` shrinks only the extrapolation (w-1)e:
    exactly CFG at w = 1, (w-1)/w of the correction otherwise."""
    sat = dict(switching="sat", phi=k * LAM, store_corrected=False)
    configs = [
        ("CFG (P-control)", SMCConfig(k=0.0), 0),
        ("SMC-CFG, paper (sign)", SMCConfig(lam=LAM, k=k), 1),
        ("SMC + boundary layer (sat)", SMCConfig(lam=LAM, k=k, **sat), 2),
        ("paper (sign), extrapolation only", SMCConfig(lam=LAM, k=k, excess_only=True), 5),
        ("sat, extrapolation only", SMCConfig(lam=LAM, k=k, excess_only=True, **sat), 7),
    ]
    rows = []
    for tag, plant in plants.items():
        for name, cfg, idx in configs:
            for w in ws:
                for seed in seeds:
                    r, _, _ = run_one(plant, 0, w, cfg, steps, n, seed)
                    rows.append(dict(plant=tag, method=name, w=w, seed=seed, **r))
    write_csv(out / "e10_excess_only.csv", rows)
    fig, axes = plt.subplots(len(plants), 2, figsize=(10, 3.8 * len(plants)), squeeze=False)
    for p_i, (tag, plant) in enumerate(plants.items()):
        for name, _, idx in configs:
            fr = [agg(rows, "frechet", plant=tag, method=name, w=w) for w in ws]
            cf = [agg(rows, "confidence", plant=tag, method=name, w=w) for w in ws]
            line(axes[p_i][0], ws, [m for m, _ in fr], idx, name, yerr=[s for _, s in fr])
            axes[p_i][1].plot([m for m, _ in cf], [m for m, _ in fr], color=COLORS[idx], marker=MARKERS[idx],
                              markersize=5, linewidth=1.8, label=name, markeredgecolor=SURFACE, markeredgewidth=0.8)
        style(axes[p_i][0], f"Fidelity vs guidance scale ({tag}, k={k})", "guidance scale w", "Frechet distance")
        style(axes[p_i][1], f"Pareto view ({tag}), lower-right is better", "mean p(class | x)", "Frechet distance")
        axes[p_i][0].set_xscale("log")
    axes[0][1].legend(fontsize=6.5, frameon=False, labelcolor=TEXT2)
    savefig(fig, out / "e10_excess_only.png")
    summary["e10"] = {tag: {name: {str(w): {"frechet": agg(rows, "frechet", plant=tag, method=name, w=w),
                                            "confidence": agg(rows, "confidence", plant=tag, method=name, w=w),
                                            "accuracy": agg(rows, "accuracy", plant=tag, method=name, w=w)}
                                   for w in ws} for name, _, _ in configs} for tag in plants}


def exp_clouds(out: Path, plant: GaussianMixtureFlow, ws: Sequence[float], k: float, steps: int, n: int) -> None:
    mt = method_table(k)
    methods = [(mt[i][0], mt[i][1], i) for i in (0, 1, 2, 6)]
    fig, axes = plt.subplots(len(ws), len(methods), figsize=(3.6 * len(methods), 3.5 * len(ws)), squeeze=False)
    truth = plant.true_class_samples(n, 0, seed=9)
    for r_i, w in enumerate(ws):
        for c_i, (name, cfg, idx) in enumerate(methods):
            _, x, _ = run_one(plant, 0, w, cfg, steps, n, 0)
            ax = axes[r_i][c_i]
            ax.scatter(truth[:, 0], truth[:, 1], s=4, color=GRID, label="true class law")
            ax.scatter(x[:, 0], x[:, 1], s=4, color=COLORS[idx], alpha=0.6, label=name)
            ax.scatter(plant.means[:, 0], plant.means[:, 1], s=30, marker="x", color=TEXT2, linewidth=1.0)
            fd, acc = plant.frechet_distance(x, 0), plant.class_accuracy(x, 0)
            style(ax, f"{name}\nw={w}  Frechet={fd:.2f}  acc={acc:.2f}", "", "")
            ax.set_aspect("equal")
            ax.set_xlim(-7, 7)
            ax.set_ylim(-7, 7)
            ax.set_xticks([])
            ax.set_yticks([])
    savefig(fig, out / "e9_clouds.png")


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/toy"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--k", type=float, default=0.1, help="paper's k for SD3.5 / Qwen-Image")
    ap.add_argument("--steps", type=int, default=30, help="paper uses 30 sampling steps")
    ap.add_argument("--only", nargs="*", default=None, help="subset of e1..e9")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    n = 1000 if args.quick else 3000
    seeds = [0, 1] if args.quick else [0, 1, 2]
    ws = [1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0]
    want = set(args.only) if args.only else {f"e{i}" for i in range(1, 11)}
    summary: Dict = {"config": dict(n=n, seeds=seeds, ws=ws, k=args.k, steps=args.steps, lam=LAM)}
    t0 = time.time()

    # Classes overlap (Bayes accuracy ~ 0.7) so that guidance has a job and the
    # alignment metrics are not saturated at w = 1.5 already.
    ring2 = ring_mixture(k=8, radius=4.0, std=1.5, dim=2)
    ring32 = ring_mixture(k=8, radius=4.0, std=2.0, dim=32)

    def log(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    if "e1" in want:
        log("E1 w sweep, ring2d")
        exp_w_sweep(args.out, ring2, "ring2d", ws, args.k, args.steps, n, seeds, summary)
        log("E1 w sweep, ring32d")
        exp_w_sweep(args.out, ring32, "ring32d", ws, args.k, args.steps, n, seeds, summary)
    if "e2" in want:
        log("E2 diagnostics")
        exp_diagnostics(args.out, ring2, 5.0, args.k, args.steps, n, summary)
    if "e3" in want:
        log("E3 step transfer")
        exp_step_transfer(args.out, ring2, 5.0, args.k, [10, 20, 30, 50, 100], n, seeds, summary)
    if "e4" in want:
        log("E4 k sweep")
        exp_k_sweep(args.out, ring2, 5.0, [0.02, 0.05, 0.1, 0.2, 0.5, 1.0], args.steps, n, seeds, summary)
    if "e5" in want:
        log("E5 measurement noise")
        exp_noise(args.out, ring2, [1.0, 3.0], args.k, [0.0, 0.1, 0.2, 0.4, 0.8], args.steps, n, seeds, summary)
    if "e6" in want:
        log("E6 assumption 2 / Jacobian")
        exp_assumption2(args.out, {"ring2d": ring2, "ring8d": ring_mixture(k=8, radius=4.0, std=1.5, dim=8)},
                        5.0, args.k, args.steps, 64 if args.quick else 256, summary)
    if "e7" in want:
        log("E7 scale transfer")
        exp_scale_transfer(args.out, 5.0, args.k, args.steps, n, seeds, summary)
    if "e8" in want:
        log("E8 bimodal classes")
        exp_bimodal(args.out, [1.0, 1.5, 2.0, 3.0, 5.0, 10.0], args.k, args.steps, n, seeds, summary)
    if "e9" in want:
        log("E9 sample clouds")
        exp_clouds(args.out, ring2, [2.0, 5.0], args.k, args.steps, 1500)
    if "e10" in want:
        log("E10 extrapolation-only shrink")
        exp_excess_only(args.out, {"ring2d": ring2, "ring32d": ring32}, [1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0],
                        args.k, args.steps, n, seeds, summary)

    # Merge into an existing summary so `--only eN` re-runs do not erase the rest.
    path = args.out / "summary.json"
    if path.exists():
        with open(path) as fh:
            old = json.load(fh)
        old.update(summary)
        summary = old
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    log(f"done; wrote {args.out}/")


if __name__ == "__main__":
    main()
