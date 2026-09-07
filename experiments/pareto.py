"""Join alignment and fidelity into the comparison that can actually rank laws.

A guidance law cannot be ranked by one number. Every arm traces a *curve* as the
guidance scale varies: more guidance buys alignment and costs fidelity. A law
that merely attenuates guidance slides along CFG's own curve and will look
better on whichever axis you happen to read alone.

So the question is not "which arm has the best FID" but: **at the same
alignment, does this arm reach a lower FID than CFG?** That is what this script
computes. For each (arm, w) it interpolates the CFG curve at that arm's
alignment and reports the ratio:

    ratio = FID(arm)  /  FID(CFG interpolated at the same alignment)

    < 1   genuinely better: below CFG's tradeoff curve
    ~ 1   on CFG's curve: the same tradeoff reached at a different w
    > 1   worse

Inputs are what the other two steps already produce:

    results/<run>/clip_summary.csv     from `evaluate.py clip`
    a CSV with columns arm,w,fid       from clean-fid, see --fid

    python experiments/pareto.py --run results/coco_test --fid results/coco_test/fid.csv

Two limits are enforced rather than papered over. The ratio is only defined
where the arm's alignment falls inside the range CFG actually measured, and it
is unstable where the CFG curve is near-vertical (alignment saturated), so both
cases are reported as undefined instead of a number.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

Point = Tuple[float, float]          # (alignment, fidelity)


def read_alignment(path: Path) -> Dict[Tuple[str, float], float]:
    """(arm, w) -> mean CLIP score, from evaluate.py's clip_summary.csv."""
    out: Dict[Tuple[str, float], float] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                value = float(row["mean"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                out[(row["arm"], float(row["w"]))] = value
    if not out:
        raise ValueError(f"{path} has no usable rows; run `evaluate.py clip` first")
    return out


def read_fidelity(path: Path, column: str = "fid") -> Dict[Tuple[str, float], float]:
    """(arm, w) -> fidelity. Any CSV with arm, w and the named column.

    `column` exists so the same join can be read on KID, which experiments/fid.py
    writes alongside FID. At a thousand images per cell KID is the sounder of the
    two: its estimator is unbiased, while FID at that size is dominated by a
    sample-size bias term.
    """
    out: Dict[Tuple[str, float], float] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {"arm", "w", column} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing column(s): {sorted(missing)}")
        for row in reader:
            try:
                value = float(row[column])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                out[(row["arm"], float(row["w"]))] = value
    if not out:
        raise ValueError(f"{path} has no usable rows")
    return out


def interpolate(curve: Sequence[Point], alignment: float,
                min_gap: float = 1e-3, saturated: float = 1e-3) -> Optional[float]:
    """Fidelity of `curve` at `alignment`, by linear interpolation.

    Returns None where the answer would not mean anything: outside the measured
    range, or across a segment whose endpoints have nearly equal alignment (the
    curve is vertical there, so the interpolated fidelity is dominated by the
    grid rather than by the data).
    """
    pts = sorted(curve)
    if len(pts) < 2 or not math.isfinite(alignment):
        return None
    if alignment < pts[0][0] or alignment > pts[-1][0]:
        return None
    if alignment >= 1.0 - saturated:
        return None
    for (a0, f0), (a1, f1) in zip(pts, pts[1:]):
        if a0 <= alignment <= a1:
            if a1 - a0 < min_gap:
                return None
            return f0 + (alignment - a0) / (a1 - a0) * (f1 - f0)
    return None


def build(align: Dict[Tuple[str, float], float], fid: Dict[Tuple[str, float], float],
          baseline: str) -> List[Dict]:
    """One row per (arm, w) that has both measurements."""
    keys = sorted(set(align) & set(fid), key=lambda k: (k[0], k[1]))
    if not keys:
        raise ValueError("no (arm, w) has both an alignment and a fidelity number")
    base_curve = [(align[k], fid[k]) for k in keys if k[0] == baseline]
    if len(base_curve) < 2:
        raise ValueError(f"baseline {baseline!r} needs at least two scales to form a curve")

    rows = []
    for arm, w in keys:
        a, f = align[(arm, w)], fid[(arm, w)]
        ref = None if arm == baseline else interpolate(base_curve, a)
        rows.append({
            "arm": arm, "w": w, "alignment": a, "fid": f,
            "cfg_fid_at_same_alignment": ref,
            "ratio": (f / ref) if ref else None,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="a grid output directory")
    ap.add_argument("--fid", type=Path, required=True, help="CSV with arm,w,fid")
    ap.add_argument("--baseline", default="cfg")
    ap.add_argument("--clip-summary", type=Path, default=None,
                    help="default: <run>/clip_summary.csv")
    ap.add_argument("--fid-column", default="fid",
                    help="which fidelity column to join on; 'kid' is the sounder "
                         "choice when each cell holds only ~1000 images")
    args = ap.parse_args()

    clip_path = args.clip_summary or (args.run / "clip_summary.csv")
    try:
        rows = build(read_alignment(clip_path),
                     read_fidelity(args.fid, args.fid_column), args.baseline)
    except ValueError as exc:
        ap.error(str(exc))

    label = args.fid_column.upper()
    print(f"{'arm':<10}{'w':>6}{'alignment':>12}{label:>10}"
          f"{'CFG at same':>13}{'ratio':>8}")
    for r in rows:
        ref = f"{r['cfg_fid_at_same_alignment']:.3f}" if r["cfg_fid_at_same_alignment"] else "--"
        ratio = f"{r['ratio']:.3f}" if r["ratio"] else ("base" if r["arm"] == args.baseline else "--")
        mark = "  *" if (r["ratio"] and r["ratio"] < 1) else ""
        print(f"{r['arm']:<10}{r['w']:>6.2f}{r['alignment']:>12.4f}{r['fid']:>10.3f}"
              f"{ref:>13}{ratio:>8}{mark}")

    out = args.run / ("pareto.csv" if args.fid_column == "fid"
                      else f"pareto_{args.fid_column}.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wri = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wri.writeheader()
        wri.writerows(rows)
    print(f"\nwrote {out}")

    beat = [r for r in rows if r["ratio"] and r["ratio"] < 1]
    print("\n  * = below the CFG curve at matched alignment, i.e. a better tradeoff,")
    print("      not merely a different point on the same tradeoff.")
    if not beat:
        print("  No arm is below the CFG curve anywhere it can be compared.")
        print("  That is a result: on this model these laws reproduce CFG's tradeoff.")
    else:
        best = min(beat, key=lambda r: r["ratio"])
        print(f"  Best: {best['arm']} at w={best['w']:g}, "
              f"{(1 - best['ratio']) * 100:.1f}% below the CFG curve.")
    print("\n  '--' means the ratio is undefined here: the arm's alignment falls")
    print("  outside CFG's measured range, or the CFG curve is vertical there.")
    print("  FID is a dataset statistic, not a per-image score, so these numbers")
    print("  carry no confidence interval. Treat small differences as unresolved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
