"""Pareto assembly: the comparison that can actually rank guidance laws."""

import csv

import pytest

from experiments import pareto


def write(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wri = csv.DictWriter(fh, fieldnames=fields)
        wri.writeheader()
        wri.writerows(rows)
    return path


# ------------------------------------------------------------- interpolation

def test_interpolates_between_measured_points():
    curve = [(0.80, 30.0), (0.90, 40.0)]
    assert pareto.interpolate(curve, 0.85) == pytest.approx(35.0)
    assert pareto.interpolate(curve, 0.80) == pytest.approx(30.0)


@pytest.mark.parametrize("alignment", [0.70, 0.95])
def test_outside_the_measured_range_is_undefined(alignment):
    """Extrapolating a tradeoff curve invents data."""
    assert pareto.interpolate([(0.80, 30.0), (0.90, 40.0)], alignment) is None


def test_vertical_segment_is_undefined():
    """Where CFG's alignment barely moves, the interpolated fidelity is decided
    by the w grid rather than by the measurement."""
    assert pareto.interpolate([(0.90, 30.0), (0.9001, 90.0)], 0.90005) is None


def test_saturated_alignment_is_undefined():
    assert pareto.interpolate([(0.998, 30.0), (1.0, 40.0)], 0.9995) is None


# ------------------------------------------------------------------- ranking

def test_an_arm_below_the_cfg_curve_scores_under_one():
    align = {("cfg", 1.5): 0.80, ("cfg", 3.0): 0.90, ("paper", 3.0): 0.85}
    fid = {("cfg", 1.5): 30.0, ("cfg", 3.0): 40.0, ("paper", 3.0): 31.5}
    row = [r for r in pareto.build(align, fid, "cfg") if r["arm"] == "paper"][0]
    assert row["cfg_fid_at_same_alignment"] == pytest.approx(35.0)
    assert row["ratio"] == pytest.approx(0.9)          # 31.5 / 35.0


def test_an_arm_on_the_cfg_curve_scores_one():
    """The case the review warns about: same tradeoff, different effective w.
    A naive 'lower FID wins' would call this an improvement."""
    align = {("cfg", 1.5): 0.80, ("cfg", 3.0): 0.90, ("paper", 3.0): 0.85}
    fid = {("cfg", 1.5): 30.0, ("cfg", 3.0): 40.0, ("paper", 3.0): 35.0}
    row = [r for r in pareto.build(align, fid, "cfg") if r["arm"] == "paper"][0]
    assert row["ratio"] == pytest.approx(1.0)
    assert row["fid"] < fid[("cfg", 3.0)]              # "better FID" yet no gain


def test_baseline_rows_have_no_ratio():
    align = {("cfg", 1.5): 0.80, ("cfg", 3.0): 0.90}
    fid = {("cfg", 1.5): 30.0, ("cfg", 3.0): 40.0}
    assert all(r["ratio"] is None for r in pareto.build(align, fid, "cfg"))


def test_build_requires_a_baseline_curve_and_overlapping_measurements():
    align, fid = {("cfg", 1.5): 0.8}, {("cfg", 1.5): 30.0}
    with pytest.raises(ValueError, match="at least two scales"):
        pareto.build(align, fid, "cfg")
    with pytest.raises(ValueError, match="both an alignment and a fidelity"):
        pareto.build({("cfg", 1.5): 0.8}, {("cfg", 9.9): 30.0}, "cfg")


# --------------------------------------------------------------------- input

def test_reads_the_files_the_other_steps_produce(tmp_path):
    write(tmp_path / "clip_summary.csv",
          [{"arm": "cfg", "w": "1.5", "mean": "0.80", "n": "1000",
            "paired_delta": "nan", "ci_lo": "nan", "ci_hi": "nan", "prompts_needed": "nan"},
           {"arm": "paper", "w": "1.5", "mean": "0.78", "n": "1000",
            "paired_delta": "-0.02", "ci_lo": "-0.03", "ci_hi": "-0.01", "prompts_needed": "50"}],
          ["arm", "w", "mean", "n", "paired_delta", "ci_lo", "ci_hi", "prompts_needed"])
    assert pareto.read_alignment(tmp_path / "clip_summary.csv") == {
        ("cfg", 1.5): 0.80, ("paper", 1.5): 0.78}

    write(tmp_path / "fid.csv", [{"arm": "cfg", "w": "1.5", "fid": "30.5"}],
          ["arm", "w", "fid"])
    assert pareto.read_fidelity(tmp_path / "fid.csv") == {("cfg", 1.5): 30.5}


def test_missing_fid_columns_are_reported(tmp_path):
    write(tmp_path / "bad.csv", [{"arm": "cfg", "score": "1"}], ["arm", "score"])
    with pytest.raises(ValueError, match=r"missing column\(s\): \['fid', 'w'\]"):
        pareto.read_fidelity(tmp_path / "bad.csv")


def test_nonfinite_rows_are_skipped(tmp_path):
    write(tmp_path / "fid.csv",
          [{"arm": "cfg", "w": "1.5", "fid": "nan"}, {"arm": "cfg", "w": "3.0", "fid": "40"}],
          ["arm", "w", "fid"])
    assert pareto.read_fidelity(tmp_path / "fid.csv") == {("cfg", 3.0): 40.0}
