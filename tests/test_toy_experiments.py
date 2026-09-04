"""Regression checks for experiment interpretation and result provenance."""

import csv
import json
import math
import statistics
import sys

import pytest
import torch

from cfgctrl import ring_mixture
from experiments import toy_smc_cfg as experiments


@pytest.fixture(autouse=True, scope="module")
def _small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_partial_summaries_keep_each_experiments_parameters_and_strict_json(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"config": {"steps": 30}, "e1_ring2d": {"ratio": 1.0}}))
    experiments.write_summary(path, {"config": {"steps": 5}, "e2": {"ratio": float("nan")}})

    def reject_nonfinite(value):
        raise AssertionError(f"invalid JSON number: {value}")

    result = json.loads(path.read_text(), parse_constant=reject_nonfinite)
    assert result["e2"]["ratio"] is None
    assert result["e1_ring2d"]["ratio"] == 1.0
    assert result["experiment_configs"]["e1_ring2d"] == {"steps": 30, "schema_version": 1}
    assert result["experiment_configs"]["e2"] == {"steps": 5, "schema_version": 2}
    assert not path.with_suffix(".json.tmp").exists()


def test_matched_alignment_ratios_omit_saturation_and_degenerate_intervals():
    rows = [
        dict(method="CFG (P-control)", w=1.0, confidence=0.5, frechet=2.0),
        dict(method="CFG (P-control)", w=2.0, confidence=0.8, frechet=5.0),
        dict(method="variant", w=1.0, confidence=0.6, frechet=1.5),
        dict(method="variant", w=2.0, confidence=0.9999, frechet=3.0),
    ]
    ratios = experiments.pareto_ratio(rows, "variant", [1.0, 2.0])
    assert ratios["1.0"] == pytest.approx(0.5)
    assert math.isnan(ratios["2.0"])
    rows[1]["confidence"] = rows[0]["confidence"]
    rows[2]["confidence"] = rows[0]["confidence"]
    assert math.isnan(experiments.pareto_ratio(rows, "variant", [1.0, 2.0])["1.0"])


def test_signal_tail_averages_use_actual_length_for_short_runs(tmp_path):
    summary = {}
    experiments.exp_signals(tmp_path, ring_mixture(), w=5.0, k=0.1, steps=3, n=12, summary=summary)
    with (tmp_path / "e2_signals.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for name, result in summary["e2"].items():
        subset = [row for row in rows if row["method"] == name]
        expected = statistics.mean(float(row["switch_activity"]) for row in subset)
        assert result["switch_activity_last10"] == pytest.approx(expected)


def test_next_surface_sensitivity_includes_corrected_memory_even_when_w_zero(tmp_path):
    summary = {}
    experiments.exp_loop_gain(tmp_path, {"test": ring_mixture(k=3)},
                              w=0.0, k=0.1, steps=3, n=4, summary=summary)
    result = summary["e4"]["test"]
    assert result["next_error_sensitivity_max"] == 0.0
    assert result["surface_sensitivity_eig_min"] == experiments.LAM - 1
    assert result["surface_sensitivity_eig_max"] == experiments.LAM - 1
    with (tmp_path / "e4_loop_gain.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert all(float(row["sigma_next"]) > 0 for row in rows)
    assert any(float(row["correction_energy_change_mean"]) != 0 for row in rows)
    assert not any("assumption2" in key or "reaching_ok" in key for key in rows[0])


def test_next_surface_sensitivity_matches_whole_step_autograd(tmp_path):
    plant = ring_mixture(k=3)
    summary = {}
    w, k, steps, n = 2.0, 0.1, 3, 3
    experiments.exp_loop_gain(tmp_path, {"test": plant}, w, k, steps, n, summary)
    with (tmp_path / "e4_loop_gain.csv").open(newline="") as fh:
        row = next(csv.DictReader(fh))
    sigma_next = float(plant.sigma_schedule(steps)[1])
    h = sigma_next - 1
    x = torch.randn(n, plant.D, generator=torch.Generator().manual_seed(0)).double()
    e = plant.error(x, 1.0, 0)
    v_u = plant.velocity(x, 1.0)
    delta = -k * torch.sign(e)
    eigenvalues = []
    for b in range(n):
        def surface(correction):
            applied = e[b] + correction
            following_x = x[b] + h * (v_u[b] + w * applied)
            return plant.error(following_x[None], sigma_next, 0)[0] + (experiments.LAM - 1) * applied
        jacobian = torch.autograd.functional.jacobian(surface, delta[b])
        eigenvalues.extend(torch.linalg.eigvalsh((jacobian + jacobian.T) / 2).tolist())
    assert float(row["surface_sensitivity_eig_min"]) == pytest.approx(min(eigenvalues), abs=1e-8)
    assert float(row["surface_sensitivity_eig_max"]) == pytest.approx(max(eigenvalues), abs=1e-8)


@pytest.mark.parametrize("args", [
    ["--only", "e6"], ["--only"], ["--steps", "0"],
    ["--only", "e4", "--steps", "1"], ["--k", "nan"], ["--threads", "0"],
])
def test_cli_rejects_invalid_runs_before_creating_outputs(monkeypatch, tmp_path, args):
    out = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["toy_smc_cfg.py", "--out", str(out), *args])
    with pytest.raises(SystemExit) as exception:
        experiments.main()
    assert exception.value.code == 2
    assert not out.exists()
