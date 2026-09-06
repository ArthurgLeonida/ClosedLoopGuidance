"""Summarising controller diagnostics, without a GPU or a real run."""

import csv
import json

import pytest

from experiments import signals


def make_run(tmp_path, arms=None, scales=(3.0,), prompts=4, steps=6,
             plateau=0.5, chatter=0.99):
    arms = arms or {"paper": {"lam": 6.0, "k": 0.1, "store_corrected": True}}
    (tmp_path / "config.json").write_text(json.dumps(
        {"arms": arms, "steps": steps, "w": list(scales), "seeds": [0],
         "prompts": ["p"] * prompts}), encoding="utf-8")
    with (tmp_path / "signals.csv").open("w", newline="", encoding="utf-8") as fh:
        wri = csv.writer(fh)
        wri.writerow(["arm", "w", "prompt_id", "seed", "step", "e_rms", "s_rms",
                      "delta_rms", "chatter", "switch_activity", "deriv_matters", "k_eff"])
        for arm in arms:
            for w in scales:
                for pid in range(prompts):
                    for step in range(steps):
                        frac = step / (steps - 1)
                        wri.writerow([arm, w, pid, 0, step,
                                      0.1 * (1 - frac) + 0.01 * frac,
                                      0.6 * (1 - frac) + plateau * frac,
                                      0.1, chatter if step > 1 else 0.0,
                                      0.2 * chatter, 0.019, 0.1])
    return tmp_path


def test_reduces_the_file_to_one_row_per_arm_and_scale(tmp_path):
    res = signals.summarise(make_run(tmp_path, scales=(1.5, 3.0), prompts=5, steps=6))
    assert res["rows"] == 1 * 2 * 5 * 6
    assert sorted(res["final_s"]) == [("paper", 1.5), ("paper", 3.0)]
    assert res["final_s"][("paper", 3.0)].n == 5          # one value per prompt
    assert res["final_s"][("paper", 3.0)].mean == pytest.approx(0.5)


def test_running_statistics_match_the_direct_computation():
    r = signals.Running()
    for v in (1.0, 2.0, 3.0, 4.0):
        r.add(v)
    assert r.mean == pytest.approx(2.5)
    assert r.sd == pytest.approx(1.2909944, rel=1e-6)
    r.add(float("nan"))                                   # ignored, not poisoned
    assert r.n == 4 and r.mean == pytest.approx(2.5)


def test_plateau_predicted_only_for_the_supported_alternating_regime():
    assert signals.predicted_plateau({"lam": 6.0, "k": 0.1, "store_corrected": True}) \
        == pytest.approx(0.5)
    # measured-error memory: the surface tracks e, there is no plateau
    assert signals.predicted_plateau({"lam": 6.0, "k": 0.1, "store_corrected": False}) is None
    assert signals.predicted_plateau({"lam": 6.0, "k": 0.0, "store_corrected": True}) is None
    assert signals.predicted_plateau({"lam": 0.5, "k": 0.1, "store_corrected": True}) is None


def test_reports_the_ratio_against_the_prediction(tmp_path, capsys):
    run = make_run(tmp_path, plateau=0.5)                 # exactly (lam-1)*k
    res = signals.summarise(run)
    signals.report(res, tail=5)
    out = capsys.readouterr().out
    assert "0.500" in out and "1.00" in out


def test_measured_memory_arm_is_labelled_rather_than_scored(tmp_path, capsys):
    run = make_run(tmp_path, arms={"excess": {"lam": 6.0, "k": 0.1,
                                              "store_corrected": False}}, plateau=0.05)
    signals.report(signals.summarise(run), tail=5)
    out = capsys.readouterr().out
    assert "tracks e" in out


def test_summary_csv_has_one_row_per_arm_and_scale(tmp_path):
    run = make_run(tmp_path, scales=(1.5, 3.0))
    res = signals.summarise(run)
    with signals.write_csv(run, res).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert {r["w"] for r in rows} == {"1.5", "3.0"}
    assert float(rows[0]["ratio"]) == pytest.approx(1.0, abs=0.01)


def test_step_count_mismatch_is_warned_about(tmp_path, capsys):
    run = make_run(tmp_path, steps=6)
    cfg = json.loads((run / "config.json").read_text())
    cfg["steps"] = 30                                     # claims more than the data has
    (run / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    signals.summarise(run)
    assert "WARNING" in capsys.readouterr().out


def test_missing_or_empty_inputs_are_reported(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"arms": {}, "steps": 3}), encoding="utf-8")
    with pytest.raises(ValueError, match="signals.csv not found"):
        signals.summarise(tmp_path)
    (tmp_path / "signals.csv").write_text("arm,w,step\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no usable rows"):
        signals.summarise(tmp_path)


def test_chatter_and_switch_use_the_same_window(tmp_path):
    """The per-sample/step square-root identity needs matching windows.
    With a constant chatter fraction within the window it also holds for means.
    """
    import csv as _csv
    steps, tail = 10, 5
    (tmp_path / "config.json").write_text(json.dumps(
        {"arms": {"paper": {"lam": 6.0, "k": 0.1, "store_corrected": True}},
         "steps": steps, "w": [3.0], "seeds": [0], "prompts": ["p"]}), encoding="utf-8")
    with (tmp_path / "signals.csv").open("w", newline="", encoding="utf-8") as fh:
        wri = _csv.writer(fh)
        wri.writerow(["arm", "w", "prompt_id", "seed", "step", "e_rms", "s_rms",
                      "delta_rms", "chatter", "switch_activity", "deriv_matters", "k_eff"])
        for step in range(steps):
            # early steps quiet, tail steps fully switching
            late = step > steps - 1 - tail
            wri.writerow(["paper", 3.0, 0, 0, step, 0.05, 0.5, 0.1,
                          1.0 if late else 0.0, 0.2 if late else 0.0, 0.02, 0.1])
    res = signals.summarise(tmp_path, tail=tail)
    key = ("paper", 3.0)
    assert res["late_chat"][key].mean == pytest.approx(1.0)
    # 0.2 / (2*0.1) == 1.0; averaging all ten steps would have given 0.5
    assert res["switch"][key].mean / (2 * 0.1) == pytest.approx(1.0)
    assert res["switch"][key].n == res["late_chat"][key].n


@pytest.mark.parametrize("override", [{"switching": "sat", "phi": 0.6},
                                     {"relative_gain": True}])
def test_no_fixed_sign_plateau_for_smooth_or_relative_variants(override):
    cfg = {"lam": 6.0, "k": 0.1, "store_corrected": True, **override}
    assert signals.predicted_plateau(cfg, w=3.0) is None


def test_excess_sign_plateau_and_switch_norm_use_applied_gain(tmp_path):
    cfg = {"lam": 6.0, "k": 0.1, "store_corrected": True, "excess_only": True}
    assert signals.predicted_plateau(cfg) is None  # cannot infer an unknown w
    assert signals.predicted_plateau(cfg, w=2.0) == pytest.approx(0.25)
    run = make_run(tmp_path, arms={"excess_sign": cfg}, scales=(2.0,), plateau=0.25)
    res = signals.summarise(run)
    with signals.write_csv(run, res).open(newline="") as fh:
        row = next(csv.DictReader(fh))
    assert float(row["ratio"]) == pytest.approx(1.0)
    # Retain the original nominal-2k CSV field; expose applied normalization too.
    assert float(row["switch_activity_applied_norm"]) == pytest.approx(
        2 * float(row["switch_activity_norm"]), abs=2e-6)


def controller_run(tmp_path, sequences, reverse_rows=False):
    """Write real controller diagnostics for prescribed measured-error sequences."""
    from dataclasses import asdict

    import torch
    from cfgctrl import SlidingModeGuidance, presets

    cfg = presets.boundary_layer_excess()
    steps = len(sequences[0])
    (tmp_path / "config.json").write_text(json.dumps(
        {"arms": {"excess": asdict(cfg)}, "steps": steps}), encoding="utf-8")
    rows = []
    for pid, sequence in enumerate(sequences):
        ctrl = SlidingModeGuidance(cfg)
        for e in sequence:
            ctrl.correct(torch.tensor([e], dtype=torch.float64), w=1.5)
            rows.append(dict(arm="excess", w=1.5, prompt_id=pid, seed=0,
                             **asdict(ctrl.history[-1])))
    if reverse_rows:
        rows.reverse()
    with (tmp_path / "signals.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


def test_final_surface_uses_previous_error_not_only_small_current_error(tmp_path, capsys):
    # Illustrative scalar example matching the pasted excess w=1.5 surface.
    run = controller_run(tmp_path, [[[0.1851], [0.0178], [0.0103]]])
    res = signals.summarise(run)
    b = res["measured_memory"][("excess", 1.5)]
    assert b["e_prev"].mean == pytest.approx(0.0178)
    assert b["s"].mean == pytest.approx(0.0993)
    assert b["lower"].mean == pytest.approx(0.0787)
    assert b["upper"].mean == pytest.approx(0.0993)
    signals.report(res, tail=5)
    text = capsys.readouterr().out
    assert "before its scheduler update" in text
    assert "not s_n = lam*e_n" in text
    with signals.write_csv(run, res).open(newline="") as fh:
        row = next(csv.DictReader(fh))
    assert int(row["n_paired"]) == 1
    assert float(row["e_rms_prev_final"]) == pytest.approx(0.0178)


def test_previous_errors_pair_by_prompt_even_with_reversed_csv(tmp_path):
    run = controller_run(tmp_path, [
        [[0.2, 0.0], [0.02, 0.0], [0.0, 0.01]],
        [[0.3, 0.0], [0.04, 0.0], [-0.01, 0.0]],
    ], reverse_rows=True)
    b = signals.summarise(run)["measured_memory"][("excess", 1.5)]
    assert b["e_prev"].n == 2
    assert b["e_prev"].mean == pytest.approx(0.03 / 2**0.5)
    assert b["lower"].mean <= b["s"].mean <= b["upper"].mean


def test_incomplete_endpoint_pairs_do_not_mix_trajectories(tmp_path):
    run = controller_run(tmp_path, [[[0.2], [0.02], [0.01]],
                                    [[0.3], [0.04], [0.03]]])
    with (run / "signals.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)
    rows = [r for r in rows if not (r["prompt_id"] == "1" and r["step"] == "1")]
    with (run / "signals.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    res = signals.summarise(run)
    assert res["final_s"][("excess", 1.5)].n == 2
    assert res["measured_memory"][("excess", 1.5)]["s"].n == 1


def test_duplicate_final_rows_are_not_silently_double_counted(tmp_path):
    run = make_run(tmp_path)
    lines = (run / "signals.csv").read_text().splitlines()
    with (run / "signals.csv").open("a") as fh:
        fh.write(lines[-1] + "\n")
    with pytest.raises(ValueError, match="duplicate endpoint"):
        signals.summarise(run)


@pytest.mark.parametrize("tail", [0, -1])
def test_nonpositive_tail_is_rejected(tmp_path, tail):
    with pytest.raises(ValueError, match="tail must be positive"):
        signals.summarise(make_run(tmp_path), tail=tail)


def test_plot_accepts_measured_memory_context(tmp_path):
    pytest.importorskip("matplotlib")
    run = controller_run(tmp_path, [[[0.1851], [0.0178], [0.0103]]])
    assert signals.plot(run, signals.summarise(run)).is_file()
