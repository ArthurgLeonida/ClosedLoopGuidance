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


def test_plateau_predicted_only_where_the_recurrence_has_that_fixed_point():
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
    """They describe one phenomenon. Averaging them over different windows made
    a sign law report chatter 0.99 alongside switch 0.86, which is impossible:
    for bang-bang switching, switch = sqrt(chatter)."""
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
