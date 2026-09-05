"""Exercise the evaluation runner without CLIP, images, or a GPU."""

import json
from types import SimpleNamespace

import pytest

from experiments import evaluate


def args(**overrides):
    values = dict(run="unused", baseline="cfg", clip_model="dummy", device="cpu",
                  check_images=6, scorer=None)
    values.update(overrides)
    return SimpleNamespace(**values)


def make_run(tmp_path, arms=("cfg", "paper"), w=(3.0,), seeds=(0,),
             prompts=("a red cube", "a blue sphere", "a green cone"), drop=()):
    (tmp_path / "config.json").write_text(json.dumps(
        {"prompts": list(prompts), "arms": {a: {} for a in arms},
         "w": list(w), "seeds": list(seeds)}), encoding="utf-8")
    for arm in arms:
        for scale in w:
            d = tmp_path / arm / f"w{scale}"
            d.mkdir(parents=True, exist_ok=True)
            for pid in range(len(prompts)):
                for seed in seeds:
                    if (arm, scale, pid, seed) in drop:
                        continue
                    (d / f"p{pid:02d}_s{seed}.png").write_bytes(b"png")
    return tmp_path


def scorer_from(table):
    """Score by (image stem, prompt) so tests control every value."""
    def score(paths, prompts, model, device):
        return [table(_stem(p), t) for p, t in zip(paths, prompts)]
    return score


def _stem(p):
    return f"{p.parent.parent.name}/{p.parent.name}/{p.stem}"


# ------------------------------------------------------------------- layout

def test_missing_config_is_reported_not_crashed(tmp_path):
    with pytest.raises(ValueError, match="point --run at a grid output directory"):
        evaluate.load_run(tmp_path)


def test_index_reports_missing_images(tmp_path):
    run = make_run(tmp_path, drop={("paper", 3.0, 1, 0)})
    cfg = evaluate.load_run(run)
    found, missing = evaluate.image_index(run, cfg)
    assert len(found) == 5 and missing == [("paper", 3.0, 1, 0)]


def test_check_fails_when_images_are_missing(tmp_path, capsys):
    run = make_run(tmp_path, drop={("paper", 3.0, 1, 0)})
    aligned = scorer_from(lambda stem, prompt: 1.0 if prompt == "a red cube" else 0.0)
    assert evaluate.cmd_check(args(run=str(run), scorer=aligned)) == 1
    assert "missing" in capsys.readouterr().out


# --------------------------------------- the silent failure this guards against

def test_check_passes_when_each_image_matches_its_own_prompt(tmp_path, capsys):
    """Correct pairing: image pNN scores high only for prompt NN."""
    def correct(paths, prompts, model, device):
        out = []
        for p, prompt in zip(paths, prompts):
            pid = int(p.stem[1:3])
            out.append(1.0 if prompt == ["a red cube", "a blue sphere", "a green cone"][pid] else 0.1)
        return out
    run = make_run(tmp_path)
    assert evaluate.cmd_check(args(run=str(run), scorer=correct)) == 0
    assert "PASS" in capsys.readouterr().out


def test_check_fails_when_prompts_are_paired_with_the_wrong_images(tmp_path, capsys):
    """A shifted mapping still yields plausible-looking numbers, so the check
    must catch it: here every image prefers its NEIGHBOUR's prompt."""
    def shifted(paths, prompts, model, device):
        out = []
        for p, prompt in zip(paths, prompts):
            pid = int(p.stem[1:3])
            names = ["a red cube", "a blue sphere", "a green cone"]
            out.append(1.0 if prompt == names[(pid + 1) % 3] else 0.1)
        return out
    run = make_run(tmp_path)
    assert evaluate.cmd_check(args(run=str(run), scorer=shifted)) == 1
    assert "mapping looks wrong" in capsys.readouterr().out


def test_check_fails_on_nonfinite_scores(tmp_path, capsys):
    run = make_run(tmp_path)
    nan = scorer_from(lambda stem, prompt: float("nan"))
    assert evaluate.cmd_check(args(run=str(run), scorer=nan)) == 1
    assert "nonfinite" in capsys.readouterr().out


# --------------------------------------------------------------- statistics

def test_paired_delta_differences_the_same_prompt_and_seed(tmp_path):
    scores = {("cfg", 3.0, 0, 0): 1.0, ("paper", 3.0, 0, 0): 1.5,
              ("cfg", 3.0, 1, 0): 2.0, ("paper", 3.0, 1, 0): 2.25}
    delta = evaluate.paired_delta(scores, "paper", "cfg", 3.0, prompts=2, seeds=[0])
    assert delta == {0: pytest.approx(0.5), 1: pytest.approx(0.25)}


def test_paired_delta_skips_prompts_without_both_arms():
    scores = {("cfg", 3.0, 0, 0): 1.0, ("paper", 3.0, 0, 0): 1.5,
              ("cfg", 3.0, 1, 0): 2.0}          # paper missing for prompt 1
    assert list(evaluate.paired_delta(scores, "paper", "cfg", 3.0, 2, [0])) == [0]


def test_bootstrap_interval_brackets_the_mean_and_needs_two_points():
    lo, hi = evaluate.bootstrap_ci([0.4, 0.5, 0.6, 0.5, 0.45], resamples=2000)
    assert lo < evaluate.mean([0.4, 0.5, 0.6, 0.5, 0.45]) < hi
    assert all(map(lambda v: v != v, evaluate.bootstrap_ci([1.0])))   # NaN for n < 2


def test_clip_writes_per_image_and_summary_with_paired_deltas(tmp_path, capsys):
    """paper beats cfg by exactly 0.2 on every prompt, so the interval must
    exclude zero and the run must be starred."""
    def scorer(paths, prompts, model, device):
        return [0.5 + (0.2 if p.parent.parent.name == "paper" else 0.0) for p in paths]
    run = make_run(tmp_path, seeds=(0, 1))
    assert evaluate.cmd_clip(args(run=str(run), scorer=scorer)) == 0
    printed = capsys.readouterr().out
    assert "+0.2000" in printed and "*" in printed

    import csv
    with (run / "clip_scores.csv").open(newline="") as fh:
        assert len(list(csv.DictReader(fh))) == 2 * 3 * 2      # arms x prompts x seeds
    with (run / "clip_summary.csv").open(newline="") as fh:
        rows = {r["arm"]: r for r in csv.DictReader(fh)}
    assert float(rows["paper"]["paired_delta"]) == pytest.approx(0.2)
    assert rows["cfg"]["paired_delta"] == "nan"                # baseline vs itself


def test_clip_refuses_an_empty_run(tmp_path):
    run = make_run(tmp_path, drop={(a, 3.0, p, 0) for a in ("cfg", "paper") for p in range(3)})
    with pytest.raises(ValueError, match="no images to score"):
        evaluate.cmd_clip(args(run=str(run), scorer=lambda *a: []))
