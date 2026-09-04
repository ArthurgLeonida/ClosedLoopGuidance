"""Exercise runner checks without diffusers, downloads, or a GPU."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import real_model


def args(**overrides):
    values = dict(check_w=7.0, check_steps=3, model="dummy", dtype="fp32", device="cpu",
                  k=0.1, lam=6.0, w=[2.0], steps=3, prompts=None, seeds=[0], out="unused")
    values.update(overrides)
    return SimpleNamespace(**values)


class DummyImage:
    def __init__(self, value):
        self.array = np.array([[value]], dtype=np.float32)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.array, dtype=dtype)

    def save(self, path):
        path.write_bytes(b"dummy image")


class Pipeline:
    def __init__(self, sensitivities=(0.0, 1.0), active=True):
        self.transformer = torch.nn.Identity()
        self.sensitivities = sensitivities
        self.active = active

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.active

    @property
    def guidance_scale(self):
        return self._guidance_scale

    def __call__(self, prompt, guidance_scale, num_inference_steps, generator):
        self._guidance_scale = guidance_scale
        prompt_id = real_model.DEFAULT_PROMPTS.index(prompt) + 1
        for _ in range(num_inference_steps):
            prediction = torch.tensor(self.sensitivities)[:, None] * prompt_id
            result = self.transformer(prediction)
        return SimpleNamespace(images=[DummyImage(float(result.sum()))])


@pytest.mark.parametrize("w", [1.0, 0.0, float("nan"), float("inf")])
def test_unsupported_scales_fail_before_loading_model(monkeypatch, w):
    def unexpected_load(*unused):
        pytest.fail("validation must precede model loading")
    monkeypatch.setattr(real_model, "load_pipe", unexpected_load)
    with pytest.raises(ValueError, match="w > 1"):
        real_model.cmd_grid(args(w=[w]))
    with pytest.raises(ValueError, match="w > 1"):
        real_model.cmd_verify(args(check_w=w))


@pytest.mark.parametrize("sensitivities", [(0.0, 0.0), (0.75, 1.0), (1.0, 0.0), (0.0, float("nan"))])
def test_verify_fails_inconclusive_or_reversed_batch_order(monkeypatch, sensitivities):
    pipe = Pipeline(sensitivities)
    monkeypatch.setattr(real_model, "load_pipe", lambda *a: pipe)
    assert real_model.cmd_verify(args()) == 1
    assert "forward" not in pipe.transformer.__dict__


def test_verify_passes_for_a_working_doubled_pipeline(monkeypatch, capsys):
    pipe = Pipeline()
    monkeypatch.setattr(real_model, "load_pipe", lambda *a: pipe)
    assert real_model.cmd_verify(args()) == 0
    assert "chatter (last 3)" in capsys.readouterr().out
    assert "forward" not in pipe.transformer.__dict__


def test_grid_rejects_silently_inactive_controller(monkeypatch, tmp_path):
    pipe = Pipeline(active=False)
    monkeypatch.setattr(real_model, "load_pipe", lambda *a: pipe)
    monkeypatch.setattr(real_model, "DEFAULT_PROMPTS", ["prompt"])
    with pytest.raises(RuntimeError, match="controller was never called"):
        real_model.cmd_grid(args(out=str(tmp_path)))
    assert "forward" not in pipe.transformer.__dict__


def test_grid_writes_active_signals_and_resets_per_image(monkeypatch, tmp_path):
    pipe = Pipeline()
    monkeypatch.setattr(real_model, "load_pipe", lambda *a: pipe)
    monkeypatch.setattr(real_model, "DEFAULT_PROMPTS", ["prompt"])
    assert real_model.cmd_grid(args(out=str(tmp_path), seeds=[0, 1])) == 0
    import csv
    with (tmp_path / "signals.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 2 * 3  # two active arms, two seeds, three steps
    for arm in ("paper", "excess"):
        for seed in ("0", "1"):
            assert [row["step"] for row in rows if row["arm"] == arm and row["seed"] == seed] == ["0", "1", "2"]
    assert len(list(tmp_path.glob("*/*/*.png"))) == 6
    assert "forward" not in pipe.transformer.__dict__
