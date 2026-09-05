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


# --------------------------------------------------------------------------
# Arm specifications: the modular part of the runner
# --------------------------------------------------------------------------

def test_bare_preset_names_itself_and_uses_the_run_defaults():
    name, cfg = real_model.parse_arm("paper", lam=6.0, k=0.1)
    assert (name, cfg.lam, cfg.k, cfg.switching) == ("paper", 6.0, 0.1, "sign")
    assert real_model.parse_arm("cfg", 6.0, 0.1)[1].is_cfg


@pytest.mark.parametrize("alias,canonical", [
    ("cfg_baseline", "cfg"), ("bl", "boundary_layer"),
    ("sat", "boundary_layer"), ("boundary_layer_excess", "excess"),
])
def test_aliases_resolve_to_the_canonical_name_and_config(alias, canonical):
    """An alias takes the canonical name, so `--arms bl boundary_layer` is
    caught as a duplicate rather than silently running one law twice into two
    directories. Pass `name=` when you do want two directories."""
    assert real_model.parse_arm(alias, 6.0, 0.1) == real_model.parse_arm(canonical, 6.0, 0.1)
    with pytest.raises(ValueError, match="both named"):
        real_model.parse_arms([alias, canonical], 6.0, 0.1)


def test_explicit_name_separates_the_output_directory_from_the_preset():
    name, cfg = real_model.parse_arm("flux=paper:k=0.7", lam=6.0, k=0.1)
    assert name == "flux" and cfg.k == 0.7 and cfg.lam == 6.0


def test_overriding_k_rederives_the_boundary_layer_width():
    """phi is derived from k*lam, so an overridden k must not leave a phi
    computed from the run default. This is the trap the parser exists for."""
    _, cfg = real_model.parse_arm("excess:k=0.3", lam=6.0, k=0.1)
    assert cfg.k == 0.3 and cfg.phi == pytest.approx(1.8)
    _, explicit = real_model.parse_arm("excess:k=0.3,phi=0.5", lam=6.0, k=0.1)
    assert explicit.phi == 0.5                      # an explicit phi still wins


def test_every_config_field_is_reachable_from_a_spec():
    _, cfg = real_model.parse_arm(
        "x=paper:lam=2,k=0.4,switching=sat,phi=0.9,"
        "store_corrected=false,relative_gain=yes,excess_only=1", lam=6.0, k=0.1)
    assert (cfg.lam, cfg.k, cfg.switching, cfg.phi) == (2.0, 0.4, "sat", 0.9)
    assert (cfg.store_corrected, cfg.relative_gain, cfg.excess_only) == (False, True, True)


@pytest.mark.parametrize("spec,message", [
    ("nosuch", "unknown preset"),
    ("paper:nosuch=1", "unknown controller field"),
    ("paper:k", "must be field=value"),
    ("paper:k=abc", "is not a number"),
    ("paper:switching=relay", "must be 'sign' or 'sat'"),
    ("paper:store_corrected=maybe", "must be true or false"),
    ("paper:k=1,k=2", "twice"),
    ("paper:k=-1", "must be finite and >= 0"),
    ("a/b=paper", "path separator"),
])
def test_bad_specs_are_rejected_with_a_useful_message(spec, message):
    with pytest.raises(ValueError, match=message):
        real_model.parse_arm(spec, 6.0, 0.1)


def test_duplicate_arm_names_are_rejected_before_they_overwrite_each_other():
    with pytest.raises(ValueError, match="both named 'paper'"):
        real_model.parse_arms(["paper", "paper:k=0.7"], 6.0, 0.1)
    table = real_model.parse_arms(["paper", "flux=paper:k=0.7"], 6.0, 0.1)
    assert list(table) == ["paper", "flux"]


def test_verify_requires_an_arm_that_actually_switches(monkeypatch):
    monkeypatch.setattr(real_model, "load_pipe",
                        lambda *a: pytest.fail("validation must precede model loading"))
    with pytest.raises(ValueError, match="every requested arm is plain CFG"):
        real_model.cmd_verify(args(arms=["cfg"]))


# --------------------------------------------------------------------------
# dry-run and resume
# --------------------------------------------------------------------------

def test_dry_run_reports_the_plan_without_loading_a_model(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(real_model, "load_pipe",
                        lambda *a: pytest.fail("a dry run must not load a model"))
    assert real_model.cmd_grid(args(out=str(tmp_path), dry_run=True,
                                    arms=["cfg", "flux=paper:k=0.7"])) == 0
    printed = capsys.readouterr().out
    assert "k=0.7" in printed and "flux" in printed
    assert not list(tmp_path.iterdir())             # nothing written at all


def test_config_json_records_resolved_settings_not_just_arm_names(monkeypatch, tmp_path):
    monkeypatch.setattr(real_model, "load_pipe", lambda *a: Pipeline())
    monkeypatch.setattr(real_model, "DEFAULT_PROMPTS", ["prompt"])
    assert real_model.cmd_grid(args(out=str(tmp_path), arms=["flux=paper:k=0.7"])) == 0
    import json
    recorded = json.loads((tmp_path / "config.json").read_text())["arms"]
    assert recorded["flux"]["k"] == 0.7 and recorded["flux"]["lam"] == 6.0


def test_resume_skips_existing_images_and_keeps_earlier_signals(monkeypatch, tmp_path):
    monkeypatch.setattr(real_model, "load_pipe", lambda *a: Pipeline())
    monkeypatch.setattr(real_model, "DEFAULT_PROMPTS", ["prompt"])
    shared = dict(out=str(tmp_path), seeds=[0], arms=["cfg", "paper"])
    assert real_model.cmd_grid(args(**shared)) == 0
    images = sorted(p.name for p in tmp_path.glob("*/*/*.png"))
    signals = (tmp_path / "signals.csv").read_text()

    calls = []
    original = real_model.generate
    monkeypatch.setattr(real_model, "generate",
                        lambda *a, **kw: (calls.append(1), original(*a, **kw))[1])
    assert real_model.cmd_grid(args(resume=True, **shared)) == 0
    assert calls == []                              # every image was already present
    assert sorted(p.name for p in tmp_path.glob("*/*/*.png")) == images
    assert (tmp_path / "signals.csv").read_text() == signals

    (tmp_path / "paper" / "w2.0" / "p00_s0.png").unlink()
    assert real_model.cmd_grid(args(resume=True, **shared)) == 0
    assert len(calls) == 1                          # only the deleted one regenerated


# --------------------------------------------------------------------------
# Preflight: fail before a multi-gigabyte download, not after
# --------------------------------------------------------------------------

def test_preflight_refuses_cuda_when_torch_cannot_use_it(monkeypatch):
    """A torch wheel built for a newer CUDA than the driver supports reports the
    driver as 'too old'. Catch that before downloading a checkpoint."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="would download the checkpoint and then fail"):
        real_model.preflight(args(device="cuda"))


def test_preflight_allows_cpu_and_working_cuda(monkeypatch, capsys):
    real_model.preflight(args(device="cpu"))            # never inspects the GPU
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "H100")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    real_model.preflight(args(device="cuda"))
    assert "H100" in capsys.readouterr().out


def test_grid_preflight_runs_before_the_model_is_loaded(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(real_model, "load_pipe",
                        lambda *a: pytest.fail("preflight must precede the download"))
    with pytest.raises(RuntimeError, match="torch cannot use CUDA"):
        real_model.cmd_grid(args(out=str(tmp_path), device="cuda"))


def test_dry_run_needs_no_gpu(monkeypatch, tmp_path):
    """Planning a matrix must work on a login node with no driver."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert real_model.cmd_grid(args(out=str(tmp_path), device="cuda", dry_run=True)) == 0


@pytest.mark.parametrize("exc,expected", [
    (type("GatedRepoError", (Exception,), {})("nope"), "accept the licence"),
    (Exception("401 Client Error"), "accept the licence"),
    (Exception("Access to model X is restricted"), "accept the licence"),
    (Exception("Repository Not Found"), "Check --model for a typo"),
])
def test_download_failures_are_translated_into_the_fix(exc, expected):
    explained = real_model._explain_load_failure(exc, "some/model")
    assert explained is not None and expected in str(explained)


def test_unrelated_load_failures_are_left_alone():
    assert real_model._explain_load_failure(OSError("disk full"), "some/model") is None
