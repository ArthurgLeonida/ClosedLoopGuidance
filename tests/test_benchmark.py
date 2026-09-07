"""Contract tests for the benchmark pipeline, without downloading model weights."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from cfgctrl import SlidingModeGuidance, presets
from cfgctrl.benchmark import adapters, config, datasets, evaluation, generation
from cfgctrl.benchmark.__main__ import main
from cfgctrl.benchmark.common import digest, file_digest, read_json, write_json
from cfgctrl.benchmark.scorers import paired_cosine


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_commit_hash="test")

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        return (0.1 * hidden_states + encoder_hidden_states,)


class Scheduler:
    def __init__(self):
        self.config = {"test": True}
        self.seen = []

    def step(self, model_output, timestep, sample, return_dict=False):
        self.seen.append(model_output.clone())
        return (sample - model_output / len(self.timesteps),)


class Pipeline:
    """Exercise all three real calling conventions, including Qwen norm rescaling."""
    def __init__(self, family, calls=None):
        self.family, self.calls = family, calls if calls is not None else []
        self.transformer, self.scheduler = Transformer(), Scheduler()

    def __call__(self, prompt, negative_prompt, width, height, num_inference_steps,
                 generator, guidance_scale=None, true_cfg_scale=None):
        self.calls.append(prompt)
        w = guidance_scale if self.family == "sd35" else true_cfg_scale
        self.scheduler.sigmas = torch.linspace(1, 0, num_inference_steps + 1)
        self.scheduler.timesteps = self.scheduler.sigmas[:-1] * 1000
        x = torch.randn(1, 2, generator=generator)
        for t in self.scheduler.timesteps:
            label = t[None] if self.family == "sd35" else t[None] / 1000
            if w == 1:
                v = self.transformer(hidden_states=x, timestep=label, encoder_hidden_states=torch.ones_like(x))[0]
            elif self.family == "sd35":
                u, c = self.transformer(hidden_states=x.repeat(2, 1), timestep=label.repeat(2),
                                        encoder_hidden_states=torch.tensor([[0.2, 0.2], [1.0, 1.0]]))[0].chunk(2)
                v = u + w * (c - u)
            else:
                c = self.transformer(hidden_states=x, timestep=label, encoder_hidden_states=torch.ones_like(x))[0]
                u = self.transformer(hidden_states=x, timestep=label, encoder_hidden_states=torch.full_like(x, 0.2))[0]
                v = u + w * (c - u)
                if self.family == "qwen":
                    v = v * (c.norm(dim=-1, keepdim=True) / v.norm(dim=-1, keepdim=True))
            x = self.scheduler.step(v, t, x, return_dict=False)[0]
        value = int(torch.sigmoid(x.mean()) * 255)
        return SimpleNamespace(images=[Image.new("RGB", (width, height), (value, value, value))])


@pytest.fixture
def experiment(tmp_path):
    refs = tmp_path / "references"
    refs.mkdir()
    rows = []
    for i in range(2):
        Image.new("RGB", (16, 16), (i * 80, 0, 0)).save(refs / f"{i}.png")
        rows.append(dict(id=f"coco_{i}", image_id=i, reference=f"{i}.png", prompt=f"object {i}"))
    datasets.save_manifest(tmp_path / "coco.json", "coco", rows, {"source": "test"})
    raw = dict(schema_version=1, models={"sd35": dict(steps=2, width=16, height=16, scales=[3.0])},
               arms=["cfg", "paper", "proximal"],
               benchmarks={"coco": dict(manifest="coco.json", reference="references", seeds=[0, 1], metrics=["clip"])})
    path = tmp_path / "config.json"
    write_json(path, raw)
    plan, base = config.load_config(path)
    out = tmp_path / "run"
    fingerprint = config.bind_plan(out, plan, base)
    return SimpleNamespace(path=path, plan=plan, base=base, out=out, fingerprint=fingerprint, raw=raw)


def generate_test(experiment, calls=None):
    generation.generate(experiment.out, experiment.plan, experiment.fingerprint, device="cpu",
                        loader=lambda family, settings, device: Pipeline(family, calls))


@pytest.mark.parametrize("family", ["sd35", "flux", "qwen"])
def test_adapter_delivers_the_same_control_law_for_all_model_calling_conventions(family):
    pipe = Pipeline(family)
    settings = dict(width=16, height=16, steps=2, embedded_guidance=1)
    ctrl = SlidingModeGuidance(presets.proximal_excess())
    _, signals = adapters.generate_image(pipe, family, settings, ctrl, "p", 12, 3)
    # Every coordinate has cond-uncond=0.8, so thresholding subtracts 0.2
    # from the velocity after multiplying the excess-only correction by w.
    assert [r["velocity_correction_rms"] for r in signals] == pytest.approx([0.2, 0.2], abs=1e-6)
    assert len(ctrl.history) == 2
    assert "forward" not in pipe.transformer.__dict__
    assert "step" not in pipe.scheduler.__dict__
    other = Pipeline("sd35")
    adapters.generate_image(other, "sd35", settings, SlidingModeGuidance(presets.proximal_excess()), "p", 12, 3)
    for actual, expected in zip(pipe.scheduler.seen, other.scheduler.seen):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("family", ["sd35", "flux", "qwen"])
def test_conditional_baseline_uses_one_prediction_per_step(family):
    pipe = Pipeline(family)
    settings = dict(width=16, height=16, steps=2, embedded_guidance=1)
    _, signals = adapters.generate_image(pipe, family, settings, None, "p", 0, 1)
    assert len(signals) == 2
    assert all(r["velocity_correction_rms"] == 0 for r in signals)


def test_adapter_fails_for_missing_unconditional_branch_and_restores_methods():
    pipe = Pipeline("flux")
    pipe.scheduler.sigmas = torch.tensor([1.0, 0.0])
    with pytest.raises(RuntimeError, match="missing conditional"):
        with adapters.SchedulerGuidance(pipe, "flux", SlidingModeGuidance(presets.paper()), 3, 1):
            pipe.transformer(hidden_states=torch.ones(1, 2), timestep=torch.tensor([1.0]),
                             encoder_hidden_states=torch.ones(1, 2))
            pipe.scheduler.step(torch.ones(1, 2), torch.tensor(1000.0), torch.ones(1, 2))
    assert "forward" not in pipe.transformer.__dict__ and "step" not in pipe.scheduler.__dict__


def test_generation_is_resumable_and_records_prompt_seed_and_checksums(experiment):
    calls = []
    generate_test(experiment, calls)
    assert len(calls) == 12
    generate_test(experiment, calls)
    assert len(calls) == 12
    cell = next(config.cells(experiment.plan))
    records = generation.image_records(experiment.out, experiment.plan, cell, experiment.fingerprint)
    assert len(records) == 4
    assert records[0]["record"]["prompt"] == "object 0"
    assert records[0]["image_sha256"] == file_digest(records[0]["path"])


def test_changed_image_is_not_silently_reused(experiment):
    generate_test(experiment)
    cell = next(config.cells(experiment.plan))
    item = next(config.samples(experiment.plan, cell))
    path, _ = generation.paths(experiment.out, cell, item)
    Image.new("RGB", (16, 16), "red").save(path)
    with pytest.raises(ValueError, match="image changed"):
        generate_test(experiment)


def test_changed_plan_cannot_mix_experiment_settings(experiment):
    changed = deepcopy(experiment.plan)
    changed["benchmarks"]["coco"]["manifest"]["records"][0]["prompt"] = "different"
    with pytest.raises(ValueError, match="different experiment"):
        config.bind_plan(experiment.out, changed, experiment.base)


@pytest.mark.parametrize("options", [{"scales": [1]}, {"scales": [3, 3]}, {"steps": 0}, {"width": 17}, {"typo": 1}])
def test_bad_generation_settings_fail_before_loading_models(experiment, options):
    experiment.raw["models"]["sd35"].update(options)
    write_json(experiment.path, experiment.raw)
    with pytest.raises(ValueError):
        config.load_config(experiment.path)


def test_paper_gains_are_model_specific_and_relative_threshold_is_explicit(experiment):
    experiment.raw["models"] = {name: {} for name in config.MODELS}
    experiment.raw["arms"] = ["paper", "proximal_relative:k=0.1", "conditional"]
    write_json(experiment.path, experiment.raw)
    plan, _ = config.load_config(experiment.path)
    matrix = list(config.cells(plan))
    assert {c["model"]: c["controller"]["k"] for c in matrix if c["arm"] == "paper"} == {
        "sd35": 0.1, "flux": 0.7, "qwen": 0.1}
    assert all(c["controller"]["k"] == 0.1 for c in matrix if c["arm"] == "proximal_relative")
    assert all(c["w"] == 1 and c["controller"] is None for c in matrix if c["arm"] == "conditional")


def test_plan_cli_is_a_dry_run(experiment, tmp_path):
    out = tmp_path / "unwritten"
    assert main(["plan", "--config", str(experiment.path), "--out", str(out)]) == 0
    assert not out.exists()


def test_coco_uses_one_caption_per_reference_and_validates_explicit_pair_ids(tmp_path):
    source = tmp_path / "captions.json"
    write_json(source, dict(images=[dict(id=i, file_name=f"{i}.jpg") for i in range(3)],
                            annotations=[dict(id=i * 10 + j, image_id=i, caption=f"caption {i}/{j}")
                                         for i in range(3) for j in range(2)]))
    a = datasets.prepare_coco(source, tmp_path / "a.json", count=2, seed=42)
    b = datasets.prepare_coco(source, tmp_path / "b.json", count=2, seed=42)
    assert a == b and len({r["image_id"] for r in a["records"]}) == 2
    pairs = tmp_path / "pairs.json"
    write_json(pairs, [dict(image_id=1, caption_id=0)])
    with pytest.raises(ValueError, match="does not belong"):
        datasets.prepare_coco(source, tmp_path / "bad.json", count=1, pairs=pairs)


def test_compbench_requires_official_validation_counts(tmp_path):
    for name in datasets.COMPBENCH_CATEGORIES:
        path = tmp_path / "examples/dataset" / f"{name}_val.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("a red dog and a blue cat\n")
    with pytest.raises(ValueError, match="expected 300"):
        datasets.prepare_compbench(tmp_path, tmp_path / "out.json")


def test_paired_metric_does_not_score_the_full_cross_prompt_matrix():
    image = torch.tensor([[2.0, 0], [0, 3.0]])
    text = torch.tensor([[1.0, 0], [0, -2.0]])
    assert paired_cosine(image, text).tolist() == [1.0, -1.0]


def test_compbench_uses_ids_even_if_evaluator_output_is_reordered(tmp_path):
    path = tmp_path / "scores.json"
    write_json(path, [{"question_id": 1, "answer": "0.8"}, {"question_id": 0, "answer": "0.2"}])
    records = [dict(id="a"), dict(id="b")]
    assert evaluation.parse_compbench_results(path, records) == [dict(id="a", score=0.2), dict(id="b", score=0.8)]
    write_json(path, [{"question_id": 0, "answer": "0.2"}])
    with pytest.raises(ValueError, match="omitted"):
        evaluation.parse_compbench_results(path, records)


def test_genai_aggregates_by_identity_and_official_skill_membership():
    records = [dict(id="b_s0", record=dict(id="b", tags=["advanced"])),
               dict(id="a_s0", record=dict(id="a", tags=["basic"])),
               dict(id="a_s1", record=dict(id="a", tags=["basic"])),
               dict(id="untagged_s0", record=dict(id="untagged", tags=[]))]
    scores = [dict(id="a_s1", score=0.9), dict(id="b_s0", score=0.2), dict(id="a_s0", score=0.7),
              dict(id="untagged_s0", score=1.0)]
    result = {r["group"]: r["value"] for r in evaluation.aggregate_scores("vqascore", scores, records)}
    assert result == pytest.approx(dict(basic=0.8, advanced=0.2, overall=0.5))
    with pytest.raises(ValueError, match="coverage"):
        evaluation.aggregate_scores("vqascore", scores[:-1], records)


def test_metric_worker_resumes_partial_batches_and_report_refuses_missing_metrics(experiment, monkeypatch):
    generate_test(experiment)
    jobs = evaluation.prepare_jobs(experiment.out, experiment.plan, experiment.fingerprint, experiment.base)
    tasks = jobs["clip"]
    calls = []
    class Scorer:
        def __call__(self, records):
            calls.extend(r["id"] for r in records)
            if len(calls) == 4:
                raise RuntimeError("interrupted")
            return [0.5] * len(records)
    monkeypatch.setattr("cfgctrl.benchmark.scorers.load_scorer", lambda *args: Scorer())
    job = dict(metric="clip", device="cpu", batch_size=2, workers=0, tasks=tasks)
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluation.worker(job)
    first = Path(tasks[0]["output"]).with_suffix(".partial.json")
    assert len(read_json(first)["scores"]) == 2
    monkeypatch.setattr("cfgctrl.benchmark.scorers.load_scorer", lambda *args: lambda records: [0.6] * len(records))
    evaluation.worker(job)
    result = read_json(tasks[0]["output"])
    assert [r["score"] for r in result["scores"]] == [0.5, 0.5, 0.6, 0.6]
    rows = evaluation.report(experiment.out, experiment.plan, experiment.fingerprint, experiment.base)
    assert len(rows) == 3 and read_json(experiment.out / "summary.json")["complete"]
    Path(tasks[0]["output"]).unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        evaluation.report(experiment.out, experiment.plan, experiment.fingerprint, experiment.base)
    assert not read_json(experiment.out / "summary.json")["complete"]


def test_loader_failure_preserves_original_error_and_releases_lock(experiment):
    def broken(*args):
        raise RuntimeError("checkpoint access failed")
    with pytest.raises(RuntimeError, match="checkpoint access failed"):
        generation.generate(experiment.out, experiment.plan, experiment.fingerprint, device="cpu", loader=broken)
    assert not (experiment.out / ".generate.lock").exists()
    assert not (experiment.out / "generation_environment.json").exists()


def test_image_without_committed_metadata_is_regenerated(experiment):
    generate_test(experiment)
    cell = next(config.cells(experiment.plan))
    item = next(config.samples(experiment.plan, cell))
    _, metadata = generation.paths(experiment.out, cell, item)
    metadata.unlink()
    calls = []
    generate_test(experiment, calls)
    assert calls == [item["record"]["prompt"]]
    assert metadata.is_file()


def test_changed_runtime_cannot_mix_new_images_into_a_partial_run(experiment, monkeypatch):
    generate_test(experiment)
    cell = next(config.cells(experiment.plan))
    item = next(config.samples(experiment.plan, cell))
    generation.paths(experiment.out, cell, item)[1].unlink()
    monkeypatch.setattr(generation, "versions", lambda: {"torch": "different"})
    with pytest.raises(ValueError, match="environment changed"):
        generate_test(experiment)


def test_evaluator_configuration_can_change_without_regenerating_images(experiment):
    generate_test(experiment)
    jobs = evaluation.prepare_jobs(experiment.out, experiment.plan, experiment.fingerprint, experiment.base)
    changed = deepcopy(experiment.plan)
    changed["evaluation"] = {"python": "metric-python"}
    new_hash = config.bind_plan(experiment.out, changed, experiment.base)
    assert new_hash == experiment.fingerprint
    loaded, base, fingerprint = config.load_plan(experiment.out)
    assert loaded == changed and base == experiment.base and fingerprint == new_hash
    assert evaluation.metric_options(loaded, "clip", base)["python"] == "metric-python"
    refreshed = evaluation.prepare_jobs(experiment.out, loaded, new_hash, base)
    assert refreshed["clip"][0]["input_fingerprint"] == jobs["clip"][0]["input_fingerprint"]
    # Scoring options affect score identity while generation identity stays fixed.
    changed["evaluation"]["scorers"] = {"clip": {"checkpoint": "different/clip"}}
    assert config.bind_plan(experiment.out, changed, base) == new_hash
    refreshed = evaluation.prepare_jobs(experiment.out, changed, new_hash, base)
    assert refreshed["clip"][0]["input_fingerprint"] != jobs["clip"][0]["input_fingerprint"]
    calls = []
    generation.generate(experiment.out, changed, new_hash, device="cpu",
                        loader=lambda *args: Pipeline("sd35", calls))
    assert calls == []


def test_plan_manual_edit_is_detected(experiment):
    path = experiment.out / "plan.json"
    saved = read_json(path)
    saved["plan"]["models"]["sd35"]["steps"] += 1
    write_json(path, saved)
    with pytest.raises(ValueError, match="modified"):
        config.load_plan(experiment.out)


def test_fid_stages_only_frozen_references_and_reuses_stats(experiment, monkeypatch):
    import sys
    generate_test(experiment)
    cell = next(config.cells(experiment.plan))
    records = generation.image_records(experiment.out, experiment.plan, cell, experiment.fingerprint)
    refs = evaluation.reference_records(experiment.plan["benchmarks"]["coco"], experiment.base)
    Image.new("RGB", (16, 16), "blue").save(experiment.base / "references/unrelated.png")
    task = dict(reference=refs, records=records, cache=str(experiment.out / ".cache"), input_fingerprint="test")
    names, made, used = set(), [], []
    def make(name, directory, **kwargs):
        names.add(name)
        made.append((set(p.name for p in Path(directory).iterdir()), kwargs))
    def compute(directory, **kwargs):
        used.append((len(list(Path(directory).iterdir())), kwargs))
        return 12.345
    monkeypatch.setitem(sys.modules, "cleanfid", SimpleNamespace(fid=SimpleNamespace(
        test_stats_exists=lambda name, **kwargs: name in names, make_custom_stats=make, compute_fid=compute)))
    for _ in range(2):
        result = evaluation.score_fid(task, "cpu", 2, 0)
        assert result == {"value": 12.345, "n": 4, "n_reference": 2}
    assert len(made) == 1
    assert made[0][0] == {"0.png", "1.png"}
    assert made[0][1]["mode"] == "clean"
    assert all(n == 4 and opts["dataset_split"] == "custom" for n, opts in used)
    with pytest.raises(ValueError, match="counts"):
        evaluation.validate_fid(dict(value=0.0, n=3, n_reference=2), task)


def test_mps_uses_official_paired_interface_and_raw_scaled_cosine(tmp_path):
    from cfgctrl.benchmark.scorers import MPS, MPS_CONDITION
    path = tmp_path / "image.png"
    Image.new("RGB", (16, 16)).save(path)
    scorer = MPS.__new__(MPS)
    scorer.device = "cpu"
    scorer.processor = lambda *args, **kwargs: {"pixel_values": torch.ones(1, 3, 2, 2)}
    texts = []
    scorer.tokenize = lambda text: texts.append(text) or torch.ones(1, 3, dtype=torch.int64)
    class Paired:
        logit_scale = torch.tensor(10.0).log()
        def __call__(self, text, images, condition):
            assert images.shape[0] == 2 and torch.equal(images[0], images[1])
            return torch.tensor([[1.0, 0.0]]), torch.tensor([[0.6, 0.8]]), torch.tensor([[0.6, 0.8]])
    scorer.model = Paired()
    assert scorer([dict(path=str(path), record={"prompt": "test prompt"})]) == pytest.approx([6.0])
    assert texts == ["test prompt", MPS_CONDITION]


def test_vqascore_requires_one_score_per_prompt_image_pair():
    from cfgctrl.benchmark.scorers import VQAScore
    scorer = VQAScore.__new__(VQAScore)
    received = []
    def batch_forward(dataset, batch_size):
        received.extend(dataset)
        assert batch_size == 2
        return torch.tensor([[[0.4]], [[0.8]]])
    scorer.model = SimpleNamespace(batch_forward=batch_forward)
    records = [dict(path="one.png", record={"prompt": "first"}), dict(path="two.png", record={"prompt": "second"})]
    assert scorer(records) == pytest.approx([0.4, 0.8])
    assert received[1] == {"images": ["two.png"], "texts": ["second"]}
    scorer.model.batch_forward = lambda **kwargs: torch.ones(2, 2)
    with pytest.raises(RuntimeError, match="shape"):
        scorer(records)


def test_doctor_checks_packages_inside_the_selected_metric_interpreter(experiment, monkeypatch, capsys):
    import subprocess
    from cfgctrl.benchmark import __main__ as cli
    experiment.raw["evaluation"] = {"python": "metric-python"}
    write_json(experiment.path, experiment.raw)
    monkeypatch.setattr(cli, "versions", lambda: {"diffusers": adapters.TESTED_DIFFUSERS})
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(dict(missing=["transformers"], cuda=True, vqa="1.1")), "")
    monkeypatch.setattr(cli.subprocess, "run", run)
    assert cli.doctor(experiment.path, "cpu") == 1
    assert commands[0][0] == "metric-python"
    assert "find_spec" in commands[0][2]
    assert "transformers" in capsys.readouterr().out


def test_delivered_low_precision_correction_is_reported_after_cast():
    pipe = Pipeline("sd35")
    pipe.scheduler.sigmas = torch.tensor([1.0, 0.0])
    pipe.scheduler.timesteps = torch.tensor([1000.0])
    ctrl = SlidingModeGuidance(presets.proximal_excess(k=0.0001))
    x = torch.ones(1, 2, dtype=torch.bfloat16)
    with adapters.SchedulerGuidance(pipe, "sd35", ctrl, 3, 1) as adapter:
        raw = pipe.transformer(hidden_states=x.repeat(2, 1), timestep=torch.tensor([1000.0, 1000.0]),
                               encoder_hidden_states=torch.tensor([[0.2, 0.2], [1.0, 1.0]], dtype=torch.bfloat16))[0]
        u, c = raw.chunk(2)
        native = u + 3 * (c - u)
        baseline = (u.float() + 3 * (c.float() - u.float())).to(native.dtype)
        pipe.scheduler.step(native, torch.tensor(1000.0), x)
    actual = (pipe.scheduler.seen[0].float() - baseline.float()).square().mean().sqrt().item()
    assert adapter.signals[0]["velocity_correction_rms"] == actual
    assert actual == 0.0
