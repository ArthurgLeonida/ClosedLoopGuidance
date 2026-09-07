"""Multi-GPU orchestration contracts; no real GPU or model download required."""
import os
from types import SimpleNamespace

import pytest

from cfgctrl.benchmark import config, generation, parallel
from cfgctrl.benchmark.common import read_json, write_json
from test_benchmark import Pipeline, experiment  # Shared small model/manifest fixture.


def test_shards_cover_each_sample_once_and_preserve_pairing(experiment):
    for world_size in (1, 2, 3, 8):
        for cell in config.cells(experiment.plan):
            expected = list(config.samples(experiment.plan, cell))
            parts = [list(generation.shard_samples(experiment.plan, cell, rank, world_size))
                     for rank in range(world_size)]
            flattened = [item["id"] for part in parts for item in part]
            assert sorted(flattened) == sorted(item["id"] for item in expected)
            assert len(flattened) == len(set(flattened))
    with pytest.raises(ValueError, match="shard"):
        generation.shard_samples(experiment.plan, cell, 2, 2)


def test_sharded_output_matches_serial_and_can_resume_with_another_gpu_count(experiment):
    calls = []
    loader = lambda *args: Pipeline("sd35", calls)
    generation.generate(experiment.out, experiment.plan, experiment.fingerprint, device="cpu",
                        loader=loader, rank=0, world_size=2)
    first_count = len(calls)
    assert first_count == 6
    # Simulate a restart on three GPUs: already committed samples move between
    # workers, but never regenerate, and metadata identity excludes GPU count.
    for rank in range(3):
        generation.generate(experiment.out, experiment.plan, experiment.fingerprint, device="cpu",
                            loader=loader, rank=rank, world_size=3)
    assert len(calls) == 12
    serial_out = experiment.out.parent / "serial"
    generation.generate(serial_out, experiment.plan, experiment.fingerprint, device="cpu", loader=loader)
    for cell in config.cells(experiment.plan):
        a = generation.image_records(experiment.out, experiment.plan, cell, experiment.fingerprint)
        b = generation.image_records(serial_out, experiment.plan, cell, experiment.fingerprint)
        assert [(r["id"], r["image_sha256"]) for r in a] == [(r["id"], r["image_sha256"]) for r in b]


def test_gpu_selection_respects_existing_visibility_and_rejects_duplicates(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3, GPU-test")
    assert parallel.gpu_devices([1, 0]) == ["GPU-test", "3"]
    with pytest.raises(ValueError, match="distinct"):
        parallel.gpu_devices([0, 0])
    with pytest.raises(ValueError, match="range"):
        parallel.gpu_devices([2])


@pytest.mark.parametrize("failure", [False, True])
def test_launcher_isolates_gpus_records_exit_status_and_cleans_terminated_rank_locks(experiment, monkeypatch, failure):
    monkeypatch.setattr(parallel, "gpu_devices", lambda _: ["3", "7"])
    processes, launches = [], []
    class Process:
        def __init__(self, rank):
            self.rank, self.pid = rank, 100 + rank
            self.returncode = (2 if failure else 0) if rank == 0 else (None if failure else 0)
            self.terminated = False
        def poll(self):
            return self.returncode
        def terminate(self):
            self.terminated, self.returncode = True, -15
        def wait(self, timeout=None):
            return self.returncode
    def popen(command, **kwargs):
        rank = int(command[-1])
        process = Process(rank)
        assert (experiment.out / ".generate.lock").exists()
        (experiment.out / ".jobs" / f".rank{rank}.lock").write_text(str(process.pid))
        launches.append((command, kwargs["env"]))
        processes.append(process)
        return process
    monkeypatch.setattr(parallel.subprocess, "Popen", popen)
    if failure:
        with pytest.raises(RuntimeError, match="failed"):
            parallel.generate_parallel(experiment.out, experiment.plan, experiment.fingerprint, [0, 1])
        assert processes[1].terminated
    else:
        parallel.generate_parallel(experiment.out, experiment.plan, experiment.fingerprint, [0, 1])
    assert [env["CUDA_VISIBLE_DEVICES"] for _, env in launches] == ["3", "7"]
    assert all("generate-worker" in command for command, _ in launches)
    status = read_json(experiment.out / "logs/generation_status.json")
    assert status["complete"] is not failure
    assert len(status["workers"]) == 2
    assert not (experiment.out / ".generate.lock").exists()
    assert not list((experiment.out / ".jobs").glob(".rank*.lock"))


def test_worker_requires_coordinator_and_uses_the_saved_shard(experiment, monkeypatch):
    from cfgctrl.benchmark.common import versions
    task = dict(parent_pid=os.getpid(), out=str(experiment.out), devices=["3", "7"],
                versions=versions(), plan_fingerprint=experiment.fingerprint, models=None, benchmarks=None)
    path = experiment.out / ".jobs/generation.json"
    write_json(path, task)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    (experiment.out / ".generate.lock").write_text(str(os.getpid()))
    received = []
    monkeypatch.setattr(generation, "generate", lambda *args, **kwargs: received.append((args, kwargs)))
    parallel.generation_worker(path, 1)
    assert received[0][1] == dict(rank=1, world_size=2, coordinated=True)
    assert received[0][0][3] == "cuda"
    assert not (experiment.out / ".jobs/.rank1.lock").exists()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with pytest.raises(ValueError, match="assignment"):
        parallel.generation_worker(path, 1)


def test_launch_failure_stops_already_started_workers(experiment, monkeypatch):
    monkeypatch.setattr(parallel, "gpu_devices", lambda _: ["0", "1"])
    process = SimpleNamespace(pid=123, returncode=None)
    process.poll = lambda: process.returncode
    process.terminate = lambda: setattr(process, "returncode", -15)
    process.wait = lambda timeout=None: process.returncode
    attempts = []
    def popen(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 2:
            raise OSError("could not start second worker")
        return process
    monkeypatch.setattr(parallel.subprocess, "Popen", popen)
    with pytest.raises(OSError, match="second worker"):
        parallel.generate_parallel(experiment.out, experiment.plan, experiment.fingerprint, [0, 1])
    assert process.returncode == -15
    assert not (experiment.out / ".generate.lock").exists()
