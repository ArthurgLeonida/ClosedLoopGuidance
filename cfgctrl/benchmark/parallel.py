"""One independent image-generation process per GPU, coordinated under one lock."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4

from .common import lock, read_json, versions, write_json
from .config import cells, load_plan


def gpu_devices(indices):
    import torch
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("--gpus requires distinct visible GPU indices")
    count = torch.cuda.device_count()
    if any(type(i) is not int or i < 0 or i >= count for i in indices):
        raise ValueError(f"GPU index out of range; this process sees {count} CUDA devices")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = [s.strip() for s in visible.split(",")] if visible is not None else None
    return [devices[i] if devices is not None else str(i) for i in indices]


def stop_workers(processes):
    # Send termination to everyone first; only then wait for GPU memory release.
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def generate_parallel(out, plan, plan_hash, gpus, models=None, benchmarks=None):
    devices = gpu_devices(gpus)
    if not list(cells(plan, models, benchmarks)):
        raise ValueError("no generation cells selected")
    out = Path(out).resolve()
    processes, streams, previous_handlers = [], [], {}
    job_id = uuid4().hex
    with lock(out, ".generate.lock"):
        task_path = out / ".jobs" / "generation.json"
        write_json(task_path, dict(id=job_id, parent_pid=os.getpid(), plan_fingerprint=plan_hash,
                                   out=str(out), devices=devices, models=models, benchmarks=benchmarks,
                                   versions=versions()))
        state = dict(id=job_id, started_at=time.time(), complete=False, devices=devices, workers=[])
        status_path = out / "logs" / "generation_status.json"
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"generation interrupted by signal {signum}")
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[sig] = signal.signal(sig, interrupted)
            for rank, device in enumerate(devices):
                log = out / "logs" / f"generation_rank{rank}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = log.open("a", encoding="utf-8")
                streams.append(stream)
                stream.write(f"\nGeneration job {job_id}; rank {rank}; CUDA_VISIBLE_DEVICES={device}\n")
                stream.flush()
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, PYTHONUNBUFFERED="1")
                command = [sys.executable, "-u", "-m", "cfgctrl.benchmark", "generate-worker",
                           "--task", str(task_path), "--rank", str(rank)]
                process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2], env=env,
                                           stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                processes.append(process)
                state["workers"].append(dict(rank=rank, device=device, pid=process.pid, log=str(log)))
                print(f"Started rank {rank}, GPU {device}, PID {process.pid}; log {log}", flush=True)
            write_json(status_path, state)
            previous_codes, last_update = None, 0.0
            while True:
                codes = [p.poll() for p in processes]
                if codes != previous_codes or time.monotonic() - last_update >= 30:
                    for row, code in zip(state["workers"], codes):
                        row["exit_code"] = code
                    state["updated_at"] = time.time()
                    write_json(status_path, state)
                    print(f"Generation workers: {sum(c is None for c in codes)} running, "
                          f"{codes.count(0)} complete; elapsed {time.time() - state['started_at']:.0f}s", flush=True)
                    previous_codes, last_update = codes, time.monotonic()
                failed = [i for i, code in enumerate(codes) if code not in (None, 0)]
                if failed:
                    raise RuntimeError(f"generation ranks {failed} failed; inspect their logs and rerun to resume")
                if all(code == 0 for code in codes):
                    state["complete"] = True
                    break
                time.sleep(1)
        except BaseException as exc:
            state["error"] = str(exc)
            raise
        finally:
            # Keep the output locked until every child has stopped, including
            # children already launched when a later process fails to start.
            for sig in previous_handlers:
                signal.signal(sig, signal.SIG_IGN)
            stop_workers(processes)
            for row, process in zip(state["workers"], processes):
                row["exit_code"] = process.returncode
                rank_lock = out / ".jobs" / f".rank{row['rank']}.lock"
                if rank_lock.exists() and rank_lock.read_text().strip() == str(process.pid):
                    rank_lock.unlink()  # A terminated child cannot release its own lock.
            state["finished_at"] = time.time()
            write_json(status_path, state)
            for stream in streams:
                stream.close()
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
    print("Multi-GPU generation complete.", flush=True)


def generation_worker(task_path, rank):
    task = read_json(task_path)
    out = Path(task["out"])
    if (out / ".generate.lock").read_text().strip() != str(task["parent_pid"]):
        raise ValueError("generation worker has no matching coordinator lock")
    if not 0 <= rank < len(task["devices"]):
        raise ValueError("invalid worker rank")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != task["devices"][rank]:
        raise ValueError("worker GPU assignment differs from its job")
    if versions() != task["versions"]:
        raise ValueError("worker packages differ from the coordinator")
    plan, _, fingerprint = load_plan(out)
    if fingerprint != task["plan_fingerprint"]:
        raise ValueError("worker plan changed after launch")
    from .generation import generate
    with lock(out / ".jobs", f".rank{rank}.lock"):
        generate(out, plan, fingerprint, "cuda", task["models"], task["benchmarks"],
                 rank=rank, world_size=len(task["devices"]), coordinated=True)
