"""Atomic per-image output, paired seeds, and verified resumability."""
from dataclasses import asdict
from contextlib import nullcontext
import gc
from pathlib import Path
import os
import time

from cfgctrl import SMCConfig, SlidingModeGuidance
from .adapters import generate_image, load_pipeline
from .common import digest, file_digest, lock, read_json, versions, write_json
from .config import cells, samples


def paths(out, cell, item):
    root = Path(out) / cell["path"]
    return root / "images" / (item["id"] + ".png"), root / "records" / (item["id"] + ".json")


def sample_fingerprint(plan_hash, cell, item):
    return digest(dict(plan=plan_hash, cell=cell, sample=item))


def inspect_sample(out, cell, item, plan_hash):
    from PIL import Image
    image, record = paths(out, cell, item)
    if not image.exists() or not record.exists():
        return None
    try:
        meta = read_json(record)
        if meta["fingerprint"] != sample_fingerprint(plan_hash, cell, item):
            raise ValueError(f"sample settings changed: {record}")
        if file_digest(image) != meta["image_sha256"]:
            raise ValueError(f"image changed after generation: {image}")
        with Image.open(image) as im:
            im.verify()
        return meta
    except (OSError, KeyError) as exc:
        raise ValueError(f"invalid generated sample {image}: {exc}") from exc


def image_records(out, plan, cell, plan_hash):
    result = []
    for item in samples(plan, cell):
        meta = inspect_sample(out, cell, item, plan_hash)
        if meta is None:
            raise ValueError(f"incomplete generation: {paths(out, cell, item)[0]}")
        result.append(dict(**item, path=str(paths(out, cell, item)[0].resolve()), image_sha256=meta["image_sha256"]))
    return result


def shard_samples(plan, cell, rank=0, world_size=1):
    if type(world_size) is not int or type(rank) is not int or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid generation shard")
    # Ownership does not depend on which files already exist. All arms retain
    # the same prompt/seed pairing, and a resumed job may change GPU count.
    return (item for index, item in enumerate(samples(plan, cell)) if index % world_size == rank)


def generate(out, plan, plan_hash, device="cuda", models=None, benchmarks=None, loader=None,
             rank=0, world_size=1, coordinated=False):
    loader = loader or load_pipeline
    matrix = list(cells(plan, models, benchmarks))
    pipe, loaded_family = None, None
    guard = nullcontext() if coordinated else lock(out, ".generate.lock")
    with guard:
        try:
            for cell in matrix:
                items = list(shard_samples(plan, cell, rank, world_size))
                pending = [item for item in items if inspect_sample(out, cell, item, plan_hash) is None]
                print(f"{cell['path']}: {len(items) - len(pending)}/{len(items)} complete", flush=True)
                if not pending:
                    continue
                current_versions = versions()
                environment_path = Path(out) / "generation_environment.json"
                if environment_path.exists():
                    previous_versions = read_json(environment_path)
                    if any(previous_versions.get(name) != current_versions.get(name) for name in
                           ("torch", "torchvision", "diffusers", "transformers", "accelerate")):
                        raise ValueError("generation environment changed during a partial run; restore its versions or use a fresh --out")
                family = cell["model"]
                settings = plan["models"][family]
                if family != loaded_family:
                    pipe = None
                    gc.collect()
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    pipe = loader(family, settings, device)
                    loaded_family = family
                    write_json(environment_path, current_versions)
                ctrl = SlidingModeGuidance(SMCConfig(**cell["controller"])) if cell["controller"] else None
                for index, item in enumerate(pending):
                    start = time.monotonic()
                    image, signals = generate_image(pipe, family, settings, ctrl, item["record"]["prompt"],
                                                    item["seed"], cell["w"])
                    if image.size != (settings["width"], settings["height"]):
                        raise RuntimeError(f"pipeline returned unexpected resolution: {image.size}")
                    destination, metadata = paths(out, cell, item)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix(".tmp")
                    try:
                        image.save(temporary, format="PNG")
                        os.replace(temporary, destination)
                    finally:
                        temporary.unlink(missing_ok=True)
                    # A crash between image and metadata commits regenerates this sample.
                    info = dict(fingerprint=sample_fingerprint(plan_hash, cell, item),
                                image_sha256=file_digest(destination), prompt=item["record"]["prompt"],
                                seed=item["seed"], versions=versions(), elapsed_seconds=time.monotonic() - start,
                                generation_worker=dict(rank=rank, world_size=world_size,
                                                       cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")),
                                scheduler=type(pipe.scheduler).__name__,
                                scheduler_config=dict(pipe.scheduler.config),
                                model_revision=getattr(getattr(pipe.transformer, "config", None), "_commit_hash", None),
                                signals=signals, controller=[asdict(s) for s in ctrl.history] if ctrl else [])
                    write_json(metadata, info)
                    print(f"  {index + 1}/{len(pending)} {item['id']} ({info['elapsed_seconds']:.1f}s)", flush=True)
        finally:
            del pipe
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
