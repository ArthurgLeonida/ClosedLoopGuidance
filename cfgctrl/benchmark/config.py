"""Resolve an explicit experiment matrix and freeze it for safe resumption."""
from dataclasses import asdict
import math
import os
from pathlib import Path

from cfgctrl.arms import parse_arm
from .common import digest, file_digest, identifier, lock, read_json, resolve, write_json
from .datasets import validate_manifest

MODELS = {
    "sd35": dict(checkpoint="stabilityai/stable-diffusion-3.5-large", scale=7.5, k=0.1),
    "flux": dict(checkpoint="black-forest-labs/FLUX.1-dev", scale=2.0, k=0.7),
    "qwen": dict(checkpoint="Qwen/Qwen-Image", scale=4.0, k=0.1),
}
PAPER_METRICS = ["fid", "clip", "aesthetic", "imagereward", "pickscore", "hpsv2", "hpsv21", "mps"]
METRICS = {"coco": PAPER_METRICS, "compbench": ["compbench"], "genai": ["vqascore"]}
DEFAULT_ARMS = ["conditional", "cfg", "paper", "excess", "proximal", "proximal_relative:k=0.1"]


def positive_int(value, label):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def load_config(path):
    path = Path(path).resolve()
    raw = read_json(path)
    if raw.get("schema_version") != 1:
        raise ValueError("configuration must have schema_version=1")
    unknown = set(raw) - {"schema_version", "models", "benchmarks", "arms", "evaluation"}
    if unknown:
        raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
    if not raw.get("models") or not raw.get("benchmarks"):
        raise ValueError("select at least one model and benchmark")
    models = {}
    for name, options in raw["models"].items():
        if name not in MODELS:
            raise ValueError(f"unknown model family {name!r}; choose {list(MODELS)}")
        defaults = MODELS[name]
        allowed = {"checkpoint", "revision", "scales", "steps", "width", "height",
                   "dtype", "offload", "embedded_guidance"}
        if set(options) - allowed:
            raise ValueError(f"unknown {name} model options: {sorted(set(options) - allowed)}")
        model = dict(checkpoint=defaults["checkpoint"], revision="main",
                     scales=[defaults["scale"]], steps=30, width=1024, height=1024,
                     dtype="bf16", offload="model", embedded_guidance=1.0)
        model.update(options)
        for dimension in ("steps", "width", "height"):
            positive_int(model[dimension], dimension)
        if model["width"] % 16 or model["height"] % 16:
            raise ValueError("width and height must be multiples of 16")
        if model["dtype"] not in ("bf16", "fp16", "fp32") or model["offload"] not in ("none", "model", "sequential"):
            raise ValueError("invalid dtype or offload setting")
        scales = model["scales"]
        if not scales or any(isinstance(w, bool) or not isinstance(w, (int, float))
                             or not math.isfinite(w) or w <= 1 for w in scales):
            raise ValueError("guided scales must be finite and >1; use the conditional arm for w=1")
        if len(set(scales)) != len(scales):
            raise ValueError("duplicate guidance scale")
        model["scales"] = [float(w) for w in scales]
        if model["embedded_guidance"] != 1.0:
            raise ValueError("paper comparison fixes embedded guidance to 1; true CFG has a separate scale")
        models[name] = model
    arms = raw.get("arms", DEFAULT_ARMS)
    if not arms:
        raise ValueError("select at least one arm")
    benchmarks = {}
    for name, spec in raw["benchmarks"].items():
        identifier(name)
        unknown = set(spec) - {"manifest", "seeds", "metrics", "reference", "models"}
        if unknown:
            raise ValueError(f"unknown {name} benchmark options: {sorted(unknown)}")
        manifest = validate_manifest(read_json(resolve(path.parent, spec["manifest"])))
        kind = manifest["benchmark"]
        seeds = spec.get("seeds", [0])
        if not seeds or any(type(s) is not int or s < 0 or s >= 2**63 for s in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique integers in [0, 2**63)")
        metrics = spec.get("metrics", METRICS[kind])
        if not metrics or len(set(metrics)) != len(metrics) or set(metrics) - set(METRICS[kind]):
            raise ValueError(f"invalid metrics for {kind}: {metrics}")
        if "fid" in metrics and not spec.get("reference"):
            raise ValueError("FID requires a real-image reference directory")
        model_names = spec.get("models", list(models))
        if not model_names or len(set(model_names)) != len(model_names) or set(model_names) - set(models):
            raise ValueError(f"{name} selects an unknown or duplicate model")
        benchmarks[name] = dict(kind=kind, manifest=manifest, seeds=seeds, metrics=metrics,
                                reference=spec.get("reference"), models=model_names)
    evaluation = raw.get("evaluation", {})
    if set(evaluation) - {"batch_size", "workers", "python", "scorers"}:
        raise ValueError("unknown evaluation options")
    positive_int(evaluation.get("batch_size", 16), "evaluation batch_size")
    if type(evaluation.get("workers", 4)) is not int or evaluation.get("workers", 4) < 0:
        raise ValueError("evaluation workers must be a nonnegative integer")
    if set(evaluation.get("scorers", {})) - set(sum(METRICS.values(), [])):
        raise ValueError("unknown metric in evaluation.scorers")
    plan = dict(schema_version=1, models=models, benchmarks=benchmarks, arms=arms,
                evaluation=evaluation,
                controller_defaults={name: dict(lam=6.0, k=MODELS[name]["k"]) for name in models})
    here = Path(__file__).resolve()
    plan["generation_implementation"] = {
        p.name: file_digest(p) for p in (here, here.with_name("adapters.py"), here.with_name("generation.py"),
                                        here.parents[1] / "controllers.py", here.parents[1] / "arms.py")}
    # Validate all arm specifications before loading any checkpoint.
    list(cells(plan))
    return plan, path.parent


def cells(plan, models=None, benchmarks=None):
    if models and set(models) - set(plan["models"]):
        raise ValueError("selected model is absent from the saved plan")
    if benchmarks and set(benchmarks) - set(plan["benchmarks"]):
        raise ValueError("selected benchmark is absent from the saved plan")
    for model_name, model in plan["models"].items():
        if models and model_name not in models:
            continue
        for bench_name, bench in plan["benchmarks"].items():
            if model_name not in bench["models"] or (benchmarks and bench_name not in benchmarks):
                continue
            names = set()
            for spec in plan["arms"]:
                if spec == "conditional":
                    name, cfg = "conditional", None
                    scales = [1.0]
                else:
                    name, cfg = parse_arm(spec, **plan["controller_defaults"][model_name])
                    identifier(name)
                    scales = model["scales"]
                if name in names or (name == "conditional" and spec != "conditional"):
                    raise ValueError(f"duplicate/reserved arm name: {name}")
                names.add(name)
                for w in scales:
                    yield dict(model=model_name, benchmark=bench_name, arm=name, w=w,
                               controller=asdict(cfg) if cfg else None,
                               path=f"{model_name}/{bench_name}/{name}/w{w}")


def samples(plan, cell):
    bench = plan["benchmarks"][cell["benchmark"]]
    for row in bench["manifest"]["records"]:
        for seed in bench["seeds"]:
            yield dict(id=f"{row['id']}_s{seed}", record=row, seed=seed)


def generation_fingerprint(plan):
    """Evaluator settings can change without invalidating expensive image generation."""
    identity = {k: v for k, v in plan.items() if k != "evaluation"}
    identity["benchmarks"] = {
        name: {k: v for k, v in bench.items() if k not in ("metrics", "reference")}
        for name, bench in plan["benchmarks"].items()}
    return digest(identity)


def bind_plan(out, plan, base):
    """Freeze generation; allow explicit evaluator updates with the same images."""
    out = Path(out)
    path = out / "plan.json"
    fingerprint = generation_fingerprint(plan)
    with lock(out, ".plan.lock"):
        if (out / ".generate.lock").exists() or (out / ".evaluate.lock").exists():
            raise ValueError("this output is in use; wait for its active process")
        if path.exists():
            _, _, previous_hash = load_plan(out)
            if previous_hash != fingerprint:
                raise ValueError(f"{out} belongs to a different experiment; choose a fresh --out")
        elif any(p.name != ".plan.lock" for p in out.iterdir()):
            raise ValueError(f"{out} is not empty and has no plan.json")
        write_json(path, dict(fingerprint=fingerprint, plan_checksum=digest(plan), plan=plan,
                             path_base=os.path.relpath(base, out.resolve())))
    return fingerprint


def load_plan(out):
    saved = read_json(Path(out) / "plan.json")
    if (digest(saved["plan"]) != saved["plan_checksum"] or
            generation_fingerprint(saved["plan"]) != saved["fingerprint"]):
        raise ValueError("saved plan was modified; use a fresh run")
    return saved["plan"], resolve(out, saved["path_base"]), saved["fingerprint"]
