"""Prepare, generate, evaluate and report the CFG-Ctrl image benchmarks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

from .common import read_json, resolve, versions, write_json
from .config import DEFAULT_ARMS, METRICS, MODELS, bind_plan, cells, load_config, load_plan, samples
from .datasets import (download_compbench_prompts, download_genai, prepare_coco,
                       prepare_compbench, prepare_genai)


def template(models, benchmarks, base):
    def location(path):
        import os
        return os.path.relpath(Path(path).resolve(), base).replace("\\", "/")
    return dict(schema_version=1,
                models={name: dict(scales=[MODELS[name]["scale"]], steps=30, width=1024, height=1024,
                                   dtype="bf16", offload="model") for name in models},
                arms=DEFAULT_ARMS,
                benchmarks={name: dict(manifest=location(f"data/benchmarks/{name}.json"), seeds=[0],
                                       metrics=METRICS[name],
                                       **({"reference": location("data/reference/val2017")} if name == "coco" else {}),
                                       **({"models": ["sd35"]} if name == "genai" and "sd35" in models else {}))
                            for name in benchmarks},
                evaluation=dict(batch_size=16, workers=4, scorers={
                    "aesthetic": {"checkpoint": location("data/weights/sac+logos+ava1-l14-linearMSE.pth")},
                    "mps": {"repo": location("external/MPS"), "checkpoint": location("data/weights/MPS_overall_checkpoint.pth")},
                    "compbench": {"repo": location("external/T2I-CompBench")},
                }))


def show_plan(plan):
    total = 0
    for cell in cells(plan):
        n = sum(1 for _ in samples(plan, cell))
        print(f"{cell['path']:<55} {n:>7} images")
        total += n
    print(f"\nTotal: {total} images. {', '.join(plan['models'])}; {', '.join(plan['benchmarks'])}.")
    print("Generation and scoring settings are declared in the saved plan; paper-exact pair IDs are not assumed.")


def doctor(config_path, device):
    plan, base = load_config(config_path)
    import torch
    problems = []
    print(json.dumps(versions(), indent=2))
    if device.startswith("cuda") and not torch.cuda.is_available():
        problems.append("CUDA unavailable in this interpreter")
    from .adapters import TESTED_DIFFUSERS
    if versions().get("diffusers") != TESTED_DIFFUSERS:
        problems.append(f"generation requires diffusers=={TESTED_DIFFUSERS}")
    from .evaluation import metric_options
    packages = {"clip": "transformers", "pickscore": "transformers", "aesthetic": "clip",
                "imagereward": "ImageReward", "hpsv2": "hpsv2", "hpsv21": "hpsv2",
                "fid": "cleanfid", "vqascore": "t2v_metrics", "mps": "transformers"}
    needed = {m for bench in plan["benchmarks"].values() for m in bench["metrics"]}
    environments = {}
    for metric in sorted(needed):
        opts = metric_options(plan, metric, base)
        python = opts.get("python", sys.executable)
        interpreter = str(resolve(base, python)) if "/" in python or "\\" in python else python
        environments.setdefault(interpreter, []).append(metric)
        for field in ("repo", "checkpoint"):
            if field == "repo" or metric in ("mps", "aesthetic"):
                if field in opts and not Path(opts[field]).exists():
                    problems.append(f"{metric}: missing {field}: {opts[field]}")
        if metric in ("mps", "aesthetic") and "checkpoint" not in opts:
            problems.append(f"{metric}: configure its official checkpoint")
        if metric in ("mps", "compbench") and "repo" not in opts:
            problems.append(f"{metric}: configure its official source repository")
    for interpreter, metrics in environments.items():
        modules = {packages[m] for m in metrics if m in packages} | {"torch", "torchvision", "PIL"}
        if "compbench" in metrics:
            modules |= {"spacy", "detectron2"}
        if "mps" in metrics:
            modules.add("einops")
        check = ("import importlib.util, importlib.metadata as md, json\n"
                 f"missing = [m for m in {sorted(modules)!r} if importlib.util.find_spec(m) is None]\n"
                 "import torch, torchvision\n"
                 f"cuda = torch.cuda.is_available() if {device.startswith('cuda')!r} else True\n"
                 f"vqa = md.version('t2v-metrics') if {'vqascore' in metrics!r} else '1.1'\n"
                 "print(json.dumps(dict(missing=missing, cuda=cuda, vqa=vqa)))\n")
        try:
            result = subprocess.run([interpreter, "-c", check], capture_output=True, text=True, timeout=60)
            if result.returncode:
                problems.append(f"{metrics}: {interpreter}: {result.stderr.strip()}")
                continue
            info = json.loads(result.stdout.strip().splitlines()[-1])
            if info["missing"] or not info["cuda"] or info["vqa"] != "1.1":
                problems.append(f"{metrics}: {interpreter}: {info}")
            else:
                print(f"{', '.join(metrics)}: packages found; torch/torchvision load in {interpreter}")
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            problems.append(f"{metrics}: evaluator check failed: {exc}")
    for name, bench in plan["benchmarks"].items():
        if "fid" in bench["metrics"]:
            directory = resolve(base, bench["reference"])
            missing = [r["reference"] for r in bench["manifest"]["records"]
                       if not (directory / r["reference"]).is_file()]
            if missing:
                problems.append(f"{name}: {len(missing)} missing FID reference images")
    for problem in problems:
        print("MISSING:", problem)
    print("Doctor checks assets, package presence and torch/torchvision imports; model inference is not exercised.")
    return 1 if problems else 0


def diagnostics(out, plan, plan_hash, models=None, benchmarks=None):
    import csv
    import statistics
    from .generation import inspect_sample
    rows = []
    for cell in cells(plan, models, benchmarks):
        values = []
        for item in samples(plan, cell):
            meta = inspect_sample(out, cell, item, plan_hash)
            if meta is None:
                raise ValueError("diagnostics require complete generation for the selected cells")
            h = meta["controller"]
            if not h:
                continue
            values.append(dict(e_last=h[-1]["e_rms"], s_last=h[-1]["s_rms"],
                               delta_late=statistics.mean(r["delta_rms"] for r in h[-5:]),
                               chatter_late=statistics.mean(r["chatter"] for r in h[-5:]),
                               switch_late=statistics.mean(r["switch_activity"] for r in h[-5:]),
                               velocity_correction_late=statistics.mean(r["velocity_correction_rms"] for r in meta["signals"][-5:])))
        if values:
            row = {k: cell[k] for k in ("model", "benchmark", "arm", "w")}
            row.update(surface_reference="current_error" if cell["controller"]["mode"] == "proximal" else "sliding_surface",
                       **{k: statistics.mean(v[k] for v in values) for k in values[0]})
            rows.append(row)
    path = Path(out) / "diagnostics.csv"
    if rows:
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    else:
        path.unlink(missing_ok=True)
    print(f"{len(rows)} diagnostic rows written to {path}")


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="write a configuration you can edit")
    init.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    init.add_argument("--benchmarks", nargs="+", choices=METRICS, default=list(METRICS))
    init.add_argument("--config", type=Path, default=Path("configs/paper.json"))
    prep = sub.add_parser("prepare", help="freeze benchmark annotations; no image generation")
    ds = prep.add_subparsers(dest="benchmark", required=True)
    coco = ds.add_parser("coco")
    coco.add_argument("--annotations", required=True, type=Path)
    coco.add_argument("--pairs", type=Path)
    coco.add_argument("--count", type=int, default=5000)
    coco.add_argument("--seed", type=int, default=0)
    coco.add_argument("--out", type=Path, default=Path("data/benchmarks/coco.json"))
    comp = ds.add_parser("compbench")
    comp.add_argument("--repo", type=Path, default=Path("external/T2I-CompBench"))
    comp.add_argument("--download", action="store_true", help="download only the official prompt files")
    comp.add_argument("--out", type=Path, default=Path("data/benchmarks/compbench.json"))
    genai = ds.add_parser("genai")
    genai.add_argument("--prompts", type=Path)
    genai.add_argument("--skills", type=Path)
    genai.add_argument("--download", action="store_true")
    genai.add_argument("--revision", default="main")
    genai.add_argument("--out", type=Path, default=Path("data/benchmarks/genai.json"))
    asset = ds.add_parser("aesthetic")
    asset.add_argument("--out", type=Path, default=Path("data/weights/sac+logos+ava1-l14-linearMSE.pth"))
    for name in ("plan", "doctor", "generate", "run"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=Path("configs/paper.json"))
        if name != "doctor":
            p.add_argument("--out", type=Path, default=Path("results/paper"))
        if name != "plan":
            p.add_argument("--device", default="cuda")
        if name in ("generate", "run"):
            p.add_argument("--models", nargs="+")
            p.add_argument("--benchmarks", nargs="+")
        if name == "generate":
            p.add_argument("--gpus", nargs="+", type=int, help="one worker per visible CUDA index, e.g. 0 1 2 3")
    for name in ("evaluate", "report", "diagnostics"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--models", nargs="+")
        p.add_argument("--benchmarks", nargs="+")
        if name != "diagnostics":
            p.add_argument("--metrics", nargs="+")
            p.add_argument("--config", type=Path, help="refresh evaluator settings; generation settings must match")
        if name == "evaluate":
            p.add_argument("--device", default="cuda")
    work = sub.add_parser("worker", help="internal metric worker")
    work.add_argument("--task", type=Path, required=True)
    generation_work = sub.add_parser("generate-worker", help="internal coordinated generation worker")
    generation_work.add_argument("--task", type=Path, required=True)
    generation_work.add_argument("--rank", type=int, required=True)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "init":
            if args.config.exists():
                raise ValueError(f"{args.config} exists; edit it or choose another --config")
            write_json(args.config, template(args.models, args.benchmarks, args.config.resolve().parent))
            print(args.config)
        elif args.command == "prepare":
            if args.benchmark == "coco":
                prepare_coco(args.annotations, args.out, args.count, args.seed, args.pairs)
            elif args.benchmark == "compbench":
                repo = download_compbench_prompts(args.out.parent / "sources/compbench") if args.download else args.repo
                prepare_compbench(repo, args.out)
            elif args.benchmark == "genai":
                if args.download:
                    args.prompts, args.skills = download_genai(args.out.parent / "sources/genai", args.revision)
                if not args.prompts or not args.skills:
                    raise ValueError("provide --prompts and --skills, or --download")
                prepare_genai(args.prompts, args.skills, args.out)
            else:
                if args.out.exists():
                    raise ValueError(f"{args.out} already exists")
                args.out.parent.mkdir(parents=True, exist_ok=True)
                url = ("https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/"
                       "main/sac+logos+ava1-l14-linearMSE.pth")
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = response.read()
                args.out.write_bytes(payload)
            print(f"Prepared {args.out}")
        elif args.command == "doctor":
            return doctor(args.config, args.device)
        elif args.command == "worker":
            from .evaluation import worker
            worker(read_json(args.task))
        elif args.command == "generate-worker":
            from .parallel import generation_worker
            generation_worker(args.task, args.rank)
        elif args.command in ("plan", "generate", "run"):
            plan, base = load_config(args.config)
            if args.command == "plan":
                show_plan(plan)
                return 0  # A dry run creates no output tree.
            fingerprint = bind_plan(args.out, plan, base)
            if getattr(args, "gpus", None):
                if args.device != "cuda":
                    raise ValueError("--gpus selects devices; keep --device cuda")
                from .parallel import generate_parallel
                generate_parallel(args.out, plan, fingerprint, args.gpus, args.models, args.benchmarks)
            else:
                from .generation import generate
                generate(args.out, plan, fingerprint, args.device, args.models, args.benchmarks)
            if args.command == "run":
                from .evaluation import evaluate, report
                evaluate(args.out, plan, fingerprint, base, args.device, args.models, args.benchmarks)
                report(args.out, plan, fingerprint, base, args.models, args.benchmarks)
        else:
            plan, base, fingerprint = load_plan(args.out)
            if getattr(args, "config", None):
                plan, base = load_config(args.config)
                fingerprint = bind_plan(args.out, plan, base)
            if args.command == "diagnostics":
                diagnostics(args.out, plan, fingerprint, args.models, args.benchmarks)
            else:
                from .evaluation import evaluate, report
                if args.command == "evaluate":
                    evaluate(args.out, plan, fingerprint, base, args.device, args.models, args.benchmarks, args.metrics)
                else:
                    report(args.out, plan, fingerprint, base, args.models, args.benchmarks, args.metrics)
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
