"""Isolated metric workers, complete coverage checks, and native-unit summaries."""
from collections import defaultdict
import gc
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from .common import digest, file_digest, lock, read_json, resolve, versions, write_json
from .config import cells
from .datasets import COMPBENCH_CATEGORIES, COMPBENCH_REVISION
from .generation import image_records

METRIC_PROTOCOLS = {
    "clip": "L2-normalized image/text cosine; OpenAI ViT-L/14",
    "pickscore": "L2-normalized image/text cosine; PickScore_v1; no softmax or logit scale",
    "aesthetic": "improved-aesthetic-predictor sac+logos+ava1-l14-linearMSE on OpenAI ViT-L/14",
    "imagereward": "ImageReward-v1.0 official normalized reward",
    "hpsv2": "HPS_v2_compressed; paired cosine; no x100",
    "hpsv21": "HPS_v2.1_compressed; paired cosine; no x100",
    "mps": "official MPS overall checkpoint; overall condition; scaled cosine logit",
    "fid": "clean-fid Inception-v3 2048; clean mode; exact manifest reference identities",
    "compbench": "official BLIP-VQA attributes and UniDet 2D spatial; validation prompts",
    "vqascore": "t2v-metrics 1.1 CLIP-FlanT5-XXL; default question/answer; official skill groups",
}


def link_or_copy(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_digest(source) != file_digest(target):
            raise ValueError(f"staged image changed: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def reference_records(bench, base):
    directory = resolve(base, bench["reference"])
    result = []
    for row in bench["manifest"]["records"]:
        path = directory / row["reference"]
        if not path.is_file():
            raise ValueError(f"missing FID reference: {path}")
        result.append(dict(name=row["reference"], path=str(path), sha256=file_digest(path)))
    if len(result) < 2:
        raise ValueError("FID needs at least two distinct reference images")
    return result


def metric_options(plan, name, base):
    options = dict(plan.get("evaluation", {}).get("scorers", {}).get(name, {}))
    if plan.get("evaluation", {}).get("python"):
        options.setdefault("python", plan["evaluation"]["python"])
    for field in ("repo", "checkpoint"):
        if field in options and (name in ("aesthetic", "mps", "hpsv2", "hpsv21") or field == "repo"):
            options[field] = str(resolve(base, options[field]))
    return options


def resource_fingerprint(options):
    result = dict(options)
    result.pop("python", None)
    checkpoint = options.get("checkpoint")
    if checkpoint and Path(checkpoint).is_file():
        result["checkpoint_sha256"] = file_digest(checkpoint)
    if options.get("repo"):
        repo = Path(options["repo"])
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        # Detect edited evaluator source as well as the commit identity.
        sources = {str(p.relative_to(repo)).replace("\\", "/"): file_digest(p)
                   for p in sorted(repo.rglob("*.py")) if ".git" not in p.parts}
        result["repository_commit"] = commit
        result["source_sha256"] = digest(sources)
    return result


def job_fingerprint(plan_hash, cell, records, metric, options, reference):
    return digest(dict(plan=plan_hash, cell=cell, metric=metric, options=options,
                       protocol=METRIC_PROTOCOLS[metric],
                       implementation=file_digest(Path(__file__)),
                       scorers=file_digest(Path(__file__).with_name("scorers.py")),
                       images=[(r["id"], r["image_sha256"]) for r in records],
                       reference=[(r["name"], r["sha256"]) for r in reference]))


def prepare_jobs(out, plan, plan_hash, base, models=None, benchmarks=None, metrics=None):
    jobs = defaultdict(list)
    references, resources = {}, {}
    selected = list(cells(plan, models, benchmarks))
    if not selected:
        raise ValueError("no experiment cells selected")
    allowed = {m for c in selected for m in plan["benchmarks"][c["benchmark"]]["metrics"]}
    if metrics and set(metrics) - allowed:
        raise ValueError(f"metrics are not configured for the selected benchmarks: {set(metrics) - allowed}")
    for cell in selected:
        bench = plan["benchmarks"][cell["benchmark"]]
        requested = [m for m in bench["metrics"] if not metrics or m in metrics]
        if not requested:
            continue
        records = image_records(out, plan, cell, plan_hash)
        for metric in requested:
            if metric not in resources:
                options = metric_options(plan, metric, base)
                resources[metric] = options, resource_fingerprint(options)
            options, resource = resources[metric]
            reference = []
            if metric == "fid":
                if cell["benchmark"] not in references:
                    references[cell["benchmark"]] = reference_records(bench, base)
                reference = references[cell["benchmark"]]
            fingerprint = job_fingerprint(plan_hash, cell, records, metric, resource, reference)
            jobs[metric].append(dict(cell=cell, records=records, reference=reference,
                                     input_fingerprint=fingerprint, options=options, resource=resource,
                                     output=str((Path(out) / cell["path"] / "metrics" / (metric + ".json")).resolve()),
                                     cache=str((Path(out) / ".evaluation_cache").resolve())))
    return dict(jobs)


def evaluate(out, plan, plan_hash, base, device="cuda", models=None, benchmarks=None, metrics=None):
    with lock(out, ".generate.lock"), lock(out, ".evaluate.lock"):
        jobs = prepare_jobs(out, plan, plan_hash, base, models, benchmarks, metrics)
        for name, tasks in jobs.items():
            options = tasks[0]["options"]
            interpreter = options.get("python", sys.executable)
            if not Path(interpreter).is_absolute() and ("/" in interpreter or "\\" in interpreter):
                interpreter = str(resolve(base, interpreter))
            path = Path(out) / ".jobs" / (name + ".json")
            write_json(path, dict(metric=name, tasks=tasks, device=device,
                                  batch_size=plan.get("evaluation", {}).get("batch_size", 16),
                                  workers=plan.get("evaluation", {}).get("workers", 4)))
            log = Path(out) / "logs" / (name + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [interpreter, "-u", "-m", "cfgctrl.benchmark", "worker", "--task", str(path.resolve())]
            print(f"Evaluating {name}: {len(tasks)} cells; log {log}", flush=True)
            # An old benchmark's dependencies stay in its own interpreter.
            with log.open("a", encoding="utf-8") as stream:
                process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace")
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                    stream.flush()
                code = process.wait()
            if code:
                raise RuntimeError(f"{name} evaluation failed (exit {code}); see {log}. "
                                   "Completed metrics are retained for resumption.")


def score_fid(task, device, batch_size, workers):
    import torch
    try:
        from cleanfid import fid
    except Exception as exc:
        raise RuntimeError("clean-fid could not load. Check the original error and ensure "
                           "torchvision matches the installed PyTorch/CUDA build; "
                           "this is an environment failure, not an FID result.") from exc
    reference = task["reference"]
    key = digest([(r["name"], r["sha256"]) for r in reference])
    root = Path(task["cache"]) / ("reference_" + key)
    for r in reference:
        link_or_copy(r["path"], root / r["name"])
    if set(p.name for p in root.iterdir()) != {r["name"] for r in reference}:
        raise ValueError("unexpected files in the staged FID reference")
    generated = Path(task["cache"]) / ("generated_" + task["input_fingerprint"])
    for r in task["records"]:
        link_or_copy(r["path"], generated / (r["id"] + ".png"))
    if len(list(generated.iterdir())) != len(task["records"]):
        raise ValueError("unexpected files in the staged FID sample")
    # clean-fid's custom cache includes both mode and content identity.
    cache_name = "cfgctrl_" + digest((key, versions(), METRIC_PROTOCOLS["fid"]))
    common = dict(mode="clean", device=torch.device(device), num_workers=workers, batch_size=batch_size)
    if not fid.test_stats_exists(cache_name, mode="clean"):
        fid.make_custom_stats(cache_name, str(root), **common)
    value = float(fid.compute_fid(str(generated), dataset_name=cache_name, dataset_split="custom", **common))
    return dict(value=value, n=len(task["records"]), n_reference=len(reference))


def compbench_command(repo, category, directory, interpreter=sys.executable):
    repo, directory = Path(repo).resolve(), Path(directory).resolve()
    if category == "spatial":
        return ([interpreter, "2D_spatial_eval.py", "--outpath", str(directory)],
                repo / "UniDet_eval", directory / "labels/annotation_obj_detection_2d/vqa_result.json")
    return ([interpreter, "BLIP_vqa.py", "--out_dir", str(directory)],
            repo / "BLIPvqa_eval", directory / "annotation_blip/vqa_result.json")


def parse_compbench_results(path, records):
    found = {}
    for row in read_json(path):
        i = int(row["question_id"])
        if i in found or i < 0 or i >= len(records):
            raise ValueError("duplicate or out-of-range CompBench question_id")
        value = float(row["answer"])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid CompBench score")
        found[i] = value
    if set(found) != set(range(len(records))):
        raise ValueError("CompBench evaluator omitted samples")
    return [dict(id=row["id"], score=found[i]) for i, row in enumerate(records)]


def score_compbench(task, device):
    if not device.startswith("cuda"):
        raise ValueError("the official CompBench evaluators require CUDA")
    if task["resource"].get("repository_commit") != COMPBENCH_REVISION:
        raise ValueError(f"CompBench source must be checked out at {COMPBENCH_REVISION}")
    values = []
    for category in COMPBENCH_CATEGORIES:
        records = [r for r in task["records"] if r["record"]["category"] == category]
        if not records:
            continue
        cache_key = digest((task["input_fingerprint"], versions()))
        directory = Path(task["cache"]) / ("compbench_" + cache_key) / category
        for i, r in enumerate(records):
            prompt = r["record"]["prompt"]
            if any(c in prompt for c in "/\\_\0") or len(prompt.encode("utf-8")) > 220:
                raise ValueError("official CompBench filename protocol cannot represent this prompt unchanged")
            link_or_copy(r["path"], directory / "samples" / f"{prompt}_{i:06d}.png")
        command, cwd, result = compbench_command(task["options"]["repo"], category, directory)
        completed = directory / "completed.json"
        if not completed.exists():
            subprocess.run(command, cwd=cwd, check=True)
            parsed = parse_compbench_results(result, records)
            write_json(completed, parsed)
        else:
            parsed = read_json(completed)
            # Revalidate the cached category's coverage and values.
            aggregate_scores("clip", parsed, records)
        values.extend(parsed)
    return values


def worker(job):
    name, device = job["metric"], job["device"]
    batch = job["batch_size"]
    if type(batch) is not int or batch <= 0:
        raise ValueError("evaluation batch_size must be a positive integer")
    scorer = None
    from .scorers import load_scorer
    for task in job["tasks"]:
        output = Path(task["output"])
        if output.exists():
            previous = read_json(output)
            if previous.get("input_fingerprint") == task["input_fingerprint"] and previous.get("versions") == versions():
                if name == "fid":
                    validate_fid(previous["aggregate"], task)
                else:
                    aggregate_scores(name, previous["scores"], task["records"])
                print(f"{task['cell']['path']} {name}: already complete", flush=True)
                continue
        print(f"{task['cell']['path']} {name}: scoring {len(task['records'])} images", flush=True)
        if name == "fid":
            result = dict(aggregate=score_fid(task, device, batch, job["workers"]))
        elif name == "compbench":
            result = dict(scores=score_compbench(task, device))
        else:
            if scorer is None:
                scorer = load_scorer(name, device, task["options"])
            scores = []
            partial_path = output.with_suffix(".partial.json")
            if partial_path.exists():
                partial = read_json(partial_path)
                if partial.get("input_fingerprint") == task["input_fingerprint"] and partial.get("versions") == versions():
                    scores = partial["scores"]
                    ids = [r["id"] for r in task["records"][:len(scores)]]
                    if [r["id"] for r in scores] != ids or any(not math.isfinite(r["score"]) for r in scores):
                        raise ValueError("invalid partial metric result")
            for start in range(len(scores), len(task["records"]), batch):
                records = task["records"][start:start + batch]
                values = scorer(records)
                if len(values) != len(records) or any(not math.isfinite(float(v)) for v in values):
                    raise RuntimeError(f"{name} returned missing/nonfinite scores")
                scores.extend(dict(id=r["id"], score=float(v)) for r, v in zip(records, values))
                write_json(partial_path, dict(input_fingerprint=task["input_fingerprint"],
                                               versions=versions(), scores=scores))
                if start % (batch * 10) == 0:
                    print(f"  {min(start + batch, len(task['records']))}/{len(task['records'])}", flush=True)
            result = dict(scores=scores)
        if "aggregate" in result:
            validate_fid(result["aggregate"], task)
        write_json(output, dict(metric=name, protocol=METRIC_PROTOCOLS[name],
                                input_fingerprint=task["input_fingerprint"], versions=versions(),
                                resource=task["resource"], **result))
        output.with_suffix(".partial.json").unlink(missing_ok=True)
    del scorer
    gc.collect()


def validate_fid(aggregate, task):
    if not math.isfinite(aggregate["value"]):
        raise ValueError("FID returned a nonfinite score")
    if aggregate["n"] != len(task["records"]) or aggregate["n_reference"] != len(task["reference"]):
        raise ValueError("FID sample counts do not match the manifest")


def aggregate_scores(metric, scores, records):
    by_id = {}
    for row in scores:
        if row["id"] in by_id:
            raise ValueError("duplicate metric sample id")
        if not math.isfinite(row["score"]):
            raise ValueError("nonfinite metric score")
        by_id[row["id"]] = row["score"]
    if set(by_id) != {r["id"] for r in records}:
        raise ValueError("metric coverage does not match generated samples")
    groups = {"overall": records}
    if metric == "compbench":
        groups = {c: [r for r in records if r["record"]["category"] == c] for c in COMPBENCH_CATEGORIES}
    elif metric == "vqascore":
        # The official evaluator's "all" is the union of skill memberships.
        # Seven of the released 1600 prompts have no tags; do not invent a group.
        groups["overall"] = [r for r in records if r["record"]["tags"]]
        groups.update({tag: [r for r in records if tag in r["record"]["tags"]]
                       for tag in ("basic", "advanced")})
    result = []
    for group, selected in groups.items():
        if not selected:
            continue
        # Average repetitions within prompt before averaging prompts.
        prompts = defaultdict(list)
        for r in selected:
            prompts[r["record"]["id"]].append(by_id[r["id"]])
        means = [statistics.mean(v) for v in prompts.values()]
        result.append(dict(group=group, value=statistics.mean(means), n_images=len(selected),
                           n_prompts=len(means), prompt_sd=statistics.stdev(means) if len(means) > 1 else None))
    return result


def report(out, plan, plan_hash, base, models=None, benchmarks=None, metrics=None):
    jobs = prepare_jobs(out, plan, plan_hash, base, models, benchmarks, metrics)
    rows, missing, environments = [], [], defaultdict(set)
    for metric, tasks in jobs.items():
        for task in tasks:
            path = Path(task["output"])
            if not path.exists():
                missing.append(str(path))
                continue
            artifact = read_json(path)
            if artifact.get("input_fingerprint") != task["input_fingerprint"]:
                missing.append(str(path) + " (stale)")
                continue
            environments[metric].add(digest(artifact["versions"]))
            cell = task["cell"]
            if metric == "fid":
                validate_fid(artifact["aggregate"], task)
            groups = ([dict(group="overall", value=artifact["aggregate"]["value"],
                            n_images=artifact["aggregate"]["n"], n_prompts=len({r["record"]["id"] for r in task["records"]}),
                            prompt_sd=None, n_reference=artifact["aggregate"]["n_reference"])]
                      if metric == "fid" else aggregate_scores(metric, artifact["scores"], task["records"]))
            for group in groups:
                rows.append(dict(model=cell["model"], benchmark=cell["benchmark"], arm=cell["arm"], w=cell["w"],
                                 metric=metric, protocol=artifact["protocol"], **group))
    if any(len(v) > 1 for v in environments.values()):
        raise ValueError("metric environments differ across cells; rerun evaluation consistently")
    write_json(Path(out) / "summary.json", dict(plan_fingerprint=plan_hash, complete=not missing,
                                               scope=dict(models=models, benchmarks=benchmarks, metrics=metrics),
                                               results=rows, missing=missing))
    import csv
    path = Path(out) / "summary.csv"
    if rows:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    elif path.exists():
        path.unlink()  # An old report must not masquerade as this incomplete evaluation.
    print(f"{len(rows)} summary rows; {len(missing)} missing/stale metric outputs", flush=True)
    if missing:
        raise RuntimeError("evaluation is incomplete; see summary.json for the missing metrics")
    return rows
