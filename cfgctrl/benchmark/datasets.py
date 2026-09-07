"""Freeze benchmark identities before generation; never infer prompts from filenames."""
from collections import defaultdict
from pathlib import Path
import random
import urllib.request

from .common import file_digest, identifier, read_json, write_json

COMPBENCH_REVISION = "4aa404212eb5d06e5adbcd9cee696c750d0d25a5"
COMPBENCH_REPO = "https://github.com/Karine-Huang/T2I-CompBench"
COMPBENCH_CATEGORIES = ("color", "shape", "texture", "spatial")
GENAI_REPO = "BaiqiL/GenAI-Bench-1600"


def validate_manifest(data):
    if data.get("schema_version") != 1 or data.get("benchmark") not in ("coco", "compbench", "genai"):
        raise ValueError("expected a version-1 coco, compbench or genai manifest")
    records = data.get("records", [])
    if not records:
        raise ValueError("benchmark has no records")
    seen = set()
    for row in records:
        identifier(row["id"])
        if row["id"] in seen:
            raise ValueError(f"duplicate benchmark id: {row['id']}")
        seen.add(row["id"])
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"empty prompt for {row['id']}")
        if data["benchmark"] == "coco":
            if not isinstance(row.get("image_id"), int) or not row.get("reference"):
                raise ValueError("COCO rows require image_id and reference filename")
            # A manifest may name files, never traverse out of the reference root.
            if Path(row["reference"]).name != row["reference"] or "\\" in row["reference"]:
                raise ValueError("COCO reference must be a filename")
        if data["benchmark"] == "compbench" and row.get("category") not in COMPBENCH_CATEGORIES:
            raise ValueError("unsupported T2I-CompBench category")
        if data["benchmark"] == "genai":
            if not isinstance(row.get("tags"), list) or any(not isinstance(t, str) for t in row["tags"]):
                raise ValueError("GenAI records require a list of official skill tags (possibly empty)")
    if data["benchmark"] == "coco" and len({r["image_id"] for r in records}) != len(records):
        raise ValueError("COCO manifest must contain one caption per unique reference image")
    if data["benchmark"] == "coco" and len({r["reference"] for r in records}) != len(records):
        raise ValueError("COCO reference filenames must be unique")
    return data


def save_manifest(path, benchmark, records, provenance):
    data = validate_manifest(dict(schema_version=1, benchmark=benchmark,
                                  provenance=provenance, records=records))
    path = Path(path)
    if path.exists() and read_json(path) != data:
        raise ValueError(f"{path} already describes a different benchmark; choose another output")
    write_json(path, data)
    return data


def prepare_coco(annotations, out, count=5000, seed=0, pairs=None):
    """Use explicit released pair IDs if supplied; otherwise freeze a declared subset."""
    data = read_json(annotations)
    images = {int(row["id"]): row for row in data["images"]}
    captions = defaultdict(list)
    by_caption = {}
    for row in data["annotations"]:
        captions[int(row["image_id"])].append(row)
        by_caption[int(row["id"])] = row
    if count <= 0:
        raise ValueError("count must be positive")
    if pairs:
        # Explicit identity list: [{"image_id": ..., "caption_id": ...}, ...].
        chosen = read_json(pairs)
        if len(chosen) != count:
            raise ValueError("pair manifest size does not match --count")
        records = []
        for pair in chosen:
            caption = by_caption[int(pair["caption_id"])]
            if caption["image_id"] != pair["image_id"]:
                raise ValueError("caption_id does not belong to its image_id")
            records.append(caption)
    else:
        ids = sorted(set(images) & set(captions))
        if len(ids) < count:
            raise ValueError(f"requested {count} distinct images, annotations contain {len(ids)}")
        rng = random.Random(seed)
        selected = sorted(rng.sample(ids, count))
        records = [rng.choice(sorted(captions[i], key=lambda c: c["id"])) for i in selected]
    rows = [dict(id=f"coco_{r['image_id']:012d}", image_id=r["image_id"],
                 caption_id=r["id"], prompt=r["caption"].strip(),
                 reference=images[r["image_id"]]["file_name"]) for r in records]
    return save_manifest(out, "coco", rows, dict(
        source="MS-COCO captions", annotations_sha256=file_digest(annotations),
        pairs_sha256=file_digest(pairs) if pairs else None, selection_seed=seed,
        exact_paper_pairs="not independently verified",
        protocol="provided pair IDs" if pairs else "declared deterministic subset; paper pair IDs unreleased"))


def prepare_compbench(repo, out):
    repo = Path(repo)
    rows, hashes = [], {}
    for category in COMPBENCH_CATEGORIES:
        path = repo / "examples" / "dataset" / f"{category}_val.txt"
        prompts = [s.strip() for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
        if len(prompts) != 300:
            raise ValueError(f"{path}: expected 300 official validation prompts, found {len(prompts)}")
        hashes[category] = file_digest(path)
        rows.extend(dict(id=f"{category}_{i:04d}", prompt=prompt, category=category)
                    for i, prompt in enumerate(prompts))
    return save_manifest(out, "compbench", rows,
                         dict(source=COMPBENCH_REPO, split="validation", files_sha256=hashes))


def download_compbench_prompts(directory):
    """Download four small prompt files, not model weights or the evaluation code."""
    directory = Path(directory)
    for category in COMPBENCH_CATEGORIES:
        relative = f"examples/dataset/{category}_val.txt"
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/Karine-Huang/T2I-CompBench/{COMPBENCH_REVISION}/{relative}"
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"{target} differs from pinned official prompts")
        target.write_bytes(content)
    return directory


def prepare_genai(prompts, skills, out):
    data, tags = read_json(prompts), read_json(skills)
    if len(data) not in (527, 1600):
        raise ValueError("GenAI-Bench must contain the official 527 or 1600 prompt set")
    membership = defaultdict(list)
    for tag, ids in tags.items():
        for i in ids:
            membership[int(i)].append(tag)
    rows = [dict(id=f"genai_{int(i):05d}", source_id=str(i), prompt=row["prompt"],
                 tags=sorted(membership[int(i)])) for i, row in sorted(data.items(), key=lambda x: int(x[0]))]
    return save_manifest(out, "genai", rows, dict(
        source="GenAI-Bench compositional generation (Li et al.)", count=len(rows),
        prompts_sha256=file_digest(prompts), skills_sha256=file_digest(skills)))


def download_genai(directory, revision="main"):
    from huggingface_hub import hf_hub_download
    # Only annotations; the benchmark's existing generated image archives are unnecessary.
    return [hf_hub_download(GENAI_REPO, name, repo_type="dataset", revision=revision,
                            local_dir=directory) for name in ("genai_image.json", "genai_skills.json")]
