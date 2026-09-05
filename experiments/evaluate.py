"""Score a `real_model.py grid` output directory.

Computes **CLIP score** (alignment) with paired, prompt-level statistics, and
tells you the command for **FID** rather than reimplementing it. That split is
deliberate:

  * CLIP score is per image, so it is meaningful on a small pilot and can be
    paired: the same prompt and seed across arms, differenced. Pairing removes
    prompt difficulty, which is the dominant source of variance.
  * FID is a property of a whole distribution, not of an image. It is strongly
    biased at small sample sizes and its value depends on the Inception weights,
    the resize, and the sample count. Reimplementing it here would produce a
    number that is internally consistent but not comparable to any published
    table, so this script points at `clean-fid` instead.

Alignment alone cannot decide whether a guidance law is better: attenuating
guidance moves fidelity and alignment together along one curve. Read this
output next to a fidelity metric, at matched alignment. See section 4.7 of
docs/CFG-Ctrl_Review_and_Improvements.md.

    python experiments/evaluate.py check --run results/pilot
    python experiments/evaluate.py clip  --run results/pilot

!! `check` exists because the failure that matters here is silent: if prompts
!! are matched to the wrong images, every score is still a plausible number.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLIP_SCALE = 2.5      # Hessel et al. 2021: CLIPScore = 2.5 * max(cos, 0)
Key = Tuple[str, float, int, int]        # (arm, w, prompt_id, seed)


# --------------------------------------------------------------------- layout
def load_run(run: Path) -> dict:
    cfg_path = run / "config.json"
    if not cfg_path.is_file():
        raise ValueError(f"{cfg_path} not found; point --run at a grid output directory")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    for field in ("prompts", "arms", "w", "seeds"):
        if field not in cfg:
            raise ValueError(f"{cfg_path} has no {field!r}; it was not written by this runner")
    if not cfg["prompts"]:
        raise ValueError("this run recorded no prompts")
    return cfg


def image_index(run: Path, cfg: dict) -> Tuple[Dict[Key, Path], List[Key]]:
    """Map every expected image to its path. Missing files are returned so the
    caller can report them rather than silently scoring a subset."""
    found: Dict[Key, Path] = {}
    missing: List[Key] = []
    for arm in cfg["arms"]:
        for w in cfg["w"]:
            for pid in range(len(cfg["prompts"])):
                for seed in cfg["seeds"]:
                    key = (arm, float(w), pid, int(seed))
                    path = run / arm / f"w{w}" / f"p{pid:02d}_s{seed}.png"
                    (found.__setitem__(key, path) if path.is_file() else missing.append(key))
    return found, missing


# --------------------------------------------------------------------- CLIP
def load_clip(model_name: str, device: str):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    return model, CLIPProcessor.from_pretrained(model_name)


def embed(model, inputs) -> Tuple["object", "object"]:
    """Projected (image, text) embeddings, as tensors.

    Do NOT use `get_image_features` / `get_text_features`: transformers 4.x
    returns the projected tensor from those, while transformers 5.x returns a
    `BaseModelOutputWithPooling` whose `pooler_output` has been replaced by that
    tensor. Normalizing the 5.x return value raises `AttributeError: 'BaseModel
    OutputWithPooling' object has no attribute 'norm'`. The `CLIPOutput` fields
    used here mean the same thing in both versions.
    """
    import torch

    out = model(**inputs)
    img, txt = getattr(out, "image_embeds", None), getattr(out, "text_embeds", None)
    if not (torch.is_tensor(img) and torch.is_tensor(txt)):
        raise RuntimeError(
            "this CLIP model returned "
            f"{type(img).__name__}/{type(txt).__name__} instead of image_embeds "
            "and text_embeds tensors; the installed transformers may have changed "
            "the CLIP output contract again. Pin a known-good version, or adapt "
            "`embed()` in this file."
        )
    return img, txt


def clip_scores(paths: Sequence[Path], prompts: Sequence[str], model_name: str,
                device: str, batch: int = 16) -> List[float]:
    """CLIPScore = 2.5 * max(cosine(image, text), 0), one per (path, prompt)."""
    import torch
    from PIL import Image

    if len(paths) != len(prompts):
        raise ValueError("paths and prompts must be the same length")
    model, processor = load_clip(model_name, device)
    out: List[float] = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
            inputs = processor(text=list(prompts[i:i + batch]), images=imgs,
                               return_tensors="pt", padding=True, truncation=True).to(device)
            img_emb, txt_emb = embed(model, inputs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            cos = (img_emb * txt_emb).sum(-1)
            out.extend((CLIP_SCALE * cos.clamp(min=0)).float().cpu().tolist())
    return out


# ---------------------------------------------------------------- statistics
def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def paired_delta(scores: Dict[Key, float], arm: str, baseline: str, w: float,
                 prompts: int, seeds: Sequence[int]) -> Dict[int, float]:
    """Per prompt, the mean over seeds of (arm - baseline) at one scale.

    Pairing on (prompt, seed) removes prompt difficulty, which dominates the
    spread of CLIP score and would otherwise swamp the effect being measured.
    """
    per_prompt: Dict[int, float] = {}
    for pid in range(prompts):
        diffs = [scores[(arm, w, pid, s)] - scores[(baseline, w, pid, s)]
                 for s in seeds
                 if (arm, w, pid, s) in scores and (baseline, w, pid, s) in scores]
        if diffs:
            per_prompt[pid] = mean(diffs)
    return per_prompt


def bootstrap_ci(values: Sequence[float], resamples: int = 10000, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile CI, resampling PROMPTS (the independent unit), not images."""
    import random
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(resamples):
        means.append(mean([values[rng.randrange(n)] for _ in range(n)]))
    means.sort()
    lo = means[int(alpha / 2 * resamples)]
    hi = means[min(int((1 - alpha / 2) * resamples), resamples - 1)]
    return lo, hi


# -------------------------------------------------------------------- check
def cmd_check(args) -> int:
    run = Path(args.run)
    cfg = load_run(run)
    found, missing = image_index(run, cfg)
    failures: List[str] = []

    print(f"run       {run}")
    print(f"arms      {list(cfg['arms'])}")
    print(f"scales    {cfg['w']}\nseeds     {cfg['seeds']}\nprompts   {len(cfg['prompts'])}")
    print(f"images    {len(found)} found, {len(missing)} missing")
    if missing:
        failures.append(f"{len(missing)} expected images are missing, e.g. {missing[:3]}")
        print(f"      FAIL  first missing: {missing[:3]}")
    if not found:
        print("\nCHECK FAILED: no images at all")
        return 1

    baseline = args.baseline if args.baseline in cfg["arms"] else list(cfg["arms"])[0]
    print(f"baseline  {baseline}")
    if len(cfg["arms"]) < 2:
        print("      NOTE  only one arm: there is nothing to compare against")

    # The silent failure: prompts matched to the wrong images. Score each image
    # against its own prompt and against a different one; the diagonal must win.
    print("\nprompt/image alignment: does each image match its OWN prompt best?")
    n = min(args.check_images, len(cfg["prompts"]))
    if n < 2:
        print("      SKIP  needs at least two prompts")
    else:
        w0, s0 = float(cfg["w"][0]), int(cfg["seeds"][0])
        keys = [(baseline, w0, pid, s0) for pid in range(n)]
        if any(key not in found for key in keys):
            print("      SKIP  the baseline arm is missing images at the first scale/seed")
        else:
            paths = [found[key] for key in keys]
            own = [cfg["prompts"][pid] for pid in range(n)]
            shifted = own[1:] + own[:1]           # each image with a NEIGHBOUR's prompt
            scorer = args.scorer or clip_scores
            matched = scorer(paths, own, args.clip_model, args.device)
            mismatched = scorer(paths, shifted, args.clip_model, args.device)
            dm, dx = mean(matched), mean(mismatched)
            print(f"      own prompt {dm:.4f}   shifted prompt {dx:.4f}   margin {dm - dx:+.4f}")
            if not math.isfinite(dm) or not math.isfinite(dx):
                failures.append("CLIP produced nonfinite scores")
                print("      FAIL  nonfinite scores")
            elif dm > dx:
                print("      PASS  images are paired with the right prompts")
            else:
                failures.append("images score no better against their own prompt than a "
                                "neighbour's: the prompt/image mapping is wrong, or the "
                                "images carry no prompt signal")
                print("      FAIL  mapping looks wrong; every later number would be meaningless")

    print("\n" + "=" * 70)
    if failures:
        print("CHECK FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("CHECK PASSED -- layout and prompt/image pairing are consistent.")
    return 0


# --------------------------------------------------------------------- clip
def cmd_clip(args) -> int:
    run = Path(args.run)
    cfg = load_run(run)
    found, missing = image_index(run, cfg)
    if missing:
        print(f"WARNING: {len(missing)} expected images are missing; "
              f"scoring the {len(found)} present. Run `check` first.")
    if not found:
        raise ValueError("no images to score")

    keys = sorted(found)
    prompts = [cfg["prompts"][pid] for (_, _, pid, _) in keys]
    scorer = args.scorer or clip_scores
    print(f"scoring {len(keys)} images with {args.clip_model} on {args.device} ...", flush=True)
    values = scorer([found[k] for k in keys], prompts, args.clip_model, args.device)
    scores: Dict[Key, float] = dict(zip(keys, values))

    out_csv = run / "clip_scores.csv"
    with open(out_csv, "w", newline="") as fh:
        wri = csv.writer(fh)
        wri.writerow(["arm", "w", "prompt_id", "seed", "clip_score"])
        for key in keys:
            wri.writerow([key[0], key[1], key[2], key[3], f"{scores[key]:.6f}"])

    arms = list(cfg["arms"])
    baseline = args.baseline if args.baseline in arms else arms[0]
    n_prompts, seeds = len(cfg["prompts"]), [int(s) for s in cfg["seeds"]]

    print(f"\nCLIP score ({CLIP_SCALE} x max(cos, 0)), baseline = {baseline!r}")
    print(f"{'arm':<12}{'w':>7}{'mean':>9}{'vs baseline':>14}{'95% CI (prompt bootstrap)':>30}")
    rows = []
    for w in [float(x) for x in cfg["w"]]:
        for arm in arms:
            vals = [scores[k] for k in keys if k[0] == arm and k[1] == w]
            if not vals:
                continue
            line = f"{arm:<12}{w:>7.2f}{mean(vals):>9.4f}"
            delta_mean = lo = hi = float("nan")
            if arm != baseline:
                per_prompt = paired_delta(scores, arm, baseline, w, n_prompts, seeds)
                if per_prompt:
                    deltas = list(per_prompt.values())
                    delta_mean = mean(deltas)
                    lo, hi = bootstrap_ci(deltas)
                    flag = "" if (math.isnan(lo) or lo <= 0 <= hi) else "  *"
                    line += f"{delta_mean:>+14.4f}   [{lo:+.4f}, {hi:+.4f}]{flag}"
            print(line)
            rows.append(dict(arm=arm, w=w, mean=mean(vals), n=len(vals),
                             paired_delta=delta_mean, ci_lo=lo, ci_hi=hi))

    with open(run / "clip_summary.csv", "w", newline="") as fh:
        wri = csv.DictWriter(fh, fieldnames=["arm", "w", "mean", "n", "paired_delta",
                                             "ci_lo", "ci_hi"])
        wri.writeheader()
        wri.writerows(rows)

    print(f"\nwrote {out_csv.name} and clip_summary.csv")
    print(f"\n  * marks a paired difference whose 95% interval excludes zero, over "
          f"{n_prompts} prompt(s).\n"
          "  With few prompts that interval is wide and a starred result is weak "
          "evidence.\n"
          "  Alignment alone does not rank guidance laws: attenuating guidance moves\n"
          "  fidelity and alignment together. Pair this with FID at matched alignment.")
    print("\nFor fidelity, use a standard FID implementation so the number is comparable\n"
          "to published tables, e.g.:\n"
          "    pip install clean-fid\n"
          "    python -c \"from cleanfid import fid; "
          f"print(fid.compute_fid('{run}/{baseline}/w{cfg['w'][0]}', 'data/reference/coco'))\"\n"
          "FID needs thousands of images; on a pilot it is dominated by sample-size bias.")
    return 0


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "clip"])
    ap.add_argument("--run", required=True, help="a grid output directory")
    ap.add_argument("--baseline", default="cfg", help="arm every other arm is differenced against")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32",
                    help="the CLIPScore convention uses ViT-B/32; keep it fixed across runs")
    ap.add_argument("--device", default="cpu", help="cpu is fine for a pilot")
    ap.add_argument("--check-images", type=int, default=6,
                    help="how many images the pairing check scores")
    args = ap.parse_args()
    args.scorer = None                      # tests inject a scorer here
    try:
        return cmd_check(args) if args.mode == "check" else cmd_clip(args)
    except ValueError as exc:
        ap.error(str(exc))
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
