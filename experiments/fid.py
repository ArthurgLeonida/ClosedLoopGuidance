"""Fidelity half of the image-quality comparison: FID and KID per (arm, w).

`evaluate.py clip` measures alignment and deliberately refuses to reimplement
FID. This script supplies the missing number by driving `clean-fid`, and writes
exactly the CSV that `pareto.py` reads, so the three steps chain:

    python experiments/evaluate.py check --run results/coco_test
    python experiments/evaluate.py clip  --run results/coco_test --device cuda
    python experiments/fid.py compute --run results/coco_test \
        --reference data/reference/val2017
    python experiments/pareto.py --run results/coco_test \
        --fid results/coco_test/fid.csv

Two things this does that a bare `compute_fid` call does not.

  * It caches the reference statistics once. A grid of 3 arms x 5 scales is 15
    comparisons; without caching, the reference images go through Inception 15
    times, which can cost more than generating the images did.
  * It refuses to compare cells holding different numbers of images. FID is
    biased downward as sample size grows, so a cell that lost 40 images to a
    killed job scores worse for a reason that has nothing to do with guidance.

KID is computed alongside because at ~1,000 images per cell the FID bias term
dominates the absolute value. That bias is common to every cell, so FID
*differences* stay usable, but KID's estimator is unbiased and is the sounder
number to rank on. Neither is comparable to a published table unless the
reference set, the sample count and the resize also match that table's.

`reference` mode is optional. It builds the subset of COCO whose captions are
exactly this run's prompts:

    python experiments/fid.py reference --images data/reference/val2017 \
        --run results/coco_test --out data/reference/coco_test_ref

Which reference to use is a real choice, not a detail. All of val2017 (5,000
images) is the usual convention and keeps reference sampling noise low, but it
includes images whose captions were never generated. The matched subset removes
that mismatch at the cost of a smaller, noisier reference. Use one or the other
for every arm; never mix them inside a comparison.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import image_index, load_run                     # noqa: E402

Cell = Tuple[str, float]
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
CSV_FIELDS = ["arm", "w", "fid", "kid", "n_gen", "n_ref", "seconds"]


def count_images(directory: Path) -> int:
    return sum(1 for p in directory.iterdir()
               if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def discover_cells(run: Path, cfg: dict) -> Tuple[Dict[Cell, Path], List]:
    """(arm, w) -> the directory holding that cell's images.

    Derived from the same index `evaluate.py` builds, so the alignment and the
    fidelity halves can never disagree about which directory is which arm.
    """
    found, missing = image_index(run, cfg)
    dirs: Dict[Cell, Path] = {}
    for (arm, w, _, _), path in found.items():
        dirs.setdefault((arm, w), path.parent)
    return dict(sorted(dirs.items())), missing


# ---------------------------------------------------------------- reference
def read_pairs(path: Path) -> Dict[str, str]:
    """caption -> file_name, from prepare_prompts.py's pairs CSV."""
    mapping: Dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        absent = {"file_name", "caption"} - set(reader.fieldnames or [])
        if absent:
            raise ValueError(f"{path} is missing column(s): {sorted(absent)}")
        for row in reader:
            # prepare_prompts.py already collapsed whitespace; repeat it here so
            # a prompt file edited by hand still matches.
            mapping.setdefault(" ".join(row["caption"].split()), row["file_name"])
    if not mapping:
        raise ValueError(f"{path} has no rows")
    return mapping


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard link where the filesystem allows it. val2017 is ~800 MB and a
    container's disk quota is rarely generous."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def cmd_reference(args) -> int:
    images = Path(args.images)
    if not images.is_dir():
        raise ValueError(f"{images} is not a directory. Download the real images:\n"
                         "  curl -L -o data/reference/val2017.zip "
                         "http://images.cocodataset.org/zips/val2017.zip\n"
                         "  unzip -q data/reference/val2017.zip -d data/reference/")
    print(f"source     {images}  ({count_images(images)} images)")

    if args.run is None:
        print(f"\nNo --run given, so nothing was subset.")
        print(f"Point `compute --reference {images}` straight at it: that uses all of")
        print("val2017 as the reference distribution, the usual COCO-FID convention.")
        return 0

    cfg = load_run(Path(args.run))
    pairs = read_pairs(Path(args.pairs))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wanted: List[str] = []
    unmatched: List[str] = []
    for prompt in cfg["prompts"]:
        name = pairs.get(" ".join(prompt.split()))
        (wanted.append(name) if name else unmatched.append(prompt))
    unique = sorted(set(wanted))

    print(f"prompts    {len(cfg['prompts'])} in the run")
    print(f"matched    {len(wanted)} captions -> {len(unique)} distinct images")
    if unmatched:
        share = len(unmatched) / len(cfg["prompts"])
        print(f"UNMATCHED  {len(unmatched)} ({share:.1%}), e.g. {unmatched[0]!r}")
        if share > 0.05:
            raise ValueError(
                f"{share:.1%} of this run's prompts are not in {args.pairs}. The "
                "reference would not correspond to what was generated. Check that "
                "the run used a prompt file written by data/prepare_prompts.py from "
                "this same annotation file.")

    absent = [n for n in unique if not (images / n).is_file()]
    if absent:
        raise ValueError(f"{len(absent)} matched images are not in {images}, "
                         f"e.g. {absent[0]}; the archive may be partly unzipped")
    for name in unique:
        link_or_copy(images / name, out / name)

    print(f"\nwrote      {out}  ({count_images(out)} images)")
    print(f"Now: python experiments/fid.py compute --run {args.run} --reference {out}")
    print("A matched reference is smaller and therefore noisier than all of "
          "val2017.\nWhichever you choose, use it for every arm in the comparison.")
    return 0


# ------------------------------------------------------------------ clean-fid
def load_backend():
    try:
        from cleanfid import fid as backend
    except ImportError:
        raise RuntimeError(
            "clean-fid is not installed:  pip install clean-fid\n"
            "It is used instead of a local implementation so the numbers stay "
            "comparable to published tables, which differ by more than the "
            "effects being measured when the resize or the Inception weights "
            "differ.") from None
    return backend


def cache_reference(backend, name: str, reference: Path, mode: str, device,
                    workers: int, batch: int, refresh: bool) -> Optional[str]:
    """Precompute the reference Inception statistics under `name`.

    Returns the name to pass as `dataset_name`, or None if caching failed and
    the caller must fall back to recomputing the reference for every cell.
    """
    try:
        cached = bool(backend.test_stats_exists(name, mode))
    except Exception:
        cached = False
    if cached and refresh:
        backend.remove_custom_stats(name, mode=mode)
        cached = False
    if cached:
        print(f"reference  statistics '{name}' already cached")
        return name
    print(f"reference  computing statistics for {reference} once ...", flush=True)
    try:
        backend.make_custom_stats(name, str(reference), mode=mode, device=device,
                                  num_workers=workers, batch_size=batch)
    except Exception as exc:
        print(f"WARNING: could not cache reference statistics ({exc}).")
        print("         Falling back to recomputing the reference for every cell: "
              "slower, same numbers.")
        return None
    return name


def score_cell(backend, directory: Path, reference: Path, stats: Optional[str],
               mode: str, device, workers: int, batch: int,
               want_kid: bool) -> Tuple[float, float]:
    common = dict(mode=mode, device=device, num_workers=workers,
                  batch_size=batch, verbose=False)
    target = (dict(dataset_name=stats, dataset_split="custom") if stats
              else dict(fdir2=str(reference)))
    fid_value = float(backend.compute_fid(str(directory), **target, **common))
    kid_value = float("nan")
    if want_kid:
        try:
            kid_value = float(backend.compute_kid(str(directory), **target, **common))
        except Exception as exc:
            print(f"           KID unavailable ({exc})")
    return fid_value, kid_value


# -------------------------------------------------------------------- compute
def read_done(path: Path) -> Dict[Cell, dict]:
    """Rows already computed, so a killed job can be resumed."""
    if not path.is_file():
        return {}
    done: Dict[Cell, dict] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                done[(row["arm"], float(row["w"]))] = row
            except (KeyError, TypeError, ValueError):
                continue
    return done


def write_rows(path: Path, rows: Dict[Cell, dict]) -> None:
    """Rewritten after every cell: a grid is a few dozen rows at most, and a
    job that dies at cell 12 of 15 should not lose the first eleven."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wri = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        wri.writeheader()
        for cell in sorted(rows):
            wri.writerow(rows[cell])


def cmd_compute(args) -> int:
    run = Path(args.run)
    reference = Path(args.reference)
    if not reference.is_dir():
        raise ValueError(f"--reference {reference} is not a directory; "
                         "see `fid.py reference --help`")
    cfg = load_run(run)
    dirs, missing = discover_cells(run, cfg)
    if not dirs:
        raise ValueError(f"no images under {run}; has the grid finished?")
    counts = {cell: count_images(directory) for cell, directory in dirs.items()}
    n_ref = count_images(reference)

    print(f"run        {run}")
    print(f"reference  {reference}  ({n_ref} images)")
    print(f"cells      {len(dirs)}")
    for cell, directory in dirs.items():
        print(f"  {cell[0]:<20}{cell[1]:>7.2f}{counts[cell]:>8} images   {directory}")
    if missing:
        print(f"\nWARNING: {len(missing)} images the config expects are absent.")

    sizes = sorted(set(counts.values()))
    if len(sizes) > 1:
        message = (f"cells hold different image counts {sizes}. FID falls with sample "
                   "size, so this comparison would partly measure which job was cut "
                   "short. Finish the grid (`real_model.py grid --resume`), or pass "
                   "--allow-unequal to score it anyway.")
        if not args.allow_unequal:
            raise ValueError(message)
        print(f"\nWARNING: {message}")
    if sizes and sizes[0] < 2048:
        print(f"\nNOTE: {sizes[0]} images per cell. FID at this size is dominated by "
              "its\n      sample-size bias. Read the KID column, and compare arms "
              "against each\n      other rather than quoting either against a "
              "published table.")

    if args.dry_run:
        print("\n--dry-run: nothing was computed.")
        return 0

    import torch
    device = torch.device(args.device)
    backend = args.backend or load_backend()
    stats = None
    if args.backend is None and args.cache_stats:
        stats = cache_reference(backend, args.stats_name, reference, args.mode,
                                device, args.workers, args.batch, args.refresh_stats)

    out_csv = run / args.out
    rows = read_done(out_csv) if args.resume else {}
    if rows:
        print(f"\nresuming: {len(rows)} of {len(dirs)} cells already in {out_csv.name}")

    print(f"\n{'arm':<20}{'w':>7}{'FID':>10}{'KID x1e3':>11}{'seconds':>10}")
    for cell, directory in dirs.items():
        if cell in rows:
            continue
        start = time.time()
        fid_value, kid_value = score_cell(
            backend, directory, reference, stats, args.mode, device,
            args.workers, args.batch, args.kid)
        elapsed = time.time() - start
        rows[cell] = dict(arm=cell[0], w=cell[1], fid=f"{fid_value:.4f}",
                          kid=("" if math.isnan(kid_value) else f"{kid_value:.8f}"),
                          n_gen=counts[cell], n_ref=n_ref, seconds=f"{elapsed:.1f}")
        kid_text = "--" if math.isnan(kid_value) else f"{kid_value * 1e3:.3f}"
        print(f"{cell[0]:<20}{cell[1]:>7.2f}{fid_value:>10.3f}{kid_text:>11}"
              f"{elapsed:>10.1f}", flush=True)
        write_rows(out_csv, rows)
    write_rows(out_csv, rows)

    print(f"\nwrote {out_csv}")
    print("\nLower is better on both columns, but neither ranks a guidance law on "
          "its own:\nweaker guidance improves fidelity and costs alignment, moving "
          "along CFG's own\ncurve. Join the two axes and read the ratio:\n"
          f"    python experiments/pareto.py --run {run} --fid {out_csv}")
    if args.kid:
        print(f"    python experiments/pareto.py --run {run} --fid {out_csv} "
              "--fid-column kid")
    return 0


# ----------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    ref = sub.add_parser("reference", help="inspect or subset the real-image set")
    ref.add_argument("--images", default="data/reference/val2017",
                     help="the unzipped COCO val2017 directory")
    ref.add_argument("--run", default=None,
                     help="subset to the images whose captions are this run's prompts")
    ref.add_argument("--pairs", default="data/prompts/coco_val2017_pairs.csv",
                     help="caption/file_name map from data/prepare_prompts.py")
    ref.add_argument("--out", default="data/reference/matched",
                     help="where the subset is written (hard links where possible)")

    com = sub.add_parser("compute", help="FID and KID for every (arm, w)")
    com.add_argument("--run", required=True, help="a grid output directory")
    com.add_argument("--reference", required=True, help="directory of real images")
    com.add_argument("--out", default="fid.csv", help="written inside --run")
    com.add_argument("--device", default="cuda")
    com.add_argument("--mode", default="clean",
                     choices=["clean", "legacy_pytorch", "legacy_tensorflow"],
                     help="clean-fid resize convention; keep it fixed across runs")
    com.add_argument("--batch", type=int, default=32)
    com.add_argument("--workers", type=int, default=4)
    com.add_argument("--no-kid", dest="kid", action="store_false",
                     help="skip KID; it costs a second Inception pass per cell")
    com.add_argument("--no-cache-stats", dest="cache_stats", action="store_false",
                     help="recompute the reference for every cell instead of caching")
    com.add_argument("--stats-name", default="cfgctrl_ref",
                     help="name of the cached reference statistics. Change it when "
                          "you change --reference, or the stale cache is reused")
    com.add_argument("--refresh-stats", action="store_true",
                     help="discard and rebuild the cached reference statistics")
    com.add_argument("--allow-unequal", action="store_true",
                     help="score cells that hold different numbers of images")
    com.add_argument("--resume", action="store_true",
                     help="keep rows already present in the output CSV")
    com.add_argument("--dry-run", action="store_true",
                     help="list the cells without loading anything")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    args.backend = None                     # tests inject a fake backend here
    try:
        return cmd_reference(args) if args.mode == "reference" else cmd_compute(args)
    except ValueError as exc:
        ap.error(str(exc))
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
