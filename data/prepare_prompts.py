"""Build prompt files from MS-COCO captions, for `real_model.py --prompts`.

The paper evaluates on "a subset of the MS-COCO dataset, comprising 5,000
image-text pairs". COCO `val2017` is exactly 5,000 images, each with about five
captions, so it is the natural match and it supplies both halves of the
comparison: the captions drive generation, and the images are the reference
distribution FID needs.

This script does not invent prompts. It reads the official annotation file and
writes deterministic, disjoint prompt lists, so a run can be reproduced from the
annotation file alone.

    # ~25 MB, or 250 MB for the whole trainval annotation bundle
    mkdir -p data/reference
    curl -L -o data/reference/ann.zip \
        http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -j data/reference/ann.zip 'annotations/captions_val2017.json' \
        -d data/reference/

    python data/prepare_prompts.py --captions data/reference/captions_val2017.json

Writes into data/prompts/:

    coco_val2017.txt        one caption per image, all 5,000, deterministic
    tune.txt                a disjoint slice for choosing w and k
    test.txt                a disjoint slice for the reported comparison
    coco_val2017_pairs.csv  image_id, file_name, caption -- to build the FID
                            reference set from the same images

Prompt files are small and worth committing: they define the experiment. The
images are not, and `data/reference/` is gitignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_pairs(captions: Path) -> List[Tuple[int, str, str]]:
    """(image_id, file_name, caption), one caption per image, deterministic.

    COCO gives several captions per image. Taking the lowest annotation id is
    an arbitrary but reproducible choice, and it avoids letting a random draw
    silently change the prompt set between runs.
    """
    with open(captions, encoding="utf-8") as fh:
        blob = json.load(fh)
    for field in ("images", "annotations"):
        if field not in blob:
            raise ValueError(f"{captions} has no {field!r}; this is not a COCO caption file")

    file_names: Dict[int, str] = {img["id"]: img["file_name"] for img in blob["images"]}
    best: Dict[int, Tuple[int, str]] = {}
    for ann in blob["annotations"]:
        image_id, ann_id = ann["image_id"], ann["id"]
        text = " ".join(ann["caption"].split())          # collapse newlines and runs of space
        if not text:
            continue
        if image_id not in best or ann_id < best[image_id][0]:
            best[image_id] = (ann_id, text)

    pairs = [(i, file_names[i], best[i][1]) for i in sorted(best) if i in file_names]
    if not pairs:
        raise ValueError(f"{captions} produced no usable image/caption pairs")
    return pairs


def split(pairs: List[Tuple[int, str, str]], n_tune: int, n_test: int,
          seed: int) -> Tuple[List, List]:
    """Disjoint tune and test slices from a seeded shuffle.

    Shuffled rather than sliced in id order, because COCO ids carry weak
    structure and an ordered slice could differ systematically between the two
    sets. The roadmap requires the sets to be disjoint: tuning on the reported
    set would invalidate the comparison.
    """
    if n_tune + n_test > len(pairs):
        raise ValueError(f"asked for {n_tune} + {n_test} prompts but only "
                         f"{len(pairs)} images are available")
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:n_tune], shuffled[n_tune:n_tune + n_test]


def write_prompts(path: Path, rows: List[Tuple[int, str, str]]) -> None:
    path.write_text("".join(f"{caption}\n" for _, _, caption in rows), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captions", type=Path,
                    default=Path("data/reference/captions_val2017.json"))
    ap.add_argument("--out", type=Path, default=Path("data/prompts"))
    ap.add_argument("--n-tune", type=int, default=250,
                    help="prompts for choosing w and k; never reported")
    ap.add_argument("--n-test", type=int, default=1000,
                    help="prompts for the reported comparison; ~200 is the minimum "
                         "to resolve the paper's CLIP gain, see evaluate.py --detect")
    ap.add_argument("--seed", type=int, default=0, help="fixes the tune/test split")
    args = ap.parse_args()

    if not args.captions.is_file():
        ap.error(f"{args.captions} not found. Download it first:\n"
                 "  mkdir -p data/reference\n"
                 "  curl -L -o data/reference/ann.zip "
                 "http://images.cocodataset.org/annotations/annotations_trainval2017.zip\n"
                 "  unzip -j data/reference/ann.zip 'annotations/captions_val2017.json' "
                 "-d data/reference/")

    pairs = load_pairs(args.captions)
    tune, test = split(pairs, args.n_tune, args.n_test, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    write_prompts(args.out / "coco_val2017.txt", pairs)
    write_prompts(args.out / "tune.txt", tune)
    write_prompts(args.out / "test.txt", test)
    with open(args.out / "coco_val2017_pairs.csv", "w", newline="", encoding="utf-8") as fh:
        wri = csv.writer(fh)
        wri.writerow(["image_id", "file_name", "caption"])
        wri.writerows(pairs)

    overlap = {r[0] for r in tune} & {r[0] for r in test}
    assert not overlap, "tune and test overlap"          # the split's whole purpose

    print(f"{len(pairs)} images with captions in {args.captions}")
    print(f"  {args.out/'coco_val2017.txt'}        {len(pairs)} prompts")
    print(f"  {args.out/'tune.txt'}                {len(tune)} prompts  (seed {args.seed})")
    print(f"  {args.out/'test.txt'}                {len(test)} prompts, disjoint")
    print(f"  {args.out/'coco_val2017_pairs.csv'}  image/caption mapping")
    print(f"\nexample prompt: {pairs[0][2]!r}")
    print("\nFor FID you also need the matching real images:")
    print("  curl -L -o data/reference/val2017.zip http://images.cocodataset.org/zips/val2017.zip")
    print("  unzip -q data/reference/val2017.zip -d data/reference/")
    print("Then compare a generated directory against data/reference/val2017.")
    print("Commit the prompt files; data/reference/ is gitignored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
