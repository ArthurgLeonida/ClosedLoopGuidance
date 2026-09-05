# Data

Nothing here is training data — nothing in this repository is trained. A
text-to-image sampler takes a prompt and seeded noise, so what you supply is
**prompts**. Real images are needed for exactly one thing: the reference
distribution that FID compares against.

## Layout

```
data/
    prompts/                    committed: these define the experiment
        coco_val2017.txt        one caption per image, all 5,000
        tune.txt                for choosing w and k, never reported
        test.txt                for the reported comparison, disjoint from tune
        coco_val2017_pairs.csv  image_id, file_name, caption
    reference/                  gitignored: large, and not ours to redistribute
        captions_val2017.json   the COCO annotation file
        val2017/                the 5,000 real images, for FID only
```

## Why COCO val2017

The paper evaluates on "a subset of the MS-COCO dataset, comprising 5,000
image-text pairs". `val2017` is exactly 5,000 images with about five captions
each, so it supplies both halves: captions to generate from, images to measure
FID against.

## Getting it

```bash
mkdir -p data/reference

# captions (~250 MB zipped, one JSON extracted)
curl -L -o data/reference/ann.zip \
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -j data/reference/ann.zip 'annotations/captions_val2017.json' -d data/reference/

# the prompt files
python data/prepare_prompts.py --captions data/reference/captions_val2017.json

# images, only if you will compute FID (~780 MB)
curl -L -o data/reference/val2017.zip http://images.cocodataset.org/zips/val2017.zip
unzip -q data/reference/val2017.zip -d data/reference/
```

`prepare_prompts.py` takes one caption per image, choosing the lowest
annotation id so the set is reproducible from the annotation file alone, and
splits a seeded shuffle into disjoint `tune` and `test`. The split is
deliberate: §5.2 of the improvement roadmap requires tuning and reporting on
different prompts, and tuning on the reported set would invalidate the
comparison.

## How many prompts

`experiments/evaluate.py clip` prints `need ~N prompts` from the spread it
observes. On the first SD3.5 pilot that came to roughly 200 to resolve the CLIP
gain the paper reports for SD3.5 (0.3681 → 0.3694 raw cosine). The defaults
here — 250 tune, 1,000 test — leave headroom above that. Prompts, not seeds,
are what narrow the interval: the bootstrap resamples prompts, so they are the
independent unit.

FID is separate and needs thousands of images regardless; on a few hundred it
is dominated by sample-size bias.
