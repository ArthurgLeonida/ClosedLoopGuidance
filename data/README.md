# Benchmark data

Run all commands from the repository root. Large data, downloaded annotations and weights are ignored by Git. Keep the frozen JSON manifests with the corresponding experiment; `plan.json` embeds their complete contents.

## COCO

Download the **2017 validation images** and **2017 train/validation annotations** from the [official COCO download page](https://cocodataset.org/#download). Extract:

- `captions_val2017.json` to `data/annotations/`.
- The 5,000 validation JPEGs to `data/reference/val2017/`.

The [nohup download command](../docs/Running_Benchmarks.md#1-prepare-annotations-and-the-aesthetic-head)
downloads and extracts both archives into these paths. The preparation command
below requires the annotation file to exist already.

~~~bash
python -m cfgctrl.benchmark prepare coco --annotations data/annotations/captions_val2017.json
~~~

The default selects 5,000 distinct image identities and one caption per image with selection seed 0. Caption choice, image IDs and source hashes are frozen in `data/benchmarks/coco.json`. The reference set for FID is exactly those image identities; unrelated files in the reference folder are excluded.

The paper does not release its exact pairs or specify the COCO split sufficiently to recover them. Val2017 and the seeded caption choice are this repository's declared choices. If the original pairs become available, provide a JSON list:

~~~json
[
  {"image_id": 139, "caption_id": 123456}
]
~~~

The IDs above illustrate the format; use real matching IDs from your annotation file. Pass `--pairs path/to/pairs.json --count 5000`. The preparation command rejects repeated images and captions assigned to another image. Use a separate manifest and output directory for a smaller smoke test.

## T2I-CompBench

~~~bash
python -m cfgctrl.benchmark prepare compbench --download
~~~

This downloads the four official validation prompt files at revision `4aa404212eb5d06e5adbcd9cee696c750d0d25a5` of [T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench). Each contains 300 prompts; the combined manifest contains 1,200. Alternatively, prepare from a local checkout with `--repo external/T2I-CompBench`.

Generation does not need the evaluator installed. Scoring needs the official BLIP-VQA/UniDet environments and weights described in [VLAB.md](../VLAB.md). Prompt files alone do not provide these assets.

## GenAI-Bench

~~~bash
python -m cfgctrl.benchmark prepare genai --download
~~~

The command downloads only the prompt and skill JSONs from [BaiqiL/GenAI-Bench-1600](https://huggingface.co/datasets/BaiqiL/GenAI-Bench-1600). It does not download the benchmark's example model images. Use `--revision COMMIT` to pin the upstream revision. Local files can instead be supplied with `--prompts image.json --skills genai_skills.json`.

The released 1,600 prompts include 722 Basic, 871 Advanced and seven untagged prompts. Generate and score all 1,600; the official Overall aggregation uses the union of skill tags, hence 1,593 prompts. The separate GenAI-Bench-527 prompt/skill files are also accepted when supplied explicitly, but the default protocol uses 1,600.

## Evaluation weights

~~~bash
python -m cfgctrl.benchmark prepare aesthetic
~~~

This obtains the official `sac+logos+ava1-l14-linearMSE.pth` predictor. Most other scoring checkpoints download through their official loaders on first evaluation. MPS and CompBench need manual setup; see [VLAB.md](../VLAB.md).

Do not tune controller gains on the final reporting prompts. Keep a separate tuning manifest and use the frozen benchmark configuration for the reported comparison.
