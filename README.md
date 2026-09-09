# CFG-Ctrl: generation and paper benchmarks

Compare the published controller with ordinary CFG and the implemented chattering fixes on SD3.5-Large, FLUX.1-dev and Qwen-Image. One configuration selects the models, controller arms, scales, benchmarks and seeds.

**Developing a new method? Start with the [research workflow](docs/Research_Workflow.md): 32-image smoke tests, 256-image development runs, and separate validation prompts.** The full matrix below is for later benchmark evaluation.

| Benchmark | Evaluation |
|---|---|
| COCO, 5,000 frozen image-caption pairs | FID, CLIP, Aesthetic, ImageReward, PickScore, HPSv2, HPSv2.1, MPS |
| T2I-CompBench validation | Official color, shape, texture and 2D spatial evaluators |
| GenAI-Bench, 1,600 prompts | VQAScore: Basic, Advanced and Overall |

These are the benchmark and metric families in [CFG-Ctrl](https://arxiv.org/html/2603.03281v2). The released paper does not identify its exact COCO pairs or every evaluation setting. This repository declares a reproducible local protocol; its results are not a claim of exact table reproduction. See [benchmark settings and limitations](docs/Benchmark_Protocol.md).

## Setup

Use Python 3.11 and a matching GPU build of PyTorch and torchvision, then:

~~~bash
python -m pip install -r requirements/generation.txt
~~~

Generation and the older evaluation packages use separate environments. Follow [VLAB.md](VLAB.md) for the evaluator environments, official checkpoints, persistent caches and background runs. The CPU controller tests only need `requirements.txt`.

Prepare the data once, from the repository root:

~~~bash
python -m cfgctrl.benchmark prepare coco --annotations data/annotations/captions_val2017.json
python -m cfgctrl.benchmark prepare compbench --download
python -m cfgctrl.benchmark prepare genai --download
python -m cfgctrl.benchmark prepare aesthetic
~~~

Place COCO validation images in `data/reference/val2017`. [Data instructions](data/README.md) explain downloads, explicit image-caption pairs and the frozen manifests. CompBench preparation downloads prompts only; evaluation also needs its official code and weights.

## Select and run

Edit [configs/paper.json](configs/paper.json). It selects all three models and all three benchmarks, with GenAI-Bench restricted to SD3.5 to match the paper's supplementary comparison. Each arm gets the same prompts and seeds. For a smaller configuration:

~~~bash
python -m cfgctrl.benchmark init --models sd35 --benchmarks coco --config configs/sd35.json
~~~

Set `evaluation.python` to the metrics environment's Python. Override `evaluation.scorers.mps.python` and `evaluation.scorers.compbench.python` for their environments. Paths in JSON are relative to the configuration file; absolute paths also work.

Inspect the image count and check setup before starting an expensive run:

~~~bash
python -m cfgctrl.benchmark plan --config configs/paper.json
python -m cfgctrl.benchmark doctor --config configs/paper.json
python -u -m cfgctrl.benchmark run --config configs/paper.json --out results/paper
~~~

For multiple GPUs and an unattended run with hardware, generation, diagnostic and
evaluation logs, follow [the nohup commands](docs/Running_Benchmarks.md). The generator
accepts `--gpus 0 1` and coordinates one worker per GPU under a single output tree.

`run` generates first, then evaluates and writes the report. The full default matrix is **121,200 images**. Remove models, benchmarks or arms from a separate configuration for an initial trial. `plan` loads no weights and creates no run directory; `doctor` checks assets and environments without testing model inference.

The stages can also run separately:

~~~bash
python -u -m cfgctrl.benchmark generate --config configs/paper.json --out results/paper
python -u -m cfgctrl.benchmark evaluate --out results/paper
python -m cfgctrl.benchmark report --out results/paper
python -m cfgctrl.benchmark diagnostics --out results/paper
~~~

Repeat a command to resume. Images and metric batches are saved as they finish. Changed image contents, prompts or generation settings are rejected. Use a fresh output directory for a different generation experiment. To change evaluator paths, checkpoints or metrics while retaining the same images:

~~~bash
python -u -m cfgctrl.benchmark evaluate --config configs/paper.json --out results/paper
~~~

Generation and evaluation accept `--models sd35` and `--benchmarks coco` to run part of the saved matrix. Evaluation/report additionally accept `--metrics fid clip`. A filtered report explicitly records its scope; it does not certify the rest of the matrix.

`--gpus` belongs to `generate`; the full workflow wrapper runs parallel generation
followed by evaluation on the first selected GPU. A model copy is loaded per worker.

## Outputs and comparisons

Each `model/benchmark/arm/w*/` directory contains PNG images, per-image records and raw metric results. The run root contains `plan.json`, evaluator logs, `summary.json` and `summary.csv`. Missing or stale results make reporting fail instead of silently reducing the sample count. `diagnostics.csv` reports late error, switching and the correction delivered to the scheduler.

The default arms are `conditional` (one branch, `w=1`), `cfg`, `paper`, `excess`, `proximal` and `proximal_relative:k=0.1`. Named ablations such as `small=proximal:k=0.05` are supported. The relative gain is dimensionless; an equal numeric absolute gain is a different intervention. [The controller guide](docs/Chattering_Fixes.md) ranks improvements and explains the observed ±0.5 surface cycle. Better image quality remains an experimental question.

The old `experiments/*.py` entry points, toy workflow and duplicate reviews have been retired. Existing results, images and weights were preserved; legacy outputs are not silently imported into the new experiment format. Pre-cleanup source backups are in `results/pre_pipeline_user_changes_20260907.zip` and `results/retired_pipeline_sources_20260907.zip`.

## Tests

~~~bash
python -m pip install -r requirements.txt
python -m pytest tests -q
~~~

The suite covers controller mathematics, all three model calling conventions using small test pipelines, identity checks, interrupted runs and metric aggregation. It does not download checkpoints or establish image quality.
