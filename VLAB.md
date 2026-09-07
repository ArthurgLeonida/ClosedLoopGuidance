# GPU setup and persistent runs

Use the repository root as the working directory. Model generation, general metrics, MPS and the official CompBench evaluators have different dependency requirements; each worker can use a separate Python interpreter.

## Generation environment

On a Linux VLAB machine with conda:

~~~bash
export CLG_PERSIST=/path/to/persistent/volume
source vlab_env.sh
python -m pip install -r requirements/generation.txt
~~~

Replace the volume path. `vlab_env.sh` keeps the environment, Hugging Face downloads, pip cache and Git settings on that volume. An existing environment is activated without upgrading it; the explicit pip command above installs the new pipeline's generation dependencies.

The script's default PyTorch index is `https://download.pytorch.org/whl/cu128`. Set `CLG_TORCH_INDEX` before sourcing if your GPU/driver requires a different build; use a matched torch/torchvision installation from [PyTorch's installer](https://pytorch.org/get-started/locally/). Diffusers is pinned to 0.35.2 because the adapters depend on its branch ordering.

Accept any required model access terms on the checkpoint's Hugging Face page and authenticate with `hf auth login`. SD3.5-Large, FLUX.1-dev and Qwen-Image require substantial GPU and host memory. The configuration uses bf16 with model CPU offload; choose sequential offload for lower VRAM use at a speed cost, or fp16 if bf16 is unavailable.

## General metric environment

Create another Python 3.11 environment, install the same compatible GPU torch/torchvision build, then the metric requirements:

~~~bash
conda create -y -p "$CLG_PERSIST/envs/clg-metrics" python=3.11
conda run -p "$CLG_PERSIST/envs/clg-metrics" python -m pip install torch torchvision --index-url "$CLG_TORCH_INDEX"
conda run -p "$CLG_PERSIST/envs/clg-metrics" python -m pip install -r requirements/metrics.txt
~~~

This environment supplies clean-fid, CLIP, Aesthetic, ImageReward, PickScore, HPSv2/v2.1 and VQAScore. VQAScore uses the image-era `t2v-metrics==1.1` implementation of CLIP-FlanT5-XXL with its default question and answer templates. Installing a current video-oriented release changes dependencies and APIs.

Set this interpreter as `evaluation.python` in your configuration:

~~~json
"evaluation": {
  "python": "/path/to/persistent/volume/envs/clg-metrics/bin/python",
  "batch_size": 16,
  "workers": 4,
  "scorers": {
    "aesthetic": {
      "checkpoint": "../data/weights/sac+logos+ava1-l14-linearMSE.pth"
    },
    "mps": {
      "python": "/path/to/persistent/volume/envs/clg-mps/bin/python",
      "repo": "../external/MPS",
      "checkpoint": "../data/weights/MPS_overall_checkpoint.pth"
    },
    "compbench": {
      "python": "/path/to/persistent/volume/envs/clg-compbench/bin/python",
      "repo": "../external/T2I-CompBench"
    }
  }
}
~~~

Merge this into `configs/paper.json`, replacing the example paths. JSON does not expand shell variables. Relative paths resolve from the configuration file. On Windows use the environment's `python.exe` path with forward slashes.

Run `python -m cfgctrl.benchmark prepare aesthetic` for the aesthetic head. The other general scorers obtain their official weights on first use. Reduce `evaluation.batch_size` if scoring runs out of memory; it does not change which images are included.

## MPS

Clone the [official MPS source](https://github.com/Kwai-Kolors/MPS) to `external/MPS`. Download its **MPS_overall_checkpoint.pth** using the checkpoint link in that repository's README and place it in `data/weights/`. This is the publicly released overall model.

Create a separate Python 3.10 or 3.11 environment, install a compatible torch/torchvision pair, then:

~~~bash
conda run -p "$CLG_PERSIST/envs/clg-mps" python -m pip install -r requirements/mps.txt
~~~

Create `clg-mps` first, as with the metrics environment. These inference requirements retain the official model's older Transformers interface. The checkpoint serializes its model class, so the matching source repository is required. The pipeline scores the raw overall logit through the official paired interface; it does not turn a single-image comparison into a constant softmax score.

## T2I-CompBench

~~~bash
nohup bash -c '
  set -e
  test -d external/T2I-CompBench ||
    git clone https://github.com/Karine-Huang/T2I-CompBench.git external/T2I-CompBench
  git -C external/T2I-CompBench checkout 4aa404212eb5d06e5adbcd9cee696c750d0d25a5
' > compbench_source.nohup.log 2>&1 &
~~~

For the H100 host with the CUDA 12.4 toolkit and GCC 11, after the source checkout completes:

~~~bash
nohup bash -c '
  set -e
  export CLG_ENV=/home/jovyan/compartilhado/envs/clg-generation
  source ./vlab_env.sh
  export CUDA_HOME=/usr/local/cuda-12.4
  export CUDA_VISIBLE_DEVICES=0
  bash scripts/setup_compbench.sh
' > compbench_setup.nohup.log 2>&1 &
~~~

The installer creates `clg-compbench` under `$CLG_PERSIST/envs`, installs inference dependencies, and compiles the upstream-pinned Detectron2 revision for H100 (SM 9.0). It uses the [official PyTorch 2.5.1/torchvision 0.20.1 CUDA 12.4 pairing](https://pytorch.org/get-started/previous-versions/), matching the local compiler as required by [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). This is an adapted inference environment, not the upstream frozen CUDA 11 training environment. Other hardware requires reviewing these defaults.

Setup downloads UniDet's required detection checkpoint from the Prismer author's repository. Official BLIP inference downloads its checkpoint and BERT tokenizer on first use. It then exercises a compiled CUDA kernel and the official BLIP-VQA and UniDet scripts on bundled examples. Only successful inference with complete, finite scores prints `COMPBENCH_SETUP_COMPLETE`. Logs, package versions and diagnostic scores are retained under `results/compbench_setup/`; these scores are not benchmark results. If inference fails, inspect the newest `smoke_*/{color,spatial}/inference.log`. This setup requires validation on the target GPU host; local CPU tests do not establish CUDA compatibility.

Set `evaluation.scorers.compbench.python` to that environment's interpreter. The pipeline stages the exact prompt-based filenames, invokes `BLIP_vqa.py` for color/shape/texture and `2D_spatial_eval.py` for spatial relations, and verifies complete result coverage. It preserves finished categories when a later category fails.

The upstream evaluators choose CUDA internally. To select a GPU consistently for generation and evaluation, set `CUDA_VISIBLE_DEVICES` and use the default `--device cuda`.

## Run and recover

For a complete background run using multiple GPUs, use the [nohup workflow](docs/Running_Benchmarks.md).
It collects GPU usage, per-worker logs, diagnostics and every configured metric.
The commands below remain useful for a single GPU.

Prepare the manifests as described in [README.md](README.md), then:

~~~bash
python -m cfgctrl.benchmark plan --config configs/paper.json
python -m cfgctrl.benchmark doctor --config configs/paper.json
mkdir -p "$CLG_PERSIST/logs"
nohup python -u -m cfgctrl.benchmark run --config configs/paper.json --out results/paper > "$CLG_PERSIST/logs/paper.log" 2>&1 &
echo $!
tail -f "$CLG_PERSIST/logs/paper.log"
~~~

The setup check verifies configured files, package presence and torch/torchvision imports in the selected interpreters. It cannot establish that all model downloads or evaluator inference will succeed. A small separate configuration is useful for the first GPU trial.

Generation saves each image and its metadata; scoring saves completed batches and metric files. Rerun the same command after an ordinary failure. An abruptly killed process can leave a lock file containing its PID: confirm the coordinator and its workers have stopped before removing that specific lock. Use one coordinated job per output directory; `generate --gpus` manages its worker processes together.

To repair evaluator settings after generation, edit the config and run:

~~~bash
python -u -m cfgctrl.benchmark evaluate --config configs/paper.json --out results/paper
python -m cfgctrl.benchmark report --out results/paper
~~~

Changed evaluator source, local weights, inputs or metric options invalidate the relevant metric cache. Changed generation settings require a new output directory. Model repositories default to `revision: "main"`; set commit revisions before a long experiment to avoid upstream changes across resumed runs.

## Git settings

The persistence helper stores Git settings at `$CLG_PERSIST/git/gitconfig`. Set your identity once with `git config --global user.name` and `user.email`. Authentication is separate; use your existing SSH or credential setup for the machine.
