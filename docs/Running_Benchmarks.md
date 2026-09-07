# All benchmarks with nohup and multiple GPUs

Run these commands from the repository root on the Linux GPU machine. They source `vlab_env.sh` inside the detached shell, so the generation environment and persistent caches are available after you disconnect.

Before the full run, install the generation and evaluator environments and configure their Python paths as described in [VLAB.md](../VLAB.md). COCO JPEGs and caption annotations must already be extracted to the paths in [data/README.md](../data/README.md). MPS and CompBench need their official source repositories and weights. The full workflow checks these prerequisites before generating images.

For the H100/CUDA 12.4 host, [the CompBench setup command](../VLAB.md#t2i-compbench) creates its dedicated environment and runs both official inference backends. Wait for `COMPBENCH_SETUP_COMPLETE` before launching the full workflow; doctor alone does not test model inference.

## 1. Prepare annotations and the aesthetic head

If `data/annotations/captions_val2017.json` or the COCO reference images are missing,
download and extract the official archives first. `prepare coco` reads an existing
annotation file; it does not download the dataset.

~~~bash
nohup bash -c '
  set -e
  command -v wget >/dev/null
  command -v unzip >/dev/null
  mkdir -p data/downloads data/annotations data/reference
  wget -c -P data/downloads http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -n data/downloads/annotations_trainval2017.zip annotations/captions_val2017.json -d data
  wget -c -P data/downloads http://images.cocodataset.org/zips/val2017.zip
  unzip -n data/downloads/val2017.zip -d data/reference
  test -s data/annotations/captions_val2017.json
  echo COCO_DOWNLOAD_COMPLETE
' > coco_download.nohup.log 2>&1 &
~~~

Wait for `COCO_DOWNLOAD_COMPLETE` in that log before continuing. Downloads resume
with `wget -c`; extraction keeps existing files. The URLs come from the
[official COCO download page](https://cocodataset.org/#download).

Run once, if the manifests/weight are not prepared yet:

~~~bash
nohup bash -c '
  set -e
  source ./vlab_env.sh
  if [ ! -f data/benchmarks/coco.json ]; then
    python -u -m cfgctrl.benchmark prepare coco --annotations data/annotations/captions_val2017.json
  fi
  if [ ! -f data/benchmarks/compbench.json ]; then
    python -u -m cfgctrl.benchmark prepare compbench --download
  fi
  if [ ! -f data/benchmarks/genai.json ]; then
    python -u -m cfgctrl.benchmark prepare genai --download
  fi
  if [ ! -f data/weights/sac+logos+ava1-l14-linearMSE.pth ]; then
    python -u -m cfgctrl.benchmark prepare aesthetic
  fi
  echo PREPARATION_COMPLETE
' > benchmark_prepare.nohup.log 2>&1 &
~~~

Wait for `PREPARATION_COMPLETE` in the log before launching the next command. This prepares small annotations and the aesthetic head, not the COCO image archive or the external evaluator installations.

## 2. Generate on multiple GPUs, then score and report

Two GPUs:

~~~bash
nohup bash -c '
  set -e
  source ./vlab_env.sh
  exec bash scripts/run_benchmarks.sh \
    --config configs/paper.json \
    --out results/paper_all \
    --gpus 0 1
' >> paper_all.nohup.log 2>&1 &
~~~

For four GPUs, replace `--gpus 0 1` with `--gpus 0 1 2 3` in the same command. Choose one launch; do not start both for the same output.

The wrapper runs these stages in order:

1. Plan and environment/asset checks.
2. Image generation, one worker per selected GPU.
3. Controller diagnostics.
4. Every configured benchmark metric, using the first selected GPU.
5. Combined CSV/JSON report.

The full default configuration generates **121,200 images**, across all six arms. Each GPU handles a disjoint subset of prompt/seed pairs for every selected model. Models are loaded successively; each active worker has its own model copy. The number of GPUs changes throughput, not the sample count, seeds or output layout.

This distributes images across GPUs. Each worker must still fit its model with the configured CPU offload, and each model copy consumes host RAM. Keep `offload: "model"` initially; choose `"none"` only when an entire pipeline fits on each GPU. A single image does not use the combined VRAM of all selected GPUs. This follows the multiple-prompts form of [Diffusers distributed inference](https://huggingface.co/docs/diffusers/v0.35.1/en/training/distributed_inference).

GPU indices are relative to the process's existing `CUDA_VISIBLE_DEVICES` list. If the scheduler exposes physical GPUs 3 and 7 through that variable, use `--gpus 0 1`. The launcher logs the resulting assignments.

The default configuration covers:

| Benchmark | Values |
|---|---|
| COCO: SD3.5, FLUX, Qwen | FID, CLIP, Aesthetic, ImageReward, PickScore, HPSv2, HPSv2.1, MPS |
| CompBench: SD3.5, FLUX, Qwen | Color, shape, texture, spatial |
| GenAI-Bench: SD3.5 | Basic, Advanced, Overall VQAScore |

GenAI is restricted to SD3.5 in the supplied configuration to match the paper's supplementary comparison. To evaluate it on all three models, remove `benchmarks.genai.models` from a separate config before generation. That expanded matrix generates 140,400 images. Other methods appearing in the paper, such as CFG-Zero*, are not implemented arms.

## 3. Results and useful logs

| File/directory | What to inspect |
|---|---|
| `paper_all.nohup.log` | Combined stage output and top-level failures |
| `results/paper_all.logs/workflow.status` | Running PID, or final exit code; 0 means the wrapper succeeded |
| `results/paper_all.logs/{plan,doctor,generate,diagnostics,evaluate,report}.log` | Output of each stage, appended across resumes |
| `results/paper_all.logs/gpu_usage.csv` | GPU utilization, used/total VRAM, temperature and power, sampled every 10 seconds |
| `results/paper_all.logs/gpu_info.txt` | Driver/device snapshot |
| `results/paper_all.logs/generation_packages.json` | Generation environment package versions |
| `results/paper_all/logs/generation_rank*.log` | Image-by-image progress and failures for each GPU worker |
| `results/paper_all/logs/generation_status.json` | Worker PIDs, GPU assignments and exit codes |
| `results/paper_all/logs/{fid,clip,...}.log` | Individual metric worker output |
| `results/paper_all/summary.csv` and `summary.json` | Combined benchmark values, counts and completeness |
| `results/paper_all/diagnostics.csv` | Final error/surface RMS, late switching and delivered velocity corrections |
| `results/paper_all/*/*/*/w*/records/*.json` | Full controller traces, sigma-step displacement corrections, seeds, timings and image hashes |
| `results/paper_all/*/*/*/w*/metrics/*.json` | Raw metric scores, protocols, environment versions and evaluator identity |

The wrapper stops its GPU monitor automatically when it exits. Metric versions are also recorded in each metric artifact. A report attempted after a scorer failure lists completed and missing metrics; `complete: false` is not a complete benchmark run.

For a detached progress snapshot:

~~~bash
nohup bash -c '
  cat results/paper_all.logs/workflow.status
  if [ -f results/paper_all/logs/generation_status.json ]; then
    cat results/paper_all/logs/generation_status.json
  fi
  for log in results/paper_all/logs/generation_rank*.log; do
    [ -f "$log" ] || continue
    printf "\n%s\n" "$log"
    tail -n 8 "$log"
  done
' > paper_all.progress.log 2>&1 &
~~~

Open `paper_all.progress.log` or the full logs using the file browser.

## 4. Resume or run stages individually

Repeat the full-run command to resume. Completed images and metric batches are reused. You may change the number of GPUs between stopped runs. Keep the generation configuration, implementation and package versions unchanged. Use a fresh output for a changed experiment.

If you prefer separate jobs, wait for each to finish before starting its dependent stage. These are alternatives to the full wrapper.

Generation only:

~~~bash
nohup bash -c '
  set -e
  source ./vlab_env.sh
  exec python -u -m cfgctrl.benchmark generate \
    --config configs/paper.json --out results/paper_all --gpus 0 1
' >> paper_all.generate.nohup.log 2>&1 &
~~~

Evaluation, report and diagnostics after generation is complete:

~~~bash
nohup bash -c '
  set -e
  source ./vlab_env.sh
  python -u -m cfgctrl.benchmark diagnostics --out results/paper_all
  evaluation_status=0
  CUDA_VISIBLE_DEVICES=0 python -u -m cfgctrl.benchmark evaluate \
    --config configs/paper.json --out results/paper_all || evaluation_status=$?
  report_status=0
  python -u -m cfgctrl.benchmark report --out results/paper_all || report_status=$?
  if [ "$evaluation_status" -ne 0 ]; then exit "$evaluation_status"; fi
  exit "$report_status"
' >> paper_all.evaluate.nohup.log 2>&1 &
~~~

In this standalone scoring example, `CUDA_VISIBLE_DEVICES=0` selects physical GPU 0. Replace it with the permitted physical GPU/UUID on your machine; the full wrapper handles an inherited visibility mapping automatically.

After a forced kill, inspect `generation_status.json` and confirm the coordinator and all workers have stopped before removing stale locks. Ordinary worker errors stop the other workers and release their locks. The wrapper also owns `results/paper_all.logs/.workflow.lock`.

The output uses the [declared local benchmark protocol](Benchmark_Protocol.md). Missing paper settings still prevent claiming exact reproduction of every published number. Multi-GPU ownership, recovery and process handling are tested on CPU stand-ins; actual GPU throughput and evaluator inference require the target machine.
