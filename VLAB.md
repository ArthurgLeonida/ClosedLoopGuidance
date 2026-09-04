# Running this on the GPU lab (VLAB)

Everything here assumes you cloned the cleaned repo. Two tracks: the **toy
experiments**, which are finished and run anywhere, and the **real model**,
which is new code that has never been executed and starts with a self-check.

---

## 1. Setup

### The one rule

**PyTorch wheels bundle their own CUDA runtime.** You do *not* need a matching
system CUDA toolkit, `module load cuda`, `nvcc`, or `$CUDA_HOME` to run anything
in this repo — only a new enough NVIDIA **driver**. Any CUDA 12.x wheel runs on
a driver >= 525.60.13, and an H100 node's driver is far newer than that.

`nvidia-smi` reports the highest CUDA version the *driver* supports, not what is
installed. Reading it as "I must install exactly this toolkit" is the usual way
this goes wrong.

So: pick a `cu12x` wheel, ignore the system toolkit, and only worry if `nvcc`
is needed — which it is not here, since nothing compiles a custom kernel.

### Which case are you in?

| you are in | what to do |
|---|---|
| **NGC PyTorch container** (`nvcr.io/nvidia/pytorch:*`) | torch is already installed and tuned. **Do not reinstall it** — see below. |
| **Plain CUDA container** (`nvidia/cuda:12.x-*`) | no Python env at all; install Python first, then a torch wheel. |
| **HPC login/compute node with modules** | `module load` a Python; ignore the CUDA modules; pip a torch wheel. |
| **Conda / mamba** | conda for the interpreter, pip for the packages; `conda env create -f environment.yml`. |
| **Plain machine with a driver** | the simple case; a venv and a pip torch wheel. |

### A. NGC PyTorch container

These ship a build of torch tuned for the card. `requirements.txt` lists
`torch` unpinned, so a plain `pip install -r requirements.txt` will see it
already installed and leave it alone — but **never add `-U`**, which would
replace it with the public wheel, and note that a few images install torch
somewhere pip cannot see, in which case pip would install a second copy that
shadows it. The safe habit is to install everything *except* torch:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # confirm it is there

pip install matplotlib pytest
pip install diffusers transformers accelerate sentencepiece protobuf            # real-model track only
```

Use a venv here only with `--system-site-packages`, or you will lose the
container's torch:

```bash
python -m venv --system-site-packages .venv && source .venv/bin/activate
```

### B. Plain CUDA container (`nvidia/cuda:12.4.1-devel-ubuntu22.04` or similar)

There is no Python environment, and on Ubuntu `venv` is a separate package:

```bash
apt-get update && apt-get install -y python3 python3-pip python3-venv git
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install sentencepiece protobuf
```

Start the container so it can see the GPUs and has enough shared memory for
dataloader workers:

```bash
docker run --gpus all --shm-size=16g -it \
    -v "$PWD":/workspace -v /scratch/$USER/hf:/hf -e HF_HOME=/hf \
    nvidia/cuda:12.4.1-devel-ubuntu22.04 bash
```

### C. HPC with environment modules

Load a **Python**, not a CUDA toolkit. Build the venv on scratch if `$HOME` has
a quota — a torch wheel is several GB.

```bash
module load python/3.11                 # names vary; `module avail python`
python -m venv /scratch/$USER/venvs/clg && source /scratch/$USER/venvs/clg/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install sentencepiece protobuf
```

If pip runs out of space mid-download, redirect its temp dir:
`TMPDIR=/scratch/$USER/tmp pip install ...`.

Do this on a node that has a GPU, or at least verify on one — a login node
usually has no driver, so `torch.cuda.is_available()` will be `False` there and
that tells you nothing about the compute nodes.

### D. Conda or mamba

Usually the right choice on a shared cluster: it gives you a Python interpreter
without modules and without root. One rule — **conda for the interpreter, pip
for the packages.** PyTorch deprecated its conda channel in 2024, so
`conda install pytorch` no longer gets you a current build.

One shot, from the spec committed in the repo:

```bash
conda env create -f environment.yml      # creates an env called "clg"
conda activate clg
```

Or by hand, which is the same thing:

```bash
conda create -n clg python=3.11 -y
conda activate clg
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install sentencepiece protobuf        # real-model track only
```

**Confirm you are actually inside it.** The most common conda mistake is a
`pip` that still belongs to `base`, which installs everything into the wrong
place and leaves the env empty:

```bash
which python pip                          # both must be under .../envs/clg/bin
python -c "import sys; print(sys.prefix)"
```

**Quotas.** Conda puts both the envs and a package cache under `$HOME`, and an
env with torch in it is 5-10 GB. On a cluster, redirect both *before* creating
anything:

```bash
export CONDA_PKGS_DIRS=/scratch/$USER/conda/pkgs
conda create -p /scratch/$USER/envs/clg python=3.11 -y
conda activate /scratch/$USER/envs/clg    # activate by path when using -p
```

or make it permanent in `~/.condarc`:

```yaml
envs_dirs:
  - /scratch/YOUR_USER/conda/envs
pkgs_dirs:
  - /scratch/YOUR_USER/conda/pkgs
```

`conda clean -a` reclaims the cache when it fills up anyway.

**In a batch job**, `conda activate` fails unless the shell hook is sourced
first — a non-interactive shell has never run `conda init`. This is the single
most common SLURM failure:

```bash
#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00

source "$(conda info --base)/etc/profile.d/conda.sh"     # <- without this, activate fails
conda activate clg

export HF_HOME=/scratch/$USER/hf
cd "$SLURM_SUBMIT_DIR"
python experiments/real_model.py verify
```

**mamba** is a drop-in with a much faster solver — `mamba env create -f
environment.yml` — and `micromamba` needs no base installation at all. Either
is worth it if `conda` sits in "Solving environment" for minutes.

**Do not mix** a conda-forge `pytorch` with a pip `torch` in one env. If
`conda list | grep -c torch` returns more than one row, that is your bug;
delete the env and rebuild rather than trying to repair it.

To hand the env to someone else, export what you asked for rather than the full
resolved graph, which is platform-specific and will not solve elsewhere:

```bash
conda env export --from-history > environment.yml
```

### E. Plain venv, no conda

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install sentencepiece protobuf
```

**Which `cu` index?** `cu124` is a safe default for an H100; `cu126` and
`cu128` are also fine on a recent driver, and `cu128` is the one to pick on
Blackwell. On Linux the default PyPI wheel is *already* a CUDA build, so plain
`pip install torch` usually works — the index only matters when you want a
particular CUDA version (an older driver, or matching something else in the
image). On Windows the default wheel is CPU-only, which is why the check below
prints `torch.version.cuda`: if that is `None`, you have a CPU build no matter
what the machine has.

### Check the environment before going further

Run this from the repository root (the last block imports `cfgctrl`):

```bash
python - <<'EOF'
import torch
print("torch          ", torch.__version__)
print("built for CUDA ", torch.version.cuda)
print("cuda available ", torch.cuda.is_available())
assert torch.cuda.is_available(), "no GPU visible: wrong node, missing --gpus all, or a CPU-only wheel"
i = torch.cuda.current_device()
cap = torch.cuda.get_device_capability(i)
print("device         ", torch.cuda.get_device_name(i), cap)
print("memory (GiB)   ", round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1))
print("bf16 supported ", torch.cuda.is_bf16_supported())        # H100 is sm_90 -> True
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
print("matmul ok      ", bool(torch.isfinite(x @ x).all()))

# the controller itself, on the GPU (the hook upcasts bf16 -> fp32, as here)
import sys; sys.path.insert(0, ".")
from cfgctrl import SlidingModeGuidance, presets
e = torch.randn(2, 16, 64, 64, device="cuda", dtype=torch.bfloat16).float()
c = SlidingModeGuidance(presets.boundary_layer_excess())
out = c.correct(e, w=7.0)
print("cfgctrl on cuda", out.device, out.dtype, f"rms(s)={c.history[-1].s_rms:.4f}")
EOF
```

Every line must print before you spend GPU hours. `bf16 supported False` means
you are not on an H100/A100 and should pass `--dtype fp16`. `built for CUDA
None` means a CPU-only build, whatever `nvidia-smi` says.

`nvidia-smi` is still worth a look for who else is on the card and how much
memory is free.

### Weights and cache

SD3.5 and Flux are gated: accept the licence on the Hub once, then log in.
Point the cache at scratch — these are 20-50 GB and will blow up a home quota.

```bash
export HF_HOME=/scratch/$USER/hf                # put this in your job script too
huggingface-cli login
```

SD3.5-large is ~8B parameters plus a ~4.7B T5; in `bf16` that is roughly 30 GB,
comfortable on an 80 GB H100. If you are on a smaller card use
`stabilityai/stable-diffusion-3.5-medium`.

Inside a container, mount the cache from the host so you download once:
`-v /scratch/$USER/hf:/hf -e HF_HOME=/hf`.

---

## 2. Sanity check

```bash
python -m pytest tests/ -q          # 20 tests, ~5 s, CPU only
```

If `test_k_zero_is_exact_cfg_for_every_variant` fails, stop — the baseline is
no longer a special case of the method, and nothing downstream means anything.

---

## 3. The toy experiments

Pure CPU. An H100 buys you nothing here; the only reason to run them on VLAB is
to have the figures next to everything else.

```bash
python experiments/toy_smc_cfg.py --quick        # ~20 s, smoke test
python experiments/toy_smc_cfg.py                # ~45 s, the full set
python experiments/toy_smc_cfg.py --only e4      # one experiment
python experiments/toy_smc_cfg.py --k 0.7        # Flux's gain instead of SD3.5's
```

Output lands in `results/toy/`: one CSV and one PNG per experiment, plus
`summary.json`, which holds every number quoted in the review document. They
are reproducible — the numbers in `docs/` came from exactly this command.

---

## 4. The real model

### 4.1 Verify first — this is not optional

`cfgctrl/diffusers_hook.py` rewrites the denoiser's output so the pipeline's own
CFG line produces the corrected velocity. The algebra is unit-tested against a
dummy denoiser; **the interface to a real pipeline is an assumption.** Two ways
it can be silently wrong: the hook is not called where you think, or the
conditional branch is the other half of the batch — which inverts every
guidance direction while still producing plausible images.

```bash
python experiments/real_model.py verify
```

Three checks, about a minute:

1. **Transparency.** With `k = 0` the hook short-circuits, so the image must be
   *bit-identical* to running with no hook at the same seed.
2. **Batch order.** Runs one step with two different prompts and the same seed
   and reports which half of the doubled batch moved. That half is the
   conditional branch; it must be the half `uncond_first` assumes.
3. **What the hook sees** — step count, latent shape, dtype, and the first real
   measurement of `rms(e)`, `rms(s)`, chatter and the derivative-decides index.

Exit code 0 means the hook is safe on that pipeline. **Do not run the grid
until verify passes.** If check 2 says the conditional is the first half, pass
`uncond_first=False`; if check 2 reports no doubled batch at all, that pipeline
does not do CFG in one pass (Flux with `true_cfg_scale` is like this) and needs
a pipeline-level wrapper instead.

That third block is also the most interesting single result available to you:
if a real model shows a derivative-decides index of 1–2 % and chatter near 1.0
late in the run, it reproduces §3.1–3.2 of the review on a real model rather
than on a Gaussian mixture.

### 4.2 The grid

```bash
python experiments/real_model.py grid \
    --w 1.0 1.5 2.0 3.0 4.5 7.0 \
    --seeds 0 1 2 \
    --steps 30 \
    --out results/real
```

Three arms (`cfg`, `paper`, `excess`) × 6 scales × 6 prompts × 3 seeds = 324
images, roughly 20–30 minutes on one H100 at 1024². Use `--prompts file.txt`
(one per line) for your own set.

Output:

```
results/real/
    config.json            exactly what was run
    signals.csv            per-step rms(e), rms(s), chatter, switching activity
    cfg/w3.0/p00_s0.png    images, one directory per (arm, scale)
    paper/w3.0/...
    excess/w3.0/...
```

**No metrics are computed on purpose.** Point whatever FID / CLIP / ImageReward
tooling you prefer at the image directories afterwards, then plot fidelity
against alignment — one curve per arm, one point per `w`. That Pareto view is
the whole point (§4.7 of the review): a law that merely lowers the effective
guidance slides *along* CFG's curve rather than beating it, and comparing at a
single `w`, as the paper's Table 2 does, cannot tell the two apart.

`signals.csv` needs no extra tooling and answers the structural questions on
its own.

### 4.3 Sweeping `k`

The paper tunes `k` per model: 0.1 for SD3.5 and Qwen-Image, 0.7 for Flux. That
per-model retuning is the thing the scale-free gain (§4.4) is meant to remove,
so it is worth measuring:

```bash
for k in 0.05 0.1 0.2 0.4 0.7; do
  python experiments/real_model.py grid --k $k --out "results/real_k$k" --w 3.0 7.0
done
```

---

## 5. Troubleshooting

| symptom | cause |
|---|---|
| `verify` check 1 fails with a tiny non-zero diff | pipeline non-determinism; pin the generator device, and check no other callback mutates latents |
| `verify` check 2 says "no doubled batch" | that pipeline runs cond and uncond in separate passes — wrap the pipeline, not the denoiser |
| `verify` check 2 is "inconclusive" | both halves moved similarly; use more distinct prompts via `--check-w`/prompt edit |
| `controller.history` empty after a run | the hook attached to the wrong module, or `guidance_scale <= 1` so the pipeline skipped CFG entirely |
| CUDA OOM on SD3.5-large | use `--dtype fp16`, `--model ...-3.5-medium`, or `pipe.enable_model_cpu_offload()` |
| gated repo 401 | accept the licence on the Hub, then `huggingface-cli login` |

---

## 6. What is deliberately missing

- **Metric computation.** The grid writes images; scoring them is left to
  standard tooling so the numbers sit next to published tables.
- **A COCO reference set.** The paper's FID is against 5,000 MS-COCO
  image–text pairs. You need that set for a FID number comparable to Table 2.
- **`relative_gain` on a real model.** It is implemented and toy-tested (§4.4)
  but the arms in `real_model.py` do not include it; add it to `arms()` if you
  want that comparison.
