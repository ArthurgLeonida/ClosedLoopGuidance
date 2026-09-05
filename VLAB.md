# Running the real-model study on VLAB

The CPU track can be run locally. The GPU track must first verify integration
with the exact checkpoint, pipeline, scheduler and dtype used for the experiment.
No trained-model run was performed during this repository audit.

## 1. Use a suitable Python and PyTorch environment

On an NGC PyTorch image, first inspect the existing installation and retain it
if it works. Install the experiment dependencies without replacing torch:

~~~bash
python -c "import sys, torch; print(sys.executable, torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip install matplotlib pytest diffusers transformers accelerate sentencepiece protobuf
~~~

If creating a venv inside that image, `python -m venv --system-site-packages .venv`
makes the existing packages visible. A plain CUDA image may need Python, pip and
venv installed by your container administrator first.

On a regular machine or cluster node, create an environment:

~~~bash
python -m venv .venv
source .venv/bin/activate
# Or: conda env create -f environment.yml && conda activate clg
~~~

On PowerShell, activate with `.\.venv\Scripts\Activate.ps1`.

Choose the torch command for your OS, Python version, GPU architecture and
supported CUDA runtime from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).
Then install the remaining dependencies:

~~~bash
python -m pip install matplotlib pytest diffusers transformers accelerate sentencepiece protobuf
~~~

The portable `environment.yml` does not guarantee a particular GPU build.
An extra pip index does not take precedence over PyPI; pip considers candidates
across indexes. It therefore cannot reliably select a CUDA runtime merely by
adding a `cu12x` URL. [pip candidate selection](https://pip.pypa.io/en/stable/cli/pip_install/#finding-packages).

Prebuilt wheels supply runtime dependencies, but the host NVIDIA driver and
GPU architecture still need to be supported. `nvidia-smi` reports driver
capability; `torch.version.cuda` reports the torch build's runtime version.
Do not assume that any CUDA 12.x wheel works fully with every driver above a
single minimum: NVIDIA documents feature and PTX limitations for minor-version
compatibility. [NVIDIA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

Run the following **on the allocated GPU node**, from the repository root:

~~~bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name()); print('bf16:', torch.cuda.is_bf16_supported()); print(torch.ones(1, device='cuda') + 1)"
python -m pytest tests/ -q
~~~

The small CUDA operation checks more than device visibility alone. Use
`--dtype fp16` if bf16 is unsupported, or `--dtype fp32` for a precision check
when memory allows. The default runner loads the full pipeline onto the device;
memory needs depend on the checkpoint and resolution.

### When `torch.cuda.is_available()` is False

A frequent case reports the driver as too old on an otherwise current node:

~~~text
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12080)
~~~

`12080` is the **driver's** CUDA version, here 12.8. The message means the
installed torch was built for a *newer* CUDA than this driver supports; it does
not mean the driver is old in absolute terms. Compare the two directly:

~~~bash
nvidia-smi | head -3                                   # driver and its CUDA version
python -c "import torch; print(torch.version.cuda)"    # what this wheel requires
~~~

Then install a build the driver supports. For a CUDA 12.8 driver:

~~~bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
~~~

A plain `pip install torch` takes the newest default build, which is how an
environment ends up ahead of its driver. `experiments/real_model.py` checks this
before downloading anything, so a mismatch costs seconds instead of a partial
multi-gigabyte download.

## 2. Storage, authentication and batch jobs

Use storage with adequate space for checkpoints and outputs. For example:

~~~bash
export HF_HOME=/scratch/$USER/hf
~~~

Access to a gated checkpoint requires two separate steps, and a `GatedRepoError`
or `401` at load time means one of them is missing. Accept the licence on the
model page **with the same account you authenticate as**, then authenticate in
this environment:

~~~bash
huggingface-cli login          # or `hf auth login` on newer huggingface_hub
# non-interactively, e.g. in a batch job:
export HF_TOKEN=<a read token>
python -c "from huggingface_hub import whoami; print(whoami()['name'])"
~~~

The last line confirms which identity the environment actually uses, which is
what distinguishes "licence not accepted" from "logged in as someone else".
The default checkpoint is `stabilityai/stable-diffusion-3.5-large`; use `--model`
to select another compatible checkpoint you can access. The runner translates
these download failures into the corresponding instruction.

For Slurm with conda, initialize the conda shell hook before activation:

~~~bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate clg
export HF_HOME=/scratch/$USER/hf
cd "$SLURM_SUBMIT_DIR"
python experiments/real_model.py verify
~~~

Adjust allocation directives to your cluster. Containers need GPU access enabled
by the runtime; run the CUDA check inside the actual job/container.

## 3. Verify before generating a grid

~~~bash
python experiments/real_model.py verify --check-w 7.0 --check-steps 8
~~~

Verification checks zero-gain transparency, prompt-sensitive branch order,
controller activity, finite diagnostics and observed denoiser calls. An
ambiguous branch probe is a failure, not permission to guess.

The hook targets pipelines that concatenate unconditional and conditional
predictions into one doubled batch while active CFG is enabled.
It follows the pipeline's active CFG flag and guidance scale. Standard SD3
uses unconditional-first ordering and enables this CFG path at scales greater
than one. [SD3 pipeline source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py).

The grid therefore requires `w>1`. At `w=1`, those pipelines omit the
unconditional branch; they cannot evaluate the paper controller's endpoint.
Use the toy or a direct sampler that explicitly computes both branches for
that comparison. Flux-style separate conditional/unconditional forward passes
require a different adapter and are unsupported here.

A passing verification covers the tested configuration, not every pipeline
feature or scheduler. Repeat it after dependency, checkpoint or integration
changes. Dummy-denoiser tests alone do not establish trained-model compatibility.

## 4. Run paired comparisons

### There is no training set, and no input images

Nothing in this repository is trained. The controller has no learnable
parameters — `SMCConfig` is seven hand-set numbers — there is no optimizer, no
gradient step and no loss anywhere in `cfgctrl/` or `experiments/`, and the
checkpoint is downloaded frozen and run in inference mode. This is a change to
how guidance is computed *during sampling*, not a model you fit.

There is also no input image. A text-to-image pipeline starts from seeded noise
and a prompt: `pipe(prompt, guidance_scale, num_inference_steps, generator)`.
What you supply is **prompts**; the images are **outputs**.

| what | where | commit it? |
|---|---|---|
| prompt lists you supply | anywhere; `--prompts path.txt`. Suggested: `data/prompts/*.txt` | yes, they are small and they define the experiment |
| generated images, `signals.csv`, `config.json` | `results/<name>/<arm>/w<scale>/pNN_sS.png` | no, `results/` is ignored |
| reference images, only if you compute FID | suggested `data/reference/` | no, large and not ours to redistribute |

**Prompts.** One per line, UTF-8, blank lines skipped. The six built-in prompts
are an integration smoke test. For a real comparison, take a standard set:
MS-COCO captions are what the paper's FID and CLIP numbers use, and
T2I-CompBench covers the compositional categories it reports. §5 of the
[improvement roadmap](docs/Improvement_Roadmap.md) requires tuning and test
prompts to be **disjoint**, so keep them in separate files and never tune on
the test file:

~~~bash
python experiments/real_model.py grid --prompts data/prompts/tune.txt --out results/tune
python experiments/real_model.py grid --prompts data/prompts/test.txt --out results/test
~~~

**Reference images are needed only for FID.** CLIP score, ImageReward, HPSv2
and PickScore each read a generated image and its prompt, so they need no
reference set. FID compares your generated distribution against a reference
distribution of real images — the paper uses 5,000 MS-COCO image–text pairs.
This repository computes no metrics at all, by design, so nothing here reads
that directory; it is only where to put the set for whichever FID tool you run
over `results/`.

~~~bash
python experiments/real_model.py grid --w 1.5 2.0 3.0 4.5 7.0 --seeds 0 1 2 --out results/real_sd35
~~~

Use `--prompts prompts.txt` for a UTF-8 file with one prompt per line. The six
built-in prompts are an integration smoke test, not enough for an image-quality
benchmark. The runner writes images by arm and scale, `signals.csv`, and
`config.json`. Choose a fresh output directory for each experiment.

### Selecting and parameterizing the arms

An arm is one guidance law. Arms are given on the command line, so changing a
comparison needs no code edit:

~~~text
[name=]preset[:field=value,...]
~~~

Presets: `cfg` (plain CFG, `k=0`), `paper` (Algorithm 1), `boundary_layer`,
`excess` (boundary layer on the extrapolation only). Aliases: `cfg_baseline`,
`bl`, `sat`, `boundary_layer_excess`. Fields: `lam`, `k`, `phi`, `switching`,
`store_corrected`, `relative_gain`, `excess_only`.

To run the published law alone, at the paper's SD3.5 settings:

~~~bash
python experiments/real_model.py grid --arms cfg paper \
    --w 1.5 3.0 7.0 --out results/paper_only
~~~

`--arms` defaults to `cfg paper excess`. Keep `cfg` in any comparison: it is the
baseline the other arms are read against, and it is also the control on the
integration itself, since its images must match an unhooked run.

Per-arm settings override the run-wide `--lam` and `--k`:

~~~bash
python experiments/real_model.py grid \
    --arms cfg "sd35=paper:k=0.1" "flux=paper:k=0.7" "excess:k=0.3"
~~~

`lam` and `k` are applied before the preset computes its derived values, so
`excess:k=0.3` gets `phi = k*lam = 1.8` rather than a width left over from the
default gain. An explicit `phi=` still wins. Note that comparing the paper at
its per-model gains is a comparison of two tunings, not evidence about either.

The name before `=` becomes the output subdirectory, which is what lets one
preset appear twice at different settings. Without a name an arm takes its
preset's canonical name, so two spellings of the same preset are rejected as
duplicates instead of silently writing one law into two directories.

### Check the matrix before allocating GPU time

~~~bash
python experiments/real_model.py grid --dry-run --arms cfg "flux=paper:k=0.7" --w 2.0 5.0
~~~

`--dry-run` resolves every arm, prints its settings and the image count, then
exits without loading a model, so a wrong matrix costs seconds rather than a
queue slot. `config.json` records the resolved controller settings per arm, not
only their names, so a result directory states which law produced it.

### Resuming an interrupted job

~~~bash
python experiments/real_model.py grid --resume --out results/real_sd35
~~~

`--resume` skips images already on disk and appends to `signals.csv`, which
suits a wall-clock-limited allocation: resubmit the same command until it
completes. A job killed between writing an image and its CSV rows leaves that
image without signals; delete that PNG before resuming to regenerate both.
Resuming does not re-check that the earlier rows came from the same settings,
so keep one output directory per configuration.

Record the environment alongside results:

~~~bash
python -m pip freeze > results/real_sd35/requirements-resolved.txt
~~~

Also record checkpoint revision, scheduler configuration, resolution, negative
prompt, GPU and model-evaluation count. The present config file is not a complete
experiment lockfile.

The CSV describes the controller computation before the hook's final branch
rounding and the pipeline's CFG arithmetic. For small corrections in fp16/bf16,
check their realized effect on the generated prediction; mathematical equivalence
does not ensure identical low-precision rounding.

Evaluate held-out fidelity, alignment and diversity across guidance scales with
paired prompts and seeds, then compare at matched alignment and compute.
Lower chatter alone does not demonstrate better images. Follow the
[improvement roadmap](docs/Improvement_Roadmap.md) for the comparison protocol.
