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

## 2. Storage, authentication and batch jobs

Use storage with adequate space for checkpoints and outputs. For example:

~~~bash
export HF_HOME=/scratch/$USER/hf
~~~

Access to a gated checkpoint may require accepting its license and authenticating
with Hugging Face in your environment. The default checkpoint is
`stabilityai/stable-diffusion-3.5-large`; use `--model` to select another compatible
checkpoint you can access.

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

~~~bash
python experiments/real_model.py grid --w 1.5 2.0 3.0 4.5 7.0 --seeds 0 1 2 --out results/real_sd35
~~~

Use `--prompts prompts.txt` for a UTF-8 file with one prompt per line. The six
built-in prompts are an integration smoke test, not enough for an image-quality
benchmark. The runner writes images by arm and scale, `signals.csv`, and
`config.json`. Choose a fresh output directory for each experiment.

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
