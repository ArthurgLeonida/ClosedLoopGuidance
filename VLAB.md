# Running the real-model study on VLAB

The CPU track can be run locally. The GPU track must first verify integration
with the exact checkpoint, pipeline, scheduler and dtype used for the experiment.

Integration has been verified once, on SD3.5-large in bf16 at 1024x1024, 8 steps,
w = 7: zero-gain transparency, unconditional-first batch order, and one
controller call per solver step. That single pass covers those settings only.
No image-quality comparison against the paper has been produced.

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

### Containers: what survives a relaunch

In a container, only the mounted volume persists. Everything else is part of
the container layer and is discarded on restart. Two things land outside the
mount by default and are therefore lost:

* **A named conda environment.** `conda create -n clg` places it inside the base
  installation, for example `/root/anaconda3/envs/clg`. Create it *on the
  volume* with a prefix instead: `conda create -p /mnt/vol/envs/clg`, activated
  by path with `conda activate /mnt/vol/envs/clg`.
* **The Hugging Face cache**, which defaults to `$HOME/.cache/huggingface` and
  holds tens of gigabytes. Losing it means re-downloading every checkpoint.

Do not guess which directory is mounted — ask, since a wrong guess silently
rebuilds everything next time:

~~~bash
findmnt -n -o TARGET,FSTYPE | grep -v overlay     # bind mounts and volumes
~~~

`vlab_env.sh` in this repository does the whole thing and is idempotent:

~~~bash
source vlab_env.sh        # after every relaunch
~~~

It puts the environment, the conda package cache, the pip cache and `HF_HOME`
on the volume (by default the directory containing this repository, which is
the mount on a Jupyter image), creates the environment on first use, activates
it afterwards, and prints the torch build and CUDA availability. Override the
defaults with `CLG_PERSIST`, `CLG_ENV` and `CLG_TORCH_INDEX` — the last must
match the driver's CUDA version, see §1.

On first creation it also writes `requirements-resolved.txt` beside the
environment. `environment.yml` deliberately does not pin a CUDA build, so it
alone will not reproduce a working GPU environment; the resolved file records
what actually worked.

### The Hugging Face cache

`HF_HOME` is the single knob: it covers both the model cache (`$HF_HOME/hub`)
and the auth token (`$HF_HOME/token`), so persisting it keeps the downloads
*and* the login. `vlab_env.sh` sets it to `$CLG_PERSIST/hf` and reports how
much is cached, so a relaunch does not re-download SD3.5.

Confirm it is actually in effect — the value the library resolves is what
matters, not the variable:

~~~bash
python -c "from huggingface_hub import constants as c; print(c.HF_HUB_CACHE)"
~~~

Three things that silently defeat it:

* **Not sourcing `vlab_env.sh` before running.** Downloads then land in
  `$HOME/.cache/huggingface`, inside the container layer, and vanish on
  relaunch. If that has already happened this session, move it rather than
  downloading again — the script prints the exact `mv` when it detects this.
* **A stale `TRANSFORMERS_CACHE`, `HUGGINGFACE_HUB_CACHE` or `HF_HUB_CACHE`.**
  These take precedence over `HF_HOME` for part of the cache. The script warns
  if any is set; unset it.
* **A read-only or full volume.** Downloads then fall back or fail partway.

Authenticate once and it persists with the cache, which also removes the
"sending unauthenticated requests" warning and its lower rate limits:

~~~bash
huggingface-cli login          # writes $HF_HOME/token
~~~

Once everything you need is cached, you can skip hub round-trips entirely,
which speeds up start-up and makes a run reproducible against a fixed cache:

~~~bash
export HF_HUB_OFFLINE=1
~~~

Anything missing then fails loudly instead of downloading, which is what you
want during an experiment and not what you want while setting one up.

### Git configuration and credentials

`~/.gitconfig`, `~/.git-credentials` and `~/.ssh/` are all in `$HOME`, so they
go the same way as the environment.

**Configuration** is handled for you: `vlab_env.sh` exports
`GIT_CONFIG_GLOBAL` to a file on the volume (git 2.32 and later), so identity
and `safe.directory` survive. Set your identity once:

~~~bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
~~~

It also registers the repository under `safe.directory`, because a container
running as a different uid than the volume's owner otherwise gets
`detected dubious ownership`. **No credential is written to that file.**

**Credentials are a different decision**, because persisting one means writing
a secret at rest on the volume. Find out who can read it first — this mount is
named *compartilhado*, "shared":

~~~bash
stat -c '%A %U %G' /home/jovyan/compartilhado
~~~

If other people can read that path, anything stored there is readable by them,
whichever method you pick. In many shared containers every session is `root`,
in which case `chmod` protects nothing.

Three options, in the order worth considering:

1. **Persist nothing.** Cache in memory for the session, re-authenticate after
   each relaunch. Nothing reaches disk.
   ~~~bash
   git config --global credential.helper 'cache --timeout=28800'
   ~~~
2. **A scoped, expiring token.** If you do persist one, limit the damage it can
   do: a GitHub *fine-grained* PAT restricted to this single repository, with
   only `Contents: read and write`, and a short expiry.
   ~~~bash
   install -d -m 700 "$CLG_PERSIST/git"
   git config --global credential.helper "store --file=$CLG_PERSIST/git/credentials"
   # the next push prompts once and writes the token in PLAINTEXT
   chmod 600 "$CLG_PERSIST/git/credentials"
   ~~~
3. **An SSH deploy key on the volume.** Prefer a per-repository deploy key over
   an account-wide key, for the same reason.
   ~~~bash
   ssh-keygen -t ed25519 -f "$CLG_PERSIST/git/id_ed25519" -N "" -C vlab
   chmod 600 "$CLG_PERSIST/git/id_ed25519"
   export GIT_SSH_COMMAND="ssh -i $CLG_PERSIST/git/id_ed25519 -o IdentitiesOnly=yes"
   ~~~
   Add that `export` to `vlab_env.sh` if you settle on this route.

Whichever you choose, keep the secret outside the repository. `CLG_PERSIST`
defaults to the repository's parent, so `git/` is already outside it; the
`.gitignore` also lists `git/` in case you point `CLG_PERSIST` at the
repository itself.

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

Resuming merges `config.json` rather than overwriting it, so the sweep axes
accumulate and `evaluate.py` still sees everything in the directory. Changes
that would make the directory self-inconsistent — a different prompt list,
model, dtype or step count, or an arm name redefined with different settings —
are refused, because `prompt_id` and the arm directories would otherwise mean
different things for different images in the same run.

### Long runs that outlive the terminal

A full sweep is many hours, and two separate things can end it: closing the tab,
and the platform stopping an idle container. **`nohup` only solves the first.**
Check the second before committing to a long job — on JupyterHub an idle culler
will stop the container regardless of what is running inside it. If there is a
culler, prefer short jobs and `--resume`.

Use `run_grid.sh`, which handles the parts that are easy to get wrong:

~~~bash
source vlab_env.sh
nohup ./run_grid.sh results/coco_test "3.0 2.0 4.5 1.5 7.0" \
    --arms cfg paper excess --prompts data/prompts/test.txt --seeds 0 \
    > "$CLG_PERSIST/logs/coco_test.out" 2>&1 &
~~~

It runs **one scale per invocation**, which matters because `cmd_grid` loops
over arms on the outside: a single call with every scale would finish all of
`cfg` before starting `paper`, so there would be no complete slice of the curve
until the very end. A scale at a time gives all arms at one scale in a couple
of hours — enough to evaluate and decide whether to continue — and an
interruption costs one scale rather than the run.

It also sources `vlab_env.sh` if the shell has not, `cd`s to the repository so
relative paths hold, passes `-u` so Python does not block-buffer the log into
looking hung, timestamps each scale, records the real exit status per scale,
continues past a failed scale rather than discarding the rest, writes a PID
file, and takes a lock on the output directory so two sweeps cannot interleave
rows in `signals.csv` and race on `config.json`. `--resume` is always on.

Logs land in `$CLG_PERSIST/logs/<outdir>_w<scale>.log`, on the volume, so they
survive a relaunch along with the images.

Wrapping the **whole loop** in `nohup` is the point: `nohup` on each `python`
protects the generation processes, but not the loop that launches them, so
closing the tab could leave the running scale alive while the remaining scales
never start.

### Spreading the sweep across free GPUs

One `grid` process uses one GPU: `pipe.to("cuda")`, batch size one, one image at
a time. On a shared multi-GPU node most of the hardware therefore sits idle
while a sweep takes a day or more. `run_sweep.sh` reads free memory per device,
uses only the cards with room, and gives each one a share of the scales:

~~~bash
CLG_PLAN_ONLY=1 ./run_sweep.sh results/coco "3.0 2.0 4.5 1.5 7.0"   # look first

nohup ./run_sweep.sh results/coco "3.0 2.0 4.5 1.5 7.0" \
    --arms cfg paper excess --prompts data/prompts/test.txt --seeds 0 \
    > sweep.out 2>&1 &
~~~

~~~text
  gpu 3: 5005 MiB free -- skipping (need 45000)
usable gpus  1 2 5 7  (4)
  gpu 1 -> 3.0 7.0
  gpu 2 -> 2.0
~~~

Each scale gets **its own output directory**, `results/coco_w3.0` and so on.
Two processes writing one directory would interleave rows in `signals.csv` and
race on `config.json`, and `run_grid.sh`'s lock would refuse the second job.
Separate directories are self-contained: run `evaluate.py check` and `clip` on
each, then assemble the Pareto curve across them.

| knob | meaning |
|---|---|
| `CLG_MIN_FREE_MIB` | memory a card must have free, default 45000. SD3.5-large in bf16 measured about 35 GB; lower this to use a card that is close but under. |
| `CLG_MAX_GPUS` | cap on cards to occupy. On a shared node, taking every free card is antisocial; 2 or 3 is usually the neighbourly choice. |
| `CLG_PLAN_ONLY` | print the assignment and exit. |
| `CLG_NVIDIA_SMI` | path to `nvidia-smi` if it is not on `PATH`. |

Two things this cannot do. It reads free memory **once, at launch**, so a
neighbour who allocates afterwards can still push you into an OOM — if that
happens, raise `CLG_MIN_FREE_MIB` and resume. And each worker loads its own
copy of the checkpoint, so five workers means five ~35 GB allocations, which is
fine on 80 GB cards but is not free.

The larger speedup is unused: generation is batch size one, which wastes an
H100. Batching several prompts per pipeline call would give a further factor on
each card, but it changes what `signals.csv` records — the controller state and
its diagnostics become per batch rather than per prompt — so it is a real
change, not a flag.

Watch it, and check progress independently of the log:

~~~bash
tail -f "$CLG_PERSIST/logs/grid_w3.0.log"
find results/coco_test -name '*.png' | wc -l
kill "$(cat "$CLG_PERSIST/logs/grid.pid")"     # to stop early
~~~

If `tmux` or `screen` is available, a detached session is nicer than `nohup`:
you can reattach and watch the live output rather than tailing a file.

Whatever ends the run, the recovery is the same command again: `--resume` skips
what is already there.

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

### Scoring a run

~~~bash
python experiments/evaluate.py check --run results/pilot   # run this first
python experiments/evaluate.py clip  --run results/pilot
~~~

`check` catches the failure that is otherwise silent: if prompts are paired with
the wrong images, every score afterwards is still a plausible number. It scores
each image against its own prompt and against a neighbour's, and fails unless
the image prefers its own.

`clip` writes `clip_scores.csv` (per image) and `clip_summary.csv`, and prints
**paired** differences against the baseline arm: the same prompt and seed,
differenced, with a 95 % interval bootstrapped over prompts. Pairing matters
because prompt difficulty dominates the spread of CLIP score and would
otherwise hide the effect. A starred row is one whose interval excludes zero;
with a handful of prompts that interval is wide and a star is weak evidence.

FID is deliberately not implemented here. Its value depends on the Inception
weights, the resize and the sample count, so a local reimplementation gives a
number that is internally consistent but not comparable to any published table.
Use a standard tool, and note it needs thousands of images — on a pilot it is
dominated by sample-size bias:

~~~bash
pip install clean-fid
python -c "from cleanfid import fid; print(fid.compute_fid('results/run/paper/w3.0', 'data/reference/coco'))"
~~~

Evaluate held-out fidelity, alignment and diversity across guidance scales with
paired prompts and seeds, then compare at matched alignment and compute.
Lower chatter alone does not demonstrate better images, and neither does a
higher CLIP score on its own: attenuating guidance moves fidelity and alignment
together along one curve, so a law that only lowers the effective scale will
show a "better" number on whichever of the two axes you look at alone. Follow
the [improvement roadmap](docs/Improvement_Roadmap.md) for the comparison
protocol.
