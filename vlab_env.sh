#!/usr/bin/env bash
# Persist the environment and model caches across container relaunches.
#
#     source vlab_env.sh
#
# Why this is needed. `conda create -n NAME` places the environment inside the
# base conda installation -- on a Jupyter image that is /root/anaconda3/envs --
# which belongs to the container layer and is discarded when the container
# restarts. Only the mounted volume survives. The same applies to the Hugging
# Face cache, which defaults to $HOME/.cache and holds tens of gigabytes of
# checkpoints. This script keeps the environment, the conda package cache, the
# pip cache and the model cache on the mounted volume, so a relaunch costs one
# `source` instead of a rebuild and a re-download.
#
# Knobs, all optional:
#   CLG_PERSIST      directory that survives a relaunch (default: repo's parent)
#   CLG_ENV          where to put the environment (default: $CLG_PERSIST/envs/clg)
#   CLG_TORCH_INDEX  PyTorch wheel index; must match the driver's CUDA version

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "source this script, do not execute it:  source ${0}" >&2
    exit 1
fi

CLG_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLG_PERSIST="${CLG_PERSIST:-$(dirname "$CLG_REPO")}"
CLG_ENV="${CLG_ENV:-$CLG_PERSIST/envs/clg}"
# `nvidia-smi` reports the driver's CUDA version; the wheel must not be newer.
CLG_TORCH_INDEX="${CLG_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

if [ ! -w "$CLG_PERSIST" ]; then
    echo "CLG_PERSIST=$CLG_PERSIST is not writable; set it to your mounted volume" >&2
    return 1
fi

export HF_HOME="$CLG_PERSIST/hf"
export CONDA_PKGS_DIRS="$CLG_PERSIST/conda/pkgs"
export PIP_CACHE_DIR="$CLG_PERSIST/pip-cache"
mkdir -p "$HF_HOME" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$(dirname "$CLG_ENV")" || return 1

__clg_base="$(conda info --base 2>/dev/null)"
if [ -z "$__clg_base" ]; then
    echo "conda is not on PATH" >&2
    return 1
fi
# A non-interactive shell has never run `conda init`, so load the hook first.
# shellcheck disable=SC1091
. "$__clg_base/etc/profile.d/conda.sh" || return 1

if [ ! -d "$CLG_ENV" ]; then
    echo "creating $CLG_ENV  (first run only; several minutes)"
    conda create -y -p "$CLG_ENV" python=3.11 || return 1
    conda activate "$CLG_ENV" || return 1
    python -m pip install --upgrade pip || return 1
    # torch first, from an index matching the driver. requirements.txt lists an
    # unpinned torch, so the later install leaves this build in place.
    python -m pip install torch --index-url "$CLG_TORCH_INDEX" || return 1
    python -m pip install -r "$CLG_REPO/requirements.txt" || return 1
    python -m pip freeze > "$CLG_ENV/requirements-resolved.txt"
    echo "recorded exact versions in $CLG_ENV/requirements-resolved.txt"
else
    conda activate "$CLG_ENV" || return 1
fi

echo "env      $CONDA_PREFIX"
echo "HF_HOME  $HF_HOME"
python - <<'PY'
import torch
print(f"torch    {torch.__version__}, built for CUDA {torch.version.cuda}, "
      f"cuda available {torch.cuda.is_available()}")
PY
