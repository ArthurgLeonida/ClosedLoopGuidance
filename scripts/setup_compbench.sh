#!/usr/bin/env bash
# Dedicated Linux/H100 evaluator environment; invoke after sourcing vlab_env.sh.
set -eo pipefail
cd "$(dirname "$0")/.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export MAX_JOBS="${MAX_JOBS:-4}"
export FORCE_CUDA=1
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export TORCH_HOME="${TORCH_HOME:-${CLG_PERSIST:?Source vlab_env.sh first}/torch}"
comp_env="${CLG_COMPBENCH_ENV:-$CLG_PERSIST/envs/clg-compbench}"
test "$(realpath -m "$comp_env")" != "$(realpath -m "$CLG_ENV")" ||
    { echo "CompBench must use a separate environment from generation." >&2; exit 1; }
comp_repo="$PWD/external/T2I-CompBench"
comp_revision=4aa404212eb5d06e5adbcd9cee696c750d0d25a5
"$CUDA_HOME/bin/nvcc" --version | tee /dev/stderr | grep -q 'release 12.4,' ||
    { echo "This installer requires the CUDA 12.4 toolkit." >&2; exit 1; }
command -v "$CXX"
command -v conda
test "$(git -C "$comp_repo" rev-parse HEAD)" = "$comp_revision" ||
    { echo "Prepare external/T2I-CompBench at $comp_revision first." >&2; exit 1; }
mkdir -p results/compbench_setup "$TORCH_HOME"
exec 9>results/compbench_setup/install.lock
flock -n 9 || { echo "Another CompBench setup is running." >&2; exit 1; }
if [ ! -x "$comp_env/bin/python" ]; then
    conda create -y -p "$comp_env" python=3.11 pip
fi
comp_python="$comp_env/bin/python"
"$comp_python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use a dedicated Python 3.11 environment"'
"$comp_python" -m pip install 'pip<26' setuptools==75.8.0 wheel==0.45.1 ninja
"$comp_python" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
"$comp_python" -m pip install -r requirements/compbench.txt
"$comp_python" -m pip install --no-build-isolation \
    'git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13'
"$comp_python" -m pip check
"$comp_python" -m pip freeze > results/compbench_setup/packages.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$comp_python" -u scripts/check_compbench.py --repo "$comp_repo" --out results/compbench_setup
echo COMPBENCH_SETUP_COMPLETE
