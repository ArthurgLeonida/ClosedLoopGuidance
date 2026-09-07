#!/usr/bin/env bash
# Run from the repository root, after preparing data/evaluator environments.
# nohup bash scripts/run_benchmarks.sh --gpus 0 1 > paper.nohup.log 2>&1 &
set -eo pipefail

benchmark_config=configs/paper.json
benchmark_out=results/paper_all
benchmark_gpus=()
while (($#)); do
    case "$1" in
        --config) benchmark_config="${2:?--config needs a path}"; shift 2 ;;
        --out) benchmark_out="${2:?--out needs a path}"; shift 2 ;;
        --gpus)
            shift
            while (($#)) && [[ "$1" != --* ]]; do
                benchmark_gpus+=("$1")
                shift
            done
            ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
if ((${#benchmark_gpus[@]} == 0)); then
    echo "Usage: bash scripts/run_benchmarks.sh --gpus 0 1 [--config configs/paper.json] [--out results/paper_all]" >&2
    exit 2
fi
for benchmark_gpu in "${benchmark_gpus[@]}"; do
    [[ "$benchmark_gpu" =~ ^[0-9]+$ ]] || { echo "GPU IDs must be visible CUDA indices" >&2; exit 2; }
done

# Uses the currently selected Python. Activate the generation environment
# before invoking it, or source vlab_env.sh inside the nohup shell.
benchmark_out="${benchmark_out%/}"
benchmark_logs="${benchmark_out}.logs"
mkdir -p "$benchmark_logs"
benchmark_workflow_lock="$benchmark_logs/.workflow.lock"
if ! mkdir "$benchmark_workflow_lock" 2>/dev/null; then
    echo "Workflow lock exists: $benchmark_workflow_lock. Check its PID before recovering a crashed run." >&2
    exit 1
fi
echo "$$" > "$benchmark_workflow_lock/pid"
benchmark_monitor_pid=""
benchmark_cleanup() {
    benchmark_exit=$?
    trap - EXIT
    if [[ -n "$benchmark_monitor_pid" ]]; then
        kill "$benchmark_monitor_pid" 2>/dev/null || true
        wait "$benchmark_monitor_pid" 2>/dev/null || true
    fi
    printf 'exit_code=%s\nfinished_at=%s\n' "$benchmark_exit" "$(date -Is)" > "$benchmark_logs/workflow.status"
    rm -f "$benchmark_workflow_lock/pid"
    rmdir "$benchmark_workflow_lock"
    exit "$benchmark_exit"
}
trap benchmark_cleanup EXIT
printf 'running_pid=%s\nstarted_at=%s\n' "$$" "$(date -Is)" > "$benchmark_logs/workflow.status"

benchmark_stage() {
    local benchmark_stage_name="$1"
    shift
    printf '\n[%s] %s\n' "$(date -Is)" "$benchmark_stage_name"
    "$@" 2>&1 | tee -a "$benchmark_logs/$benchmark_stage_name.log"
}

python -m pip list --format=json > "$benchmark_logs/generation_packages.json"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "$benchmark_logs/gpu_info.txt"
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv -l 10 >> "$benchmark_logs/gpu_usage.csv" 2>&1 &
    benchmark_monitor_pid=$!
fi

# Each metric worker sees one GPU, including legacy evaluators that hard-code
# cuda:0. Respect an inherited CUDA_VISIBLE_DEVICES mapping.
benchmark_eval_gpu="${benchmark_gpus[0]}"
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    IFS=',' read -r -a benchmark_visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
    benchmark_eval_gpu="${benchmark_visible_gpus[${benchmark_gpus[0]}]:?GPU index is outside CUDA_VISIBLE_DEVICES}"
    benchmark_eval_gpu="${benchmark_eval_gpu//[[:space:]]/}"
fi

benchmark_stage plan python -u -m cfgctrl.benchmark plan --config "$benchmark_config"
benchmark_stage doctor env CUDA_VISIBLE_DEVICES="$benchmark_eval_gpu" \
    python -u -m cfgctrl.benchmark doctor --config "$benchmark_config"
benchmark_stage generate python -u -m cfgctrl.benchmark generate --config "$benchmark_config" \
    --out "$benchmark_out" --gpus "${benchmark_gpus[@]}"
benchmark_stage diagnostics python -u -m cfgctrl.benchmark diagnostics --out "$benchmark_out"

# Always attempt a report after evaluation, so completed metrics and missing
# ones are visible even when a later scorer fails. Preserve a nonzero exit.
benchmark_evaluation_status=0
benchmark_stage evaluate env CUDA_VISIBLE_DEVICES="$benchmark_eval_gpu" \
    python -u -m cfgctrl.benchmark evaluate --config "$benchmark_config" --out "$benchmark_out" \
    || benchmark_evaluation_status=$?
benchmark_report_status=0
benchmark_stage report python -u -m cfgctrl.benchmark report --out "$benchmark_out" \
    || benchmark_report_status=$?
if ((benchmark_evaluation_status)); then
    exit "$benchmark_evaluation_status"
fi
exit "$benchmark_report_status"
