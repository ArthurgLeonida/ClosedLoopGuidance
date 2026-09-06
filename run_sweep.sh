#!/usr/bin/env bash
# Spread a guidance-scale sweep across whichever GPUs are actually free.
#
#     nohup ./run_sweep.sh results/coco "3.0 2.0 4.5 1.5 7.0" \
#         --arms cfg paper excess --prompts data/prompts/test.txt --seeds 0 \
#         > sweep.out 2>&1 &
#
#     $1        output prefix; each scale gets its own "<prefix>_w<scale>"
#     $2        quoted list of guidance scales
#     $3...     passed through to `real_model.py grid`
#
# One output directory per scale, deliberately. Two processes writing one
# directory would interleave rows in signals.csv and race on config.json, so
# `run_grid.sh` takes a lock that would refuse the second job anyway. Separate
# directories are self-contained: `evaluate.py check/clip` works on each, and
# the Pareto curve is assembled across them.
#
# Knobs:
#   CLG_MIN_FREE_MIB   memory a GPU must have free to be used (default 45000;
#                      SD3.5-large in bf16 measured ~35 GB, plus headroom)
#   CLG_MAX_GPUS       cap on GPUs to occupy, 0 = no cap (default 0)
#   CLG_PLAN_ONLY      1 = print the assignment and exit without launching
set -uo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 OUT_PREFIX \"W1 W2 ...\" [args passed to real_model.py grid]" >&2
    exit 2
fi

PREFIX="$1"; SCALES="$2"; shift 2
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MIN_FREE_MIB="${CLG_MIN_FREE_MIB:-45000}"
MAX_GPUS="${CLG_MAX_GPUS:-0}"
PLAN_ONLY="${CLG_PLAN_ONLY:-0}"
LOGS="${CLG_LOGS:-$here/logs}"
mkdir -p "$LOGS" || exit 1

# Overridable so it can be pointed at a non-standard path, and stubbed in tests.
NVSMI="${CLG_NVIDIA_SMI:-nvidia-smi}"
command -v "$NVSMI" >/dev/null 2>&1 || { echo "$NVSMI not found" >&2; exit 1; }

# Free memory now, not total: this is a shared node and other jobs hold cards.
avail=()
while IFS=, read -r idx free; do
    idx="${idx//[[:space:]]/}"; free="${free//[[:space:]]/}"
    case "$idx$free" in *[!0-9]*|"") continue;; esac
    if [ "$free" -ge "$MIN_FREE_MIB" ]; then
        avail+=("$idx")
    else
        echo "  gpu $idx: ${free} MiB free -- skipping (need $MIN_FREE_MIB)"
    fi
done < <("$NVSMI" --query-gpu=index,memory.free --format=csv,noheader,nounits)

if [ "${#avail[@]}" -eq 0 ]; then
    echo "no GPU has $MIN_FREE_MIB MiB free; lower CLG_MIN_FREE_MIB or wait" >&2
    exit 1
fi
if [ "$MAX_GPUS" -gt 0 ] && [ "${#avail[@]}" -gt "$MAX_GPUS" ]; then
    avail=("${avail[@]:0:$MAX_GPUS}")
fi

n_scales=$(set -- $SCALES; echo $#)
echo "usable gpus  ${avail[*]}  (${#avail[@]})"
echo "scales       $SCALES  ($n_scales)"
echo "prefix       ${PREFIX}_w<scale>"
echo "logs         $LOGS"

# Round-robin. Every scale costs the same (same arms, prompts and seeds), so a
# static split is within one scale of optimal and needs no coordination.
declare -a plan
slot=0
for w in $SCALES; do
    gpu="${avail[$(( slot % ${#avail[@]} ))]}"
    plan+=("$gpu:$w")
    slot=$((slot + 1))
done

for gpu in "${avail[@]}"; do
    mine=""
    for entry in "${plan[@]}"; do
        [ "${entry%%:*}" = "$gpu" ] && mine="$mine ${entry#*:}"
    done
    [ -n "$mine" ] && echo "  gpu $gpu ->$mine"
done

if [ "$PLAN_ONLY" = "1" ]; then
    echo "CLG_PLAN_ONLY=1: nothing launched"
    exit 0
fi

tag="$(basename "$PREFIX")"
for gpu in "${avail[@]}"; do
    mine=()
    for entry in "${plan[@]}"; do
        [ "${entry%%:*}" = "$gpu" ] && mine+=("${entry#*:}")
    done
    [ "${#mine[@]}" -eq 0 ] && continue
    (
        for w in "${mine[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpu" "$here/run_grid.sh" "${PREFIX}_w${w}" "$w" "$@" \
                > "$LOGS/${tag}_w${w}.out" 2>&1
            echo "[$(date -Is)] gpu $gpu finished scale $w (exit $?)"
        done
    ) &
done

echo "launched ${#avail[@]} worker(s); waiting"
wait
echo "[$(date -Is)] sweep complete"
