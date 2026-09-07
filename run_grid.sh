#!/usr/bin/env bash
# Launch a guidance-scale sweep that outlives the terminal.
#
#     nohup ./run_grid.sh results/coco_test "3.0 2.0 4.5 1.5 7.0" \
#         --arms cfg paper excess --prompts data/prompts/test.txt --seeds 0 \
#         > /dev/null 2>&1 &
#
#     $1        output directory
#     $2        quoted list of guidance scales, in the order to run them
#     $3...     passed through to `real_model.py grid`
#
# One scale per invocation, deliberately. `cmd_grid` loops over arms on the
# outside, so a single call with every scale would finish all of `cfg` before
# starting `paper`: no complete slice of the curve until the very end. Running
# a scale at a time gives all arms at one scale in a couple of hours, which is
# enough to evaluate and decide whether to continue.
#
# --resume is always on, so re-running after an interruption costs nothing and
# `config.json` accumulates the scales rather than being overwritten.
set -uo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 OUT_DIR \"W1 W2 ...\" [args passed to real_model.py grid]" >&2
    exit 2
fi

OUT="$1"; SCALES="$2"; shift 2

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The environment is what most often goes wrong: a shell that never sourced
# vlab_env.sh has the container's python, no persistent HF cache, and no
# $CLG_PERSIST for the logs.
if [ -z "${CLG_PERSIST:-}" ]; then
    # shellcheck disable=SC1091
    source "$here/vlab_env.sh" || { echo "could not prepare the environment" >&2; exit 1; }
fi
cd "$here" || exit 1

# Logs default to logs/ inside the repository. The repository is itself on the
# mounted volume, so this is exactly as persistent as $CLG_PERSIST/logs, and it
# means `tail -f logs/...` works from where you already are, without depending
# on $CLG_PERSIST being set in the launching shell. Override with CLG_LOGS.
LOGS="${CLG_LOGS:-$here/logs}"
mkdir -p "$LOGS" "$OUT" || exit 1

# One sweep per output directory at a time. Two `grid` processes writing the
# same directory interleave rows in signals.csv and race on config.json.
LOCK="$OUT/.run_grid.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "another sweep is already writing $OUT (remove $LOCK if that is stale)" >&2
    exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

tag="$(basename "$OUT")"
echo "sweep      $OUT"
echo "scales     $SCALES"
echo "extra      $*"
echo "logs       $LOGS/${tag}_w<scale>.log"
echo "pid        $$"
echo "$$" > "$LOGS/${tag}.pid"

failed=0
for w in $SCALES; do
    log="$LOGS/${tag}_w${w}.log"
    started="$(date -Is)"
    echo "[$started] scale $w -> $log"
    # A subshell, not a { } group: the group's status would be the trailing
    # echo's, so every scale would look successful no matter what python did.
    (
        echo "=== $started  scale $w  args: $* ==="
        # -u because Python block-buffers stdout into a redirect, which makes a
        # working job look hung for minutes at a time.
        python -u experiments/real_model.py grid --w "$w" --out "$OUT" --resume "$@"
        rc=$?
        echo "=== exit $rc at $(date -Is) ==="
        exit $rc
    ) >> "$log" 2>&1
    status=$?
    if [ "$status" -ne 0 ]; then
        failed=$((failed + 1))
        echo "[$(date -Is)] scale $w FAILED (exit $status); see $log" >&2
        # Keep going: one scale failing (an OOM, a transient hub error) should
        # not throw away the scales that would still succeed.
    else
        echo "[$(date -Is)] scale $w done"
    fi
done

echo "finished; $failed of $(set -- $SCALES; echo $#) scale(s) failed"
[ "$failed" -eq 0 ]
