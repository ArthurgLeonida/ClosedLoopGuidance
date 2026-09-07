#!/usr/bin/env bash
# Score a finished grid: alignment, then fidelity, then the join that ranks.
#
#     ./run_eval.sh results/coco_test
#     CLG_REF=data/reference/matched ./run_eval.sh results/coco_test
#
#     $1        a `real_model.py grid` output directory
#     $2...     passed through to `evaluate.py clip`
#
# The order is not cosmetic. `check` runs first and aborts on failure because
# the failure that matters here is silent: if prompts are matched to the wrong
# images, every later number is still a plausible-looking float. Then alignment
# (CLIP) and fidelity (FID/KID) are measured separately, and only the join
# decides anything -- neither axis ranks a guidance law alone, because weaker
# guidance buys fidelity with alignment along CFG's own curve.
#
#     CLG_REF     real images for FID   (default data/reference/val2017)
#     CLG_DEVICE  torch device          (default cuda)
#     CLG_LOGS    log directory         (default ./logs)
set -uo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 RUN_DIR [args passed to evaluate.py clip]" >&2
    exit 2
fi
OUT="$1"; shift

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${CLG_PERSIST:-}" ]; then
    # shellcheck disable=SC1091
    source "$here/vlab_env.sh" || { echo "could not prepare the environment" >&2; exit 1; }
fi
cd "$here" || exit 1

REF="${CLG_REF:-data/reference/val2017}"
DEV="${CLG_DEVICE:-cuda}"
LOGS="${CLG_LOGS:-$here/logs}"
mkdir -p "$LOGS" || exit 1
LOG="$LOGS/$(basename "$OUT")_eval.log"

# Scoring rewrites clip_scores.csv, fid.csv and pareto.csv in place; two of
# these at once would interleave.
LOCK="$OUT/.run_eval.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "another evaluation is already writing $OUT (remove $LOCK if stale)" >&2
    exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

echo "run        $OUT"
echo "reference  $REF"
echo "device     $DEV"
echo "log        $LOG"

step() {                      # step "name" cmd...
    local name="$1"; shift
    echo "" | tee -a "$LOG"
    echo "=== $name  $(date -Is) ===" | tee -a "$LOG"
    "$@" 2>&1 | tee -a "$LOG"
    return "${PIPESTATUS[0]}"
}

step "check" python -u experiments/evaluate.py check --run "$OUT" --device "$DEV" || {
    echo "check FAILED: the prompt/image pairing is wrong, so every score below" >&2
    echo "would be meaningless. Nothing else was run." >&2
    exit 1
}

step "clip" python -u experiments/evaluate.py clip --run "$OUT" --device "$DEV" "$@" \
    || echo "WARNING: CLIP scoring failed; the Pareto join needs it" >&2

if [ -d "$REF" ]; then
    step "fid" python -u experiments/fid.py compute --run "$OUT" --reference "$REF" \
        --device "$DEV" --resume \
        || echo "WARNING: FID failed; see $LOG" >&2
else
    echo "" | tee -a "$LOG"
    echo "no reference images at $REF, so fidelity was skipped. Get them with:" | tee -a "$LOG"
    echo "  curl -L -o data/reference/val2017.zip http://images.cocodataset.org/zips/val2017.zip" | tee -a "$LOG"
    echo "  unzip -q data/reference/val2017.zip -d data/reference/" | tee -a "$LOG"
fi

if [ -f "$OUT/clip_summary.csv" ] && [ -f "$OUT/fid.csv" ]; then
    step "pareto (FID)" python -u experiments/pareto.py --run "$OUT" --fid "$OUT/fid.csv"
    step "pareto (KID)" python -u experiments/pareto.py --run "$OUT" --fid "$OUT/fid.csv" \
        --fid-column kid
fi

step "signals" python -u experiments/signals.py --run "$OUT"

echo ""
echo "full log: $LOG"
