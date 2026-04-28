#!/usr/bin/env bash
# Automated ablation harness for Zelda. Each variant runs headless in its
# own --checkpoint-dir; their metrics.jsonl files are then merged into one
# chart via scripts/compare_runs.py.
#
# Variants:
#   1. baseline     — full hybrid (BC + GA + PPO) with retuned reward
#   2. no_items     — remove item-related rewards (tests item signals)
#   3. no_explore   — remove exploration bonus (tests exploration signal)
#   4. motion_zero  — force motion=0 (already default, sanity check)
#   5. aggressive   — exploration/items bumped, higher entropy
#
# Usage:
#   ./scripts/run_ablations.sh                 # default 30 gens
#   ./scripts/run_ablations.sh 50 8 16 1500    # gens/instances/pop/max-steps
set -euo pipefail

cd "$(dirname "$0")/.."

GENS=${1:-30}
INSTANCES=${2:-8}
POP=${3:-16}
MAX_STEPS=${4:-1500}
ROM=${ROM:-roms/zelda.nes}
PROFILE=${PROFILE:-configs/zelda.yaml}
STATE=${STATE:-roms/zelda.ines1_start.state.bin}

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: activate the venv first: source .venv/bin/activate" >&2
    exit 1
fi

OUT=./checkpoints_ablation_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
echo "==> Ablations → $OUT  (gens=$GENS, instances=$INSTANCES, pop=$POP, max_steps=$MAX_STEPS)"

run_variant() {
    local name=$1; shift
    local dir="$OUT/$name"
    mkdir -p "$dir"
    echo "==> [$name] starting"
    python -m src.training.trainer \
        --rom "$ROM" \
        --profile "$PROFILE" \
        --start-state "$STATE" \
        --bc-demo "$STATE" \
        --instances "$INSTANCES" \
        --population "$POP" \
        --generations "$GENS" \
        --max-episode-steps "$MAX_STEPS" \
        --checkpoint-dir "$dir" \
        --seed 42 \
        "$@" \
        2>&1 | tail -5
    echo "==> [$name] done"
}

run_variant baseline
run_variant no_items      --reward-overrides configs/overrides/zelda_no_items.yaml
run_variant no_explore    --reward-overrides configs/overrides/zelda_no_exploration.yaml
run_variant motion_zero   --reward-overrides configs/overrides/zelda_motion_zero.yaml
run_variant aggressive    --reward-overrides configs/overrides/zelda_aggressive_exploration.yaml

echo ""
echo "==> All variants done. Merging metrics:"
python scripts/compare_runs.py "$OUT"/*/metrics.jsonl
echo ""
echo "Charts saved as $OUT/comparison.png"
