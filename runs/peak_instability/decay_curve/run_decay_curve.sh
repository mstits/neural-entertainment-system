#!/bin/zsh
# Decay-curve shape measurement: honest protocol across a checkpoint ladder
# for two v28 runs. Sequential, single eval-seed, 30 episodes — a shape
# probe, not a gate number. See runs/peak_instability/decay_curve/ for
# receipts.
set -e
cd /Users/stits/Documents/macos-emulation-and-training

OUT=runs/peak_instability/decay_curve
mkdir -p "$OUT"

START_STATE=runs/live_show/smb_4_4_micro/entrance_start.state

run_one () {
  local seed="$1"
  local iter="$2"
  local profile="configs/mario_1_1_v28_seed${seed}.yaml"
  local iter_padded=$(printf "%05d" "$iter")
  local ckpt="checkpoints/mario_1_1_v28_capacity_seed${seed}/vanilla_ppo_iter_${iter_padded}.pt"
  local out_json="${OUT}/seed${seed}_iter${iter_padded}.json"
  local out_log="${OUT}/seed${seed}_iter${iter_padded}.log"

  if [ ! -f "$ckpt" ]; then
    echo "MISSING CHECKPOINT: $ckpt" | tee -a "${OUT}/RUN_ERRORS.txt"
    return 1
  fi

  # Skip if we already have a good receipt from a prior (interrupted) pass.
  if [ -f "$out_json" ]; then
    local prior_st
    prior_st=$(.venv/bin/python -c "import json,sys
try:
    print(json.load(open('$out_json')).get('status','MISSING'))
except Exception:
    print('PARSE_FAIL')")
    if [ "$prior_st" = "ok" ]; then
      echo "seed${seed} iter${iter}: SKIP (already ok)" | tee -a "${OUT}/progress.log"
      return 0
    fi
  fi

  echo "=== seed${seed} iter${iter} : $(date -u +%FT%TZ) ===" | tee -a "${OUT}/progress.log"

  .venv/bin/python scripts/eval_game.py --game mario \
    --profile "$profile" \
    --checkpoint "$ckpt" \
    --start-state "$START_STATE" \
    --episodes 30 --max-steps 1500 --sticky-prob 0.25 --start-jitter 16 \
    --eval-seed 0 --action-select greedy --eval-workers 4 --eval-rng per-episode \
    > "$out_json" 2> "$out_log"

  local run_status
  run_status=$(.venv/bin/python -c "import json,sys
try:
    print(json.load(open('$out_json')).get('status','MISSING'))
except Exception:
    print('PARSE_FAIL')")

  if [ "$run_status" != "ok" ]; then
    echo "BAD STATUS seed${seed} iter${iter}: $run_status" | tee -a "${OUT}/RUN_ERRORS.txt"
    cat "$out_json" | tee -a "${OUT}/RUN_ERRORS.txt"
  else
    local cr
    cr=$(.venv/bin/python -c "import json; print(json.load(open('$out_json'))['clear_rate'])")
    echo "seed${seed} iter${iter}: clear_rate=${cr}" | tee -a "${OUT}/progress.log"
  fi
}

# seed3 ladder: peak=90 (winners/best.json) -> includes peak-10=80, peak+10=100
SEED3_ITERS=(20 50 70 80 90 100 120 160 200 240)
# seed0 ladder: peak=70 (winners/best.json) -> includes peak-10=60, peak+10=80
SEED0_ITERS=(20 50 60 70 80 90 120 160 200 240)

for it in "${SEED3_ITERS[@]}"; do
  run_one 3 "$it"
done

for it in "${SEED0_ITERS[@]}"; do
  run_one 0 "$it"
done

echo "ALL DONE: $(date -u +%FT%TZ)" | tee -a "${OUT}/progress.log"
