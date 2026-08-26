#!/bin/bash
# Protocol ablation: is late-training collapse a stickiness-robustness
# failure or genuine capability loss? v28 seed3 (best-of-8 run): peak
# checkpoint (honest 0.670) vs final checkpoint (honest 0.000), crossed
# with sticky_prob {0.0, 0.25} x action_select {greedy, sampled}, holding
# start_state (cold entrance), start_jitter=16 and eval_seed=0 fixed at
# the honest protocol's values. 30 episodes, single eval seed: a SHAPE
# probe, not a gate number. Run strictly sequentially -- shared machine.
set -euo pipefail
cd /Users/stits/Documents/macos-emulation-and-training

OUT=runs/peak_instability/protocol_ablation
PROFILE=configs/mario_1_1_v28_seed3.yaml
ROM="roms/Super Mario Bros. (World).nes"
START_STATE=runs/live_show/smb_4_4_micro/entrance_start.state
PEAK_CKPT=checkpoints/mario_1_1_v28_capacity_seed3/winners/best.pt
FINAL_CKPT=checkpoints/mario_1_1_v28_capacity_seed3/vanilla_ppo_iter_00240.pt

run_cell () {
  local tag="$1" ckpt="$2" sticky="$3" select="$4"
  local out_json="$OUT/${tag}.json"
  local out_log="$OUT/${tag}.log"
  echo "=== $tag : ckpt=$ckpt sticky=$sticky select=$select ===" | tee -a "$OUT/manifest.log"
  .venv/bin/python scripts/eval_game.py \
    --game mario \
    --profile "$PROFILE" \
    --rom "$ROM" \
    --checkpoint "$ckpt" \
    --episodes 30 \
    --max-steps 1500 \
    --start-state "$START_STATE" \
    --sticky-prob "$sticky" \
    --start-jitter 16 \
    --eval-seed 0 \
    --action-select "$select" \
    --eval-workers 4 \
    --eval-rng per-episode \
    > "$out_json" 2> "$out_log"
  status=$(.venv/bin/python -c "import json,sys; print(json.load(open('$out_json')).get('status','MISSING'))" 2>/dev/null || echo "PARSE_ERROR")
  echo "    -> status=$status  ($out_json)" | tee -a "$OUT/manifest.log"
}

for ckpt_pair in "peak:$PEAK_CKPT" "final:$FINAL_CKPT"; do
  ckpt_tag="${ckpt_pair%%:*}"
  ckpt_path="${ckpt_pair#*:}"
  for sticky in 0.0 0.25; do
    for select in greedy sampled; do
      tag="${ckpt_tag}_sticky${sticky}_${select}"
      run_cell "$tag" "$ckpt_path" "$sticky" "$select"
    done
  done
done

echo "done" | tee -a "$OUT/manifest.log"
