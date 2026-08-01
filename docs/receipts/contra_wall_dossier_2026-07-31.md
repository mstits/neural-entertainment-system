# Contra stage-1 base wall — status dossier (2026-07-31 evening)

## What is receipted
- Wall = fixed-camera fight at progress 3072 (screen 12); progress-invisible
  to coverage (v2: 2h/20.3M steps/313k cells, frontier never moved).
- Real per-slot HP array at $04BF-$04C2 (causal: fire dropped a slot 3->0,
  slot emptied). Guns die to ordinary fire.
- $0311-$0314 are ANIMATION/PHASE CODES, not stable type IDs (codes cycle
  30<->44<->54<->68<->69/70 on live objects; the "wall-resident types"
  receipt and therefore boss_typed's hp ladder are PARTLY phase-aliasing —
  "hp 12" cells are no-typed-object-live fallbacks, "hp 2" cells are
  single-object-in-typed-phase moments).
- The final object survives everything tried: 4.5h of solver (v4-v6,
  coverage/kill/typed-hp/time-bins arms) + scripted sweeps from banked
  states (prone fire 12x120, standing fire 30x300, facing-controlled
  stationary fire 40x400): zero transitions.
- Screenshots (/tmp/contra_wall_*.png): the core is the X-plated circle,
  low on the wall; the plate is armor — the core presumably takes damage
  only during a periodic open phase.

## Next levers (designed, not blind)
1. Pixel-phase mining (Observatory v2 / v9 pixel modality): render the
   core's screen region across a cycle from banked wall states, find the
   RAM byte whose value tracks the open/closed pixel phase -> the
   open-window observable -> cells keyed on (core hp, open-phase) and/or
   fire timed to open windows.
2. Slot-identity fix: find the per-slot stable ID array (the code bytes
   cycle; some parallel array should hold the object class) by diffing
   slot columns across a single object's lifetime in one trace.
3. Once core HP is truly keyed: boss dimension on it; expect the v5-style
   grind to finish the job.

## Runs
- v2 coverage control: stage1_baseline_collapsed_cells (bad observables),
  stage1_v2 (clean control)
- v3 cumulative kk (saturated): stage1_v3_kk_saturated
- v4 local kk: stage1_v4_localkk
- v5 boss_typed: stage1_v5_bosstyped
- v6 resume + time-bins: stage1_v6_resume (650k cells)
