# SMB World 1 — Autonomous RL Training

Record of training agents to play through *Super Mario Bros.* World 1
entirely by self-play — **no human demonstrations**. The trainer is the
unified `vanilla_ppo` path (single policy, N parallel envs, batched GAE,
K-epoch PPO) on the RAM-decoded tile encoder (`smb_tiles`, 175-dim × 4
frame-stack). Per-level dense reward ladders live in
`nes_core/src/rewards.rs` (`MarioReward`), keyed to the **area byte**
(`$0760`); the trainer warm-starts each level from a captured save-state.

## Results

| Level | Area byte | Status | How |
|------|-----------|--------|-----|
| 1-1 | 0 | ✅ greedy clear | vanilla_ppo + tile policy |
| 1-2 (underground) | 2 | ✅ greedy clear | area-byte ladder fix + entropy-decay consolidation |
| 1-3 | 3 | ✅ greedy clear | seed-transfer from the 1-1 winner (running long-jump) |
| 1-4 (castle) | 4 | ⚠️ reachable, not consolidated | dense platform/bridge ladder; crosses to World 2 ~10% but oscillates |

Preserved policies (`checkpoints/winners/`):
`smb_1-1_greedy_clear_iter70.pt`, `smb_1-2_underground_greedy_clear_iter370.pt`,
`smb_1-3_greedy_clear.pt`, `smb_1-4_best_iter560.pt`.

Greedy-eval any of them warm-started at the matching curriculum state:
```bash
python scripts/eval_game.py --game mario --checkpoint checkpoints/winners/<name>.pt
```

## What worked (transferable lessons)

- **Area-byte semantics.** A displayed level can span multiple area
  bytes; 1-2 = area 1 (entrance) + area 2 (underground). The dense
  checkpoint ladder must key off `$0760` and be mapped to the area byte
  the obstacles actually live on. The original `LEVEL_1_2` table was
  wired to area 1 (the short entrance), leaving the underground main with
  zero dense signal — fixing the mapping was what unlocked 1-2. Confirm
  semantics empirically via `$075C` (display level), not the approximate
  `AREA_TO_LEVEL` log helper.
- **Seed-transfer: the skill has to come from a policy that already has
  it.** 1-3's wide gap needs a running long-jump. Seeding from the 1-2
  *underground* winner (a floored corridor that atrophied the jump) stalled
  every time; re-seeding from the **1-1 winner** (which clears 1-1's pits
  with that exact jump) transferred the skill and cleared 1-3 — *without*
  any demonstration.
- **Jump-shaping vs camping on platform levels.** Platform levels
  (1-3) need `air_bonus` + `jump_clear_bonus` (reward productive jumps),
  and `survival_bonus` must be **off** — on a platform level "survive"
  means camp on a safe ledge, a local optimum that competes with the
  risky jumps needed to advance. (`survival_bonus` is fine on floored
  levels like 1-1 where surviving == progressing.)
- **Entropy-decay consolidation.** When the stochastic policy *finds* a
  crossing but greedy lags (1-2), decaying `entropy_coef` (e.g.
  0.01 → 0.002) sharpens the argmax onto the found behavior — converting
  a stochastic clear into a deployable greedy one.
- **Dead-zone bridges.** A reward-silent span between checkpoints
  (>~400 px) leaves PPO without a gradient across an obstacle; a thin
  intermediate checkpoint makes "make progress past it" detectable.
- **Judge by eval, not the metric.** `vanilla_ppo_max_x` was originally
  a *cumulative* running-max (a missing per-iter reset) that preserved an
  old peak and masked a policy collapse. Fixed to per-iter; still, always
  confirm the *current* policy by evaluating its checkpoint (median over
  envs + greedy), never the max-over-envs metric, which overstates.

## 1-4 — the autonomous ceiling

The castle is reachable but not consolidatable with this setup. The
trajectory: x814 is a jump-up onto an elevated platform-hop; the final
~2430→2560 stretch is the Bowser bridge. A dense ladder through both got
the agent **across into World 2** — but only ~10–17% of training iters
had a crossing env (~0.5%/episode), and the policy **oscillates** between
rare crossings and collapse to the x814 safe-stall (a **risk-aversion**
optimum: the high-variance bridge loses to the low-variance stall in the
value estimate). It never stabilizes, so the crossing stays too rare to
consolidate to greedy.

Levers tried and exhausted: full-level training, anti-collapse entropy,
seed-transfer, **within-level curriculum** (fails outright — cold
mid-castle warm-starts are *death-traps* because the hazards are timed
relative to the approach), anti-stall time penalty, dense reward, denser
bridge, lower learning rate. The remaining lever is a **recurrent (GRU)
policy** for the non-Markovian firebar/bridge timing (`recurrent: true`,
see `configs/smb_1_4_recurrent.yaml`) — a from-scratch undertaking
(a GRU can't seed from the feed-forward winners) with uncertain payoff
(recurrence hurt the Markovian levels).

## Reproduce

Per-level configs: `configs/mario_vanilla_ppo.yaml` (1-1),
`configs/smb_underground_dense.yaml` (1-2 underground),
`configs/smb_1_3.yaml` (1-3), `configs/smb_1_4_dense.yaml` (1-4).
Curriculum save-states: `checkpoints/super_mario_bros/smb_curriculum/`.

```bash
# e.g. train 1-3 (jump-shaping, seeded from the 1-1 winner):
cp checkpoints/winners/smb_1-1_greedy_clear_iter70.pt \
   checkpoints/mario_1_3/vanilla_ppo_iter_00070.pt
python scripts/train_game.py --profile configs/smb_1_3.yaml \
   --rom "roms/Super Mario Bros. (World).nes" --num-envs 32 --iters 350
```

A human-recorded demo path exists as a shelved fallback
(`scripts/record_demo.py` + `scripts/bc_pretrain.py --start-state`) but is
intentionally unused — the goal is fully autonomous learning.
