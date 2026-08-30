# Full Technical Dossier: Six Months of Learning Super Mario Bros (NES) — Architecture, Every Approach, Every Wall

**For:** a deep-research investigation. This is the complete record — system
architecture with code, the full chronological history of what we tried and why
each failed or succeeded, the recurring failure taxonomy, what is ruled out (with
evidence), and the precise open problem. Companion briefs (read after):
`LEARNED_STICKY_DISTILLATION_BRIEF_2026-07-22.md` (v1) and `..._v2_...md` (the
failure of the first proposed solution).

**One-paragraph orientation.** We have a fast, deterministic Rust NES emulator
and a PyTorch RL stack. Over ~6 months we got model-free PPO to *genuinely learn*
SMB World 1-1 (63–67% under the honest stochastic-eval protocol) and to
*search-solve* (Go-Explore) 15 of 32 levels. The unsolved science: converting a
search-found solution for a **long** level (1-2, ~787 action-steps) into a
**learned, closed-loop policy robust to 25% sticky-action noise** evaluated from
the level entrance. Four independent method families all fail this. The most
recent (a deep-research-proposed recipe: recurrent GRU + wavefront PBRS + DART
recovery + staged BC→PPO) also failed, in a newly diagnostic way (§6, and v2
brief).

---

# PART I — ARCHITECTURE (with code)

## I.1 Hardware & execution model
- **Machine:** one Apple M4 (12 P-cores + 4 E-cores = 16 logical), 128 GB unified
  memory, latest macOS. "The agent" = this one machine.
- **Emulator:** cycle-accurate NES core in Rust (`nes_core`), pyo3-bound.
  Deterministic; **microsecond full-state save/restore**; headless; parallel
  worker `Pool`, rayon-parallel with the GIL released. `pool.step_all(actions)`
  returns per-worker `(frame_240x256x3, preprocessed_84x84, ram_2048_bytes, done)`.
- **Apple-Silicon reality (measured, load-bearing):** small nets are
  **dispatch-latency-bound on MPS** (25–80 µs GPU driver dispatch/op vs <1 µs
  compute) → the ~48k-param tile MLP runs **~3000–4400 env-steps/s on CPU** vs
  ~420–580 on MPS. We train tile policies on **`device: cpu`**. `torch.compile`
  on MPS is unreliable. MPS only wins for large batched CNN/RSSM.

## I.2 Observation: "tile" mode (`smb_tiles_pos`)
RAM-decoded, not pixels. A 13×13 tile grid around Mario + scalar position/state
features = **178 features**, 4-frame stacked = **712-dim** input. (A pixel CNN
path, Nature-DQN 84×84×4 ~1.7M params, also exists but is ~10× slower per rollout
on this hardware; tile is the workhorse.) Key RAM addresses used throughout:
```
x_abs   = (RAM[0x006D] << 8) | RAM[0x0086]      # horizontal position
y        = RAM[0x00CE]                            # vertical
world    = RAM[0x075F]   level(displayed) = RAM[0x075C]   area = RAM[0x0760]
lives    = RAM[0x075A]   player_state = RAM[0x000E]  (6 or 11 = dying)
```

## I.3 Policy networks (`src/models/tile_policy.py`)
Intentionally tiny (single-level-tuned defaults). Feedforward:
```python
class TilePolicyNetwork(nn.Module):
    def __init__(self, num_actions, feature_dim=175, hidden_dim=64, trunk_dim=32):
        self.fc1 = nn.Linear(feature_dim, hidden_dim)   # 712 -> 64
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, trunk_dim)     # 64 -> 32  (the bottleneck)
        self.norm2 = nn.LayerNorm(trunk_dim)
        self.actor = nn.Linear(trunk_dim, num_actions)  # 6 actions
        self.critic = nn.Linear(trunk_dim, 1)
        # ~48k params at 712/64/32. SiLU activations, orthogonal init.
```
Recurrent (for POMDP levels), reused unchanged by the recent recipe:
```python
class TileRecurrentPolicyNetwork(nn.Module):
    def __init__(self, num_actions, feature_dim=175, hidden_dim=64, gru_dim=32):
        self.input_proj = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                         nn.LayerNorm(hidden_dim), nn.SiLU())
        self.gru = nn.GRUCell(hidden_dim, gru_dim)
        self.norm_h = nn.LayerNorm(gru_dim)
        self.actor = nn.Linear(gru_dim, num_actions); self.critic = nn.Linear(gru_dim, 1)
        # forward_ac_recurrent(x, h) + initial_hidden(); ~112k at 712/128/64.
```
Checkpoints round-trip the widths (`to_dict`/`from_dict`), and
`build_tile_policy_from_checkpoint` infers widths from weight shapes so eval
rebuilds any size.

## I.4 Action space & reward
6 actions: `[noop], [right], [right+A], [right+B], [right+A+B], [A]`; frame_skip 4.
Generic per-level reward (computed in Rust `MarioReward`, folded in the trainer):
```yaml
reward_weights: {forward_progress: 1.0, time_penalty: -0.01,
                 death_penalty: -15.0, completion_bonus: 50.0, ...}  # dense knobs default 0
```
`forward_progress` = Δ(max x). `completion_bonus` fires on the flag. We learned
(the hard way, §II) that **dense per-step shaping DILUTES the terminal clear
signal — the agent farms the cheap dense reward** — so the proven 1-1 recipe uses
only forward+completion (shaping OFF).

## I.5 Trainer (`src/training/trainer.py`, `_run_vanilla_ppo`)
Model-free PPO: GAE-λ, Huber value loss (value_coef 0.25), symlog rewards, 60
parallel envs, entropy coef with an adaptive floor controller, optional two-head
RND. Sticky-actions applied *in training* to match eval. Multiple curriculum
modes coexist (all gated): staged save-state `smb_curriculum`, `substage_ladder`,
`consolidate_level`, `plr` (prioritized level replay), `go_explore` restart. A
recurrent PPO path (`_recurrent_ppo_update`, truncated BPTT, per-env hidden reset
on `done`). Per-step reward (the wavefront-shaping injection point):
```python
reward, rew_done, level_id = reward_fns[i].compute(ram, action=...)
done = bool(r.done) or bool(rew_done)
if wave_pot is not None:                          # PBRS shaping (recent recipe)
    _phi = wave_pot.potential(ram)
    if wave_prev_phi[i] is not None and not done:
        reward += wave_pot.gamma * _phi - wave_prev_phi[i]
    wave_prev_phi[i] = None if done else _phi
reward_buf[t, i] = reward
```

## I.6 The HONESTY PROTOCOL (the only headline metric)
Machado et al. 2018: cold, **greedy (argmax)**, from the level entrance, under
**25% sticky-actions + no-op start-jitter (0–16 frames)**, single life, ≥2 seeds.
"Cleared" = warp-guarded forward level transition (a warp-zone pipe does NOT
count). Binding rule set by the project owner (see §II, 2026-07-19):
**no game-internals cheating** — no disassembly, level maps, or hand-scripted
inputs; training-time state restoration + search-as-teacher ARE allowed, but the
*deployed/claimed* agent must be a learned policy judged by this protocol.
```bash
python scripts/eval_game.py --game mario --profile <p> --checkpoint <ckpt> \
   --sequential --level-clear --sticky-prob 0.25 --start-jitter 16 --eval-seed {0,1}
```

## I.7 Go-Explore (`src/training/go_explore.py`, `scripts/go_explore_solve.py`)
First-return-then-explore over a cell archive (µs restore = free returns).
Honest solver: roots ONLY at the level entrance, no hand-crafted seeds,
warp-guarded clear, lives-based death, dumps replayable action traces.
```python
def cell_fn(ram):   # spatial-ish discretization
    return (int(ram[0x0760]), (int(ram[0x0009])>>2)&7, int(ram[0x00CE])//32,
            ((int(ram[0x006D])<<8)|int(ram[0x0086]))//16)   # (area, step-phase, y-band, gx//16)
```

---

# PART II — THE SIX-MONTH HISTORY (chronological, with issues)

### Phase A — Making PPO learn at all (May 2026)
- **Tile-mode rewrite (`smb_tile_mode`).** Pixel CNN PPO's `policy_loss` sat at
  ~0.01 forever — gradients too weak to move 1.7M weights against sparse reward +
  long trajectories. Root causes (5, compounding): sparse reward, oversized net,
  broken credit assignment, weak PPO inner loop, elite-clone-overwrite killing
  diversity. **Fix:** RAM tile obs + a ~14k-param MLP. Additive, dispatched on
  `reinforce.encoder`.
- **PPO structural fix (commit 69a6e68).** `time_penalty=0`+`death_penalty=0`
  made **no-op a Nash equilibrium** (standing still ≡ moving+dying in return);
  unnormalized value loss (returns [0,5000+]) dominated 100% of gradient vs
  policy's −0.005; entropy pinned at ln(8). **Fix:** −15 death / −0.01 time,
  Huber value loss, value_coef 0.25. Early "wins" (fitness 7716) were **base-rate
  luck**, not learned (~1% of random policies clear 1-1 given the right-biased
  action set). *Lesson: verify a "win" with a GREEDY eval — sampled/GA wins are
  often luck.*
- **Pre-PPO elite freeze (4606def).** A single regressive PPO step corrupted the
  GA elite in place → a genuine gen-151 winner destroyed in one step. Fix:
  snapshot the elite before PPO.

### Phase B — 1-1 solved + the save-state curriculum (May 2026)
- **Save-state curriculum.** vanilla_ppo's iter boundary teleports every env to
  cold-boot 1-1, so even at 70–80% 1-1 clear the policy never trained a single
  1-2 frame. Fix: mid-rollout capture of the first level-entry state, warm-start
  all envs there, advance on a rolling-mean clear gate. Many subtle bugs
  (capture-too-late, dead-env capture, cutscene-byte transitions, a Rust
  `frame_cycle_target=None`-after-apply_state bug that froze restored states).
- **★ 1-1 GENUINELY LEARNED (2026-06-23).** Proven recipe = `mario_vanilla_ppo`
  (tile + PPO, **shaping OFF** = forward+completion only). Greedy argmax eval of
  iter-70 = **clear_rate 1.0 (5/5 deterministic)**. This is the first committed,
  reproducible learned clear.

### Phase C — The "why don't agents win" diagnosis (Jun 2026)
- **69-agent adversarial audit.** META-FINDING: win-blockers were mostly broken
  instrumentation + 2 reward misspecs, not missing capability. Contra's
  `episode_success()` was hardcoded `false` and reward was 71× score-dominated;
  SMB's "collapse" was the pooled cross-stage success-rate metric cratering when
  the curriculum advanced (the 1-1 skill was intact). The real wall = a genuine
  **1-2 exploration limit**. Huge DEBUNK list (do not re-chase): catastrophic
  forgetting, entropy-schedule/SAC (entropy↔success corr −0.26), value
  normalization (symlog already on), gamma tweaks, RND-as-primary, LSTM/bigger
  net as the 1-1 ceiling.
- **Recurrence RULED OUT for SMB (clean negative).** GRU tile net at 1.29× the FF
  net's 1-1-mastery budget got ~1% success vs FF's 88% — SMB 1-1 is
  reactive/Markovian, so the GRU adds optimization difficulty with no memory
  payoff, *crippling the 1-1 mastery that is the prerequisite to ever testing the
  1-2 memory hypothesis.* (Note: the recent recipe re-introduced recurrence for
  1-2 specifically; still no clear — §6.)
- **Dense-shaping farming (recurring).** Heavy per-step shaping (checkpoint/jump/
  survival) made the agent farm shaping instead of committing to the flag (same
  class as Contra's score-farm). For 1-1, shaping-OFF is proven; shaping is only
  for exploration past 1-2.

### Phase D — Per-level World 1, and the entropy-consolidation lever (Jun–Jul 2026)
- **Area-byte reward bug (319b01c).** The 1-2 dense ladder was mapped to the wrong
  area byte (entrance, not the underground main) → zero dense reward where the
  agent plays. *The "x=980 wall" was a greedy-replay artifact* — training
  warm-started in the underground blows past it.
- **★ Entropy-decay consolidation (key reusable lever).** A stochastically-found
  clear (e.g., 44–94% sampled) stays greedy-0 under a constant entropy bonus; a
  **consolidation run decaying entropy 0.01→0.002 sharpens it into a deployable
  GREEDY clear.** This cracked 1-2-underground and 1-3.
- **Seed transfer matters:** seeding 1-3 (needs a running long-jump) from the
  1-2-underground winner FAILED (the floored corridor atrophied the jump); seeding
  from the 1-1 winner transferred the skill and cracked it.
- **1-4 (castle) = autonomous ceiling for PPO+tile.** Every autonomous lever
  (ladders, seed-transfer, within-level curriculum, anti-stall) collapsed to a
  safe-stall at the x814 platform-hop; mid-castle warm-starts are death-traps
  (timed firebars). Go-Explore + tuned hyperparams got 1-4 clearing stochastically
  (peak ~7/24) but not consolidated.

### Phase E — The composite router + World 1 "one-shot" (mid-Jul 2026)
- **World-1 one-shot (2026-07-16): seq_clear_rate 1.0 deterministic**, cold →
  1-1→1-2→1-3→1-4, via a **composite** of per-level BC "pilot" nets + a router
  with hysteresis + per-level entry opts. Extended to **Worlds 1–3 (12 levels,
  2026-07-19)** and **1-1→4-3 (15 levels)** — the mechanical loop = capture true
  arrival → Go-Explore-solve → replay_to_demos → BC-clone → route.
- **★ THE HONEST-EVAL REVELATION.** Running the World-1 composite under
  **sticky-0.25 + jitter-16 → 0/50, all die in 1-1** (vs deterministic 1.0). The
  composite is a **trajectory-replay system** — Machado's "Brute" in the
  literature; 0/50 is its signature. Sticky+jitter became the only reported
  metric.

### Phase F — The "no cheating" course-correction (2026-07-19)
- **Binding directive (project owner):** "You're disassembling the game… cheating.
  We've gone way off course." The 4-4 maze work used the disassembly's LoopCmd
  tables; 4-2's flag used hand-driven inputs. **All such assets banned.** The
  product is an AI that *learns* autonomously (watchable on Twitch, reproducible
  from source). Pilots/routed chains are solver artifacts, NOT the product. The
  honest sticky eval (0/50) had been telling the truth all along.

### Phase G — Toward a genuinely-learned agent (2026-07-20 → 07-22)
- **Capability attempt #1 FAILED.** "Train the whole level from cold under sticky
  + entropy floor" DEGRADED 1-1 (1.0 det → 0/20 det). Wrong shape; the entropy
  floor is sound but insufficient.
- **Generalist plan + Phase-A.** ONE model-free PPO policy, tile obs, PLR,
  sticky. **Phase-A PASSED:** a from-scratch, generic-reward, sticky-trained tile
  net cleared **1-1 at 63–67% cold sticky+jitter** (2 seeds) — the first
  genuinely-learned honest clear. *But it does not generalize:* multi-level PLR
  (3 levels) collapsed to a non-clearing average even with a 4.5× wider net —
  ruling out capacity and entropy-schedule as the cause. Diagnosis: **1-2 can't
  be learned from scratch (exploration wall) — it must be search-solved then
  distilled.**
- **Stability audit (~70 agents):** 10 verified crash/corruption/integrity fixes
  (Rust-panic-blind supervisor, non-atomic checkpoints, seq_clear warp-latch,
  etc.). 918 tests + 146 parity green. (Infra, not the learning problem.)
- **Research-grounded pivot (2026-07-21).** Two external research docs → the
  Go-Explore-solves + distill pipeline. **Breadth track WORKS:** the honest
  Go-Explore solver clears **1-2, 1-3, 1-4, and the hands-off chain 2-1→4-3**
  (15 levels), seconds each, warp-guarded, clean entrances. **Depth track (the
  hard, goal-aligned part) does NOT:** turning those solutions into a learned
  sticky-robust policy — see §III/§IV.

---

# PART III — THE RECURRING FAILURE TAXONOMY (cross-cutting)

1. **LEARNED ≠ SEARCH-SOLVED ≠ SAMPLED-clear.** Search (Go-Explore) solves
   levels; a *learned* policy must clear them greedily under sticky. GA/sampled
   "wins" are usually luck; always greedy-eval.
2. **Long-horizon sticky-fragility.** Clear probability decays with the remaining
   horizon: near the goal, sticky-robust ≈ 1.0; past ~250–320 action-steps it
   collapses to ~0. 1-1 (learnable from scratch, reactive) works; 1-2 (~787 steps
   + a precise obstacle) does not.
3. **Greedy ≠ sampled (newest, §6).** A dense-shaped run can raise SAMPLED return
   while the GREEDY policy (the eval mode) collapses to early death.
4. **Dense-shaping farming.** Dense per-step reward gets farmed without
   completing; PBRS is theory-invariant but with a finite-training value function
   it can still dominate the sparse terminal.
5. **Over-consolidation / entropy collapse.** Runs peak then decline as entropy
   collapses onto a non-completing mode. Entropy floor prevents the crash but can
   also block greedy consolidation.
6. **Covariate shift.** BC on clean trajectories → drift under noise → OOD states
   with no supervision → catastrophic failure.
7. **Metric artifacts (repeatedly bit us).** Pooled cross-stage success-rate;
   cumulative (non-reset) max_x masking collapse; area-byte-filtered telemetry
   lying about multi-area levels; Go-Explore-return-inflated "clear rates."

---

# PART IV — WHAT IS RULED OUT (with evidence)

- **Recurrence as the 1-1/general lever** — negative (cripples Markovian 1-1).
  (Still testing it specifically for 1-2's POMDP platforms; §6.)
- **Capacity (bigger MLP)** — 4.5× width didn't fix multi-level or long-horizon.
- **Entropy schedule alone** — floors 0.0/0.1/0.3/0.5 + sticky-vs-greedy
  acceptance all tested; wall persists.
- **Dense wavefront reward alone** — raised sampled return but neither reached the
  goal nor produced a good greedy policy (§6); partially farmable.
- **DQfD/DART recovery anchor alone** — insufficient; greedy still collapses.
- **From-scratch PPO on long levels** — exploration wall (never reaches the flag).
- **DreamerV3/world-models, EfficientZero, VLM/LLM-guidance** — killed with
  receipts (wrong regime: our bottleneck is not sample efficiency).
- **Maze levels (4-4/7-4/8-4) via honest search** — the maze needs its RAM
  loop-byte (disassembly → ruled out); random-burst Go-Explore loops. Separate
  problem from the core.

---

# PART V — CURRENT STATE & THE OPEN PROBLEM

**Solid, honest deliverable:** 1-1 *learned* (63–67% cold sticky+jitter);
**15 levels search-solved** (1-2/1-3/1-4 + chain 2-1→4-3), receipted, hands-off.

**Open problem (the crux):** produce a **learned, closed-loop policy that clears
a long level (1-2 testbed, ~787 steps) from its entrance under greedy + 25%
sticky + jitter, ≥60%**, distilled from / bootstrapped by the Go-Explore
solution, on one M4, without game internals. Four method families fail it:
(1) self-imitation backward-curriculum robustifier — welds the back third, stalls
at a front-half precise obstacle; (2) demo-action-augmented distiller — same
horizon-decay (1.0 near goal → 0 by ~320 steps); (3) RL-from-distilled-init —
0.0; (4) the recurrent+wavefront+DART recipe — §6.

See v1/v2 briefs for the sharp questions. The single most diagnostic new datum:
**the recipe's greedy policy dies at gx 196 of ~3266 in the underground (~6%)
while its sampled training return reached 1692 and it NEVER once reached the flag
in 766 iters** — i.e. it optimized the wrong (sampled, not greedy) objective and
the goal was never actually reached from the entrance.

---

# PART VI — CODE APPENDIX (the recent pipeline, verbatim)

## VI.1 Wavefront PBRS (`src/utils/wavefront_reward.py`) — non-cheating dense reward
Distance-to-goal D(cell) from Go-Explore SOLUTION traces (not the ROM); cell =
`(area, x//16, y//32)`. Φ(s)=−α·D, F=γΦ(s')−Φ(s). Verified on 1-2: 313 cells,
D∈[0,748], monotone, +556 total shaping along the solution.
```python
def wave_cell(ram):  # (area, x-bucket, y-band) from RAM — game-agnostic
    return (int(ram[0x0760]), ((int(ram[0x006D])<<8)|int(ram[0x0086]))//16,
            int(ram[0x00CE])//32)
# build_distance_map: replay each solution; D(cell)=min(len(actions)-index) over visits.
# WavefrontPotential.potential(ram) = -alpha * D(cell); off-map -> nearest same-area x-bucket.
```

## VI.2 DARS recovery (`scripts/generate_dars_recovery.py`) — DART core
For N samples: restore a solution state, inject 1–5 frames of 25% sticky noise
(drift), then EXPERT-RELABEL the drifted rollout (solution actions become
targets), keep survivors that progress (wavefront distance decreases). 1-2: 1722
recoveries → 50,410 unique (obs,action) pairs (deduped — the demo-anchor loader
requires ~100% unique obs to catch buffer-view aliasing).

## VI.3 The recurrent recipe config (`configs/mario_1_2_robust_recurrent.yaml`)
```yaml
reinforce:
  encoder: smb_tiles_pos
  recurrent: true            # GRU 128/64 (~112k params)
  tile_hidden_dim: 128
  tile_trunk_dim: 64
  sticky_action_prob: 0.25
  entropy_coef: 0.01
  entropy_floor: 0.1
  wavefront_reward: {enabled: true, dmap: checkpoints/wavefront/mario_1_2_dmap.pkl, alpha: 0.15}
  demo_anchor_enabled: true              # DQfD imitation of DARS recoveries
  demo_anchor_paths: [runs/dars_1_2/recovery_demos.npz]
  demo_anchor_coef: 1.0                  # -> 0.02 (imitate then wean = staged BC->PPO)
  num_envs: 48
```
**Result:** gate 0.0; sampled return peaked 1692 then declined; 0 training clears;
greedy dies at gx 196. (§6, v2 brief — the crux for the researcher.)

## VI.4 Reusable tools (all shipped, tested)
`go_explore_solve.py` (honest entrance-rooted solver), `go_explore_chain.py`
(hands-off multi-level chain), `distill_level.py` (demo-action-augmented backward
curriculum), `robustify_level.py --accept-sticky` (self-imitation ladder, parallel
collection), `replay_to_demos.py`, `generate_dars_recovery.py`, `wavefront_reward.py`.
Banked artifacts: 27 solution trajectories + clean entrance states (1-2…4-3), the
1-2 wavefront dmap, the 50k DARS recovery set, the learned-1-1 net (the ONE
positive example — study why from-scratch reactive PPO with NO shaping/imitation
succeeded where the heavily-engineered pipelines failed), and the failing
recurrent artifact (greedy dies gx 196) for probing.

**The ask, restated:** given all of the above, return the concrete recipe
(algorithm, reward balance, net, schedule, hyperparameters, sample budget) that
makes the **greedy** policy clear 1-2 from the entrance ≥60% under sticky+jitter,
plus the single diagnostic that disambiguates "recipe needs X" from "we have an
implementation bug" (given greedy-dies-at-196 vs sampled-return-1692).
