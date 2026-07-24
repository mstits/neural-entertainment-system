# Research-Grounded Plan — ending the six-month thrash

Source: the two research docs (NES AI Planning Research; Unified Generalist
Retro-Console AI Agent Blueprint). This supersedes the ad-hoc "make model-free
PPO generalize" loop that has repeatedly failed.

## The honest diagnosis: why we've been stuck

Every failed run this campaign shares one root cause: **we tried to make a
model-free PPO policy both (a) solve hard levels from scratch AND (b) generalize
across level types, directly.** The research says plainly that this does not
work, and both docs open with the exact failure we measured — "Generalization
Collapse in Multi-Stage Retro Environments" driven by visual, physics, and
scrolling shift.

Our concrete wall — 1-2 never clears, even trained solo (dies ~270 steps into
the underground gauntlet, zero flag touches in 350 iters) — is the textbook
**hard-exploration problem** (sparse/delayed reward, one long correct path).
That is the exact problem **Go-Explore** was invented to solve (Montezuma's
Revenge, Pitfall). We have Go-Explore in the repo but never used it as the
primary trajectory-solver — we kept asking PPO to explore its way to a distant
flag, which it can't.

## What the research says actually works

Winning retro-console agents are **structured**, not raw model-free PPO:

| Game | Proven approach (from the research) |
|------|-------------------------------------|
| SMB | State-space search/planning (Mairio) OR RSSM world model; **falling-velocity reward filter** (halt reward when vy<0 → no pit-diving) |
| Super Metroid | **Go-Explore + abstraction** (cell = tile+item flags; log-linear progress selection); 25 action primitives held 1-20 frames |
| Mega Man | **Wavefront (Dijkstra) distance-to-goal map** + **asymmetric reward** (backward penalty ≫ forward reward); spike/pit → truncate; dual-phase curriculum |
| Tetris | Afterstate value + Dellacherie-Thiery features + bitboard |
| Zelda | Centered viewport crop + RAM multi-modal + tile-grid action constraint + **RAM action masking** (skip locked frames) + predictive save-state rollouts |
| Contra/Castlevania | reward = v+d+b, clipped [-15,15], 4-frame stack, multi-key combos |

## THE core pipeline (Blueprint §2C) — this is the answer

**Go-Explore Trajectory Distillation:**
1. Run **Go-Explore natively in Rust** on the clean, deterministic emulator,
   exploiting our **microsecond save/restore** to build a progressive trajectory
   graph that SOLVES the level (reaches the flag). Search, not gradient descent,
   does the hard exploration.
2. **Distill the solved trajectories into ONE generalist policy** via
   **Behavioral Cloning + DAgger**, **with 25% sticky actions** so the policy
   learns to RECOVER from stochastic slips (not memorize a brittle path).
3. **Evaluate honestly**: sticky-0.25 + jitter from power-on. This is legit
   under the field's purity line (Go-Explore/Nature restore states in TRAINING,
   evaluate sticky) — the policy genuinely learns; it is not hand-driven.

This directly breaks our 1-2 wall: Go-Explore finds a 1-2 solution the PPO
explorer never could, then the policy LEARNS it (robustly, via sticky BC/DAgger).

## We already have most of the pieces

- ✅ Microsecond save/restore (measured), Rust core, deterministic emulator
- ✅ Go-Explore (`src/training/go_explore.py`) — needs promotion to primary solver
- ✅ Behavioral cloning (`src/training/behavior_cloning.py`), DAgger-able
- ✅ Sticky actions (25%) wired; honest sticky+jitter eval (`eval_game.py`)
- ✅ Parallel diverse envs / PLR (built 2026-07-21) — Blueprint §2A "parallel
     environmental diversity" is exactly this
- ✅ RSSM/DreamerV3 scaffold (dead — Blueprint §2B revives it as an OPTION, not
     required for the core pipeline)
- ⚠️ Missing glue, all specified with concrete code in Blueprint §4:
  - Asymmetric wavefront-distance reward (`AsymmetricWavefrontReward`, §4-Task3)
    — dense distance-to-goal, backward penalty ≫ forward. Fixes the sparse-reward
    exploration problem AND pit-diving/oscillation.
  - RAM action masking (`NESStateReader.get_action_mask`, §4-Task1) — skip
    animation-locked frames (no wasted gradient).
  - SMB falling-velocity reward filter (PDF1) — halt reward when vy<0.
  - Centered viewport crop (§4-Task2) — for Zelda/open-world later.

## The re-grounded plan (phased, each phase gated on the prior)

**Phase 1 — Prove the pipeline on our exact blocker (1-2).**
1. Add the **asymmetric wavefront-distance reward** for SMB (dense progress,
   backward penalty, fall-filter). Cheap; fixes the reward sparsity that starves
   1-2.
2. Run **Go-Explore in the Rust/emulator loop to SOLVE 1-2** (reach the flag),
   using µs save/restore + the progress-weighted cell archive. Success =
   Go-Explore produces ≥1 full 1-2 trajectory.
3. **Distill** those trajectories into the tile policy via BC + 25% sticky +
   DAgger. Success = 1-2 clears cold-greedy under sticky+jitter (the honest gate
   it has NEVER passed).
GATE: if 1-2 clears honestly this way, the pipeline is proven and we scale it.
If Go-Explore can't even solve 1-2, that's a much narrower, well-defined
problem (its abstraction/archive) than "PPO won't generalize."

**Phase 2 — Generalist via distillation.** Go-Explore-solve 1-1/1-2/1-4/2-1,
distill ALL trajectories into ONE policy (BC/DAgger + sticky + PLR diversity).
Add RAM action masking. This is the Blueprint's generalist — learned, honest,
multi-level.

**Phase 3 — Scale + optional world model.** More levels/worlds; add the RSSM
(§2B) only if cross-physics generalization plateaus (per PDF1 it buys physics
invariance). Extend to other games via the per-game structured approaches.

## Hardware note (reconciling the two docs)
PDF1: run SMALL nets on CPU (dispatch latency kills MPS for tiny nets) — matches
our `device: cpu` tile runs (correct, keep). Blueprint §4-Task4: MPS + 64
parallel + `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85` + pre-allocated tensors — for
the LARGER batched visual/RSSM generalist. Rule: tile MLP → CPU; pixel/RSSM →
MPS batched. Both docs agree: parallel envs + pre-allocation, avoid torch.compile
on MPS.

## What we STOP doing
- Stop trying to make model-free PPO explore its way through hard levels.
- Stop treating "the generalist won't generalize" as the problem — the problem
  is per-level EXPLORATION (solved by Go-Explore) + reward sparsity (solved by
  wavefront reward), then DISTILLATION into one policy.
- Stop the entropy-knob roulette — it was treating a symptom.
