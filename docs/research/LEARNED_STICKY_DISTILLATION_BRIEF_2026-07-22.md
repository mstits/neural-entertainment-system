# Research Brief: Learned, Sticky-Robust Clearing of Long NES Levels

**Prepared for:** an external deep-research investigation.
**Date:** 2026-07-22.
**Ask:** read this brief and return concrete, implementable solutions (with
citations and, ideally, pseudocode or algorithmic recipes) to the single core
problem stated below. Everything here is measured on our own system; numbers are
real, not aspirational.

---

## 0. The one core problem (read this first)

> **We can (a) *learn* a short NES level from scratch with model-free RL, and
> (b) *search-solve* long/hard levels with Go-Explore. We CANNOT turn a
> search-found solution (or otherwise learn a policy) that clears a *long*
> (~500–1100 action-step) level from its entrance under stochastic evaluation
> (25% sticky actions + no-op start jitter). Every method we tried is
> sticky-robust near the goal but the robustness *decays to zero as the required
> horizon grows*. We need methods that produce a genuinely learned, closed-loop
> policy robust over the full long horizon — without cheating (no
> disassembly/level-maps/hand-scripted inputs) — on a single Apple-Silicon
> machine.**

If you solve nothing else, solve horizon-robust distillation/learning for a
single long level (SMB World 1-2) under our honesty protocol.

---

## 1. System & hardware

- **Emulator:** custom cycle-accurate NES core in Rust (nes_core), exposed to
  Python via pyo3. Deterministic; **microsecond save/restore** of full machine
  state; runs headless. Parallel worker pool (`Pool(num_workers=N)`), rayon-
  parallel emulation with the GIL released.
- **Hardware:** one Apple M4 (12 performance + 4 efficiency cores, 16 logical),
  128 GB unified memory, latest macOS. This is "the agent" — one machine.
- **Measured throughput:** tile-obs PPO ~3000–4400 env-steps/s at 60 envs on
  CPU; Go-Explore search ~3000 env-steps/s per worker headless.
- **Apple-Silicon note (important for any proposed net):** small nets are
  **dispatch-latency-bound on MPS** (25–80 µs GPU driver dispatch per op vs
  <1 µs of actual compute), so we run the small tile policy on **CPU** (faster
  than MPS for this size). MPS only pays off for large batched CNN/RSSM work.
  `torch.compile` on MPS is unreliable. Pre-allocated tensors to avoid MPS
  fragmentation OOM.

## 2. The task and the HONESTY PROTOCOL (central — solutions must respect it)

Goal: an AI that **genuinely learns** to play and beat Super Mario Bros (NES),
watchable live and reproducible by anyone who compiles the code.

**Binding honesty rules (non-negotiable, set by the project owner):**
1. **No game-internals cheating.** No disassembly-derived knowledge, no level
   maps, no hand-scripted/hand-driven input sequences, no per-level memorized
   scripts presented as "learning." (This specifically rules out, e.g., using
   SMB 4-4's maze-layout RAM byte to route the maze.)
2. **Honest evaluation = Machado et al. 2018 protocol.** The only headline
   metric is **cold, greedy (argmax) clear rate from the level entrance under
   25% sticky-actions (prob. of repeating the previous action) + no-op start
   jitter (0–16 random no-op frames before control)**, single life, averaged
   over ≥2 seeds × ≥12 episodes. "Cleared" = reached the flag / forward
   level-transition (warp-guarded: a warp-zone pipe does NOT count).
3. **Training-time state restoration IS allowed** (Go-Explore / backward
   algorithms restore states during *training*; the field considers this
   legitimate as long as *evaluation* is the sticky-from-entrance protocol).
   Using a search as a *teacher* is allowed; the deployed agent must be a
   learned policy.
4. **LEARNED vs SEARCH-SOLVED must stay distinct** in all claims. A level a
   search cleared is "search-solved," not "learned," until a policy clears it
   under the protocol above.

## 3. The learning stack (be precise — propose changes against this)

- **Observation ("tile" mode, `smb_tiles_pos`):** a RAM-decoded feature vector
  — a 13×13 tile grid around Mario plus scalar position/velocity features =
  **178 features**, 4-frame stacked = **712-dim** input. (Pixel CNN 84×84×4
  exists but is ~10× slower per rollout on this hardware; tile is the workhorse.)
- **Policy net (`TilePolicyNetwork`):** MLP `712 → Linear(hidden=64) →
  LayerNorm → SiLU → Linear(trunk=32) → LayerNorm → {actor head (6 actions),
  critic head}`. **~48k parameters.** (We also tested hidden=256/trunk=128 =
  ~217k params; see §5.) Orthogonal init. Runs on CPU.
- **Action space (6):** `[noop], [right], [right+A], [right+B], [right+A+B],
  [A]`. frame_skip 4.
- **Algorithm:** model-free PPO (GAE-λ, Huber value loss, value_coef 0.25,
  entropy coef ~0.005 with an adaptive floor, RND intrinsic two-head option,
  symlog rewards). 60 parallel envs. Sticky-actions applied *in training* to
  match the eval. An entropy floor controller (raise coef when entropy < floor)
  prevents the collapse-to-deterministic failure mode.
- **Reward (generic, per-level):** `+forward_progress (Δx)`, `−0.01 time`,
  `−15 death`, `+50 completion (flag)`; clipped. No level-specific shaping.

## 4. What currently WORKS (the honest ledger)

| Item | Status | Measured |
|------|--------|----------|
| **World 1-1** | **LEARNED** | 63–67% cold clear under sticky-0.25 + jitter-16, 2 seeds (iter-130 tile net). First genuinely-learned honest clear. From-scratch PPO (reactive controller). |
| **1-2, 1-3, 1-4** | search-solved | Go-Explore from clean entrances, seconds each, warp-guarded, receipted. |
| **2-1 … 4-3 (11 levels)** | search-solved | hands-off chain (solve → extract next entrance from the clear → solve next). Action counts: 2-1:763, 2-2:1100(water), 2-3:652, 2-4:595(castle), 3-1:853, 3-2:663, 3-3:506, 3-4:594, 4-1:737, 4-2:921, 4-3:485. |
| **4-4** | STALL | maze; wrong branches loop. Honest random-burst Go-Explore can't find the route; cracking it needs the maze RAM byte (disassembly → ruled out). |

**So: 1 level *learned*, 14 levels *search-solved* (~half the 32-level game).**
Go-Explore search is strong and general (handles water/castle physics that broke
the model-free generalist). The gap is turning search solutions into *learned*
policies.

Key contrast that frames the problem: **1-1 became sticky-robust because it was
learned by from-scratch RL (a reactive closed-loop controller). Longer levels
cannot be learned from scratch (exploration wall — that's why we need Go-Explore
to solve them), and distilling the search solution yields sticky-FRAGILE
imitation.**

## 5. What we TRIED for "learned long levels" and the exact results

All targeting **World 1-2** (a ~787-action underground level with a precise
mid-level obstacle). All evaluated by the §2 protocol. **All produced 0.0 cold
sticky+jitter clear from the entrance.**

**(a) From-scratch model-free PPO on 1-2 (incl. with RND, tuned entropy).**
Never reaches the flag — dies ~270 steps in, 0 training clears in 350+ iters.
1-2's long single correct path with sparse reward is a hard-exploration problem;
gradient descent can't explore it. (This is *expected*; it's why Go-Explore is
used.)

**(b) Go-Explore SOLVES 1-2 (search).** From the clean entrance, right-biased
sticky random bursts + first-return-then-explore over a save-state cell archive
(cell = area, step-phase, y-band, gx-bucket). Solves in ~12 s; produces
replayable action traces (~787–829 actions) that legitimately reach 1-3.
**This works.** The solutions are the "teacher."

**(c) Behavioral cloning of the solution trajectory.** BC the tile net on the
~3200 (obs, action) pairs from 4 solutions → 81.5% train action-accuracy →
**0.0 deterministic** clear (not even greedy). 81.5% per-action over ~800 steps
compounds into desync/death. Classic BC covariate shift.

**(d) Demo-anchored sticky PPO from scratch (DQfD-style).** PPO + a per-update
imitation loss toward the solution demos (coef 1.0→0.02) + sticky-0.25 + dense
reward. Never reaches the flag (dies ~270 steps, clears=0 through iter 190).
The anchor pulls toward demo actions on demo states, but the policy's own
(off-demo) rollouts get no signal → same exploration wall as (a).

**(e) Backward-curriculum self-imitation robustifier (Salimans & Chen 2018
style).** Restart from states along the solution, deepest (near flag) → shallow
(entrance); at each rung collect the policy's own *sticky* clears, BC them,
advance the restart backward. **Welds ~60% of the level (greedy 1.0) then
STALLS** at precise obstacles where the policy cannot stochastically self-clear
(0 clears collected) and no harvested intermediate exists to bridge. Tried
gx-spaced and action-spaced (6-action) rungs; tried greedy-acceptance and
sticky-acceptance. Stalls at the same obstacle region either way.

**(f) Demo-action-augmented backward-curriculum distillation (our best idea).**
At each backward rung K, the BC dataset = the solution's *exact* (obs, action)
suffix `seq[K:]` (guarantees a signal even where self-imitation can't cross) +
the policy's self-collected sticky clears (recovery); accept under sticky. This
**extends the robust region deeper** than (e) but the measured per-rung sticky
clear rate vs remaining horizon is the crux finding:

```
rung K (action index)   remaining horizon   sticky clear rate
747 → 547               ≤ 40 → 240 actions   1.00   (robust)
507                     ~280 actions          0.50
467 and earlier         ~320+ actions         0.00   (one 0.12 flicker at 387)
0  (entrance)           787 actions           0.00
```

**(g) RL fine-tuning from the distilled init (last lever).** PPO warm-started
from (f)'s net (which is robust on the back third) + restart-from-waypoints
(the full solve archive, 8180 states) + demo-anchor + sticky-0.25 + dense
reward, 500 iters. Training clears stayed modest (9–16/iter from waypoint
restarts); **entrance cold sticky+jitter = 0.0** at iters 400 and 490, both
seeds. The reactive fine-tuning did not make the front half robust in-budget.

**Capacity check:** widening the net to ~217k params (hidden 256 / trunk 128)
did *not* help multi-level generalization or long-horizon robustness (same
collapse), so raw MLP width is not the bottleneck.

## 6. The characterized wall (the empirical heart of the problem)

**Sticky-robust clear probability decays with the required horizon length.**
Near the goal (short remaining path) the learned/distilled policy clears at
1.0 under sticky; as the required horizon grows past ~250–320 action-steps it
collapses to ~0. Intuition: under 25% action-repeat noise, surviving a long
*precise* sequence without a fatal desync becomes exponentially unlikely unless
the policy is genuinely *reactive* (recovers from perturbations), and neither
imitation nor short-budget RL fine-tuning produced that reactivity over the full
length. 1-2 additionally has a **front-half precise obstacle** (~action 460–510,
underground gx≈2000–2430) that is sticky-fragile: even with the exact expert
actions available (demo-suffix BC), the policy clears it only ~0–50% under
sticky, and that fragility compounds with everything downstream.

## 7. Hypotheses we hold, and what we've ruled out

**Ruled out (with evidence):**
- *It's a capacity problem* → No; 4.5× wider net didn't help.
- *It's the entropy schedule* → No; tested floors 0.0/0.3/0.5, sticky vs greedy
  acceptance; wall persists.
- *It's rung spacing / missing intermediates* → Partially; finer rungs help but
  the obstacle is sticky-uncrossable even with dense rungs + expert actions.
- *4-4 is solvable by honest search* → No, not without the maze RAM byte
  (disassembly → ruled out). 4-4 is a separate exploration problem, not the
  core distillation problem.

**Open hypotheses (candidates for the researcher to confirm/refute/replace):**
- The reward is too sparse over the horizon; **dense intermediate rewards
  (checkpoint/wavefront distance-to-goal) that shorten the *effective* horizon**
  might make RL learn reactivity end-to-end.
- Action-imitation is the wrong distillation target; **using the solution as a
  *value* prior** (RL with a demo-bootstrapped value function) might yield a
  reactive policy.
- The sticky-fragility is intrinsic to open-loop-ish policies; **explicit
  recovery training** (inject perturbations during training and reward
  return-to-path) might build robustness.
- A **recurrent** policy (memory) or a richer observation might be needed for
  the precise obstacle (we've mostly used a feedforward tile MLP).

## 8. Constraints any proposed solution MUST satisfy

1. Honest eval unchanged (sticky-0.25 + jitter, cold from entrance, warp-guarded).
2. No game-internals/disassembly/hand-scripting (see §2.1). Training-time state
   restoration and search-as-teacher ARE allowed.
3. Runs on one Apple M4 (128 GB). Prefer CPU-friendly small nets (or justify
   MPS/large-net cost against the dispatch-latency reality in §1).
4. Reasonable wall-clock: a per-level solution should be hours, not weeks, on
   this machine (we can run ~10 parallel search workers and 60 PPO envs).
5. Reproducible from source; no external services required at train/eval time.

## 9. Assets available to a solution (you may assume these exist)

- Deterministic emulator with µs save/restore; parallel pool; RAM + framebuffer
  access. Key RAM: x = `$006D<<8 | $0086`, world `$075F`, displayed-level
  `$075C`, area `$0760`, lives `$075A`, player-state `$000E` (6/11 = dying).
- **Go-Explore solver** (`scripts/go_explore_solve.py`): honest, entrance-only,
  warp-guarded; dumps solution action-traces. **Chain driver**
  (`go_explore_chain.py`): hands-off multi-level solve. Banked: 27 solution
  trajectories + clean entrance states for 1-2…4-3.
- **Distillation tools:** `replay_to_demos.py` (trace → (obs,action) npz),
  `distill_level.py` (demo-action-augmented backward curriculum),
  `robustify_level.py` (self-imitation backward curriculum, `--accept-sticky`,
  parallel collection).
- **Trainer** (`vanilla_ppo`): sticky training, RND, demo-anchor (DQfD),
  go-explore restart, entropy floor, PLR/curriculum, honest cold-probe.
- **Eval** (`eval_game.py --sequential --level-clear --sticky-prob --start-jitter`).
- A **learned 1-1 net** (the one positive example of a sticky-robust learned
  policy — study what made it work) and a **distilled 1-2 net** robust on the
  back third.

## 10. Specific questions we want answered

1. **What is the state-of-the-art method to distill a single long
   demonstration into a policy that is robust to action-persistence (sticky)
   noise over the full horizon?** (Beyond BC, DQfD, and the backward algorithm,
   which we've tried.) Cite methods that demonstrably handle 500–1000-step
   precise sequences under 25% sticky.
2. **Reward design:** is a dense, honestly-computable distance-to-goal reward
   (e.g., wavefront/BFS over reachable RAM-derived positions, NOT a
   disassembled map) the key to shortening the effective horizon? How to compute
   it without game-internals cheating?
3. **Why did 1-1 learn (reactive) but 1-2 won't**, and what specifically makes a
   policy "reactive" enough to absorb sticky perturbations — is it an
   architectural property (recurrence, larger receptive field), a training
   property (perturbation/recovery training), or a reward property?
4. **Is per-segment learning + a learned high-level switch** a legitimate,
   robust answer, or does it just reintroduce the brittle "router" we retired?
   If legitimate, how to make the switch and segments sticky-robust and
   composable?
5. **Exploration for maze levels (4-4/7-4/8-4):** how to solve a looping maze
   with honest, game-agnostic exploration (no maze RAM byte)? (Secondary to the
   core problem, but blocks full-game breadth.)
6. **Given the Apple-Silicon constraints (§1), what net/algorithm choices** best
   fit CPU-bound small-model RL at 60–256 parallel envs? Is there a case for
   MPS/large-net here that beats CPU despite dispatch latency?
7. Concretely: **give us the recipe** (algorithm, reward, net, schedule,
   hyperparameters, expected sample budget) most likely to get 1-2 to ≥60% cold
   sticky+jitter from the entrance on this hardware, and how to generalize it to
   the other search-solved levels.

## 11. Prior art we've already applied (don't re-recommend as-is)

Go-Explore (Ecoffet et al. 2021); backward algorithm / learning from a single
demonstration (Salimans & Chen 2018, arXiv:1812.03381); DQfD (demo-anchored RL);
behavioral cloning + DAgger; Machado et al. 2018 sticky-actions; PLR
(Jiang et al. 2021); RND (Burda et al. 2018); DreamerV3 RSSM (considered,
rejected on MPS throughput). We've implemented and hit walls with the first
five. Novel combinations or methods beyond these are what we need.

---

**Bottom line for the researcher:** search solves our levels; the unsolved
science is *learning a closed-loop policy that is robust to sticky-action noise
over a long precise horizon, distilled from (or bootstrapped by) that search,
without cheating, on one Apple-Silicon machine.* Point 1-2 is the concrete
testbed; a recipe that gets it to ≥60% cold sticky+jitter from the entrance is
the win.
