# Research Brief v2: Your Recipe Was Implemented Faithfully — Here Is How It Failed

**Prepared for:** the same deep-research investigation.
**Date:** 2026-07-22.
**Read first:** Brief v1 (`LEARNED_STICKY_DISTILLATION_BRIEF_2026-07-22.md`) — the
system, honesty protocol, constraints, and prior failures. This v2 reports the
result of implementing **your v1 solution** (DARS recovery + recurrent GRU +
non-cheating wavefront PBRS + staged BC→PPO + sticky). We implemented all of it,
faithfully, on the real stack. **It did not clear World 1-2 (gate = 0.0), and it
failed in a specific, highly diagnostic way we need you to now explain and fix.**

---

## 0. TL;DR of the new result

- We built exactly what you specified (details in §1), verified each piece.
- **Honest gate (cold greedy + 25% sticky + jitter-16 from the 1-2 entrance): 0.0**
  (2 checkpoints, confirmed). Same 0 as all prior methods.
- **The diagnostic surprise:** during training the *sampled* policy's shaped
  return climbed strongly (826 → **1692**) — the wavefront reward clearly pulled
  it deeper — **but the *greedy/argmax* policy (the eval mode) reaches only
  gx ≈ 196 of ~3266 in the underground (~6%) before dying.** And **training
  clears were 0 for all 766 iterations** (it never reached the flag even while
  sampling/exploring). Then it over-consolidated: return peaked at iter 500 and
  *declined* (1692 → 1434) as entropy collapsed (1.78 → 0.31).
- **So the failure is no longer "sticky-fragility over a long horizon."** It is:
  **the recipe optimized *sampled* return via dense shaping, but the *greedy*
  policy — the one the honest protocol evaluates — is near-useless (dies ~6% in),
  and the process never actually reached the goal even once.** The dense
  wavefront reward was farmable enough to raise sampled return without producing
  a completable greedy controller.

The single question for you: **§0 of v1 still stands, but now specifically — why
did dense-shaped recurrent PPO+imitation raise sampled return while the greedy
policy collapsed and the goal was never reached, and what recipe produces a good
GREEDY controller (not just high sampled return) for this long level?**

---

## 1. What we implemented (your v1 recipe → our real stack)

All faithful to your spec; only file paths differ (your guessed
PolicyNetwork.py / Trainer.py / rust_pool_adapter.rs don't exist — mapped to
our real `src/models/tile_policy.py`, `src/training/trainer.py`, etc.).

- **Recurrent policy (your §3):** `TileRecurrentPolicyNetwork` — `712 →
  Linear(128) → LayerNorm → SiLU → GRU(128→64) → {actor(6), critic}`, ~112k
  params. Our trainer already had a full recurrent PPO path (truncated BPTT,
  per-env hidden-state reset on `done`). CPU. ✔ exactly your architecture.
- **Wavefront PBRS (your §2):** `src/utils/wavefront_reward.py`. Distance-to-goal
  D(cell) built from the Go-Explore **solution traces** (verified search output,
  no ROM/disassembly), cell = `(area, x//16, y//32)` from RAM. Φ(s)=−0.15·D,
  F=γΦ(s')−Φ(s). Verified: 313 cells, D∈[0,748], Φ monotone toward goal, total
  shaping along the solution = +556. Injected into the per-step reward on live
  transitions (Φ(terminal) skipped so death/clear get no spurious potential). ✔
  (Adaptation: built from solution traces, not Dijkstra-over-archive — our
  archive stores cells, not edges; the solution IS a verified geodesic path.)
- **DARS (your §4):** `scripts/generate_dars_recovery.py`. For 2500 samples:
  restore a solution state, inject 1–5 frames of 25% sticky noise → drift, then
  **expert-relabel the drifted rollout** (solution actions become targets),
  keep only survivors that progress (wavefront distance decreases). Yield: 1722
  recoveries → **50,410 unique (obs,action) recovery pairs** (deduped; our
  demo-anchor loader requires ~100% unique obs). ✔
- **Staged BC→PPO (your §7):** realized as recurrent sticky PPO with the DARS
  set as a DQfD demo-anchor, coef **1.0 → 0.02** (imitate → wean) — this is your
  "Stage A BC pretrain then Stage B fine-tune" folded into one schedule — plus
  the wavefront reward, entropy floor 0.1, lr 3e-4, 48 envs, 1200-iter budget,
  from the 1-2 entrance. Config: `configs/mario_1_2_robust_recurrent.yaml`.
  (Deviation from your §7: we used our DQfD demo-anchor for the imitation phase
  rather than a separate sequence-BC pretrain, and did NOT add a KL-to-BC
  constraint — see Q4. We also did NOT use waypoint-restarts — faithful to your
  Phase 3, which resets from the level start and relies on the wavefront to
  supply the horizon-shortening signal.)

## 2. Exact measured result

Honest gate (cold, greedy, sticky-0.25, jitter-16, from 1-2 entrance),
iter-250 and iter-760 checkpoints, 15 episodes each: **clear_rate = 0.0**,
furthest = 1-2 (never left the level).

Training trajectory (sampled rollouts, 48 envs, wavefront-shaped return):

```
iter    shaped_return   mean_len   TRAIN clears   entropy
0        826            238        0              1.78
100      888            242        0              1.51
200     1088            254        0              1.37
300     1239            257        0              1.25
400     1601            306        0              1.05
500     1692            306        0              0.73   <- return peak
600     1539            288        0              0.51
700     1506            280        0              0.47
766     1434            274        0              0.31   <- declining, entropy collapsing
```

Greedy-policy depth probe (argmax from the entrance, no sticky, 5 episodes):
**max gx in the underground = 196** (flag/exit ≈ 3266; solution ≈ 787 actions).
i.e. the greedy policy dies right after entering the underground pipe.

## 3. The key diagnostic (this is new and important)

1. **Sampled ≫ greedy.** The dense wavefront reward raised *sampled* shaped
   return to 1692 (the exploring policy got materially deeper), but the
   *greedy/argmax* policy — the ONLY thing the honest protocol runs — collapsed
   to a ~6%-depth early death. The recipe optimized the wrong mode.
2. **The goal was never reached, even once, in 766 iters.** Unlike our
   backward-curriculum runs (which at least self-cleared near the flag), this
   from-entrance run never got a single training clear. The wavefront's dense
   signal was enough to reward *partial* progress but not to carry a full
   traversal to the flag.
3. **Over-consolidation.** Return peaked (iter 500) then fell as entropy
   collapsed (0.73 → 0.31): the policy sharpened onto a non-completing mode.
4. **Recurrence didn't visibly help** at this budget (still 0 clears, greedy
   collapse) — though we can't rule out that it would matter *after* the goal is
   reachable.

## 4. What v2 rules out / refines

- **"Dense wavefront reward alone fixes the exploration/horizon problem"** →
  refuted here: it raised sampled return but neither reached the goal nor
  produced a good greedy policy. It appears partially *farmable* (accumulate
  shaping over the early section without completing).
- **"DQfD demo-anchor of DART recoveries + PPO yields recovery robustness"** →
  insufficient on its own here; the greedy policy still collapses early.
- **"From-entrance training + shaping will traverse the level"** → did not, in a
  1200-iter/48-env budget; the policy never reached the flag from the entrance.
- Still **open**: recurrence's value (untestable until the goal is reachable);
  whether waypoint-restarts (which we deliberately omitted) are actually
  necessary; whether a KL-to-BC anchor (which we omitted) stabilizes the greedy
  mode.

## 5. Refined questions for you (given your recipe failed this way)

1. **Greedy vs sampled:** how do we make the recipe optimize the GREEDY
   (argmax) policy — the eval mode — rather than sampled return? Options we're
   weighing: evaluate/greedy-anneal during training; add an argmax-consistency
   loss; deterministic-policy-gradient-style objective; or gate acceptance on a
   periodic greedy probe (we have this in other modes). Which is SOTA for
   "the deployed mode must be good, not just the exploration mode"?
2. **Farmable shaping:** is our wavefront potential farmable (raising return
   without completion)? PBRS is theory-invariant, but with a *value function*
   and finite training, a dense potential can dominate the sparse terminal.
   Should Φ be scaled down late, or the terminal completion bonus scaled up, or
   the potential clipped near the start? Give a concrete balance.
3. **Never reaching the goal from the entrance:** should we ABANDON
   from-entrance-only and use **backward-curriculum waypoint restarts** (restart
   from states along the solution, walking the start back), *combined* with the
   wavefront + recurrence + DARS? Our prior backward-curriculum runs (without
   wavefront/recurrence) welded the level's back third but stalled at a
   front-half precise obstacle. Would your recipe's components make the backward
   curriculum cross that obstacle? Give the combined recipe if so.
4. **Stabilization we omitted:** does the KL-to-BC constraint (δ=0.015 in your
   §7) materially prevent the greedy collapse we saw? Should the demo-anchor be
   a *sequence*-BC (through the GRU) rather than our per-state DQfD anchor?
5. **The precise obstacle:** 1-2 has a front-half spot (underground gx≈2000–2510)
   that is sticky-fragile even with exact expert actions available. Is there a
   principled way (control-theoretic funnel, local closed-loop stabilization,
   sub-goal) to make a policy robust at a *specific* precise obstacle without
   game internals?
6. **Sanity check on us:** given the greedy policy dies at gx 196 (barely into
   the underground) while sampled return is 1692, is there a likely
   *implementation* bug you'd look for (e.g., train/eval obs-pipeline mismatch,
   GRU hidden-state handling at eval, shaping double-counting), or is this the
   expected behavior of the recipe under-budget? What single diagnostic would
   you run to disambiguate "recipe needs more/other" vs "we have a bug"?

## 6. The one in-pipeline lever we have NOT pulled

**Waypoint-restarts added to this recipe.** We have the full Go-Explore solve
archive (8180 states) and the backward-curriculum machinery; adding restart-from-
waypoints to the recurrent+wavefront+DARS run would let it practice the
flag-finish and late segments it currently never reaches from the entrance. We
held off because (a) the project owner is (rightly) wary of endless variant
cycling, and (b) we want your read on whether this is the missing piece or a
band-aid before spending another multi-hour run. Please advise explicitly.

## 7. Assets now available (in addition to Brief v1's)

- `checkpoints/wavefront/mario_1_2_dmap.pkl` (313-cell distance map).
- `runs/dars_1_2/recovery_demos.npz` (50,410 recovery pairs).
- `checkpoints/mario_1_2_robust_recurrent_wavefront_dars/` (the recurrent run's
  checkpoints — greedy dies at gx 196; a concrete failing artifact to probe).
- The learned 1-1 net (63–67% sticky — the ONE positive example; note it was
  learned by from-scratch reactive PPO with NO shaping/imitation, which is worth
  contrasting against why this heavily-engineered pipeline failed).
- All tools from v1 (`go_explore_solve`, `go_explore_chain`, `distill_level`,
  `robustify_level --accept-sticky`, `generate_dars_recovery`, wavefront module).

---

**The crux for you:** your recipe *moved the sampled policy* but did not produce
a **good greedy controller** and **never reached the goal from the entrance**.
We need the recipe correction that (a) makes the *greedy* policy the training
target, (b) prevents farmable-shaping / over-consolidation, and (c) either makes
from-entrance traversal actually reach the flag or justifies switching to a
waypoint-restart backward curriculum built on top of these components. Concrete
recipe + the one disambiguating diagnostic (bug vs recipe) requested.
