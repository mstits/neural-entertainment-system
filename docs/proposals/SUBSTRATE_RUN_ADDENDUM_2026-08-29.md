# Substrate run addendum — registered before any collection or training compute

**Date:** 2026-08-29. **Status:** registered; no S1/S2/S3 compute spent.
**Base registration:** `scripts/eval_shared_substrate.py`'s in-file CONFIG
(mirrored by `tests/test_eval_shared_substrate.py`, 58 tests) and
`scripts/train_shared_substrate.py`'s documented pipeline. This addendum
closes three holes found in cross-session review BEFORE compute; the
reviewing session (personal-os-6c) independently re-derives the final
verdict from raw receipts, against THIS addendum, having run none of it.

## 1. Threshold correction (review Hole A) — the 5-point protocol bias

CONFIG's `baseline_sum = 153` (43+38+21+51) embeds 1-2's banked 38/100,
measured under a protocol the harness itself marks
`matches_harness_protocol: False` (shared-stream, 1 worker, no
sequential/level-clear). The like-for-like figure exists, receipted:
`runs/interference/interference.jsonl`, row `{type: probe, level: 1-2,
role: control}` — the identical `_preserved/consol2_40pct_strict_
iter01120.pt` checkpoint under the harness's exact protocol shape
(sequential, level-clear, workers 5, per-episode RNG, sticky 0.25,
jitter 16, stage_03 entrance), 100 episodes at eval-seed 20260816:
`clear_rate = 0.43` → **43**. The seed/episode split (1×100 vs 2×50)
differs from the harness's re-eval split in the same way 1-1's banked
leg does, which the harness's own provenance block records as
immaterial to the count. (`clear_rate_strict = 0.42` in the same row is
a different, stricter estimator; the harness scores `clear_rate` — see
`test_strict_clears_uses_clear_rate_not_seq_clear_rate` — so 43 is the
like-for-like number.)

**Registered adjudication threshold: `baseline_sum_adj = 43+43+21+51 =
158`, re-derived at adjudication time from the four named receipts, not
quoted from here.** CONFIG and its pinned tests stay byte-identical
(153 is history); the harness will print its CONFIG verdict, and the
BANKED verdict is the one against 158:

- `shared_sum > 158` and nothing collapsed → SUPERSEDES.
- `153 < shared_sum ≤ 158` → **MIXED-BY-ADDENDUM** (indistinguishable
  from the protocol artifact), even if the harness prints SUPERSEDES.
- Everything else: the harness's registered classes unchanged.
- The registered exact-binomial significance gate applies against 158.

Choosing among threshold options after seeing the delta is off the
table; this section is that choice, made now.

## 2. Training-set composition (review finding: unregistered researcher DoF)

- **1-1:** `runs/interference/success_1_1.npz`, sha256 `6476eea9b179e0fc…`
  (80,327 pairs). **1-2:** `runs/interference/success_1_2.npz`, sha256
  `d1e1fb1017e59f23…` (61,245 pairs). Both as collected 2026-08-16;
  no re-collection.
- **1-3, 1-4:** collected via `interference_falsifier`'s imported
  collection loop (`night2_runner.collect_trajectories` mechanics),
  strict-episode-success-only, with the falsifier's registered knobs
  verbatim: 300 episodes, max_steps 3000, sticky 0.25, jitter 16,
  sampled T=1.0, per-episode RNG, collect_seed 20260815, 5 lanes.
  Specialists = the eval harness's own banked baseline specs
  (`one_three_FINAL_consol2_iter00690.pt` / `mario_1_3_online_v1`;
  1-4's banked spec likewise). Outputs:
  `runs/substrate_pairs/success_1_{3,4}.npz`, sha256s recorded in the
  collection receipts at write time.
- **Balance:** the trainer's pairwise trim to the smallest level, as
  documented in its own header. No re-weighting.
- **Hyperparameters:** the trainer's defaults verbatim — epochs 50,
  lr 3e-4, batch 256, hidden_dim 256, trunk_dim 64, seed 0,
  val_frac 0.1. **One training run.** Any change after seeing any
  result — including per-level BC accuracy — is a new registration.
- BC accuracy is a diagnostic, never a verdict input (trainer header;
  same proxy-metric discipline as CLAIMS.md's 3-episode-gate rule).

## 3. Claim scope (review Hole C) — interference only, not transfer

Nothing is held out: the trunk trains on success pairs from the same
four levels it is scored on. A SUPERSEDES therefore licenses exactly:
*"one shared trunk fits all four levels under the honest gate without
the pooled-net collapse (pooled 1-2: 4/100 vs specialist 42/100)."*
It licenses **no transfer claim** of any kind. The transfer question —
frozen trunk, new head trained on a held-out level's data, versus that
level's specialist — is named as follow-on work requiring its own
registration before any of it runs.

## 4. VOID licence

A VOID (any leg errored / UNUSABLE / collection failure) licenses only
the sentence "the substrate run did not measure interference," confirms
and refutes nothing, moves no bar, and retires nothing.

## 5. Compute plan (for the operator's go)

S1: two collections (1-3, 1-4), 300 eps each, 5 lanes — emulator-bound.
S2: offline BC, no emulator, minutes (dry-run on existing data passes).
S3: the registered eval, 4 levels × 2 seeds × 50 eps = 400 episodes.
Emulator held via `src/utils/run_lock` CLI throughout S1/S3; nothing
else steps the pool concurrently. Roles: executing session runs S1–S3;
reviewing session independently re-derives per-leg strict clears, the
aggregate, the binomial, and the verdict class against §1 from the raw
receipts before anything is banked.

## ADDENDUM 2 (2026-08-29, post-result, pre-control) — the design was confounded, and the control that isolates sharing

Registered AFTER the main run's verdict (FAILED, shared_sum 42 vs 158,
all four levels collapsed, dual-derived independently by both sessions
with exact agreement) and BEFORE any control compute. Found in the
verifier's independent read: **the main run changed two variables at
once** — the shared arm is offline BC on success pairs; the baseline
arm is online-RL specialists with curricula (all four `_preserved`
checkpoints carry optimizer + curriculum state). A FAILED is therefore
fully explained by "offline BC at this data size does not reach
RL-specialist performance," and licenses NO interference claim. (The
original pooled falsifier shared this structure but was rescued by its
within-run contrast — pooled BC took 1-1 ABOVE its specialist, 52 vs
43, which training method cannot explain. This run produced no such
contrast: nothing was captured.)

**What the main run banks, exactly:** *a shared-trunk multi-head policy
trained by offline BC on banked success pairs does not approach
RL-trained specialists on any of the four levels (42 vs 158, 1-2 and
1-3 at zero).* Interference: UNTESTED by that run.

**The control, registered now:** four SEPARATE nets, one per level —
identical trainer, identical defaults (epochs 50, lr 3e-4, batch 256,
hidden 256, trunk 64, seed 0, val_frac 0.1), identical eval — with each
level's training set pre-trimmed to its LEADING 37,834 pairs, the exact
per-level exposure the shared run's pairwise balance produced (same
leading-order trim rule; trimmed-file SHAs recorded in the trim receipt
at write time). The single changed variable vs the main run is
SHARING.

**Decision rule, fixed now.** Let S_sep = the four separate nets'
summed strict clears under the identical registered eval:
- **S_sep − 42 ≤ 10** (one collapse margin): separate-trunk BC collapses
  too → the method explains the main result; **interference untested**,
  and the generalist question moves to on-policy/corrective designs.
- **S_sep − 42 > 10 AND every level's separate net clears its
  like-for-like baseline minus the collapse margin**: BC per se
  suffices → the main run's collapse is attributable to sharing;
  **interference demonstrated** at this recipe.
- **Anything between**: PARTIAL — per-level pattern reported, no
  aggregate interference claim; any further claim needs a new design.
- Advisory exact binomial on S_sep vs 42 and vs 158, reported beside,
  never substituting.
- VOID licence: a VOID control licenses only "the control did not
  measure BC-sufficiency," and the main run's banked sentence above
  stands unaffected.

Roles unchanged: executor runs, verifier re-derives blind from raw
receipts before anything is banked.
