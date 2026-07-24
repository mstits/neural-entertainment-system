# Results: SMB World 1-2 Under the Honest Protocol — a Measured Negative

**Date:** 2026-07-24. **Status:** two of three protocol seeds complete and
concurring; seed 3 in progress (formality). **Ledger:** LEARNED (documented
negative). **Companion documents:** `DOSSIER_V3_2026-07-23.md` (the
elimination record), the SMB 1-2 consulting report (external), and the
CGSA-PPO implementation (`configs/mario_1_2_cgsa.yaml`,
`src/training/trainer.py`).

## The claim this documents

Under the honest evaluation protocol (cold greedy, sticky-0.25, start-jitter
16, from the level entrance, no test-time state loads), **no method in an
extensive, externally-reviewed campaign produced a World 1-2 clear from a
compact feedforward policy on tile observations** — and the failure is now
*characterized*, not merely observed. A literature audit (external deep
research, 2026-07-23) found **no published agent by any method that clears
SMB 1-2 under this protocol**; the level is where reported sticky results
stop. This is an open problem at the frontier of the field, and this
document is our measured contribution to it.

## What was tried (all falsified or eliminated, each with data)

1. **Model-free PPO** + non-farmable wavefront shaping + Go-Explore restarts
   + DQfD margin anchoring (the "JAVI" recipe): plateaus at ~36% depth;
   invariant across four reward regimes after five verified shaping exploits
   were closed (Dossier v3 §2).
2. **The entire imitation space** (Dossier v3 §3): BC at any capacity
   (71→84% agreement ceiling ≈ expert self-consistency), pixel-resolution
   features (null), action-history conditioning (+6.7 points, insufficient),
   GRU sequence-BC (crawling), DAgger/DARS variants. The decisive artifact:
   a perfect clone (train acc 1.000) of a working trajectory scores 0.00
   under sticky-0.25 from that trajectory's own start state.
3. **Backward-ladder welding** (5 variants): small-sample acceptance gates
   shown to be statistically meaningless ("gate mirage": welds passing
   3-episode sticky gates have true success <1/400).
4. **CGSA-PPO** (the external researcher's cell-granular stochasticity
   annealing recipe, implemented faithfully and debugged to full mechanical
   health): the definitive experiment. Seed 0: 1,940 of 2,742 zones
   SPRT-welded locally sticky-robust (α=.01, β=.05), local success p50 =
   1.00 at each zone's annealed noise — **and Signpost 3 composition = 0/30**
   (every episode dead by gx 1827), with the residual ~208-zone gauntlet
   core's noise retreating to p≈0.03 (a measured local sticky ceiling).
   Seed 1: replicates (0/30, uniform deaths at gx 1635). Seed 2: running.

**The one-sentence finding: verified local sticky-robustness does not
compose into segment traversal, and the gauntlet's hard core has a local
sticky ceiling far below the protocol's 0.25 — the failure is a property of
the policy class, not of the training procedure.**

## The measured robustness profile (the honest positive statement)

From the 1-2 entrance, best available learned policy per point, 25 episodes
each, greedy + jitter (`/tmp/robustness_profile.txt`, 2026-07-23):

| p_sticky | clear rate | depth (mean) | depth (best) |
|---|---|---|---|
| 0.00 | 0/25 | 63% | 63% |
| 0.05 | 0/25 | 45% | 65% |
| 0.10 | 0/25 | 41% | 63% |
| 0.15 | 0/25 | 37% | 47% |
| 0.20 | 0/25 | 38% | 63% |
| 0.25 | 0/25 | 39% | 63% |

Note the honest surprise: **no learned-from-entrance policy clears 1-2 even
deterministically** (63% depth at sticky 0). Complete 1-2 traversals exist
only in the EXHIBITION ledger (Go-Explore solutions, routed chains). Any
claim of the form "deterministic traversal + bounded robustness" belongs to
the search system, not to a learned policy, and must be labeled accordingly
per CLAIMS.md.

## What may be claimed (CLAIMS.md-compatible)

- LEARNED: "World 1-1 is learned honestly (63–67% cold greedy sticky+jitter,
  from-scratch PPO)." Unchanged, still the flagship learned result.
- LEARNED (negative): "World 1-2 under the honest protocol is unsolved by
  this project's policy class, with the failure characterized to zone
  granularity: local robustness verified by SPRT at 1,900+ zones does not
  compose, and the gauntlet core (gx 1600–2200) has a measured local sticky
  ceiling ≈0.03–0.05."
- CONTEXT: "No published agent by any method is known to clear SMB 1-2
  under Machado sticky-0.25 from the level start" (audit 2026-07-23).
- EXHIBITION: full-level and multi-level traversals by the search system,
  labeled as search.

## Reproduction

- CGSA runs: `configs/mario_1_2_cgsa*.yaml`; telemetry `[cgsa]` lines in the
  run logs; curriculum state `cgsa_stats.json` per checkpoint dir.
- S3 probe: `/tmp/s3_probe.py <ckpt> 30` (gx1500 → gx2200 under sticky).
- Robustness profile: `/tmp/profile_probe.py <ckpt> <sticky> <eps>`.
- Honest depth probe: `/tmp/probe_javi.py <ckpt> <eps> <seed>`.
- Falsification pre-registration: the consulting report's criteria (3 seeds,
  S1+S2 health, 0/400 entrance success, elevated gauntlet value-MSE).

## Standing follow-ups

1. Seed 2 completes the formal record (no decision rests on it; two seeds
   and the entire elimination history concur).
2. The one untried policy-class lever with a plausible mechanism: a
   recurrent policy trained *inside* the CGSA curriculum (recurrence failed
   only at imitation, never given RL-at-scale with per-zone noise
   annealing). Pre-registered protocol required before any run (Dossier v4).
3. The robustness profile should be re-measured per-level as other levels
   get learned policies, making the level-by-level noise-ceiling map the
   project's standing scientific artifact.
