# Receipts Index — 2026-08-23/24 window

Curated claim-to-receipt map for the two-day push (commits 6e565e5 →
793f640). Every claim below names its ledger (LEARNED / EXHIBITION /
FORGE), its verdict, and its receipt paths relative to the repo root.
VOID and retracted items are listed with their corrections — nothing is
silently dropped. Discrepancies (orphan receipts, claims whose receipts
are missing or uncommitted) are enumerated at the end.

Companion docs: docs/research/{RECURRENT_AB_VERDICT_2026-08-23,
V26_ADJUDICATION_2026-08-23, RECOVERY_ASSAY_VERDICT_2026-08-24,
OPTIONS_NEGATIVE_2026-08-23, PROCESS_AUDIT_2026-08-23}.md and
docs/proposals/RECOVERY_DISTILL_1_1_2026-08-24.md.

---

## 1. Odometer campaign (FORGE + EXHIBITION)

**CLAIM (FORGE): PPU scroll odometer shipped in-core and certified 5/5**
— commit 3429d8a; loopy_v per-line sampling, modal filter, wrap-aware
fold, v3 savestate envelope.
- Certification: `runs/odometer/cert_smb_2026-08-23.json` (5/5 checks:
  hold-forward monotonic dx=551/0 regress, HUD-split immunity,
  hold-still flat, restore-exact, no restore discontinuity).
- Spec: `docs/proposals/ODOMETER_CORE_SPEC_2026-08-23.md`.

**CLAIM (FORGE): progress-signal gate verdicts** — Rygar SOUND (117
distinct/470px), Ninja Gaiden SOUND (126/1384px), Contra SOUND and
cross-validated against the RAM pair (odometer 162 vs RAM 163 distinct),
Kung Fu UNUSABLE-as-camera-static = skill wall, not instrument fault.
- `runs/odometer/gate_rygar_2026-08-23.json`
- `runs/odometer/gate_ninja_gaiden_2026-08-23.json`
- `runs/odometer/gate_contra_2026-08-23.json`
- `runs/odometer/gate_kungfu_2026-08-23.json`,
  `runs/odometer/gate_kungfu_left_2026-08-23.json` (OAM churn 540 =
  agent active, camera static).

**CLAIM (FORGE): scene detection shipped** — commit 741a953, v4
envelope; scene ordinal keys Go-Explore cells; NG boss-room aliased
dx=−511 wall root-caused.
- Headline receipt: `runs/ng_odo_scene/` (max_area 8, 8 scenes crossed
  in 12 min vs 2.5 h pinned at 6144 pre-scene).

**CLAIM (EXHIBITION, depth-only): NG and Rygar are gate-SOUND, and
their clear status is UNMEASURED** — scene keying moved the frontier
(area 0 → 9, best_score 74783). No clear predicate was wired in any of
the 15 odometer-celled probe runs, so none of them could bank a
solution; this claim is about frontier depth only.

> **CORRECTION 2026-08-26.** Previously: *"NG and Rygar are gate-SOUND
> but UNSOLVED — … 0 solutions in all 15 odometer-celled probe runs."*
> That "0 solutions in all 15 runs" is one constant reported fifteen
> times, not fifteen independent negatives. All 15 ran on profiles
> shipping `level_key: []` with no `clear:` block, which makes
> `GenericGame.is_clear`'s opening test `() > ()` — False for every RAM
> state. **The "gate-SOUND" half stands**: that gate is the odometer,
> certified separately 5/5 and untouched by this. What is struck is the
> word UNSOLVED and the solution count behind it — neither game is
> scorable as solved or unsolved, because the question was never asked.
> Source: `docs/research/CLEAR_DETECTION_CAMPAIGN_2026-08-26.md`.
- Pre-scene plateaus: `runs/ng_odo_night/` (3 h, 17.7M steps, best
  6144), `runs/rygar_odo_night/` (3 h, 21.5M steps, 5680px); also
  `runs/{ng,rygar}_odo_{smoke,v2,debounce,deep}/`.
- Scene-keyed: `runs/ng_odo_scene_long/`, `runs/ng_odo_scene_long2/`
  (4 h each), `runs/ng_odo_throttle/`, `runs/ng_odo_doa/` (max_area 9,
  max_sect 7, 0 solutions).
- Log tails: `runs/odometer/{ng_deep,rygar_deep,rygar_v2}.log`
  committed (161a20b); `runs/odometer/{ng_night,ng_scene_long,
  ng_scene_long2,rygar_night}.log` exist on disk, uncommitted
  (see discrepancy #2).

**CLAIM (FORGE): two generic death-detector fixes** — (a) death
debounce ≥3 consecutive dead observations (Rygar door transition-blip),
commit baa3ac7; (b) wrap-aware modular lives decrement (NG 0→255
underflow hid every death), commit 6bf0dfd.
- Debounce falsifier: `runs/rygar_odo_debounce/` (gx 1536 → 5360 in
  6 min) — on disk, uncommitted (discrepancy #7).
- NG underflow probe: inline in commit message only (discrepancy #10).

**CLAIM (FORGE): dead-on-arrival cell retirement** — commit 49fe332.
- Receipt run: `runs/ng_odo_doa/` — exists, uncited by any doc
  (discrepancy #2).

## 2. GRU / recurrent-bottleneck A/B (LEARNED, negative)

**CLAIM: recurrent policy-class treatment FAIL — mechanism untested** —
treatment (TileRecurrentPolicyNetwork, 48,975 params) best-of-4 honest
sticky **0.06** (seeds 0.00/0.06/0.01/0.01, 100 eps each) vs banked
control **0.76**; deterministic ≈ sticky ≈ 0 on every seed = learning
failure, not robustness failure. FAIL not VOID: policy class armed,
hidden-reset audit clean (trainer.py:6576, trainer.py:7461).
- Verdicts: `runs/gru_ab/verdict_seed{0,1,2,3}.json`
- Training: `runs/gru_ab/train_seed{0,1,2,3}.log`,
  `checkpoints/mario_1_1_backward_gru_seed{0..3}/`
- Control (banked, not re-run):
  `checkpoints/_preserved/backward_1_1_seed3_iter140.pt` (honest 0.76).
- Docs: `docs/proposals/RECURRENT_BOTTLENECK_AB_2026-08-23.md`
  (pre-registration), `docs/research/RECURRENT_AB_VERDICT_2026-08-23.md`.
- RETRACTION recorded in verdict doc: the registration's acceptance
  that BC-through-stateless-fallback "cannot explain a between-arm
  difference" was wrong.

**CLAIM: sticks are detectable statelessly (v25 POMDP premise
weakened)** — stick-detection probe AUC, banked-control policy:
- `runs/gru_ab/stick_probe.json` (4-frame stack: MLP 0.83 / GRU 0.87)
- `runs/gru_ab/stick_probe_nostack.json` (single frame: 0.77 / 0.82)
- SUPERSEDED for the recurrence-delta by the post-loader-fix re-run:
  `runs/gru_ab/stick_probe_realpolicy.json` (real policy,
  divergent-stick base rate 0.0575: MLP 0.76 / GRU 0.74 — recurrence
  adds nothing; the GRU edge flipped sign). The 0.83/0.87 matrix quoted in
  `docs/research/V26_ADJUDICATION_2026-08-23.md` and the v25 memory
  predates the fix and is not yet annotated (discrepancy #11).

**CLAIM: v26 gate PASS, strategically overridden** — detection was
never the binding constraint; recurrent-RL overhaul declined.
- Doc: `docs/research/V26_ADJUDICATION_2026-08-23.md`.

## 3. Recovery assay (LEARNED measurement, solver-adjudicated)

**CLAIM: 1-1 sticky wall decomposed — fatal window real, trainable
slice real** — 60 honest eps reproduced banked 0.767; 1,592 post-stick
snapshots; timeout sticks 5/5 recovered (sanity), true death-sticks 3/9
(33%), sticks 1–2 steps pre-death 0/3 → honest ceiling ~0.83–0.85 for
ANY policy class.
- `runs/recovery_assay/manifest.json`, `runs/recovery_assay/verdict.json`
- Recovery tapes: `runs/recovery_assay/solve_ep*/`
- `runs/recovery_assay/probe_ep15_10min/` — the manual probe that
  exposed the first pass's too-short budget.
- Doc: `docs/research/RECOVERY_ASSAY_VERDICT_2026-08-24.md` (0d114bd).

**INTEGRITY RETRACTION (recorded): first pass scored 0/14 TWICE** —
(a) 3-minute solver budget too short, (b) stdout-grep success detector
that could never fire. Both fixed in `scripts/recovery_assay.py`; final
verdict filesystem-scored (ground truth = solutions/ dirs).
- Debris of the void first pass: `runs/recovery_assay_bad/` (manifest +
  states only, no verdict; its state paths point at
  `runs/recovery_assay/`) — uncited orphan (discrepancy #1).

**CLAIM: 1-2 wall is mostly physics — BANKED verdict gains a
mechanical explanation** — collection run cleared 25/60 per
`manifest.json` (the verdict doc's baseline-reproduction figure is
0.367); 16/35
death-sticks adjudicated, 3/16 recovered (19%); 11/16 sticks ≤4 steps
pre-death → honest ceiling ~0.53; banked ~0.37–0.40 sits near it.
- `runs/recovery_assay_1_2/verdict.json`, `manifest.json`,
  `solve_ep*/` (commit d44bbe2).

**ROUTING RULE (binding, new): run this assay before spending training
effort on any level's sticky rate** — the recoverable share is the
budget-worthiness signal.

## 4. Distillation gauntlet (LEARNED, negative ×3 → isolated-optimum meta-finding)

Registration + all verdicts:
`docs/proposals/RECOVERY_DISTILL_1_1_2026-08-24.md`.

**VOID (retracted): first distill attempt trained a random net** —
`build_tile_policy_from_checkpoint` silently returned a RANDOM net for
path inputs; loader fixed in commit 38f2358 (root cause of every
08-24 standalone-loop anomaly, including the retracted "0/60
unjittered argmax-tie receipt"). Receipt = test counts + reproduction
at pre-odometer commit; suite 3687 passed.

**CLAIM: base BC distill FAIL-by-drift at epoch 0** — honest greedy
0.767 → 0.033 after 13 Adam steps; sampled 0.17.
- `runs/recovery_distill/train_history.json` as committed (6d84cdf:
  epoch-0 clear 0.033, loss 3.35) — the working-tree copy was later
  overwritten by variant A rung 2's history (uncommitted);
  `runs/recovery_distill/ckpts/` (preserve-on-peak
  `ckpts/distill_epoch00.pt`).
- Fuel/anchors: `runs/recovery_distill/fuel/`,
  `runs/recovery_distill/demos/` (82 files incl. anchor_selfplay.npz).

**CLAIM: variant A (KL-anchored cloning, 2-rung LR ladder) FAIL both
rungs** — lr 1e-4 drift-stop at epoch 0; lr 1e-5 best 0.70 < baseline
0.767 < gate 0.80. Cross-entropy on solver recovery actions is
net-destructive at every tested strength.
- Rung-2 history is the current working-tree `train_history.json`
  (epoch 0 0.70 / epoch 1 0.60, uncommitted overwrite of the
  committed base-run receipt); rung-1 history only in task logs
  (discrepancy #4).

**CLAIM: variant B (on-policy recovery PPO from 27 mined post-stick
states, KL-anchored) FAIL** — honest by ckpt: iter10 0.33 → iter30 0.0;
recovery-pool clears 0.49 → 0.30; entrance rate 0.31 → 0.06.
- `runs/recovery_distill/variant_b_train.log` + ckpts — on disk,
  uncommitted at verdict time (discrepancy #5).

**META-CONCLUSION: the consolidated 48k 1-1 artifact is an ISOLATED
OPTIMUM** — every gradient touching it makes it worse; post-hoc
improvement CLOSED on this artifact; 1-1's honest number remains the
untouched control's 0.767. Registered next: v27 fresh-run with recovery
states in curriculum from the start (not run).

## 5. Onboarding wave 1 (EXHIBITION, IN FLIGHT)

Launched 08-24 17:45 by the Wednesday Push orchestration (793f640,
`docs/proposals/WEDNESDAY_PUSH_2026-08-24.md`). No verdict doc yet — by
design, but wave-1 verdicts will need one when the lane lands
(discrepancy #13).
- Progress-signal gates (21 JSONs):
  `runs/onboard_wave1/gate_*.json` — SOUND-still-advancing:
  castlevania 905/3055, excitebike 1116/11062, gradius 940/1714;
  SOUND-game-stops: ducktales, ghosts_n_goblins, megaman2 (+rampair),
  metroid-right; camera-static: bubble_bobble (agent inert),
  double_dragon (agent active), kirby_left; UNUSABLE: kid_icarus (all
  4 drivers), punchout (both), kirby main, metroid_left.
- Solver smokes (writing at index time): `runs/onboard_wave1/smoke_*/`
  (bubble_bobble 2 solutions, castlevania 3 at last check).

## 6. Adjacent adjudications in window (options, shelf) — for completeness

**CLAIM (LEARNED, negative): options treatment FAIL by overcommitment**
— control 8/100, treatment 0/100 strict honest on 1-2; 93.6% of 4,000
real-state decisions chose k=4. Preceded by a VOID (frozen-actor arms,
commit 28fd32c) that this rerun corrected.
- `runs/options/rerun2_eval_control_seed{7,101}.json`,
  `runs/options/rerun2_eval_treatment_seed{7,101}.json`,
  `runs/options/verdict.json`
- Doc: `docs/research/OPTIONS_NEGATIVE_2026-08-23.md`.
- Side finding (measured): continued PPO collapsed the consolidated
  control 31/100 → 8/100 in 200 iters → preserve-on-peak mandated for
  all future A/Bs. Banked 38/100 preserved checkpoint untouched.
- Options integration smoke claims in b52b0b1 (iter 1121, 3,302 sps)
  carry no receipt file (discrepancy #8).

**CLAIM (LEARNED): shelf dispositions answered** — joint policy 32/100
(1-1) / 1/100 (1-2): naive pooling stays falsified; 1-4 endpoint
51/100 = exactly the banked rate (0.633 probe = winner's curse).
- `runs/engine/logs/shelf_joint_1_1_seed{7,101}.log`,
  `runs/engine/logs/shelf_joint_1_2_seed{7,101}.log`,
  `runs/engine/logs/shelf_1_4_endpoint_seed{7,101}.log` — on disk,
  uncommitted (discrepancy #6).
- Doc: `docs/research/PROCESS_AUDIT_2026-08-23.md` (shelf section).

## Standing LEARNED scoreboard after this window

1-1 0.767 (untouched control; ceiling ~0.83–0.85 measured); 1-2 38/100
banked (ceiling ~0.53 measured, BANKED with mechanical explanation);
1-3 21/100; 1-4 51/100 (endpoint re-confirmed). GRU class 0.06 FAIL;
options 0/100 FAIL; post-hoc training on the consolidated 1-1 artifact
CLOSED.

---

## Discrepancies

### Orphan receipts (exist on disk, no doc cites the path)

1. `runs/recovery_assay_bad/` — void first-pass debris (manifest +
   states, no verdict; state paths point into `runs/recovery_assay/`).
   Should be named in the verdict doc's retraction note or deleted;
   silent orphan is the one state it must not stay in.
2. 14 of 15 odometer-celled probe dirs — only `runs/ng_odo_scene` is
   cited anywhere. Uncited: `runs/ng_odo_{smoke,v2,debounce,deep,night,
   scene_long,scene_long2,throttle,doa}` and `runs/rygar_odo_{smoke,v2,
   debounce,deep,night}`. Partially mitigated by log tails
   (`ng_deep,rygar_deep,rygar_v2` committed in 161a20b;
   `ng_night,ng_scene_long{,2},rygar_night` on disk, uncommitted),
   but `ng_odo_doa` (DOA-retirement receipt for 49fe332) and both
   4-hour scene_long runs are pathless in docs.
3. `runs/odometer/` itself — cited only in session memory
   (project_odometer_shipped_2026-08-23.md), not in any docs/ file;
   the shipped-5/5-certified claim had no in-repo doc naming its
   receipts until this index.

### Unreceipted or uncommitted claims (headline numbers whose receipt files are absent from git or absent entirely)

4. Variant A FAIL (commit 6e6fd54) — doc-only commit; rung-1 history
   lives only in task logs, and rung-2's history exists solely as an
   uncommitted working-tree overwrite of
   `runs/recovery_distill/train_history.json` (banking it as-is would
   clobber the committed base-run history, 6d84cdf). **No committed
   receipt file exists for either rung.**
5. Variant B FAIL (commit e5d702d) — doc-only commit;
   `runs/recovery_distill/variant_b_train.log` + ckpts exist on disk
   but were uncommitted at verdict time. Bank them.
6. Shelf evals 32/100, 1/100, 51/100 (commit c39c9ee) —
   `runs/engine/logs/shelf_*.log` exist on disk, uncommitted.
7. Rygar death-debounce falsifier (commit baa3ac7) —
   `runs/rygar_odo_debounce/` exists on disk, uncommitted.
8. Options integration smoke (commit b52b0b1: resumes iter 1121,
   3,302 sps, converted seed clears) — no receipt file anywhere.
9. Rygar odometer-consumer smoke (commit 1021077: 307 cells, gx 1536
   in 2 min) — inline claim only; partially covered by later-banked
   `runs/odometer/rygar_*.log`.
10. NG doomed-cell probe (commit 6bf0dfd: hold-right frozen, t=41) —
    inline claim only, no file.

### Stale annotations (correct at write time, superseded by post-fix data, not yet marked)

11. Stick-probe AUC matrix 0.77/0.83/0.82/0.87 ("recurrence adds
    ≈+0.05") in `docs/research/V26_ADJUDICATION_2026-08-23.md` and
    memory project_v25_verdict_recurrent_bottleneck.md predates the
    38f2358 loader fix; post-fix real-policy receipt
    (`runs/gru_ab/stick_probe_realpolicy.json`: MLP 0.76 / GRU 0.74)
    flips the GRU edge's sign. Direction of the adjudication is
    strengthened, but the figures should be marked pre-fix/superseded.
12. MEMORY.md index line "1-1 → ~0.83–0.85 possible" understates the
    file's final verdict: all salvage families FAIL, post-hoc
    improvement CLOSED on the consolidated artifact — reaching the
    ceiling requires a fresh run (v27), not touching the banked policy.

### In flight (doc pending by design)

13. `runs/onboard_wave1/` — wave-1 gates + smokes writing now; needs a
    verdict doc when the Wednesday Push onboarding lane lands.
