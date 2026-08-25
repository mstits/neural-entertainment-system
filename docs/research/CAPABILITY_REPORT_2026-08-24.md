# Capability report — 2026-08-24 vs two weeks ago

Window: 2026-08-11 → 2026-08-24. Audience: project owner reviewing two
weeks of work. Every claim below carries its receipt path and its
ledger tag — LEARNED (honest-eval learning results), EXHIBITION
(solver/search results), FORGE (mechanisms built). FAILs, VOIDs, and
retractions are reported as first-class results: catching them is a
capability of the process, and this window produced more adjudicated
negatives than positives. Nothing here is silently dropped; the
retraction register is at the end.

Baseline (window start, 2026-08-11): the learning track had been
reopened three days (project_learning_reopen_2026-08-08 — B1–B3
baselines, flagship 0.65 pooled reproduction); no per-level banked
honest scoreboard existed yet. The solver had SMB game-complete
(EXHIBITION, 2026-07-27) and Castlevania blocks 0–1; Rygar, Ninja
Gaiden, and Kung Fu were blocked at the gate level — no game-agnostic
progress signal existed, deaths in Rygar/NG were invisible to the
solver, and scene changes were not detected at all.

---

## 1. Instrument capabilities [FORGE]

### 1.1 PPU scroll odometer — did not exist → shipped in-core, certified 5/5

Two weeks ago the odometer was a reference note (wideNES write
interception, "candidate fix for Rygar/KungFu/NG"). Today it is a
core-resident instrument: loopy_v + fine_x sampled per scanline at
dot 256, modal filter over 240 scanlines, wrap-aware fold
dX=((Xc−Xp+256) mod 512)−256 with the attribute-table trap handled,
accumulator inside the savestate blob (v3 envelope). Shipped commit
c556e40; spec docs/proposals/ODOMETER_CORE_SPEC_2026-08-23.md; 623
Rust tests green at ship.

Certification (scripts/odometer_cert.py, receipt
runs/odometer/cert_smb_2026-08-23.json): 5/5 — hold-forward monotonic
(dx=551, 0 regressions), HUD-split immunity, hold-still flat,
restore-exact across savestate, no restore discontinuity. Fail-any →
build quarantined.

Per-game progress-signal gate, before → after (receipts:
runs/odometer/ gate JSONs, committed c3d7405):

| Game         | 2026-08-11                      | 2026-08-24 |
|--------------|---------------------------------|------------|
| Rygar        | UNUSABLE (no RAM scalar found)  | SIGNAL SOUND — 117 distinct cells / 470 px |
| Ninja Gaiden | UNUSABLE                        | SIGNAL SOUND — 126 distinct / 1,384 px |
| Contra       | RAM-pair signal (163 distinct)  | odometer 162 distinct / 635 px — cross-validated within one bin |
| Kung Fu      | UNUSABLE, cause unknown         | 1 distinct — correctly diagnosed: camera static, agent ACTIVE (OAM churn 540) = SKILL wall, not instrument fault |
| SMB          | RAM x-scalar (game-specific)    | certification target, 5/5 |

The Kung Fu row is the instrument doing its second job: a flat
odometer measures the CAMERA; OAM churn splits agent-active from
agent-inert. That diagnostic rule is now in memory as binding.

### 1.2 Scene detection — did not exist → shipped, envelope v4

Commit 074f888 (2026-08-24): rendered-cut scene detection (bipartite:
masked FNV-1a frame hash AND scroll discontinuity together), scene
ordinal keys Go-Explore cells, v4 savestate envelope, 626 Rust tests
green. Measured effect: the NG boss room aliases dx=−511 and pinned
the frontier at gx 6144 for 2.5 hours; with scene keying the solver
crossed 8 scenes in 12 minutes (receipt runs/ng_odo_scene/). This is
the live validation of the Zelda/Metroid room-navigation architecture
(docs/proposals/MEMORY_ARCHITECTURE_2026-08-23.md, L2).

### 1.3 Death semantics — two generic detector fixes + DOA retirement

Both game-agnostic (purity-clean: hardware/telemetry surface only):

- Transition-blip debounce (commit 1610093): death requires ≥3
  consecutive dead observations. Rygar's door transitions blip lives
  through 0 for 2 steps; before the fix every door was a false death.
  Falsified live: Rygar gx 1536 → 5360 in 6 minutes (3.5× depth).
  Receipt runs/rygar_odo_debounce/ (on disk, uncommitted — see §7).
- Wrap-aware lives decrement (commit 084362c): (start−cur) mod 256 in
  1..8. NG's 0→255 underflow previously hid EVERY death; probe showed
  hold-right frozen at t=41 with no death registered.
- Dead-on-arrival cell retirement (commit 6a47a71, receipt
  runs/ng_odo_doa/): archive cells whose restores die immediately are
  retired instead of resampled forever.

### 1.4 Gate instrumentation at scale — wave 1 (in flight)

runs/onboard_wave1/ (21 gate JSONs, launched 2026-08-24 17:45,
commit 787cf3e lane): progress_signal_gate now classifies ten games
in one pass — SOUND-still-advancing: Castlevania 905 distinct/3,055 px,
Excitebike 1,116/11,062, Gradius 940/1,714; SOUND-then-game-stops:
DuckTales 104/319, Ghosts 'n Goblins 117/285, Mega Man 2 91/808,
Metroid-right 87/508; camera-static: Bubble Bobble (agent inert),
Double Dragon (agent active), Kirby-left; UNUSABLE: Kid Icarus (all 4
drivers), Punch-Out (both), Kirby main, Metroid-left. Verdict doc
pending by design — the lane has not landed.

---

## 2. Search capabilities [EXHIBITION]

Frontiers reached, with the honest caveat up front: on the two new
odometer games the frontier MOVED but neither game is solved — 0
solutions in every ng_odo_*/rygar_odo_* run.

- Ninja Gaiden: pre-scene best gx 6144 (runs/ng_odo_night: 3 h,
  17.7 M steps — the boss-room aliased-dx wall). Post-scene: max_area
  9, max_sect 7, best_score 74,783 (runs/ng_odo_scene_long and
  _long2, 4 h each; ng_odo_throttle and ng_odo_doa are shorter
  follow-ups). 8 scenes in 12 min once
  scene keying landed. Remaining wall: knockback-death gauntlet —
  a real skill wall, queued for longer runs.
- Rygar: best 5,680 px (runs/rygar_odo_night: 3 h, 21.5 M steps); the
  death-debounce fix alone took 6-minute depth from gx 1536 to 5360.
- Kung Fu: no search progress — correctly reclassified a skill wall
  (camera static, agent active), not an instrument gap.
- Wave-1 smokes (in flight, runs/onboard_wave1/): Bubble Bobble
  already 2 solutions, Castlevania 3, in 6-minute × 3-worker smokes.
- Solver as adjudicator (new use of search): 10-min 8-worker
  Go-Explore runs from post-stick snapshots adjudicated the recovery
  assays (§3.1) — 5/5 timeout sticks recovered, 3/9 true death-sticks
  on 1-1; 3/16 on 1-2. Receipts runs/recovery_assay/solve_ep*/.

Two weeks ago none of the NG/Rygar numbers were reachable: no
progress signal, invisible deaths, no scene identity. The search
capability delta is almost entirely the instrument delta.

---

## 3. Learning-science state [LEARNED]

This is the section where most results are negative. They are the
product of the window: four registered experiments ran to adjudicated
verdicts, and the sticky wall now has a measured decomposition
instead of a hypothesis.

### 3.1 The sticky-wall decomposition (recovery assays)

docs/research/RECOVERY_ASSAY_VERDICT_2026-08-24.md; receipts
runs/recovery_assay/{manifest,verdict}.json, runs/recovery_assay_1_2/
verdict.json.

1-1 (banked 0.76 control; the collection run itself reproduced
0.767 over 60 eps): 1,592 post-stick snapshots; 14 non-clear
episodes adjudicated by solver. Timeout sticks 5/5 recovered
(sanity). True death-sticks 3/9 recovered (33%) — a real trainable
slice, with the recovery tapes as curriculum fuel. Sticks 1–2 steps
pre-death 0/3 recovered — a real FATAL window: under sticky-0.25 the
honest ceiling is strictly <1.0 for ANY policy class. Arithmetic:
perfect recovery training moves 1-1 from 0.767 to ~0.83–0.85. That
ceiling is now a measured number, not a hope.

1-2 (banked consol2; the collection run cleared 25/60 by its
manifest, alongside the verdict doc's 0.367 baseline reproduction): 16
death-sticks adjudicated, 3 recovered (19%); 11/16 sticks were ≤4
steps pre-death (69%) → implied honest ceiling ~0.53. This is the
mechanical explanation for the 1-2 BANKED verdict: the wall is
mostly PHYSICS at p=0.25, and the banked 0.37–0.40 already sits near
its ceiling. 1-2 stays closed, now with a reason.

### 3.2 The recurrent-bottleneck line: premise tested, premise dead

- DR v25 claimed the sticky wall is a POMDP artifact and chose a
  policy-class change (GRU). Pre-registered A/B
  (docs/proposals/RECURRENT_BOTTLENECK_AB_2026-08-23.md): treatment
  48,975 params vs banked control 0.76, single-variable diff.
- VERDICT: FAIL — treatment best-of-4 honest sticky 0.06
  (0.00/0.06/0.01/0.01) vs control 0.76; deterministic ≈ sticky ≈ 0
  on every seed, so this is a LEARNING failure and the mechanism was
  never tested. FAIL not VOID: policy class verified armed, hidden-
  reset audit clean (trainer.py:6576, trainer.py:7461). Receipts
  runs/gru_ab/verdict_seed{0..3}.json;
  docs/research/RECURRENT_AB_VERDICT_2026-08-23.md.
- Stick-detection probe (v26's gate): sticks are detectable
  STATELESSLY. Post-loader-fix real-policy numbers: MLP AUC 0.76 vs
  GRU 0.74 — recurrence adds NOTHING (receipt
  runs/gru_ab/stick_probe_realpolicy.json). The earlier matrix
  (0.83/0.87 etc.) predates the 28dc163 loader fix and is
  superseded. v26's PASS-branch authorization of a recurrent-RL
  overhaul was overridden with this data: detection was never the
  binding constraint (feedforward had AUC-0.83-grade stick access at
  the old plateaus). docs/research/V26_ADJUDICATION_2026-08-23.md.

### 3.3 The isolated-optimum finding (three salvage families, all FAIL)

docs/proposals/RECOVERY_DISTILL_1_1_2026-08-24.md, verdicts appended:

- Naive CE distillation on solver recovery actions: FAIL-by-drift at
  epoch 0 — honest greedy 0.767 → 0.033 after 13 Adam steps at lr
  1e-4 (sampled 0.17: genuine degradation, not argmax-tie noise).
  Receipts runs/recovery_distill/train_history.json as committed
  (8bded0d); the working-tree copy was later overwritten by variant
  A's rung-2 history (uncommitted).
- Variant A, KL-anchored cloning, 2-rung LR ladder: FAIL both rungs
  (lr 1e-4 drift-stop at epoch 0; lr 1e-5 best 0.70 < baseline
  0.767 < gate 0.80). Cross-entropy on recovery actions is
  net-destructive at every tested strength.
- Variant B, on-policy recovery PPO from 27 mined post-stick states,
  KL-anchored, 60 iters: FAIL (honest 0.33 @ iter10 → 0.0 by
  iter30; recovery-pool clears 0.49→0.30; entrance rate 0.31→0.06).
  Receipt runs/recovery_distill/variant_b_train.log.

META-CONCLUSION: the consolidated 48k-param 1-1 artifact is an
ISOLATED OPTIMUM — every gradient that touches it makes it worse.
Post-hoc improvement of this artifact is CLOSED. 1-1's honest number
remains the untouched control's 0.767.

Corroborating measurement from the options experiment: continued PPO
collapsed a consolidated 1-2 peak 31/100 → 8/100 in 200 iters with
the KL anchor active (−74% relative). Consolidated peaks are
transient under further training; preserve-on-peak is now mandatory
in both arms of every A/B (docs/research/OPTIONS_NEGATIVE_2026-08-23.md).

### 3.4 Other adjudicated negatives this window

- Options (temporally-extended actions) on 1-2: FAIL — control 8/100,
  treatment 0/100; mechanism failed by OVERCOMMITMENT (93.6% of
  4,000 real states chose k=4 — v22's predicted advantage-
  accumulation pathology). Receipts runs/options/rerun2_eval_*.json,
  runs/options/verdict.json. Consequence: v23's Castlevania options
  dependency sits on a failed mechanism pending a new registered
  salvage.
- Joint-policy pooling: the 0.52 flicker does NOT replicate — pooled
  honest 100-ep: 1-1 32/100 (vs specialist 43), 1-2 1/100 (vs banked
  38). Naive pooling stays falsified. Receipts
  runs/engine/logs/shelf_joint_*.log (on disk, uncommitted — §7).
- 1-4 endpoint: 51/100 pooled — exactly the banked rate; the 0.633
  probe was winner's curse. Consolidation HELD here (contrast with
  1-2's 31→8 collapse).

### 3.5 Closed vs open

CLOSED this window: 1-2 training (BANKED, now with measured ~0.53
ceiling); post-hoc gradient work on the consolidated 1-1 artifact
(all three families); the recurrent-RL overhaul (declined on data);
options as registered (no-rescue clause bars retuning); naive
pooling (re-falsified with receipts).

OPEN, registered, not yet run: v27 — is untrainability a property of
consolidation or of parameter budget? Fresh run with recovery states
in the curriculum FROM THE START (configs/mario_1_1_recovery_ppo.yaml
pattern). Ranked salvages not run: sequence-BC for the GRU line; SWA
over 1-2 peaks; hazard→Go-Explore weighting (Castlevania module
dependency). Wednesday Push lanes (787cf3e) in flight, no verdicts.

---

## 4. Process capabilities

Two weeks ago the process could verify configs. Today it verifies
that mechanisms are ALIVE, routes budget with measurements, and — the
evidence is this window itself — catches its own false results
before they reach the ledger. Every catch below is a capability.

### 4.1 Assay-as-router (new, binding)

Run the recovery assay BEFORE spending training effort on any
level's sticky rate: the recoverable share IS the budget-worthiness
signal. First application already paid: 1-2's assay (~0.53 ceiling,
fatal window dominant) converts "keep trying 1-2" from a judgment
call into a measured no.

### 4.2 Mechanism-aliveness preflight (born from the audit)

docs/research/PROCESS_AUDIT_2026-08-23.md named the root defect
class behind every void this month: "configs verified, mechanism-
aliveness never verified — an assay with no positive control."
Shipped fixes (commit 3bb10f6 + trainer changes): mandatory positive
controls in scripts/experiment_preflight.py; trainer sentinel
enforcement (the actor_freeze_steps 1e12 sentinel that silently
froze actors in Phase-3 and options v1 now trips a tripwire);
adjudicator fingerprint-identity refusal (refuses to score two arms
with identical weights). Falsifier design rules now binding: single
variable, preserve-on-peak both arms, ≥100 eps / 2 seeds honest vs
banked baseline, explicit actor_freeze_steps: 0.

### 4.3 The falsifier discipline — this window's catches

1. Frozen-actor VOID discovery (e4e111c): Phase-3 hazard-veto AND
   options v1 had trained only critics. Options FAIL correctly
   replaced by VOID; Phase-3 scope-corrected, with the surviving
   eval-time claim (veto collapse 31/100→0/100) explicitly kept.
2. Recovery assay scored 0/14 TWICE before the true 8/14 — a 3-min
   solver budget (exposed by a 10-min manual probe that solved,
   runs/recovery_assay/probe_ep15_10min/) and a stdout-grep success
   detector that could never fire. Fixed; final verdict is
   filesystem-scored against solutions/ ground truth.
3. Loader footgun (commit 28dc163):
   build_tile_policy_from_checkpoint silently returned a RANDOM net
   for path inputs — root cause of every standalone-loop anomaly on
   08-24 (the "0/60 unjittered argmax-tie receipt" is RETRACTED; the
   stick-probe was re-run with real weights and flipped the GRU
   edge's sign). Harness/solver receipts audited unaffected.
4. Winner's-curse correction: 1-4's 0.633 probe → endpoint re-run
   51/100, exactly banked.
5. Non-replication run: joint-policy 0.52 flicker → 32/100 and
   1/100 under the full protocol.
6. FAIL-vs-VOID defense: the GRU FAIL survived a hidden-reset audit
   (c61b711, 1e94445) before standing.
7. Registration honesty: the A/B prereg's claim that BC-through-
   stateless-fallback "cannot explain a between-arm difference" was
   retracted as wrong in the verdict doc.
8. Hot-machine false KILL on the hazard benchmark → retracted,
   needs_quiet gate added.
9. Strategic override with data: v26's own PASS gate was honored
   (probe PASSed) and its spending authorization declined anyway,
   on the measurement that detection was never the wall — twice
   confirmed after the loader fix.

### 4.4 Two-ledger + prereg adherence

Prereg/no-rescue followed on all four adjudicated experiments;
two-ledger discipline held; "read never infer" violated once and
caught by the fingerprint check; quiet-bench violated once and
retracted. The 2-1 attempt-1 VOID (950e37c, trained on 1-3's restart
ladder) is recorded in git but has no memory/docs coverage — carried
here so it is not silent.

---

## 5. Honest scoreboard [LEARNED unless marked]

Protocol: cold power-on, sticky-0.25 + start-jitter-16, single-life,
greedy, ≥50–100 eps, harness eval_game (per CLAIMS.md). Ceilings are
measured by the recovery assays, not estimated.

| Level / test | 2026-08-11 | 2026-08-24 | Ceiling (measured) |
|---|---|---|---|
| SMB 1-1 (specialist, banked) | in flight | **43/100** (banked mid-window) | — |
| SMB 1-1 (backward control) | — | **0.767** (untouched; reproduced on 60-ep collection; checkpoints/_preserved/backward_1_1_seed3_iter140.pt) | **~0.83–0.85**; post-hoc path CLOSED (isolated optimum) |
| SMB 1-2 | in flight | **38/100** banked (preserved ckpt), BANKED-closed | **~0.53** — fatal window dominant |
| SMB 1-3 | — | **21/100** | not assayed |
| SMB 1-4 | — | **51/100**, endpoint re-confirmed at exactly 51 | not assayed |
| GRU policy class (1-1) | — | **0.06** best-of-4 — FAIL | — |
| Options (1-2) | — | **0/100** treatment (control 8/100) — FAIL | — |
| Joint pooled policy | 0.52 flicker (unreplicated) | **32/100** (1-1) / **1/100** (1-2) — pooling falsified | — |
| Distill/recovery salvage (1-1) | — | 0.033 / 0.70 / 0.0 — all three families FAIL | — |

Two weeks ago the per-level banked scoreboard did not exist; the
learning track had just reopened with a pooled 0.65 flagship
reproduction. Today it holds four banked levels, one measured
ceiling per wall, and five adjudicated negatives with receipts.

EXHIBITION (unchanged headline, new frontiers): SMB all 32 levels
solver-complete (2026-07-27, db44fc7). New this window: NG frontier
area 9 / score 74,783, Rygar 5,680 px — both UNSOLVED (0 solutions);
Bubble Bobble 2 and Castlevania 3 smoke solutions in wave-1 (in
flight).

FORGE (no rates claimed, per ledger rules): odometer shipped +
certified 5/5; scene detection shipped; two generic death-semantics
fixes; preflight/sentinel/fingerprint enforcement; engine_driver
self-running shelf dispositions; assay + distill tooling.

---

## 6. Retraction / VOID register (window-complete)

1. Recovery assay first pass 0/14 ×2 → VOID (budget + detector
   defects); re-scored from filesystem. (§4.3.2)
2. Recovery-distill first attempt trained a random net → VOID;
   loader fix 28dc163. (§4.3.3)
3. "0/60 unjittered argmax-tie receipt" → RETRACTED (same root
   cause).
4. A/B prereg claim "BC fallback cannot explain a between-arm
   difference" → retracted as wrong. (§4.3.7)
5. v26 PASS-branch recurrent-RL authorization → overridden with
   data; re-confirmed post-fix (MLP 0.76 vs GRU 0.74). (§3.2)
6. Options v1 + Phase-3 training-side claims → VOID (frozen actor);
   Phase-3 eval-time finding survives, precisely scoped. (§4.3.1)
7. Hazard-model KILL on hot machine → retracted; needs_quiet added.
8. Joint-policy 0.52 flicker → does not replicate. (§3.4)
9. 1-4 0.633 probe → winner's curse; endpoint 51/100 = banked.
10. Pre-fix stick-probe AUC matrix (0.83/0.87, "+0.05 GRU") →
    superseded by real-policy re-run (0.76/0.74); the v25 memory and
    V26 doc figures predate the fix and should be read as such.
11. 2-1 attempt 1 (950e37c) → VOID, trained on 1-3's restart ladder
    (git-only record, surfaced here).

## 7. Receipt debt (uncommitted or orphaned receipts, named)

- runs/recovery_distill/variant_b_train.log + ckpts (variant B FAIL,
  380f306): on disk, uncommitted. Variant A (ba68f93): rung-2 history
  sits as an uncommitted working-tree overwrite of
  train_history.json (the committed copy, 8bded0d, is the base-run
  history — don't clobber it when banking); rung-1 only in task logs.
- runs/engine/logs/shelf_joint_*.log, shelf_1_4_endpoint_*.log
  (0848ca1's 32/100, 1/100, 51/100): on disk, uncommitted.
- runs/rygar_odo_debounce/ (1610093's falsifier): uncommitted.
- cd00fdf options smoke claims (iter 1121, 3,302 sps): no receipt
  file exists.
- runs/recovery_assay_bad/: void first-pass debris, uncited — needs
  a named retraction pointer or deletion; silent orphan is the one
  state it must not stay in.
- runs/odometer/ receipts are cited only in session memory; no
  docs/ file names them except this report and the cert/gate JSON
  commit (c3d7405).
- 14 of 15 ng_odo_*/rygar_odo_* dirs are uncited by path in docs;
  committed log tails runs/odometer/{ng_deep,rygar_deep,rygar_v2}.log
  (c1f9dbe) partially cover them, while the ng_night, ng_scene_long,
  ng_scene_long2, and rygar_night tails exist on disk uncommitted;
  ng_odo_doa is the DOA-retirement receipt and is named here.

---

One-paragraph summary. In two weeks the system gained a certified
game-agnostic progress instrument (odometer + scene identity + death
semantics) that converted three UNUSABLE games into two sound-signal
frontiers and one correctly-diagnosed skill wall; the sticky wall on
SMB went from hypothesis (POMDP) to measured decomposition (fatal
window + 33% trainable slice, ceilings 0.83–0.85 / 0.53), killing
the recurrent overhaul, the options mechanism, and all three
post-hoc salvage families with pre-registered receipts; and the
process grew the machinery (preflight positive controls, sentinel
tripwires, fingerprint refusal, assay-as-router, preserve-on-peak)
that caught two false verdicts, one silent random-net loader bug,
and one winner's curse before any reached the ledger. The honest
numbers stand at 1-1 0.767 / 1-2 38% / 1-3 21% / 1-4 51%, and the
one open registered question is v27: fresh-run curriculum vs
parameter budget.
