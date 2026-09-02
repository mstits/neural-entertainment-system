# Proposals index

This directory is the project's design/decision record: architecture
proposals, pre-registrations, campaign specs, strategy documents, and
point-in-time status/resume notes. Unlike `docs/research/` (externally-
reviewed Deep Research consultations) and `docs/receipts/` (per-run
evidence), this directory mixes live plans, closed experiments, and
plans that a later document superseded in place. Read here for **current
status**, not history. A doc's own text is often stale by the time you
read it, because several are amended or superseded by a later doc without
being deleted.

Status legend:

- **ACTIVE**: current plan of record, still governing work today.
- **SUPERSEDED**: a later, named document replaces it.
- **COMPLETED**: the work it proposed shipped or ran to a verdict
  (positive or negative); the doc is now a record, not a plan.
- **HISTORICAL**: a design doc for something now built and stable,
  kept for context; no action follows from reading it.
- Anything I could not confidently place is labeled **status unclear,
  needs a closer read**, not guessed.

Grouped by work-stream and roughly chronological within each group.

---

## 1. Rust migration, CPU/PPU performance, engine architecture

| Doc | Date | Status |
|---|---|---|
| `unified_rust_plan_v3.md` | 2026-04-21 | **COMPLETED.** Its own §9 cross-reference table already marks several siblings historical (see below); its Batches F–J (scaling fix, fs=1 parity, audio sign-off, 24h ASM soak, nes-py retirement) are all landed. `nes-py` is confirmed absent from `requirements.txt` and fails to import in this environment, satisfying Batch J's DoD. `docs/ARCHITECTURE.md` still cites it as "the authoritative roadmap," so it also serves as a historical reference, not only a closed plan. |
| `aarch64_cpu_asm.md` | 2026-04-20 (design) | **HISTORICAL.** The AArch64 threaded-code 6502 core it specifies was built; `unified_rust_plan_v3.md` §9 called it "still live as the reference." It remains the default CPU path today (commit `cee9f2b`, "asm-cpu: ... asm_cpu stays default", 2026-08-25) and is still being extended to new mappers (Gradius, Punch-Out, commits `a11d8cb`/`f963c0a`). Superseded as a *plan* only in the sense that the thing it planned now exists. |
| `asm_fuzz_result.md` | 2026-04-21 | **SUPERSEDED** by the 2026-04-24 overnight soak recorded in `nes_core/SECURITY.md` (100M streams / 1.6B instructions / 0 divergences, run specifically to gate the MMC1 RMW + NES-2.0 fixes). `unified_rust_plan_v3.md` §9 predicted exactly this rewrite ("rewritten in Batch I with 24-h numbers"). |
| `cpu_bulk_stepping.md` | 2026-04-20 (design) | **HISTORICAL.** `unified_rust_plan_v3.md` §9 labels it in-repo: "historical experiment; archive." |
| `pgo_results.md` | 2026-04-20 | **HISTORICAL.** PGO is now a standing part of the build pipeline (`scripts/pgo_build.sh`); `unified_rust_plan_v3.md` §9 labels this doc "historical; archive after Batch J" (Batch J is done). Still linked from `docs/ARCHITECTURE.md` as the measurement writeup. |
| `ppu_event_driven_catchup.md` | 2026-07-14 | **SUPERSEDED** by `event_driven_ppu_design_2026-07-31.md`, which says so explicitly in its own header ("Supersedes for the fidelity track... That design partially landed and is default-OFF"). |
| `event_driven_ppu_and_subcycle_bus_2026-07-31.md` | 2026-07-31 | **COMPLETED** (the WHY note). The design it argues for (advancing the PPU to an absolute target dot instead of per-cycle) is implemented: `nes_core/src/ppu.rs` has `advance()`/`advance_to()` and `nes.rs:96` documents that hot loops "route through `Ppu::advance_to(target_dot)`." |
| `event_driven_ppu_design_2026-07-31.md` | 2026-07-31 | **COMPLETED** (the HOW blueprint, companion to the above). Same evidence: `advance_to`/`target_dot` machinery is live in `nes_core/src/ppu.rs` and `nes.rs`. |
| `archive_engine_v2_2026-08-01.md` | 2026-08-01 | **ACTIVE, unimplemented.** Proposes replacing the pickle-backed `GoExploreArchive` with an mmap arena. As of this read, `src/training/go_explore.py` is still `pickle.dump`/`pickle.load`-based. No `mmap`/`madvise` anywhere in `src/`. No later doc supersedes or withdraws it; it appears to be a live but deprioritized proposal, not a shipped one. |
| `crate_hygiene_eviction.md` | 2026-07-20 | **ACTIVE (backlog, not fully actioned).** Report-only dead-code audit. Spot-checked item A1 (the dead `simd` Cargo feature) is still present unremoved in `nes_core/Cargo.toml` today, so this list has not been comprehensively worked. Status of the other ~9 items is unverified here. Treat the list as an open backlog. |
| `rust_actor_learner_design.md` | 2026-07-20 | **ACTIVE, unimplemented.** Explicitly "design + acceptance test only. Not implemented." as written. `tests/test_actor_learner_parity.py` still carries `skipif` guards keyed on "not-yet-built actor entry point" (confirmed still true today). No successor or abandonment found. |
| `trainer_decomposition_plan.md` | 2026-07-20 | **ACTIVE, partially executed.** Proposed five extracted modules (`RolloutCollector`, `PPOUpdater`, `ExplorationController`, `Curriculum`, `CheckpointManager`). Four of five exist today as separate files (`ppo_updater.py`, `exploration_controller.py`, `curriculum.py`, `checkpoint_manager.py`); `trainer.py` is still ~10,000 lines and no dedicated `RolloutCollector` module was found. Read as in-progress, not shipped or superseded. |

## 2. Parity / fidelity harness

| Doc | Date | Status |
|---|---|---|
| `nes_parity_harness.md` | 2026-04-22 | **HISTORICAL.** The spec for what is now the standing `make parity` harness, built and stable, kept for the design rationale (why hybrid `cross_emulator`/`golden_hash` modes exist). |
| `parity_design_pattern.md` | n/d (same era, added 2026-04-27) | **HISTORICAL/still-descriptive.** Describes the five-layer test stratification (`test_library_buckets.py` … `test_zelda_input_replay.py`) that the harness still uses; no successor doc found. |
| `parity_coverage_map.md` | n/d (same era) | **HISTORICAL: numbers stale.** A frozen `scripts/parity_sweep.py --frames 120` snapshot (byte-exact/tight/moderate/loose/wide buckets across 439 comparable ROMs). The project has since reached 793/794 ROMs *booting* (a different metric: boot success, not RAM-parity bucket), and numerous mapper/IRQ fixes have landed since this sweep (e.g. MMC3 A12 edge fix, MMC1 restore-write-drop fix), so the specific bucket counts here are almost certainly outdated. No newer coverage-map doc exists in this directory to supersede it with. |
| `parity_harness_proof_log.md` | 2026-04-22 | **HISTORICAL.** Dated regression-proof entries (e.g. T13 palette-perturbation proof) for the harness above; a log, not a live plan. |
| `mmc3_crash_n_boys_investigation.md` | ~2026-04-23/24 (open at time of writing) | **COMPLETED.** Marked `Status: OPEN` in its own text with the fix "not yet identified," but the real root cause was found and fixed one investigation later: `nes_core/KNOWN_ISSUES.md` "CLOSED" section records commit `dbfad69` (2026-04-27): missing `I`-flag set on IRQ entry in `interrupt_push_status`, not the MMC3-specific hypothesis this doc was chasing. `KNOWN_ISSUES.md` also confirms "No open issues. All 794 tested ROMs... boot." |

## 3. SMB campaign history (2026-07): decision points, plans, thesis docs

| Doc | Date | Status |
|---|---|---|
| `unified_learning_thesis.md` | authored 2026-05-16 | **SUPERSEDED in substance** (not by name) by `RESEARCH_GROUNDED_PLAN_2026-07-21.md`, which explicitly retires "the ad-hoc 'make model-free PPO generalize' loop that has repeatedly failed": the approach this thesis proposed (one `vanilla_ppo`-derived framework, configured per game, beats all six games directly). |
| `breakthrough_roadmap_2026-07.md` | 2026-07-16 | **HISTORICAL.** Its top bet (BET 1, "the Level Factory": systematized Go-Explore + backward robustification per level) is exactly the mechanism that later shipped and beat the full 32-level game (see `docs/research/README.md` Thread 2, and `smb_oneshot_campaign.md` below). Its BET 2 (honest sticky/jitter eval everywhere) also shipped and is now the standard protocol referenced throughout later docs. No doc formally supersedes it; it reads as executed-in-spirit. |
| `smb_oneshot_campaign.md` | added 2026-07-15, updated through 2026-07-19 | **COMPLETED.** The doc is itself the live record: it accumulated "Status" sections in place through "WORLD 1 ONE-SHOT: ACHIEVED" (2026-07-16), the World 2 campaign, and finally commit `dd1fc60` ("GATE MET 7h early — chain clears 1-1 through 2-3 cold on both seeds... verdict GO", 2026-07-19), which appended the go/no-go resolution directly into this file. |
| `morning_plan_2026-07-17.md` | 2026-07-16/17 | **HISTORICAL.** A single morning's hour-by-hour execution plan (10-lane synthesis); overtaken by the day's actual events, which are recorded in `smb_oneshot_campaign.md` and `decision_memo_2026-07-17.md`. |
| `decision_memo_2026-07-17.md` | 2026-07-17 | **COMPLETED.** Set a Sunday 2026-07-19 18:00 go/no-go on the SMB campaign continuing. Resolved GO, seven hours early, per commit `dd1fc60` (folded into `smb_oneshot_campaign.md`, see above). The memo's own criteria were met. |
| `PROJECT_STATUS_2026-07-20.md` | 2026-07-20 | **SUPERSEDED**, and says so in its own addendum: "this snapshot is now stale as a status report. For current plan, priorities, and gates, read `docs/proposals/STRATEGY_2026-08-08.md`... it supersedes the roadmap/priority sections below." (That successor is itself now superseded: see §5.) The addendum is explicit that the scorecard and drift-and-correction history underneath remain accurate as history, not retracted. |
| `RESEARCH_GROUNDED_PLAN_2026-07-21.md` | authored 2026-07-21, added 2026-07-24 | **SUPERSEDED in substance.** Argued Go-Explore-solves → BC/DAgger-distills as the fix for the 1-2 hard-exploration wall. The Go-Explore-solves half shipped (full game beaten by search). The imitation/distillation half was later tested exhaustively and eliminated for 1-2 specifically by `HONEST_1_2_FINDINGS_2026-08-14.md` and closed by `SMB_1_2_AUTORESOLVE_2026-08-14.md` (§4 below). |

## 4. The SMB 1-2 honest-protocol wall (closed)

| Doc | Date | Status |
|---|---|---|
| `HONEST_1_2_FINDINGS_2026-08-14.md` | 2026-08-14 | **SUPERSEDED** by `SMB_1_2_AUTORESOLVE_2026-08-14.md`, which names it directly: "this closes the sequence begun in `HONEST_1_2_FINDINGS_2026-08-14.md`." |
| `SMB_1_2_AUTORESOLVE_2026-08-14.md` | 2026-08-14 | **COMPLETED.** Self-declared terminal record: "Status: CLOSED — DECISION: BANK." Every method in both paradigms (imitation and on-policy sticky RL) scored 0.0 honest on 1-2; the harness's own positive control (1-1 at 0.76) rules out a plumbing artifact. This is the closing document for the 1-2 learned-policy line as of this writing. Later work (RECOVERY_DISTILL, V27/V28) targets 1-1, not 1-2. |

## 5. Strategy and the "totality basis"

| Doc | Date | Status |
|---|---|---|
| `TOTALITY_BASIS_2026-08-08.md` | 2026-08-08 | **ACTIVE framework, stale status column.** Still the operative "ten mechanism classes / eight-game basis" framing (`README.md` at repo root still points here for "what 'any game' would actually take"). But several per-class statuses in its table are now out of date: class 6 (Punch-Out, "open"): `FIGHTGATE_MECHANISM_2026-08-25.md` shipped and scored SUCCESS on exactly this; class 8 (Metroid, "open — big research lift"): `ROOMGRAPH_ENGINE_2026-08-24.md` shipped room-graph navigation and RG-2 (Metroid) has since run; class 9 (Zelda, "open"): item-semantics IS-1a has since run (FAIL, see §7). Read the class *framework* as current and the class *statuses* as a 2026-08-08 snapshot. |
| `NES_LIBRARY_CLASSIFICATION_2026-08-17.md` | 2026-08-17 | **ACTIVE.** Companion to the above, applying the basis to the full 795-ROM library (headline: the basis is an action-game basis, ~75% coverage by title count). No successor found. |
| `STRATEGY_2026-08-08.md` | 2026-08-08 | **SUPERSEDED** by `STRATEGY_2026-08-14.md`, explicitly: "Supersedes the day-30 framing of STRATEGY_2026-08-08 where they conflict." Note that `README.md` at the repo root has **not** been updated to reflect this. It still calls this doc "the current plan of record" in three places. Treat the root README's strategy pointers as stale until someone repoints them at 2026-08-14. |
| `STRATEGY_2026-08-14.md` | 2026-08-14 | **ACTIVE, with caveats.** No later document names it as superseded, and the three-flywheel framing (Truth / Learning / Scale) is the most recent standing strategy doc. Two things have visibly moved past its text since: the "no-overnight compute rule stands" line has been superseded by a later compute-policy change (unattended/overnight compute is now permitted, per this project's own tracked history), and the `WEDNESDAY_PUSH_2026-08-24` → `_DAY2` → `_DAY3` documents (§9) run a much larger multi-lane operating mode than this doc's "operating rhythm" section describes, without formally citing it as superseded. |
| `WORLDCLASS_PROGRAM_2026-08-11.md` | salvaged 2026-08-12 (source audit 2026-08-11) | **ACTIVE (open backlog).** A 20-critical/83-high finding list salvaged from a stopped audit workflow before its synthesis phase ran. Later docs describe drawing it down in named "hardening waves" (`WEDNESDAY_PUSH_DAY2_2026-08-25.md`: "a hardening wave (16 confirmed bugs fixed...)"; `WEDNESDAY_PUSH_DAY3_2026-08-25.md`: "Lane K — hardening wave 3," naming specific still-untouched modules). Partially actioned, not closed. |

## 6. The GATE-OPENER campaign (closed negative)

| Doc | Date | Status |
|---|---|---|
| `gate_opener_arm_2026-08-11.md` | 2026-08-11 | **SUPERSEDED** by `GATE_OPENER_CAMPAIGN_2026-08-11.md`, which says so directly: "`gate_opener_arm_2026-08-11.md` (828 lines) is a starting point only; where it conflicts with this doc, this doc rules." Its vocabulary (`GATED`/`saturated`) is also independently struck campaign-wide after the K-FALSIFIER ran and failed. |
| `GATE_OPENER_CAMPAIGN_2026-08-11.md` | 2026-08-11 (rev 4) | **COMPLETED: closed negative.** This is a living document amended in place across four revisions plus in-line strike-throughs, not a static plan. Its own §7 K-FALSIFIER ran and FAILED (no statistic separates the Castlevania "hall" from four already-SOLVED archives, so the `GATED` wall class was retired mid-document to the purely descriptive `UNRESOLVED-CONCENTRATED`). Its own §13 records "K0 VERDICT (2026-08-11): FAIL — CAMPAIGN DISARMED," and §15 records a third tuning attempt ("K0-v3, Kirby $004F: TUNING MISS") with the verdict "campaign stays DISARMED; K0 loop HALTED by decision." Read this document for its *history*, not as a plan to execute. The arm it specifies never got past instrument calibration. |

## 7. Room-graph / memory-architecture / item-semantics (the Metroid-Zelda capability lane)

| Doc | Date | Status |
|---|---|---|
| `MEMORY_ARCHITECTURE_2026-08-23.md` | 2026-08-23 | **HISTORICAL: realized elsewhere.** Proposed three layers (metric position / topology / semantics). Layer 1 shipped same-day as `ODOMETER_CORE_SPEC_2026-08-23.md` (below); Layer 2 shipped as `ROOMGRAPH_ENGINE_2026-08-24.md`; Layer 3 shipped (with a negative result on its first test) as `ITEM_SEMANTICS_ENGINE_2026-08-25.md`. The vision in this doc is not wrong or replaced, just fully built out elsewhere now. Read the three successor docs for current status. |
| `ODOMETER_CORE_SPEC_2026-08-23.md` | 2026-08-23 | **COMPLETED.** The v24 research-round spec for the PPU scroll odometer. Shipped same day, commit `3429d8a` ("odometer: PPU scroll integration in-core — v3 savestate envelope, certified 5/5"); `nes_core/src/ppu.rs` carries the `odometer_x`/`odometer_y`/`odometer_enabled` fields this doc specifies. |
| `ROOMGRAPH_ENGINE_2026-08-24.md` | 2026-08-24 | **COMPLETED.** Synthesis of three competing designs (D0 minimal-diff won as chassis, with grafts from D1/D2). Shipped: `docs/receipts/room_graph/RG1_zelda_2026-08-25.md` records a live Zelda run adjudicated against a locked pre-registration, PASS on 4 of 5 RG-1a sub-checks (room-count floor, fade-edge presence, zero warp-minted edges, false-merge audit all PASS; stability at 86.6%, above the kill line, below the 95% target, with a root cause identified). |
| `ITEM_SEMANTICS_MINIMAL_2026-08-25.md` | 2026-08-25 | **SUPERSEDED** by `ITEM_SEMANTICS_ENGINE_2026-08-25.md`, which synthesizes it as "Lens A" and selects it as the winning chassis. |
| `ITEM_SEMANTICS_INDEPENDENT_2026-08-25.md` | 2026-08-25 | **SUPERSEDED** by `ITEM_SEMANTICS_ENGINE_2026-08-25.md`, which synthesizes it as "Lens B" and grafts its behavioral-verification/classification ideas onto the winning MINIMAL chassis. |
| `ITEM_SEMANTICS_ENGINE_2026-08-25.md` | 2026-08-25 | **COMPLETED (shipped; first gate FAILed).** The synthesis of the two docs above. Implemented as `scripts/discover_item_bits.py`. Its pre-registered validation gate IS-1a (Zelda negative control) ran and returned **FAIL** (`runs/item_semantics/is1a/IS1A_VERDICT_2026-08-25.md`): the byte-identity check passed, but candidates were promoted to `confirmed` over already-banked Zelda data when the gate required zero. The mechanism is real and wired; the first live test of it did not pass. |

## 8. Fight-gate progress mechanism

| Doc | Date | Status |
|---|---|---|
| `FIGHTGATE_MECHANISM_2026-08-25.md` | 2026-08-25 | **COMPLETED (shipped on its primary target; extension mixed).** Marked "DESIGN ONLY" in its own header at time of writing, but the mechanism was subsequently built and wired live: `tests/test_fight_gate.py`, `scripts/go_explore_solve.py`, and `runs/fight_gate/` all exist, and commit history shows a SUCCESS on Punch-Out (commit `d4933ac`, "fight-gate SUCCESS") followed by an extension attempt to Kung Fu / Ice Climber / Galaga that did **not** validate (commits `5238a88`, `fc6ed95`: "1-for-4, Kung Fu/Ice Climber/Galaga don't validate"). Read as: mechanism shipped and proven on its motivating case, generalization beyond that case is an open negative result. |

## 9. The sticky-wall training-science chain (1-1, honest protocol)

This is the chain that actually runs `V21 → OPTIONS_PREREG → RECURRENT_BOTTLENECK_AB → RECOVERY_DISTILL_1_1 → V27_FRESH_RECOVERY → V27_SUCCESSOR_PASS / V28_CAPACITY`. Note: **`V23_SYNTHESIS_2026-08-23.md` is not part of this chain** despite the adjacent version number (see §7/§10); it is a same-day, broader-scope research round (the five-game SIMCE plan) that references V21's hazard-model ranking but does not feed into V27. Treat "V21 → V23 → V27" as two different threads sharing an ancestor, not one line.

| Doc | Date | Status |
|---|---|---|
| `V21_SYNTHESIS_2026-08-22.md` | 2026-08-22 | **COMPLETED/ACTIONED.** Diagnosed why the Phase-3 hazard veto failed (behavior-policy mismatch + gradient corruption) and re-ranked "options/temporal abstraction" to the #1 build. Its own "Supersessions" section closes `RESEARCH_SYNTHESIS_2026-08-17` Phase 3 as "closed, negative." Its build-order recommendation produced `OPTIONS_PREREG_2026-08-22.md` the same day. |
| `OPTIONS_PREREG_2026-08-22.md` | 2026-08-22 | **COMPLETED: FAIL.** Pre-registered commitment-options mechanism for the sticky wall. First run VOID (both arms had `actor_freeze_steps: 1e12` and trained only critics: `docs/research/PROCESS_AUDIT_2026-08-23.md`). Re-run and adjudicated live: **control 8/100, treatment 0/100 (FAIL)** by overcommitment (`docs/research/OPTIONS_NEGATIVE_2026-08-23.md`, `runs/options/verdict.json`). |
| `RECURRENT_BOTTLENECK_AB_2026-08-23.md` | 2026-08-23 | **COMPLETED: FAIL.** Tested DR round v25's prescription (recurrent policy beats feedforward under sticky-noise POMDP). Verdict: treatment best-of-4 honest sticky **0.06** vs control **0.76**, FAIL, and per the verdict doc the mechanism itself went untested (stick-detection was never the actual bottleneck). See `docs/research/RECURRENT_AB_VERDICT_2026-08-23.md`. |
| `RECOVERY_DISTILL_1_1_2026-08-24.md` | 2026-08-24 | **COMPLETED: ALL THREE FAMILIES FAIL.** Verdict appended in place and in `CLAIMS.md` ("RECOVERY DISTILLATION — THREE FAMILIES, ALL FAIL"): base method, KL-anchored cloning, and on-policy recovery PPO all failed to lift the banked 1-1 control past 0.767. The meta-finding, "the consolidated 48k artifact is an isolated optimum," is the premise `V27_FRESH_RECOVERY` was built to split (post-hoc consolidation vs. from-scratch curriculum). |
| `V27_FRESH_RECOVERY_2026-08-24.md` | 2026-08-24 | **COMPLETED: FAIL.** Pre-registered before any training ran. Verdict in `CLAIMS.md`: best-of-4 pooled honest **0.530** against PASS ≥0.80 / FAIL ≤0.767 (commit `2ec0d58`): "not close on either bound." Per the verdict's own text, "the sticky-wall research line is CLOSED on curriculum shape"; capacity (`V28_CAPACITY`) is the standing next step, not a contingency. |
| `V27_SUCCESSOR_PASS_2026-08-25.md` | 2026-08-25 | **SUPERSEDED / MOOT.** Written ahead of the v27 verdict per its parent lane's "prepare regardless" instruction, to activate only if v27 PASSED. v27 FAILed (0.530, see above), so this document's design never activates. Confirmed against project tracking: the FAIL/MARGINAL branch (`V28_CAPACITY_2026-08-25.md`) is the one that actually fired. |
| `V28_CAPACITY_2026-08-25.md` | 2026-08-25 | **ACTIVE: in progress, no verdict yet.** The pre-registered FAIL/MARGINAL branch from `V27_FRESH_RECOVERY`, testing whether more capacity (`tile_hidden_dim` 64→96, 48,135→72,039 params) helps where curriculum shape didn't. As of the latest `CLAIMS.md` entry: preflight checks (V1–V4) all PASS, seeds 0–1 complete, seed 2 running, seed 3 not started. This is the current standing experiment in this lane. Do not read any of the docs above as still-open; this is the one that is. |

## 10. Research synthesis (five-round digest)

| Doc | Date | Status |
|---|---|---|
| `RESEARCH_SYNTHESIS_2026-08-17.md` | 2026-08-17 | **COMPLETED/ACTIONED.** Digests five Deep Research rounds (v16, v18, v19, v20, and the Phase-3 veto). Each recommendation has since been individually carried out by a dedicated doc: the hazard/veto question closed negative via `V21_SYNTHESIS_2026-08-22.md`; the "shared substrate, per-level heads" ranking (v20) shipped as commit `740cac6` ("substrate: trunk-plus-heads experiment," 2026-08-17, same day); the room-graph design (v16) shipped as `ROOMGRAPH_ENGINE_2026-08-24.md`. Read this doc as background for those, not as an open plan. |
| `V23_SYNTHESIS_2026-08-23.md` | 2026-08-23 | **status: mixed, partially unclear.** A same-day, broader-scope round (the five-game SIMCE plan: Contra, Zelda+Metroid, Castlevania, Final Fantasy) that shares an ancestor with the chain in §9 (it credits V21's hazard-model ranking) but is not part of it (see the note at the top of §9). Its unifying SIMCE mechanism (irreversible-bit tracking + counterfactual splicing) was later operationalized in a different, redesigned form by `ITEM_SEMANTICS_ENGINE_2026-08-25.md` (§7), not built exactly as specified here: a repo-wide search found no `irreversible`-bit Contra key, no `z_obj` cell-key implementation, and no Final-Fantasy probe artifacts. The Zelda/Metroid room-identity portion shipped via `ROOMGRAPH_ENGINE_2026-08-24.md`. **Whether the Contra, Castlevania hazard+options, and Final Fantasy sub-experiments specifically named here ever ran is not established by this read: treat those three as status unclear, needs a closer read**, distinct from the parts confirmed shipped. |

## 11. Point-in-time resume notes and the Wednesday push

These are operational handoff notes, not designs. Each is accurate only for the moment it was written, and each names its own resume point.

| Doc | Date | Status |
|---|---|---|
| `RESUME_2026-08-17.md` | 2026-08-17 | **HISTORICAL.** A post-reboot resume checklist (bank World 1-4, run three no-training analyses). Overtaken by `RESUME_2026-08-18.md` and all subsequent work; the specific numbers it flags as pending (1-4 60% probe) have long since been superseded by the `CLAIMS.md` ledger. |
| `RESUME_2026-08-18.md` | 2026-08-18 | **HISTORICAL.** A stopped-session handoff (engine unloaded, 2-1 consolidation paused). Superseded by `RESUME_2026-08-20.md`. |
| `RESUME_2026-08-20.md` | 2026-08-20 | **HISTORICAL.** Records the hazard Phase-1 KILL retraction (a benchmarking-on-a-hot-machine false negative) and a work queue. Superseded by the much larger `WEDNESDAY_PUSH_2026-08-24.md` operating mode four days later. |
| `WEDNESDAY_PUSH_2026-08-24.md` | 2026-08-24 | **SUPERSEDED** by `WEDNESDAY_PUSH_DAY2_2026-08-25.md`, its explicit continuation. |
| `WEDNESDAY_PUSH_DAY2_2026-08-25.md` | 2026-08-25 | **SUPERSEDED** by `WEDNESDAY_PUSH_DAY3_2026-08-25.md`, its explicit continuation. |
| `WEDNESDAY_PUSH_DAY3_2026-08-25.md` | 2026-08-25 | **ACTIVE.** The most recent document in this sequence as of this index; no Day 4 exists yet. Names the currently-running work (`V28_CAPACITY` training, an RG-1 re-verification pass) and the next lanes (RG-1 final verdict + RG-2 Metroid, hardening wave 3). Read this and `CLAIMS.md` together for the actual current state of the project. |

---

## Notes on accuracy for future readers

- Several documents in this directory are **living documents amended in
  place** (`GATE_OPENER_CAMPAIGN_2026-08-11.md` most extremely, four
  revisions plus inline strike-throughs, but also `gate_opener_arm`,
  `STRATEGY_2026-08-08.md` before it was superseded, and `TOTALITY_BASIS`).
  Their *filename date* is when they were created, not when they were
  last true. Read to the end, including any addenda/strike-throughs,
  before citing one.
- The repo root `README.md` still points to `STRATEGY_2026-08-08.md` as
  "the current plan of record" in three places. That doc has been
  superseded by `STRATEGY_2026-08-14.md` since 2026-08-14 (§5). This is
  a real staleness in the root README, not a status this index invented.
- `CLAIMS.md` is the ground truth for experiment verdicts cited above;
  where a proposal doc and `CLAIMS.md` disagree, trust `CLAIMS.md`. It
  is updated as results land, proposal docs sometimes are not.
