# Parity + Perf Plan: close the remaining nes-py gap, then optimize

## Overview

Three commits landed this session (ca08374 → 9388bed → 18553a9) that:
1. Fixed a real skip_render PPU overshoot bug (user-visible — no more HUD sprite bouncing).
2. Built the `run_lockstep` harness + ratcheted baseline tests that catch RAM divergence automatically.
3. Removed the `NESEnvironment::reset` warmup frame, aligning step phase with nes-py. Mega Man went byte-exact; Zelda went from 40 bytes divergence to 1.

Current state of the baseline (600 cold-boot idle frames, RAM bytes diverged vs nes-py):

| ROM | Current | Target |
|-----|---------|--------|
| Mega Man | **0** | — (already byte-exact) |
| Zelda | **1** | 0 |
| Contra | 71 | ≤5 |
| Metroid | 170 | ≤5 |
| SMB | 177 | ≤5 |

Two user-facing acceptance tests:
- `tests/parity/test_zelda_input_replay.py` — feeds a 3354-frame scripted Zelda playthrough into both emulators; sword must be picked up at end. Currently `xfail(strict=True)`: xpasses when the last Zelda byte is fixed.
- `tests/parity/test_lockstep_baseline.py` — fail-on-improvement ratchet on all 5 games. Every fix below will make one of these fail with "improved! was X now Y" and prompt a ceiling tightening.

Once parity is in hand, the baselines and the skip_render / scanline-0 tests guard any performance work from regressing correctness.

## Architecture Decisions

- **Byte-exact parity with nes-py, not cycle-exact with real hardware.** nes-py is the reference; it has its own timing quirks relative to real NES. Matching nes-py maximizes compatibility with the training-data ecosystem and lets us reuse nes-py's bug-for-bug behavior for RL purposes.
- **Diagnose before fixing.** Every task starts by running `run_lockstep` to find the first diverging byte and the instruction class that wrote it. Past session burned hours on hypotheses (jemalloc, vblank race, RESET pulse) that had zero measurable effect; this plan requires a measurement before any change.
- **Ratchet via fail-on-improvement.** `test_lockstep_baseline.py::CASES` screams when divergence drops. That's how you commit progress: tighten the ceiling, add the fix, commit together. No fix is "landed" until the ceiling moves.
- **Performance work is gated by parity.** Don't chase performance while correctness is shifting; the baseline tests must be green before any bench-driven refactor lands.

## Task List

### Phase 1: Close Zelda to byte-exact (unblocks the user-facing xfail test)

#### Task 1: Diagnose the $01FD divergence on Zelda 600f idle — Size: XS
**Description:** Zelda at 600 idle frames differs from nes-py by exactly 1 byte: `$01FD` with `ours=0xA4 theirs=0x25`. This stack slot holds the P-register value pushed during an interrupt. Find which interrupt, which CPU cycle, and what sets the flags to different values.

**Acceptance criteria:**
- [ ] Identify whether the differing push is NMI, IRQ, BRK, or PHP.
- [ ] Identify the exact flag bits that differ (bit-AND the two values; narrow to N/V/Z/C/I/D/U/B).
- [ ] Hypothesis for root cause written down, even if not yet proven.

**Verification:**
- [ ] Added a short debug probe to `Cpu::push_byte` logging scanline/cycle/opcode when writing to `$01FD`; captured the final value written in both Zelda idle 600f and (as a control) Mega Man idle 600f. Remove the probe after.

**Dependencies:** None.

**Files likely touched:** (diagnostic only — no source changes) `nes_core/src/cpu.rs` temporarily.

---

#### Task 2: Fix the Zelda $01FD flag divergence — Size: S
**Description:** Apply the minimum CPU change to make the flag bits that differ come out matching nes-py. Likely suspects: per-opcode flag-affecting behavior on an ADC/SBC/CMP/BIT instruction, or an interrupt-push that captures the wrong P value.

**Acceptance criteria:**
- [ ] `test_lockstep_baseline.py::test_lockstep_ram_divergence_within_ceiling[Zelda]` fails with "improved! was 1 now 0" — tighten ceiling to 0, commit.
- [ ] `test_zelda_input_replay.py` either xpasses (in which case swap `xfail(strict=True)` for a normal assertion) or still fails but the first critical divergence in the replay moves later than frame 1011.

**Verification:**
- [ ] Run the full lockstep baseline — Zelda shows 0, no regressions on other games.
- [ ] Run `make parity` — 52 passing.
- [ ] Run `pytest tests/ -q` — no regressions.
- [ ] Run `cargo test --release --lib --test skip_render_parity` — 22/22 + 3/3.

**Dependencies:** Task 1.

**Files likely touched:** `nes_core/src/cpu.rs`, `tests/parity/test_lockstep_baseline.py`, possibly `tests/parity/test_zelda_input_replay.py`.

---

### Checkpoint A: After Tasks 1-2

- [ ] Zelda 600f idle is byte-exact with nes-py.
- [ ] If xfail still applies, the replay test is CLOSER (first-divergent-frame moved later) and a specific reason for the remaining gap is documented.
- [ ] Commit message references the diagnosis from Task 1 so future parity work has a trail.

---

### Phase 2: Close Contra (71 bytes) and the MMC1 games

Contra uses mapper 2 (UxROM), not MMC1 — the MMC1 suspicion from the earlier plan doc was wrong for Contra but still applies to Zelda. Before attacking Contra, first verify Task 2's fix didn't also drop Contra; likely a shared-root fix since both games' divergences originate in CPU-level behavior.

#### Task 3: Measure Task 2's spillover effect on Contra, Metroid, SMB — Size: XS
**Description:** Re-run the baseline test after Task 2 lands. Measure whether the Zelda fix incidentally dropped the other games' counts. Common CPU-core fixes often affect multiple games.

**Acceptance criteria:**
- [ ] Document the new counts for SMB / Contra / Metroid in `test_lockstep_baseline.py` history comment.
- [ ] Any game whose count dropped has its ceiling tightened with a commit.

**Verification:** Baseline test is green on the new ceilings.

**Dependencies:** Task 2.

**Files likely touched:** `tests/parity/test_lockstep_baseline.py`.

---

#### Task 4: Diagnose the first Contra divergence byte — Size: S
**Description:** Use `run_lockstep` with Contra to find the first address where nes_core and nes-py differ in the 600-frame idle run. Name the address (zero-page variable or stack), identify which CPU write created it, and document the hypothesis.

**Acceptance criteria:**
- [ ] First diverging address identified, e.g. `$0007`.
- [ ] Write-path traced: which opcode + PC + cycle wrote it in nes_core.
- [ ] Hypothesis written down.

**Verification:** Same diagnostic probe technique as Task 1.

**Dependencies:** Task 3.

**Files likely touched:** diagnostic-only.

---

#### Task 5: Fix Contra root cause — Size: S-M
**Description:** Apply the fix from Task 4's diagnosis. Likely candidates: APU frame counter timing, $2002 race, or a per-opcode cycle-count bug.

**Acceptance criteria:**
- [ ] Contra baseline ceiling drops to ≤5.
- [ ] No regressions in the other 4 games' ceilings.
- [ ] Pixel parity tape `contra_title` still passes (re-author only if phase shift is unavoidable and document why).

**Verification:** Full test suite green.

**Dependencies:** Task 4.

**Files likely touched:** `nes_core/src/{cpu.rs,ppu.rs,apu.rs,mapper/*.rs}` depending on diagnosis.

---

### Checkpoint B: After Tasks 3-5

- [ ] Two games byte-exact (Zelda, Mega Man), Contra near byte-exact.
- [ ] Remaining bytes on SMB / Metroid documented with first-divergence addresses.
- [ ] Visual spot-check in GUI: load Zelda, verify sword pickup works.

---

### Phase 3: SMB + Metroid — likely tougher (APU / mid-frame effects)

SMB and Metroid have the highest residuals (177, 170). Both use mapper 1 (MMC1). Likely the MMC1 consecutive-write detection, or APU frame counter, is the shared root cause.

#### Task 6: Diagnose the first SMB divergence, cluster it with Metroid — Size: S
**Description:** First-divergence byte on SMB. Then same for Metroid. Check whether they share a mapper-level or APU-level root cause.

**Acceptance criteria:**
- [ ] First SMB and first Metroid diverging addresses named.
- [ ] If both point at the same subsystem (e.g. both in audio-related zero-page), note the shared cause.

**Dependencies:** Task 5.

**Files likely touched:** diagnostic-only.

---

#### Task 7: Fix the shared SMB/Metroid root cause — Size: M
**Description:** Apply the fix. If there are two independent root causes, split into 7a and 7b.

**Acceptance criteria:**
- [ ] SMB ceiling ≤5, Metroid ceiling ≤5.
- [ ] No regressions.

**Dependencies:** Task 6.

**Files likely touched:** depends on diagnosis — probably `nes_core/src/{apu.rs,mapper/mmc1.rs}`.

---

### Checkpoint C: After Tasks 6-7

- [ ] All 5 baseline games at ≤5 bytes.
- [ ] `test_zelda_input_replay.py` xpasses (xfail gets removed).
- [ ] Add 2-3 new replay tests covering SMB + Metroid gameplay.

---

### Phase 4: Performance optimization (parity-locked)

This phase is ONLY unblocked once all Phase 3 acceptance criteria pass. The baseline tests from Phase 1-3 guard any perf change from regressing correctness.

#### Task 8: Re-run PGO build from scratch — Size: XS
**Description:** `make build-pgo`. The PGO profile is stale after the CPU/PPU changes from Phase 1-3. Per `feedback_pgo_rerun_after_code_change.md` memory, always regen before trusting bench numbers.

**Acceptance criteria:**
- [ ] Fresh `nes_core.profdata` committed.
- [ ] `make bench-hot` output compared against `docs/proposals/hot_path_baseline.md`.

**Verification:** Parity test suite still green after PGO build.

---

#### Task 9: Baseline perf numbers — Size: S
**Description:** Measure single-env fs=1 throughput vs nes-py, and 12-parallel fs=16 trainer throughput. Update `docs/proposals/unified_rust_plan_v3.md` with the new numbers.

**Acceptance criteria:**
- [ ] Numbers captured for: single-env fs=1, single-env fs=4, single-env fs=16, 12-parallel fs=4, 12-parallel fs=16.
- [ ] Each number includes a nes-py comparison (ratio).

---

#### Task 10: Pick the highest-leverage perf optimization from the ranked list in `unified_rust_plan_v3.md` batches G/H/I/J — Size: M-L
**Description:** Per the v3 plan, remaining batches are G (fs=1 gap), H (audio verify), I (24h fuzz), J (retire nes-py). Pick one, apply, measure, commit. Repeat per session.

**Acceptance criteria:** depends on which batch.

---

### Checkpoint D: Complete

- [ ] Every ROM in the baseline test at ≤5 bytes RAM divergence.
- [ ] At least one replay-based gameplay test per baseline ROM, all passing (not xfail).
- [ ] Trainer throughput measured and documented vs nes-py.
- [ ] One parity-locked perf optimization shipped.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Fix for one game regresses another | Medium | Baseline ratchet catches regressions immediately; commit after full suite green only. |
| Residual divergence is inherent (RAM init non-determinism, open-bus reads) | Low | Accept ≤5 bytes as the floor; document. |
| Parity fixes invalidate existing golden tapes | Low | Re-author if needed (5-minute operation); pattern established in 18553a9. |
| Performance optimization regresses pixel correctness | Medium | All perf work must pass skip_render_parity + parity harness + lockstep baselines; no exceptions. |
| Zelda replay xfail never xpasses even with 0-byte idle parity | Medium | Indicates the divergence happens only under active input; write a `run_lockstep` variant that feeds the replay tape and finds the first divergence under input. |

## Open Questions

- Should we add a reference nestest ROM + log to the repo? Would make per-opcode cycle-count bugs trivially bisectable. User acceptance of the GPL/freely-redistributable ROM needs confirmation.
- Should the baseline test fail-on-improvement pytest.fail be softened to a warning once the count hits 0? Currently strict; fine for now but annoying if a ROM genuinely settles at 0.
