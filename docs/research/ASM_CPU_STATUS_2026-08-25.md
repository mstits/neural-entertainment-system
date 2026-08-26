# ASM CPU status — what is lockstep-verified, and why `asm_cpu` stays on by default

**Date:** 2026-08-25
**Decision:** keep `asm_cpu` compiled into the default `make build`; buy coverage
with more lockstep tests instead of with a global perf regression.
**Scope:** the AArch64 6502 ASM fast path (`asm_cpu` Cargo feature) and the
runtime `disable_asm_cpu` flag.

This document exists because the decision is a correctness-vs-performance
tradeoff, and project doctrine says those get receipted rather than drifted
into. It also exists so onboarding knows when `disable_asm_cpu` is a real
safety net and when it is superstition.

---

## 1. The decision

`Makefile`'s `build` target runs:

```
(cd nes_core && ../.venv/bin/maturin develop --release --features "python,asm_cpu")
```

That stays. The alternative on the table was "restore skip-by-default" per the
April 2026 memory *ASM CPU disabled — Mario walks again*. It was rejected for
five reasons, in descending weight.

**1. It would not be a restoration.** The `asm_cpu` feature has been in the
`build` target since this repo's first commit (`55e5333`). The April memory
describes a `pyproject.toml` in a pre-squash history that does not exist here,
and cites commit `6a4304d`, which is not in this repo. Removing the feature now
would be a **new** policy change wearing a revert's clothes. Framing a new
change as a revert is exactly the goalpost-move the claims ledger forbids.

**2. The fidelity premise behind skip-by-default is refuted on every ROM we
actually test.** The original bug (task #40: the MMIO callback ticked the PPU
only for the current instruction's cycles, and the NMI line only updated at
end-of-batch) produced "Mario falls through the floor," with a first divergence
at iter 782, PC=`$800D`. `nes_core/tests/asm_vs_slow_smb.rs` — the diagnostic
written for that exact bug, on that exact ROM — now runs 1.5M instructions with
zero divergence. So does MMC1, and so does every other family in §2. Whatever
landed for task #40 landed and stuck.

**3. The one live suspicion against `asm_cpu` was chased to ground and is not
an ASM bug.** Journey to Silius produced a flat odometer under the default
Pool, and toggling `disable_asm_cpu` "fixed" it — which looked damning. It is
an MMC1 state-restore defect (§4). Disabling `asm_cpu` by default would have
suppressed the symptom while leaving the real defect in place, which is the
worst available outcome: a masked bug **plus** a permanent perf loss.

**4. The cost is immediate and re-baselines everything.** Every perf number in
the project — the PGO ladders, the ~3500 samples/s core-bound ceiling, the
engine-wide ASM MMIO fixes, the Punch-Out/Gradius ASM work, the 32-level SMB
solve, the live show — was measured with `asm_cpu` on. Flipping the default
silently invalidates all of them, and re-earning them is expensive.

> **Correction (verification pass, same day).** An earlier draft of this
> section cited "206.7 ms with the ASM path vs 325.6 ms without (~1.58x)" on
> Journey to Silius. **That figure did not reproduce and is withdrawn.** A
> direct A/B (identical single-threaded `Nes::step` loop to a fixed
> 4,658,918-cycle mark, three runs per arm, both arms landing on the identical
> final `PC=$C8B6` and cycle count) gave overlapping distributions:
> ASM 176.7 / 222.8 / 243.0 ms, interpreter 197.2 / 214.9 / 235.8 ms. That is
> **no measurable single-ROM speedup**, not 1.58x. Two caveats keep this from
> being a finding in the other direction: the machine was under concurrent load
> from other agents (hence the ±30% spread), and a single-threaded attract-mode
> loop is not the workload the ASM path was tuned for (parallel pool stepping
> under PGO). The honest status is that **the per-ROM speedup is unmeasured**,
> and reason 4 rests on the re-baselining cost, not on a multiplier. A proper
> isolated benchmark on an idle machine is owed before anyone quotes a number.

**5. What (a) actually buys is coverage, not correctness — and coverage is
cheaper to buy directly.** A lockstep test costs nothing at runtime. Turning
off the fast path costs every run forever. When a mapper family is unverified,
the fix is to write its `asm_vs_slow_*.rs` test (§5), not to slow down the 29
mapper implementations that already work.

---

## 2. Lockstep-verified: what is actually covered

Each test lockstep-runs the bulk path (`nes.step`, ASM engine) against the slow
path (`nes.tick` loop) on a **real commercial ROM**, comparing CPU registers and
cycle count after every instruction plus full 2 KB system RAM at a stride.
All are gated `#![cfg(feature = "asm_cpu")]` and all currently pass clean.

| Mapper | Family | ROM | Test | Instrs |
|---|---|---|---|---|
| 0 | NROM | Super Mario Bros. | `asm_vs_slow_smb.rs` | 1.5M |
| 1 | MMC1 | Castlevania II - Simon's Quest | `asm_vs_slow_mmc1.rs` | 1.5M |
| 1 | MMC1 | Journey to Silius | `asm_vs_slow_silius.rs` | 1.5M |
| 2 | UNROM | Castlevania | `asm_vs_slow_unrom.rs` | 1.5M |
| 3 | CNROM | Gradius | `asm_vs_slow_gradius.rs` | 1.5M |
| 4 | MMC3 | Kirby's Adventure | `asm_vs_slow_mmc3.rs` | 1.5M |
| 9 | MMC2 | Punch-Out!! + Mike Tyson's | `asm_vs_slow_punchout.rs` | 1.0M |

Plus `asm_fuzz_indexed.rs` (opcode-level indexed-addressing fuzz) and
`asm_bulk_override.rs` (bulk-budget behavior on Zelda and Contra).

`asm_vs_slow_silius.rs` is **new as of 2026-08-25**. It was added because
Journey to Silius is the only heavy-DPCM ROM in the family (Sunsoft driver), so
it is the only test exercising DMC-DMA stall accounting against an MMC1 ASM
window. It is not vacuous: the ROM reports `prg_asm_ptr.is_some() == true` with
`asm_bulk_cycles = 1`, and running with the flag on vs off lands on an identical
final `PC=$C8B6` and an identical cycle count of 4,658,918.

## 3. NOT verified — the honest denominator

Do not read §2 as "the ASM CPU is verified." Three gaps, stated plainly.

- **Mapper breadth.** 6 mapper numbers are lockstep-covered. `mapper.rs`
  dispatches **37**, and **29 mapper implementations expose `prg_asm_ptr`** —
  meaning nearly the whole 793-ROM library routes onto the ASM path, mostly
  without a lockstep test behind it.
- **Depth.** Every test runs cold boot through the title/attract sequence:
  1.0-1.5M instructions, roughly 100-160 frames. Nothing covers deep gameplay,
  long runs, or level transitions.
- **Restored state.** No lockstep test calls `apply_state` at all. Both machines
  run from `reset()` only, with no input injected. The harness is structurally
  blind to any bug living in the restore path — which is precisely where §4's
  defect lives.
- **Oracle.** These are self-consistency checks (ASM vs our own interpreter),
  not external ground truth. External comparison is the separate nes-py parity
  harness and the Mesen oracle.

## 4. The MMC1 restore defect (open, not an ASM bug)

Found while investigating the Journey to Silius flat odometer.

`Mapper1::apply_state` (`nes_core/src/mapper/mapper1.rs:540`) parks
`last_register_write_cycle` at `u64::MAX`, commented *"reset so the next write
is unconditionally honored."* The RMW consecutive-write filter
(`mapper1.rs:392`) tests:

```rust
if self.cur_cpu_cycle == self.last_register_write_cycle.wrapping_add(1) {
```

`u64::MAX.wrapping_add(1) == 0`. A `Pool` that was constructed and immediately
restored has never ticked, so `cur_cpu_cycle` is still `0`. The sentinel that
promises "always honored" therefore delivers the exact opposite: **the first
MMC1 bank-select write after a state load is silently dropped.** One dropped
write desynchronizes MMC1's 5-write shift sequence, the commit lands on the
wrong register with the wrong value, the game gets a wrong PRG/CHR bank and
stops functioning — flat odometer, dead agent.

**Receipts.** (a) End-to-end, from two banked gate receipts on the same ROM,
same start state, same held input, same 1200 steps, both on the default ASM
path — the ONLY difference being the `reset_all()` ordering repair.
`gate_journey_to_silius_right.json` (unrepaired): 2 distinct values, range
0..4, `oam_churn` **1**/1199 — a dead machine.
`gate_journey_to_silius_right_resetall.json` (repaired): 137 distinct, range
0..783, `oam_churn` **842**/1199 — alive. (b) A source-level falsifier now
lives in the tree:
`mapper1.rs::restore_consecutive_write_tests::first_register_write_after_apply_state_is_honored_at_cycle_zero`.
It is `#[ignore]`d **because it fails on `main` on purpose** — `shift` stays at
`0x10` where it should advance to `0x18`. Run it with
`cargo test -- --ignored`. Its sibling
`consecutive_cycle_writes_are_still_filtered` is not ignored and guards the
original Bill & Ted's `INC $FFFF` behavior against any fix.

*(c) UNBANKED, cited for completeness only.* An earlier pass reported a
one-line A/B across 8 ROMs — guarding the predicate with
`self.last_register_write_cycle != u64::MAX &&` flips the symptom off, and
reverting flips it back on, with only Journey to Silius (first divergence at
step 0, 109 bytes) and Zelda (step 0, 291 bytes) affected while CV2, SMB,
Mega Man, Kirby, Metroid and Final Fantasy were clean either way. **No artifact
for that sweep was banked and it was not reproduced in the verification pass.**
Treat it as a lead, not a receipt; (a) and (b) are what actually carry the root
cause, and neither depends on it. Re-running it belongs in the parity sweep the
fix is waiting on anyway.

**Where `asm_cpu` comes in.** The ASM MMIO callback path never pushes
`Mapper::set_cpu_cycle` — only `Nes::tick` does (`nes.rs:854`). That is why
`cur_cpu_cycle` stays pinned at `0` under the ASM path, and why toggling
`disable_asm_cpu` appears to fix a mapper bug. Coincidence, not fidelity.

**Status: open, deliberately unapplied.** The one-line guard is an emulator-core
fidelity change. The last change to this same filter required a parity and
library re-baseline, so this one waits for sign-off plus a parity sweep rather
than riding in on a subagent's receipt.

**The caller-side fix is ordering, and for this profile it fully replaces the
flag.** Calling `pool.reset_all()` before the first `load_worker_state` runs a
frame (`Worker::reset` ends in `advance_one_frame()`, ~29,781 cycles), which
gets `cur_cpu_cycle` far off zero. The Journey to Silius gate was re-run on
current `HEAD` with the repaired ordering and no `disable_asm_cpu` patch, on the
default ASM path: `runs/onboard_wave4/gate_journey_to_silius_right_resetall.json`
reports range 0..783, 137 distinct, `oam_churn` 842 over 1200 forward-held
steps — the same trace as the older receipt that needed the flag. The ASM path
and the interpreter agree on this ROM; the flag was papering over the restore
defect, nothing more.

**Exposed callers — one repaired, eight open.** `go_explore_solve.py` and
`discover_observables.py` already `reset_all()` first and were never exposed.
`scripts/progress_signal_gate.py` did not; it has been repaired (2026-08-25).
A repo-wide sweep for the same never-ticked ordering (construct a `Pool`, then
`load_worker_state` with nothing in between to tick it) finds eight more, none
of them touched here — each needs its own check before a blind `reset_all()` is
inserted, since some restore inside a loop where an extra frame changes the
measurement:

`scripts/odometer_cert.py` (`fresh_pool`, :42→:48 — this is the odometer
certification harness, so any MMC1 title it has certified is suspect),
`scripts/recovery_assay.py` (:107→:117), `scripts/stall_discriminator.py`
(:124→:136), `scripts/room_fp_calibrate.py` (:130→:135 and :182→:187),
`scripts/merge_recovery_ladder.py` (:82→:100), `scripts/find_wrap_pair.py`
(:121→:124), `scripts/stick_probe.py` (:76→:86), and
`scripts/hazard_collect.py` (:835 → `_resolve_states_and_rng`).

> **Onboarding rule:** never construct a `Pool` and immediately
> `load_worker_state` into it. Always `reset_all()` first. On an MMC1 title the
> unsafe ordering corrupts **every** episode, and Zelda is a live target.

## 5. When `disable_asm_cpu` is a real safety net

**Currently: no known case.** The single situation it was ever used for in
anger — Journey to Silius — has a correct fix that is not this flag.

- **Unnecessary caution:** reaching for it on any family in §2 because of a
  general "graphics wrong" / "won't learn" symptom. Follow the project protocol
  instead: check the start state first (the title-screen-demo bug is the most
  common cause of "won't learn"), then lockstep against nes-py, then the Mesen
  oracle.
- **Legitimate use:** as a **diagnostic toggle** when bisecting a new suspicion
  on an unverified mapper (§3). If toggling it changes behavior, that is a lead,
  not a diagnosis — it also toggles `set_cpu_cycle` delivery, so a mapper bug
  can masquerade as an ASM bug, which is exactly what happened here.
- **Not a production setting.** If a ROM genuinely needs it, that ROM needs a
  lockstep test and a root cause, not a flag in its config.

**Entry criteria for trusting `asm_cpu` on a new ROM or mapper family:**

1. Add an `asm_vs_slow_<family>.rs` lockstep test on the real ROM.
2. Run a vacuity check — confirm `prg_asm_ptr.is_some()`, and confirm the flag
   on vs off produce an identical final PC and identical cycle count. A test
   that passes because the ASM path was never reached proves nothing.
3. Only then remove any `disable_asm_cpu` note from that game's config.

## 6. Corrections to prior records

- The memory *ASM CPU disabled — Mario walks again* (2026-04-26) says default
  builds skip the ASM CPU. **Stale.** Already annotated in the memory itself;
  §1 is the durable version.
- `configs/journey_to_silius.yaml` attributed its flat odometer to "the known
  ASM-CPU PPU/NMI-batching bug already on record for SMB." **Wrong on both the
  mechanism and the SMB attribution; retracted.** The config note has been
  rewritten to §4's root cause, keeping `disable_asm_cpu` documented as a
  workaround for the MMC1 restore defect rather than as an ASM fidelity fix.
- A prior analysis pass flagged `scripts/discover_observables.py:230` as likely
  exposed to §4. **It is not** — it calls `reset_all()` at line 197, immediately
  after constructing its Pool.
- An earlier draft of this document claimed `progress_signal_gate.py` was **the
  only** exposed caller. **Wrong; retracted.** A repo-wide sweep (every `.py`
  that both constructs a `Pool` and calls `load_worker_state`) finds eight more
  callers with the same never-ticked ordering — see §4's exposed-caller list.
  Only `progress_signal_gate.py` is repaired as of this pass.

## 7. Suite status under `--features asm_cpu` (measured 2026-08-25)

`cargo test --release --features asm_cpu --no-fail-fast` over 19 test targets:
**18 green, 1 red.** Every `asm_vs_slow_*` lockstep target passes, including the
new Silius one. Two caveats a future reader will otherwise trip over.

**`opcode_cycle_audit` fails 7/7 — pre-existing, and not a fidelity finding.**
It builds a synthetic NROM per opcode, runs `Nes::step()` once, and compares the
returned cycle count to the LaiNES base table. Under the ASM path that return
value is not the interpreter's per-instruction accounting, so every expectation
reads `got 0 cycles`. The same target passes **7/7 with the feature off**, and
the failure reproduces identically on unmodified `HEAD`. It is an ungated
interpreter-semantics test, not evidence of wrong cycle counts — the
`asm_vs_slow_*` family compares cumulative cycle counts on real ROMs and agrees.
The fix is to gate it `#[cfg(not(feature = "asm_cpu"))]` or teach it to read
cycles the way the ASM path reports them. Left alone here: out of scope, and it
should be a deliberate choice rather than a drive-by.

This is also why it went unnoticed. The project's routine gate is
`make test`, which runs `cargo check --lib` plus pytest — it never runs
`cargo test`, let alone with this feature.

**`cpu_asm::tests::diff_lda_abs_x_pagecross` is an intermittent flake.** It
fired in 2 of ~6 full-suite runs and never in isolated runs. The *same compiled
binary* both passes and fails across repeated invocations, so it is
nondeterministic rather than content-dependent; running that binary directly, at
default threads and at `--test-threads=1`, passed every time. The failure
signature is `PC rust=0xC003 asm=0xC000` — the ASM engine did not advance PC at
all, the same "did nothing" shape as the `opcode_cycle_audit` zeros. Worth
chasing (a JIT/W^X mapping that intermittently fails to take would explain
both), but it is pre-existing and independent of anything in this pass.

**Python side:** `pytest tests/ -k "config or profile" -q` → 695 passed,
21 skipped, 3526 deselected.
