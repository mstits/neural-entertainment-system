# World-Class Hardening Program — Phase-1 Audit Backlog

*Salvaged 2026-08-12 from the P1 audit workflow (`wf_004279b6-15b`,
69/~55+ agents completed before the run was stopped at low budget).
The Fable synthesis phase never ran; this backlog was harvested directly
from the cached agent findings. **20 critical + 83 high** findings.
Resume the workflow by runId to regenerate the full synthesis, or work
this backlog directly.*

## ACTIONABLE BUGS (real defects, not just test gaps — do these FIRST)

1. **`system_bus.rs:219-222` — `write_word()` writes big-endian.** High
   byte first instead of little-endian low-byte-first. Any caller gets
   swapped bytes. Fix: `write_byte(addr, val&0xFF); write_byte(addr+1, val>>8)`.
   Add a round-trip test. **Verify who calls it before assuming impact.**
2. **`oam_dma.rs` + `nes::State` — OAM DMA state never serialized.**
   The in-file doc claims byte-exact round-trips, but `get_state`/
   `apply_state` are never called from `nes::State`. A savestate taken
   mid-DMA is not byte-exact.
   **DEFERRED — requires a savestate format-version bump; bincode
   serde(default) does not give backward compat, so adding the field
   breaks all banked blobs. Needs a versioned State envelope, done as its
   own wave with a migration/round-trip test over old blobs.**
   *(A first attempt appended `oam_dma` to `nes::State`; because bincode
   is non-self-describing, every banked blob then failed to load with
   `unexpected end of file`. Reverted 2026-08-12 to restore backward
   compat.)*
3. **`python.rs:314` — `reset()` bypasses `apply_state_guarded`.**
   Restores the cached start-state via raw `apply_state`, so a corrupted
   snapshot crashes the interpreter instead of falling back to `reset()`.
   Route through the guard.
4. **`nes.rs:375/428/461` — raw `ram_ptr` aliases `&mut self.ram`.**
   `ram_ptr = self.ram.as_mut_ptr()` is passed to `try_step_asm`
   alongside a `SystemBus` built from a fresh `&mut self.ram` — a
   later `&mut` retained across the raw pointer's use. Give the pointer
   single provenance derived from the bus. (Memory-safety / UB risk.)
5. **`system_bus.rs:145-154` — SipHash HashMap lookup on every bus
   read** even though `cheats` is empty for the whole training pipeline.
   Guard with `if !self.cheats.is_empty()`. Cheap, hot-path perf win.
6. **`system_bus.rs:173-180` — `peek_byte()` mis-routes `$4016-$4017`**
   to the APU (returns 0) instead of controller state. Add explicit
   match arms.
7. **`cartridge.rs:308-320` — `apply_state()` clones+reallocs chr/prg_ram**
   instead of `copy_from_slice` into existing buffers. Allocation churn
   on every state load.

## WAVE LEDGER

- **Wave 1 (a94a51d, 2026-08-12)** — 6 actionable audit bugs fixed
  (write_word byte order, empty-cheats hot-path guard, peek_byte input
  routing, cartridge in-place restore, ram_ptr provenance, guarded
  reset). OAM-DMA (#2) reverted → DEFERRED. Test-drift it exposed fixed
  in 61a2846.
- **Wave 2 (3be9f4d, 2026-08-12)** — mapper test coverage: 27 untested
  mappers now have 294 biting tests (lib suite 214 → 508 green). Surfaced
  **7 suspected implementation defects** (below) — filed, not fixed, so
  the suite stayed green.
- **Wave 3 (1eba6e1 / 23cf9d5, 2026-08-12)** — fixed 3 of the 7 (VRC6
  expansion audio forwarding [HIGH, demo-audible], AxROM OOB mask, mapper34
  guard); diagnosed the other 4 as unsafe-to-fix-blind (2 DR dossiers
  submitted). Gated cargo 511 / make test 2538 / pool-test 546.
- **Wave 4 (4abd0df / 5ab0b1b, 2026-08-12)** — PPU/APU/CPU coverage: +65
  tests (cpu.rs 1 → 30), lib 511 → 576. Surfaced + FIXED a latent APU DMC
  bug (pointer wrap used `+= 1`, panics under overflow-checks; now
  wrapping_add — release byte-identical).

## MAPPER DEFECTS SURFACED BY WAVE 2 — DISPOSITION (Wave 3, 2026-08-12)

Wave 3 split them: fix the clearly-safe ones; DIAGNOSE (don't fix) the
behavioral ones against the compat suite first. Result — half were fixed,
half proved unsafe to fix blind. Falsifier source: `runs/rom_compat_audit.json`.

FIXED (1eba6e1), each with a test that fails on the old code:
1. **vrc6.rs (HIGH) — VRC6 expansion audio was silent.** forward_vrc6!
   now forwards tick_audio()/audio_mix() for Mapper24/26; also
   uses_scanline_irq()->true. (bundled the LOW #7 here.)
2. **mapper7.rs (MED) — AxROM PRG bank index** now masked modulo
   32K-bank-count (identity for full 256KB carts). No more sub-256KB OOB.
3. **mapper34.rs (LOW) — sub-32KB PRG read** now guarded `off & (len-1)`.

NOT FIXED — diagnosis says unsafe blind (verify-with-ROM / DR):
4. **mapper234.rs (Maxi-15) — CONFIRMED_BUG_RISKY.** Write-only latch;
   real hw latches on READ of $FFxx (menu launches a game via a fetch,
   never STA). 1 cart in library (boots, but the harness never drives a
   menu selection, so the failure path is unproven). Fix is textbook
   (override prg_read_byte to latch prg_peek_byte(addr) in the reg
   windows) but must be validated against the ROM. → VERIFY-WITH-ROM.
5. **mapper19.rs (Namco163) — CONFIRMED_BUG.** Register map shifted one
   0x800 slot (CHR reg 7 unreachable, NT regs one slot low, $D800
   dropped). BUT zero library ROMs use mapper 19 (boot-scaffold only),
   AND nametable/CHR-RAM select logic is also absent — a register-map-only
   fix won't render a real N163 title. → DR dossier written +
   VERIFY-WITH-ROM. Dossier: research-consult/prompts/
   n163_register_map_verification_2026-08-12.md (USER TO SUBMIT).
6. **mapper64.rs (RAMBO-1) — flagged item NOT_A_BUG** (the R8==0 fallback
   is a deliberate, commented boot guard). BUT diagnosis surfaced a
   DEEPER latent bug: the impl uses bank_registers[8] for the third
   swappable PRG bank, whereas real RAMBO-1 uses R15 — and bank_registers
   is [u8;10] so a real R15 PRG write is silently dropped. 5 shipping
   games (Klax, Road Runner, Rolling Thunder, Shinobi, Skull&Crossbones)
   boot clean (vectors/main in fixed+R6/R7 banks), so this is a latent
   in-gameplay bug with real regression risk. → DR dossier written.
   Dossier: research-consult/prompts/
   rambo1_mapper64_prg_register_verification_2026-08-12.md (USER TO SUBMIT).

Real-impact priority for follow-up: #64 root cause (5 shipping games) >
#19 (clean bug, nothing uses it) > #234 (single multicart, menu-switch).

## TEST-COVERAGE BACKLOG (largest cluster — the fidelity risk surface)

Grouped by area; each is "add unit tests for X" unless noted. Ranked by
blast radius.

- **Mappers: DONE (Wave 2, 3be9f4d)** — was 27 of 33 untested; all now
  covered with 294 tests. VRC6/MMC5 audio subsystems included.
- **PPU**: sprite-evaluation state machine (`:958`), VRAM read buffer
  (`:869`), address-increment row 30/31 wrap (`:1702`), palette mirror
  (`:2952`), scroll write-toggle (`:816`), coarse-X wrap (`:1693`),
  nametable mirroring dispatch (`:2997`).
- **APU**: DMC address wrap at 0x10000 (`:1130`), DMC loop/IRQ (`:1135`),
  five-step frame counter (`:421`), noise mode-1 6-tap LFSR (`:1005`),
  triangle linear-counter reload (`:896`), hw_dmc_stall_timing path.
- **CPU**: `apply_state` mid-instruction recovery (`:377`),
  `poll_interrupts` under all nmi-timing flag combos (`:576`),
  page-cross fetch (`:1624`), ADC overflow / SBC borrow edge cases.
- **Cartridge**: `mirror_address` per mirroring mode (`:28`), trainer
  skip (`:251`), NES2.0 CHR size + MAX_ROM_BYTES cap, OneScreen/
  FourScreen variants.
- **PyO3 (`python.rs`)**: NEON video unpack on ARM64 (`:1014`),
  hw_frame_anchor boundary logic (`:876`), pace_to_realtime hang risk
  (`:940`), replay_actions legacy format (`:974`), bulk-step margin
  overshoot (`:913`).
- **Pool**: load_worker_state frame_cycle_target reset, vblank-race
  prerequisite atomicity, load_start_state error paths.

## MAINTAINABILITY

- **`trainer.py:4807-9634` — `_run_vanilla_ppo` is one 4,828-line method**
  (rollout + PLR + CGSA + backward curriculum + RND + PR-MDP +
  Go-Explore + checkpointing + metrics), 16 nested closures. The
  highest-value refactor: extract phases into free functions/dataclasses.
  Do this behind byte-identity tests (the project has determinism
  lineages that must not shift).

## HOW TO RESUME

- Full synthesis (state-of-stack per pillar, demo matrix, gate-opener
  A/B, phased plan): `Workflow({scriptPath: '…/worldclass-p1-audit-wf_004279b6-15b.js', resumeFromRunId: 'wf_004279b6-15b'})`
  — the 69 completed agents replay from cache; only verify+synthesis run.
- Or work this backlog directly in waves (worktree-isolated, gated by
  tests). Actionable bugs 1-7 first; then mappers; then per-subsystem
  test suites.
- DR-worthy candidates flagged so far: none new from this audit (all
  findings are in-house fixable); the CV-hall gate-opener remains the
  standing DR candidate (see `GATE_OPENER_CAMPAIGN_2026-08-11.md` §13-15).
