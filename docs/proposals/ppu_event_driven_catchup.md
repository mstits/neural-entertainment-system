# SYNTHESIS — Event-driven PPU catch-up: the campaign plan

Target: `nes_core` on M4 / AArch64, PGO pipeline (`scripts/pgo_build.sh`). Problem
(measured 2026-07-14): the per-dot PPU state machine costs ~545 µs/frame under
`skip_render` ≈ 54% of `pool_step`; `Ppu::tick` is entered 82–89k×/frame and **68%
of PPU-state time is per-cycle dispatch residual** — the cost of *entering* tick and
running its branch cascade, not leaf work. Goal: advance the PPU in computed slices
between observable boundaries. Ceiling: >9% of `pool_step`, likely more, and it may
unlock CPU bulk batching (inert today at pool scale).

---

## 1. Chosen architecture — scanline-granular `advance` (Design 2), grafted

**WINNER: ANGLE 2 (scanline-granular).** Both judges scored it highest on the three
axes that decide this campaign — fidelity-provability (9/9), staged-shippability (9/9),
and gain-realism (9/9) — while staying competitive on layout (8/8) and lowest on
implementation risk (8/8). It wins for one structural reason the other two concede:

- It **does not predict the sprite-0 hit dot at all.** `can_batch_line()` returns
  false whenever `sprite_0_on_scanline && !hit`, so that one scanline runs the
  *existing per-dot `tick` body* and latches the flag on its true dot — exact by
  construction, zero new math, robust even for games that never poll `$2002`. The
  hardest, most-sacrosanct boundary costs ~1 scanline/frame (measured: 0.6–2.0
  mid-scanline reg-writes/frame across SMB/SMB3/Zelda), not a proof burden.
- Its gain estimate is the only one grounded in a **measured** batchable-scanline
  distribution (≥99% of visible lines carry no register write), not asserted.
- It keeps `tick` byte-for-byte untouched as the fallback callee, so the layout-
  fragile hot path is never perturbed.

### Grafts (credited)

**From ANGLE 3 (compile-time dual-path) — the layout-safety machinery, adopted wholesale:**
- **L1 golden-disassembly gate → `make ppu_layout_check`.** Commit `llvm-objdump -d`
  of the hot symbols (`Ppu::tick`, `tick_three`, `render_pixel`, the `fetch_*`,
  `shift_background_registers`, `update_sprite_rendering_registers`) from a fresh PGO
  build; every rung diffs against golden and is **rejected if any hot symbol's machine
  code changed.** This converts the historical *silent* layout-perturbation failure
  (tick_n −55%, inline-tick −52%, skip_bg-under-PGO removed) into a hard, local,
  deterministic gate. This is the single most valuable idea in the whole panel and it
  is not native to the winner.
- **L4 dual-corpus PGO.** Once `advance` becomes the common skip-render path, PGO could
  demote `tick` to cold `.text.unlikely` and regress the **full-render / spectator**
  path that still uses it. `pgo_build.sh` stage-2 workload MUST include a full-render
  (skip_render OFF, spectator-style) segment *and* skip-render training frames, and the
  A/B gate measures full-render sps explicitly and rejects any spectator regression >1%.
- **Trust-SM Shadow compare — repurposed as a TEST-TIME differential oracle, not a
  production path.** Both judges dinged Angle 3 for shipping analytic sprite-0 on the
  mainline under invariants that never check the hit dot. We take the *good* half: a
  `PPU_FF_SHADOW=1` mode that runs `advance` and a reference per-dot advance on a cloned
  PPU and diffs `observable_digest` — used in fuzz/CI, never as the production
  correctness argument. Production correctness rests on per-dot fallback + the pre-ship
  tape/Mesen gates, as in Angle 2.

**From ANGLE 1 (event-queue) — the analytic pieces, deferred to their proper rungs:**
- **The fully-worked sprite-0 predictor math** (§4.4: fetch replication for 8×16 +
  H/V-flip, AND of both opacity masks column-by-column, span clamp to ≤254, `x<8`
  left-mask, `x<255` quirk, mid-line scroll handled by horizon-segmentation). This is
  the best sprite-0 derivation of the three. It is **optional Rung 4 (perf-only)** —
  only for games that *don't* poll `$2002` near the hit, never on the correctness path.
- **`cpu_cycles_until_irq_fire()`** — the closed-form MMC3 A12 IRQ-fire cap, mirroring
  the existing `cpu_cycles_until_nmi_fire()` (ppu.rs:486). Drives the Rung-3 IRQ horizon.
- **The overflow de-risk**: a cheap first pass counts in-range sprites; if ≤8, overflow
  is impossible → skip the buggy `n/m`-diagonal walk entirely (the common case).
- **Lazy-`v` framing** for the rare rendering-enabled `$2007`.

---

## 2. Why this is sound — the premise all three share (verified in source)

Every CPU access to `$2000–$2007` routes through `SystemBus::read_byte`/`write_byte`,
and in **every** CPU engine the PPU is synced to that exact cycle *before* the access
runs: the ASM MMIO callback `tick_impl` → `bus.tick_cpu_cycles_with` (nes.rs:246–253)
materializes mid-block; the bulk/interp path's catch-up loop
(`for _ in 0..cycles { apu.tick(); ppu.tick_three() }`, nes.rs:293–296, 352–355) only
ever contains non-MMIO cycles, so a register write is by construction the *first* cycle
of the next step. **Therefore inside any slice the catch-up loop is asked to advance,
PPUCTRL/PPUMASK/scroll-`t`/`x`/OAMADDR are constant** — a `$2000–$2007` write can only
land at a slice boundary, never mid-slice. The only things that change inside a slice
are PPU-internal and computable in closed form: `v`, the (unobserved) shift registers,
sprite-eval scratch, the sticky flags, vblank, and A12/IRQ.

**The bulk==1 reality that dictates the two-level structure** (verified nes.rs:196–213,
Judge 1 & 2): for **every banked mapper — MMC1, MMC3, UxROM, CNROM — `raw_bulk_cycles
== 1`**, so a per-step PPU catch-up is only 3 dots. Only NROM (SMB) and a few others
(AxROM=4) get a wider window. Batching a whole 341-dot scanline for the majority of the
library REQUIRES deferring the PPU across many CPU cycles — that is **Level B** below.
Level A (in-loop `advance`) is the safe NROM down-payment that provably cannot regress
bulk==1 mappers because `advance(3) == tick_three` exactly.

---

## 3. State-advance algorithm

Add **one** new hot routine, `Ppu::advance(&mut self, dots: u64, mapper, sink)`. The
existing `tick` body is retained **verbatim** as the fallback callee (revertability +
the parity oracle + layout preservation).

```
fn advance(dots):
  remaining = dots
  while remaining > 0:
    dot  = scanline_cycle()
    line = scanline
    end  = CYCLES_PER_SCANLINE - odd_skip(line)      // 341 or 340
    span = min(remaining, end - dot)                  // dots left on this line
    if can_batch_line(line, dot, span):               // dot==0 && span==full && no event inside
        step_whole_scanline(line)                     // closed form, O(1)
    else:
        for _ in 0..span { tick() }                   // EXISTING per-dot body, untouched
    remaining -= span
```

`can_batch_line(line, dot, span)` = `dot == 0 && span == end` **and** the line is not
"interesting":
- not the sprite-0 line: `!sprite_0_on_scanline || already_hit`,
- not an overflow-polled ROM (per-ROM `overflow_polled` bool, default forces per-dot),
- for A12 mappers: heuristic owns the clock (common: emit one `on_scanline_tick` at dot
  260/324) **or** A12 is fetch-driven (8×16/same-table) → per-dot (Rung 3 relaxes this).

A partially-owed scanline (`dot != 0` or `span < end`, i.e. a slice started/ended
mid-line because of MMIO) always runs per-dot — that is the measured ~1–2 lines/frame.

### `step_whole_scanline` — CPU-observable side effects ONLY (skip_render, rendering on)

Under skip_render the BG/sprite pixel pipeline is unobserved (`render_pixel` bails,
ppu.rs:1349). So the closed form does only:

- **Scroll `v`** (exact transcription of `inc_coarse_x_with_wrap`/`inc_y_with_wrap`,
  ppu.rs:1494–1520). Across a full visible line the 32 coarse-x incs in 8..256 are
  overwritten by the dot-257 `v←t` horizontal copy, so with disjoint bit-fields
  (`0x041F` horiz, `0x7BE0` vert):
  ```
  v_vert  = inc_y(v) & 0x7BE0                    // one inc_y_with_wrap
  v_horiz = (t & 0x041F); inc_cx(v_horiz)×2      // two prefetch coarse-x incs at 328/336
  v = v_vert | v_horiz
  ```
  Pre-render (261): `v_vert = t & 0x7BE0` (the 280–304 vertical copy) then the two
  prefetch incs. Rendering OFF: `v` unchanged. **Correctness-first alternative** (from
  Angle 3, recommended for the first cut): run a *bounded ≤33-iteration loop* of the
  real inc functions instead of the hand-derived formula — still 10× cheaper than 341
  dispatches, immune to wrap-edge (`y==29/31`, nametable-toggle) bugs a formula invites.
  Promote to the closed form only once the loop version is green and PGO shows it worth.
- **Sprite evaluation** — one-pass "first 8 in-range sprites" scan of primary OAM,
  producing `sprite_0_found` (→ next line's `sprite_0_on_scanline`) and, for the
  ≤8-in-range common case, `SPRITE_OVERFLOW`. Overflow's exact set-dot/value under the
  hardware `n/m` bug is gated behind `overflow_polled` (default per-dot).
- **A12/IRQ** — emit one `on_scanline_tick` at dot 260/324 (heuristic case).
- **Counters** — `cycles += end; scanline += 1;` plus the frame-boundary work
  (chr_cache refresh, `bg_elision_allowed`, and at wrap the `write_frame`/frame++/
  skip_render reset the tick already does).

No fetches, no shifts, no `render_pixel`, no 341-way dispatch — the 68% residual for
that line is gone. `#[inline(never)]` keeps it off `advance`'s driver and off `tick`'s
cache lines.

---

## 4. Sprite-0 exactness strategy (spelled out)

**Rungs 1–3: do not predict — exploit the poll (Angle 2's economy).** The CPU that
cares about the hit is spin-reading `$2002` to time its split (SMB HUD ≈ scanline 31,
Zelda HUD, MMC3 status bars). Each `$2002` read is a slice boundary, so on the sprite-0
scanline the PPU is already handed short slices and never owns a whole scanline there.
`can_batch_line` returns false for `sprite_0_on_scanline && !hit`, so **that one line
runs the reference per-dot body and latches on the true dot, byte-identical to today.**
Cost: ~1 line/frame. This is exact by construction and needs zero new math.

**Every way a naive predictor is wrong (why per-dot fallback dodges all of them):**
transparent BG columns, transparent sprite columns, left-8 masking, `x==255` suppression,
8×16 tile-half + H/V flip, mid-scanline scroll change, sticky re-latch. The per-dot body
already handles every one. A mid-scanline scroll write is MMIO → a slice boundary → the
pre-write and post-write dots are separate spans, so a split line is never whole-batched.

**Rung 4 (optional, perf-only): the analytic predictor, grafted from Angle 1 §4.4**, for
games that DON'T poll `$2002` near the hit. Hit at first `x` in `[x0, min(x0+7,254)]`
with `s0_opaque[x−x0] ∧ bg_opaque[x] ∧ show_bg ∧ show_spr ∧ ¬(x<8 ∧ ¬(bg_left8 ∧
spr_left8))`, dot `= x+1`:
1. `sprite_0_on_scanline` = `y0 ≤ line < y0+height`.
2. Sprite-0 opaque mask: fetch sprite-0's two pattern bytes for row `line−y0`
   (replicate `fetch_sprite_tile` 8×16 half-select + V-flip; H-flip = bit-reverse);
   opaque = `lo | hi`. 2 CHR reads.
3. BG opaque mask over the ≤3-tile span from `v`-at-scanline-start + `fine_x`; opaque =
   pattern bits nonzero. `xstart = 8` if either left-8 off, else 0.
4. First `x` with both opaque → `hit_dot = x+1`; else no hit.
This is on the correctness path **only if** its own gate (SMB/Zelda HUD pixel-diff +
Mesen sprite-0 lockstep + off-vs-on parity on the sprite-0 line) is green; otherwise it
stays a pure perf optimization behind its own sub-gate and per-dot remains the default.

---

## 5. Layout-safety protocol (the make-or-break axis — history: −55%, −52%, skip_bg removed)

Mandatory for **every** rung, in order:

1. **`tick` is never modified.** It stays the full-render path and the per-dot fallback.
   `advance`/`step_whole_scanline` are NEW functions in their own region, `#[inline(never)]`,
   nothing inlined into `tick`.
2. **`make ppu_layout_check` (grafted L1).** Fresh-PGO `llvm-objdump -d` of the hot
   symbols, diffed against committed golden. **Any change to a hot symbol's machine code
   = rung rejected.** Run before trusting any perf number — this catches the silent
   global-codegen perturbation that has no other detector.
3. **PGO-in-the-loop, fresh, per rung.** `cargo clean` between PGO and plain (stale
   profdata masquerades as regression — repo rule). Regenerate profdata on the *new*
   code on the real trainer workload before trusting any %.
4. **Dual-corpus PGO (grafted L4).** Stage-2 workload includes a full-render segment +
   skip-render frames so PGO keeps both `tick` and `advance` hot. A/B measures
   full-render sps explicitly; reject any spectator regression >1%.
5. **Per-mapper A/B + leaf self-time bisect.** Bench a `bulk==1` ROM (Zelda, Level-B
   territory) AND a `bulk>1` ROM (SMB, Level-A) AND Kirby (worst mapper-write case, no-
   regression guard). Sample leaf self-time (`perf_discovery` harness), not just wall
   time — a +2% wall hiding a moved cache line is the exact skip_bg failure signature.
6. **Revert, don't tune in place.** Any rung that doesn't beat the gate-off baseline
   post-PGO by a clear margin is reverted (flip the runtime bool), not massaged.
7. **Gate is a runtime bool** (`ppu_scanline_advance`, mirroring `refined_skip_render`,
   ppu.rs:457), default **OFF**; gate-OFF codegen must be byte-identical (A/B 0.0% delta).

---

## 6. Rung ladder

Universal gates (all rungs, fresh PGO): **146-tape parity + Mesen 33/33 lockstep + the
4 `skip_render_parity` tests + `refined_off_vs_on_state_parity` observable-digest**
(skip_render_parity.rs:255–280) extended with a `scanline_advance` on/off axis + the
`make ppu_layout_check` disasm gate + dual-corpus PGO A/B (no spectator regression >1%).

**Rung 0 — Instrumentation + layout gate (0% perf; pure de-risk).**
- Scope: extend the `ppu_neon_stats` counter to a per-mapper histogram — fraction of
  visible scanlines with (a) zero MMIO in dots 1–256, (b) a reg write, (c) a `$2002`
  read, (d) a mapper write — to establish the *true* batchable fraction per mapper on
  in-game start-states (not just title screens). Land `make ppu_layout_check` (golden
  disasm) and the dual-corpus `pgo_build.sh` stage-2 workload. Add the `PPU_FF_SHADOW`
  differential oracle harness (test-only).
- Gates: suite green; disasm unchanged; A/B 0.0% delta.  **Revert:** delete counter/module.
- **Est: 0%** (measurement + safety net before any analytic code exists).

**Rung 1 — `Ppu::advance` + Level-A in-loop (NROM / bulk>1 only).**
- Scope: closed-form (or ≤33-iter bounded) `step_whole_scanline` for skip_render,
  non-interesting, rendering-on/off lines; sprite-0/overflow/A12-fetch lines fall to
  per-dot. Wire `advance` into the three catch-up loops (Level A). Since a bulk==1 slice
  is 3 dots, `can_batch_line` is always false there and `advance(3) == tick_three` — so
  bulk==1 mappers provably cannot regress. Detailed spec in §8.
- Gates: universal + SMB (NROM sprite-0 canary) + an AxROM title + **brute-force
  `advance(n) == n×tick` equivalence test** over one frame from many `(entry, n)` +
  byte-identical full-render framebuffers.  **Revert:** flip bool / one-line swap to
  `tick_three`.
- **Est: ~4–8% of pool_step** on NROM-dominated workloads (SMB); ~0 on Zelda/MMC1
  (that's Level B). Reasoning: captures ~40–55% of the per-line dispatch residual on the
  batched lines, and NROM slices owe up to 192 dots (bulk=64), so whole scanlines are
  actually owed.

**Rung 2 — Level-B deferred PPU for MMC1 / UxROM (no A12 IRQ). THE HEADLINE.**
- Scope: add `ppu_owed_dots` at the sync layer; the per-cycle catch-up ticks **APU
  only** (audio/DMC/frame-IRQ stay per-cycle) and does `ppu_owed_dots += 3`. Flush
  (`ppu.advance(ppu_owed_dots)`, then zero) at: (1) `$2000–$2007` access (flush first,
  then service — subsumes today's `tick_cpu_cycles_with`); (2) the NMI horizon (already
  the CPU-batch cap). MMC1/UxROM have no `on_scanline_tick`/A12, so no IRQ horizon
  needed. Sprite-0 line still per-dot via the poll. This is what turns the PPU
  O(scanlines) for the bulk==1 majority.
- Gates: universal + Zelda MMC1 observable-digest parity + Zelda fs=1-vs-fs=16
  framebuffer parity + the `PPU_FF_SHADOW` fuzz oracle at *every* `$2002`/IRQ boundary
  across MMC1/UxROM/NROM with randomized input scripts.  **Revert:** feature-gate the
  deferral; flush-every-cycle == today.
- **Est: ~10–18% of pool_step** on MMC1/UxROM ROMs (large slice of the library).

**Rung 3 — Level-B for MMC3 via A12 / IRQ horizon.**
- Scope: add `cpu_cycles_until_irq_fire()` (grafted from Angle 1 — closed-form: with one
  A12 heuristic clock/scanline at dot 260, IRQ asserts at `scanline_now + irq_counter`;
  cap the CPU horizon there exactly like the NMI cap, flush, recompute after any MMC3
  `$8000–$FFFF` write since those are slice boundaries). Emit `on_scanline_tick` at
  260/324 in the batch; keep the heuristic authoritative (`HEURISTIC_OWNS_CYCLES`,
  mapper4.rs:482). Fetch-driven-A12 lines (8×16/same-table, Kirby) force per-dot.
- Gates: universal + SMB3 status-bar split pixel-diff (fs=1 vs fs=16) + an MMC3 8×16
  title + **Kirby** (no-regression + IRQ-exactness) + **Mesen IRQ-cycle lockstep on an
  MMC3 raster title (mandatory)**. Fallback: per-dot when the counter is within N of 0.
  **Revert:** sub-gate; MMC3 falls back to Rung-1/per-dot.
- **Est: ~10–18% on MMC3 raster titles**, ~0 elsewhere; Kirby ~neutral.

**Rung 4 — (optional, perf-only) analytic sprite-0 predictor + closed-form fetch-A12.**
- Scope: Angle 1 §4.4 predictor for games that don't poll `$2002` near the hit; removes
  the per-dot fallback on sprite-0 lines. Closed-form fetch-driven A12 reconstruction for
  §3.4 lines. Ship only if Rung-0 data shows residual per-dot lines worth it.
- Gates: universal + SMB/Zelda HUD pixel-diff + Mesen sprite-0 lockstep + off-vs-on
  parity on the sprite-0 line.  **Revert:** sub-gate → per-dot sprite-0 line.
- **Est: ~1–3%**. Ship last, or never if Rungs 1–3 land the target.

Library-weighted, Rungs 1–3 plausibly reach **~10–15% of pool_step** (brief's ">9%,
likely more"), before the CPU-bulk follow-on that Level B unlocks (raising MMC1's
`asm_bulk_cycles` clamp becomes a separate lever once the PPU is no longer the
3-dots-per-cycle anchor).

---

## 7. Top 5 risks (ranked) + detection

1. **icache/layout regression under PGO** (the historical killer — skip_bg, tick_n,
   inline-tick). *Detect:* `make ppu_layout_check` golden-disasm gate (rejects on any
   hot-symbol machine-code change) + fresh-PGO dual-corpus A/B on a bulk==1 and a bulk>1
   bench + leaf self-time bisect. *Mitigate:* `tick` untouched; `advance` isolated
   `#[inline(never)]`; revert-not-tune.
2. **Deferred-PPU staleness at an unmodelled observable** (Level B) — a read path that
   samples PPU-derived state without a flush (mapper `irq_pending` between horizons, a
   `$2004` OAM read during rendering, a DMC interaction). *Detect:* extend the
   `observable_digest` parity to fire at *every* `$2002`/IRQ-poll boundary in a fuzz
   test with randomized input across MMC1/MMC3/NROM (`PPU_FF_SHADOW`); Mesen lockstep is
   the ground-truth backstop. *Mitigate:* enumerate flush points; any digest divergence
   = a missing flush.
3. **Sprite-overflow flag exactness in batch-eval** — the hardware `n/m`-diagonal bug's
   precise set-dot/value in adversarial OAM. *Detect:* targeted test with pathological
   OAM (9+ sprites straddling a line, mid-eval `$2002` bit-5 read) comparing batch vs
   per-dot. *Mitigate:* `overflow_polled=false` default forces per-dot; ≤8-in-range fast
   path skips the walk entirely; ship batch-eval only behind a proven per-ROM allowlist.
4. **MMC3 IRQ-horizon mis-prediction** (Rung 3) — a mid-frame `$C000/$C001` reload or
   table-select shifts the assert scanline; status-bar split lands on the wrong line.
   *Detect:* those writes are `$8000–$FFFF` MMIO → already slice boundaries → recompute
   the horizon after any MMC3 write; SMB3/Kirby raster pixel-diff + Mesen IRQ-PC-of-
   service lockstep + a synthetic reload-mid-frame case. *Mitigate:* per-dot when counter
   within N of 0; own sub-gate; keep the dual A12 path.
5. **odd-frame dot-skip / frame-boundary drift in the closed form** — `advance` is now
   the authoritative cycle counter; an off-by-one in `odd_skip` or the wrap
   `write_frame`/skip_render reset desyncs `cpu_cycles_until_nmi_fire` → NMI at the wrong
   PC → the "Mario falls through floor" class. *Detect:* the parity harness's per-frame
   29781-cycle lock is a canary; the brute-force `advance(n)==n×tick` equivalence test
   (Rung 1 gate) asserts `(scanline, scanline_cycle, cycles, v, flags)` identical after
   `advance(n)` vs `n×tick`, exhaustive over one frame. *Mitigate:* prefer the bounded
   inc-loop over the hand-derived `v` formula until proven.

**Secondary:** OAM-DMA island (preserve the per-cycle path — DMA reads no PPU state, so
`advance` with tiny n or eager `tick` during `oam_dma.active`); the `$2002` read-on-set
race (match current tick ordering, not "real hardware" — the 146 tapes pin it); MMC2/MMC4
CHR-latch fetches (Punch-Out) — exclude from fetch-elision, per-dot fallback.

---

## 8. Rung 1 first-rung spec (hand-off ready)

**Objective:** land `Ppu::advance` + Level-A in-loop wiring, batching whole
non-interesting scanlines when a slice owns ≥1 full scanline. Behavior byte-identical to
today under gate-OFF and observably-identical under gate-ON; measurable win only on
NROM/bulk>1 ROMs (SMB), ~0 on bulk==1.

**Files touched**
- `nes_core/src/ppu.rs`: add `advance`, `step_whole_scanline`, `can_batch_line`,
  `odd_skip`; add the `ppu_scanline_advance: bool` gate field + `set_ppu_scanline_advance`
  setter (mirror `refined_skip_render`, ppu.rs:457). `tick`/`tick_three` UNCHANGED.
- `nes_core/src/nes.rs`: in the two catch-up loops (nes.rs:293–296, 352–355) and the
  ASM remainder loop, replace `for _ in 0..cycles { apu.tick(); ppu.tick_three() }` with
  `for _ in 0..cycles { apu.tick() }` followed by, when the gate is on,
  `self.ppu.advance(cycles as u64 * 3, &mut self.mapper, video_frame_sink)`; when off,
  the verbatim `ppu.tick_three()` loop (byte-identical fallback). Keep APU per-cycle.
- `nes_core/tests/skip_render_parity.rs`: add the `scanline_advance` on/off axis to
  `refined_off_vs_on_state_parity`; add the brute-force equivalence test (below).
- `Makefile`: add `ppu_layout_check` target (golden disasm of the hot symbols).
- `examples/ppu_state_profile.rs` (or `ppu_neon_stats`): add a `scanline_advance` A/B mode.

**`advance` signature & body:** exactly §3. `can_batch_line(line, dot, span)` =
`dot == 0 && span == (CYCLES_PER_SCANLINE - odd_skip(line))` **and**
`(!self.sprite_0_on_scanline || self.regs.ppu_status.contains(SPRITE_ZERO_HIT))`
**and** `!self.overflow_polled` (new per-ROM bool, default `false` ⇒ **for Rung 1,
force per-dot on ALL visible rendering-on lines is too coarse** — instead: Rung 1 only
batches vblank scanlines (240–260) and rendering-DISABLED visible lines via
`step_whole_scanline`, plus rendering-enabled visible lines that pass `can_batch_line`;
sprite-eval/overflow lines with >0 in-range sprites fall to per-dot until Rung 2's
sprite-eval batch is proven). This keeps Rung 1's `step_whole_scanline` to pure
counter/`v`/vblank math — no overflow risk — and still collapses the ≥99% of lines that
carry no sprite activity of interest. `odd_skip(line)` returns 1 iff `line ==
PRE_RENDER_SCANLINE && frame_is_odd && rendering_enabled_at_prerender_end`, else 0.

**`step_whole_scanline(line)`** (Rung 1 scope): vblank scanlines → counter math + the
(241,1) `set_vblank`; rendering-disabled visible lines → pure `cycles += end;
scanline += 1;` + frame-boundary work; rendering-enabled non-sprite lines →
bounded-loop `v` evolution (recommended) + counter/frame-boundary work + the dot-260/324
`on_scanline_tick` for A12 mappers. Pre-render (261) → include the 280–304 vertical
`v←t` copy and the odd-frame skip. `#[inline(never)]`.

**Correctness gate (the load-bearing test):** `advance(n) == n × tick`. For a matrix of
entry states `(scanline, scanline_cycle)` spanning a full frame and every `n` in
`{1,3,7,341,683,…}`, clone a `Ppu`, run `advance(n)` on one and `n×tick` on the other,
assert `observable_digest`-relevant PPU sub-state identical: `scanline`,
`scanline_start_cycle`, `cycles`, `frame`, `regs.v`, `regs` file, `sprite_0_on_scanline`,
`nmi_occurred`, `ppu_status`. This is cheap (one frame) and exhaustive over the entry
lattice — it is the primary defense against risk #5.

**Acceptance:** all universal gates green; `make ppu_layout_check` reports hot symbols
unchanged; gate-OFF A/B is 0.0%; gate-ON SMB shows a positive fresh-PGO delta with no
full-render (spectator) regression >1%; Zelda shows ~0 (expected — Level B is Rung 2).
**Revert:** `set_ppu_scanline_advance(false)` (zero rebuild) or revert the one-line
loop swap.
