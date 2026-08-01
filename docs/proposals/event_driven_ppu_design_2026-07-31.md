# Event-driven PPU catch-up — implementation blueprint

**Status: design/blueprint (execute tomorrow). Date: 2026-07-31.**
**Companion WHY note:** `docs/proposals/event_driven_ppu_and_subcycle_bus_2026-07-31.md`
(read first — this doc is the HOW).
**Supersedes for the fidelity track:** the 07-14 perf plan
`docs/proposals/ppu_event_driven_catchup.md` (scanline-granular `advance(dots)`,
Rung 1/Level A). That design partially landed and is **default-OFF** (see §0.2);
this blueprint reuses its closed-form building blocks but re-anchors the entry
point at the bus layer with an absolute target dot.

All file:line citations are relative to `nes_core/src/` unless a `tests/` or
`scripts/` prefix is given. Every empirical number is receipted in
`runs/event_ppu_design/mmio_access_receipt.json` (regenerate with
`runs/event_ppu_design/mmio_probe.py`).

---

## 0. Ground truth: what exists today

### 0.1 The interleave

`Nes::tick` (`nes.rs:512-557`) is the per-cycle reference. Order within one CPU
cycle:

1. `apu.tick` (`nes.rs:518`) — 1 APU cycle, may return a DMC/OAM stall.
2. **3× `ppu.tick` + `cpu.set_nmi_line` after each dot** (`nes.rs:524-528`).
3. `mapper.set_cpu_cycle(self.cycles)` (`nes.rs:532`) — feeds MMC1 RMW filter and
   MMC3 A12 cycle clock.
4. `update_irq_line` (`nes.rs:534`).
5. `cpu.tick(&mut bus)` (`nes.rs:549`) — the CPU cycle; **any bus access here
   observes the PPU as of the end of step 2** (3 dots already advanced).
6. `update_irq_line` again (`nes.rs:552`); `cycles += 1` (`nes.rs:554`).

So a bus access is resolved against PPU state at the **end** of the CPU cycle's
three dots — up to ±3 dots from where the real access lands. That is precisely
the race class the WHY note targets.

The hot production paths do NOT run per-cycle: they run the ASM/bulk fast paths
in `Nes::step` (`nes.rs:152-446`) and catch the PPU up in a tail loop
(`for _ in 0..cycles { apu.tick(); ppu.tick_three() }`, `nes.rs:347-350` and
`nes.rs:423-426`). The mid-instruction MMIO callback `tick_cpu_cycles_with`
(`system_bus.rs:81-91`, installed as `tick_impl` at `nes.rs:280-296`) syncs the
PPU *before* each MMIO access inside an ASM batch — so the "PPU is current at
every bus access" invariant already holds; only the **dot granularity** (whole
CPU cycles, never sub-cycle) is wrong.

### 0.2 The existing (dead-default) `Ppu::advance(dots)`

`Ppu::advance` (`ppu.rs:2157-2280`), `step_whole_visible_scanline`
(`ppu.rs:2305-2341`), `scanline_boundary_advance` (`ppu.rs:2290-2294`) are the
07-14 Rung-1 scanline batcher, wired into the catch-up loops at `nes.rs:334-345`
and `nes.rs:410-421` behind `ppu.scanline_advance_enabled()`
(`ppu.rs:620-622`). **It defaults OFF** (`ppu.rs:462`) for a receipted reason
that dictates this blueprint's architecture:

> "at bulk=1 the Level-A slices are 3 dots, so per-slice classification runs
> ~89k times/frame and regresses the 16-worker pool ~28% (PGO A/B 2026-07-15)"
> — `ppu.rs:457-461`.

The lesson: **do not classify per CPU cycle.** Flush the PPU only at the events
that matter. Receipt (`runs/event_ppu_design/mmio_access_receipt.json`, SMB
in-game): **454 PPU-register accesses per frame**, vs 89342 dots or ~29781 CPU
cycles — a ~65× smaller flush count. That is the whole perf case for anchoring at
the bus layer instead of the per-cycle loop.

### 0.3 The five hardware-timing flags (must be preserved byte-for-byte)

All default OFF, config not savestate, mirrored env+pool setters:

| flag | field | effect | receipt anchor |
|---|---|---|---|
| `hw_mmio_read_timing` | `cpu.rs:266` | abs-mode PPU **reads** commit on the instruction's FINAL cycle (`defer_ppu_read` `cpu.rs:1844-1847`; late handlers `abs_late_ppu_read_{a,x,y,bit}` `cpu.rs:1859-1891`) instead of the LaiNES cycle-0 early commit (`cycle_zero_early_commit` `cpu.rs:1751-1839`) | CV boot vblank-wait `LDA $2002` at PPU 241,8 |
| `hw_mmio_write_timing` | `cpu.rs:280` | abs-mode PPU **writes** defer to final cycle (`defer_ppu_write` `cpu.rs:1851-1853`; rides the `$4014` late-write plumbing `sta/stx/sty_abs_4014_late_write` `cpu.rs:1705-1731`) | CV frame-11 `STA $2000` NMI re-enable |
| `hw_nmi_poll_timing` | `cpu.rs:293` | interrupt poll uses `nmi_poll_latch` (state as of the previous cycle's end) so an edge on the final cycle defers service one instruction (`poll_interrupts` `cpu.rs:561-586`); **disables the ASM/bulk fast paths** (`nes.rs:194`) | CV frame-3792 NMI entry 241,23 vs 241,31 |
| `hw_reset_alignment` | `nes.rs:52` | reset lands first fetch at CYC=7, PPU dot 25 (`nes.rs:788-804`, `SystemBus::tick_ppu_dots_discard` `system_bus.rs:120-131`) instead of CYC=16/dot 24 | OAM-DMA 513/514 parity |
| `hw_dmc_stall_timing` | `apu.rs:70` | DMC fetch stall 3 (aligned) vs flat 4 (`apu.rs:459-460`, `step_reader` `apu.rs:1100-1126`) | long-tape drift |

**The intra-cycle-offset machinery already half-exists:** the late-commit
handlers mark the exact **cycle** of each MMIO access (final cycle of the
instruction). What is missing is the sub-cycle **dot** the access samples within
that cycle. Receipt: under the shipped default, **93% (2113/2270) of PPU accesses
land at `abs_dot % 3 == 2`** — the last of the cycle's three dots — confirming
the "observe PPU at end of CPU cycle" model empirically
(`runs/event_ppu_design/mmio_access_receipt.json`). Calibrate the true offset the
same way boot constants were (CYC=7/dot 25 against Mesen).

---

## 1. EVENT INVENTORY

Every CPU-observable PPU effect, its exact current dot/cycle semantics, and
whether the target design can predict it in closed form or must fall to a per-dot
simulation window. "Predictable" = computable from `(cycles, scanline,
scanline_start_cycle, regs.ctrl, regs.mask, regs.v/t/x, frame parity)` all of
which are **constant across any catch-up slice** (a register write is MMIO → a
slice boundary; verified `ppu.rs:1818-1823`, old-plan §2).

Frame geometry: `CYCLES_PER_SCANLINE = 341` (`ppu.rs:14`); visible 0..=239
(`ppu.rs:76-77`); vblank start 241 (`ppu.rs:78`); pre-render 261 (`ppu.rs:79`);
frame total `262*341 = 89342` dots (`ppu.rs:654`). `scanline_cycle() = cycles -
scanline_start_cycle` (`ppu.rs:1707-1709`). **Display-convention note:** we label
pre-render as scanline **261**, not Mesen's **-1**; our dot labels run +1 vs Mesen
at the same machine moment (naming only — receipted in
`memory/project_mesen_lockstep_dawn_2026-07-31.md`: "scanline 261-vs-(-1) naming
is display only").

### 1.1 vblank set — **PREDICTABLE**
Set at the tick whose start state is `(scanline=241, scanline_cycle=1)`, i.e.
frame dot `241*341+1 = 82182`. Handled by the vblank fast-path
`ppu.rs:1743-1747` calling `set_vblank` (`ppu.rs:1699-1701`, sets
`nmi_occurred=true`). Closed form already exists as the target position in
`cpu_cycles_until_nmi_fire` (`ppu.rs:653`). Independent of scroll/mask.

### 1.2 vblank clear + flag clears — **PREDICTABLE**
At `(261,1)`: `clear_vblank` + `SPRITE_OVERFLOW=false` + `SPRITE_ZERO_HIT=false`
(`ppu.rs:1993-1997`). Frame dot `261*341+1 = 89012`.

### 1.3 NMI line assert / deassert — **PREDICTABLE edge, boundary deassert**
The line the CPU sees is `nmi_output && nmi_occurred` (`nes.rs:527`,
`system_bus.rs:237-239`). **Assert edge** = the vblank-set dot (§1.1) iff
`nmi_output` (PPUCTRL bit 7, `write_ppu_ctrl` `ppu.rs:791`). Closed form:
`cpu_cycles_until_nmi_fire` (`ppu.rs:644-663`) — returns `None` if `!nmi_output ||
nmi_occurred`, else CPU cycles to the fire tick. **Deassert** happens via a `$2002`
read (`read_ppu_status` sets `nmi_occurred=false`, `ppu.rs:758`) or the (261,1)
clear — both are already slice boundaries (a read is a bus access; the clear is
the predicted §1.2 event). The edge feeds the CPU through `set_nmi_line`
(currently hoisted to end-of-batch, `nes.rs:353-355`, `nes.rs:434-436`).

### 1.4 sprite-0 hit set — **PER-DOT WINDOW (default)**
Set in `render_pixel` (`ppu.rs:1507-1520`) at dot `x+1` on the sprite-0 scanline
when BG-opaque ∧ sprite-opaque ∧ `show_bg` ∧ `show_sprites` ∧ ¬(x<8 left-mask) ∧
`x<255` ∧ not-already-set. Three code paths must stay identical:
- **full-render / per-dot:** the body above.
- **skip_render fast path:** `render_pixel` still runs the hit test but bails
  before pixel extraction when the flag is already latched or sprite-0 isn't on
  this line (`ppu.rs:1476-1481`, `ppu.rs:1504-1520`). The dependency on OAM +
  scroll + pattern opacity makes the exact hit dot non-closed-form-cheap.
- **existing scanline-batch:** `advance` deliberately **refuses to batch** a
  sprite-0-pre-hit line — the gate at `ppu.rs:2246-2256` requires
  `(!oam.sprite_0_found && !sprite_0_on_scanline) || SPRITE_ZERO_HIT` (note the
  two-flag subtlety: the x=0 pixel consults the stale `sprite_0_on_scanline`
  before dot-1's `sprite_evaluation_init` swap — regression-tested
  `whole_line_batch_latches_x0_sprite0_hit_from_stale_flag`).

Target: same policy — the sprite-0 line runs a **per-dot window**. Optional
closed-form analytic predictor is deferred (old plan §4 Rung 4), on the
correctness path only behind its own pixel-diff gate.

### 1.5 sprite overflow — **PER-DOT WINDOW unless ≤8 in range**
Set in `sprite_evaluation_write_byte` (`ppu.rs:952`) during the dots-65..=256
eval, governed by the hardware n/m diagonal bug (`ppu.rs:936-969`). The existing
`step_whole_visible_scanline` reproduces it verbatim by replaying the 96 read/write
pairs (`ppu.rs:2320-2323`) — exact because a batched line carries no mid-line
`$2002` read to observe an intermediate value. Fast pre-pass: count in-range
sprites; if ≤8, overflow is impossible → skip the n/m walk.

### 1.6 A12 rise for MMC3 (260/324 rule) — **PREDICTABLE (heuristic) + PER-DOT (fetch-driven)**
Two clock sources, deliberately non-overlapping:
- **Heuristic** (`ppu.rs:1839-1862`): on rendering-enabled visible+prerender
  lines, `mapper.on_scanline_tick()` fires at dot **260** if `(BG=$0000,
  sp=$1000)` or dot **324** if `(BG=$1000, sp=$0000)`; same-table → no clock. For
  MMC3 this decrements the IRQ counter (`on_scanline_tick` `mapper4.rs:609-619` →
  `clock_irq` `mapper4.rs:436-447`). PREDICTABLE from PPUCTRL (constant in slice).
- **Fetch-driven** (`clock_a12` `mapper4.rs:477-493`, invoked from
  `chr_read_byte` `mapper4.rs:582-585`): tracks real A12 off the CHR address bus,
  clocks on a filtered low→high edge, but only while the heuristic is *silent*
  (`HEURISTIC_OWNS_CYCLES=240`, `mapper4.rs:481-486`). Needed for 8×16 / same-table
  (Kirby). Depends on the actual fetch stream → **PER-DOT WINDOW**.
IRQ **fire** is closed-form given `irq_counter`/`irq_latch`: assert when the
counter reaches 0 (`clock_irq` `mapper4.rs:444-446`) → a
`cpu_cycles_until_irq_fire()` mirroring `cpu_cycles_until_nmi_fire`. Recompute
after any MMC3 `$8000-$FFFF` write (a bus boundary). Line feeds the CPU via
`irq_pending()` (`mapper4.rs:722-724`) → `update_irq_line` (`nes.rs:560-563`).

### 1.7 odd-frame dot skip — **PREDICTABLE**
Pre-render 261 is 340 dots (skips the last) iff `rendering_enabled && odd frame`
(`ppu.rs:2002-2008`: end-of-scanline at `scanline_cycle == 339` normally, or
`== 338` on odd-frame pre-render). `cpu_cycles_until_nmi_fire` explicitly ignores
this (±1 dot, `ppu.rs:641-643`); the target's authoritative counter must NOT.

### 1.8 `$2002` / `$2004` / `$2007` read side effects
- **`$2002`** `read_ppu_status` (`ppu.rs:751-761`): resets `w` toggle
  (`ppu.rs:753`), returns `nmi_occurred<<7 | status&0x60 | gen_latch&0x1F`,
  **clears `nmi_occurred`** (`ppu.rs:758`). This is the vblank/sprite-0 poll — the
  fidelity-critical access. Receipt: **~3.6 `$2002` reads/frame** on SMB, landing
  mid-scanline (e.g. (43,173),(241,146),(246,160),(176,142)) — exactly the
  mid-cycle races the ±3-dot model mis-resolves.
- **`$2004`** `read_oam_byte` (`ppu.rs:763-773`): returns `0xFF` during dots
  1..=64 of a visible line (sprite-eval clear), else `oam[oam_addr]`. Reads the
  live sprite-eval scratch → sensitive to the exact dot on a rendering line.
- **`$2007`** `read_ppu_data_byte` (`ppu.rs:847-865`): buffered read (non-palette
  returns the old buffer then refills; palette returns immediately + refills the
  mirror), then `inc_ppu_addr` (`ppu.rs:812-825`). During rendering `inc_ppu_addr`
  does coarse-x + y increments (`ppu.rs:814-818`) — a scroll glitch.
Every `read_byte` first fills `ppu_gen_latch = val` (`ppu.rs:2362`); `note_ppu_reg_access`
(`ppu.rs:491-511`) is stats-only.

### 1.9 `$2000/$2001/$2005/$2006/$2007` write effects
Dispatch `write_byte` (`ppu.rs:2387-2496`); every write fills `ppu_gen_latch =
value` first (`ppu.rs:2410`, open-bus, load-bearing on cold boot). **29658-cycle
warmup:** PPUCTRL/MASK/SCROLL/ADDR writes are ignored while `cycles < 3*29658`
(`ppu.rs:2419-2426`).
- **`$2000`** `write_ppu_ctrl` (`ppu.rs:786-792`): sets `t` bits 10-11
  (`ppu.rs:788`), `ppu_ctrl`, and **`nmi_output`** (`ppu.rs:791`) — can raise the
  NMI edge mid-vblank (the CV frame-11 case → `hw_mmio_write_timing`).
- **`$2001`** (`ppu.rs:2485-2488`): sets `ppu_mask`, refreshes `ppu_mask_cache` +
  `grey_mask` (`refresh_ppu_mask_cache` `ppu.rs:544-556`). Toggles
  `rendering_enabled` — which the target treats as constant per slice, so this MUST
  be a slice boundary.
- **`$2005`** `write_ppu_scroll` (`ppu.rs:794-810`): first write sets `x` (fine-x)
  + `t` coarse-x; second sets `t` fine-y/coarse-y; toggles `w`.
- **`$2006`** `write_ppu_addr` (`ppu.rs:827-845`): first write sets `t` high;
  second sets `t` low then **`v = t & 0x7FFF`** (`ppu.rs:841`) — mid-frame scroll
  change.
- **`$2003`** OAMADDR (`ppu.rs:2489`); **`$2004`** `write_oam_byte`
  (`ppu.rs:775-784`, ignored during rendering); **`$2007`** `write_ppu_data_byte`
  (`ppu.rs:867-912`) → VRAM/palette write + `inc_ppu_addr`.
Receipt register mix (SMB, /frame): `$2007:W` 432, `$2006:W` 7, `$2005:W` 5,
`$2000:W` 4, `$2002:R` 3.6, `$2001:W` 1.4, `$2003:W` 0.8. The `$2007` flood is
vblank nametable streaming — cheap to batch (no CPU-observable feedback except the
buffer/`v`), the polls are the rare fidelity-critical minority.

### 1.10 OAM DMA interplay — **KEEP PER-CYCLE (island)**
`$4014` write arms it (`system_bus.rs:195-196` → `oam_dma.activate`
`oam_dma.rs:36-41`); `handle_oam_dma` (`nes.rs:565-611`) runs a 513/514-cycle
alternating dummy/read/write transfer, each even cycle a bus read, each odd cycle
a `ppu.write_byte($2004)`. The DMA reads **no PPU-derived state**, but it drives
`$2004` writes at exact cycles and its 513/514 length is parity-load-bearing
(`hw_reset_alignment`). Frame drivers already single-cycle-tick while
`oam_dma.active` (`pool.rs:212-213`, `python.rs:777-778`). Receipt: ~1 DMA/frame
on SMB. **The event PPU must not batch across an active DMA** — flush, then let the
per-cycle DMA path run, exactly as today.

### 1.11 palette / VRAM access windows
Palette RAM read/write mirror handling `PaletteRam::index` (`ppu.rs:2768-2793`);
`$3F00` backdrop drives forced-blank `clear_pixel` (`ppu.rs:1560`). VRAM
nametable writes via `$2007` mutate render inputs but produce no CPU-observable
feedback except the `$2007` read buffer + `v`. No timing window of its own beyond
`inc_ppu_addr` rendering-vs-not (§1.8).

---

## 2. CALL-SITE MAP

Every driver that advances the PPU, whether it is PGO-hot, and how the target
design touches it.

| # | call site | file:line | PPU advance mechanism | hot? | disposition |
|---|---|---|---|---|---|
| A | `Nes::tick` per-cycle | `nes.rs:524-528` | `3× ppu.tick` + `set_nmi_line`/dot | compat/DMA only (hot only under `hw_nmi_poll_timing` or active DMA) | **reference path** — never modified; the per-dot oracle |
| B | ASM MMIO callback | `system_bus.rs:81-91` (`tick_cpu_cycles_with`), installed `nes.rs:280-296` | `for cpu_cycles { apu.tick; ppu.tick_three }` | **HOT** (mid-instruction PPU sync before every ASM MMIO) | **primary sub-cycle integration point** (§3.4) |
| C | ASM remainder catch-up | `nes.rs:334-351` | `advance(remaining*3)` gated, else `tick_three` loop | **HOT** (every ASM instruction's non-MMIO tail) | flush point + closed-form slice |
| D | bulk-step catch-up | `nes.rs:410-427` | `advance(cycles*3)` gated, else `tick_three` loop | **HOT** (Rust bulk opcodes) | flush point + closed-form slice |
| E | `step_lda_immediate_fast` | `nes.rs:482-487` | `for cpu_cycles { apu.tick; 3× ppu.tick }` | warm (bench/legacy) | flush point |
| F | `SystemBus::tick_cpu_cycles_discard` | `system_bus.rs:97-115` | `tick_three` loop, null sinks | reset/boot only | leave (config path) |
| G | `SystemBus::tick_ppu_dots_discard` | `system_bus.rs:120-131` | raw `ppu.tick` ×dots | `hw_reset_alignment` boot only | leave |
| H | `Pool::advance_one_frame` | `pool.rs:180-222` | drives B/C/D via `nes.step`; single-cycle `tick` during DMA + trailing to hit 29781 | **HOTTEST** (all training) | unchanged; benefits transitively |
| I | `Environment::advance_one_frame` | `python.rs:728-787` | same as H, plus `hw_frame_anchor` PPU-frame mode (`python.rs:731-748`) | hot (single-env + spectator) | unchanged |
| J | `handle_oam_dma` | `nes.rs:565-611` | per-cycle `$2004` writes | per-DMA | **island — stays per-cycle** (§1.10) |
| K | `Ppu::advance` (existing) | `ppu.rs:2157-2280` | scanline-granular `dots` batcher | default OFF | **refactor into `advance_to`** (§3.1) |
| L | Mapper `step` (unused MMC3) | `mapper4.rs:596-603` | dead alt scanline hook | cold | ignore |

Bus access dispatch that must gain the catch-up hook: `SystemBus::read_byte`
`0x2000..=0x3FFF` arm (`system_bus.rs:140`), `write_byte` `$4014` and
`0x2000..0x4000` arms (`system_bus.rs:195-200`). The `note_mapper_write_during_render`
call (`system_bus.rs:210`) marks MMC3 register writes as horizon-recompute points.

**PGO-critical set:** B, C, D, H (and transitively I). Everything the layout gate
protects. The event PPU changes B/C/D; H/I are untouched call graphs that inherit
the speedup.

---

## 3. TARGET ARCHITECTURE

### 3.1 `Ppu::advance_to(target_dot)` — the single entry point

Replace the `dots`-relative `Ppu::advance` (`ppu.rs:2157`) with an **absolute**
target on the PPU's own monotonic `cycles` counter (`ppu.rs:92`):

```
#[inline(never)]
fn advance_to<V: VideoSink>(&mut self, target: u64, mapper, sink):
    debug_assert!(target >= self.cycles);
    while self.cycles < target:
        let next = self.next_event_dot(mapper);      // §3.2, closed form
        let stop = next.min(target);
        if self.can_closed_form(stop):               // whole event-free run
            self.run_closed_form_to(stop, mapper);   // §3.3
        else:
            // per-dot simulation window (sprite-0 line, overflow eval,
            // fetch-driven A12, pre-render 261, the 3 dots straddling
            // an event, and the <3-dot remainder): drive tick_three
            // chunks then <3 tick() — the EXACT existing fallback
            // (ppu.rs:2270-2278), byte-for-byte with tick.
            self.run_per_dot_to(stop, mapper, sink)
```

`self.cycles` is the authoritative clock (already true — `tick` owns it,
`ppu.rs:1748`,`1999`,`2339`). `advance_to(target) == (target - start)/... ticks`
by construction; the closed forms are proven-equal to the per-dot body they
replace (existing `step_whole_visible_scanline` `ppu.rs:2305-2341` is the model).

### 3.2 Event-prediction functions — one per event type

`next_event_dot(mapper) -> u64` returns the **soonest** absolute dot at or after
`self.cycles` at which a CPU-observable effect fires, so the loop can fast-forward
to just before it and then simulate the containing window per-dot. Each predictor
is closed-form from slice-constant state:

- `next_vblank_set_dot()` — §1.1: `frame_base + 82182` if not yet past this frame,
  else `+89342`. Statically predictable.
- `next_vblank_clear_dot()` — §1.2: `frame_base + 89012`. Static.
- `next_nmi_edge_dot()` — §1.3: `= next_vblank_set_dot()` iff `nmi_output`, else
  ∞. (Deassert is a boundary, not a horizon.) Derived from
  `cpu_cycles_until_nmi_fire` logic (`ppu.rs:644-663`), promoted to absolute dots
  and made odd-skip-exact.
- `next_a12_heuristic_dot()` — §1.6: on a rendering-enabled visible/prerender
  line, `line_base + 260` or `+324` per PPUCTRL table select; ∞ if same-table or
  rendering off. Static within slice.
- `next_irq_fire_dot(mapper)` — §1.6: `cpu_cycles_until_irq_fire()` = current
  scanline + `irq_counter` heuristic clocks, converted to a dot; recompute after
  any MMC3 write. Predictable given counter/latch.
- `next_odd_skip_dot()` — §1.7: `line 261 dot 340` iff `rendering_enabled && odd
  frame`. Static.

**Require a per-dot simulation window (NOT closed-form on the correctness path):**
- sprite-0 hit line (§1.4): return `line_base` for the whole sprite-0 line so the
  window covers it; gate exactly as `ppu.rs:2246-2256`.
- overflow eval when >8 sprites in range (§1.5): the eval window dots 65..=256.
- fetch-driven A12 (8×16 / same-table, MMC2/MMC4 CHR latch) (§1.6, §1.10): force
  per-dot — reuse the existing mapper gate `chr_static_ptr().is_some() &&
  !uses_scanline_irq()` (`ppu.rs:2175-2176`, `nes.rs:735-737`) to pick the whole
  mapper into per-dot when it can latch/clock off fetches.

`can_closed_form(stop)` is true when the run from `self.cycles` to `stop` is one
of the existing proven-safe classes: pure-counter lines (post-render 240, vblank
242..=260, forced-blank visible — `ppu.rs:2215-2217`), or a whole
rendering-enabled visible line owned from dot 0 with no sprite-0/overflow
(`ppu.rs:2246-2252`). Everything else → per-dot window.

### 3.3 `run_closed_form_to` — reuse the landed building blocks

- pure-counter run: `cycles += span` + `scanline_boundary_advance`
  (`ppu.rs:2218-2224`).
- whole visible line: `step_whole_visible_scanline` (`ppu.rs:2305-2341`) —
  sprite-eval replay + `v` scroll sequence + counters, verbatim per-dot order,
  skipping only the unobserved render pipeline.
- **New:** emit `mapper.on_scanline_tick()` at the predicted A12 dot **inside** a
  batched rendering line (the closed form currently omits it because the existing
  `advance` only batches when `!uses_scanline_irq()`; the event design adds it so
  MMC3 lines can batch up to the horizon).

### 3.4 Bus catch-up integration point (sub-cycle offset)

The fidelity payoff lives here. Two sub-parts:

**(a) Owed-dots deferral at the sync layer (Level B, the perf win).** Add
`ppu_owed_dots: u64` alongside the CPU batch accounting (a `Nes` field, or thread
it through `tick_cpu_cycles_with`). The per-cycle catch-up loops (call sites B/C/D)
tick **APU only** and accumulate `ppu_owed_dots += 3` per CPU cycle. Flush
(`ppu.advance_to(ppu.cycles + ppu_owed_dots)`, zero the counter) at exactly:
1. a `$2000-$3FFF` or `$4014` bus access (`system_bus.rs:140`,`195-200`) — flush
   **before** servicing;
2. the NMI horizon (already the ASM batch cap, `nes.rs:238-247`);
3. the MMC3 IRQ horizon (`next_irq_fire_dot`);
4. an active OAM DMA (§1.10) and any of the per-dot-window mappers.
APU/DMC/frame-IRQ stay per-cycle (`apu.tick` `nes.rs:339`,`348`) — never batched.

**(b) Sub-cycle dot offset (the fidelity win).** At a bus access, the flush target
is not `cpu_cycle * 3` (end of cycle) but `cpu_cycle_of_access * 3 +
intra_cycle_offset`. The access's **cycle** is already pinned by the hw flags'
late-commit handlers (final instruction cycle; `cpu.rs:1859-1891`,
`1705-1731`). The **dot offset** is a small calibrated constant:
- convention: reads sample at `READ_DOT_OFFSET`, writes commit at
  `WRITE_DOT_OFFSET` within the cycle (∈ {0,1,2}); the shipped model is
  effectively offset 2 for both (receipt: 93% at `abs_dot%3==2`).
- calibrate against Mesen's catch-up-at-master-clock the way boot CYC=7/dot-25 was
  (`memory/project_mesen_lockstep_dawn_2026-07-31.md`), using the CV tape's
  first-divergence finder.
Under legacy (`hw_event_ppu=false`) the offset is forced to the end-of-cycle value
so behavior is byte-identical.

### 3.5 NMI/IRQ flow back to the CPU between events

After each flush / at each batch end, sample and push the lines exactly as the
current hoisted pattern does:
- `cpu.set_nmi_line(ppu.nmi_output && ppu.nmi_occurred)` (`nes.rs:353-355`,
  `434-436`);
- `update_irq_line()` (`apu.irq_pending() || mapper.irq_pending()`,
  `nes.rs:560-563`);
- `cpu.poll_interrupts()` at the instruction boundary (`nes.rs:357`,`438`).
The correctness argument that made end-of-batch sampling valid still holds
(neither signal can de-assert mid-slice because a `$2002` read or mapper write is a
flush point — `nes.rs:308-324`). With sub-cycle offset the **edge dot** is now
exact, so `hw_nmi_poll_timing`'s second-to-last-cycle poll (`cpu.rs:568-586`) sees
the edge at the true cycle. `hw_event_ppu` should, in its first landing, **imply
per-cycle NMI sampling** (like `hw_nmi_poll_timing` disables the ASM/bulk paths,
`nes.rs:194`) so edge placement is unconditionally correct; relax to hoisted
sampling only once the horizon math is receipted.

---

## 4. STAGED PLAN

New runtime gate `hw_event_ppu: bool` on `Cpu`/`Nes` (config, not savestate),
default **OFF**, mirrored env+pool setters, exactly like the five flags
(`python.rs:451-496`, `pool.rs:1090-1138`). Gate-OFF must be byte-identical
(A/B 0.0%). Universal gate for **every** stage: `make parity` (146 tapes,
`Makefile:91-92`) + lib suite (158, `cd nes_core && cargo test`) +
`tests/ppu_shadow_oracle.rs` extended with a `hw_event_ppu` on/off axis +
`refined_off_vs_on_state_parity` `observable_digest`
(`tests/skip_render_parity.rs:249-337`) + `make ppu_layout_check`
(`Makefile:107-108`, golden disasm of `Ppu::tick`) unchanged.

### Stage 1 — `advance_to` skeleton, equal-by-construction
Scope: add `advance_to(target)` (§3.1) with **only** the per-dot window inner path
(no closed forms yet) — a pure re-parameterization of the existing per-dot loop.
Wire call sites C/D to call `advance_to(ppu.cycles + cycles*3)` under the gate;
legacy path unchanged. Add `hw_event_ppu` field + setters.
Gate: universal + a brute-force `advance_to(t) == (t-start)/3 × tick_three`
equivalence test over a full frame of entry states (mirror the old plan's Rung-1
gate). **Proves the plumbing; zero behavior change.** Revert: flip bool.

### Stage 2 — vblank / NMI closed form
Scope: `next_vblank_set_dot`, `next_vblank_clear_dot`, `next_nmi_edge_dot`,
`next_odd_skip_dot`; enable closed-form pure-counter + vblank runs in
`run_closed_form_to`. This makes `advance_to` fast-forward all 21 vblank lines +
post-render in O(1).
Gate: universal + a `cpu_cycles_until_nmi_fire`-vs-actual equivalence assertion
(clone PPU, predict, run, compare fire dot) + **Mesen NMI-entry lockstep** on the
CV tape (the d_cyc==0 × 11-frame receipt, `memory/…dawn…`). Revert: sub-gate to
per-dot windows.

### Stage 3 — sprite-0 prediction / window
Scope: sprite-0 line policy (§1.4) — default the per-dot window (reuse
`ppu.rs:2246-2256` gate); enable whole-visible-line closed form
(`step_whole_visible_scanline`) for non-sprite-0 rendering lines + the ≤8-in-range
overflow fast pass (§1.5).
Gate: universal + SMB HUD sprite-0 split pixel-diff (fs=1 vs fs=16) + shadow
oracle at 2000 frames on SMB/Zelda/Contra. Revert: force sprite lines per-dot.

### Stage 4 — A12 / MMC3 horizon
Scope: `next_a12_heuristic_dot`, `next_irq_fire_dot(mapper)`; emit
`on_scanline_tick` inside batched rendering lines; recompute the horizon after any
MMC3 `$8000-$FFFF` write (`system_bus.rs:210`). Fetch-driven-A12 mappers (Kirby,
MMC2/MMC4) stay per-dot via the mapper gate.
Gate: universal + SMB3 status-bar split pixel-diff (fs=1 vs fs=16) + a Kirby 8×16
no-regression run + **Mesen IRQ-cycle lockstep on an MMC3 raster title
(mandatory)**. Revert: MMC3 falls back to per-dot.

### Stage 5 — bus catch-up wiring (owed-dots + sub-cycle offset)
Scope: `ppu_owed_dots` deferral at call sites B/C/D (§3.4a); flush hooks in
`SystemBus::read_byte`/`write_byte` (`system_bus.rs:140`,`195-200`); the
`READ_DOT_OFFSET`/`WRITE_DOT_OFFSET` sub-cycle convention (§3.4b), calibrated
against Mesen. This is the fidelity headline — the ±3-dot race closes.
Gate: universal + `PPU_FF_SHADOW`-style differential oracle firing at **every**
`$2002`/IRQ boundary across MMC1/MMC3/NROM with randomized input +
**CV-tape first-divergence frame must not regress** (currently ~4118, target:
push past it) via `scripts/tracing/` first-divergence finder. Revert: flush
every cycle at end-of-cycle offset == today.

### Stage 6 — perf pass with PGO receipts
Scope: promote the bounded inc-loops in `step_whole_visible_scanline` to hand
-derived closed forms only where PGO shows it worth; tune flush granularity;
consider raising `asm_bulk_cycles` for MMC1/UxROM now the PPU is no longer the
3-dots/cycle anchor (`mapper.rs:156-167`).
Gate: `make ppu_layout_check` (hot symbols unchanged) + fresh dual-corpus PGO
(`scripts/pgo_build.sh`, `cargo clean` between modes) + `bench_hot_path`
(`Makefile:122-123`) A/B on a bulk==1 ROM (Zelda) and a bulk>1 ROM (SMB) with
**no full-render/spectator regression >1%**. Receipt every % before trusting it.

---

## 5. RISK REGISTER

1. **Savestate compat (ppu State fields).** `ppu::State` (`ppu.rs:378-405`) is
   serialized; `ppu_owed_dots` and any new counter MUST be flushed to zero at
   `get_state` boundaries (the CPU sits between instructions there) OR excluded
   like `nmi_poll_latch` (`cpu.rs:295-300`, re-synced in `apply_state`
   `cpu.rs:368-370`). `apply_state` nulls `chr_cache_ptr` (`ppu.rs:704`); the
   event path must re-query the mapper the same way. *Mitigate:* keep `advance_to`
   stateless beyond `ppu.cycles`; flush owed-dots before any `get_state`. Gate:
   existing skip_render_parity `load_state` round-trip (`tests/…:227-236`) + a
   mid-batch save/load fuzz.

2. **Batched / skip-render interactions.** `advance_to`'s closed forms omit
   `render_pixel` (`ppu.rs:1463-1547`), so they are **only valid under
   `skip_render`** — the existing `advance` enforces this (`ppu.rs:2183-2188`) and
   `advance_to` must too. `BatchedRenderMode::{Verify,Replace}` (`ppu.rs:351-376`)
   and `skip_pixel_writes_this_scanline` (`ppu.rs:228`,`1488-1493`) run inside
   `tick`; under full render / batched modes, `advance_to` MUST fall to the per-dot
   body (`ppu.rs:2183`). *Mitigate:* mirror the `!skip_render || !mapper_batchable`
   guard; the shadow oracle already holds `skip_render` on to force the fast path.

3. **ASM CPU path.** `hw_nmi_poll_timing` already **disables** the ASM/bulk fast
   paths (`nes.rs:190-194`) because they batch cycles and would take NMIs at the
   wrong boundary — the precedent. `hw_event_ppu` changes the interleave under the
   ASM MMIO callback (call site B); its first landing should likewise force the
   per-cycle NMI-sample path (§3.5) or disable ASM/bulk, then re-enable per-stage
   once the horizon + sub-cycle offset are receipted. The ASM path was structurally
   disabled once before for a PPU+NMI batching bug
   (`memory/project_asm_cpu_disabled_2026-04-26`) — do not fight that history;
   correctness first, re-enable behind receipts.

4. **PGO bench discipline.** The PPU is the most PGO-sensitive symbol in the tree;
   history: tick_n −55%, inline-tick −52%, skip_bg silently removed. **Mandatory:**
   `tick`/`tick_three` (`ppu.rs:1719-2128`) NEVER modified (they are the reference
   + the layout-gate fingerprint); `advance_to`/predictors are new
   `#[inline(never)]` symbols in their own region (as the existing `advance` is,
   `ppu.rs:2130-2156`); `cargo clean` between PGO and plain
   (`memory/feedback_pgo_clean_between_modes`); regenerate profdata on the new code
   before trusting any % (`memory/feedback_pgo_rerun_after_code_change`); measure
   before/after on the hot-path refactor (`memory/feedback_measure_before_after`).
   `make ppu_layout_check` rejects any hot-symbol machine-code change.

5. **Deferred-PPU staleness at an unmodelled observable (Level B).** A read path
   that samples PPU-derived state without a flush — a mapper `irq_pending` between
   horizons, a `$2004` OAM read mid-render, a DMC-vs-PPU interaction. *Detect:*
   the differential shadow oracle (`tests/ppu_shadow_oracle.rs`) at **every**
   `$2002`/IRQ boundary with randomized input; Mesen lockstep is the backstop.
   *Mitigate:* enumerate flush points (§3.4a list); any digest divergence = a
   missing flush.

6. **odd-frame skip / frame-boundary drift in the authoritative counter.**
   `advance_to` becomes the cycle authority; an off-by-one in `next_odd_skip_dot`
   or the wrap `write_frame`/`skip_render` reset (`ppu.rs:2052-2084`) desyncs
   `cpu_cycles_until_nmi_fire` → NMI at the wrong PC → "Mario falls through floor"
   class. *Detect:* the parity harness's per-frame 29781-cycle lock is a canary;
   the Stage-1 `advance_to(t) == n×tick` exhaustive equivalence test asserts
   `(scanline, scanline_cycle, cycles, frame, v, flags, nmi_occurred)` identical.
   *Mitigate:* prefer bounded inc-loops over hand-derived `v` formulas until proven
   (as `step_whole_visible_scanline` already does, `ppu.rs:2329-2335`).

7. **Display-convention quirks in Mesen calibration.** We label pre-render 261
   (Mesen -1) and our dot labels are +1 vs Mesen at the same machine moment
   (naming only, `memory/…dawn…`). The sub-cycle offset calibration (§3.4b) must
   compare **machine moments**, not labels — anchor on the CV tape's cycle-exact
   instruction trace (34k+ instructions match Mesen post-boot-fix), not on
   scanline/dot string equality. Boot constant CYC=7/dot 25 is the reference frame.

8. **Secondary.** OAM-DMA island must stay per-cycle (§1.10, the DMA drives
   `$2004` writes at exact cycles); MMC2/MMC4 CHR-latch (Punch-Out) excluded from
   fetch-elision via the mapper gate; the `$2002` read-on-set race must match
   current `tick` ordering (the 146 tapes pin it, not "real hardware"); a stale
   `cached_ppu_batchable` hint (`nes.rs:72-82`) only ever costs a batch, never
   fidelity (`advance_to` re-queries the live mapper).

---

## Appendix — receipts

- `runs/event_ppu_design/mmio_access_receipt.json` — SMB in-game, 5 frames:
  454 PPU-reg accesses/frame; `$2007:W` dominated; **~3.6 `$2002` reads/frame**;
  ~1 OAM DMA/frame; **93% (2113/2270) of accesses pinned to `abs_dot%3==2`**
  (end-of-cycle) — the empirical ground for §0.3 and the ±3-dot race.
- `runs/event_ppu_design/mmio_probe.py` — regenerator (single `NESEnvironment`,
  `NES_TRACE_BUS`, no Pool; safe alongside the running `runs/cv_chain_hw2`).
- Prior-art building blocks already in-tree: `Ppu::advance` `ppu.rs:2157-2280`,
  `step_whole_visible_scanline` `ppu.rs:2305-2341`, `cpu_cycles_until_nmi_fire`
  `ppu.rs:644-663`, shadow oracle `tests/ppu_shadow_oracle.rs`, layout gate
  `Makefile:107-108` + `scripts/ppu_layout_check.sh`.
