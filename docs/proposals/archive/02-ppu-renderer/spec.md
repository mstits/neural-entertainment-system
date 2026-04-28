# Split 02 — PPU Renderer

## Context

Source proposal: [`../full_rust_refactor.md`](../full_rust_refactor.md)
Interview decisions: [`../deep_project_interview.md`](../deep_project_interview.md)
Manifest: [`../project-manifest.md`](../project-manifest.md)

The PPU renderer is the highest-risk split — sprite priority,
sprite-0 hit, scrolling, and palette handling are the classic NES
bugs that take days to track down. The non-cycle-accurate design
choice means per-scanline rendering, not per-pixel.

## Key decisions inherited from interview

- **Non-cycle-accurate, per-scanline rendering.** This is the deliberate
  tradeoff for speed over fidelity. Games requiring per-cycle PPU
  timing (Battletoads' raster split, some MMC5 demos) are
  out-of-scope per the proposal's non-goals.
- **256×240 RGB output** matching the existing transport layout.
- **Optional sub-resolution rendering** (e.g. 84×84 directly for
  headless workers) is a stretch goal in this split — design the
  renderer so a future PR can add it without restructuring.

## Deliverables

1. **Background renderer.** Nametable + attribute table + pattern
   table fetches per scanline, palette lookup, color emphasis bits.
2. **Sprite renderer.** Object Attribute Memory (OAM), 8×8 and 8×16
   modes, sprite priority vs background, sprite limit per scanline
   (eight; doc the silent drop).
3. **Sprite-0 hit detection.** Cycle-approximate (not cycle-exact),
   gated by the sprite-0 mask register. Triggers the PPU status
   bit at the right scanline so games using sprite-0 hit for
   scroll splits (zelda, super mario bros) work.
4. **PPU registers** ($2000-$2007) with correct read/write side
   effects, including the latched VRAM address writes ($2006) and
   the post-fetch VRAM increment.
5. **Palette + color generation.** Hardcoded NES classic palette
   (configurable via const for future user-supplied .pal files).
   SIMD palette → RGB conversion is a stretch goal here; finalize
   in split 05's perf pass.
6. **Frame buffer export:** `pub fn frame(&self) -> &[u8; 256*240*3]`
   returning RGB bytes ready for the existing transport layout.
7. **Skip-render fast path** — biggest single perf win: a `step_no_render`
   mode that advances PPU state machines (vblank flag, sprite-0 hit,
   NMI generation, scroll registers, mapper IRQ ticks via PPU A12)
   WITHOUT actually fetching pixels or composing the frame buffer.
   With `frame_skip=4` the trainer skips rendering on 75% of frames,
   so this path runs the vast majority of the time. Must produce
   identical CPU-visible state (registers, IRQs, NMI timing) to the
   full-render path; the only delta is the frame buffer is left stale.
   Test: a step+full-render, vs (4× step_no_render then 1× step+render),
   must produce byte-identical RAM after the same input sequence.

## Dependencies

- **Provided by 01-foundation-fork-and-cpu:**
  - `Bus` for VRAM/CHR access.
  - `Cpu` cycles drive PPU pacing (3 PPU cycles per CPU cycle).
- **Provided by 03-mappers** (only at integration time, not while
  developing this split): real CHR bank switching. While this split
  is in flight, use the placeholder `Mapper` trait — most test ROMs
  are NROM and don't bank-switch CHR.

## Provides to other splits

- `Ppu` type with `step(cycles: u32)` advancing PPU state and
  raising NMI on vblank.
- Frame buffer accessor for split 05's PyO3 zero-copy export.

## Risks for this split

- **Sprite priority bugs.** Foreground vs background sprite layering
  + transparent pixel handling is famously fiddly. Mitigation:
  golden-frame regression test against Zelda title screen and
  dungeon entry sprites; both have characteristic priority
  arrangements.
- **Sprite-0 hit timing.** Off-by-one scanline triggers Mario's HUD
  flicker bug. Mitigation: explicit test ROM (blargg's
  `sprite_hit_tests`) + visual verification on Mario.
- **Scrolling timing.** $2005 and $2006 register interaction is the
  most error-prone area of the PPU spec. Reference: nesdev wiki PPU
  scrolling page; implement strictly.
- **Spending too long chasing cycle accuracy.** Ship per-scanline
  first; only revisit if a target game (Zelda, Mario) is visibly
  broken.

## Acceptance criteria

1. blargg's `ppu_vbl_nmi`, `sprite_hit_tests`,
   `oam_read`, `oam_stress` test ROMs pass.
2. Golden-frame regression: Zelda boots to title screen; the rendered
   frame matches a checked-in golden PNG within ≤1% per-pixel
   difference (allows for palette tolerance).
3. Mario boots to title screen and the HUD coin counter doesn't
   flicker (sprite-0 hit working).
4. `cargo bench` (or similar) reports PPU step time; baseline for
   split 05's perf pass.
