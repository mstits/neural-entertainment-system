# Event-driven PPU catch-up + sub-cycle bus scheduling — one build, two payoffs

**Status: proposed (design note). Date: 2026-07-31.**

## Where the fidelity program landed

Five gated hardware-timing flags now ship (mmio read, mmio write, reset
alignment, DMC stall, NMI poll; plus env-level frame anchoring). Each was
receipted against Mesen by instruction-trace forensics on the CV power-on
tape. Together they hold RAM-level lockstep with Mesen from power-on to
~frame 4118 (~68 s) — at session start the first behavioral fork was frame
3992, and before the boot fix the machines disagreed about OAM-DMA length
at essentially every DMA.

The remaining divergence class is **not flag-sized**. Receipted residual:
CV re-enables NMI at every screen/block transition and polls sprite-0 every
frame; whenever one of these edges lands *within* a single CPU cycle's
three PPU dots, our engine and Mesen can resolve the race differently:

- Our interleave is `[3 PPU dots] → [1 CPU cycle]` (per-cycle path). A bus
  access inside the CPU cycle observes PPU state as of the *end* of that
  cycle's dots — up to ±3 dots from where the hardware access actually
  lands within the cycle.
- Mesen advances the PPU to the exact master-clock of each bus access
  (catch-up at access time), resolving these races at dot granularity.

Each mis-resolved race costs ±1 instruction of idle-loop phase; phases then
quantize later NMIs differently (±2–3 cycles/frame of jitter), and a few
hundred frames later a sprite-0 poll flips and gameplay forks. Measured
post-flags: phases converge and diverge repeatedly (lockstep at 3600 and
3790, split at 3700; ±2-cycle swings across 3848–3858). There is no single
remaining "seed" to fix — the race class itself must close.

## The claim

The fix for this **is** the top-of-queue perf item ("event-driven PPU
catch-up", est. >9% ceiling from the 07-14 campaign): restructure the PPU
to advance lazily to a target dot, invoked (a) at every CPU bus access
touching PPU-observable state, and (b) at end of each CPU batch. That
architecture gives:

1. **Perf**: the PPU stops ticking dot-by-dot through dead scanline time;
   batches collapse into closed-form advances (the existing `Ppu::advance`
   rung generalizes; skip-render paths get cheaper too).
2. **Fidelity**: bus accesses observe the PPU at the exact dot of the
   access (the Mesen catch-up semantics) — the ±3-dot race class closes,
   and with it, plausibly, the whole CV tape goes lockstep (the last
   measured gap).

## Sketch

- `Ppu::advance_to(master_dot)` — idempotent, event-driven: computes next
  observable event (vblank set/clear, sprite-0 hit dot [predictable from
  OAM+scroll once per scanline], NMI edge, A12 rise for MMC3, odd-frame
  skip) and fast-forwards between events; falls back to per-dot stepping
  inside render-critical windows.
- `SystemBus::read/write` for $2000-$3FFF (and $4014): call
  `ppu.advance_to(cpu_cycle*3 + intra_cycle_offset)` before servicing.
  The five hw flags remain the compatibility gates: legacy mode keeps the
  current interleave byte-for-byte (parity suites, banked receipts).
- CPU side already has the hooks: the late-read/late-write handlers mark
  the exact cycle of each MMIO access; the intra-cycle offset convention
  (which of the 3 dots the access lands on) gets calibrated against Mesen
  the same way boot constants were (CYC=7/dot-25).

## Cost/risk

- The PPU is the most PGO-sensitive hot path in the tree; every step of
  this must go through the clean-PGO bench discipline (receipts before and
  after, `cargo clean` between modes).
- The event list must be provably complete or the fallback per-dot window
  generous; the parity suite (554) + Mesen lockstep traces are the gates.
- Est. effort: a focused multi-day build, not a flag. Prereq for: TAS
  corpus Grade-A sync (v10 #2), Mesen-side bat soak, and the >9% perf
  ceiling.

## Receipts trail

- `memory/project_mesen_lockstep_dawn_2026-07-31.md` — full forensic log.
- Commits: 5c6dd8e (boot/DMC/anchor), 2540d11 (NMI poll, MMIO write).
- Traces: cycle-anchored fork scans, NMI-entry tables (d_cyc=0 × 11
  frames), frame-11 STA $2000 byte-identical sequence.
