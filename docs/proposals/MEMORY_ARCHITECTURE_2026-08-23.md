# The memory architecture: spatial, topological, semantic — and where
# Apple Silicon actually helps

Direction set by the user 2026-08-23: to beat Zelda/Metroid/Battletoads-
class games the system needs MEMORY — a map of the world it can navigate
(spatial), and an understanding of what things do: key opens door,
weapon kills enemy, item heals (semantic). Sources now in hand: the
wideNES engineering + academic-reference reports (full implementation
math), our v23 plan, v24 (in flight) on odometer implementation.

## The three layers, all purity-legal

**Layer 1 — metric position (the odometer).** Integrate PPU scroll
per frame into global camera coordinates. The reports supply the exact
mechanics: modulo-delta wrap thresholds (±128 of 256 horizontal, ±120
of 240 vertical), HUD isolation by split-scanline (S_irq < 120 = top
bar: sample below; else sample above), $2001 bit-1/2 left-8px masking,
sprite-layer stripping for clean background sampling. OUR STRUCTURAL
ADVANTAGE over wideNES: it intercepted $2005 writes and therefore needed
"game-specific memory sniffers" for Zelda's $2006-based transitions —
we own the core, so we sample the PPU's internal v/t registers directly,
which $2006 writes also update. One mechanism covers both register
styles (v24 will confirm the sampling dot). CRITICAL DESIGN RULE: the
odometer accumulator (global x/y, prev scroll, scene id) lives INSIDE
the savestate blob — Go-Explore restores thousands of states per run
and an external accumulator desyncs on the first restore.

**Layer 2 — topology (the scene graph).** Rooms/scenes as nodes,
transitions as edges — G = (V, E) exactly as wideNES models it, fused
with v23's nametable fingerprinting: tile-ID hashing is palette-
invariant (fades and flashes don't fork rooms), scroll-discontinuity
plus phash-spike (Δ intensity sum over background) confirm transitions,
counterfactual rollback validates edges (>=80% reproducibility under
sticky). The reports' key gift: NON-EUCLIDEAN worlds (Lost Woods warp
loops, one-way ledges, pipe networks) are handled by making each scene
its own isolated sub-canvas node — local geometry stays Euclidean,
weirdness lives only in the edges. Drift over long sessions is a solved
problem: pose-graph loop closing (SLAM-style global batch adjustment)
when a mapped region is re-entered.

**Layer 3 — semantics (capability memory).** What items DO, discovered
from our own play, never authored: SIMCE's irreversible-bit tracking
finds candidate flags (key acquired), counterfactual splicing proves
their function (splice the bit into a blocked state; if the door edge
opens, the bit IS the key — 20 rollouts, random-bit false-positive
controls). Weapon-vs-enemy and item-effect associations are the same
pattern at the statistics level: learned co-occurrence between own
actions, OAM entity disappearance, and health/state deltas. The purity
line is preserved because every association is measured from the
system's own experience — the difference between learning "this opens
that" and being told.

## The Apple Silicon mapping — honest version

Three engines, three lanes, no contention:
- **CPU (P-cores)**: the emulator pool, as today (~2,500-3,300
  worker-steps/s). Odometer/scene-hash instrumentation is per-frame or
  per-scanline — noise on this budget.
- **GPU (MPS)**: PPO training, as today; plus batch training of any
  map-level models (tile autoencoders for room embeddings, MDMC-style
  spatial models — the reports cite Jain 2016 / Snodgrass 2018 as the
  academic line).
- **ANE (Core ML)**: the genuinely NEW opening. Prior ruling
  "ANE-for-collection rejected" stands — per-step policy inference is
  latency-bound and tiny. But the MAP pipeline is batch and
  asynchronous: room-fingerprint embeddings, VQ-VAE latent-cell
  encoding, hazard-model scoring of archived states, perceptual-hash
  variants — thousands of items, no step-loop deadline. That is the
  ANE's exact sweet spot, and it runs WITHOUT stealing CPU pool cycles
  or GPU trainer cycles. Caveat kept from the old ruling: tiny models
  may still win on CPU/AMX; every ANE placement gets benchmarked on a
  quiet machine before adoption, per the standing bench rule.

## Sequencing (unchanged critical path)

1. Options verdict (running now) — the sticky-wall experiment decides
   the immediate learning-track direction.
2. Core work bundle, ONE rebuild: expose scroll/v-register sampling +
   nametable peek + savestate-embedded odometer state. v24's report
   refines the sampling details before this lands.
3. Re-run the progress-signal gate on Rygar / Kung Fu / Ninja Gaiden
   with the odometer — three dead onboardings retested for pennies.
4. v23 Experiment 1 (Zelda D1 / Metroid shaft) on the fused Layer-1+2
   stack; SIMCE build for Layer 3.
5. Show overlay: the stitched map IS the League's live-stream artifact.

## What this is NOT

Not a fallback for an impasse — it is the v23 class-8/9 plan with its
math now filled in. And none of it touches the honest eval protocol:
maps and memory are training/search infrastructure; the cold sticky
greedy 100-episode standard stays the only headline number.
