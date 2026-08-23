# PPU odometer core spec (v24 synthesis) — ready to implement

Source: responses/20260823T052734Z_v24. This supersedes the raw wideNES
recipe on four points and confirms two of our judgments. Implementation
target: nes_core (Rust), one bundle with the v23 nametable peek.

## The design, beyond wideNES

1. **Modal Scanline Histogram, not write-interception.** Sample the
   PPU's internal loopy_v + fine_x at dot 256 of EVERY visible scanline
   (240 samples/frame); the frame's playfield scroll is the MODE (ties:
   largest contiguous scanline block). A HUD occupies a minority of
   scanlines by construction, so the mode is playfield — no IRQ
   tracking, no sprite-0 heuristics, no wideNES manual padding. Immune
   to top HUDs, bottom HUDs, multi-splits, per-scanline glitches.
2. **Absolute coordinate + wrap math.**
   X = NTbit10*256 + coarseX*8 + fine_x; Y = NTbit11*240 + coarseY*8 +
   fineY. Deltas: dX = ((Xc-Xp+256) mod 512) - 256; dY = ((Yc-Yp+240)
   mod 480) - 240. ATTRIBUTE-TABLE TRAP: if coarse Y enters 30-31 the
   vertical modulus becomes 512 for the duration. Accumulate into i64
   odometers.
3. **$2006/Zelda subsumed automatically.** Any camera mutation must
   propagate into loopy_v to render; sampling the rendering address
   makes the mutation path irrelevant. wideNES's Zelda sniffer is
   unnecessary here — confirmed, and stronger than we assumed.
4. **OAM fusion fixes camera-vs-player divergence.**
   Player_Global = Odometer + player_sprite_screen_pos, player found by
   the lowest-variance-relative-to-viewport-center heuristic over a
   rolling 60-frame window (sprite-0 as a prior). This keeps the signal
   ALIVE where the camera parks — fixed-camera arenas are exactly the
   Contra base wall, so the fused observable is a candidate second
   attack on that wall, hardware-only and purity-clean.
5. **Savestate coherence (non-negotiable).** global_odometer_x/y,
   scene_hash_id, prev_modal_x/y live INSIDE the core state struct and
   serialize with it. External accumulators desync on the first
   Go-Explore restore.
6. **Bipartite scene change.** Static-masked nametable hash (tiles
   volatile within a 30-frame window are masked; FNV-1a over the rest)
   AND a scroll discontinuity (|dX|>128 or |dY|>120) must fire
   TOGETHER: flashes change pixels but not the masked hash; shake moves
   scroll but under threshold. Converges with v23's fingerprint design.
7. **RL integration.** Cells: quantize Player_Global by 32. Reward:
   odometer delta with runtime dominant-axis detection (variance ratio
   k>=4 locks an axis; else Euclidean). ANTI-INFLATION GATE: odometer
   deltas count only while the agent has agency (input-divergence probe
   — the same controllability construct as the agency-recognition
   blueprint), so death resets and autoscroll cutscenes cannot mint
   progress.
8. **Certification = five automated checks** (extends
   progress_signal_gate): hold-forward monotonicity, hold-still exact
   flatness, HUD-split immunity, death/door discontinuity flagging, and
   savestate restore reverting the odometer exactly. Fail any -> the
   build is quarantined.

## Sequencing

Implementation is core work: PPU sampling hook (dot-256 per scanline),
state-struct fields + serialization, Rust-side modal/wrap math, pyo3
surface (read odometer, read fused player-global, read scene hash).
Runs AFTER the options verdict (no core rebuild under a live
pre-registered run), bundled with the nametable peek. First consumers:
re-run the progress-signal gate on Rygar / Kung Fu / Ninja Gaiden; then
v23 Experiment 1 (Zelda D1 / Metroid shaft) on the full stack.
