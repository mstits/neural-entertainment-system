# room_fp calibration receipt — The Legend of Zelda (2026-08-24, T4)

Profile: `configs/zelda_roomfp.yaml` · config_sha `2d698930`
Falsifier: `tests/test_rg0_roomgraph.py` (RG-0 — all five gate assertions PASS)
Tool: `scripts/room_fp_calibrate.py` (capture / mint / mask / replay)
ROM: `roms/Legend of Zelda, The (USA) (Rev A).nes` · root state `roms/zelda_start_ctrl.state.bin`
(re-verified live 2026-08-24, `runs/onboard_wave2/mint_legend_of_zelda.json`)

Purity: every capture reads hardware surfaces only (2 KB physical nametable VRAM,
PPU scroll odometer, scene ordinal, rendered-line vote). RAM bytes appear below
solely to validate probe semantics against already-banked observables ($0670 HP
ladder, $0070 Link-x — `docs/receipts/games/` onboarding receipts); none feed the
mask, the hash, or the classifier.

## Mask (auto volatility, §4)

Captures (both frame_skip 1, from the root state):

```
capture --script "noop*300"                                   -> tests/fixtures/roomgraph/zelda_idle_fs1.npz
capture --script "right*40,left*40,up*25,down*25,right*40,left*40" -> tests/fixtures/roomgraph/zelda_walk_fs1.npz
mask zelda_idle_fs1.npz zelda_walk_fs1.npz --game zelda
```

Result: **exactly one volatile byte — NT 214**, the HUD heart tile (byte-diff
verified: it is the only byte that changes on every HP transition, including
both octorok hits of the death stream). `mask: [[214, 215]]`.

Idle-stability numbers (the T4 done-when, locked by
`test_masks_reproduce_probe_idle_stability` + RG-0.3):

| capture | rendered frames | distinct hashes pre-mask | post-mask |
|---|---|---|---|
| zelda_idle_fs1 | 300 | 2 | **1** |
| zelda_walk_fs1 | 210 | 3 | **1** |

Known residual: byte 214 is the only heart tile this one-heart root exercises;
a full 16-heart HUD may span neighbouring bytes (212–215). Recalibrate with a
higher-HP capture before trusting the mask on late-game states.

## Measured transition signatures (fs4 fixtures)

| event | fixture | Δodo | Δscene | churn shape | classifier |
|---|---|---|---|---|---|
| east screen exit | `zelda_east_exit_fs4.npz` | (+256, 0) | +1 (pan-end snap) | draw/scroll interleave, **8-step stable window mid-pan** | pan-E |
| north screen exit | measured in mint stream | (0, −181) | +1 | row churn every step | pan-N |
| west screen exit | `runs/onboard_wave2/gate_legend_of_zelda_left.json` | −256 | — | 18 distinct odometer values | pan-W |
| death (2 octorok hits) | `zelda_death_fs4.npz` | (0, 0) | **+2** | 2-step attribute-table flash rewrite, then 49 steps stable | **warp** |
| death aftermath | `zelda_death_full_fs4.npz` | (0, 0) | 0 | spark rewrite; 3 blank frames; continue menu | 2 fades |

Fixture generation:

```
mint    --frame-skip 4 --script "right*6,up*6,...(x7)...,right*2"   # 86-step weave
        -> tests/fixtures/roomgraph/zelda_north_screen.state.bin    # settled north screen, hp 125
capture --state zelda_north_screen.state.bin --frame-skip 4 \
        --script "right*4,up*6,right*6,up*6,right*6,up*6,right*6,up*6,right*6,noop*40"
        -> zelda_east_exit_fs4.npz
capture --state roms/zelda_start_ctrl.state.bin --frame-skip 4 --script "right*72"  -> zelda_death_fs4.npz
capture --state roms/zelda_start_ctrl.state.bin --frame-skip 4 --script "right*120" -> zelda_death_full_fs4.npz
```

(The design session's `ck_zelda.state` / `zelda_start.state.bin` checkpoints
were both measured input-dead — Link never moves under held input — so the
east-exit root was re-minted from the verified controllable state. Hold-right
from the root is NOT an east exit: Link dies to octoroks first; the design
draft's probe stream conflated that death with the pan, which is why every
constant below was re-measured.)

## Constants — deviations from the design draft, and why

1. **settle = 14 at fs4 (`ceil(56/frame_skip)`), draft said 3.** The nametable
   snapshot is scroll-invariant: during a horizontal pan Zelda pre-draws
   columns, then scrolls over them with zero VRAM writes for 8 straight steps
   (32 frames). At settle 3 the east-exit fixture mints a hybrid half-drawn
   room and a truncated +132 px pan (3 rooms / 2 edges — locked by
   `test_settle_three_would_mint_a_hybrid_mid_pan_room`); at 14 it is one
   pan-E of +256 exactly. 56 frames also outlasts Metroid's 12-step (48-frame)
   mid-run stable window (see metroid.md) with a 2-step margin.
   Cost: room adoption after a restore needs 14 stable samples (~1 s at fs4).
2. **Churn-onset baseline = previous rendered sample** (not the diverging
   sample). The death flash bumps the scene ordinal in the very frames of its
   first attribute rewrite, so a current-sample onset reads Δscene +1-of-2 and
   the death classifies fade. Implemented in `replay_room_stream` AND in the
   live hot loop (`Solver._room_step`, `c["fp_base"]` — locked by
   `test_onset_baseline_is_the_pre_churn_sample_so_a_straddled_warp_classifies`
   and `test_a_blank_breaks_the_onset_baseline` in tests/test_room_fp.py).
3. **Pan Δscene is +1, not 0**: every measured pan ends with one scene snap
   (scroll discontinuity when the game re-homes $2005). The draft's pan rule
   (Δscene ≤ 1) already tolerates it; recorded here so nobody "tightens" it.
4. **Δodo comes from the integrated odometer** and does integrate real pans
   (+256/−256/−181 measured); the odometer stays flat through the death flash.
   The draft's "modal 16→272→16" note described the same death correctly, but
   its "Zelda pans churn 64 straight frames" hid the mid-pan stable window.

## RG-0 verdicts (Zelda assertions)

- **RG-0.1 east exit ⇒ pan-E, exactly one new node: PASS** (2 nodes, 1 edge
  `0→1 pan E +256`, 0 warps).
- **RG-0.2 death ⇒ warp, zero edges: PASS** (Δscene +2, odo flat, 0 edges,
  warp_count 1) on the death-event window. Full-stream residual (locked by
  `test_zelda_death_aftermath_is_fades_never_warp_edges`): the game-over
  screens arrive as 2 *fade* edges out of the warp-adopted flash room — never
  warp-minted, but routable in principle. In live runs the `lives: 0x0670`
  proxy kills the lineage at the first hit, so the aftermath is unreachable;
  for observable-less games this residual stands and RG-1a's audit wording
  ("zero warp-minted edges") is the enforceable form.
- **RG-0.3 300-frame idle ⇒ exactly 1 hash post-mask: PASS** (and 2 pre-mask —
  the mask is load-bearing).
