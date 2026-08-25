# room_fp calibration receipt — Metroid (2026-08-24, T4)

Profile: `configs/metroid_roomfp.yaml` · config_sha `e376f983`
Falsifier: `tests/test_rg0_roomgraph.py` (RG-0 — all five gate assertions PASS)
Tool: `scripts/room_fp_calibrate.py`
ROM: `roms/Metroid (USA).nes` · start state `roms/Metroid (USA)_start.state.bin`
Door-approach state: `tests/fixtures/roomgraph/metroid_onplat.state.bin` — minted
in the 2026-08-24 design-session probes (Samus on the first corridor's door
platform, odometer 508 / scene 2 riding its v4 envelope); adopted here as the
banked probe state the doors fixture starts from.

Purity: hardware surfaces only (nametable VRAM, odometer, scene, rendered
lines). RAM appears only in the death-observable probe below, which is the
standard experiment-discovered-observable pipeline (differential analysis over
our own rollouts — the same class as every onboarded `lives:` byte). The
`quarantined_external_knowledge` block in configs/metroid.yaml stays untouched.

## Mask (auto volatility, §4)

```
capture --frame-skip 1 --script "noop*300"                             -> metroid_idle_fs1.npz
capture --frame-skip 1 --script "right*24,noop*24,a*12,noop*24,right*12,noop*24" -> metroid_walk_fs1.npz
mask metroid_idle_fs1.npz metroid_walk_fs1.npz --game metroid
```

Result: **zero volatile bytes** → `mask: []`. Idle 300 frames = 1 hash,
in-room walk 120 frames = 1 hash, both unmasked (locked by
`test_masks_reproduce_probe_idle_stability`).

Calibration lesson (kept as a warning): a first walk script
(`right*40,left*40,...`) let the camera scroll and flagged 83 redrawn column
bytes as "volatile" — masking those would blind room identity. Calibration
walks must stay camera-still; the committed script does.

Known residual: the EN energy-digit HUD tiles are background and DO rewrite on
damage — invisible to damage-free calibration, so they are unmasked. A damage
event inside a room can fork its fingerprint (bounded by max_rooms + intern
dedupe; the `lives: 0x0107` proxy terminates damaged lineages anyway). If RG-2
telemetry shows damage-forked rooms, recalibrate with a damage capture.

## Measured transition signatures (fs4)

| event | Δodo | Δscene | churn shape | classifier |
|---|---|---|---|---|
| door 1 (corridor → item room) | +254 | +1 | NT flips EVERY step of the 32-step door scroll | pan-E |
| door 2 (item room → next room) | +254 | +1 | same | pan-E |
| in-room run (camera scrolling) | +6/step | seam/clamp bumps | 1–2-step flips; **one 12-step stable window** over repeating terrain (odo 256..328) | no settle at 14 |
| in-room run, camera still | 0 | 0 | NT static (Samus/enemies are OAM) | nothing |
| motion reversal | 0 while camera holds | — | stable ≥ 50 steps until the camera re-engages | settles a camera-view node |

Fixture generation:

```
capture --state tests/fixtures/roomgraph/metroid_onplat.state.bin --frame-skip 4 \
        --script "noop*20,b%2*4,right*36,noop*30,right+a/4*52,b%2*10,right*46,noop*50"
        -> metroid_doors_fs4.npz            # both doors, hp steady (no damage)
capture --state "roms/Metroid (USA)_start.state.bin" --frame-skip 4 \
        --script "noop*16,right*100,left*100,right*100,left*100,noop*24"
        -> metroid_scene_noise_fs4.npz      # two identical laps inside room 1
```

Door route facts (they justify the profile's `room_advance.buttons`): both
doors opened only from stand-and-shoot (`B` pulses) followed by a rightward
walk — run-and-gun `right+b%2` from the start state never opens door 1 (odo
clamps at 508 indefinitely, 300+ steps measured); the item room is crossed
with run-jump hops (`right+a/4`), plain right walks into the bubble wall.

## Scene-ordinal noise vs fingerprint identity

Two-lap fixture, all inside rooms already visited: the scene counter reaches
**8** (bumps at the 256-px seam and the 508-px clamp, both directions, both
laps) — D0's scene identity would mint a node per bump, unbounded. The
fingerprint mints **3** nodes total (spawn view, clamp view, −170 px reversal
view), and lap 2 mints **zero** new nodes — both reversal settles re-intern
lap 1's ordinals on byte-identical VRAM (camera reproducibility within 1 px).

**Deviation from the draft's RG-0.5 wording**: "zero extra nodes" is
literally unsatisfiable — racking 4 scene bumps inside room 1 requires motion
reversals, and a reversal parks the camera long past any usable settle, so
stable camera views become nodes. This also falsifies §7.3's premise ("hash
never settles while moving") for Metroid's repeating terrain (the 12-step
window) and camera-lag rests. The enforced falsifier form (RG-0.5): scene ≥ 8
inside room 1, zero warps, lap-2 settles all dedupe to known ordinals, node
count strictly below the scene-ordinal count. Camera-view granularity in
seamless-scroll rooms is hereby a *named property* of the identity layer, not
a surprise: bounded, deduplicating, aliasing-audit territory (§7.1/§7.2).

## Death-observable probe (30-min budget; the probe SUCCEEDED)

Method: three independent scripted death runs (weave-into-enemies variants,
two from the start room, one from the door-platform room) + four live
controls (noop 400/1500, damage jiggle, the full doors script, the two-lap
script), full 2 KB RAM recorded per step; candidate = byte with a novel
sustained post-death value in every death run, never seen in any control.

Result — 12 surviving candidates, of which the cleanest:

| byte | death value | fires (steps before render-off) | false fires in controls |
|---|---|---|---|
| **$0020** | **4** | ~10 before | none |
| $0614 | 31 | ~10 before | none |
| $0673 | 6 | ~10 before | none |
| $001E | 4 | at render-off | none |
| $0610 | 140 | ~10 before | **fires during door scrolls — struck** |

Nominated observable: **`ram[0x0020] == 4` = death sequence active**
(experiment-assigned semantics: enters at Samus's explosion, holds through
game-over; location-independent across the two rooms tested). Deliberately
**NOT wired** into any profile: the warp veto is the design's game-agnostic
death defense, RG-2 is report-only, and `lives: 0x0107` already terminates
damaged lineages. Wiring is a T5/T6 decision; validation across other areas /
power-up states is still owed before it gates anything.

Death-sequence surface signature, for the record: energy tens ($0107)
3→2→1→0, ~120 quiet steps while the fine digits drain, then $0020=4, 10 steps
of explosion, 1–3 frames rendered-lines 0, game-over screens. No scene bump —
a Metroid death is invisible to the scene core and near-invisible to the
classifier at fs4 (the blank is shorter than one step's 4 frames), which is
exactly why the RG-2 corpse-frontier caveat stands.

## RG-0 verdicts (Metroid assertions)

- **RG-0.4 door1/door2 ⇒ exactly two pan edges: PASS** (3 nodes; edges
  `0→1 pan E +254`, `1→2 pan E +254`; 0 warps).
- **RG-0.5 scene noise ⇒ zero extra nodes: PASS in the enforced form above**
  (FAIL as literally drafted — measured impossibility, documented; lap-2 = 0
  new nodes while scene climbs to 8; 3 fingerprint nodes < 9 scene ordinals).
