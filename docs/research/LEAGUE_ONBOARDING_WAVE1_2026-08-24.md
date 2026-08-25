# League Onboarding — Wave 1 Classification Table (2026-08-24)

Odometer-era onboarding pass over 12 games with existing start states. Doctrine held throughout:
purity line (observables only from the repo's own 3-probe discovery or hardware surfaces),
receipts for every claim, smokes at `--workers 3` / 6 min / seed 0.

## Classification table

| Game | Signal | Classification | Receipts | Next lever |
|---|---|---|---|---|
| Bubble Bobble (USA) | SOUND_ADVANCING | Fixed-screen arcade; progress on receipted RAM round counter `$0401` (odometer camera-static both axes, inapplicable). 2 replay-verified round clears in smoke; cells monotone 708→1360. | `runs/onboard_wave1/gate_bubble_bobble.json` (+`_left`), `runs/onboard_wave1/smoke_bubble_bobble/` | None needed for search; terminal open problem stays the (99,1) boss room skill wall (diagnosed 2026-08-09). |
| Castlevania (USA) | SOUND_ADVANCING | Gate PASS (axis x, 905 distinct, 0..3055 px). Smoke banked 3 replay-verified block-0→1 clears via level_key; gx-751 flat line = block-0 extent, not a frozen frontier. | `runs/onboard_wave1/gate_castlevania.json`, `runs/onboard_wave1/smoke_castlevania/` | Longer solve run down the block chain; odometer `{source: odometer, axis: x}` documented as certified fallback to the RAM pair. |
| Double Dragon (USA) | SOUND_ADVANCING | Fight-gated brawler: odometer legitimately flat (camera clamps at every enemy wave, both directions), but own-discovered RAM frontier (`0x005A \| 0x00B2<<8` per-room counter) advanced to room 15 with frontier still growing at cutoff (cells 385→793). | `runs/onboard_wave1/gate_double_dragon.json` (+`_left`), `runs/onboard_wave1/smoke_double_dragon/` | Verify mission counter `0x0030` increment (win predicate under-triggers safely today); a combat-blip-safe clear detector remains open (confluence rejected 2026-08-06). |
| DuckTales (USA) | SOUND_ADVANCING | Gate PASS (axis x, 104 distinct); progress deliberately kept on the decoded-money HUD odometer (game objective; `score_jump` clear threshold is in money units). Frontier advanced every window, stall 0. | `runs/onboard_wave1/gate_ducktales.json`, `runs/onboard_wave1/smoke_ducktales/` | If the money frontier ever stalls, swap to the now-certified PPU odometer fallback (`{source: odometer, axis: x}`) recorded in the profile. |
| Excitebike (Japan, USA) | SOUND_ADVANCING | Gate PASS under `--forward A` (throttle; 11,062 px, 1116 distinct) — the flat right-hold was a probe-driver artifact (left/right are mid-air lean controls). Smoke reached section 2 of 3, still expanding at cutoff. | `runs/onboard_wave1/gate_excitebike.json`, `runs/onboard_wave1/smoke_excitebike/` | Longer run: a finish is plausibly in budget, and the receipted finale hook (`0x000E==2`) banks a solution on first finish. |
| Ghosts'n Goblins (USA) | SOUND_ADVANCING | Gate PASS. Seed-0 3-window freeze at gx 2605 diagnosed as a TRAP-CELL (all 11 holds byte-identical; fatal hit latched in banked state, death at step 15). Seed 1 routed +722 px past it — wall is cell-local, not global. | `runs/onboard_wave1/gate_ghosts_n_goblins.json`, `smoke_ghosts_n_goblins/` (+`_seed1/`), `diag_deepest_ghosts_n_goblins.json` | Engine tune: retire-on-instant-death threshold fires only at `died_at_burst_step<=5`; raise it or probe-on-bank so late-latched death cells don't drain bursts. |
| Gradius (USA) | SOUND_ADVANCING | Gate PASS (940 distinct, 0..1714 px). Progress switched to the odometer (mirrors the legit RAM forced-scroll accumulator); frontier climbed 681→3121 across windows; deeper = surviving further into the difficulty ramp. | `runs/onboard_wave1/gate_gradius.json`, `runs/onboard_wave1/smoke_gradius/` | Stage byte discovery — `level_key` is empty, so the solver searches but cannot bank a stage clear. This is the known blocker for a real win detector. |
| Kid Icarus (USA, Europe) | SOUND_ADVANCING | Search progresses on the profile's own 16-bit altitude odometer (`0x0750 \| 0x04D1<<8`); 5-min gx-289 plateau broke on its own (early-search variance, not a wall). PPU odometer unusable as gate instrument: vertical climb needs repeated jump presses no hold-driver produces. | `runs/onboard_wave1/gate_kid_icarus.json` (+`_up`, `_A`; formal FAIL is a schema false positive on the documented lives sentinel), `runs/onboard_wave1/smoke_kid_icarus/` | Longer solve run to bank the `0x0130` 0→1 stage clear (already wired as level_key). Gate schema: recognize documented no-op lives sentinels. |
| Kirby's Adventure (USA) (Rev A) | NEEDS_SCENE_KEYS | Scene-cut wall: grounded UP fires a real door (gx 991→40, verified new content 0/7 nametable-hash overlap) but scene byte `0x004F` doesn't change, so post-door states are dominated and discarded. Generic `area: 0x803` tested and rejected — render blanking re-anchors the odometer so doors never bump the ordinal, plus 2 spurious bumps. | `runs/onboard_wave1/gate_kirby.json` (+`_left`), `smoke_kirby/`, `diag_kirby_deepcell.json`, `diag_kirby_door_probe.json`, `diag_kirby_scene_key.json`, `diag_kirby_cut_forensics.json` | Nametable-fingerprint scene key: core `peek_nametables` exists; the solver config surface is the missing piece. Search is capped at 2 rooms until then. |
| Mega Man 2 (USA) | SOUND_ADVANCING | Gate PASS; odometer resolved ~2.6× finer than the receipted RAM pair under the identical probe (808 px/91 distinct vs 312/47), so progress switched to odometer. Frontier advanced every window, zero stalls. | `runs/onboard_wave1/gate_megaman2.json` (+`_rampair` cross-check), `runs/onboard_wave1/smoke_megaman2/` | `level_key` deliberately empty (coverage baseline) — needs a stage/boss observable. First lever if a long run walls at a boss door or vertical transition: scene keys (`area: 0x803`). |
| Metroid (USA) | SOUND_ADVANCING | NEEDS_SCENE_KEYS → keys applied (`area: 0x803`), re-smoke SOUND. Deepest-cell diagnostic proved the frozen 1045 cell was a door-blip (scene counter 8→9, odometer 1049→1446), not a skill wall. Re-smoke: 2596 cells, 16 room transits. | `runs/onboard_wave1/gate_metroid.json` (+`_left`), `diag_metroid_deepcell.json`, `smoke_metroid_scene/` | Raise `--sect-cap` to 64+ for long runs (default 16 saturated in a minute); note `0x803` is a cut COUNTER not room identity (backtracking inflates max_area). No reachable clear byte — coverage baseline. |
| Mike Tyson's Punch-Out!! (Rev A) | CAMERA_STATIC_AGENT_ACTIVE | Fixed-screen fight-gate game: 28–46 px screen shake only, y-range 0, OAM churn shows agent animating. Scroll odometer is the wrong surface for this class; no smoke run (gate not SOUND, per pipeline rules). | `runs/onboard_wave1/gate_punchout.json` (+`_left`) | Fight-gate progress mechanism (opponent-defeat / bout-outcome from purity-clean surfaces). Gate script improvement: sub-threshold wobble should classify as static camera instead of SIGNAL UNUSABLE. Purity note: legacy `ram_mapping` cites internet maps — excluded from solve block. |

Score line: **10 of 12 SOUND_ADVANCING** (one of those, Metroid, via a scene-key fix inside the wave), 1 blocked on a representational mechanism (Kirby), 1 out of instrument class entirely (Punch-Out).

## Synthesis

### Mechanisms that earned their keep this wave

- **Scene keys (`area: 0x803`)** — the wave's clearest win *and* its clearest boundary. On Metroid they converted a frozen 272-cell smoke into a 2596-cell, 16-room-transit run, and the deepest-cell diagnostic proved the counter registers Metroid's doors. On Kirby the same key was tested and correctly *rejected*: render blanking at door loads re-anchors the odometer so the ordinal never bumps, and it double-fired during plain scrolling. One generic mechanism, one game fixed, one game with a precise refutation receipt — that is exactly what a generic lever should look like.
- **Deepest-cell diagnostic** — earned tenure. Three deployments, three distinct verdicts with receipts: GnG trap-cell (input-locked death latched in a banked state), Kirby representational freeze (door fires but archive can't see it), Metroid door-blip (progress real, banking absent). Without it, all three walls would have read as identical "frontier frozen" lines.
- **Debounce** — quietly correct on Bubble Bobble: the known 52→51→53 round blip is real ROM behavior and the debounce rode it without a false round-clear.
- **Sentinel/modular lives handling** — did its job (Excitebike's constant-anchor keeps the lives clause inert; Kid Icarus's monotone no-op sentinel is the documented right call) but exposed a gate-schema gap: the gate's "death byte reads 0" INSTRUMENT finding false-positives on documented sentinels. Small fix, wave-2 candidate.
- **The odometer itself** — 4 profiles switched to it as primary (MM2, Gradius, GnG, Metroid), 3 more carry it as a certified fallback (Castlevania, DuckTales, Excitebike). On MM2 it out-resolved the receipted RAM pair 2.6×. Its two documented non-fits (fixed-screen arcade, fight-gate) failed *legibly*, with both-direction receipts.
- **Probe-driver discipline** — Excitebike's flat right-hold vs sound throttle-hold is the canonical receipt that a flat gate must be attributed to driver, instrument, or game before classification.

### One-mechanism-away games

| Game | The one mechanism |
|---|---|
| Kirby | Nametable-fingerprint scene key — core `peek_nametables` already exists; only the solver config surface is missing. Highest-leverage single build of the backlog. |
| Punch-Out | Fight-gate progress (opponent-defeat/bout-outcome detection from purity-clean surfaces). Also unlocks the whole fixed-screen-fight class. |
| Gradius | Stage-byte discovery for `level_key` — search is sound, wins are invisible. |
| Mega Man 2 | Same shape as Gradius: stage/boss observable for `level_key`; coverage is already excellent. |
| Double Dragon | Combat-blip-safe clear detection (mission counter `0x0030` verification); progress axis already sound. |
| Ghosts'n Goblins | Engine tune, not config: retire late-latched death cells (`died_at_burst_step` > 5) or probe-on-bank. |

Cross-cutting gate-script fixes earned by this wave: (1) sub-threshold wobble (≤~50 px shake) should classify as static camera; (2) documented lives sentinels shouldn't trip the INSTRUMENT finding.

### Recommended wave-2 game list

**Precondition: none of these 12 have start states in `roms/` — every one needs a start-state mint first** (the cheap onboarding recipe: mint → clone config → gate → smoke). Selection favors families adjacent to already-onboarded games (observable shapes likely transfer) plus well-scrolling action games where the odometer should certify cleanly:

1. Mega Man 3 (USA) — MM2 adjacency; odometer + same observable shapes likely transfer
2. Castlevania III - Dracula's Curse (USA) — CV adjacency; block/level_key pattern should port
3. Ninja Gaiden II - The Dark Sword of Chaos (USA) — NG adjacency (NG already odometer-certified)
4. Super C (USA) — Contra adjacency; tests the vertical-axis odometer on alternating stage types
5. DuckTales 2 (USA) — DuckTales adjacency; money-odometer pattern may port directly
6. Blaster Master (USA) — scrolling run-and-gun + room transitions; good scene-key stress test
7. Bionic Commando (USA) — scrolling platformer with area select; strong odometer candidate
8. Adventure Island II (USA) — pure right-scroller, SMB-class; should be a fast SOUND certify
9. Chip 'n Dale Rescue Rangers (USA) — scrolling platformer, forgiving physics
10. Darkwing Duck (USA) — Capcom platformer, MM-engine relative
11. Power Blade (USA) — free-roaming scroller; exercises bidirectional odometer
12. Journey to Silius (USA) — linear scroller run-and-gun; clean odometer candidate

Alternates if any mint proves hostile (menus/passwords/mode selects): Shatterhand, Little Nemo - The Dream Master, Bucky O'Hare, Tiny Toon Adventures, Batman - The Video Game.

Also note: five games *with* start states already minted remain un-onboarded in this era and can slot in ahead of any minting work — Contra, Kung Fu, Ninja Gaiden, Rygar (all four already odometer-certified 2026-08-23), Lost Levels, and Tetris (B-type). They are wave-1.5 material, not wave-2.
