# Maze Mini-Dossier: SMB Castle-Maze Search Without Game Internals

**Date:** 2026-07-24. **Scope:** one tightly-bounded search problem.
**Context documents (optional):** the project's honesty policy bans game
disassembly, level maps/walkthroughs, and hand-authored action sequences.
Everything below is derived from the system's own emulator rollouts.

## The problem

Super Mario Bros castle mazes (4-4, 7-4, 8-4) silently loop the player back
when the "wrong" route is taken through invisible check zones. Our Go-Explore
solver (deterministic emulator, microsecond save/restore, first-return-then-
explore over a cell archive, ~2,500 rollout steps/s on 14 workers) solves
every non-maze SMB level from the entrance in minutes, but saturates on 4-4:
**five cell-representation variants all pin at max gx ≈ 2060–2102 with zero
solutions** (level exit is well beyond). We need the cell representation or
search modification that cracks silent-checkpoint looping mazes — using only
observations of our own rollouts.

## Observed loop mechanics (measured, not documented)

- Wrong-route traversal triggers a discontinuous **gx → 0 collapse** (screen
  section reload); observable 100% reliably. Loop trigger positions vary
  (observed at prev_gx ≈ 269, 325, 779, 989, 1502 across passes).
- From deep "wrong-route" states, **every action** loops back at the check
  zone — the route decision was made earlier in the pass and is held in
  internal game state invisible to coordinates.
- Differential RAM analysis over 32 passes (log all 2048 bytes, correlate
  change events with loop events / positions): loop events redraw the tile
  buffer ($05xx–$07xx region, consequence not cause); two candidate
  route-tracker bytes found ($0742: 4 changes in 32 passes, single position
  cluster gx≈768; $07F8: two clusters). Keying on their values did NOT
  unblock the search (variant 5).

## The five failed variants (each with its measured failure mode)

| # | Cell key addition | Result | Failure mode |
|---|---|---|---|
| 1 | none — (area, phase, y-band, gx-bucket) | pins gx 2064, 5,970 cells saturated | first-pass and looped states alias; frontier dies |
| 2 | + trajectory loop-count (gx-collapse events, cap 8) | pins gx 2102 | right/wrong-route states still alias at same coords; gx-domination *prefers* wrong-route spirals (they rack up gx) |
| 3 | variant 2 + low-loop selection bias (70% of deep draws to minimal-loop lineages) | pins gx 2102 | same aliasing; bias alone can't separate what the key can't see |
| 4 | + coarse RAM hash (40 bytes sampled 0x0300–0x0800) | gx 2068, **archive explodes to 585k one-visit cells, throughput 1700→289 sps** | timers/enemies churn every frame; archive becomes a trajectory log; search degenerates to a random walk |
| 5 | + route-signature (trajectory's y-band at each 512px gx boundary, reset on loop) and separately + discovered bytes ($0742,$07F8) | gx 2076–2082, 273k cells | either the signature doesn't capture the true check variable, or check zones/height bands are misaligned with our discretization |

## Constraints

- **No disassembly, no level maps, no walkthroughs, no hand-authored
  inputs.** Empirical analysis of our own rollouts' RAM/frames is allowed
  (the cell keys already read coordinate bytes).
- Machine: one M4 MacBook Pro; the solver runs ~1,000–2,500 steps/s at
  14 workers; runs of 30–90 minutes are cheap; multi-hour runs acceptable.
- The Go-Explore archive supports arbitrary cell keys (any function of the
  RAM snapshot plus per-lineage trajectory features we thread through
  restore), domination scoring, and frontier-biased selection.

## Questions (ranked)

1. **What cell representation is known to work for silently-looping mazes?**
   Montezuma/Pitfall Go-Explore used downsampled *frames* — but SMB's maze
   gives no visual route feedback (the screen is identical on every pass),
   so frame cells seem to alias exactly like coordinates. Is there published
   work on Go-Explore-style search in environments with hidden route state
   (POMDP mazes), and what did it key cells on?
2. **Is the right move a different search structure rather than a richer
   key?** E.g., treat each inter-loop segment as a macro-node and search the
   graph of (segment, route-choice-sequence) — a two-level search where the
   inner level replays a segment and the outer level enumerates route
   choices. The route choices per segment are discoverable by clustering
   trajectories that DON'T loop at each check zone.
3. **Empirical route-state extraction:** given black-box RAM snapshots and
   the ability to generate unlimited labeled rollouts (loop / no-loop at
   each check), what's the principled way to identify the minimal RAM state
   that predicts the loop decision (feature selection over 2048 bytes,
   thousands of labeled passes are cheap)? Our differential analysis was
   crude; a supervised probe (predict loop-at-next-check from RAM at
   position X) would isolate the true route bytes. Confirm this approach is
   sound and specify the cleanest protocol.
4. **Fallback:** if maze route-state truly requires byte-level
   identification, is there any principled objection to keying cells on a
   RAM byte identified by a *learned probe* over our own rollouts (vs.
   reading a disassembly)? We believe this stays on the legal side of the
   project's no-internals line — flag any published-precedent concerns.

## Deliverable requested

One concrete recipe (cell key or search structure + selection policy +
budget estimate for our measured throughput) with 2–3 measurable signposts,
and the falsification condition under which we should conclude the maze
needs the supervised route-byte probe (question 3) rather than a richer key.
