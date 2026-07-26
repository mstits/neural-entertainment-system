# Maze Dossier v3: 8-4 — Eight Attempts, ~110M Steps, and a RAM-Semantics Reckoning

**Date:** 2026-07-26. **Read first:** maze dossiers v1–v2 and your two prior
responses (the heuristic-inversion recipe solved 4-4 in 45 minutes and 7-4 in
59 minutes, first try — that part of your prescription is fully vindicated).
**This dossier is about World 8-4 only**, where the final wall stands, and
where our attempt to apply your event-driven adaptation exposed that our RAM
interpretations — including one from your response — do not match measured
behavior. We need semantic ground truth, observationally definable.

## 1. Where the game stands

31 of 32 levels solved and verified (4-4 and 7-4 fell to your
heuristic-inversion recipe: velocity-signed cells + inverted sampling in the
self-measured saturation window). 8-4 resists. Eight attempts (A–H),
~110M search steps, four mechanism generations. The measured shape:

- The first pipe gate (loop trigger at gx≈3849, warp back to ≈2828) was
  crossed by attempt C after the inversion-window floor was extended to the
  observed warp destination.
- The search then reliably reaches an underwater stretch (swim flag $001D=1)
  and saturates there: frontier pinned at in-water gx≈2467–2498 across every
  variant; driving further right triggers a warp to in-section gx 0.
- Segment solves rooted at validated in-water states (twin roots, 28M
  concentrated steps) died 30px from the warp point with frontier=1 — no
  local maneuver from those lineages exits the water.
- Full-level re-runs with richer keys (below) reproduce the same ceiling.

## 2. The mechanism generations and what each taught

| Gen | Key/mechanism addition | Result |
|---|---|---|
| E | loop-count + inversion (the 4-4 recipe) + low-loop archive gate | first gate crossed (C); water reached; water exit never |
| F | + "$0750 section-transit count" leading the key and the score | frontier metric climbed to 24 "sections"; water ceiling unchanged |
| G/H | + "$0750 value-sequence" (pipe-path identity, last 4) | same ceiling (sect 24, in-water gx ~2086–2498); archive ballooned (10GB; flush overhead fixed separately) |

## 3. The reckoning (today's decisive measurement)

Your 8-4 adaptation prescribes an event-driven probe: label pipe entries by
outcome, isolate the global destination-routing variable, key on it. We
implemented the labeling as: **transit = $0750 changed** (per your citation
of $0750/$074E as the sub-area indicators) vs **loop = gx collapse with
$0750 unchanged**. The collection returned **14,478 "transits" vs 29
"loops" in 2.5M steps — $0750 changes ~6 per 1,000 steps during ORDINARY
play.** It is not a stable sub-area pointer in our measurements; it appears
to update continuously as part of the column-streaming engine. Consequently:

- F/G/H's "section transits" were streaming ticks (distance-correlated,
  which is why the frontier metric seemed meaningful), not pipe events.
- The "pipe-path identity" key was a position echo, not a route.
- The probe's labels were meaningless, so no routing variable was isolated.
- Our one solid earlier datum — pre/post-first-gate states differ at $0750
  (2 vs 229) — is consistent with *any* position-dependent variable and
  proves nothing about routing.

We are done guessing RAM semantics. Every 8-4-specific interpretation we
took (from your response or our inference) must now be observationally
defined and measured before further use.

## 4. What is verified, observationally

- Warp-back events: discontinuous gx collapse (>100px backward) — reliable.
- The water stretch: $001D (swim/float flag) = 1 — behaviorally verified
  (probes swim). Water reachable from the entrance robustly.
- The water exit wall: at in-water gx≈2498, rightward/upward/downward
  approaches all produce a warp to in-section gx≈0. 28M+ steps of local
  variation (11-action full controller, y-band diversity, velocity-signed
  cells, inverted sampling) never produced a non-warping continuation.
- The clear detector (world/level bytes advancing) is independent of all of
  the above and warp-guarded — a true 8-4 clear cannot be missed or faked.

## 5. Questions (ranked)

1. **Reconcile $0750.** Our measurement says it churns with scroll/streaming.
   What is the OBSERVATIONALLY STABLE indicator of 8-4 sub-area/room
   identity available in RAM state — one that changes exactly at room
   transitions and nowhere else? If no single byte is stable, what derived
   predicate is? (We can measure any candidate's change-rate in minutes —
   give us candidates with expected behavior, we will verify before use.)
2. **Define "pipe entry" observationally.** For the event-driven probe to
   label correctly, we need a detector for the moment a pipe entry begins
   (player-state byte value? y-snap + input freeze? sprite behavior?), so
   transits and loop-warps are windowed to actual pipe events rather than
   inferred from side effects.
3. **The water exit.** Given the measured wall (every approach at in-water
   gx≈2498 warps to in-section 0), what are the plausible mechanics
   consistent with this signature, and what experiment separates them:
   (a) the exit is a pipe requiring an entry state our action pattern hasn't
   produced; (b) the exit checks route-history from before the water (our
   in-water lineages are uniformly poisoned); (c) the warp at 2498 is the
   MANDATORY section wrap and the exit lies elsewhere in the section
   (e.g., must be entered from a different y or an earlier x); (d) other.
4. **The routing variable, correctly.** Once (1) and (2) give sound labels,
   the Lasso+mutation protocol is ready to run within the hour. Anything
   else about 8-4's engine behavior we should measure while instrumented?

## 6. Assets

Solver with all mechanism generations (`scripts/go_explore_solve.py` —
velocity cells, saturation-gated inversion with warp-destination window
floors, transit/psig plumbing to be re-pointed at correct semantics, atomic
flushes, section telemetry); archives F/H (6.4M cells, intact, with traces);
validated in-water roots; the routing-probe harness (`/tmp/pipe_probe.py`,
relabel-ready); `runs/ge_chain_w8/entrances/entrance_after_8-3.state` (the
verified 8-4 entrance); 31 verified level solutions and all chain handoffs
power-on → 8-4.
