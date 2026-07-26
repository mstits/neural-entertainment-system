# Driving prompt for the 8-4 consult (v3)

Attach: MAZE_DOSSIER_2026-07-24.md, MAZE_DOSSIER_V2_2026-07-24.md, both prior
maze consultation responses, MAZE_DOSSIER_V3_2026-07-26.md.

---

You are continuing a consultation on Go-Explore search in Super Mario Bros
castle mazes. Your track record in this loop is two-for-two: the
heuristic-inversion recipe from your second response solved World 4-4 in 45
minutes and World 7-4 in 59 minutes, unchanged. Read the attached Dossier v3
fully — it concerns the final level, 8-4, where eight attempts and ~110M
search steps have produced a precisely-characterized wall, and where our
attempt to implement your event-driven adaptation surfaced a factual
contradiction that must be resolved before anything else is built.

**The contradiction, stated plainly:** your 8-4 adaptation identified $0750
(area address) and $074E (background type) as the sub-area transition
indicators. Our measurement: **$0750 changes ~6 times per 1,000 steps during
ordinary play** — it behaves like part of the column-streaming engine, not a
stable room pointer. Three search generations built on it turned out to be
keying on position echoes. We will not use any further RAM interpretation
without measuring it first, so every claim in your answer about a RAM
location must come with its **expected observable behavior** (when it
changes, when it must not) phrased as a predicate we can verify in minutes
with unlimited instrumented rollouts.

**Answer the dossier's four questions in order, with these requirements:**

1. **Room identity:** name the candidate indicator(s) of 8-4 sub-area/room
   identity — single byte, byte combination, or derived predicate — each
   with its expected change-behavior (changes exactly at room transitions;
   stable during scroll, enemy activity, and death-respawn). If your sources
   conflict with our $0750 measurement, say which source is wrong and why.
   If no stable indicator exists in RAM, say so and give the derived
   alternative (e.g., a predicate over multiple bytes, or an event-window
   definition).
2. **Pipe-entry detection:** define the observable signature of the moment a
   pipe entry begins (player-state byte values, input-freeze behavior,
   y-motion signature — whatever is checkable), so warp events can be
   windowed to actual pipe entries and our Lasso/mutation probe harness
   (already built, relabel-ready) gets sound labels.
3. **The water exit:** the measured wall — every approach at in-water
   gx≈2498 warps to in-section gx≈0; 28M+ concentrated steps of
   full-controller variation from validated in-water states never produced a
   non-warping continuation. Rank the candidate mechanics in the dossier
   (unproduced entry state / route-history poisoning from before the water /
   mandatory wrap with the exit elsewhere in the section / other), and for
   the top two, give the single cheapest experiment that separates them,
   with its expected observation under each hypothesis.
4. **The corrected recipe:** assuming (1) and (2) resolve as you specify,
   state the exact cell-key and probe protocol for 8-4, with two signposts
   (step budgets at ~2,500–3,400 steps/s, 14 workers) and the abandonment
   condition.

Constraints unchanged: no disassembly, no maps, no hand-authored inputs;
empirical measurement of our own rollouts is unrestricted; solutions must
run on one M4 MacBook Pro. Do not re-prescribe: $0750-keyed cells (measured
invalid), transit-count or value-sequence keys built on it (attempts F–H,
all at the same ceiling), raw RAM-region hashes (archive explosion, dossier
v1), or segment-rooting from deep frontier cells (twice shown to inherit
poisoned state). If one of these is correct *with a modification*, name the
modification and the measured failure mode it defeats.

One meta-request: this level is the last obstacle to a complete
power-on→axe machine run of the game. If your sources describe 8-4's
mechanics in ways that would require knowledge we've banned (route maps,
walkthrough directions), translate them into *observables and experiments*
rather than directions — tell us how to measure, never where to go.
