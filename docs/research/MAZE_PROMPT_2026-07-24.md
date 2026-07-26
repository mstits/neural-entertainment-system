# Driving prompt for the maze consult (attach MAZE_DOSSIER_2026-07-24.md)

You are consulting on a narrow, well-instrumented search problem in a NES
emulation project. Read the attached maze dossier fully before answering — it
contains measured data on five failed approaches, and repeating any of them
unmodified wastes the consult.

**The problem in one sentence:** A Go-Explore-style solver (deterministic
emulator, microsecond save/restore, ~2,500 rollout steps/s, arbitrary cell
keys over RAM snapshots plus per-lineage trajectory features) solves every
non-maze Super Mario Bros level from the entrance in minutes, but saturates
on castle maze 4-4 — which silently loops wrong-route passes back via hidden
internal state — with five different cell representations all pinning at max
gx ≈ 2060–2102 and zero solutions.

**Hard constraints:** No game disassembly, no level maps or walkthroughs, no
hand-authored inputs. Empirical analysis of the system's own rollouts (RAM
traces, frames, unlimited labeled replays from save states) is fully
allowed. One M4 MacBook Pro; 30–90 minute search runs are cheap.

**Answer the dossier's four ranked questions directly, then deliver:**

1. **ONE primary recipe** — the cell representation *or* search-structure
   change most likely to solve silently-looping mazes on this stack. If your
   answer is the supervised route-byte probe (dossier Q3), specify the exact
   protocol: how to generate labeled passes (loop / no-loop at each check
   zone), the probe model and feature-selection method over the 2048-byte
   RAM, the validation test that confirms the identified byte(s) truly carry
   route state, and how the byte then enters the cell key. If your answer is
   the two-level segment-graph search (dossier Q2), specify node/edge
   definitions, how route choices per segment are enumerated from observed
   trajectories, and the inner/outer search budgets.
2. **Two or three measurable signposts** (with wall-clock or step budgets at
   ~2,500 steps/s) at which we abandon the recipe if unmet.
3. **The precedent check:** is keying search state on RAM bytes identified
   by a learned probe over the system's own rollouts methodologically
   accepted in published work (system identification / black-box state
   discovery), or does any published norm treat it as equivalent to reading
   a disassembly? Cite what exists.
4. **The generalization note:** whatever you prescribe must also apply to
   mazes 7-4 and 8-4 (8-4 additionally has non-linear pipe routing). Flag
   anything in your recipe that is 4-4-specific and how it adapts.

Do not propose: raw RAM-region hashing (measured archive explosion, dossier
variant 4), coordinate-only or loop-count-only keys (variants 1–3), or
downsampled-frame cells without addressing the dossier's objection that the
maze gives zero visual route feedback. If you believe one of these is
actually correct *with a specific modification*, name the modification and
why it defeats the measured failure mode.
