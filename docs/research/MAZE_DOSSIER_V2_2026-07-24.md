# Maze Dossier v2: Your Probe Protocol Was Executed — Here Is the Full Elimination Record

**Date:** 2026-07-24 (evening). **Read first:** `MAZE_DOSSIER_2026-07-24.md`
(v1) and your consultation response ("Resolving State-Space Aliasing in
Castle Mazes"). We implemented your Supervised Route-Byte Probe Protocol
faithfully, plus four follow-on experiments its results forced. **4-4 remains
unsolved**, but the day produced a chain of decisive eliminations and one
live, untested hypothesis whose measurement instrument we haven't built
correctly yet. We need your read on the reframe in §4.

## 1. Your protocol, executed — and what its checks caught

- **Checkpoint X=240 (your suggested pre-trigger):** class skew 3,789 pass /
  14 loop — the first gate is NOT at gx≈269; early loop observations were
  rare stragglers. Moved X to 740 (before the dominant observed trigger at
  ~779): skew inverted, 9 pass / 3,763 loop (random-play pass rate 0.24%).
- **Archive-boosted classes:** Y1 from replaying the search archive's
  never-looped ("blessed") lineages, Y0 fresh. Lasso reached 100% holdout —
  and **your causal mutation test correctly REJECTED all candidates**: the
  probe had learned the *collection channel* (timer-like bytes $00F1/$07BB),
  not route state. The methodology's built-in filter worked exactly as
  designed.
- **Same-channel redesign** (restore pre-gate archive cells, random
  continuations, label pass/loop): produced the day's key inversion —
  continuations from blessed cells pass the 779 gate at **95–99% from as
  early as gx 80**, vs 0.24% from the level root. Route state appeared to be
  fixed before gx 80.
- **Bisect-mutation of the root** (transfer the 345 bytes constant across 8
  blessed states into the root, binary-search the causal subset): transfer
  lifted pass rate **not at all** (0%→0% baseline on that measurement
  batch). Root inspection explained why: the chain-handoff root is a
  PRE-INITIALIZATION state ($000E=0, empty enemy slots) — the game re-inits
  those bytes during the level intro, wiping mutations. Root-corruption
  hypothesis then tested directly: we built a **clean entry** by replaying
  the 4-3 solution and letting the game itself carry the transition into
  settled 4-4 gameplay. **Clean entry and old root behave identically**
  (8% vs 8% pass-779 in a matched batch; same depth). Root hypothesis dead.

## 2. The deeper eliminations (beyond your protocol)

- **The blessed lineages' wall:** never-looped lineages reach gx ~2080 and
  then **loop (30/48), die (10/48), or stall (8/48)** — the ~2100 barrier
  applies to lineages that passed every earlier gate. No archived route
  variant passes the final section.
- **Trajectory-shape saturation:** fine-cell micro-search (gx/8, y/16 cells)
  of just the final section (gx 1400→2100) from two blessed roots: 218k
  cells each, saturation coverage, 0 solutions, pinned 2076/2096.
- **Action space:** the solver ran with rightward-only actions all campaign.
  Re-ran with the full controller (left, left+A, down, down+right): pinned
  at 2059. **Not a missing control primitive.**
- **Totals:** seven cell-key/search variants, two roots, two action spaces,
  ~8M search steps today. Max gx across everything: 2059–2102. Zero
  solutions.

## 3. The live hypothesis (untested — instrument failure, not refutation)

**Seam-advance:** the gx→0 wrap at the seam may fire on the CORRECT route
too, loading the *next section's content* — in which case our loop-detector
counts success as failure, gx-based metrics are blind to progress by
construction, and every variant above was measuring the wrong thing. The
observable discriminator should be level CONTENT at equal coordinates.
Our first instrument — hash of the tile buffer $0500–$06BF — turned out
noisy: 496 of 501 (loops, area, gx-bucket) positions carry 2–20 distinct
hashes (animated tiles / coin-block state churn inside the buffer), so it
separates nothing. The hypothesis stands unmeasured.

## 4. Questions

1. **Is the seam-advance model consistent with your understanding of the
   4-4 trigger mechanic** (your v1 answer described coordinate-boundary
   resets)? If yes: what is the clean content signature? Candidates we can
   build without internals: (a) tile-buffer hash restricted to
   structurally-stable bytes (mask out bytes that churn within a single
   lineage while stationary — measurable), (b) a short-horizon FUTURE probe
   (from each post-wrap state, roll K deterministic steps and hash the
   coordinate trace — different sections produce different motion traces),
   (c) frame-pixel hash of the non-animated screen region. Rank these or
   name a better one.
2. **If seam-advance is wrong**, what mechanic is consistent with ALL of:
   every route variant looping at ~2100 (including never-looped lineages,
   full controller, saturated trajectory shapes); pass/fail at the FIRST
   gate being ~fixed by gx 80; and a clean entry behaving identically to a
   pre-init root?
3. **The success detector:** our clear condition is the world/level byte
   advancing (warp-guarded). If 4-4's exit fires only after Bowser/axe, and
   the bridge room lies BEYOND several correct seam-wraps, is there any
   observable intermediate success signal we should also watch (e.g., a
   content signature never seen in wrong-route repeats)?
4. Given the totality above, give the single next experiment with a
   signpost and abandonment condition.

## 5. Assets

Everything from v1, plus: `runs/maze_roots/clean_4_4.state` (game-carried
entry), `runs/maze_roots/blessed_*.state` (never-looped mid-level roots),
`checkpoints/maze_micro_{0,2}` and `maze_full_ctrl` and `maze_content`
(saturated archives with traces), `/tmp/probe_*.log` experiment logs,
`scripts/route_probe.py` (your protocol, reusable), the solver with
pluggable trajectory-feature cell keys and full-controller profile
(`configs/smb_4_4_micro.yaml`).
