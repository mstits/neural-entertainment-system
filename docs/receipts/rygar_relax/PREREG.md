# Pre-registration — Rygar deep-relax arm (2026-08-27)

Written BEFORE any compute. Bar is fixed; a moved goalpost is a fabricated result.

## The measured pathology this targets
`deep` is a hard `key[0] == max_sect` filter. Once a cold run reaches sect=2 the
entire deep pool becomes cells in the static far room, where dx=0 in 83/83
visits. Both prior cold runs spent ~90% of budget pinned there.
`--transit-deep-relax` is the knob the design specified for exactly this and it
has never had a clean cold run: D1 ran at default 0; D2 (the only run passing
relax 2) was a contaminated resume whose max_sect never left 1, so the relax had
nothing to relax.

## Prior (fixed)
- max_x artifact-free: **4,608 px**
- novel areas from cold: **2**
- areas known: `(0,29)` deepest, adjudicated a dead end by a pre-registered
  falsifier (634,283 death-terminated steps, no fourth area)

## Arms
Two cold from-power-on runs, `--workers 2`, equal budget:
- A: `--transit-deep-relax 1`
- B: `--transit-deep-relax 2`

## Verdicts — decided now
- **PASS**: artifact-free max_x > 4,608 **OR** a third novel area reached.
- **FAIL**: neither, AND the relax demonstrably engaged (deep pool contains
  cells with `key[0] < max_sect` at some point in the run).
- **VOID**: relax did not engage (max_sect never exceeded the relax window, so
  the knob had nothing to relax) — the D2 failure mode, and not a result.

## Abort
If both arms are VOID, stop. The knob cannot be tested from a cold start and the
next lever is the area-key coarseness question, not more budget here.

---

## RESULT (2026-08-27, adjudicated against the bar above)

| arm | max_gx (raw) | max_sect | cells | steps | sps | solutions |
|---|---|---|---|---|---|---|
| relax 1 | 5,084 | 2 | 725 | 406,726 | 484 | 0 |
| relax 2 | 5,027 | 2 | 732 | 406,202 | 484 | 0 |

**VERDICT: FAIL** (not VOID — the relax engaged: `max_sect` reached 2 in both
arms, so the deep-pool filter had a live window to widen).

- No third novel area. `max_sect` 2 in both, matching the prior 2.
- Raw frontier 5,084 / 5,027 vs a prior raw HEAD ceiling of 5,893 and a banked
  tape of 6,242. Below both, so it does not beat 4,608 artifact-free either and
  the ratchet audit is unnecessary to reach that conclusion.

**SCOPE, stated so this is not over-read:** 14 minutes and ~406k steps per arm,
against prior campaigns spending far more. This FAILs the knob AT THIS BUDGET.
It does not refute the lever at scale — the same armed-compute caveat the Contra
lock campaign attached to its own negative.

**What this closes:** `--transit-deep-relax` was the one knob the transition
design specified and never got a clean cold run. It has now had two. The measured
pathology it targets (once `sect` reaches 2, the hard `key[0] == max_sect` filter
collapses the deep pool into the static far room where dx=0 in 83/83 visits) is
real, and widening the window does not by itself move the frontier.

**Next per the design's own ordering:** adjudicate the area-key coarseness
question before spending more compute here. `$0014/$001C` take 3 values across
38,650 cells plus 1.7M fresh steps, and the nametable-hash probe saturated its
64-hash cap in ~20 s, so it neither confirms nor refutes. A new hash under a seen
area key would be the instrument saying the key is too coarse to see the very
event this axis exists to catch.
