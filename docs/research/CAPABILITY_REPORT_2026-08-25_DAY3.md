# Capability report — 2026-08-25, Day 3

Scope: this is a further addendum on top of
`docs/research/CAPABILITY_REPORT_2026-08-25.md` (itself an addendum to
the 2026-08-11 → 2026-08-24 report), not a rewrite of either — their
numbers stand unchanged except where this document says otherwise.
This covers exactly what landed across Wednesday Push Day 2/3
(commits `9a7a4ca` through `982fe8b`): the v27 verdict and v28 launch,
hardening wave 2, and League onboarding wave 3. Same ledger tags —
LEARNED, EXHIBITION, FORGE — same rule: nothing claimed past what has
a receipt. Full entries for everything below are in `CLAIMS.md`; this
is the narrative form. No spin: a FAIL and a retraction are reported
here with the same weight as a PASS, because they are capabilities of
the *process*, not failures of it — the process's job is to find out
which hypotheses are true, and it just did that twice.

---

## 1. The sticky-wall research line [LEARNED] — CLOSED on curriculum shape, OPEN on capacity

### What changed

The prior addendum left v27 "in progress, no verdict." It has one now,
and it closes a question that has been open since the recovery
distillation FAILs one day earlier.

**The question, precisely stated:** the consolidated 48k-parameter 1-1
artifact is a measured isolated optimum — every post-hoc gradient
touch (naive CE distillation, KL-anchored cloning, on-policy recovery
PPO) made it worse. Is that because post-hoc fine-tuning is the wrong
delivery mechanism for the mined recovery states — and a fresh run
with those states in the curriculum from iteration 0 would behave
differently — or because 48k parameters is not enough capacity to
represent both the base policy and the recovery behavior no matter how
the training is shaped?

**v27 tested the first half of that split: curriculum shape.** 4 seeds
× 250 iterations, vanilla PPO on the SMB-tiles-pos encoder, the same
785-rung merged recovery ladder present from iteration 0, ReDo
dormant-neuron recycling mandatory-on per the DR amendment (ruling out
primacy-bias/dormant-collapse as a confound). Full honest-protocol
scoring, 16 gate evals: seed rates 0.04 / 0.29 / 0.53 / 0.17,
**best-of-4 = 0.530** against a PASS bar of ≥0.80 and a FAIL bar of
≤0.767. Not close on either bound — the best individual seed doesn't
even reach the untouched control's own 0.767.

**Verdict, in the registration's own words: from-the-start inclusion
adds nothing at 48k parameters — the parameter-budget hypothesis takes
the floor.** Combined with the prior post-hoc FAILs, both delivery
shapes have now failed at the same parameter budget:

| Delivery shape | Result | Rate | Receipt |
|---|---|---|---|
| Post-hoc (distillation onto consolidated artifact) | FAIL | best 0.70 (Variant A, gate 0.80) | `docs/proposals/RECOVERY_DISTILL_1_1_2026-08-24.md` |
| From-scratch (v27, recovery-in-curriculum) | FAIL | best-of-4 0.530 (gate 0.80) | `docs/proposals/V27_FRESH_RECOVERY_2026-08-24.md` |

**The sticky-wall research line is CLOSED on the curriculum-shape
hypothesis.** It is not closed on capacity — that is a live, distinct
hypothesis, and it is the one v28 is spending compute on right now.

### Two things learned along the way, banked as their own findings

**Peak instability is not a post-hoc artifact — it's intrinsic to the
recipe.** Every one of v27's 4 seeds degraded from its own in-run peak
to its own final checkpoint: 0.02 vs 0.04, 0.02 vs 0.29, 0.00 vs 0.53,
0.01 vs 0.17. This is the same collapse pattern measured two days
earlier when continued PPO destroyed a *post-hoc-consolidated* 1-2
peak (31/100 → 8/100, −74%) — but v27's checkpoints were never
post-hoc fine-tuned; they degraded from within a single from-scratch
run. That moves the standing explanation from "an artifact of
resuming a consolidated checkpoint" to "a property of this
architecture/training-recipe class." Preserve-on-peak, already
standard practice, turns out to be load-bearing rather than a
convenience: without it, v27's reported number is ~0.01, not 0.53.

**Training telemetry is unreliable at a magnitude this project hasn't
seen before.** The in-training `entrance_trailing_rate`
(0.87/0.93/1.00/0.97 across the four seeds) overestimated the honest
rate by 2–25×, while still correctly rank-ordering the seeds (seed 2
best, seed 0 worst). That split — useless as an absolute number,
still useful as a relative one — is exactly why this project's honest
protocol exists as a separate, mandatory step rather than a
convenience check, and this is the starkest confirmation of that rule
banked to date.

### What's running right now

**v28 tests the capacity hypothesis directly, and it is live on this
machine as this report is being written.** Single-variable change from
every v27 seed config: `tile_hidden_dim` 64 → 96 (`tile_trunk_dim`
unchanged), 48,135 → 72,039 parameters (+1.50×). Same ladder, same
ReDo treatment, same seeds, same iteration budget, same 0.80/0.767
gate — deliberately the identical bar v27 was measured against, so a
PASS or FAIL answers "does capacity move the v27 number" without
also asking "what's a fair bar for a bigger net."

Before spending any compute, four preflight checks ran and passed:
the config diff touches only the intended two lines (V1); the live
parameter count printed by the trainer matches the arithmetic exactly,
72,039 (V2); the ReDo dormant-neuron detector was re-calibrated at the
new width rather than assuming the old tau transfers — the recycle
boundary shifted from ~2 units to ~6 units at tau 0.30 when the layer
widened, and the soft-VOID trigger was updated to match rather than
guessed proportionally (V3); and the mechanism was confirmed armed,
not silently inert, in the actual seed-0 training log (V4).

As of this writing, only seed 0 has started — iteration 8 of 250,
~2,400–2,550 env-steps/s, ~162–170× realtime. Seeds 1–3 are queued
behind it in a sequential driver. **No honest-protocol number exists
for v28 yet, at any seed, and none should be inferred from training
telemetry** — the same rule the secondary finding above just
re-confirmed at 2–25× the previously-seen gap. This machine is
dedicated to the run; nothing else heavy should contend with it until
it finishes or reaches a checkpoint worth pausing at.

---

## 2. Engine hardening [FORGE, process] — a second P0-class bug found and fixed

### What changed

A second, independent audit pass covered the surfaces the first pass
hadn't reached: `scripts/engine_driver.py` (the scheduler this project
depends on to run unattended for weeks), the pyo3 bindings broadly,
`checkpoint_manager.py`, `checkpointing.py`, and `go_explore.py`. It
found a P0 as serious as the one hardening wave 1 fixed, of the exact
same shape: a piece of retry/recovery logic that had been silently
dead the entire time, discovered from evidence already sitting in the
system's own state — not from a walkthrough, not from a human hunch.

**The P0:** `engine_driver.py`'s `_reap_one` marked every single-shot
action (`honest_eval`, `config_*`, `mint_*`, `select_*` — anything
without a recurrence marker) as permanently `completed` the moment it
terminated, regardless of *how* it terminated — a clean success, a
verified failure, or an ambiguous crash were all treated identically.
Because the scheduler's `offer()` step permanently excludes anything
already marked `completed`, this made the entire retry-then-abandon
cap (`MAX_ATTEMPTS_PER_ACTION`) dead code for every action of this
class: a single transient fault — an OOM, a disk hiccup, a
momentarily-locked ROM file — would silently and permanently freeze
that level at whatever stage it was on, with no error surfaced and no
diagnostic trail. Live evidence that this had already happened was
sitting in `runs/engine/state.json` before the audit even started
looking. This is precisely the failure class the "walk away for three
weeks" operating mode cannot absorb: an engine that silently stops
working on one lane while reporting nothing wrong is worse than one
that crashes loudly.

**The fix:** `completed` is now written only after a *verified*
success. A crash or a verified failure leaves the action retry-eligible
up to the attempt cap; only then does the (previously unreachable)
abandon branch in `tick()` do its job and write a real terminal status
with a diagnostic trail attached.

**Six further fixes landed in the same pass**, each a real correctness
defect rather than a hardening nicety: a timeout-killed heavy job
never arming the post-kill settle timer, meaning a SIGKILLed
multi-hour campaign could be immediately followed by a benchmark on a
still-hot machine; three places that silently dropped a skip reason
instead of journaling it; a Rust-side worker-state loader that
panicked uncatchably on a malformed savestate blob and never cleared a
worker's dead flag even after a later successful restore, permanently
stranding that worker; a two-file checkpoint write (`best.pt` +
`best.json`) that wasn't transactional, so a kill between the two
writes could let a worse checkpoint silently overwrite the true best;
and a checkpoint temp-file path that wasn't unique per writer, letting
two racing processes tear each other's writes.

### The honest bound

29 new regression tests target these paths directly; the full suites
pass clean (`make test-fast`: 3,942 passed; `cargo test --lib`: 634
passed). That proves each fixed path now behaves correctly under
test. It does **not** yet prove the system has survived a real
multi-week unattended run with an actual transient fault in it — no
live crash-then-successful-retry event has been observed in production
since the fix landed, because none has occurred yet. **The system is
measurably safer for unattended multi-week operation; it is not yet
proven safe by having run unattended for multiple weeks.** Two
independent audit passes have now each found and closed one P0 of this
exact shape (a recovery mechanism that looked wired but was dead) —
that repetition is itself informative about where this class of bug
hides, and argues for a third pass rather than treating two as
sufficient.

---

## 3. League roster growth [EXHIBITION/process] — 22 games classified across two onboarding waves, one generic mechanism fix

### What changed

Two onboarding waves have now run against the odometer-era pipeline:
wave 1 (`docs/research/LEAGUE_ONBOARDING_WAVE1_2026-08-24.md`, 12
games with existing start states) and wave 3
(`docs/research/LEAGUE_ONBOARDING_WAVE3_2026-08-25.md`, 10 further
games) — 22 games classified in total, each with a receipted signal
and an honest next lever rather than a bare pass/fail label. (Wave 3's
own receipts reference reused "wave 2" gate JSONs for a few games;
those were inputs carried forward, not a third classification
document.)

**Wave 3's rollup:** 5/10 SOUND_ADVANCING (1942, Blaster Master,
Bionic Commando, Batman — preliminary, Paperboy), 2/10
CAMERA_STATIC_AGENT_ACTIVE (Ice Climber, Galaga — both genuinely
fixed-camera genres, correctly distinguished from a skill wall), 2/10
SOUND_GAME_STOPS with a diagnosed, receipted cause (Arkanoid: a
genuine one-life ball-tracking wall; Chip 'n Dale: a camera clamp plus
an unconfirmed room-counter fix), 1/10 BLOCKED on missing tooling
rather than any property of the game itself (Tetris needs a HUD digit
decoder this pipeline doesn't have yet, not a game-specific hack —
building one from outside knowledge would break the purity line).

### The headline mechanism fix

1942's solver smoke froze at 7 cells. The deepest-cell diagnostic
measured, directly and without appeal to any outside knowledge of the
game, that every held action drove the odometer's y-reading strictly
negative from a verified-live start state — 1942 is a
vertically-scrolling shooter whose scroll register counts *down*
during forward flight, and the existing odometer code clamped the
camera integral to ≥0 on the baked-in assumption (true of every
horizontal right-scroller onboarded before it) that forward always
increases the reading.

The fix generalizes rather than patches: `progress.axis` now accepts
an optional leading sign (`x`, `-x`, `y`, `-y`) that flips the raw
reading before the clamp, implemented so every existing profile that
doesn't specify a sign is untouched. Same wall-clock smoke budget:
7-cell freeze → 106 cells, frontier open, SOUND_ADVANCING. Verified
against the full suite — `tests/test_room_fp.py` plus the
go_explore_solve suites, 609 passed — so the fix is confirmed not to
have disturbed any of the 21 other onboarded profiles.

**Honest bound:** 1942 itself remains unsolved — this closed a search
freeze, not the game — and the fix is validated on exactly one title.
Galaga was flagged in the same wave as a plausible carrier of the same
latent defect class, not yet tested. Two more generic mechanisms
proved their worth again this wave without being new work: the
3-probe observable-discovery tool prevented a `KeyError` crash on at
least six profiles, and the deepest-cell diagnostic distinguished
three flat-frontier causes (a scripted non-interactive window, an
automatic scene transition, a real hazard past a camera cap) that are
indistinguishable from the raw progress log alone.

---

## Unchanged from the prior windows

Everything in `CAPABILITY_REPORT_2026-08-24.md` and
`CAPABILITY_REPORT_2026-08-25.md` not superseded above stands as
written: the odometer and scene-detection instruments, the death-
semantics fixes, the room-graph engine (RG-0 green, RG-1 still
registered and not yet run), the honest scoreboard (1-1 0.767 / 1-2
38% / 1-3 21% / 1-4 51%), and the retraction register. This document
adds three rows to the ledger — a closed hypothesis with an open
successor running live, a second hardening pass, and 10 more games —
not a reconciliation of the rest.
