# The False-Death Fanout

*2026-08-26. `discover_observables.py`'s zero-start lives guard shipped
as commit `94e52b0`. This is what happened when it was pointed at all
eleven profiles the Mechanism Coverage Matrix said it would unblock.*

*Companion: `docs/research/MECHANISM_COVERAGE_MATRIX_2026-08-25.md`
§4-5, which enumerated the affected set and ranked the guard as "the
highest-value single fix available in this codebase — 11 games, hours."
That projection is the thing this document scores.*

---

## 0. The headline, stated plainly

| | |
|---|---|
| Profiles confirmed defective by first-hand root peek | **11 of 11** |
| Profiles that got a verified replacement lives byte | **5** |
| Profiles measurably unblocked (cells up in a matched smoke) | **3** |
| Profiles where no candidate survived verification | **6** |
| Profiles where the replacement was neutral or worse | **2** |

The tool fix works. The projection did not.

Eleven-for-eleven on the diagnosis: every profile the Matrix flagged
does read 0 at its own root state, re-confirmed here independently of
the sweep agents (`Pool` + `load_worker_state`, one NOOP step, direct
RAM read). And on every one of the eleven, the fixed ranker demoted the
old zero-start address exactly as designed — in most cases out of the
top five entirely, in the rest to a rank strictly below every
`starts_nonzero` candidate.

But **a demotion is not a replacement.** The guard can only reorder the
candidates the death-drive probe already found. On six of eleven games
it reordered a list that contained no real lives byte to promote, and
the honest outcome there is an unchanged config and a named instrument
gap. Three games got a real unblock. Two got a verified byte whose
smoke did not improve.

---

## 1. Confirmed defective (11 of 11)

Re-verified here, not inherited from the sweep. Each row is a fresh
`Pool`, that profile's own declared `start_state_path`, one NOOP step,
direct read of the old address.

| Profile | Old `solve.lives` | Root value | `player_state` fallback |
|---|---|---|---|
| 1942 | `$00C9` | 0 | none |
| Bad Dudes | `$034F` | 0 | none |
| Chip 'n Dale Rescue Rangers | `$0052` | 0 | none |
| DuckTales 2 | `$000B` | 0 | none |
| Galaga | `$0073` | 0 | none |
| Journey to Silius | `$00B9` | 0 | none |
| Kid Icarus | `$0130` | 0 | `$00A6` (wired) |
| Mega Man 3 | `$001A` | 0 | none |
| Ninja Gaiden | `$001F` | 0 | none |
| Paperboy | `$000B` | 0 | none |
| Shatterhand | `$0051` | 0 | none |

The Matrix's "10 of 11 have no fallback" holds. Kid Icarus is the
exception and it is a real exception, not a lucky one — see §4.

---

## 2. Verified replacements (5)

A replacement counts only if it was checked behaviourally, not accepted
on rank. Every row below was peeked at root by this audit and confirmed
`> 0`.

| Profile | New `solve.lives` | Root value | Independent check |
|---|---|---|---|
| 1942 | `$0524` | 1 | flat over 300 idle steps; 1→0 at t=65, refill 0→1 at t=471, 1→0 at t=479 (stock-with-continue) |
| Chip 'n Dale | `$0646` | 2 | flat over 400 idle steps; 2→0 in one frame, only in runs ending in a reload-sized churn event |
| Galaga | `$0464` | 2 | flat over 240 idle steps; one clean 2→0 at step 276 of a 900-step play rollout |
| Mega Man 3 | `$0029` | 1 | flat over a 150-step idle rollout; 5/5 driven death runs agree on a −1 to empty with refill |
| Shatterhand | `$0199` | 2 | pinned at 2 across both an 80-step idle hold and a 700-step constant-RIGHT hold |

**Shatterhand's caveat, kept:** no death transition was witnessed in the
verifier's own short deterministic probe (constant-RIGHT does not
provoke a death inside 700 steps). The `reaches_empty` /
`spends_its_stock` / `refilled_runs` evidence there is the tool's
randomized death drives, 5/5 agreeing; the hand check establishes only
"behaves like a stock, not a free-running counter."

---

## 3. Before/after, measured

All pairs are `go_explore_solve.py --workers 2 --minutes 1`, same root
state, the profile's lives byte as the only difference. Numbers read
from the archived `archive.stats.json`, not from prose.

| Profile | Cells before | Cells after | Δ | Verdict |
|---|---|---|---|---|
| **Shatterhand** | 16 | **232** | **14.5×** | Real unblock. Frontier 0 → 175 (still open), max_gx 53 → 1023. SOUND_GAME_STOPS → SOUND_ADVANCING. |
| **1942** | 101 | **121** | **+20%** | Real unblock. Frontier +48%, best_score 764 → 1265. |
| **Chip 'n Dale** | 76 | **81** | **+6.6%** | Real but small. max_sect 7 → 11, best_score +57%. Single 1-minute runs; treat +5 cells as directional. |
| Galaga | 1 | 1 | **0** | Null. Instrument fixed (roots seed `lives=0` → `lives=2`), search unmoved. |
| Mega Man 3 | 475 | **351** | **−26%** | Regression on the day's measurement. See below. |

### Shatterhand is the one that matches the projection

It is the only game in the set whose failure mode was the one the Matrix
described: a frozen frontier caused by the lives byte and nothing else.
Removing the false death moved it from a stalled 16-cell archive to a
232-cell archive with an open frontier, in the same minute of compute.
This is the Mega Man (USA) 10 → 16 result reproduced, larger.

### Mega Man 3 went down, and the honest answer is "confounded"

Cells 475 → 351, per-step cell rate 0.0212 → 0.0188 (≈11%), so it is not
purely a step-count artifact. But load average was ~24 during both
windows — the F0 eval ladder (`--eval-workers 8`) plus at least three
sibling sweep agents running their own solves — and sps differed 374 vs
311 (~17%) between two runs with identical seed, profile and worker
count. That is enough contention to produce this delta on its own.

What is *not* in doubt: `$001A` reads 0 and `$0029` reads 1, both
verified here. The edit is correct on the instrument's own terms. What
this smoke does **not** supply is confirming evidence of an unblock, and
it is not claimed. A longer, uncontended re-run is the outstanding item.

The narrower point the number does make: Mega Man 3's frontier was
already at 475 cells *with* the broken byte. Whatever the false death
was costing it, it was not the frozen-at-10-cells failure the projection
generalized from.

### Galaga: the instrument is fixed, the game is still unscorable

`roots.json` seeds corrected from `lives=0` to `lives=2` — the agent is
no longer born dead. Cells stayed at 1 because Galaga is
CAMERA_STATIC_AGENT_ACTIVE: the four-driver PPU-scroll-odometer gate
measured `distinct=1, min=max=0` under every input. The screen never
scrolls, so Go-Explore's position frontier has no spatial gradient to
exploit no matter what the death byte says. Two separable defects; this
sweep closed one of them.

---

## 4. No candidate found (6)

Configs left functionally unchanged. In each case the fixed ranker
behaved correctly and the candidate pool was still empty of a real
stock.

| Profile | Fixed tool's top pick | Why it was rejected |
|---|---|---|
| **Bad Dudes** | `$00CD` (root 2) | Combat/attack-animation counter. Toggles 2→0→2 five to eighteen times per 700-step run, first toggle at step 3 under an attack mash. Wiring it fires "death" on ordinary A/B presses — strictly worse than the status quo. |
| **DuckTales 2** | `$000B` (still) | The old address survived the guard: `lives_from_death_drives` samples `start` from the death-drive log's first entry, which is *after* one forced forward-hold step, and `$000B` flips 0→1 in that one step. All eight candidates traced share a single periodic RAM-clear/redraw signature clustered at `$07EF-$07F8`. |
| **Journey to Silius** | `$0135` (root 1) | Drops 1→0 within 4 steps of *any* rightward hold — fires on movement onset, not death, in a forced scroller. Wiring it collapsed the smoke 774 → 2 cells (reverted immediately). Ranks 2-9 are movement triggers, animation counters, or free-runners. |
| **Kid Icarus** | `$0129` (root 0) | Still reads 0 — fails the bar outright. Sits at its unchanged start value for 6-267 frames, then makes one 0→255 wrap. The four nonzero candidates are blink/sound/init-transient bytes. **This game was never actually degraded**: see below. |
| **Ninja Gaiden** | `$0386` (root 1) | Passed every static gate and failed the behavioural one. Cross-checked against the odometer, forward progress keeps climbing for 100+ steps *after* `$0386` empties (94 px → 426 px). Wiring it collapsed the smoke 1096 → 24 cells, frontier → 0. |
| **Paperboy** | `$00B2` (root 4) | Free-runs under pure NOOP on a ~117-step cadence: 4→3→2→1→0, refill to 4, repeat, four full cycles in 3000 idle steps. The tool's 240-step idle probe is too short to see it. Every other top-ranked candidate free-runs on the same shared clock. |

### Kid Icarus is the one profile that was never broken

It is on the list of eleven because `$0130` reads 0. But `$0130` was
chosen *deliberately* as a monotone no-decrement sentinel — it only ever
increases, on a real stage clear — precisely to dodge the underflow
trap, with real death governed independently by the wired
`player_state: $00A6, death_states: [0]`. `GenericGame.is_dead`'s
wrap-aware path (`(start - cur) % 256 ∈ 1..8`) structurally cannot fire
off a byte that never decrements. Its baseline smoke confirms it:
32 cells, frontier 18, zero stall windows, 81 improvements. Healthy.
It should be removed from the affected set, not re-flagged as
unaddressed.

### Ninja Gaiden: the origin case is still live

`progress_signal_gate.py`'s own module docstring cites `$001F` as the
canonical example of this trap, and `configs/ninja_gaiden.yaml` still
declares `lives: 0x001F`. The fixed ranker demotes it correctly (rank 15
of 22, below every `starts_nonzero` candidate; it is a flicker artifact
making 12-26 unpaired 0↔255 ticks per run). The replacement is simply
not in the pool. Its config carries a documentation-only diff recording
the audit; the byte is unchanged.

**One caveat this run does not resolve.** Ninja Gaiden's 1096-cell
"before" number may not be a clean baseline either. `$001F` is
uncorrelated with real death, and the profile has no `player_state`
fallback, so real in-game deaths may go entirely *undetected* under the
old byte too — some of that frontier could be post-death exploration.
Settling it needs a validated ground-truth death signal for this ROM,
which this campaign does not have.

---

## 5. Against the projection

The Matrix ranked this fix **"11 games, hours"** and called it the
highest-confidence item on its list. Scoring it honestly:

**"11 games" → 3 measurably unblocked, 5 configs corrected, 6 gaps
named.** The count was a count of *diagnosed* profiles, and the
diagnosis was perfect. It was read as a count of *fixable* ones. The
step the projection skipped is that the guard is a filter, not a
generator: it can reject a bad nomination but cannot manufacture a good
one out of a probe that never sampled a real lives byte.

**"Hours" was right.** The whole sweep, eleven profiles including the
behavioural verification that caught three false positives, fit in a
single session.

**"Do it as a tool fix, not eleven config edits" was right, and is the
most durable finding here.** The one guard demoted the vacuous
nomination on all eleven profiles without per-game work. Every hour
after that went into verification — which is exactly where it should
have gone, because verification is what caught `$0386`, `$00CD`,
`$0135` and `$00B2` before they shipped.

### The four "mis-marked UNSCORABLE" games

The framing of "11 games plus 4 more mis-marked UNSCORABLE" conflates
two different sets in the Matrix, and the correction matters:

- The **four UNSCORABLE** games — Mega Man 3, Ninja Gaiden, Paperboy,
  Shatterhand — are a **subset of the eleven**, not additional. They are
  the ones that FAIL `progress_signal_gate.py` on the lives byte alone
  despite excellent odometer signals.
- The **other four** in §4 of the Matrix — Mega Man (USA), Power Blade,
  Darkwing Duck, Ice Climber — are the already-defused-by-hand cases.
  They were never live and are untouched by this sweep.

Scoring the four UNSCORABLE:

| Game | Instrument finding cleared? | Status |
|---|---|---|
| Shatterhand | Yes — `$0199`, root 2 | Cleared *and* unblocked (14.5×) |
| Mega Man 3 | Yes — `$0029`, root 1 | Cleared; smoke did not confirm |
| Ninja Gaiden | **No** — still `$001F`, root 0 | Would still be marked UNSCORABLE |
| Paperboy | **No** — still `$000B`, root 0 | Would still be marked UNSCORABLE |

**Two of four are recoverable from the League denominator on this
evidence; two are not.** And "cleared the instrument finding" is not the
same as "passes the gate" — `progress_signal_gate.py` has not been
re-run on Shatterhand or Mega Man 3 here. That re-run is the concrete
next step, and it is cheap.

---

## 6. Three residual defects this sweep surfaced

The guard closed one loophole in `lives_from_death_drives`. Verification
found three more, each independently reproducible, none patched here
(scope was config re-nomination, not a second tool rewrite).

1. **`start` is sampled one step late.** It comes from the death-drive
   log's first entry, which already has one forced forward-input step
   applied — not the true idle root. Any byte driven by currently-held
   input can read nonzero there and slip past `starts_nonzero` while
   genuinely reading 0 at root. *Confirmed on DuckTales 2 (`$000B`
   0→1 in that one step) and Chip 'n Dale (`$0052` is a facing-direction
   flag: 1 holding right, 2 holding left, 0 otherwise).*

2. **`_regime_split` truncates the spend window at the first rise.** A
   byte that falls and recovers within a few frames shows exactly one
   clean down-tick inside the truncated window, trivially satisfying
   both `reaches_empty` and `spends_its_stock`. *Confirmed on Bad Dudes
   (`$00CD`, an attack-animation counter, ranked #1).*

3. **The 240-step idle probe is too short.** `moves_while_idle: false`
   is a false negative for any decorative counter with a period longer
   than the window. *Confirmed on Paperboy, where every top candidate
   free-runs on a ~117-step cycle invisible at 240 steps and obvious at
   3000.*

A fourth, softer one: a byte can pass every static shape gate and still
be death-uncorrelated. Only the odometer cross-check caught Ninja
Gaiden's `$0386`. **Cross-checking a lives candidate against the
progress signal — does forward motion continue after the byte empties? —
should be a standing gate, not a manual step.** It is the single cheapest
addition suggested by this sweep.

---

## 7. Receipts

Under `runs/false_death_fanout/<game>/`, one directory per profile:
the post-fix `discover_*.json` with the full re-ranked candidate table,
the before/after `smoke_*/archive.stats.json` and `roots.json` (the
`lives=` field in `roots.json` is the seeding proof), and per-game
verification logs. Large `archive.pkl` / `traces.pkl` are excluded, per
standing practice.

Independent re-verification for this document — the eleven root peeks in
§1 and the five in §2 — was run fresh against each profile's own
declared ROM and start state, not read from the sweep's own JSON.

`.venv/bin/pytest tests/test_profile_configs.py -q` → **273 passed,
21 skipped.**
