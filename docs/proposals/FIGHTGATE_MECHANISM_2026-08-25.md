# Fight-gate progress mechanism — design + pre-registration

Status: DESIGN ONLY. Nothing in this document has been implemented in
`scripts/discover_observables.py` or `scripts/go_explore_solve.py` yet. The
three RAM reads under "Pilot evidence" below are brief, single-worker,
few-second manual probes run to calibrate the pre-registered prediction in
§5 — they are not the validation run itself, and no solver smoke was run.

## 1. The problem this closes

Every progress signal this project has shipped so far — the discovered
RAM byte pair (`find_progress`), the PPU scroll odometer
(`progress: {source: odometer}`), the decoded HUD tile field
(`progress: {tiles: ...}`) — assumes the game has a **spatial or
monotone-scalar frontier** to climb: a world-X position, a camera pan, a
bank balance. A class of games has none of that: the camera never scrolls
and the "world" never gets bigger, but the game still gates real
progression behind combat. The League onboarding pipeline has now hit
this class directly and named it in its own receipt. From
`configs/punchout.yaml`'s `solve:` block (2026-08-24 odometer-gate pass,
`runs/onboard_wave1/gate_punchout.json` / `gate_punchout_left.json`):

> forward=right, 1200 steps: axis=x range 0..28, 16 distinct, oam_churn
> 373/1199 … forward=left, 1200 steps: axis=x range 0..46 … y range = 0 in
> BOTH runs, so axis: y is also dead. The 28-46 px x-wobble is screen
> shake, not scrolling. Verdict: **CAMERA_STATIC_AGENT_ACTIVE** — Punch-Out
> is a fixed-screen fight-gate game … this game needs a fight-gate
> progress mechanism (opponent-defeat / bout-outcome detection) built
> from purity-clean surfaces before go_explore_solve can drive it.

That is the literal request this document answers, in the same words the
onboarding pass used. Three other configs in the roster show the same
family of problem from different angles, and all three are load-bearing
precedent for the design below:

- **Bubble Bobble** (`configs/bubble_bobble.yaml`) — already ships
  `progress: {lo: 0x0401}`, a round/level counter, because "no world-X
  frontier exists" in this game at all. This is the ONE place in the
  roster where a round-gated counter is already the primary progress
  axis, and it works — proof the *consumption* side of this idea
  (a non-spatial byte driving the ordinary lo/hi progress path) needs no
  new Solver machinery, only a byte.
- **Kung Fu** (`configs/kungfu.yaml`) — `progress: {lo: 0x0094}` is a
  player POSE byte (stand/duck/airborne), not a floor counter; no
  `level_key`; the generic `clear: {mode: confluence}` detector was
  tried and rejected because Kung Fu is "exactly the combat-heavy shape
  (frequent knockdowns) that misfired" when tested on a neighboring
  game. Kung Fu's real floor-gate signal has never been found.
- **Double Dragon** (`configs/double_dragon.yaml`) — the confluence
  detector's `coord` signal fired on a **combat RAM blip**: progress
  spiked 72 → 846 → 88 in 5 steps during a hit/knockback exchange, read
  as a level-load, and shipped a false clear. This is the single
  best-documented failure mode a fight-gate progress signal has to
  survive, and it already has a receipt
  (`runs/detector_gate_20260810/`, verdict `state_artifact`) plus a
  fix primitive already shipped for it: `progress_median: K` on the
  confluence detector (median-filters exactly this kind of impulse).

Punch-Out is the sharpest case because it has *zero* spatial signal at
all (both odometer axes read flat/noise) — every other roster game with
a combat problem still has some room/floor counter or coordinate to fall
back on. It is the correct validation target because success there means
the mechanism works with no spatial crutch to lean on.

## 2. Design constraints (unchanged project doctrine)

- **Purity line.** Every candidate address is adjudicated from THIS
  core's own scripted rollouts under THIS project's own inputs — no
  RAM maps, no disassembly, no game-specific address literals seeded
  into the discovery code. (`configs/punchout.yaml`'s existing
  `ram_mapping:` block, sourced from Data Crystal / TASVideos, is
  explicitly flagged in-file as **not usable** as a solver observable
  for exactly this reason — see the `PURITY NOTE` in its `solve:`
  section. This document treats that block as background only: it
  informs the falsifiable prediction in §5, and the discovery code
  described in §3 must never read it, import it, or special-case
  Punch-Out's ROM hash.)
- **Generic action patterns only.** No per-game move lists, no
  telegraph-reading, no frame-perfect counters. The only primitives
  available are the same ones `Discoverer` already uses elsewhere:
  hold a direction, hold an attack button, hold both, idle. "Approach"
  and "retreat" are defined relative to whatever `--forward` already
  means for the game (the same `fwd`/`rev` resolution `_resolve_dirs`
  already does) — for a fixed-ring game with no forward axis at all,
  the closest generic analog is the LEFT/RIGHT dodge pair the action
  space already exposes for every side-view combat game in this
  roster, not a Punch-Out-specific "dodge" concept.
- **Reuse before invention.** Three primitives already in the codebase
  do most of the work this mechanism needs and are reused, not
  rebuilt: the mass-RAM-reset detector (`_first_reset` /
  `reset_threshold`, already used to find deaths and level reloads),
  the decrementing-stock consensus test (`lives_from_death_drives`,
  already separates a life counter from debris without ever locating
  the death event), and the anti-correlated-pair tally finder
  (`clear_detect.score_tally_windows`, already discovers a
  timer→score conversion cadence from raw RAM history with no assumed
  addresses).
- **VOID is not FAIL.** An under-powered probe that never lands a hit
  is an instrument finding, not a game-difficulty verdict — the same
  split `progress_signal_gate.py` already enforces between
  "instrument findings" (block the profile) and "behaviour findings"
  (report the wall).

## 3. Mechanism, part A — discovery (extends `discover_observables.py`)

### 3.1 Two new generic drives on `Discoverer`

```
def attack_mash(self, reps: int = 5, n: int = ATTACK_N) -> list[dict]:
    """Randomized attack-heavy drive, modeled on death_drives: never
    reloads mid-run, seeds differently per rep, hands back the whole
    log. Action distribution over {A, B, A|B, fwd, rev, NOOP} —
    the SAME six-symbol vocabulary every game in this roster already
    defines, weighted toward A/B (0.60 combined) rather than a fixed
    rhythm. A fixed short cycle (e.g. alternating A,B,A,B...) is
    EXPLICITLY rejected as the default: it is what configs/punchout.yaml
    already calls out as insufficient ("the generic masher does NOT
    reach live gameplay on this dump"), and Pilot Evidence #1/#2 below
    reproduces exactly that failure independently. Randomized sampling
    over a long-enough budget is required for a fight-gate probe the
    same way ADVANCE_N-scale budgets (not CLEAN_N-scale) are required
    for spatial probes.
    """

def approach_retreat(self, reps: int = 5, n: int = APPROACH_N) -> list[dict]:
    """Alternating fwd/rev bursts (lengths drawn from a small
    distribution, e.g. 3-10 steps) with NO attack input at all —
    the pure-defense control condition. Its only job is to give the
    self/foe discriminator (3.3) a probe where the player is NOT
    landing hits, so foe-HP is provably distinguishable from an
    always-decrementing timer AND from a byte that merely tracks
    "the player is doing anything."
    """
```

Both are cached the same way `clean_forward` / `advance` / `death_drives`
already are, and both reuse `_reload` / `_step` / `_first_reset` /
`reset_threshold` verbatim — a "bout ends" event (new opponent's data
loads, HUD resets, a large fraction of RAM rewrites) is the same shape of
mass-RAM-rewrite event `_first_reset` already exists to find for a death
or a level reload. No new reset-detection code is needed; only the
threshold's calibration window changes per probe.

### 3.2 `find_fight_health(rom, state, *, disc=None) -> dict`

Ranks candidate "foe HP" bytes. Method, directly parallel to
`lives_from_death_drives` but inverted (a foe's stock empties from
*our* offense, not from an event we can't locate either):

1. Run `attack_mash` (N reps) and `approach_retreat` (N reps).
2. For every RAM address, compute the wrap-aware delta series
   (`_wrap_deltas`, already shared code) over both probe sets.
3. **Gate FH1 — decrements under offense.** The byte must show a
   net decrease across the attack-mash reps, by consistent
   per-event magnitude (same consensus test as
   `lives_from_death_drives`: `DEATH_MIN_AGREE`-style agreement
   across reps with different seeds).
4. **Gate FH2 — flat or non-decreasing under pure defense.** The same
   byte must NOT show the same decrement pattern across
   `approach_retreat` reps (which contain no attack input). This is
   the self/foe discriminator and is load-bearing: see §3.3, it is
   exactly the gate a naive port of `lives_from_death_drives` lacks.
5. **Gate FH3 — refill-aware regime split.** Reuse
   `lives_from_death_drives`'s existing "a run that outlives the
   stock gets it refilled" regime logic (judge the counter only up to
   its first rise, treat what follows as a new regime) rather than
   assuming strictly monotone decay to zero. Pilot Evidence #3 below
   shows this firing on Punch-Out's own Mac-HP byte inside a single
   600-step probe, which is exactly the shape a foe-HP byte's
   round-to-round refill will also take.
6. **Corroboration, not a gate.** Run `clear_detect.score_tally_windows`
   over the attack-mash RAM history and check whether the top FH1/FH2
   survivor appears as one half of a nominated anti-correlated pair
   (a "hit landed → tally moves" cadence). Agreement raises confidence
   in the report; disagreement does not disqualify the candidate,
   because `score_tally_windows` was built for a *periodic* tally and
   a foe-HP byte is not required to be periodic.

Output shape mirrors `find_hp_lives`: ranked candidates each carrying
which gates passed, plus a `self_hp_conflict` field naming the
`find_hp_lives` candidate it was checked against.

### 3.3 The self/foe discriminator (the one truly new idea here)

`find_hp_lives` never had to solve this problem because a life counter
is unambiguous — it decrements on death and nothing else looks like it.
A fight-gate probe introduces a *second* decrementing-stock byte in the
same RAM image (the player's own health) that decrements from the exact
same generic input (attacking without defending draws return fire), and
a naive port of the existing gate cannot tell them apart. Pilot Evidence
#1 demonstrates the failure directly: 400 steps of alternating A/B
mashing drove `mac_hp` 96 → 0 while `opp_hp` never left 96 — an
attack-only probe with no defensive component makes the PLAYER's stock
look exactly like what the gate is searching for, and the true foe-HP
byte looks inert.

The fix is Gate FH2 above: a real foe-HP byte must decrement under an
offense-heavy probe and must NOT decrement under a defense-only probe
(no attack input at all), while the player's own HP decrements under
BOTH (it takes damage whether or not the player is attacking, as long as
it isn't dodging effectively) or under neither if the defense probe
happens to dodge well. Any byte satisfying FH1 without FH2 is reported
as a `self_hp_candidate`, not a `foe_hp_candidate` — this is a
classification the discovery report makes explicit rather than a
silent rejection, so a game where the two really are inseparable (shared
health pool, PvP-style) shows up as a named finding instead of a report
that lies by omission.

### 3.4 `find_round_gate(rom, state, *, disc=None) -> dict`

Parallel to `find_room_counter`, but keyed to the mass-RAM-reset
detector instead of a screen/door transition:

1. Run `attack_mash` and `approach_retreat` long enough to plausibly
   end at least one bout by chance (budget set from the FH1/FH2 probe
   above — if no bout ever ends, this returns `insufficient_probe`,
   an INSTRUMENT finding, not a `no_round_signal` BEHAVIOUR finding;
   see §2's VOID/FAIL split).
2. At every mass-reset boundary `_first_reset` finds, diff RAM
   immediately before vs. after.
3. A candidate must be: stable within a bout (flat between resets,
   same test `find_room_counter` already applies within a room),
   monotone non-decreasing across resets, and — the new check this
   axis needs that a room counter didn't — **NOT** reset back to its
   start value at a bout boundary the way a life-counter-triggered
   level reload does (a lost bout on a multi-life game reloads the
   SAME opponent; a won bout advances to the NEXT one). This
   distinguishes "round/opponent index" from "attempt counter."

### 3.5 CLI surface

`discover_observables.py --fight-gate` runs 3.2 + 3.4 in addition to the
existing four passes and folds `fight_health` / `round_gate` blocks into
`--emit-solve`'s YAML the same way `find_y` / `find_hp_lives` already do.
No existing flag's behavior changes; a profile that never asks for
`--fight-gate` gets byte-identical output to today.

## 4. Mechanism, part B — consumption (extends `go_explore_solve.py`)

### 4.1 A new progress source, symmetric to the odometer

```
progress:
  source: fight_gate
  foe_hp: 0x0398        # from find_fight_health
  foe_hp_start: 0x60    # captured at load, not hardcoded
  round: 0x0006         # from find_round_gate, optional
```

`GenericGame.__init__` gets a third `elif` beside the existing
`source: odometer` branch (§`self.odometer_axis` in the current code).
Exactly like the odometer, the computed value is NOT read from `ram[lo]`
directly — it is an accumulated integral the Solver writes into the
pseudo-RAM extension every step, so every existing consumer (cells, the
glitch filter, macros, `progress_cap`) reads it through the same
unmodified `lo`/`hi` path the odometer already uses. Concretely, in
`Solver._xram` (which already owns pseudo-addresses `0x800..0x802` for
the odometer integral, `0x804/0x805` for the room-fp ordinal, and
`0x806` for the odometer's orthogonal axis):

```
# new pseudo-address pair, next free slot after the odometer/room-fp block
FIGHT_LO, FIGHT_HI = 0x807, 0x808

# per-worker running total, exactly like the odometer's clamped camera
# integral: cumulative_damage[wid] += max(0, prev_foe_hp - foe_hp_now)
# whenever foe_hp DROPS (a landed hit), and on a round-gate transition
# (or the mass-reset detector, for a fixed-ring game with no round byte)
# prev_foe_hp is RE-ARMED to the fresh opponent's foe_hp_start instead of
# being read as a (foe_hp_start - 0) fake windfall. Clamped >=0 and to
# 16 bits, same non-negative-clamp discipline as the odometer's 24-bit
# integral.
```

This gives the archive a monotone-increasing "total damage dealt"
frontier with NO new cell/archive code — a Go-Explore cell that reaches
a higher cumulative-damage value is strictly better banked material, the
same relationship `gx` already has to a spatial cell.

### 4.2 The round axis plays the role `area` already plays

If `find_round_gate` nominates a byte, it is wired to `solve: area`
exactly like a room-based game's screen counter — the archive already
partitions cells by `(area, progress-bucket)` for every room-based
profile; a fight-gate game's `(round, cumulative-damage-bucket)` is the
same partition with different labels. Punch-Out has no such byte
(confirmed dead axis, §1), so its profile omits `area` and search is
scored on the cumulative-damage integral alone within the one bout —
this is the honest degraded case, not a blocker.

### 4.3 Win/clear detection needs no new code

`solve: clear:` already supports exactly the two modes this needs:

- `byte_change {addr: <round>, direction: up}` — a bout won (round
  advances) fires a clear the same way a stage-clear byte does
  elsewhere in the roster.
- `confluence {...}` — for a game with no round byte at all (a
  fixed-ring one-bout profile like the Punch-Out validation target),
  the existing multi-signal detector's `audio` (fanfare / music-cut)
  and `apu` (channel-activity vote) signals are purity-clean win
  detectors that need no RAM byte whatsoever. `progress_median: K`
  is applied by default on any fight-gate profile — it is the
  Double Dragon combat-blip fix (§1) and a fight-gate game is
  *maximally* exposed to that exact blip, being combat by
  construction.

### 4.4 This subsumes, and should replace, the manual `boss:` config

`GenericGame` already has `boss: {hp, start}` and `boss_typed: {...}`
(§ comments above `self._boss_hp` in the current code) — a
hand-discovered boss-HP byte that "must be discovered by differential
analysis of our own rollouts and receipted like every other solve
address." That sentence describes exactly the manual process §3.2
automates. Once `find_fight_health` exists, `boss:` becomes the SAME
field populated by an automated discovery pass instead of a bespoke
one-off investigation per game — no schema change, no consumption-side
change, only a faster and more repeatable path to filling it in.

## 5. Pre-registered validation — Punch-Out

### 5.1 Target and why it is closest to ready

`configs/punchout.yaml` already has: a captured, verified-live start
state (Glass Joe bout); a generic action space (no move is
Punch-Out-specific — NOOP / A / B / up+A / up+B / left / right / down,
the same attack+dodge vocabulary any side-view combat game in this
roster exposes); and an explicit, receipted verdict
(`CAMERA_STATIC_AGENT_ACTIVE`, quoted in full in §1) that BOTH odometer
axes are dead and that this exact mechanism is the named prerequisite
before `go_explore_solve.py` may be pointed at it. Nothing else in the
roster has as complete a starting rig with as clean a "blocked on
exactly this" receipt. The one thing standing between this profile and
a real search is the discovery pass this document specifies.

### 5.2 Procedure (not yet run)

```
.venv/bin/python scripts/discover_observables.py \
    --rom "roms/Mike Tyson's Punch-Out!! (Japan, USA) (Rev A).nes" \
    --state "roms/Mike Tyson's Punch-Out!! (Japan, USA) (Rev A)_start.state.bin" \
    --fight-gate --emit-solve --out runs/fightgate_punchout/discover.json
```

followed by a `progress_signal_gate.py`-style long-rollout sanity check
of whatever `foe_hp` candidate is nominated, then — ONLY if that gate
passes — a brief `go_explore_solve.py` smoke (a few minutes, single
worker, the standard onboarding-wave smoke budget) to confirm the
cumulative-damage integral actually accrues over an unattended run. No
smoke has been run for this document; it is scoped entirely to the
discovery pass and the prediction below.

### 5.3 Pilot evidence (calibration only, NOT the validation run)

Three brief, single-worker, headless probes were run directly against
`nes_core` from the captured start state to check that the *phenomenon*
this design assumes is real before specifying gates around it. Each is
a few hundred steps (a few seconds of wall clock); none used the
`ram_mapping:` addresses as a discovery method — they were read only to
print what a hand-known byte does under each probe, exactly the way a
`--selftest` checks new code against already-verified ground truth.

**Probe 1 — fixed-rhythm attack mash (alternating A, B; no defense),
400 steps.**
```
baseline  opp_hp(0x398)=96  mac_hp(0x392)=96  round=1  match=0
after 400 steps:
          opp_hp(0x398)=96  mac_hp(0x392)=0   round=1  match=0  opp_down=0
```
`opp_hp` never moved; `mac_hp` was driven to zero. This reproduces, from
direct measurement, the config's own comment that "the generic masher
does NOT reach live gameplay on this dump" — a fixed-rhythm mash is an
INSTRUMENT problem (§2), not evidence that no foe-HP byte exists.

**Probe 2 — fixed-cycle dodge+attack (LEFT×4, RIGHT×4, A×3, B×3,
DOWN×2, repeating), 600 steps.**
```
baseline  opp_hp=96  mac_hp=96
after 600 steps: opp_hp=96  mac_hp=28  (min over run: mac_hp=28, opp_hp=96)
```
Still zero movement on `opp_hp`; `mac_hp` degraded less than Probe 1 but
still took damage. A deterministic cycle, even one mixing offense and
defense, is a fixed rhythm this opponent's AI evidently tracks well
enough to counter — consistent with §3.1's requirement that
`attack_mash` sample RANDOMLY rather than cycle.

**Probe 3 — randomized weighted mash (A 0.25, B 0.25, LEFT 0.15,
RIGHT 0.15, DOWN 0.10, NOOP 0.10; independent per-step draws), 1000
steps.**
```
baseline       opp_hp=96  mac_hp=96
min opp_hp over run: 70   (final 70)  — opp_hp DID move
min mac_hp over run: 0    (final 72)  — mac_hp hit zero, then RECOVERED
round=1  match=0  losses=0 (unchanged)
```
This is the calibration finding that sets the design, not just confirms
it: (a) `opp_hp` only decremented once the probe stopped following a
fixed rhythm — direct support for §3.1's randomized-sampling
requirement; (b) `mac_hp` dropped to zero and then rose back to 72
*inside a single continuous probe with no death and no round change* —
direct, concrete confirmation that Gate FH3's refill-aware regime split
(§3.2) is necessary, not a hypothetical edge case, because it already
happens inside a single 1000-step window on this exact game.

### 5.4 Pre-registered prediction

**SUCCESS looks like:** `find_fight_health`, run blind (no address
list consulted, per §2), nominates a `foe_hp_candidate` at or adjacent
to `0x0398`/`0x0399` in its top-3 by Gate FH1+FH2 score, correctly
separates it from a `self_hp_candidate` at or adjacent to
`0x0392`/`0x0391`, and reports the Probe-3-style refill event as a
named regime split rather than a rejection. `find_round_gate` correctly
reports `insufficient_probe` (VOID, not a nomination) if the probe
budget doesn't include a full bout's end, OR nominates `0x0006` if it
does. A subsequent `go_explore_solve.py` smoke with
`progress: {source: fight_gate, foe_hp: 0x0398, ...}` shows the
cumulative-damage integral (§4.1) rising above zero across the run
without operator-supplied addresses anywhere in the profile except
what the discovery pass itself emitted.

**FAILURE modes named in advance:**

1. **Self/foe aliasing survives the gate.** `find_hp_lives` (already
   shipped) and `find_fight_health` (new) both nominate the same
   address, or FH2 is too permissive and lets `0x0392` (mac_hp) through
   as a foe candidate. Directly reproducible risk — Probe 1 shows the
   naive signature (decrements under offense) alone cannot separate
   them.
2. **The randomized probe still never lands enough hits within a
   realistic discovery budget** (hundreds to a few thousand steps,
   the `ADVANCE_N`-class budget, not a `--selftest`-class one), and
   `opp_hp` reports flat — a correct VOID (instrument-insufficient),
   but one that is indistinguishable from a real "no purity-clean
   foe-HP byte exists" finding without the instrument/behaviour split
   this design insists on. Mitigation is a probe-budget floor derived
   from how many random steps Probe 3 needed (a decrement appeared
   within 1000 steps here; the shipped default should sit meaningfully
   above that, not at it).
3. **The refill event is mis-scored as a death/reset** because it
   crosses `_first_reset`'s mass-rewrite threshold even though no
   actual bout ended — Probe 3's `mac_hp` 0→72 recovery happened with
   `round` and `match` both unchanged, so if the refill also happens to
   coincide with a large RAM churn window, `find_round_gate` could
   misfire a spurious round boundary. This is the same class of bug
   the Double Dragon combat-blip already is (§1); `progress_median`
   defaulting ON (§4.3) is the mitigation, not a cure — this should be
   checked directly against a captured Probe-3-style trace before the
   discovery pass is trusted.
4. **No purity-clean win signal exists for Punch-Out at all** — if
   neither `audio` nor `apu` fires cleanly on a real KO/TKO (untested;
   this game's fanfare-on-win audio profile has never been measured by
   this project's confluence detector), the fight-gate PROGRESS axis
   could still succeed (cumulative damage rises) while the CLEAR
   detector remains a separate, unresolved problem — these are
   deliberately decoupled in §4.1/§4.3 so a partial result (progress
   works, clear detection doesn't yet) is reportable as such rather
   than forced into a single pass/fail bit.

**VOID conditions**, distinguished from the above per §2's doctrine:
probe budget too small to exercise a single hit or a single bout
end (instrument), vs. a probe that ran long enough and genuinely found
no decrementing byte matching FH1+FH2 anywhere in RAM (behaviour — the
honest "this game's opponent HP is not represented as a scalar RAM
counter" finding, which would itself be a receipted, useful result).

## 6. Non-goals / what this document does not do

- Does not modify `configs/punchout.yaml`, `kungfu.yaml`,
  `double_dragon.yaml`, or `bubble_bobble.yaml`.
- Does not implement `find_fight_health`, `find_round_gate`,
  `attack_mash`, `approach_retreat`, the `fight_gate` progress source,
  or the `FIGHT_LO`/`FIGHT_HI` pseudo-RAM slots.
- Does not run `discover_observables.py --fight-gate` or any
  `go_explore_solve.py` smoke. §5.3's three probes are direct `nes_core`
  reads used only to calibrate the prediction in §5.4, clearly
  distinguished from the pre-registered run they inform.
- Does not claim Kung Fu's or Double Dragon's fight-gate signal is
  solved by this design — both are named in §1 as precedent for the
  problem shape, not as validation targets. Punch-Out is the sole
  pre-registered target because it is the one game with an explicit
  "needs exactly this" receipt and no spatial fallback to confound the
  result.
