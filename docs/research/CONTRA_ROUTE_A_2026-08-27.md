# Route A — does the Contra gx-3072 lock look different if you get there another way?

**Date:** 2026-08-27
**Ledger: EXHIBITION, without exception.** Every number below is Go-Explore
search output or instrument measurement. No policy was trained for this game
and no honest-protocol evaluation was run. Nothing here may be described with
"the AI learned", "the AI plays", or "the AI beat" — see `CLAIMS.md`.
**Emulator:** `nes_core` sha256_16 `54366c20d32f71cc` (the same core the
characterisation used).
**ROM** `roms/Contra (USA).nes` sha256
`26541a5550ee22deeb3d5484e4a96130219b58cff74d068fb1eb6567fa5e5519`.
**Receipts:** `runs/contra_routeA/` (gitignored) — every manifest, every
screen result, and the eight harness scripts.
**Builds on, does not re-derive:** `docs/research/CONTRA_WALL_2026-08-27.md`.

---

## 1. The two answers, up front

**Do wall states differ by approach? YES, and materially.** Four fresh
from-power-on searches — different mint, different action prior, different
selection rule, different seed — arrive at the lock in configurations that a
nearest-neighbour test separates at **92% purity against a 14% chance
baseline**, with a permutation p of 0.0005 while a sham split of one arm
scores p = 0.86. Approach fixes `$00AA` (the profile's `weapon` label) to a
single value per approach, and three of the four sit at values (**16**, **19**)
that appear in **none** of the 69 reference states from the prior campaign's
lineage. Approach also fixes the life count at entry, moves steps-to-arrive
into non-overlapping bands (772–816 / 848–862 / 941–1050 / 1371–1520), and
moves the fraction of arrivals that are input-live from **0.47 to 0.73**.
The brief's premise was right: *no attack had ever reached the lock another way,
and it does look different when you do.*

**Is Route A therefore worth a fan-out? No — and this is not the "they are
identical" branch.** Everything that differs is measurably inert on the only
axes that matter. Across 238 input-live wall states drawn from six independent
approach populations, held-mask probing returns the camera pair `(12, 0)`,
progress `3072`, and `px_max` **136** — one value each, in every single arm,
with no exception. A matched pilot (identical prior, burst, budget, cell key;
only the root population differs) gave the prior campaign's own legacy roots
the **largest** cell count of the four, not the smallest. And the one apparent
exception — px reaching 144 from one approach's roots, replicated across four
seeds and never once from any control — **resolves to an already-committed
input-dead window in 156 of 156 screened witnesses.** `beat_3072` is false
everywhere: 2,417,912 worker-steps of fresh from-power-on search plus
~1.25M alive steps resident at the lock produced no trajectory above 3072.

> **The headline is a correction to the *class* of the previous negative.** The
> characterisation could only falsify "no escape from banked roots of one
> lineage". This work removes that caveat: the lock now holds against roots
> reached by four genuinely independent from-power-on routes, and the
> approach-dependence that Route A was betting on is real but does not buy
> anything.

---

## 2. What was run

Four arms, concurrent, one pool worker each, `--workers 4` throughout. Each
arm differs in **all** of {root mint, action prior, archive selection rule,
RNG seed}.

| Arm | Root | Action prior | Selection | Seed |
|---|---|---|---|---|
| `A_rush` | `mint_0` | right-family p=0.80, sticky 0.75, burst 48 | greedy-deep, inverse-visit within a 96-px band | 1001 |
| `B_coverage` | `mint_1` | right-family p=0.45, sticky 0.35, burst 32 | inverse-sqrt(visits) over the whole archive | 2002 |
| `C_cautious` | `mint_2` | safe-set baseline, right only as 2–5 step taps, burst 64 | survival × gx weighted | 3003 |
| `D_macro` | `mint_3` | held macros 8–40 steps, 55% right-family, burst 64 | deep-bias 0.4 | 4004 |

### The mints are genuinely different power-on entries

`mint_powerons.py` boots the ROM from **reset** and mashes into gameplay under
a variant-specific menu schedule (different idle prefix, pulse period, duty
cycle, button rotation, settle length). Gameplay is detected game-agnostically
by the RIGHT-vs-LEFT divergence test `capture_start_state.py` already uses —
growth measured on the split itself so shared animation cancels, plus
directionality so a demo that ignores input fails. No ROM knowledge is used.

The four mints differ from each other by **183–407 RAM bytes** and from the
single shared savestate every prior Contra run booted from by **231–328
bytes**. Preflight: a forward hold from each moves progress 0 → 635, caps
`player_x` at 128 (the scroll-lock signature), and dies at step **63 / 76 / 63 /
80** — four different phases, not four copies.

### Reaching the lock

| Arm | max gx alive | first reach (lineage steps) | cells | deaths | `beat_3072` |
|---|---|---|---|---|---|
| `A_rush` | 3072 | 852 | 554 | 11,034 | false |
| `B_coverage` | 3072 | 829 | 4,145 | 5,565 | false |
| `C_cautious` | 2524 → 2918 → reached in stage 2 | — | 9,405 | 15,795 | false |
| `D_macro` | 3072 | 955 | 3,667 | 6,679 | false |

Stage 1 was 1,613,292 worker-steps over 30 min. `C_cautious` did not reach the
lock there, so it was given the whole 4-worker pool under its **own** prior and
**own** selection rule for another 804,620 worker-steps (`stage1b`) — compute,
not method, was changed — and got to 2918. It finally arrived in stage 2, where
96-step bursts from its own 2800–2918 frontier produced 111 arrivals. **That an
approach can be 6× less likely to arrive at all is itself an approach
difference**, and it is why the input-live fractions below are not
interchangeable.

### Sampling arrivals

Stage 2 resumes **only** from each arm's own sub-wall corridor (gx 2800–3071,
the same 272-px-earlier offset the characterisation used for its camera
positive control), replays that arm's **own** prior, and banks the first step of
each burst that reads exactly 3072 while alive, from a root that read below
3072 at its priming step. Arrivals are **reservoir-sampled**, never deduped, so
the banked 60 per arm are unbiased over 5,724 / 4,294 / 111 / 5,570 total
arrivals.

Two further populations were added:

* **`A_rush_life1` / `A_rush_life0`** — under a strict never-cross-a-death
  discipline every burst inherits its root's life count, so all four approaches
  necessarily arrive on the first life, and so does the prior campaign
  (59 of 61 verified legacy cells read `lives == 2`). **Nobody has ever varied
  life count.** These two arms derive lives-1 and lives-0 roots from `A_rush`'s
  own frontier by holding a lethal action until the byte decrements, settling,
  re-verifying, and re-screening for input-liveness. 12 of 12 derived roots
  screened LIVE at each level.
* **`LEGACY_solve20` (n=61) and `PRIOR8_headwall` (n=8)** — the prior
  campaign's own lineage, as the reference every earlier measurement was taken
  on. Legacy cells were lifted from `runs/play_one_well/contra/solve20/archive.pkl`
  and kept only if, one NOOP priming step after **restore on today's core**,
  they read progress exactly 3072 with lives > 0 (61 kept of 652 scanned; the
  9.4% keep rate matches the archive's 8.2% wall-bucket share). Savestates, never
  traces — cross-build trace replay is known broken for this ROM.

**429 wall states banked and screened in total.**

---

## 3. The input-liveness screen, and why you can believe it

The brief is explicit that 2 of 8 previously banked wall states are
already-committed fatal windows and that including them poisons any
conclusion. Every state above is screened before it is counted.

The screen is a **differential**, never a churn count: 14 held masks (the 11
profile actions plus `up`, `left+B`, `A+B`) for 32 steps from the same restored
state, and the score is how many bytes take *different* values under *different*
inputs — union over every step at which at least four masks are still alive.
Each mask's run is truncated at the first lives change **or** at a large
backwards collapse of progress, because on this ROM the lives byte holds flat
past the final death (measured directly: 2 → 1 → 0, then 1,058 further steps
pinned at 0 while progress collapses to 0 and stays there). A lives-equality
gate alone is not a liveness gate here; it is paired with a positional
invariant everywhere in this work.

Three calibrations, all run through the identical code path:

| Population | verdict | input-dep RAM | input-dep OAM | sprite slots | latched |
|---|---|---|---|---|---|
| **DEAD control** (900 forward steps from a mint — past the final death) | 0/3 LIVE | **2** | **0** | **0** | **2** |
| Open-field control (gx 700–2200, arms' own archives) | 6/8 LIVE | 442 (med) | 120 | 52 | 70 |
| **`PRIOR8_headwall` — ground truth fixed by someone else** | **5/8 LIVE** | 487 (med) | 79 | 21 | 51 |

The prior-8 result is the strongest anti-vacuity evidence available, because
the labels predate this screen. It flags `head_wall_0` (all 14 masks die at the
identical step, 5 input-dependent bytes) and `head_wall_1` (identical step, 4
bytes) — **exactly the two the characterisation identified**, at the same
steps modulo the priming offset. It additionally flags `head_wall_3`, in which
all 14 masks survive the full window and *nothing responds* (4 input-dependent
bytes, 0 OAM) — an unresponsive state that is not a fatal window, and one the
prior labelling did not separate out.

The gap between LIVE and DEAD is not marginal: 475–588 input-dependent RAM
bytes against 2–5.

**Input-live fraction, by population:**

| Population | screened | LIVE | fraction |
|---|---|---|---|
| `C_cautious` | 60 | 44 | **0.73** |
| `B_coverage` | 60 | 40 | 0.67 |
| `PRIOR8_headwall` | 8 | 5 | 0.62 |
| `LEGACY_solve20` | 61 | 32 | 0.52 |
| `D_macro` | 60 | 30 | 0.50 |
| `A_rush_life0` | 60 | 30 | 0.50 |
| `A_rush_life1` | 60 | 29 | 0.48 |
| `A_rush` | 60 | 28 | 0.47 |
| **Total** | **429** | **238** | **0.55** |

Roughly **45% of wall arrivals are input-dead**, far above the 2-of-8 rate the
prior campaign's hand-picked sample implied. Anyone rooting a future search at
unscreened wall states is spending about half their budget on corpses.

---

## 4. What differs by approach

All figures are over the input-LIVE states only.

| Arm | n | `$00AA` | lives | px range (distinct) | y range | sprites | latched med/max | steps to arrive |
|---|---|---|---|---|---|---|---|---|
| `A_rush` | 28 | **16** | 2 | 87–128 (10) | 76–164 | 12–19 | 57 / 122 | 848–862 |
| `B_coverage` | 40 | **19** | 2 | 33–131 (28) | 42–196 | 11–25 | **112 / 255** | 772–816 |
| `C_cautious` | 44 | **16** | **1** | 25–127 (28) | 42–155 | 8–16 | 88 / 162 | 1371–1520 |
| `D_macro` | 30 | 0 | 2 | 25–127 (17) | 55–196 | 9–20 | 93 / 142 | 941–1050 |
| `A_rush_life1` | 29 | 0 | **1** | 80–132 (18) | 74–196 | 8–17 | 73 / 129 | — |
| `A_rush_life0` | 30 | 0 | **0** | 25–131 (18) | 44–177 | 9–17 | 95 / 154 | — |
| `LEGACY_solve20` | 32 | 0 / 4 | 2 (30), 1 (2) | 25–**136** (22) | 32–196 | 12–26 | 80 / 182 | — |
| `PRIOR8_headwall` | 5 | 0 | 2 | 95–**136** (5) | 74–196 | 11–20 | 56 / 174 | — |

### The separation is statistically real, and the machinery is not vacuous

Distance is Hamming over all 2048 RAM bytes at the arrival step.

| Test | Result |
|---|---|
| Permutation, arm labels (2000 perms) | within 364.3, between 497.9, **stat +133.6, p = 0.0005** |
| **Sham control** — random half/half split of `C_cautious` (1000 perms) | stat **−3.2**, p = **0.86** |
| **Positive control** — open-field vs wall, same metric (1000 perms) | stat **+257.3**, p = 0.001 |
| Nearest-neighbour arm purity | **0.920** vs chance 0.138 |
| NN purity on the 777 non-churning ("quiet") bytes only | 0.303 vs chance 0.138 |
| Bytes constant inside every arm and differing across arms | **8** (`$011A`, `$0148`, `$0149`, `$014B`, and four seen only in the prior-8) |

A fake arm boundary scores zero and a genuinely different population scores
double the real effect, so the +133.6 is neither noise nor an artefact of the
metric.

### The four differences that are actually decision-relevant

1. **`$00AA` is approach-determined and the fresh values are disjoint from the
   reference set.** Constant across all 60 banked states of each arm:
   `A_rush` = 16, `B_coverage` = 19, `C_cautious` = 16, `D_macro` = 0, both
   life-variants = 0. The legacy lineage carries only {0, 4} across 61 cells;
   the prior-8 carry only {0}. **180 fresh arrival states sit at a value that
   appears in zero of the 69 states every one of the eight prior attacks was
   measured on.** (Purity note: this document does not say what any value of
   `$00AA` corresponds to. It reports that the byte is fixed per lineage and
   that the fresh and reference value sets do not intersect. It also records,
   without interpretation, that the derived life-variant roots read 0 — losing a
   life resets it.)
2. **Life count at entry is approach-determined.** `C_cautious` arrives at
   `lives == 1` in all 44 of its live states; the other three arrive at 2; the
   derived variants at 1 and 0. Nobody had varied this before.
3. **Steps-to-arrive bands do not overlap.** 772–816 (`B`) < 848–862 (`A`) <
   941–1050 (`D`) < 1371–1520 (`C`).
4. **Irreversible-action capacity differs.** `B_coverage` reaches states with
   **255** agent-latched bytes, against the characterisation's stated per-state
   range of 20–74 and against `A_rush`'s 122. Its median (112) is double
   `A_rush`'s (57).

### The differences that turn out to be small

Interleaving — the median nearest-neighbour distance to the arm's own states
versus to any other arm — is 7.86 for `A_rush` and 6.22 for `D_macro` (both
arrive in a *narrow* set of configurations), but only **1.10** for
`A_rush_life0`, 1.68 for `A_rush_life1`, 1.21 for `LEGACY` and 1.20 for the
prior-8. **The life-count axis barely moves the population**; it is the
strategy/mint axis that does.

---

## 5. And none of it buys anything

### 5.1 The invariants are approach-invariant

Every one of the 238 input-live states, 14 held masks, 32 steps, alive-only,
death-truncated on the paired gate:

| Population | camera pairs seen | gx max | px max |
|---|---|---|---|
| `A_rush` (28) | `{(12, 0)}` | 3072 | 136 |
| `A_rush_life0` (30) | `{(12, 0)}` | 3072 | 136 |
| `A_rush_life1` (29) | `{(12, 0)}` | 3072 | 136 |
| `B_coverage` (40) | `{(12, 0)}` | 3072 | 136 |
| `C_cautious` (44) | `{(12, 0)}` | 3072 | 136 |
| `D_macro` (30) | `{(12, 0)}` | 3072 | 136 |
| `LEGACY_solve20` (32) | `{(12, 0)}` | 3072 | 136 |
| `PRIOR8_headwall` (5) | `{(12, 0)}` | 3072 | 136 |
| **union over every approach** | **`{(12, 0)}`** | **3072** | **136** |

Positive control on the same code path: open-field states report **20–63
distinct camera pairs**, so the probe can see camera motion when there is any.

`px_max = 136` is worth pausing on. Arrivals land at px ≤ 132 in every fresh
arm, yet holding masks drives px to **exactly 136 and no further** in all eight
populations — the characterisation's number, reproduced from four independent
routes it was never measured on.

### 5.2 The matched pilot: fresh roots buy less search surface, not more

Four pool workers, identical in every respect — same action prior (iid over the
11 profile actions, sticky 0.5), same 48-step bursts, same inverse-visit
resume, same cell key `(px//8, y//16, six zero-page cyclers)`, same budget —
differing **only** in which population's input-live wall states they are rooted
at. 600 s, ~125k alive steps each.

| Roots from | alive steps | distinct cells | camera | gx max | px max | beat |
|---|---|---|---|---|---|---|
| `LEGACY_solve20` | 125,062 | **20,436** | `{(12,0)}` | 3072 | 136 | false |
| `C_cautious` | 125,796 | 19,988 | `{(12,0)}` | 3072 | 136 | false |
| `A_rush` | 124,614 | 15,160 | `{(12,0)}` | 3072 | 136 | false |
| `B_coverage` | 125,039 | 12,475 | `{(12,0)}` | 3072 | **144** | false |

The prior campaign's own single-lineage roots produced **the most** distinct
cells of the four. If the fresh configurations opened anything up, this is where
it would show, and it does not.

### 5.3 px 144 — replicated, matched-controlled, and dead

`B_coverage` was the only population to read px above 136. It replicates:

| Arm | alive steps | px max | steps at px > 136 |
|---|---|---|---|
| `B_coverage` seed 11 | 123,905 | **144** | 246 |
| `B_coverage` seed 999 | 123,984 | **144** | 260 |
| `LEGACY_solve20` seed 11 | 123,825 | 136 | **0** |
| `D_macro` seed 11 | 124,354 | 136 | **0** |

plus two further `B_coverage` seeds (4242, 8675) at 144 and 144 in a shorter
replicate. Witness savestates cold-restore to px 137 / 142 / 144 at progress
exactly 3072 with `lives == 2`. Under a tightened gate — priming lives > 0 and
progress **exactly** 3072 on every scored step — this is a real reading, in a
population reached only by a fresh approach, of a value the characterisation
declared universal at 136.

**It is not a capability.** 160 witnesses were banked and put through the same
input-liveness screen:

```
verdicts:      DEAD_UNRESPONSIVE 140 | DEAD_FATAL_WINDOW 16 | LIVE 4
by entry px:   137 -> 70 dead + 8 fatal      142 -> 2 dead
               138 -> 39 dead                144 -> 29 dead + 8 fatal
               (the 4 LIVE have entry px 0 and 48 — the excursion is over
                by their priming step)
input-dep RAM across the dead ones: 0-4      input-dep OAM: 0
```

**156 of 156 states whose priming step reads px > 136 are input-dead**, at the
dead-control's own signature. Their `median_alive` counts down monotonically
across consecutive witnesses (18, 17, 16, 15, 14, 13 …) — the fingerprint of a
single already-committed commitment being sampled one step further along each
time. The lives byte still reads 2 throughout, which is precisely why a
lives-equality gate would have called every one of them alive and published
"px_max is 144, the characterisation is wrong".

**Corrected statement, which supersedes nothing:** `px_max = 136` holds for
every *input-live* state in every approach. What is approach-dependent is which
already-committed dead windows a population can fall into — and a dead window
is not an attack surface.

---

## 6. Verdict, and what it licenses

**`states_differ_by_approach: true`. `route_a_worth_pursuing: false`.**

These are not in tension. The brief's decision rule had two branches — identical
states kill Route A, differing states make the difference the attack surface.
The measured outcome is a third: the states differ, substantially and
reproducibly, and the difference is **inert**. That is a stronger negative than
either branch anticipated, because it removes the caveat the characterisation
had to attach to its own result. `attackable_by_search` is now falsified for
wall states reached by four independent from-power-on routes, at three life
counts, across `$00AA` values the prior work never held, not merely for the
boundary-resident family of one lineage.

Total this campaign: **2,417,912 worker-steps** of fresh from-power-on search,
**~757k** alive steps of arrival sampling, **~1.25M** alive steps resident at
the lock under matched pilots, **429** wall states banked, **238** screened
input-live, **160** px-excursion witnesses screened. `beat_3072: false`
everywhere. **No tape exceeds 3072, so `docs/receipts/` gains nothing.**

### What this does hand forward

1. **Screen your roots.** 45% of wall arrivals are input-dead — five times the
   rate the hand-picked 2-of-8 sample suggested. The screen in
   `runs/contra_routeA/routeA_stage3_screen.py` reproduces the prior campaign's
   independently-fixed labels and separates LIVE from DEAD by two orders of
   magnitude (475–588 input-dependent bytes vs 2–5). Any Route B search should
   run through it first; a Route B objective that scores survival-at-wall would
   otherwise spend half its selection pressure on corpses.
2. **Prefer `C_cautious`-shaped root stock.** 0.73 input-live against
   `A_rush`'s 0.47, and 19,988 cells in the matched pilot against 15,160 — the
   best root population of the fresh four, and within 2% of legacy.
3. **The lives byte is not a liveness gate, twice over.** It holds flat past the
   final death *and* it reads a healthy 2 throughout the px-144 dead windows.
   Pair it with a positional invariant, always. Both traps fired in this
   campaign and both were caught by the pairing.
4. **Route B is untouched and remains the open branch.** Nothing here bears on
   the `max_gx_in_max_area` defect: selection still scores every one of these
   demonstrably-distinct wall states 3072. This campaign has now added a
   further reason to fix it — the archive can discriminate approach-determined
   configurations that the objective cannot see at all.

### Limits of this claim, stated as plainly as the result

* Approach is confounded on purpose. Each arm varies mint, prior, selection and
  seed together, which maximises independence and buys the binary answer
  cheaply — but it means "`$00AA` = 19 causes the px-144 dead windows" is
  **not** established. `B_coverage` differs from the controls in four ways at
  once and only one arm ever produced the excursion.
* `C_cautious` was given extra compute to arrive at all (its own prior and
  selection rule, four workers instead of one). Method was held fixed; budget
  was not.
* The matched pilots are 600 s and 300 s at one action prior and one cell key.
  They are the size of an individual attack in the previous fan-out, not the
  size of the fan-out.
* Under the never-cross-a-death discipline, life count at entry could only be
  varied by deriving roots — so the lives axis is tested from one lineage
  (`A_rush`), not four.
