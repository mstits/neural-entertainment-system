# The purity line was breached, by two independent paths, for months

**2026-08-26.** Status: both paths closed, with tests that fail when the mechanism is
absent. Four further leaks of the same class were found during the fix and closed. This
document is the record.

---

## Why this is filed as the most serious defect class in the project

The purity line is the central claim. Everything this project asserts — that it beat Super
Mario Bros. from power-on, that its solver finds its own way through a maze, that a policy
learned rather than memorised — rests on one premise: **the system discovers its own
observables rather than being handed them.** Strip that premise and the artifacts do not
degrade gracefully into weaker results. They stop being results. A search that was told
where the win flag lives has not searched for it, and a receipt from an instrument that was
told which bytes not to look at is a receipt for a question that was never asked.

That is why a breach here outranks any performance regression or correctness bug in this
repository. A wrong number can be re-measured. A contaminated claim cannot be
decontaminated after the fact; it can only be withdrawn or re-run from scratch.

Two independent breaches were live simultaneously. Neither was found by a test. Both were
found by a human reading code during an adjudication that was nominally about something
else. **The quarantine mechanism, which exists precisely to prevent this, did not detect
either one — and one of the two breaches ran *through* the quarantine's own machinery,
turning it into the delivery vehicle.**

---

## Breach path 1 — a text-clean profile inherited a disassembly-sourced win predicate

`configs/legend_of_zelda.yaml` is 31 lines. It has no `ram_mapping`. It has no quarantine
block. It contains no RAM address of any kind. It declares no reward weights. It describes
itself as "Not a training profile". It passes the outside-provenance lint in
`scripts/onboard_game.py` that `configs/zelda.yaml` trips.

At runtime it was `ZeldaReward` — including the win predicate built on
`RAM_GANON_DEFEATED = 0x0672`, whose provenance in `nes_core/src/rewards.rs` reads
"aldonunez disassembly + Data Crystal" and whose status is `UNVERIFIED_EXTERNAL`.

The mechanism was one line of dispatch:

```rust
if name.contains("zelda") { return Some(Reward::Zelda(...)) }
```

The profile's `name:` is `"The Legend of Zelda"`. That was the entire qualification.

Measured on the pre-fix binary, against a 2 KB buffer of zeros with a single byte flipped:

| | reward | done | `episode_success()` |
|---|---|---|---|
| zeroed RAM | −0.001 | False | False |
| `ram[0x0672] = 1` | **19999.999** | **True** | **True** |

One flipped byte, sourced from a disassembly, took a profile that declared nothing at all
from "no progress" to "game won".

**The quarantine did not cost this project Zelda. The substring dispatch did.** The
quarantine was working exactly as designed — `configs/zelda.yaml` stores its contaminated
addresses as `q_`-prefixed *strings* specifically so `int(a)` raises and no consumer can
use them. The dispatch reached around it, because it never consulted a config at all.

Two more profiles were inheriting the same way, neither named in the original adjudication:

- `configs/zelda_roomfp.yaml` — reproduced the identical 0x0672 jackpot.
- `configs/metroid_roomfp.yaml` — silently acquired `MetroidReward`, which `CLAIMS.md`
  lists as **BLOCKED** from producing any Learned-ledger claim.

And the same mechanism failed in the opposite direction at the same time:
`configs/smb_4_4_micro.yaml` (name `"SMB 4-4 micro (full controller)"` — contains "smb", not
"mario") declares nine `MarioReward`-only weights and got `GenericReward`, so all nine were
inert. A name is not a usable selector in either direction.

### How long it was live

| | from | to | duration |
|---|---|---|---|
| `name.contains(...)` dispatch existed | `55e5333` 2026-04-27 (initial commit) | `2a15dfb` 2026-08-26 | **121 days** |
| A quarantined predicate was reachable by inheritance | quarantine established `9553682` 2026-08-25 | 2026-08-26 | 1 day |
| The three silently-inheriting profiles existed | `794270d` / `09a1c03` 2026-08-24 | 2026-08-26 | 2 days |

The honest reading of those rows: the *defect* is four months old; the *breach of the
quarantine* is one day old, because the quarantine is one day old. The mechanism was
waiting for the quarantine to be created so it could defeat it. Had the two `roomfp`
profiles been minted a week later, or the quarantine a week earlier, the exposure window
would have been correspondingly longer for no reason connected to anyone's diligence.

### Closed by

`build_reward` in `nes_core/src/rewards.rs` no longer takes a display name. Its signature is
`build_reward(reward_id: &str, weights)`, matching exactly, and the sixteen substring
branches are deleted. The display name is not a parameter of the function, so substring
dispatch on it is not merely discouraged — it is not expressible without re-plumbing a
parameter. Same treatment for `GameKind::from_name` → `from_reward_id`. `reward_id` is a
top-level profile key; absent means `generic` (axis-free, no win predicate, cannot witness a
clear); an unknown value raises `ValueError` naming the valid set. 126 configs were migrated
mechanically from a baseline frozen before any Rust changed.

Verified on the rebuilt binary: `{"name": "The Legend of Zelda"}` with no `reward_id`
resolves to `generic`, and flipping 0x0672 moves it 0.010 → 0.009, `episode_success()`
False. The positive control still fires — `{"name": "totally unrelated", "reward_id":
"zelda"}` still jackpots — so this is a real negative from a live instrument.

---

## Breach path 2 — an external RAM map was steering the discovery instrument away from the quarantined bytes

This is the worse of the two, and it is worse in kind, not degree.

`configs/zelda_gui_tuned.yaml` is a fourth Zelda profile, picker-visible and pinned bootable
by `tests/test_profile_configs.py`. It carried a live, unquarantined 13-entry `ram_mapping`
under the comment *"Win chain (disassembly + emulator-verified)"*. Six of its values were
the exact quarantined addresses: `dungeon_level` 0x10, `current_hearts` 0x66F, `max_hearts`
0x66F, `triforce_pieces` 0x671, `ganon_defeated` 0x672, `song` 0x609.

They were plain ints, not the `q_`-prefixed strings the quarantine convention uses. So
`scripts/observatory.py` — this project's own conditional-observable discovery instrument —
parsed them:

```python
mapped = {int(a) for a in (profile.get("ram_mapping", {}) or {}).values()}
known = coord_bytes | mapped
excluded |= known
```

and `excluded` is the set consulted at `if b in excluded: continue`, the gate on **candidate
predicate generation**. An excluded byte is never probed, never scored, and never appears in
the receipt.

Read that composition slowly, because the summary is not "a stale address was in a config":

> An external RAM map, copied from a disassembly and a third-party wiki, was deciding which
> bytes this project's own discovery instrument was permitted to nominate — and the bytes it
> was removing were **precisely the ones under quarantine.**

The quarantine's stated exit condition is independent rediscovery by this instrument. The
exclusion made that rediscovery structurally impossible. It is not a weakened quarantine; it
is the quarantine inverted, with the contamination record repurposed as the blindfold.

Nobody wrote it on purpose. The exclusion's stated rationale in the module docstring —
"already-known observables; the tool discovers NEW ones" — is a **reporting** concern:
don't clutter the top-10 with things we already named. It was implemented as **blindness**.
Those are not the same thing, and every consequence here follows from conflating them.

### How long it was live

| | from | to | duration |
|---|---|---|---|
| `zelda_gui_tuned.yaml` carried the 13 int-valued addresses | `a2a5644` 2026-07-12 | 2026-08-26 | **45 days** |
| `observatory.py` folded `ram_mapping` into the exclusion set | `f3b97b4` 2026-07-31 | 2026-08-26 | **26 days** |
| Both together, i.e. an external map able to steer discovery | 2026-07-31 | 2026-08-26 | **26 days** |
| Those addresses under formal quarantine | 2026-08-25 | 2026-08-26 | 1 day |

### Closed by

`scripts/observatory.py` no longer folds `ram_mapping` into `excluded`. Only solve-block
coordinate bytes are excluded — those *are* the archive's cell key, so excluding them is
correct and the guard-of-the-guard tests prove they still are. `ram_mapping` is now
annotation: each scored row carries `"known": true/false` and the receipt records the block
with `"excludes": false`.

The structural half matters as much as the behavioural half. The whole computation moved
into `pre_probe_exclusions(profile, full)`, which returns `(excluded, is_known, excl_log)`
where **`is_known` is a predicate, not a set.** `main()` never receives the mapping
addresses as a container, so `excluded |= mapped` is not a line anyone can add — it would
require re-plumbing a return value. The cost of the fix is 12–17 extra bytes entering a
~1500-byte candidate space. That is what the blindness was buying.

`configs/zelda_gui_tuned.yaml`'s 13-entry block was converted to the string-valued
`quarantined_external_knowledge` form, and the comment claiming a disassembly-verified win
chain was deleted. That line was the whole claim in miniature, and it is not this project's
claim.

---

## Breach path 3 — the test that could not have caught either

`tests/test_zelda_purity_quarantine.py` hardcoded `PROFILE = configs/zelda.yaml`. Five Zelda
configs exist. One was tested. Nothing anywhere asserted that a profile declaring no reward
gets the generic one — if that dispatch were wrong, no test failed. It was wrong, and no
test failed.

Closed by re-parametrising the guard over the surface the contamination actually travels on
(`"zelda" in filename + name`, the same predicate the dispatch used) and adding a tree-wide
sweep sourced from the quarantine blocks themselves, so a third game's future quarantine is
covered without anyone remembering to extend a list.

---

## Four more leaks of the same class, found while fixing these

None of these were in the original adjudication. All are closed.

**1. A second door into the exclusion set: `solve:` blocks.** Closing the `ram_mapping` fold
left `solve.progress.lo` / `.hi` / `y` / `lives` / `level_key` / `area` still (correctly)
excluded, because they are the archive's cell key. A quarantined address copied there is
excluded exactly as before, by a different door — and the new sweep read only `ram_mapping`.
Four profiles were doing this: `configs/legend_of_zelda.yaml` (0x0070, 0x0609),
`configs/zelda.yaml` (0x0070, 0x0084, 0x00EB), `configs/zelda_roomfp.yaml` (0x0070, 0x0084).

A blanket ban would be wrong — independent rediscovery is the quarantine's documented *exit*,
and each of these was genuinely re-derived by measurement. But the mechanism cannot tell
rediscovery from leakage by looking at a number, and a prose comment claiming re-derivation
is not auditable. The rule now: a quarantined address may appear in a solve block only
alongside a machine-readable `rediscovered_addresses:` entry naming the role, the method,
and a **receipt path that exists in the tree** (not under `runs/`, which is gitignored and
would vanish on a fresh clone).

To satisfy that rule honestly rather than by writing a citation, the Zelda coordinates were
**re-derived live during this work**, from this project's own start state, by held-direction
differential with no map consulted:

| byte | rule that nominated it | measurement | rank |
|---|---|---|---|
| `$0084` | reversible on the vertical axis, flat on the horizontal and under no-op | DOWN 134→189, UP 131→85 | 1 of 3 |
| `$00EB` | reversible horizontally over the full hold; the only byte settling to *different* constants across arms | RIGHT 119→120, LEFT 119→118 | 1 |
| `$0070` | a screen-relative x re-bases at screen edges and so fails a whole-hold test by construction; re-run on the within-screen prefix | RIGHT 89→240, LEFT 78→0 | 1 of 3 |

Receipt: `docs/receipts/rediscovery/zelda_coordinates_2026-08-26.json`. The `$0070` endpoints
reproduce `configs/zelda.yaml`'s banked prose exactly, which is what independent
re-derivation landing on the same byte is supposed to look like.

**2. Cross-game address collisions made the first version of that sweep lie.** Zelda's
quarantined `q_dungeon_level: 0x10` flagged Arkanoid's `solve.progress.hi: 0x0010`. RAM
addresses are numbered per ROM; that comparison is meaningless, and a guard that reports
meaningless hits trains people to ignore it. Quarantine blocks now declare
`applies_to_rom:` and are compared only against profiles running that ROM. A block that
omits the key falls back to comparing against everything — the fail-closed direction — and a
test requires the key so nobody hits that strictness by accident.

**3. Two undisclosed diagnostics files were doing breach path 1 with the same bytes.**
`src/diagnostics/worker_debug.py` selected `ZELDA_LINK_X = 0x0070` / `ZELDA_LINK_Y = 0x0084`
under `if "zelda" in self.game`. `scripts/diagnose.py` read the identical pair under
`"zelda" in profile["name"].lower()`. Both quarantined addresses, both arriving by inference
from a title, in files neither the adjudication nor two rounds of review had named. They are
diagnostics and feed no reward or clear claim, which caps the severity — but they are live
instances of the exact pattern the Rust fix was announced to have made impossible.
`src/audio/ram_music.py` carried a whole substring→address table of the same shape
(`{"zelda": 0x0605, "mario": 0x00F7, ...}`), dead since the Rust mixer stopped reading song
bytes. All three are fixed; the audio table is deleted rather than disabled.

**4. There is no way to be sure by hand that this class is gone.** These were found by
grepping, twice, after the breach was declared closed. `tests/test_no_new_name_dispatch.py`
now holds a **shrink-only inventory** of every remaining display-name substring dispatch in
`src/` and `scripts/` — eight sites, each with the reason it survives. The list may lose
entries and may never gain one, and a stale entry is a hard failure, so fixing a site forces
the count down instead of leaving a line behind. Five of the eight are the disclosed SMB
training sites (`reward_id == "mario"` is the drop-in, but `trainer.py`'s
`"mario" in name or "smb" in name` is a strictly wider set that includes
`smb_4_4_micro.yaml`, so it changes training behaviour and needs its own before/after).
Three are offline tooling keyed on file paths. None selects a RAM address.

---

## What the banked results are now worth

### Directly affected: 13 of 14 banked observatory receipts

Every `observatory_v2.json` in `runs/` except `lost_levels` was produced with a folded
exclusion set. Their own logs say so, verbatim: `"region": "profile coordinate + ram_mapping
bytes"`, `"reason": "already-known observables (cell key / verified map); discover NEW bytes
only"`.

The distinction that decides what each is worth is **where the folded addresses came from**:

**Externally sourced — the real breach, four profiles.** These excluded addresses that
originate outside this project, so an outside map genuinely constrained the search:

| receipt | folded | the contradiction |
|---|---|---|
| `runs/megaman/mm2_bootstrap2/` | 9 | The config's own `solve:` comment claims the datacrystal map "is deliberately NOT reused here". Seven of the nine excluded bytes *are* that map — `player_x` 0x0460, `player_x_sub` 0x0480, `player_y` 0x04A0, `player_health` 0x06C0, `boss_health` 0x06C1, `weapon_energy` 0x09C0, `lives` 0x00A8. Not reused as a *source*; used as a *filter*. The receipt cited as the independence proof is the receipt recording the dependence. |
| `runs/gradius/gradius_bootstrap/` | 14 | Six excluded bytes (0x0040–0x0046) are the ones the config states outright "are the datacrystal.tcrf.net/wiki/Gradius_(NES) RAM-map values", never delta-verified. The run's stated blocker was that the stage byte "is undocumented on datacrystal" — a pass hunting an undocumented byte, pre-blinded to six documented ones on an outside source's say-so. |
| `runs/kid_icarus/kid_icarus_bootstrap/` | 17 | Eight are Data-Crystal-documented and explicitly not verified live, in a file whose own section header names the source, while its `solve:` block claims "NO external RAM maps / disassembly". |
| `runs/kungfu/kungfu_bootstrap/` | 11 | The best-behaved file on this list — it runs an honest three-tier legend and even records two debunked datacrystal labels. Two excluded bytes are still CROSS-SOURCED (0x04A5, 0x0058). Honest labelling does not undo the steering: the exclusion set does not read the legend. |

**Verdict for these four: the negative results are void; the positive results stand.**
Anything each run *found* was found by probing and scoring and remains valid. What is not
valid is any claim of the form "the instrument looked and there is nothing there" — for the
excluded bytes the instrument did not look. Concretely, Gradius's headline finding, that its
stage-number byte was NOT FOUND, is **withdrawn**: that pass could not have found six of the
candidate bytes, and the conclusion of absence is unsupported. The same applies to any
"exhaustively searched" phrasing attached to the megaman, kid_icarus and kungfu bootstraps.
Each is cheap to re-run under the fixed instrument, and re-running is the only thing that
restores the negative.

**Self-derived — mechanism defect, not contamination, nine receipts.** `contra`,
`bubble_bobble`, `castlevania`, `double_dragon`, `ducktales`, `excitebike`,
`ghosts_n_goblins`, `kirby`, `metroid` folded only addresses this project derived itself
(each carries `[VERIFIED: ...]` receipts, and five of the configs explicitly state "no
external RAM maps, no disassembly"). Excluding your own verified observables from a
"discover NEW bytes" pass was the stated intent, badly implemented. **These results stand.**
They are narrower than they should have been — a handful of bytes that could have been
re-nominated were not — but no outside knowledge entered, and no purity claim depending on
them is weakened.

### The four CONFIRMED profiles are unaffected

`bubble_bobble`, `castlevania`, `excitebike`, `tetris_b` — the only four profiles on the
roster that can witness their own clear — carry **no external RAM map**. Checked line by
line: their only external-provenance mentions are negative citations ("no external RAM
maps/disassembly"). Two have observatory receipts with self-derived folds, covered above.
**No CONFIRMED clear rests on either breach path.**

### The Zelda archives were already void, and are now void twice

`zelda_roomfp`'s headline number — 443,419 archive cells over 4 × 90-minute 12-worker runs,
zero solution increments — was already struck by the 2026-08-26 adjudication, on the ground
that there was no admissible target signal to build against. The breach adds an independent
second reason: **that profile was silently running the quarantined win predicate the whole
time.** It declared no reward weights and inherited `ZeldaReward` by title. The number was
worth nothing before this document and is worth nothing after it; what changes is that there
are now two separate reasons, and the second one would have voided it on its own.

`legend_of_zelda` could not construct at all (`KeyError: 'y'`, fixed separately), so it
produced nothing to void.

### Nothing in the Learned ledger was built on the quarantined structs

`CLAIMS.md` already recorded, on 2026-08-25, that neither `ZeldaReward` nor `MetroidReward`
appears anywhere else in that file: no clear rate, no honest eval, no capability claim. That
was checked again during this work and still holds. **No Learned-ledger claim is retracted
by this document.** What changed is that the rule now has a mechanism behind it instead of a
paragraph.

---

## What is still open, stated plainly

**`ZeldaReward` and `MetroidReward` remain quarantined-knowledge-derived, and are still
live for the profiles that explicitly ask for them.** `configs/zelda.yaml` and
`configs/zelda_gui_tuned.yaml` declare `reward_id: zelda`, and flipping 0x0672 still returns
`episode_success() == True` for them. That is now an *explicit declaration* rather than an
inference from a title, which is the whole point of the dispatch fix — but the underlying
struct is built almost entirely from quarantined addresses (link x/y, map x/y, hearts,
rupees, triforce, dungeon level, song, the win flag). The `CLAIMS.md` rule stands: neither
struct may produce a Learned-ledger claim. Gutting them was not done here because it is a
real behaviour change on training profiles with no measured benefit today, and it belongs in
its own change with its own before/after.

**D2 — six detector signals reach no production vote.** `entity_wipe`,
`room_fp_transition`, `input_lock`, `lock_release`, `oam_quiesce` and `scene_cut` were built
and are not wired into either the live vote (`tally + coord >= min_signals`) or the offline
harness. Checked during this work: all six do have their own capability tests, so this is
six *working, unwired* signals rather than six unverified ones, and `clear_quorum` prints
every one of them as `NOT_WIRED` in the table it emits at launch. That is disclosed debt with
an accurate readout. Wiring uncalibrated signals into a live clear vote would risk
manufacturing false clears, which is strictly worse than leaving them out; the fix is
calibration, then wiring, in its own change.

**The five SMB name-substring sites still gate training behaviour** (`trainer.py` ×3,
`exploration_controller.py`, `gui/main_window.py`). Tracked by the shrink-only inventory.
None selects a RAM address.

**`scripts/gen_auto_configs.py`'s skip list is now unnecessary but not removed**, so
Dr. Mario, Mario Bros., SMB2 and SMB3 stay off the roster.

---

## The discipline lesson, which is the part worth keeping

Every fix here ships with a test that fails when the mechanism is absent, and each of those
tests was verified by putting the defect back and watching it go red. That discipline caught
a real failure *inside this work*: the first version of the observatory guard asserted
against a copy of the fixed line pasted into the test body, so restoring
`excluded |= mapped` in production left all three of its tests green. The single most
important fix in the batch was, for a while, guarded by nothing. It was found by mutation,
not by review — review had already passed it twice.

The generalisable rule: **a test that reconstructs the production decision inside itself has
tested its own copy.** If the decision cannot be reached without a ROM, that is a signal to
move the decision, not to simulate it. Three of the fixes in this batch (observatory
exclusions, the progress-gate truncation, the camera-static branch) needed exactly that
restructuring before they could be honestly guarded.

Second rule, from breach path 2: **when a filter's stated purpose is about reporting, do not
implement it as blindness.** Tag and show; never subtract from what an instrument is allowed
to consider. The difference cost 26 days of steered discovery and four withdrawn negatives.

---

## Verification performed for this document

- 8 mutations run against the new guards; every one turned at least one named test red:
  restore `excluded |= mapped`; delete a `rediscovered_addresses` block; restore the
  `start_lives == 0` exemption; delete the stasis detector; delete the D1-blind finding;
  restore the `passed = not findings` idiom; delete the truncation call; remove the churn
  floor.
- `scripts/progress_signal_gate.py`'s new frozen-surface detector demonstrated capable on
  the real case (`ninja_gaiden_ii`: "surface frozen from step 746: 454 of 1200 steps at or
  under 14 bytes/step against a live median of 286") and swept before/after across all 43
  constructible profiles with **zero verdict changes** and exactly one profile arming — the
  one D1 identified. Receipt: `docs/receipts/progress_gate_stasis_sweep_2026-08-26.json`.
- Build freshness confirmed by hash: `target/release`, `target/maturin` and both venv
  `.so` paths are byte-identical, and the loaded module exposes `reward_ids()`.
- Every config touched re-parsed as YAML.
