# Unwitnessed semantics in the ENGINE — `nes_core/src/rewards.rs`, 2026-08-27

Follow-up to `UNWITNESSED_SEMANTICS_2026-08-27.md`, which swept `configs/`
(994 entries: 7 quarantined, 24 downgraded, 963 kept) and named its own scope
limit in one sentence:

> "Quarantining the YAML retracts the DOCUMENTATION claim, NOT the Rust
> constant."

That gap is this pass. The YAML quarantine was clean; the engine was not.

## The discriminant, unchanged

> **An assertion of semantics tied to an event this project has never
> witnessed is injected knowledge, because it could not have been measured
> here.**

Two corollaries carried over verbatim, and both did real work:

* **An honest null is not a breach.** Kid Icarus's "held 0 across 300k+
  frames" and Kung Fu's 32/32-banked-cells zero are measurements of absence,
  correctly reported. They stay. What does not stay is a false-positive
  *rate* asserted on top of a null.
* **Over-withdrawal is its own defect.** Contra's `clear_screen == 255`
  sentinel, Gradius's "NO byte is trustworthy as the stage index yet",
  Ghosts' disabled `stage_addr`, Bubble Bobble's `enemy_count_addr` and Kung
  Fu's opt-in `$04A5` are already the right discipline and were left exactly
  as they were.

## What changed, and what deliberately did not

**No reward arithmetic was changed. Not one address, threshold or weight
moved.** The disposition was provenance + guard, chosen over disarmament and
stated here explicitly because the alternative was available and rejected:
three of the tagged constants are *active shaping terms* on unwitnessed
identities (Mega Man `boss_damage` 5.0 / `boss_killed` 75.0, Castlevania
`boss_damage` 3.0 / `boss_killed` 50.0, Kid Icarus `boss_killed` 500.0), and
gating them off behind the disabled-by-default pattern the engine already
uses elsewhere would be a *behaviour* change. That belongs in its own
separately-approved step, not smuggled into an annotation pass.

| | count |
|---|---|
| constants covered by a `PURITY: UNWITNESSED-EXTERNAL` tag | 27 across 11 games |
| tag blocks written (some cover a sibling group) | 22 |
| retracted clauses pinned dead in this layer | 8 |
| SMB constants touched | **0** |
| reward arithmetic changed | **0** |
| new Rust guard tests | 5 |
| new Python guard assertions | 126 |

## The structural finding

For **Kid Icarus** (`$0130`) and **Double Dragon** (`$0030`), the config
sweep retracted a specific sentence — and that exact sentence survived
**verbatim** in the Rust:

* `configs/kid_icarus.yaml` quotes and retracts *"an unambiguous stage clear
  (never a false positive)"*. `rewards.rs` still said it.
* `configs/double_dragon.yaml` retracts *"never a false positive; if wrong,
  it simply never fires"* and notes the address *"hard-defaults to 48 =
  0x0030 in nes_core/src/rewards.rs, so this key is documentation, not the
  wiring"*. The wiring still carried the retracted claim.

So the configs-only sweep did not merely leave the engine uncovered — it
moved the documentation while the executing layer stayed put, and the two
layers ended up **disagreeing in writing**. Double Dragon carried its clause
in *two* places (the arm header and again in `compute()`'s win branch), so
retracting only the header would itself have been a half-fix.

## The sharpest breach: Castlevania's `DRACULA_STAGE`

Most tagged constants cite an external RAM map. This one did something
different in kind — it resolved the question by **reading the ROM's
disassembled code** and quoted an instruction as proof:

> "Dracula's stage is 0x12 (18) — proven by the game's own code: LoadStage
> indexes a 22-entry StageDataTable and the ending is a special-case
> `cmp #$12`."

That is precisely the Tier-3 line: a question answered by knowing the game.
The *proof* is withdrawn; the *value* is left in place, because removing it
is a behaviour change and its predecessor (`>= 17`, from a wrong "0-indexed
block 18" inference) was worse. 0x12 is now recorded as believed, not proven.

Also newly named, because address-shaped sweeps structurally cannot see it:
**DuckTales' win predicate is not an address at all.** It is the
game-knowledge pair *$1,000,000 boss treasure* vs *$50,000 largest gem*,
encoded as two dollar figures — an asserted false-positive rate for a boss
defeat with zero observations anywhere in this repo.

## I measured rather than argued

`scripts/probe_unwitnessed_bytes.py` drives each ROM from its shipped start
state with **uniform-random actions and no game knowledge**, and reports only
the distinct values each byte took. Receipt:
`docs/receipts/purity/rust_unwitnessed_probe_2026-08-27.json`.

| game | address | result |
|---|---|---|
| Castlevania | `$01A9` boss health | held **64**, 1 distinct value / 20k steps |
| Castlevania | `$0044` health | **65** distinct, 0..64 — a draining bar |
| Castlevania | `$0045` contested | **9** distinct over the same window |
| Castlevania | `$0071` hearts | 11 distinct, 0..10 |
| Castlevania | `$0028` stage | 0 -> 1 (reproduces the witnessed chain) |
| Kid Icarus | `$0130`, `$006B` | both held **0** / 15k steps |
| Metroid | `$0098` `$0099` `$007A` `$007B` | all held **0** / 15k steps |
| Zelda | `$0609`, `$0672` | held **1** and **0** / 600 NOOP + 60k random |
| Kung Fu | `$0058` | held **0** / 15k steps |
| Punch-Out | `$0001`, `$000A` | both held **0** / 15k steps |
| Punch-Out | `$0398` opp HP | **40** distinct, 0..96 — the control |
| Mega Man 2 | `$06C1`, `$06C0`, `$00A8` | each a **single** value / 15k steps |

Three of these deserve calling out because they do not simply confirm the
suspicion that prompted them:

1. **The Punch-Out null has a built-in control.** `$0001` and `$000A` never
   moved *while `$0398` took 40 distinct values over the same window* — so
   the bout was genuinely being fought, punches landed, and the outcome
   bytes still never moved. That is a much stronger null than "nothing
   happened."
2. **A hypothesis of mine was refuted and is reported as refuted.** Zelda's
   `$0609` held 1 across 600 NOOP and 60,000 random steps, with 0 frames
   equal to `SONG_ENDING` and 0 frames setting `SONG_GANON`. The feared
   spurious 30,000-point `ganon_reached` payout is **refuted** for this
   start state, not merely unobserved.
3. **A cited number did not survive my own re-measurement.** The inherited
   sweep described Castlevania's `$0071` as taking "5 values in 5..14,
   rising only". My probe measured **11 distinct values across 0..10** — it
   both rises *and* falls. The annotation records what I measured, and
   explicitly declines to assert "rising only". (Falling is what sub-weapon
   ammo does when thrown, so the constant still stands; the *claim* about it
   was wrong.)

Two measurements settle a dispute `configs/castlevania.yaml` flagged as
CONTESTED but could not resolve. `rewards.rs` asserted in a comment that
"$0045 is actually Simon's real HP" three lines below `RAM_PLAYER_HEALTH =
0x0044`. **The constant is right and the comment was wrong** — $0044 is the
smooth 0..64 bar, $0045 takes 9 values. Separately, the config objected that
`RAM_HEARTS = 0x0071` "has no receipt anywhere under docs/ or runs/"; it has
one now. Both are comment fixes; neither constant moved.

## Did any banked claim break?

**No.** Every tagged constant is *unfired*: not one sits under a quoted
number. All belong to games with no witnessed clear, and **no boss defeat has
ever been witnessed here on any game.** SMB needed no change at all, so the
standing concern — that a behaviour change to SMB's reward invalidates banked
runs — does not arise.

SMB's block is now *positively* marked `PURITY: WITNESSED`, with a note that
its constants fired 32 times in one cold-boot tape with `state_loads=0` and a
rendered ending frame. Recording what **is** earned is half the discipline:
without it, the next sweep sees only unwitnessed tags and over-withdrawal
starts to look like rigour.

## The guard

`WIN_WITNESS_LEDGER` in `rewards.rs` classifies **every** reward arm's
`episode_success()` as `Witnessed` / `Unwitnessed` / `Disarmed`, exported to
Python as `nes_core.win_witness_ledger()`. It is pure data — nothing in the
reward path reads it, so a row changes no arithmetic. Its job is that a
success reported by an arm resting on semantics nobody here has ever seen
**cannot be reported silently**.

* `Witnessed` (4): mario, excitebike, bubble_bobble, tetris
* `Unwitnessed` (9): zelda, castlevania, metroid, punch_out, kung_fu,
  ducktales, kid_icarus, double_dragon, ghosts
* `Disarmed` (4): generic, mega_man, contra, gradius

Ghosts is in the unwitnessed set for a different reason from the rest and the
row says so: its *addresses* are earned, but its only live win path is the
Path-B positional inference, whose thresholds no witnessed stage transition
has ever calibrated. Mega Man is `Disarmed` for its **win only**, and its row
discloses that `boss_damage`/`boss_killed` remain live on `$06C1` — otherwise
"disarmed" would read as "nothing depends on the unwitnessed byte", which is
false.

Five Rust tests keep the ledger from becoming decoration: totality over
`REWARD_IDS` (a new arm cannot land unclassified); every `Unwitnessed` arm is
**driven to a reported success** through the byte its row names; every
`Disarmed` arm is driven hard and proven unable to; and the `Witnessed` set is
pinned in both directions.

The Python guard adds 126 assertions, including one that compares the
**compiled** ledger against the source — so a stale `.so` (a standing failure
mode here) becomes a loud test failure instead of silently voiding every
measurement taken afterwards.

## Verified by actual revert

Per `scripts/verify_unwitnessed_guard_by_revert.sh`, receipt in
`docs/receipts/purity/guard_revert_verification_2026-08-27.txt`:

| arm | failures |
|---|---|
| baseline (fixed tree) | **0** |
| full revert to pre-annotation `rewards.rs` | **63** |
| half-fix: only Punch-Out `$0001` reverted, `$000A` left tagged | **2** |
| sibling group: Metroid's one shared tag deleted (4 bytes + threshold) | **10** |
| over-withdrawal: SMB `RAM_FLOAT_STATE` moved `0x001D` -> `0x001E` | **1** |

The predecessor's bar was 14/14 on full revert and 7 on a half-fix. Full
revert clears it at 63. The single-constant half-fix lands at 2 rather than 7
because this registry is keyed **per constant** rather than per pair — the
finer granularity is why reverting one member of a pair is caught at all —
and the sibling-group arm (10) is the like-for-like comparison.

## Named follow-ups, not done here

1. **Gate the three active boss-shaping terms** (Mega Man, Castlevania, Kid
   Icarus) behind the disabled-by-default pattern the engine already uses.
   This is a behaviour change and needs its own approval.
2. **`depth_tracker.rs` and `pool.rs`** duplicate SMB constants and carry
   Zelda depth-key literals. Positional only, no win/boss semantics attached,
   so the discriminant does not bite — but the duplication is a maintenance
   smell worth closing.
3. **Punch-Out `$0002`** (Glass Joe = index 0) is tagged but cosmetic; it
   feeds the `level_id` label only.
