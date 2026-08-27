# Engine purity — the layer the config quarantine could not reach

**2026-08-27.** Third purity sweep of the day. The first two swept
`configs/`. This one swept the layer that executes.

---

## The finding, stated plainly

The config quarantine retracted **documentation** claims. The executing
layer kept its own copies, and **nothing in the tree could tell.**

A 994-entry sweep of `configs/` (commits `0557896`, `bd5d3d6`) quarantined
7 entries, downgraded 24 and kept 963. It named its own scope limit in its
commit message:

> "Quarantining the YAML retracts the DOCUMENTATION claim, NOT the Rust
> constant."

That limit was not a footnote. It was the whole exposure. For **Kid Icarus
(`$0130`)** and **Double Dragon (`$0030`)** the YAML retracted a specific
sentence and *that exact sentence survived verbatim in the Rust*:

* `configs/kid_icarus.yaml` quoted and retracted "an unambiguous stage
  clear (never a false positive)". `rewards.rs` still said it.
* `configs/double_dragon.yaml` retracted "never a false positive; if wrong,
  it simply never fires" and noted the address "hard-defaults to 48 =
  0x0030 in `nes_core/src/rewards.rs`, so this key is documentation, not
  the wiring." The wiring still carried the retracted claim.

So the configs-only sweep did not merely leave the engine uncovered. **It
moved the documentation while the executing layer stayed put, and the two
layers then disagreed in writing** — with the wrong one running.

---

## The numbers

| | count |
|---|---|
| RAM-address constants swept in `nes_core/src/` | **134** |
| — of them in `rewards.rs` as `const RAM_*` | 109 |
| Non-address constants carrying semantics (magic values) | 23 |
| Sweep findings ruled SEMANTIC-and-UNWITNESSED | 21 |
| Individual constants those findings cover, now pinned | **27**, across 11 games |
| Constants **annotated** with a provenance tag | **27** |
| Reward arithmetic changed | **0** |
| Banked claims retracted | **0** |
| Python sites corrected | 12 |
| Enforced quarantined-address sites, now disclosed | 24 |

**Behaviour changed: zero in the reward engine.** Diffing non-comment
lines shows no existing executable line in `rewards.rs` was removed or
modified. SMB's block is byte-identical and is now *positively* marked
`PURITY: WITNESSED`. One behaviour change did land **outside** the engine —
the ROM resolver fix below — and it is called out separately because it is
a correctness fix, not a purity retraction.

---

## The engine came back mostly clean, and that is the good outcome

This is worth saying without hedging, because a third sweep that finds
little is evidence the first two worked.

**Not one of the 27 unwitnessed constants is load-bearing for a banked
number.** Every one is *unfired*: none sits under a quoted result, all
belong to games with no witnessed clear, and **no boss defeat has ever been
witnessed on any game in this repo.** That is precisely why this was cheap
to fix now and would have been expensive after one of them fired.

The witnessed side is legitimately earned and was left alone: SMB's block
(which fired 32 times in the single cold-boot tape with `state_loads=0` and
a rendered ending frame), Castlevania's `$0028` level key, Bubble Bobble's
`$0401`, Excitebike's section chain, Tetris's line bytes, and Punch-Out's
`$0398` — the one address in the unwitnessed-game set with real numbers
behind it.

Several blocks are already models of the right discipline and were
explicitly **not** changed: Contra's `clear_screen` 255 sentinel, Gradius's
"NO byte is trustworthy as the stage index yet", Ghosts' disabled
`stage_addr`, Bubble Bobble's `enemy_count_addr`, Kung Fu's opt-in
`$04A5`, and Contra's knowingly-dead `$002C` with the working path named
beside it. An assertion that honestly records a null is a measurement, not
a breach.

**Over-withdrawal was guarded as the equal-and-opposite defect.**
`test_smb_constants_are_untouched_and_untagged` fails if anybody strips or
tags SMB's earned constants to look rigorous. Mutation-tested: moving
`RAM_FLOAT_STATE` fails 1 assertion; tagging SMB's `RAM_LIVES` as
unwitnessed fails 4.

---

## The sharpest individual finding

`DRACULA_STAGE = 0x12` did not merely cite an external RAM map. It
justified itself by **quoting the ROM's own disassembled instruction** —
"the ending is a special-case `cmp #$12`". That is the Tier-3 line exactly:
a question resolved by knowing the game. The proof is withdrawn; the value
stays (removing it is behaviour) and is recorded as *believed, not proven*.

Runners-up: Zelda's whole win chain sourced in-code to the aldonunez
disassembly; Metroid's whole win chain plus a hardcoded 32-hit boss-death
threshold, self-declared "from the disassembly + Data Crystal, NOT yet
fight-verified"; and DuckTales' win predicate, which is game knowledge
encoded as **two dollar figures** ($1,000,000 boss treasure vs a $50,000
red diamond) rather than an address — which is exactly why every
address-shaped sweep had missed it.

---

## Measured rather than argued

Four rulings were re-derived on this repo's own emulator, with no game
knowledge used, and one of them cut against the hypothesis that produced
it:

* **Punch-Out** has a built-in control. `$0001` and `$000A` each held 0
  across 15,000 random-action steps while `$0398` took **40 distinct
  values** over the *same* window — so the bout was genuinely being fought,
  punches landed, and the outcome bytes still never moved.
* **Metroid** `$0098/$0099/$007A/$007B`: all held 0 across 15,000 steps.
* **Kid Icarus** `$0130`/`$006B` and **Castlevania** `$01A9`: measured
  nulls (`$01A9` held 64 across 20,000 steps — the "reads 64 idle" half is
  now genuinely ours; the identity is not).
* **Zelda** `$0609` held 1 across 600 NOOP and 60,000 random-action steps,
  with **0 frames** setting bit `0x02` and **0 frames** equal to `0x10`.
  This **REFUTES** a spurious-30,000-point payout hypothesis formed earlier
  in the pass. The refutation is reported, not the suspicion.

And a number inherited from the sweep did not survive its own probe:
Castlevania's `$0071` was described as "5 values in 5..14, rising only";
the re-measurement found **11 values across 0..10 — it falls too.** The
annotation records what was measured and explicitly declines to assert
"rising only".

---

## What is now mechanically checked

Four mechanisms, all mutation-tested by actual revert.

**1. A provenance-tag registry** (`tests/test_rust_unwitnessed_semantics.py`,
126 assertions). Each of the 27 constants must still be declared *at its
recorded value* — so a behaviour change that moved an address fails here —
and must carry a tag stating what it ASSERTS, that there is NO WITNESS, and
what would EARN IT.

**2. A retraction lint.** The 8 sentences the sweeps withdrew are pinned
dead, so layer drift cannot recur silently.

**3. `WIN_WITNESS_LEDGER`** — 17 rows, one per reward arm, classifying every
`episode_success()` as Witnessed / Unwitnessed / Disarmed and exported as
`nes_core.win_witness_ledger()`. Five Rust tests drive each arm through the
byte its own row names; the Disarmed arms are driven hard and proven unable
to report success.

**4. A tree-wide derived scanner** (`tests/purity_engine_scan.py`, wired
into `make test` via `purity-check`). It derives quarantined addresses from
the `quarantined_external_knowledge:` blocks themselves and ownership from
the source's own dispatch table, so neither can drift from what it guards.
24 enforced sites, every one disclosed in
`docs/purity/engine_quarantine_disclosures.yaml` with a ratchet that only
moves down.

### Anti-vacuity: what the guards were *actually* mutated against

Seven vacuous gates have shipped in this project, so the bar is revert, not
assertion. The predecessor sweep set it at 14 failures on full revert and 7
on a half-fix.

| mutation | result |
|---|---|
| full revert of all 27 tags | **63 failures** |
| sibling-group revert | 10 |
| **every one of the 27 tags deleted individually** | **27/27 caught** |
| SMB over-withdrawal (move `RAM_FLOAT_STATE`) | 1 |
| SMB over-withdrawal (tag an earned constant) | 4 |
| ledger status flip `zelda` → Witnessed, stale `.so` | caught |
| ledger predicate re-pointed, stale `.so` | caught |
| retracted clause restored in its own arm header | caught (2) |
| retracted clause restored far from any marker | caught (2) |
| Contra's disarmed win re-armed (`clear_screen` 255→3) | `make purity-check` exits 2 |
| renamed alias `GANON_DEFEATED = 0x0672` in `src/` | caught |
| subscript evasion `r[0x0672]` | caught |
| dropping one row of a sibling pair | caught (the half-fix case) |

---

## Three defects found in the guards themselves, and fixed before landing

The guards shipped earlier in the day were verified independently, and two
of them did not hold. Recording this is the point of the exercise.

**1. The half-fix guard was largely vacuous — 79% escape rate.**
`TAG_LOOKBACK_LINES = 60` searched a fixed 60-line window above each
declaration, so a *neighbouring* constant's tag satisfied it. Deleting a
constant's entire 9-line provenance block passed **126/126**. Measured
across the registry: **19 of 24 rows escaped**, including every headline
retraction — the Zelda, Metroid and Castlevania win chains.

The fix removes proximity from the rule. A tag is now the constant's **own
contiguous comment block**. Four blocks legitimately tag a group of
adjacent constants; those are handled by **naming** the group
(`[group: metroid_win_chain]`) in both the shared header and each member,
so deleting either half fails. Result: **27/27 caught, up from 5/24.**

**2. The stale-`.so` guard did not do what its docstring claimed.** It
compared only the *set of reward ids* between the compiled binary and the
source. Any edit that neither adds nor removes an arm — a flipped status, a
re-pointed predicate, a rewritten basis, i.e. exactly the edits this sweep
makes — left it green against a stale binary. In a repo where a stale dylib
silently voiding every measurement is a documented failure mode, that
under-delivered badly. It now compares **all three fields on all 17 rows**.
(For the record: the binary had *not* drifted — 0 real mismatches. The
guard simply could not have told.)

**3. The retraction lint had a blind zone covering all 8 clauses at exactly
the position each one used to live.** A 50-line marker lookback treated any
occurrence near a retraction as a quotation — but an arm header is one long
comment run *containing its own retraction*, so restoring the sentence a
few lines below the marker passed silently, in 8/8 cases. Two candidate
rules were tried and measured before landing a third: comment-run scoping
failed (the header is one run), and quote-span detection failed (backtick
parity over a long run creates bogus spans that swallow unquoted prose).
The rule that holds is an **exact quotation census** — each clause records
how many times the sweep quotes it, so a new occurrence fails wherever it
lands.

---

## Two more, found outside the engine

**The tree-wide scanner detected this repo's naming convention, not the
address.** It required an identifier matching `RAM_*` / `*_addr`, or a
subscript on a variable literally called `ram`. So `GANON_DEFEATED =
0x0672`, or a read through a variable called `r`, walked straight through
`make purity-check` while reading the quarantined byte live. Three real
in-tree sites were invisible for that reason — including
`src/gui/play_window.py`, production code carrying two quarantined Zelda
addresses, which had *also* evaded `test_no_new_name_dispatch.py` because
that guard is a conjunction (`has_addr AND has_token`) and the file names
no game.

Detection is now address-shaped. False positives were measured rather than
assumed: flagging every literal would report **354 sites across 50 files**;
the three shapes actually used report 96, of which **24 are enforced**. Two
narrow negative allowlists keep the known false-positive classes out
(bitmask names like `_BTN_UP = 0x10`; quantity names like `PRG_BANK`), and
hardware-fidelity modules are exempt by standing rule — which is what let
the identifier-prefix requirement go. `max_enforced_sites` moved 17 → 24,
and the file records *why*: not because breaches landed, but because the
detector got stronger. Every one of the 7 newly disclosed sites pre-dates
the change.

**The ROM resolver's franchise-collision check was one-directional — and
this one is a real correctness bug, not a documentation issue.** The
existing check compared `candidate_markers - canonical_markers`, a set
difference that is empty whenever the *candidate* carries no installment
marker. It blocked original ← sequel and left sequel ← original wide open.
Measured against the real canonical names in `configs/`, **five profiles
bound the wrong dump**:

    lost_levels      -> Super Mario Bros. (World).nes
    castlevania_iii  -> Castlevania (USA).nes
    ducktales_2      -> DuckTales (USA).nes
    ninja_gaiden_ii  -> Ninja Gaiden (USA).nes
    double_dragon_ii -> Double Dragon (USA).nes

`lost_levels` is **on the witness ledger**. The comparison is now symmetric,
with care not to overcorrect: a marker glued to the name (`megaman2.nes`, a
legitimate retag) still matches, while a dump that does not encode the
installment at all is rejected. Five new tests cover the direction that had
none.

---

## The reusable finding

> **A quarantine that covers only the declarative layer is incomplete by
> construction.**

This generalises well past this repo. Wherever a claim is written in one
place and executed in another — config vs code, schema vs migration, spec
vs implementation, policy document vs enforcement rule — retracting it in
the declarative layer *feels* like retracting it, produces a clean diff, and
leaves the behaviour untouched. The failure is silent by construction,
because the layer that was corrected is also the layer everyone reads.

Two corollaries earned the hard way today:

1. **The retraction makes it worse before it makes it better.** Before the
   config sweep, both layers carried the same wrong claim — consistent, and
   discoverable by reading either one. After it, they disagreed, and the
   authoritative-looking one was wrong. A partial retraction is not a
   partial fix; it is a new defect class.

2. **The enforcement must be derived from the declaration, not listed
   beside it.** A hand-maintained list of "places the quarantine also
   applies" is the same defect one level up. The scanner that works derives
   its addresses from the quarantine blocks and its ownership from the
   dispatch table, so neither can drift from what it guards.

And one about guards generally, which is why three of them were rewritten
today: **a guard that locates evidence by proximity is a guard that can be
satisfied by a neighbour.** All three defects above are the same shape — a
fixed-size window (60 lines, 50 lines) or a naming convention standing in
for the thing actually being checked. Scope a check to the artifact it
guards, or mutation-test it until it bites.

---

## Gates

* `.venv/bin/pytest tests/ -q --timeout=120` → **5807 passed, 30 skipped,
  3 xfailed, 1 failed** — the known-environmental
  `test_night2_runner.py::test_dry_run_passes_live`, left alone. No new
  failures.
* `cargo test --lib` in `nes_core/` → **664 passed, 0 failed.**
* `make build` re-run and the venv `.so` refreshed; dylib and `.so` are the
  same bytes, and a test now compares the *compiled* ledger against source
  so a stale binary fails loudly.
* `make purity-check` green, and it now runs the Rust `win_witness_guard`
  tests too — `make test` previously ran `cargo check`, which compiles but
  executes nothing, so the five tests that actually drive each reward arm
  were in no default gate.
