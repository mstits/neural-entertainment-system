# Metroid profile — external-knowledge quarantine

**Date:** 2026-08-10
**File quarantined:** `configs/metroid.yaml`
**Guard:** `tests/test_metroid_purity_quarantine.py` (15 tests)
**Behaviour change:** none. Every functional value in the profile is
byte-identical to `HEAD` (verified key-by-key: `solve`, `reward_weights`,
`action_space`, `curriculum`, `reinforce`, `ram_mapping`, `frame_skip`,
`max_episode_steps`, `start_state_path`, `name`, `rom_hashes`).

## What was contaminated

Two pieces of external RAM-map knowledge were living in the working
profile, unlabelled, where they read as our own measurement:

1. **The cartridge item block.** `ram_mapping` carried commented entries
   for `missiles 0x6879`, `missile_cap 0x687A`, `energy_tanks 0x6877`
   and `item_flags 0x6878`, the last with a full claimed equipment
   bitfield (bit → morph ball / bombs / high jump / long beam / screw
   attack / varia / wave / ice). None of it was ever observed on this
   ROM by us — the bytes are unreachable through `get_ram`, which masks
   to `$07FF`, so they could not have been. The same source supplied the
   claim that the upper nibble of `0x0107` counts filled energy tanks.
   The block header attributed the whole mapping to an external map
   ("VERIFIED against datacrystal + empirical differential scan"),
   which laundered the external half through the empirical half.

2. **The pairing justification in `solve:`.** The `progress` comment
   justified pairing the fine camera byte with the coarse screen counter
   by appeal to an outside prescription ("the lore-prescribed camera-byte
   + screen-counter pairing"). The *measurement* underneath it is ours
   and is sound — `0x0051` clamps at 220, `0x0050` keeps stepping — so
   the appeal was decoration on a real result, but it is exactly the
   sentence that lets external knowledge become load-bearing later.

## What was done

- Added a top-level `quarantined_external_knowledge:` block holding the
  four cartridge addresses and the two unverified semantic claims, under
  renamed `q_*` keys, with `provenance`, `status: UNVERIFIED_EXTERNAL`
  and a `rediscovery_rule`.
- **The rediscovery rule:** a byte named in that section is dead until
  it is re-derived from scratch by our own differential protocol on this
  ROM — drive the input from the verified start state, watch the byte
  move, record the wrap / reversibility / saturation-and-persistence
  evidence in a receipt. A byte that passes enters the profile as a *new*
  entry under `ram_mapping` or `solve:` with its own `[VERIFIED: ...]`
  tag. The quarantined entry is never promoted in place and never cited
  as the reason a byte was chosen.
- Quarantined values are **strings, not ints**, so a tool that folds a
  name→address dict the way `scripts/observatory.py` folds `ram_mapping`
  (`int(a)` over `.values()`) raises instead of silently consuming banned
  addresses.
- Scrubbed every reference to the quarantined addresses out of the
  working blocks (`description`, `ram_mapping`, the `reward_weights`
  note, `solve:`). The disarmed-channel note now says the reward may be
  repointed only at re-derived, receipted bytes — the quarantined
  addresses may not be pasted in as the target.
- Replaced the external-prescription wording in `solve.progress` with
  the measurement it was decorating.
- Scrubbed one item-lore comment in `action_space` (`"morph ball (once
  Maru Mari obtained)"` → a plain button label). Found by the new lint,
  not by inspection.

## Open re-derivations (flagged, not quarantined)

- **`solve.area = 0x0074`.** Held constant `0` through every scripted
  probe *and* the whole 4-minute solve, so what has been measured is its
  inertness, not that it indexes a region. Harmless as the leading
  cell-key term — a constant term partitions nothing — but the "region
  index" label is not re-derived. Noted inline in the profile. Left
  wired: removing it would be a behaviour change, and the spec adjudicated
  only the item block and the pairing comment.
- **`ram_mapping.samus_health_hi = 0x0107`.** The tens-digit behaviour is
  independently verified (start 3 = 30 energy, decrements under damage)
  and that verified part alone is what `solve.lives` rests on. The
  filled-tank-nibble reading of the same byte is quarantined.

## Deliberately not edited

`runs/metroid/metroid_receipt.json` and its copy at
`docs/receipts/games/metroid_receipt.json` both contain the phrase *"the
room/screen counter the lore prescribes pairing the camera byte with"*
in `step2_discovery`, contradicting the same receipts' `method` field
("no external RAM maps or disassembly"). Both are left byte-intact: a
receipt is a record of what was done, and editing one after the fact to
look cleaner is a worse integrity failure than the leak it would hide.
The leak is logged here instead, and `configs/metroid.yaml` now cites
that receipt as a historical record rather than as authority for the
pairing.

## Guard

`tests/test_metroid_purity_quarantine.py` pins: the block and its rule
survive; no external-provenance token and no quarantined cartridge
address appears outside it; the quarantined values cannot be folded into
an address map and no source file reads the key; `ram_mapping` still
folds to system-RAM ints for the discovery tools; `solve:` reads only
verified system RAM at the receipted addresses; the in-tree
`0x0008|0x0065` persistence-probe rejection note is preserved; and the
profile still meets the trainer boot contract. Both text lints are
mutation-checked — a contaminated copy is fed in and must fail.
