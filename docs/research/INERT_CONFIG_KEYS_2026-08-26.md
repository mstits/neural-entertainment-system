# A config key that lies: `symlog_rewards` is inert under `vanilla_ppo`

**Date:** 2026-08-26
**Status:** verified in source; 8 banked experiment configs affected.
**Severity:** the runs are not invalidated — but their configs misdescribe them,
and the class of defect is one this project has now hit twice in two days.

## The finding

All eight v27 and v28 seed configs declare:

```yaml
reinforce:
  symlog_rewards: true
```

It does nothing in those runs.

- `symlog_rewards` is read once, at `src/training/trainer.py:1216`.
- It is consumed at exactly one site, `trainer.py:4505`.
- That site is inside `_reinforce_update`, which begins at `trainer.py:4424` —
  the GA path.
- `_run_vanilla_ppo` begins at `trainer.py:4820`. Neither
  `src/training/ppo.py` nor `src/training/ppo_updater.py` contains the string
  `symlog` at all.

All eight configs set `trainer_mode: vanilla_ppo`. **Their rewards were never
symlog-transformed.**

## How it was found

Not by a schema check — by the F1 instrument agent building
`scripts/critic_explained_variance.py`. Its first implementation honoured the
config key and applied symlog before computing explained variance, and measured
**EV = −0.21** (mean V 801 against symlogged R 128). On the raw reward scale the
same checkpoint scores **+0.38**.

Explained variance is offset-invariant but **not scale-invariant**, so applying a
transform the training run never applied is silently fatal to the measurement —
it would have produced a confidently wrong F1 verdict, on the exact gate that
decides whether v29 spends 8 hours. The agent caught it by checking whether the
key was actually reachable rather than trusting the config, and pinned it with a
regression test.

## Why this is not "just a stale key"

This is the second time in two days that a treatment which *looked armed* was
*inert*:

- **ReDo** (v27/v28): implemented, tau-swept, preflight-verified as armed,
  logged `[redo] ENABLED` — and performed **zero recycles across ~2,000
  per-iteration checks in all 8 runs**. v27's own verdict text quoted a claim
  that ReDo "mathematically guarantees all 48k parameters were active." It
  guaranteed nothing; it never fired.
- **`symlog_rewards`** (v27/v28): declared in all 8 configs, consumed by none of
  them.

Both were found only because someone went looking at the mechanism rather than
the declaration. `src/training/config_schema.py` catches *unknown* keys loudly
(`unknown reinforce key ... NOT consumed by the trainer`), which is good — but it
has no concept of a key that is known, spelled correctly, registered, and
**inert for the selected `trainer_mode`**. That is the gap.

## What was done

The eight configs are **annotated, not edited**. The value stays `true` because
those runs are banked and their configs must keep describing the file they
actually ran with. The comment records where the key is read, where it is
consumed, why that site is unreachable under `vanilla_ppo`, and an explicit
warning against "fixing" it by making `vanilla_ppo` honour the key — doing so
would change the reward scale and silently break comparability with every banked
v21–v28 number.

## What was NOT done, and why

The real fix is a **mode-aware inertness check** in `config_schema.py`: a
registered key that the selected `trainer_mode` cannot reach should warn as
loudly as a misspelled one. That was not implemented here because
`src/training/config_schema.py` and `tests/test_config_schema.py` both carry
uncommitted edits from a concurrent workflow, and editing a file another lane is
mid-change on is how two correct fixes become one broken merge.

It is a small, well-specified change and should be done when the tree is quiet:
build the inert set per mode from the same source-parsing machinery
`consumed_reinforce_keys_from_source` already uses (it re-derives consumed keys
by parsing `<cfg>.get("key")` sites), extended to attribute each consumption site
to its enclosing method, and warn when a set key's only consumers are unreachable
from the configured mode.

## Scope check

`symlog_rewards` is the only key confirmed inert here. This document does not
claim it is the only one — a full audit needs the mode-aware checker above. The
honest statement is: one key was checked because a measurement tripped over it,
and it was inert. Nobody has swept the rest.
