# Process audit: where we stopped following our own methodology

Requested by the user 2026-08-23 after the frozen-actor discovery voided
two experiments. This audit is against OUR OWN rules — the prereg
discipline, the two-ledger standard, the "read, never infer" rule — not
against an ideal.

## The one defect class behind every void this week

Every failed experiment shipped with pre-registration, single-variable
config assertions, adjudicators written before results, and smoke tests
— and still measured nothing. The common gap: **we verified CONFIGS and
never verified the MECHANISM WAS ALIVE at runtime.**

| incident | config checks passed | what was actually dead |
|---|---|---|
| 2-1 attempt 1 | dry-run all green | training consumed 1-3's ladder |
| 1-4 campaign | dry-run all green | backward ladder inert |
| Phase-3 masked v1 | arms single-variable | actor NaN-frozen; eval veto unarmed |
| Phase-3 BOTH arms + options | arms single-variable, smoke ran | actor sentinel-frozen (1e12) |

A biologist would call this running an assay with no positive control.
The configs said the experiment existed; nothing proved the treatment
was operating or that ANY learning was occurring. The frozen-actor case
is the purest form: the profile even documented the freeze as a
controller-managed sentinel ("unfreeze is gate-coupled"), i.e. the base
profile was DESIGNED to be un-runnable standalone — and the process let
us clone it into standalone arms because comments are not enforcement.

## Rule adherence review

- **Prereg / no-rescue**: followed. Verdicts were never renegotiated.
- **Two ledgers / honest protocol**: followed. No contaminated claim
  shipped; every void was caught by receipt forensics.
- **Read, never infer**: violated once late (accepting a FAIL whose
  numbers were the seed fingerprint would have been the inference; the
  fingerprint check caught it — but only because the void run had taught
  us the fingerprint).
- **Smoke tests**: followed in form, insufficient in content: a smoke
  run proved "no crash + throughput," never "learning is happening."
  Also skipped once in a relaunch hurry, and the skipped run crashed on
  the very guard added minutes earlier.
- **Machine-calendar / quiet-bench**: violated once (hazard benchmark on
  a hot machine -> false KILL), fixed with needs_quiet.

## Discarded prematurely / left on the table (dispositions)

1. **Joint-policy transfer signal — the biggest one.** The interference
   falsifier's joint checkpoint scored 0.52 strict on 1-1 vs the
   specialist's 0.43, and we filed the experiment as "naive pooling
   falsified" and never followed the POSITIVE signal. Disposition:
   honest 100-episode eval of that joint checkpoint on 1-1 AND 1-2 —
   two eval runs, cheap, could revise the whole specialists-vs-shared
   picture.
2. **1-4 re-consolidation at 0.633 (30-ep probe)** — never honestly
   evaluated; the winner's-curse rule says it will land lower, but 0.633
   vs banked 0.51 deserves its 100 episodes. Disposition: queue honest
   eval.
3. **Hazard salvage (Go-Explore cell weighting)** — v21's Rank-1 use,
   v23's Castlevania experiment #2 depends on it; built model, weighting
   never implemented. Disposition: part of the Castlevania module.
4. **Substrate experiment (multihead)** — built + tested 08-17, never
   run for want of 1-3/1-4 trajectory collection. Item 1 may supersede:
   if the joint signal replicates, this experiment rises sharply.
5. **latent_cells** — built, unproven, deprioritized by v23 (Rank 3 for
   Contra). Correctly shelved, not prematurely.
6. **SWA over 1-2 peaks** — queued since 08-17, never run. Cheap (4h,
   no training). Disposition: engine action once machine frees.
7. **Options k-collapse tripwire** — prereg named the duration
   distribution as the collapse signal; never logged. Disposition: the
   post-hoc duration measurement is mandatory in the rerun adjudication.

## Claims-integrity sweep after the frozen-actor finding

Campaign-produced claims (1-1/1-2/1-3/1-4 banked rates) are UNAFFECTED:
campaigns run under the controller, which manages the sentinel per
phase; the honest evals measure behavior regardless of training
provenance. Affected and already corrected: Phase-3 training-side
claims (void), options v1 (void). smodice/night2 lineages checked:
their runs either used their own freeze partitions explicitly (night2
dry-run verifies the partition) or were offline (no actor freeze
concept). No further contamination found.

## The fix, mechanical not aspirational

1. **experiment_preflight.py — mandatory positive controls.** Before
   any experiment's full budget: run ~2 iters per arm, then ASSERT from
   artifacts (checkpoints + logs): (a) the ACTOR moved (non-critic
   parameter delta > 0); (b) the treatment mechanism left runtime
   evidence (armed line + activity metric); (c) arms' runtime differs
   only where the design says (control actor moved too, and identically
   configured); (d) no sentinel value outlasts the planned budget.
   An experiment that fails preflight never gets its budget.
2. **Sentinel enforcement in the trainer** (shipped): loud warning when
   a freeze outlasts the run. Preflight turns the warning into a stop.
3. **Fingerprint check in adjudication**: the adjudicator refuses a
   verdict when an arm's eval fingerprint (rate + mean_len per seed)
   is byte-identical to the shared seed's known fingerprint — that is
   evidence the arm did not train, not evidence of a null effect.

## Capability acknowledgment (the second half of the directive)

The workflows are excellent at exactly one family (Mario-engine
platformers: solver seconds-per-level, learned policies 21-51%) and
unproven-to-poor elsewhere, for measured reasons per class. The module
roadmap (MEMORY_ARCHITECTURE + V23_SYNTHESIS) is the response: spatial
mapper (odometer + scene graph), semantic capability engine (SIMCE +
item-effect association learning), and a CLASSIFIER-ROUTER — the
progress-signal gate and TOTALITY class detection choosing which
modules arm per game, so "classify and navigate a given game from
beginning to end" is the standard pipeline rather than per-game craft.

## Shelf dispositions: ANSWERED (engine-run, 2026-08-23)

All six evals ran autonomously the moment the machine freed. Answers:

1. **Joint-policy transfer signal: CLOSED.** Pooled honest 100-episode
   rates: 1-1 = 32/100 (vs specialist 43), 1-2 = 1/100 (vs banked 38).
   The falsifier's 0.52 flicker does not replicate; naive pooling stays
   falsified on both levels, now with full-protocol receipts
   (runs/engine/logs/shelf_joint_*.log). The shared-substrate line does
   not revive on this evidence; the trunk-plus-heads experiment remains
   distinct and unscheduled.
2. **1-4 endpoint: 51/100 pooled — EXACTLY the banked rate.** The 0.633
   probe was winner's curse, as the discipline predicted (0.633 -> 0.51).
   Two further reads: (a) the corrected-ladder re-run did NOT raise the
   rate, so the memo's falsifier resolves — the backward ladder was not
   a material driver of 1-4's 51, and the inert-ladder caveat closes
   with "no measurable effect either way"; (b) unlike 1-2's continued
   PPO (31 -> 8), 1-4's consolidation endpoint HELD its banked level —
   consolidation and raw continued training age very differently, which
   sharpens Finding 2 of the options negative.
3. Remaining from the audit shelf: SWA (needs a script; still queued),
   hazard->Go-Explore weighting (Castlevania module dependency).
