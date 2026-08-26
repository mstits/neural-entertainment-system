# Field-literature dimension — peak-instability investigation

Scope: external literature on late-training PPO collapse after an early
peak, mapped against the 8-run v27/v28 phenomenon. WebSearch/WebFetch
citations and quotes are in the parent agent's response; this directory
holds the one piece of NEW computation this dimension produced (everything
else is cited reasoning against data other siblings already extracted).

## Files

- `feature_rank_probe.py` — loads every saved checkpoint
  (`vanilla_ppo_iter_000{10..240}.pt`, all 8 runs) via the same
  `build_tile_policy_from_checkpoint` dispatch and fixed 1536-observation
  batch the `checkpoint_autopsy` sibling validated, forwards them through
  `fc1->norm1->SiLU->fc2->norm2->SiLU` (the shared trunk feature the
  actor/critic heads read), and computes the effective rank of that
  representation two ways: `srank_delta01` (Kumar et al. 2020 / Moalla et
  al. 2024's threshold-rank, delta=0.01) and the participation ratio.
  Output: `feature_rank.csv` (192 rows, 8 runs x ~24 checkpoints).
- `correlate_rank_entropy.py` — correlates `srank_delta01` against
  iteration and against `ppo_entropy` (from each run's `metrics.jsonl`),
  and reports srank at the honest-peak iter (from the `curriculum_ladder`
  sibling's `peak_iter_bestjson`) vs. the final checkpoint. Output:
  `correlate_rank_entropy_output.txt`.

## What this tests and what it found

Moalla et al. 2024 ("No Representation, No Trust", arXiv:2405.00662)
propose that PPO's late-training collapse is preceded by a DECLINE in
penultimate-layer effective rank (their "capacity loss"), measurable
independent of dormant-unit counts. This project's own ReDo audit already
found zero dormant units across all 8 runs, so rank is the one plasticity
diagnostic in that paper's toolkit this project had not yet checked.

Result: the opposite sign, 8/8 runs. `srank` rises through training
(r(srank, iter) = 0.69 to 0.94, all positive) and anti-correlates with
`ppo_entropy` (r = -0.45 to -0.82, all negative) — i.e. as the policy
sharpens (entropy falls) the shared-trunk representation gets *more*
differentiated, not less, and rank at the honest-peak iter is LOWER than
rank at the fully-collapsed final iter in all 8 runs (e.g. v28 seed3,
the best run: srank 10/32 at the honest peak (iter 90, honest 0.67) vs.
16/32 at iter 240 (honest 0.0)). Reproduce with:

```
.venv/bin/python runs/peak_instability/field_literature/feature_rank_probe.py
.venv/bin/python runs/peak_instability/field_literature/correlate_rank_entropy.py
```

This is a clean null on the specific, cheap, literature-sourced diagnostic
(effective rank of the shared trunk) as applied to THIS architecture —
not a null on the paper's full "capacity loss" claim, which is defined via
a fit-to-random-target probe over the course of training and was not run
here (would need new training compute to do properly — a frozen-checkpoint
probe is a reasonable proxy for capacity but the paper's own metric is
dynamic, not a single-checkpoint snapshot). See the parent response for
the full mapping of this and five other literature threads against the
phenomenon, including a sourced correction to the project's own claim that
ReDo's zero-dormancy result rules out "primacy bias" (Nikishin et al. 2022
define and diagnose primacy bias via learning-curve behavior under
replay-buffer priming, not via dormant units, and study it exclusively in
off-policy replay-buffer algorithms (SAC/DrQ/SPR) — the paper does not
test or address on-policy, buffer-free algorithms like PPO at all).
