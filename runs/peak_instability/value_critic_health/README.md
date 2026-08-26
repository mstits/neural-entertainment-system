# Value function / critic health — receipts

Dimension: does `ppo_value_loss` (and related PPO loss fields) destabilize
around each run's honest-eval peak, and if so, does it precede the
performance collapse (causal-shaped) or only follow it (symptom-shaped)?

Run these in order to reproduce everything below:

1. `extract_value_trajectories.py` — pulls `ppo_value_loss`, `ppo_policy_loss`,
   `ppo_loss`, `ppo_entropy`, `success_rate`, `vanilla_ppo_max_x` per
   iteration for all 8 runs, cross-checks each run's peak iter against
   `winners/best.json`, writes `value_trajectories.json` (raw series) and
   `summary.csv` (one row per run).
2. `changepoint_analysis.py` [`--exclude-warmup`] — finds each run's
   value-loss trough and the iter it regime-changes upward, finds the iter
   on-policy `success_rate` permanently collapses, and reports the lead
   (positive = value loss moved first). Writes
   `changepoint_summary_raw.json` / `changepoint_summary_excl_warmup.json`.
   **Use `--exclude-warmup`** — the raw/no-flag mode is kept only to show
   why the flag is needed (see Finding 3).
3. `threshold_sensitivity.py` — sweeps the changepoint detector's two free
   parameters (jump factor, sustain window) to check the lead/lag finding
   isn't an artifact of one convenient threshold choice.
4. `correlations.py` — cross-run (N=8) Pearson/Spearman between value-loss
   summary stats and honest peak score.
5. `analyze_out/*.txt` — `scripts/analyze.py --split "generation>PEAK_ITER"`
   run per-run against each run's own `metrics.jsonl`, full field ranking
   (this is the tool the task asked to be exercised; see its stdout header
   for row counts and split sizes).

All raw stdout captured in `*_output.txt` / `analyze_out/*.txt` next to each
script for exact reproducibility.

## Findings (see full writeup for hedges/confounds)

1. Config check: `value_loss: huber`, `value_coef: 0.25` identical across
   all 8 configs — constant, cannot explain between-run variance. Current
   value-loss magnitudes (10–65 throughout) and actively-decaying entropy
   are a different regime than the pre-2026-05-06 pathology that fix
   targeted (value loss pinned 300–660, entropy pinned near max, policy
   loss ~0). That old failure mode is not recurring here.
2. Whole-run pre/post-peak split (`scripts/analyze.py`): `ppo_value_loss`
   ranks 11th–28th of 31 fields by effect size in every one of 8 runs,
   with two runs showing near-zero effect (0.001, 0.065). `ppo_entropy`
   ranks #1–#3 in all 8 runs with 4–8x larger effect size. In aggregate,
   value loss is not a standout divergence axis; entropy dominates.
3. Localized changepoint timing (excl.-warmup): a post-warmup value-loss
   trough correlates with the honest-peak window in 7/8 runs (the 8th,
   the worst run v27_seed0, never forms a clean trough — its only
   sustained minimum sits deep in its terminal degenerate/near-zero-
   entropy plateau, where low value loss reflects deterministic failure,
   not health). Among those 7, a **loose** (1.3–1.5x) regime-change
   threshold puts value loss upticking a median 15–22 iters before
   success-rate permanently collapses (7/7 positive) — but a **strict**
   (2.0x) threshold flips this to near-zero/negative in 5 of 7 (2/7
   positive) — see `threshold_sensitivity.py`. Verdict: a mild, gradual
   value-loss drift begins somewhat before collapse; a genuinely large
   value-loss blowup is contemporaneous with or slightly lags collapse,
   not a clean leading indicator.
4. Cross-run: lower global-minimum value loss correlates with higher
   honest peak score (Pearson r=-0.735, Spearman r=-0.595, N=8) — a lead,
   not a finding, and consistent with reverse causation (a policy that's
   good for other reasons produces lower-variance, easier-to-fit returns).
5. Rare large `ppo_policy_loss` spikes (4 events, 3/8 runs, up to 1777)
   all occur well after each run's own peak/collapse, with `ppo_value_loss`
   completely unremarkable (36–45, normal range) at every one — value loss
   does not co-explode with these; they are post-collapse, policy-side
   artifacts, not part of the peak-to-collapse mechanism.

## What the data cannot answer

`metrics.jsonl` logs one scalar `ppo_value_loss` per iteration only — no
explained variance, no advantage magnitude/skew, no per-batch value-target
variance. So this dimension can show that the scalar value-loss signal and
the on-policy success-rate signal move in a certain order, but cannot show
whether a destabilizing critic is actually corrupting the advantage
estimates fed to the actor (the classic causal mechanism) versus merely
tracking a policy that is already becoming less consistent for unrelated
reasons. Adjudicating that would need additional instrumentation
(explained-variance per iter, advantage std/skew per iter) or a controlled
ablation (e.g. freeze the critic or zero `value_coef` after each run's own
peak and see whether the collapse still happens on the same schedule).
