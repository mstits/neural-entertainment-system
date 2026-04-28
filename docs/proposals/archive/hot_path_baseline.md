# Hot-path baseline — M4 Max, 2026-04-20

## IMPORTANT UPDATE (2026-04-20 late session)

**PGO landed +81% throughput — the single biggest perf win of this
session.** See `docs/proposals/pgo_results.md` for full data. Post-PGO
numbers:

- `pool_step`: 20.1 ms → **10.5 ms** per trainer step (-48%)
- Worker-steps/sec: 710 → **1289** (+81%)
- 1-gen trainer wall-time: 2.54 s → **1.35 s** (-47%)
- `pool_step`'s share of trainer wall-time dropped from 89% → 84%
  (still dominant but less so).

The breakdowns below reflect the **pre-PGO** run; they're kept as the
unoptimized-baseline snapshot for future regression comparison. All
"next-priority" recommendations are still valid as relative rankings
but their absolute magnitudes are smaller post-PGO (e.g. the
scanline-PPU ceiling of 14% applies to the new, lower absolute
emulator cost).

Measured via `scripts/bench_hot_path.py` on zelda.nes.

**Config:** 16 workers × 200 steps × frame_skip=16 = 3200 worker-steps,
51,200 NES frames.

| Layer                   | ms       | % of total | ms/step |
|-------------------------|---------:|-----------:|--------:|
| `pool_step` (Rust emu)  | 4022.9   |   **89.3** |  20.114 |
| `policy_forward` (MPS)  |  455.6   |     10.1   |   2.278 |
| `stacker_push` (NumPy)  |   15.0   |      0.3   |   0.075 |
| `narrator` (Rust)       |    5.1   |      0.1   |   0.026 |
| `audio_drain` (Rust)    |    3.3   |      0.1   |   0.017 |
| `reward_compute` (Rust) |    2.7   |      0.1   |   0.014 |
| `depth_tracker` (Rust)  |    1.2   |      0.0   |   0.006 |
| **TOTAL**               | **4505.8** |  **100.0** | **22.529** |
| pool.reset_all (init)   |    2.1   |            |         |

**Throughput: 710 worker-steps/sec.**

## Implications for the optimization phase

**`pool_step` is 89% of wall-time. Drill-down via
`scripts/bench_emulator_phases.py` splits that further:**

| Inside pool_step (per frame_skip=16 step) | μs    | % of step |
|-------------------------------------------|------:|----------:|
| CPU + APU + PPU(state-ticks)              | 7370  |    84.1%  |
| PPU pixel rendering (1 of 16 frames)      | 1398  |    15.9%  |

**Per-frame cost at fs=16: 460.6 μs CPU+APU+PPU_state, 1398.2 μs full
PPU render.** Cross-checked at fs=1 and fs=64; the linear model holds
(predicted fs=64 30.4 ms vs measured 31.8 ms).

### Revised priority order (data-driven)

1. **CPU + APU + per-cycle dispatch** — 84% of emulator time = ~75%
   of trainer wall-time. The real bottleneck. Architectural changes
   needed: CPU bulk-stepping, reduced per-cycle polling of the APU
   frame counter, flattened trait dispatch. Not easy, but where the
   headroom is.

2. **MPS-native async forward** — **INCONCLUSIVE / NOISE-LEVEL.**
   Reorder so `net(obs)` is kicked off at the END of an iter and
   `.cpu()` materialised at the START of the next. Metal kernel runs
   async during `pool.step_all` (20 ms, GIL released).
   - First bench pass: sync 853 sps, mps-async 869 sps (+1.8%).
   - Second pass: sync 881, mps-async 878 (-0.4%).
   - Third pass: thread-async +1.3%, mps-async -0.4%.

   At the batch sizes (16 genomes) and policy shape (84×84×4 →
   linear → 6 actions) in use, the overlap window is too small
   for reliable gains. Prototyped-and-reverted in `trainer.py` —
   kept `scripts/bench_async_pipeline.py` for future re-eval if
   the policy net grows or batches change.

   **REJECTED**: `threading.Thread`-based pipeline — bench showed
   -4.2% on first pass. Thread-spawn + GIL overhead dwarfs any
   overlap at this workload size.

3. ~~**MLX migration (Batch 10)**~~ — **TESTED, REGRESSES.**
   `scripts/bench_mlx_vs_mps.py` on the Nature-DQN CNN at batch=16:
   PyTorch-MPS 0.447 ms/fwd, MLX-Metal 0.682 ms/fwd — MLX is **34.5%
   SLOWER**. Likely reasons: MLX's Conv2D kernel is less mature than
   MPS's MPSCNNConvolution on the specific stride=4/kernel=8 shape;
   also MLX's NHWC layout may be slower than MPS's native NCHW
   without a good transpose fuser. Close this item — do not port.

4. **Core ML / ANE for single-frame inference** — **SHIPPED, 8×
   faster at batch=1**. `scripts/bench_coreml_ane.py` on the same
   policy: PyTorch-MPS 0.649 ms/fwd, Core ML 0.081 ms/fwd at
   batch=1. At batch=16 (training) all backends are within 4%, so
   MPS stays for training. Replay viewer now prefers Core ML via
   a pre-exported `<checkpoint>.mlpackage` saved at checkpoint
   time — instant load, no JIT wait on genome pick.

4. **Per-scanline PPU rewrite (Batch 6)** — **downgraded from the
   plan's top priority**. Ceiling is 15.9% of emulator cost ≈ 14%
   of trainer wall-time. Still real, but weeks of work for a modest
   return. Do this only AFTER (1) and (2) land and we need the next
   percentage point.

5. **Bulk step_n (Batch 5) is dead.** Per-frame skip-render already
   fires inside `pool_step`. No FFI hops to amortize.

6. **GA ops in Rust (Batch 4)** — not in the per-step profile at
   all. Confirmed deferred correctly.

### What the plan got wrong

`final_rust_plan.md` claimed Batch 6 would deliver 2-3× emulator
step speed. That was guessed from first principles; the actual
measurement shows full PPU render is only 15.9% of step cost at
production frame_skip on M4 Max (NEON-accelerated path). The CPU
per-cycle fetch/decode + APU frame-counter dispatch dominates. That
matches what you'd expect on modern Apple Silicon: the branch
predictor + 128KB L1 + M4's wide pipeline makes the "slow part" of
an emulator the inherently-serial 6502 state machine, not the
parallelisable pixel array.

**Update to `final_rust_plan.md`**: downgrade Batch 6 priority, add
async-pipeline as a new top priority.

## Method notes

- `pool_step` wall-time includes the 16-frame frame_skip loop per
  step, so `20.114 ms/step ÷ 16 frames ≈ 1.26 ms/NES-frame` across
  16 workers in parallel → ~**0.078 ms/NES-frame/worker** in the
  emulator.
- The bench uses argmax-deterministic policy actions, no
  exploration. Real training runs have the same profile shape; only
  the MPS forward timing is identical.
- `policy_forward` includes `torch.mps.synchronize()` — the reported
  number is wall-clock, not dispatch.
- `reward_compute` = 16 workers × one reward call each per step.
  The 0.014 ms/step figure confirms the overnight Rust port's
  claimed 4.1M-calls/sec hot path.
- Audio drain only hits `pool.drain_audio(i)` for each worker — the
  mixer's actual push is gated on mode ≠ "mute" in the real
  trainer, so audio cost in headless runs is ≤ what's shown here.

## Worker-count scaling (2026-04-20)

`scripts/bench_worker_scaling.py` measures headless `pool.step_all`
throughput at several worker counts on M4 Max:

### Pre-PGO (historical)

| workers | total sps | per-worker sps | efficiency vs 1-worker |
|--------:|----------:|---------------:|-----------------------:|
|       1 |       107 |          107.1 |                  100.0% |
|       4 |       435 |          108.8 |                  101.6% |
|       8 |       813 |          101.6 |                   94.8% |
|      12 |       947 |           78.9 |                   73.7% |
|      16 |       936 |           58.5 |                   54.6% |
|      20 |      1026 |           51.3 |                   47.9% |  ← old peak
|      24 |       772 |           32.2 |                   30.0% |
|      32 |       769 |           24.0 |                   22.4% |

### Post-PGO (current)

| workers | total sps | per-worker sps | efficiency vs 1-worker |
|--------:|----------:|---------------:|-----------------------:|
|       1 |       180 |          179.8 |                  100.0% |
|       4 |       764 |          191.1 |                  106.3% |
|       8 |      1427 |          178.4 |                   99.2% |
|      12 |      1511 |          125.9 |                   70.0% |
|      16 |      1624 |          101.5 |                   56.4% |
|      20 |      1866 |           93.3 |                   51.9% |
|    **24** | **1958** |        81.6 |                   45.4% |  ← new peak
|      32 |      1945 |           60.8 |                   33.8% |  plateau

**Post-PGO peak: 24 workers at 1958 sps.** PGO shaved per-worker cost
enough that more workers can pack in before context-switching
dominates. 32 is effectively flat with 24 — no gain, no loss.

**Actionable (landed 2026-04-20)**: default `num_instances` bumped
16 → 20 → **24** in the GUI and CLI to match M4 Max's post-PGO
throughput peak. Users on smaller chips should drop to their physical
core count.

## Pinned as the baseline

Re-run `scripts/bench_hot_path.py` after any Batch 6 or Batch 10
change to verify the claimed speedup actually lands here, not just
in micro-benchmarks. Re-run `scripts/bench_worker_scaling.py` if
binning changes (M4 Pro vs M4 Max vs M4 Ultra all have different
P/E-core mixes).

## Bench reproducibility — **not thermal**

An earlier version of this doc claimed the drops in throughput
between back-to-back bench runs were thermal throttling. That was
wrong. User confirmed the machine stays cool to the touch under
the bench workload.

The real cause of the sub-baseline numbers seen earlier: **PGO
was silently dropping off**. `cargo test`, a plain `maturin
develop --release`, or any rebuild path that doesn't go through
`scripts/pgo_build.sh apply` produces a non-PGO wheel and throughput
falls from ~1500 sps → ~700 sps (i.e. pre-PGO levels).

Confirmed by running 4 consecutive benches on a cool machine with
the PGO wheel in place: 1564, 1545, 1561, 1550 sps — σ ~10, no
degradation.

**Rule for reproducibility**: after any edit to `nes_core/src/*`,
run `bash scripts/pgo_build.sh apply` to restore PGO on the wheel.
A bare `maturin develop` silently gives you a non-PGO build and
everything downstream looks like a perf regression.
