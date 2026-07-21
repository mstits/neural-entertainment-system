# Rust actor / learner split — design and acceptance criterion

Status: **design + acceptance test only. Not implemented.**
Date: 2026-07-20
Grounds every number in `runs/throughput_split_2026-07-20.json` (the
measured per-iteration wall-clock split), read against `src/training/trainer.py`
(the rollout loop and PPO update), `src/emulation/rust_pool_adapter.py`
(the current FFI adapter), and `nes_core/src/pool.rs` (the GIL-release
boundary at `step_all`, ~L817).

The acceptance criterion this document argues for is codified as a
runnable test scaffold in `tests/test_actor_learner_parity.py`, which
skips until the actor exists. **The test is written before the build so
the bar is fixed before the first line of actor code.**

---

## 1. What the measurement says (and does not say)

From `runs/throughput_split_2026-07-20.json`, `mario_2_1_runB`, mean over
75 iters (60 workers × 1024 steps = 61,440 env-steps/iter):

| Bucket | ms/iter | % wall | Where |
|---|---:|---:|---|
| **MPS PPO update** | 22,595 | **58.6%** | K-epoch minibatch SGD on MPS (`trainer.py:6253`) |
| **Rust emulation** | 12,360 | **32.1%** | `pool.step_all` — NES core, GIL released (`trainer.py:5457`, `pool.rs:817`) |
| iter reset | 1,289 | 3.3% | per-iter start-state restore |
| rollout fwd/obs/sample | 861 | 2.2% | policy forward + logits→cpu + multinomial + sticky (`trainer.py:5446`) |
| RND intrinsic | 40 | 0.1% | full-rollout novelty pass (`trainer.py:5971`) |
| **GAE** | 2.9 | **0.0%** | batched backward sweep (`trainer.py:5995`) |
| reward-loop glue (untimed) | ~1,380 | 3.6% | per-worker Python reward loop (`trainer.py:5467`) |

Grouped as the review framed it: **32.1% substrate / 58.6% learner /
9.2% Python glue.** Torch-free collection of the exact same config runs
**2.5–2.8× faster** than the full training loop (4,052 mean sps vs 1,601
training sps). The learner is the binding constraint; the substrate is not.

**The one number that governs this whole design:**

> If collection is fully overlapped with the learner, iteration time drops
> to `max(collection, update) = max(~13.6 s, 22.6 s) = 22.6 s`. That is
> `61440 / 22.6 = 2719 sps = 1.70× the current 1601`. **The ceiling is set
> by the MPS update, not by the emulator.**

Two consequences that bound scope, both non-negotiable:

- **Do not migrate the learner or GAE to Rust.** GAE is 2.9 ms (0.0%).
  The autograd update is the thing we are trying to hide collection
  *behind*, not replace. Migrating either buys nothing and forfeits
  PyTorch/MPS + the entire training stack. (Reaffirms the standing
  memory: keep PyTorch/MPS + Qt in Python; migrate everything else.)
- **Beyond ~2719 sps the update dominates and emulator work is wasted
  effort.** Further speedup needs a faster/larger-batch MPS update or
  fewer PPO epochs — a *separate* investigation, out of scope here.

---

## 2. The architecture in one paragraph

Run the rollout collection as a **native, GIL-free Rust actor** that
double-buffers: while the Python learner runs the MPS PPO update on
rollout buffer *N−1*, the actor collects buffer *N* entirely in Rust —
policy forward, action sample, sticky override, `frame_skip` env step,
reward, and per-step record — releasing the GIL for the whole rollout so
the 32% emulation overlaps the 59% update. One FFI call returns the whole
rollout as contiguous arrays, replacing today's 1024 `step_all` calls and
their 61,440 per-worker Python touches. The learner is unchanged: it
takes the returned buffers and runs RND + GAE + the K-epoch update + the
entropy-floor controller + demo-anchor exactly as it does today.

This is the classic decoupled actor/learner (IMPALA / Sample Factory /
APPO), specialized to one machine and one GPU where inference is cheap
(861 ms) and the win is CPU↔GPU overlap, not distribution.

---

## 3. Two axes, kept strictly separate — this is the honesty of the design

The whole design lives or dies on not conflating two independent claims.

### Axis A — correctness (byte-identical, testable, mandatory)

> Given identical initial worker states, identical `frame_skip`, and an
> identical sequence of per-step action bitmasks, the Rust
> `collect_rollout` path must produce **byte-identical** env-side buffers
> — `ram`, `reward`, `done`, `obs`, `bonus` — to today's Python
> `step_all` + per-worker reward loop.

This is byte-exact-able because it is the *same deterministic NES core*
(no RNG in `Worker`; `pool.rs` L59–98) driven by the *same integer/fixed
reward math* (`compute_rewards_batch`, already byte-parity-tested on the
GA path in `tests/test_compute_rewards_batch.py`). The action *sampling*
is factored out — fed as a fixed sequence — because it is the only
non-deterministic, backend-dependent part. This axis is what
`tests/test_actor_learner_parity.py` asserts. **It is the acceptance gate.**

### Axis B — the policy forward (numerical agreement, tolerance, not byte)

A Rust-native tile-policy forward (Step 4) will **not** bit-match a
PyTorch-MPS forward on the same weights — different BLAS, different
reduction order. That is expected and fine. The requirement is that the
Rust forward, on a faithful copy of the PyTorch weights, agrees within
tolerance (`atol` on logits), so the Rust snapshot *represents the same
policy* and PPO importance ratios sit at `1 ± ε` on the first epoch. This
is a **tolerance** test, never a byte test. Conflating it with Axis A is
the mistake that would sink the review.

### Axis C — overlap introduces one-iteration policy lag (off-policy, validated by learning, not byte)

Running the actor a snapshot behind the learner makes the behavior policy
exactly **one iteration stale** — this is APPO, deliberately off-policy by
a bounded amount. It is **not** byte-identical to synchronous PPO and must
not be claimed to be. It is validated by **learning-curve equivalence**
over M iters (mean episode return / clear-rate within noise of the
synchronous baseline), gated behind a flag, reversible. The synchronous
`collect_rollout` (Step 5) is the on-policy, byte-parity-tested fallback.

| | Axis A: env path | Axis B: policy fwd | Axis C: overlap |
|---|---|---|---|
| Claim | byte-identical | numerically close | learns the same |
| Test | `assert_array_equal` on ram/reward/done/obs | `allclose(atol)` on logits | return/clear within noise over M iters |
| Gate | **hard** (blocks merge) | hard (blocks Step 4) | soft (flag stays off until met) |

---

## 4. The exact FFI boundary

### Today (measured, per iteration)

- **1024 × `Pool.step_all(actions: uint8[N]) -> PyList[(frame, pp, ram, done)]`**
  (`pool.rs:778`, adapter `rust_pool_adapter.py:193`). One numpy array in
  (zero-copy), a Python list of **N tuples** out.
- Python then, **61,440 times/iter**, unpacks a tuple, copies RAM with
  `bytes(ram_bytes)` (`rust_pool_adapter.py:275`), and calls
  `reward_fns[i].compute(ram, action)` (`trainer.py:5603`).
- The "61,440 FFI crossings" the review cites *are these per-worker
  Python touches* — one materialize + one reward compute per env-step.

### Proposed (one call per rollout)

```python
# nes_core.Pool
def collect_rollout(
    self,
    rollout_steps: int,
    *,
    policy: "nes_core.TilePolicy | None",   # Rust-native fwd; GIL-free.
    replay_actions: "np.ndarray | None" = None,  # (T, N) uint8 — PARITY/TEST
                                                  # hook: bypass policy, feed
                                                  # exact executed actions.
    reward_fns: "list[nes_core.RewardFunction]", # per-worker, mutated in place
    sticky_p: float = 0.0,
    seed: int = 0,                # seeds action-sample + sticky RNG streams
    obs_mode: str = "tile",       # "tile" | "pixel"
    want_ram: bool = False,       # ship (T,N,2048) RAM (curriculum capture / debug)
    gx_count_beta: float = 0.0,   # count-based frontier bonus, if enabled
) -> "nes_core.RolloutBatch": ...
```

`RolloutBatch` — every field one contiguous allocation, `(T, N, …)`:

| field | shape | dtype | meaning |
|---|---|---|---|
| `obs` | (T, N, obs_dim) | int8 (tile) / uint8\|f16 (pixel) | policy input, per step |
| `actions` | (T, N) | int32 | **executed** action id (post-sticky) |
| `log_probs` | (T, N) | float32 | behavior-policy log-prob of executed action, clamped `min=-13.0` |
| `values` | (T, N) | float32 | critic head output |
| `rewards` | (T, N) | float32 | extrinsic reward (`RewardFunction.compute`) |
| `dones` | (T, N) | bool | `r.done \|\| rew_done` |
| `bonus` | (T, N) | float32 | count-based frontier bonus (0 if disabled) |
| `ram` | (T, N, 2048) | uint8 | only if `want_ram` (curriculum/GX capture, parity) |

**FFI accounting:** 1024 calls → **1**; 61,440 `bytes()` copies → **0**
(RAM ships as one array, gated by `want_ram`); 61,440 Python reward calls
→ **0** (reward runs in Rust, `compute_rewards_batch`'s per-worker
`RewardFunction` already borrows in place). The learner receives numpy
views over the `RolloutBatch` arrays — no per-step object churn.

### What stays in Python (unchanged, by design)

- The **learner**: RND intrinsic pass (`trainer.py:5933`), `fold_intrinsic_into_rewards`,
  `batched_gae` (`trainer.py:5991`), advantage normalization, the K-epoch
  minibatch update (`trainer.py:6137`), `ppo_losses`, optimizer, grad clip,
  NaN backstop. It consumes `RolloutBatch` exactly where it consumes the
  numpy `*_buf` arrays today. **Zero learner-math change.**
- The **entropy-floor controller** and **demo-anchor** — see §7.
- **Curriculum capture / Go-Explore archiving.** These read RAM
  mid-rollout (`trainer.py:5520`, `5576`) and call `save_worker_state`.
  In the actor design they either (a) run in Rust from `want_ram=True`
  buffers post-rollout, or (b) stay a Python post-pass over the returned
  RAM. Kept as an explicit, separately-tested seam — **not folded into the
  first actor build** (see §6, Step 5 scope note).

---

## 5. Thread-safety and determinism

### The existing model (must be preserved)

`Worker` lives in `WorkerCell(UnsafeCell<Worker>)` (`pool.rs:476`).
`step_all`/`reset_all` dispatch `rayon::par_iter().enumerate()`, each task
touching a **unique** index → unique `&mut Worker` (`worker_mut`,
`pool.rs:501`). All Python entry points are **sequential** (never
overlapping an in-flight rayon dispatch). The NES core carries **no RNG**;
a worker is fully deterministic given (start state, action sequence,
`frame_skip`). This is why Axis A can be byte-exact.

### What the actor adds

A second OS thread (the actor) running concurrently with the learner
thread. New invariants, each with a single enforcement point:

1. **Params double-buffer.** Two Rust policy snapshots, A/B. The learner
   trains the PyTorch `net`; at the iter barrier it serializes updated
   weights into the *idle* snapshot. The actor reads only its *current*
   snapshot during collection. The copy happens **only at the barrier,
   when the actor is between rollouts** — never concurrent. No lock on the
   hot path.
2. **Rollout double-buffer.** Two `RolloutBatch` allocations, ping/pong.
   The learner reads buffer *N−1* while the actor writes buffer *N*.
   Distinct memory; the only synchronization is the barrier handoff.
3. **One barrier per iter.** Actor and learner rendezvous once: actor
   publishes buffer *N*, learner publishes new weights, both swap. Outside
   the barrier there is **no shared mutable state** — this keeps the
   concurrency surface to a single, auditable point.
4. **Rayon pool is shared but non-overlapping.** The actor owns
   `step_all_native`; the learner never touches rayon. Since the actor is
   single-threaded into rayon and the learner is off rayon entirely, the
   existing `UnsafeCell` soundness argument (`pool.rs:537–562`) holds
   unchanged. `save_worker_state` from a Python post-pass must still be
   sequenced *after* the barrier (documented precondition).

### Determinism

- **Substrate:** deterministic, unchanged (no `Worker` RNG).
- **Action sampling + sticky:** the *only* RNG. `collect_rollout` takes an
  explicit `seed` and **must document its RNG stream** (order of draws:
  per step, per worker, sample-then-sticky). Note the standing caveat
  already in the code (`trainer.py:5403`): the CPU multinomial draw is
  statistically identical to the old MPS Philox stream but does not
  bit-reproduce pre-change trajectories — the actor inherits, and must
  document, the same "reproducible under its own seed, not across a
  sampler change" contract.
- **Async timing is non-deterministic; the math is not.** The one-iter
  lag (Axis C) is *exactly* one iteration because the barrier is a hard
  rendezvous — thread scheduling jitter changes wall-time, never which
  weights collected which buffer. This is what makes Axis C validate-able
  by learning curve rather than dissolving into irreproducible noise.

---

## 6. Incremental build order (smallest first)

Each step ships and is tested **independently**; each shrinks FFI/glue or
adds one enabling capability; none requires the next. The 1.70× arrives
only at Step 6, but every prior step is a standalone, reversible win.

**Step 0 — baseline gate (done).** `runs/throughput_split_2026-07-20.json`
is the before-number. Any step that regresses the 32.1% emulation bucket
or the substrate byte-parity fails.

**Step 1 — batched RAM egress (smallest).** `step_all` returns one
contiguous `(N, 2048)` uint8 array instead of N `bytes()` objects. Kills
61,440 `bytes()` allocs/iter. Pure adapter + Rust egress change; RAM
content byte-identical. Test: extend `test_rust_pool_adapter.py`.
*Independent shippable win, no actor.*

**Step 2 — batched reward on the vanilla path.** Route the vanilla
per-step reward through the existing `compute_rewards_batch`
(`python.rs:921`), already byte-parity-tested on the GA path. Removes the
61,440 per-worker Python reward calls + the 3.6% reward-loop glue. Test:
`test_compute_rewards_batch.py` already pins batched == per-call.
*Independent shippable win, no actor.*

**Step 3 — lean per-step recorder.** `step_all` writes obs/action/reward/
done into caller-provided contiguous buffers instead of building per-step
Python objects. Still 1024 calls/iter, but each is allocation-free on the
Python side. Test: buffer contents byte-match the object path.
*Independent shippable win, no actor.*

**Step 4 — `nes_core.TilePolicy` (Rust-native forward).** Load the tile
MLP weights (14k–116k params); forward + critic in Rust f32. **Axis B**
gate: logits `allclose(atol)` vs the PyTorch net on shared weights.
Enables GIL-free collection. **Scope: tile mode only.** Pixel-CNN Rust
forward is a larger lift and is explicitly deferred — the flagship runs
(runB, curriculum) are all tile mode, so tile-first captures the win.
*Enabling capability; no throughput change yet.*

**Step 5 — `collect_rollout` (synchronous, one FFI call).** The full
1024-step loop in Rust: TilePolicy forward → seeded sample → sticky
override → `frame_skip` step → batched reward → record into `RolloutBatch`.
Learner still waits (no overlap). **This is where the Axis A byte-identical
acceptance test bites** (`test_actor_learner_parity.py`). Curriculum/GX
capture stays a Python post-pass over `want_ram` buffers (not yet folded
in). *Erases the FFI/glue; still update-bound at ~1× — the correctness
milestone.*

**Step 6 — double-buffered actor/learner overlap (the 1.70×).** Actor
thread collects buffer *N* with the *N−1* snapshot while the learner
trains buffer *N−1* on MPS. Gated behind `--actor-overlap` (default off).
**Axis C** gate: learning-curve equivalence over M iters vs the Step-5
synchronous baseline. Ceiling: `max(13.6, 22.6) = 22.6 s → 2719 sps →
1.70×`. *The headline; reversible via the flag.*

---

## 7. How sticky / entropy-floor / demo-anchor are preserved

The key realization: **two of the three live entirely in the learner and
are preserved for free; only sticky touches the actor.**

### Entropy-floor — learner-side, untouched

The adaptive controller (`trainer.py:7147–7160`) runs *after* the update,
reading `last_entropy` and nudging `self.entropy_coef`:

```
if last_entropy < entropy_floor:      coef = min(coef*1.5 + 1e-4, entropy_coef_max)
elif last_entropy > 1.5*entropy_floor: coef = max(coef*0.9, base)
```

`entropy_coef` feeds only `ppo_losses` (`trainer.py:6177`), inside the
K-epoch update. The actor never sees it. In async mode the coef the
learner computes at iter *N* applies to update *N*; the actor collecting
*N+1* is indifferent. **Preserved verbatim, zero actor code.** The parity
test pins the controller formula so a change to it flags that the design
assumption moved.

### Demo-anchor — learner-side, untouched

The DQfD-style term is added *inside the minibatch loop*
(`trainer.py:6185–6197`): draw a demo minibatch, `demo_anchor_loss` with
margin, scaled by `_demo_coef` (linear decay over `demo_anchor_decay_iters`
keyed on absolute `global_it`, `trainer.py:6124`). It is a learner loss on
a fixed demo bank; the actor's rollout has nothing to do with it.
**Preserved verbatim, zero actor code.** The parity test pins
`demo_anchor_loss` as update-side.

### Sticky — the one actor-side semantic; must be replicated exactly

Sticky lives in the rollout (`trainer.py:5415–5435`), so the actor must
reproduce it bit-for-bit:

1. Only for `t > 0` and with `_sticky_p > 0`.
2. Draw `np.random.random(num_envs) < _sticky_p`; on hit, replace the
   sampled action with `_prev_exec_action[row]`.
3. Record the **executed** action's log-prob (importance ratio parity),
   **clamped `min=-13.0`** — the NaN backstop that keeps a near-deterministic
   policy's stuck-action log-prob from exploding the PPO ratio
   (`trainer.py:5431`).
4. `_prev_exec_action[:] = executed actions` after every step.

Determinism requires the actor's sticky RNG to be seeded and its stream
documented (§5). The parity test provides (a) a pure-Python **executable
spec** of this exact algorithm that runs today, and (b) a
seed-matched byte-parity assertion against the Rust actor once it exists.

---

## 8. Acceptance gates (codified in `tests/test_actor_learner_parity.py`)

| Gate | Axis | Test | Blocks |
|---|---|---|---|
| env buffers (ram/reward/done/obs) byte-identical given fixed actions | A | `test_collect_rollout_byte_identical_env_buffers` | Step 5 merge |
| deterministic under a fixed seed (two runs identical) | A | `test_collect_rollout_deterministic_under_seed` | Step 5 merge |
| sticky override matches the Python reference bit-for-bit | A | `test_collect_rollout_sticky_matches_reference` | Step 5 merge |
| one call returns the whole `(T, N, …)` rollout | — | `test_collect_rollout_is_single_call` | Step 5 merge |
| TilePolicy logits `allclose(atol)` vs PyTorch | B | (Step-4 test, not in this file) | Step 4 merge |
| learning-curve equivalence over M iters | C | (Step-6 soak, not in this file) | `--actor-overlap` default-on |

The runnable-now tests (reference reproducibility, the sticky executable
spec, the entropy-floor/demo-anchor learner-side pins) validate the
harness and freeze the contracts today; the actor tests skip with a clear
reason until `collect_rollout` exists, then assert without further edits.

---

## 9. What this design explicitly is NOT

- **Not** a learner migration. GAE and autograd stay in PyTorch/MPS.
- **Not** a claim that async == synchronous PPO. Axis C is off-policy by
  one iteration and is validated by learning, not parity.
- **Not** a pixel-CNN port. Tile-first; pixel is future work.
- **Not** the implementation. This is design + a codified acceptance test.
  No actor code exists; `tests/test_actor_learner_parity.py` skips.
