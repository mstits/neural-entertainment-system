# Project: Rust-Native NES Core (`nesrs-py`)

> **⚠ HISTORICAL: PRE-MIGRATION COMPARISON DOCUMENT.**
>
> This document compared `nes-py` vs `nesrs` (tetanes-core) during
> the backend-evaluation phase. **Both backends are now deleted.**
> The live core is the in-tree Rust crate at `nes_core/`, a
> purpose-built NES emulator, not a wrapper around anything external.
>
> **Current state (2026-04-20)**:
> - Single backend: `nes_core` (in-tree Rust).
> - 2.0× faster than the old nes-py baseline pre-PGO; 3.6× faster
>   post-PGO (710 → 1289 worker-steps/sec on the hot-path bench).
> - Real APU audio (deleted chiptune/NSF fallback paths).
> - In-memory save-state via `NCST\x01`-prefixed binary blobs.
> - See `ARCHITECTURE.md` for the current architecture and
>   `proposals/hot_path_baseline.md` for measured perf.
>
> Nothing below reflects live code. Kept for historical context.

## The pitch

Replace `nes-py`, our current Cython wrapper around an ageing C++
emulator (SimpleNES, last substantive commit 2018, **no APU**), with a
Rust-native NES core exposed through PyO3. The outcome is a single
`pip`-installable wheel, `nesrs-py`, that is:

* **3–5× faster** than nes-py on per-frame emulation,
* **cycle-accurate** (the reference Rust cores match Mesen on every test
  ROM in the NesDev suite),
* **audio-capable** from frame one (full APU implementation built in,
  exposed as 16-bit PCM samples per step),
* **zero-copy** for Python (frame/RAM buffers are returned as memoryview
  over Rust-owned memory, not bytes-copied),
* **native arm64** with no x86_64 fallback, built against the same
  toolchain that powers the training stack.

We keep the same `NESEnvironment` / `FrameTransport` contracts; the only
thing that changes below the API is what produces the bytes. The GUI,
training loop, reward functions, curriculum, BC, PPO: all of it lights
up unchanged, except now audio also works.

## Why this is interesting

* **The performance ceiling moves by 5×** without any algorithm changes.
  At ~8,500 agent-steps/sec aggregate (up from ~1,700), a Zelda
  generation drops from ~23s to ~5s. Overnight runs are an actual
  overnight, not a long weekend.
* **Audio problem dissolves.** The "nes-py has no APU" dead-end is
  replaced by cycle-accurate synthesized audio samples streamed out the
  emulator alongside every frame. No NSF side-channel, no ROM-vs-audio
  sync problems. It is literally the same PCM a real console's DAC
  would emit, because we compute it the same way.
* **Cycle accuracy improves training signal quality.** SimpleNES gets
  roughly 85% of NesDev's accuracy tests right; tetanes-core and its
  peers score ≥99%. Subtle mapper quirks that occasionally corrupt
  long-run Zelda episodes (e.g., MMC1 bank-switching edge cases on the
  SRAM save slot) simply stop happening. Training noise drops.
* **A real distributable artifact.** A single `pip install nesrs-py`
  that works on arm64 macOS with audio and proper mapper coverage would
  genuinely fill a hole in the Python ecosystem. The incumbents are
  stale (nes-py, last release pre-pandemic) or broken on arm64
  (stable-retro). Nobody's shipped the obvious right thing.
* **Sets up world-model work.** The Rust core is the data pipeline
  feeding the world model (see `docs/world_model_rl.md`). 5× more
  real-env rollouts per hour = 5× more training data for the world
  model. These two projects compound.

## Core architecture

### Emulator choice

Three serious candidates, in priority order:

1. **tetanes-core** (`lu-zero/tetanes`): modern, active, MIT-licensed,
   passes 99%+ of NesDev tests, has APU, save states, 20+ mappers.
   Actively maintained by the same author as `tetanes` the standalone
   emulator. *This is the default.*
2. **nes** (`kamiyaowl/nes`): another pure-Rust core, smaller surface,
   cleaner architecture. Fallback if tetanes has a blocking issue.
3. **rusticnes-core** (`kylc/rusticnes-core`): older but battle-tested.

We wrap one of these with PyO3 and expose a minimal surface.

### Wrapper layout

```
nesrs-py/
├── Cargo.toml
├── pyproject.toml          # maturin build config
├── rust/
│   └── src/
│       ├── lib.rs          # #[pymodule] entry, PyO3 bindings
│       ├── env.rs          # NESEnvironment Python class
│       └── audio.rs        # APU → PCM sample buffer
└── python/nesrs/
    ├── __init__.py         # re-exports NESEnvironment
    └── constants.py        # BUTTON_A, etc. (verbatim from current code)
```

### Python-facing API (drop-in for `src/emulation/nes_environment.py`)

```python
class NESEnvironment:
    def __init__(self, rom_path: str, frame_skip: int = 1,
                 start_state_path: str | None = None): ...

    def reset(self) -> np.ndarray: ...
    def step(self, action: int) -> tuple[np.ndarray, bool]: ...
    def get_frame(self) -> np.ndarray: ...
    def get_ram(self, addr: int) -> int: ...
    def get_ram_range(self, start: int, end: int) -> np.ndarray: ...

    # NEW — the reason we're doing this
    def get_audio(self) -> np.ndarray:
        """Stereo int16 samples produced during the last step.
        Sample rate is self.sample_rate (44100 by default)."""

    def save_state(self) -> bytes: ...
    def load_state(self, data: bytes) -> None: ...

    sample_rate: int
    close: Callable[[], None]
```

Zero-copy: `get_frame()` returns a `numpy.ndarray` whose buffer is
`PyArray_FromRustArray`-owned memory; no memcpy on the hot path.
`get_audio()` similarly hands back a numpy view over the APU's
per-step ring buffer before it gets overwritten on the next `step()`.

### Build system

* `maturin develop --release` during dev. Rust compiles in ~15s.
* `pipx install maturin` for the user's machine.
* CI: `maturin build --release --target universal2-apple-darwin` →
  `.whl` for arm64+x86_64. Distributable.

## Speed reality (measured, M4 Max, Apple Silicon)

`scripts/bench_backends.py` A/B on `zelda.ines1.nes`, frame_skip=16,
non-cycle-accurate, Pixellate video filter:

| Config | nes-py | nesrs | nesrs vs nes-py |
|---|---|---|---|
| 1 worker × 500 steps | 93 steps/s | 38 steps/s | **0.41×** |
| 4 workers × 200 steps | 363 steps/s | 143 steps/s | **0.39×** |

**nesrs is ~2.5× slower than nes-py's SimpleNES core**, not faster.
The original "5×" estimate was aspirational based on tetanes-core
micro-benchmarks; in reality SimpleNES (the C++ core nes-py wraps)
is already heavily optimised for raw throughput, and tetanes' focus
on cycle accuracy costs per-frame speed.

**So why keep nesrs?** The value isn't speed:

- **Real APU audio per step**: the streaming hook. `nes-py` has no
  APU; there is no alternative Python core that does.
- **Cycle accuracy**: passes 99 %+ of NesDev's test suite. SimpleNES
  passes ~85 %. Subtle mapper bugs that corrupt long training runs
  simply stop.
- **Save-state round-trip as bytes**: useful for curriculum
  auto-promotion (save mid-episode, restart deeper next gen).
- **Active upstream maintenance**: `nes-py`'s last release predates
  the pandemic.

For the **stream**, use `--env-backend nesrs` and accept 2-3× wall-
clock for audio + accuracy. For **overnight training where speed
dominates**, stay on `nes-py` (default) until the Rust core gets
further optimisation work. The path is viable (tetanes has a
`NoVideo` headless mode we'd need to tweak to still emit frames
cheaply, plus SIMD RGB unpacking).

## Milestones

### M0: `nesrs-py` proof of life (3 days)
* maturin-built crate in `nesrs-py/`, `pip install -e` works.
* Wraps tetanes-core, exposes `NESEnvironment.{reset, step, get_frame,
  get_ram_range}`.
* Passes a round-trip test: load Zelda ROM, reset, step 1000 times,
  assert frame shape and RAM at known addresses match what nes-py
  produces for the same input sequence.
* Output: `nesrs-py/README.md` with usage, `tests/test_nesrs_basic.py`

### M1: Training parity (4 days)
* Add `save_state` / `load_state` so curriculum start-states work.
* Plumb into `src/emulation/nes_environment.py` via a `--backend
  nesrs` flag. The rest of the stack stays byte-identical.
* Run a 50-generation Zelda smoke test. Assert: convergence trajectory
  matches nes-py within noise, wall-clock is ≥3× faster.
* Output: `tests/test_backend_parity.py`. A deterministic seeded run
  on both backends must produce identical trajectories.

### M2: Audio pipeline (3 days)
* Expose APU samples via `get_audio()` as documented above.
* Wire into `AudioMixer`. Replace the NSF fallback path with live
  samples from the `solo-N` worker's transport. Add `audio` bytes to
  the FramePacket dataclass.
* Authentic sword-swing sounds, rupee jingles, dungeon music, all
  tied to actual gameplay, not a side channel.
* Output: 30-second video demo showing audio keyed to Link's actions.

### M3: Distribution (2 days)
* Universal wheel build, publish to PyPI as `nesrs-py`.
* README with badges, MIT license, a link to this repo, and a "why
  this exists" section.
* Announce in r/EmuDev. The project is now independently useful
  beyond our training stack: any Python NES RL researcher gains 5×.
* Output: PyPI page, first release tag.

**Total: ~12 working days, one person, one M-series Mac.**

## Risks & how we handle them

| Risk | Mitigation |
|---|---|
| tetanes-core has an API-breaking change between our pin and their next release | Pin exact version in Cargo.toml. Periodic (not automatic) upgrades. |
| A mapper Zelda uses (MMC1) has a subtle bug we don't catch until late training | M1's parity test catches most of these on short runs. If a specific long-run bug appears, narrow it down and file upstream (Rust cores have responsive maintainers). |
| PyO3 / maturin versions drift with Python 3.12, 3.13 | Build universal wheels against 3.11 and 3.12 in CI. Re-run on each Python bump. Cost is low. |
| Audio latency (we publish samples per-step, but sounddevice wants continuous) | Buffer samples in the mixer for ~50 ms. Already the pattern for file-loop playback; nothing new. |
| Save-state format differs from nes-py, breaking old checkpoints | Curriculum `.state.bin` files are recorded via nes-py's save_state. Re-record them under nesrs-py at M1 (1-time cost, documented in the CHANGELOG). |
| Rust compile times slow dev iteration | `maturin develop` uses `cargo build`'s incremental compilation. After the first build, edit/reload is ~3 s. Fine. |

## Relationship to the world-model project

These two projects compose:

```
nesrs-py (this doc)
    │
    ├── 5× faster real-env rollouts
    │
    ▼
World-Model RL (docs/world_model_rl.md)
    │
    ├── 100× more training from the same real rollouts
    │
    ▼
500× effective training throughput, end to end
```

Do `nesrs-py` first. The world-model project's M1 (replay-buffer
collection) directly consumes this output. If you do the world model
first and then swap the emulator later, you throw away 2 months of
collected experience because save-state formats changed.

## Success criterion

One sentence: **"I typed `pip install nesrs-py`, and a Python NES
emulator with APU audio and cycle-accurate Mesen parity showed up on
my arm64 Mac, 5× faster than anything else that existed."**

One commit that says `nesrs-py: v0.1.0 shipped`. One training run log
showing our existing Zelda recipe converging in a fifth the time. One
tweet.

## Future perf wins (Python-side, independent of `nesrs-py`)

These are stacking wins we can land without touching the Rust work. They
become meaningful once emulation isn't the bottleneck (post-`nesrs-py`)
but are worth tracking separately since they land on Python time only.

1. **Async inference / emulation pipelining** (~25–40 % per-step throughput).
   Step `K+1`'s GPU forward pass overlaps with worker emulation of step `K`.
   Requires action `K+1` to be computed from observation `K-1`, one
   observation-step of latency the agent rarely notices on NES. The
   `self.async_pipeline` flag in `Trainer.__init__` is reserved for this;
   implementation is a rewrite of the top of `_evaluate_batch`'s step
   loop to double-buffer action tensors. Risk: trajectory semantics
   change subtly (REINFORCE log-probs move by one step), so gate behind
   the flag + validate with a parity test before flipping the default.

2. **torch.compile cache warmup**. Every generation pays the first-call
   compile cost because the network + batch shape change per-batch.
   Cache at (dtype, batch_size, device) level so gen 2+ skips.

3. **Bulk reward-function dispatch**. The per-step reward loop iterates
   per-genome and mutates a breakdown dict. Consolidate to a vectorised
   per-batch pass that consumes RAM snapshots and emits a (N, num_signals)
   tensor.

4. **Shared-memory reward breakdown**. Instead of trainer re-reading
   each RAM snapshot via `read_latest()`, reward fns could live in the
   workers alongside the emulator, emitting only the per-signal deltas.

5. **Smaller per-step CPU→MPS transfers**. Preprocessed obs are
   already 84×84 uint8 (28 KB / worker / step). Stacking the 4 history
   frames at CPU time sends 4×28KB contiguously in one transfer rather
   than 4 separate ones.

## Appendix: prior art we stand on

* **tetanes** (`lu-zero/tetanes`): the Rust NES core we'll wrap.
  MIT-licensed, actively maintained, Mesen-compatible.
* **PyO3**: the gold standard for Rust↔Python bindings; used by
  `cryptography`, `polars`, `ruff`.
* **maturin**: PyO3's build tool, produces real wheels.
* **nes-py** (Kautenja): what we're replacing. Credit where due: it
  was the right thing in 2019. It has aged out.
* **stable-retro**: the existing libretro-based alternative; arm64
  wheel is broken, build-from-source works but is heavy (10–20 min
  and a pile of brew deps). We do less and do it more cleanly.

Nobody has shipped a well-maintained, arm64-native Python NES
emulator with audio in 2026. The hole is there; this project fills it.
