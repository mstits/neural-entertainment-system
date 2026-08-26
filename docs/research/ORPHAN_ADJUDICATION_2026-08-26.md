# The Orphan Adjudication

*2026-08-26. A ~600-agent campaign across roughly a dozen workflows ran
against this repo overnight. Most workflows committed their own work. This
document is about what they left behind: 19 modified files and 3 untracked
ones sitting in the working tree with no owner coming back for them.*

*This is the sweep that collected them. It is also, more usefully, the
record of a process gap that will recur at this campaign size unless
something is built to close it.*

---

## 0. The headline, stated plainly

| | |
|---|---|
| Orphaned paths triaged | **21** |
| Landed (committed, with a test that discriminates) | **17** |
| Reverted | **2** (1 whole file, 1 half of a bundled file) |
| Handed back to a live owner, uncommitted | **2** |
| CORE-tier changes that had no test when found | **1** (`cpu.rs`) |
| CORE-tier changes that a test disproved | **1** (`latent_cells.py`) |

Two of the twenty-one were not orphans at all — they belonged to a workflow
still editing them. One was a silent regression wearing a green test. One
was a real fix bundled with a real defect in the same file. The other
seventeen were good work that would have been lost at the next checkout.

---

## 1. Why anything was in the tree at all

The mechanism is worth stating precisely, because it is not misconduct by
any agent involved.

A fix agent produces a change. A verifier agent then commits a *scoped*
subset of the tree — correctly refusing to stage files outside the change
it was asked to land. The verifier journals the refusal and exits. The fix
agent has already exited. The file stays modified, unstaged, and unowned.

The clearest instance is `nes_core/src/cpu.rs`. Verifier
`a552e53db755c7f9e` journaled:

> HELD BACK, not landed ... carries an out-of-scope change: a new
> `i_poll_latch` field ... they are simply someone else's to commit.

That judgment was **correct**. A verifier landing a CPU interrupt-timing
quirk it was not asked to review, did not audit, and could not vouch for
would have been the worse failure. The same verifier held back
`metrics_sink.py` on the grounds that "the defect is not closed
end-to-end" — also correct, and also true (see §4).

So: every individual decision in the chain was defensible, and the
aggregate outcome was still a tree full of unowned work. That is the
finding. It is structural, not behavioral.

---

## 2. The hazard runs in both directions

This matters because the failure is symmetric, and only one direction is
intuitive.

**Direction 1 — good work rots.** An unstaged modification survives until
someone runs `git checkout`, `git stash`, or a worktree reset. Untracked
files are worse: `tests/test_ppo_updater.py` and `tests/test_watchdog.py`
were the *proofs* for two real fixes, and an untracked test disappears
leaving no diff to remind anyone it ever existed.

**Direction 2 — bad work launders itself in.** An unowned modification
that nobody reverts is still sitting there the next time an unrelated
agent runs `git commit -a` or stages a directory. It rides in under
someone else's message, with someone else's review, attributed to someone
else's change. This repo has been burned by exactly this before.

`video_sink.rs` is the worked example of direction 2, and it is the reason
"passes the existing suite" is not the bar.

---

## 3. The two things that did not land

### 3.1 `nes_core/src/sink/video_sink.rs` — REVERTED whole

**What it claimed:** a `pending_emphasis` field so `set_emphasis` stages
the PPUMASK emphasis bits and `write_frame` promotes them afterward,
deferring emphasis by one frame so a value written during vblank cannot
retint the frame already rendered.

**What it actually did:** disabled color emphasis entirely.

The staging latch can never fire in production. All three construction
sites build a *fresh sink per frame* — `pool.rs:201`, `python.rs:940`,
`python.rs:1056` each do
`let mut video = Xrgb8888VideoSink::new(&mut self.video_buf);` inside a
one-frame advance — and `reset_frame_written` has **zero call sites
repo-wide** (verified by grep during this sweep, not taken on trust). A
sink instance therefore renders exactly one frame and is dropped.

Per frame, in order: `emphasis = 0` and `pending_emphasis = 0` at
construction; `ppu.rs` calls `set_emphasis(e)`, which now writes only
`pending_emphasis`; `write_frame` renders using `self.emphasis`, still 0;
the promotion `self.emphasis = self.pending_emphasis` then runs and dies
with the sink. Emphasis is permanently 0 on the training-pool observation
path and on the python/parity path.

The pre-change code applied emphasis correctly, possibly one frame early —
which may well be a real defect. This traded "one frame early" for "never
applied." That is strictly worse.

**The part worth internalizing:** its two tests passed. They passed
because they called `write_frame` twice on one sink instance — asserting
"the staged value is active starting here" — encoding an object lifetime
that production never produces. And `make parity` stayed at the known-good
146, because emphasis is 0 for nearly every frame and the base-palette
path is byte-identical either way.

A green test and a green parity suite both certified a regression. The
thing that caught it was reading the call sites.

The author's own returned text conceded the change was "a compensating
latch confined to this file, not a fix to the actual root cause
(`ppu.rs:2140-2142` sampling PPUMASK after the vblank window)." It was
right about that. If the one-frame-early misattribution is worth fixing,
the fix belongs in `ppu.rs`, latching emphasis when the frame's pixels are
produced.

### 3.2 `src/training/latent_cells.py` — HALF REVERTED

This file bundled two independent changes with very different evidence.
They were split.

**Kept (landed):** `reinit_pass_count += 1` on the `n_dead == 0`
early-return path of `_reinit_dead_codes`. One line, correct — the
early-return already performs the meaningful `_usage_since_reinit.zero_()`
side effect, so callers must be able to tell that a pass ran. Verified
both ways: against pure HEAD the shipped test fails `assert 0 == 1`.

**Reverted:** a per-slot `_codeword_generation` counter folded into
`encode_to_cell`'s return value as `raw_id + generation * codebook_size`.

The motivation was legitimate. `_reinit_dead_codes` overwrites a slot that
was live in an earlier era, so a raw index genuinely is not a permanent
identity. But the fix broke the invariant that a cell id lies in
`[0, codebook_size)` — which `_VQVAE.quantize`'s own docstring states, and
which three consumers depend on as a denominator.

Two independent adjudicators measured the harm, by different routes, and
agreed:

- At production `K=512`, 20 reinit passes: max cell id **10364**, distinct
  ids **680 > 512**, `occupancy(log, 512)` reports **1.0000** while live
  occupancy is 0.9570, and `dead_code_count` reports **0** while 22 slots
  were never used at all.
- Single-slot stress, 600 revisits: log occupancy reports **1.0** against a
  live occupancy of **0.00195** (1/512), and
  `check_discovery_kill_criterion` **never fires** on a run that visited
  one state region 600 times and discovered nothing.

That last one is the sharpest. These ids are the Go-Explore cell identity.
The change makes a codebook report *fully saturated* and *zero dead*
regardless of actual state, and it disarms a pre-registered kill criterion
on its own worst case. A pre-registered kill that cannot fire when it
should is worse than no kill — this is the false-capability-signal class
this project has been burned by repeatedly.

The shipped test passed only because it used a single reinit pass on an
8-slot codebook, and it asserted `occupancy(log, 8) == 2/8` where exactly
one slot was ever selected — i.e. it blessed the divergence rather than
testing it. It was **shaped to confirm the fix rather than to stress it**,
which is the second distinct way a green test misled this sweep.

A correct version would return `(slot, generation)` or make the log
diagnostics generation-aware. That is a design change, not a two-line
patch. Filed, not built.

Blast radius today is zero — `latent_cells` has no importer outside its own
test; it is the shelved v23 Rank-3 module. The hazard was latent and would
have landed the moment it was wired.

---

## 4. What landed, and what proves each one

Nine commits, grouped by subsystem. Every one carries a test that was
verified to **discriminate** — to fail with the change reverted and pass
with it present. That is the bar this sweep applied, and it is higher than
"the suite is green."

### `nes_core` — CPU

`nes_core/src/cpu.rs` carried two unrelated changes:

1. **`i_poll_latch`** — the 6502 CLI/SEI/PLP interrupt-poll latency quirk.
   Real hardware polls the interrupt line on those instructions' final
   cycle, so the mask write becomes visible one instruction later than a
   naive implementation makes it.
2. **`None`-opcode fallback** — replaces a flat `pc += 1` with a per-opcode
   operand-length match, and corrects a stale comment claiming the arm was
   unreachable.

Half 2 was independently re-verified by parsing the `OPCODES` table:
exactly 240 populated, 16 missing (`0x02/0x12/0x22/0x32/0x42/0x52/0x62/
0x72/0x92/0xB2/0xD2/0xF2` plus `0x93/0x9B/0x9F/0xBB`). The arm is live
code; the old "unreachable" comment was simply false.

Half 1 arrived with **no test of its own**, and there is no blargg
`cpu_interrupts_v2` ROM in `roms/` to validate against. Given that this
project's two worst recent bugs were both CPU/mapper timing — the ASM-CPU
PPU/NMI batching bug that made Mario fall through floors, and the MMC1
restore write-drop that silently killed Zelda and Journey to Silius — "not
harmful by available evidence" was not good enough to land it.

So six adjudication tests were written for it, covering all four documented
directions of the quirk plus two negative controls:

| Test | Claim |
|---|---|
| `..._cli_defers_service_by_one_instruction` | IRQ not taken at CLI's own boundary; pushed PC `$0203`, not `$0202` |
| `..._sei_cannot_mask_an_irq_pending_at_its_final_cycle` | IRQ *is* still taken at SEI's boundary — a live-flags implementation swallows it |
| `..._plp_defers_service_like_cli` | PLP pulls I in its final cycle; same one-instruction deferral |
| `..._survives_a_mid_cli_savestate_round_trip` | latch is unserialized and re-derived from `state.flags.i` — pins the MMC1-restore failure class |
| `..._rti_restores_i_with_no_delay` | **negative control.** blargg: "RTI affects IRQ inhibition immediately" |
| `irq_entry_masks_a_held_line_and_does_not_re_enter` | **negative control** for the stack-drain class (Crash 'n' the Boys) |

The pushed return address is the load-bearing receipt: it names which
instruction boundary the IRQ was taken at, so each test asserts a
cycle-level outcome rather than "an IRQ appeared." No test writes
`i_poll_latch` directly — every setup reaches its state through the real
per-cycle `Cpu::tick` pipeline, so each assertion is behavioural rather
than a restatement of the implementation.

Discrimination verified both ways: reverting only the behavioural line in
`poll_interrupts` fails tests 1-4 with their exact expected messages, while
**both negative controls still pass** — proving they are genuine controls
and not co-varying with the fix.

**Blast radius, characterized.** The changed expression differs from the
old one only when `irq_line_low` is true *and* `flags.i != i_poll_latch` —
i.e. only during the final cycle of CLI/SEI/PLP while an IRQ is asserted.
Every ROM with no live IRQ source (SMB and the whole NROM solver-tape
corpus) is bit-identical, which is what makes the parity suite and the
`asm_vs_slow_*` trajectory-equality tests meaningful negatives here rather
than vacuous ones.

**Residual risk, recorded not hidden.** Unlike its direct precedent
`nmi_poll_latch`, this is applied unconditionally rather than behind an
`hw_*` flag. That was judged acceptable because the risk classes differ:
NMI timing moves every frame of every game, whereas IRQ poll latency only
fires on the CLI/SEI/PLP-with-pending-IRQ coincidence. One genuine
residual: in the aarch64 ASM lane, `poll_interrupts` runs once per *block*
(64 cycles for NROM), so the latch degrades from "as of the previous cycle"
to "as of the previous block boundary." That lane already samples the IRQ
line only at block end and delivers interrupts up to 64 cycles late, so
this is the lane's existing granularity rather than a new error class. If
zero default-behaviour delta is ever wanted, gating behind
`hw_irq_poll_timing` remains a one-line change.

### `nes_core` — APU

`nes_core/src/apu.rs`: a `$4010` write with bit 7 clear now also clears
`dmc.irq_pending`, per NESdev APU_DMC ("If clear, the interrupt flag is
cleared"). Closes a real hang class — a ROM that acks the DMC IRQ by
rewriting `$4010` rather than reading `$4015` previously left the IRQ line
permanently asserted, so every `RTI` re-entered the handler forever. The
direction of the change is strictly safe: it can only ever clear a pending
IRQ that hardware also clears, never raise one.

### Training-loop correctness

`src/training/ppo_updater.py`: the RND `update_normalization` call now
masks by `valid_buf`, so post-done frozen padding no longer feeds the
obs_rms/reward_rms Welford stats. When an env dies, `trainer.py`'s
`active_in_iter` freeze stops stepping it, so every remaining `obs_buf`
row is a byte-identical copy of the death frame. Folding those duplicates
in mis-scales the intrinsic bonus divisor and every subsequent
`_normalize_obs` call for the rest of the run — and the bias is worst
exactly in late-training rollouts where most envs die early. The fix makes
this call consistent with the two sibling consumers of the same rollout in
the same function, both of which already index by valid rows.

`src/training/wall_taxonomy.py`: `frozen_windows` now reads the last
record rather than the max over the trailing window. `stall_flat_windows`
resets to 0 the instant a new cell is recorded, so taking the max let a
single historical stall peak outlive a genuine mid-window recovery — a
search actively discovering new cells read as permanently frozen and got
misclassified. Every sibling field in the same function already reads only
the last record; `max(...)` was the lone inconsistent case. No existing
test caught it because every prior test applied a *constant* stall value
across the synthetic series, under which max and last are identical.

`src/models/tile_policy.py`: `build_tile_policy_from_checkpoint` now
raises when `load_weights` is set but no usable state_dict resolved,
instead of silently returning a freshly-initialized random policy. A
silently random policy returned from a checkpoint loader is the exact
mechanism that fabricates eval numbers — a confident-looking result with
no exception and no log line, indistinguishable from a genuine capability
finding. An abort is recoverable and visible; a laundered eval number is
neither.

### Run bookkeeping and observability

`src/training/run_manifest.py`: a second same-run writer can no longer
clobber a correct `rom_md5` with `null`. `src/training/config_schema.py`:
registers `episode_metrics` and `tensorboard`, two keys `trainer.py`
already reads, which `--strict-config` was aborting launches over.
`src/training/metrics_sink.py`: a non-serializable value in one
generation's dict no longer aborts `emit()` before the queue and
TensorBoard fan-outs run. `src/gui/watchdog.py`: per-reason stack-dump
cooldown, so a "growth" dump can no longer suppress a strictly more urgent
"threshold" dump; and a failed `ps -M` now omits `threads` rather than
recording a plausible-looking `0`.

**One follow-up is filed, not done.** `metrics_sink.truncate()` gained a
`resume: bool = False` parameter that is currently **inert** — the only
caller, `trainer.py:1947`, still calls it with no argument, so a resumed
run still wipes the canonical `metrics.jsonl` out from under a continuous
generation counter. The withholding verifier was right that the defect is
not closed end-to-end. The remaining work is one line at that call site:

```python
truncate(resume=bool(resume_from and Path(resume_from).exists()))
```

It was not done here because `trainer.py` belongs to a concurrently
running workflow. The parameter and its test landed anyway: both default
to today's exact behavior, so they cannot regress anything, and the test is
what will prove the follow-up correct when someone makes it.

---

## 5. The two that were not orphans

`scripts/onboard_game.py` and `tests/test_onboard_game.py` were left
**uncommitted and un-reverted**, and handed back.

They belong to the lives-nomination-hardening workflow that was editing
`scripts/discover_observables.py` during this sweep. The coupling is
direct and checkable: `onboard_game.py` reads
`findings['hp_lives']['behaviour_gate']` and `gate['rejections']`, both
produced by `discover_observables.py`; it reads `BEHAVIOUR_N` from that
same module; and `git show HEAD:scripts/discover_observables.py | grep
BEHAVIOUR_N` returns nothing — neither `BEHAVIOUR_N` nor `behaviour_gate`
exists in HEAD. It is a consumer of work that has not landed yet.

The change itself is sound and was verified to discriminate (4 failed / 1
passed against HEAD's `onboard_game.py`, where the 1 pass is the intended
negative control). Its `getattr(d, "BEHAVIOUR_N", 0)` default is
load-bearing: without it the `AttributeError` would be swallowed by
`_try_probe_budget`'s bare `except` and silently drop the *entire* probe
budget from every generated profile's provenance.

Landing it independently would strand a half-wired consumer in HEAD.
Reverting it would destroy work its owner is still building on — the
"good work rots" hazard, executed by the very sweep meant to prevent it.
Neither is right. **Handing it back is the disposition**, and it is a
disposition, not a deferral: it has a named owner who is live.

**One gap for that owner.** The new note is nested under
`if hl.get("addr") is None`, so the `hp_death_proxy` case is silent. When
discovery falls back to a HUD-health proxy *and* every lives nomination was
rejected, `emit_solve_yaml` still warns that the byte "is the HUD-health
death proxy, which this gate does not validate — CONFIRM before trusting
it", but the onboard report's notes say nothing. Same hazard class, one
branch short. Recommend widening that branch when the file is committed.

---

## 6. Process observation — the part worth keeping

A campaign this size leaves unowned work in the tree. This is not a bug in
any agent; it is a property of the topology.

The chain is: **fix agent produces → verifier scopes → verifier exits →
fix agent has already exited → file is unowned.** Every step is correct in
isolation. The verifiers that declined to stage out-of-scope work were
right to decline, and should keep declining. Nothing in the campaign was
responsible for collecting what they set down.

That is the missing role. Not stricter verifiers — **a collector.**

Three observations from running it once by hand:

**1. The green-suite bar is the wrong bar for orphans.** Two of the three
defects found here were wearing passing tests. `video_sink.rs` passed
because its test constructed an object lifetime production never produces.
`latent_cells.py` passed because its test used a codebook size and pass
count too small to reach the failure regime — it was shaped to confirm the
fix rather than to stress it. Both also left `make parity` at the exact
known-good baseline. The bar that worked was **discrimination**: revert the
change, watch the test fail, restore, watch it pass. Applied to all 21
paths, it caught both, and it promoted `cpu.rs` from "not proven correct"
to landable.

**2. Bundled files need splitting before judging, not after.** Two of the
three CORE-tier files carried two unrelated changes each. `cpu.rs` bundled
a proven zero-risk fix with an unproven timing quirk; `latent_cells.py`
bundled a correct one-liner with a defect that disarms a pre-registered
kill criterion. A single verdict on either file would have been wrong in
one direction or the other — landing a defect, or reverting a good fix.
The unit of adjudication is the *change*, not the file.

**3. Untracked test files are the highest-attrition artifact in the
tree.** `tests/test_ppo_updater.py` and `tests/test_watchdog.py` were the
sole proofs for two real fixes. A modified file at least shows up in
`git diff`; an untracked one vanishes at the next checkout leaving no trace
that it existed, taking the evidence for its fix with it. Any future
collector should treat `git status --porcelain` output starting with `??`
as the top-priority queue.

**The cheap version of the fix:** a campaign is not finished when its last
workflow commits. It is finished when `git status` is clean. A terminal
sweep that reads the tree, splits bundles into changes, demands a
discriminating test per change, and dispositions every path — land, revert,
or hand back to a named live owner — is roughly the cost of one workflow
and closes both directions of the hazard at once.

---

## 7. Disposition table

| Path | Disposition | Why |
|---|---|---|
| `nes_core/src/cpu.rs` (`None`-fallback) | LAND | `OPCODES` table re-parsed; arm is live code, old comment false |
| `nes_core/src/cpu.rs` (`i_poll_latch`) | LAND | 6 adjudication tests written; discriminates both ways; 2 negative controls hold |
| `nes_core/src/apu.rs` | LAND | Spec-cited; can only clear an IRQ hardware also clears |
| `src/training/ppo_updater.py` | LAND | Frozen padding skewed RND Welford stats |
| `tests/test_ppo_updater.py` | LAND | Untracked proof for the above |
| `src/training/wall_taxonomy.py` | LAND | Stale stall peak outlived recovery; misclassified the wall |
| `tests/test_wall_taxonomy.py` | LAND | First non-constant stall series in the file |
| `src/models/tile_policy.py` | LAND | Silent random policy fabricates eval numbers |
| `tests/test_tile_recurrent_policy.py` | LAND | Pins the present-but-empty `{}` case |
| `src/training/run_manifest.py` | LAND | Second writer clobbered `rom_md5` with null |
| `tests/test_run_manifest.py` | LAND | Reproduces the real two-writer sequence |
| `src/training/config_schema.py` | LAND | Keys `trainer.py` already reads; strict mode aborted launches |
| `tests/test_config_schema.py` | LAND | Strict-mode test is the load-bearing one |
| `src/training/metrics_sink.py` | LAND | JSONL guard effective now; `resume` inert pending a `trainer.py` one-liner |
| `tests/test_metrics_sink.py` | LAND | Asserts queue fan-out survives the dropped line |
| `src/gui/watchdog.py` | LAND | Shared cooldown suppressed the more urgent dump |
| `tests/test_watchdog.py` | LAND | Untracked proof; fake clock, no sleeps |
| `src/training/latent_cells.py` (`reinit_pass_count`) | LAND | One line; fails `0 == 1` against HEAD |
| `src/training/latent_cells.py` (`_codeword_generation`) | **REVERT** | Breaks `[0, K)`; occupancy pins at 1.0; disarms the kill criterion |
| `tests/test_latent_cells.py` | LAND | Green at 37 once the fold is reverted; the disproof is the deliverable |
| `nes_core/src/sink/video_sink.rs` | **REVERT** | Latch can never fire; disables emphasis entirely |
| `scripts/onboard_game.py` | **HAND BACK** | Live owner; consumes `BEHAVIOUR_N`, absent from HEAD |
| `tests/test_onboard_game.py` | **HAND BACK** | Rides with the file above |

Untouched throughout, by instruction and on purpose:
`scripts/discover_observables.py`, `tests/test_discover_observables.py`,
`tests/test_lives_behaviour_gate.py`.
