# Project status — authoritative snapshot, 2026-07-20

> **Addendum, 2026-08-08:** this snapshot is now stale as a status
> report. For current plan, priorities, and gates, read
> `docs/proposals/STRATEGY_2026-08-08.md` ("Final Determination, post
> red-team") — it supersedes the roadmap/priority sections below. The
> scorecard, drift-and-correction record, and the 2026-08-06 addendum
> underneath remain accurate as history and are not retracted.

One honest place to understand where this project actually stands, after
the drift-and-correction, the stability audit, and the PhD/E8 review.
Supersedes scattered claims in older campaign docs where they conflict.
Reads against `CLAIMS.md` (the ledger + tier policy) as ground truth.

## What this project is

An AI tool that **learns** to play NES games, on one Apple-Silicon
laptop, watchable live and reproducible by anyone who compiles it. Two
assets sit under that banner, and the distinction is the whole story:

1. **A learning engine** — PPO + Go-Explore + RND + self-imitation on a
   Rust NES substrate. This is the product.
2. **A search-and-replay solver** — Go-Explore/beam finds a clearing
   trajectory, a BC pilot overfits it, a router replays it. Real and
   useful, but by `CLAIMS.md` it is **EXHIBITION**, not learning (the
   ALE "Brute", Machado 2018; Baumgarten's A*, 2009).

## The honest scorecard (PhD-AI + E8 review, 2026-07-20, verified)

| Dimension | Grade | One-line reason |
|---|---|---|
| Emulator core | B+ → **A−** | byte-exact (nestest+CYC), Mesen lockstep, 793/794 boot, save/restore now *measured* at ~1–2 µs; blargg gauntlet still unrun |
| Rust maximization | B− | strong substrate; the RL system is Python; the actor/learner migration (~1.7×, update-bound) is un-built |
| Engineering quality | C+ → ~B+ | audit fixed the crash/corruption trail + guards; monolith not yet decomposed (plan written) |
| Demonstrated capability | **C−** | search solves ~15 levels; 3 levels learned in-distribution; **0/50 under the honest sticky protocol** |
| Broad generality | D+ | 16 hand-authored per-game reward structs; one game has any non-zero result, via search |
| Research novelty | D+ | competent reimplementation on a *solved* benchmark; no algorithmic novelty |
| Groundbreaking | C− | the measurement culture is above field norm; a measuring stick with no passing measurement is not a contribution |

**Composite ≈ high-2s/low-3s (B−/B).** A 4.0 is not honestly reachable:
novelty and groundbreaking are capped at ~B by the solved-benchmark
choice, and capability/generality need long, uncertain training. The
honest ceiling if everything achievable lands is ~3.2–3.5. We do not
game the grader to inflate this (that is the meta-Brute).

## The single most important number

`runs/composite_honest_pair/eval.jsonl` → **0/50**. Cold power-on,
sticky-actions 0.25 + start-jitter, single-life, 50 episodes × 2 seeds:
the best LEARNED policy dies in 1-1 every time. Everything that "clears
Worlds 1–3/1–4" above that is EXHIBITION (search replay) or a specialist
that clears in-distribution and evaporates under 25% sticky.

## What is genuinely real (keepable)

- **The emulator.** From-scratch cycle-accurate Rust NES core; nestest
  byte-exact incl. CYC; 31-ROM Mesen-oracle lockstep; 146 parity tapes;
  793/794 library boot; 36 mappers; save/restore **measured** save p50
  ~1.75 µs / restore ~1.0 µs (`scripts/bench_save_restore.py`,
  `runs/emulator_bench_2026-07-20.json`); ASM 6502 core (perf path,
  240M-instruction differential fuzz vs the interpreter reference).
- **The measurement culture.** Two-ledger LEARNED/EXHIBITION split, Tier
  0–3 injection taxonomy, provenance allowlist enforced at the demo
  loader + `make provenance-check` (content-hashed, bypass-resistant),
  ALE sticky protocol with self-replaying receipts, and the discipline
  to publish the 0/50. This is above the field norm.
- **Three learned levels.** 1-1/1-2/1-3 clear deterministically
  in-distribution — real learning, categorically different from replay.

## The drift, and the correction (recorded so it is not repeated)

The campaign drifted into (a) presenting BC-replay pilots as "the AI
playing" and (b) using the game's disassembly + level maps + hand-driven
inputs to route 4-2/4-4 — which the founder correctly called cheating.
Correction (binding, see `CLAIMS.md` + memory
`feedback-no-game-internals-no-cheating`): Tier-3 bans disassembly/maps/
hand-driven inputs; tainted artifacts quarantined
(`checkpoints/QUARANTINE_tier3/`); the manifest split into
`composite_world1.yaml` (EXHIBITION, labeled) and
`composite_learned.yaml` (trained nets only).

## Stability audit (2026-07-20) — shipped, all suite+parity green

75-agent audit, adversarially verified. Fixed with regression tests:
emulator **use-after-free** on state restore (`chr_cache_ptr`);
supervisor **blind to Rust PanicException**; **non-atomic checkpoint
save** + non-tear-tolerant resume (silent restart-from-random);
`catch_unwind` gated off in prod; unguarded `apply_state` (GUI hard
crash); **systemic NaN backstop** (isfinite-loss guard at all 3 PPO
sites; GA sticky clamp; poison-save guard extended to RND+optimizer);
`seq_clear` warp/world guard; receipt-jitter self-replay; off-map
episode isolation; fresh-start checkpoint fence; run-lock; bypassable
provenance gate. Details: `runs/stability_audit_FULL_2026-07-20.json`.

## Capability attempt #1 — FAILED (honest negative result)

Sticky-from-cold PPO + adaptive entropy floor on the warm 1-1 net
**degraded** it: 1.0 deterministic → **0/20 deterministic** (harness
verified: original net 10/10 same harness). The entropy floor worked as
a mechanism (no entropy collapse) but the *approach* was wrong — training
a whole level from cold under sticky+forced-entropy pushes the policy off
its learned solution. See
`project-capability-attempt1-failed-2026-07-20`. The correct build is the
**backward-algorithm / self-imitation-under-sticky** robustification
(anchor to the working solution, harden against perturbation), now in
progress via `scripts/robustify_level.py --sticky-prob`.

## Forward roadmap (ranked, honest)

1. **Capability: pass the honest test on ONE level (1-1).** Self-imitation
   robustification under sticky (collect the net's own sticky-surviving
   clears, BC them; backward ladder if cold-sticky clears are ~0).
   Target ≥0.6 sticky-0.25, 2 seeds. The one number that flips "0
   demonstrated" → "learned play demonstrated". Uncertain, multi-attempt.
2. **Rust A−: decoupled GIL-free actor/learner** (~1.70×, update-bound
   ceiling confirmed by `runs/throughput_split_2026-07-20.json`). The one
   migration the "as much Rust as possible" value demands.
3. **Engineering A−: decompose the 3,130-line `_run_vanilla_ppo`** per
   `docs/proposals/trainer_decomposition_plan.md` (characterization tests
   first, one module per commit).
4. **Emulator A: run the blargg/Mesen accuracy gauntlet** (needs the test
   ROMs locally) + publish the pass table; ship as a clean crate.
5. **Novelty (the only earnable "first"): a single LEARNED policy clearing
   multiple SMB worlds from power-on under the ALE protocol** — gated on
   (1). Or the determinism-vs-sticky generalization study as a workshop
   paper. Or the emulator-crate systems contribution.

## How to verify any claim here

`make provenance-check` · `make parity` · `make test` ·
`scripts/eval_composite.py --manifest configs/composite_learned.yaml
--sticky-prob 0.25 --start-jitter 16 --episodes 50` (the honest number) ·
benchmarks in `runs/*_2026-07-20.json`.

## Addendum, 2026-08-06 — the direct-PPO learning track is shelved

The framing above ("a learning engine ... this is the product") describes
the intent as of 2026-07-20, not the last 17 days of actual work. On
2026-07-30 the direct-PPO track (PR-MDP → SHAPO/SAM-PPO on 1-2) was
stopped after its two pre-registered gates both failed — six method
families total have now walled on learned-sticky-1-2 — with no fourth
pivot recommended. `trainer.py`'s PPO path has had zero commits and no
training run since (`vanilla_ppo_iter_25630.pt`, 2026-07-30 22:39, is
still the newest PPO checkpoint anywhere). Every commit since has been
search/solver/show/adapter work — see the `[GAME COMPLETE 2026-07-27]`,
`[Multi-game product vision 2026-07-27]`, and world-by-world memory notes
for what actually shipped in that window.

This is not a retraction — the item-2 forward-roadmap goal (a learned
policy passing the honest sticky test) is still the ceiling this project
would need to hit to earn the grades above. It just hasn't been the
active work, and won't be revisited without a genuinely new idea per the
2026-07-30 call. Read this addendum before assuming the scorecard above
describes current effort, and update it again the day the learning track
either resumes or is formally retired rather than paused.
