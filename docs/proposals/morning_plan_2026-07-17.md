# Morning execution plan — 2026-07-17

Synthesized overnight from ten design/scouting lanes (demo-augmented PPO,
DAgger seam repair, npz root-cause, gx-keyed routing, policy-based returner,
PLR, decoupled actor/learner, kickstarted distillation, Worlds 3-8 scout,
TAS receipts pipeline). Full lane outputs in the session workflow journal.

ning execution plan",
  "agentCount": 11,
  "logs": [],
  "result": {
    "plan": "All load-bearing claims verified on disk: npz corruption confirmed (obs_0 535 rows / 1 unique, obs_1 582 rows / 1 unique — L3's finding is real), all three gx seed states + ep231 actions exist, and L1's trainer plug points (`ppo_losses` at trainer.py:5862, `_burst_seeds` at 6894) are exact. Synthesis follows.

---

# THE MORNING EXECUTION PLAN
**Mission: (P0) close the 2-1 seam · (P1) cold chain to 3-1 · (P2) full-game run setup with TAS-grade receipts.**

**Coordinator's read:** P0 is gated by a 15-minute fix, not research. `demos_2_1_runway.npz` is corrupt — L3 root-caused it (view-aliasing into the reused `TileFeatureStacker._out` buffer at `gru_weld_2_1.py:66`), L2 independently confirmed, and I re-verified on disk this session (1/535 and 1/582 unique obs rows). Every demo-consuming design gates on fix + re-collect. The seam closer is then a two-layer move: **L1's DQfD demo-anchor (policy layer, patch-ready) produces a front half that survives gx~1658; L4's gx-routing (deployment layer, pure-unit-ready) hands it to the already-welded post-barrier net at gx1963.** L2 is the decision instrument, not a day-one build: its alias diagnostic on clean demos decides whether a feed-forward anchor can cross 1658 at all, and pre-names the contingency. L10 receipts build in parallel (emit_fm2 is offline-testable). L5 parks itself; L8 parks on unmet prerequisites but its baseline measurement runs tomorrow; L7 parks because it rewrites the same trainer lines L1 edits, on mission day.

## 1. HOUR-BY-HOUR BUILD ORDER

| Time | Build | Status | Why now / evidence |
|---|---|---|---|
| 08:00–08:20 | **GATE — L3 fix**: `np.array(..., copy=True)` at `gru_weld_2_1.py:66`; delete poisoned npz; re-collect; land L3's uniqueness assert in the collector **and** inside L1's `DemoBank.from_npz` (L1's own assert checks shape only, not uniqueness) | READY (one line + re-collect) | Verified corrupt on disk; gates L1/L2/L6. Also grep `np.asarray(obs` repo-wide per L3's closing note |
| 08:20–09:00 | **Alias diagnostic on clean demos** (L2 Part-A empirics brought forward): group identical 700-d rows, measure conflict mass, bucket by gx. In background, launch **sticky-0.25 W1 composite baseline eval** (`eval_composite --sticky-prob 0.25 --start-jitter`, N≥100 fixed seeds) | READY (harness already wired, L8 §1) | The diagnostic pre-positions the 15:00 fork (L2 predicts ~19% conflict, concentrated at the piranha wait). The sticky baseline is L8-prereq #1 and the honest denominator for every claim this week (roadmap L0) |
| 09:00–12:00 | **L1-b DQfD demo-anchor** — new `demo_bank.py`, `demo_anchor_loss` in `ppo.py`, 8-line interleave at trainer.py:~5869 (verified: `ppo_losses` at 5862), knobs ~L451. Coef 1.0→0.02 over 400 iters, margin 0.8, mb 256, entropy 0.01→0.002. **Launch the seam run** from the 2-1 curriculum warm-start | READY (exact patch given) | Only design putting demo CE and reward gradient in the same backward pass — targets exactly the 0.81-cap/greedy-death failure the out-of-loop BC replay cannot fix (L1). Smallest diff of the three L1 options |
| 09:00–12:00 (parallel worktree) | **L4 gx-routing** — resolver + Schmitt deadband + forward-latch + controller branch + unit tests + **capture keyed on `active_key`** (banks `handoff_2-1@gx>=1963.state`). First: 10-min empirical hazard-map check — replay a demo logging gx of the 1658 compound vs the 1963 line, because L4 §0 and §4 disagree on which side the compound sits, and that decides `reset_on_entry` | READY (pure functions, no training run needed; L4 §8 test plan) | Deployment layer for ANY front half; reuses the welded suffix instead of re-learning it; honest-mode surface (`intra_level_switches`) ships with it |
| 12:00–13:00 | **L10-1 `emit_fm2.py`** + round-trip unit test against `convert_fm2.py`'s `_FM2_BIT_AT_POS` | READY (offline, zero emulator dep) | Unblocks the whole receipts chain while the seam run trains (L10 build order #1) |
| 13:00–15:00 | **Unified harvest tool**: replay an action sequence from an entry state → dump (obs, act, gx, savestate) every 48–64 px | ITERATE-light (small new tool) | One tool, three consumers: JSRL rungs (L1-c), PLR corpus materialization (L6 §3), DAgger reference corridors (L2 B-2). Also verify the robustify harvest driver loops on `keep_exploring` (L5's quick-win — shorter demos weld easier) |
| 15:00–16:00 | **Seam-run checkpoint + FORK.** Green (greedy max-gx advancing past 1658): wire **L1-c JSRL** branch at trainer.py:~6908 (verified `_burst_seeds` at 6894) on harvested rungs, `jsrl_frac 0.75`, start just before 1658. Red: fork on the 08:30 diagnostic — drift regime → build L2's `dagger_repair`; aliasing regime → front-half contingency (enemy-phase/timer scalar or recurrent clone via lifting `robustify_level.py:307`, L2 B-5) | JSRL: ITERATE-light; DAgger: ITERATE (new oracle/search code) | JSRL concentrates the exploration budget on the exact ~200 px around 1658 (L1-c); L2's B-5 byte-identical-obs check is the drift-vs-aliasing arbiter |
| 16:00–18:00 | **L10-2/3** — default-None capture hook in `run_episode`; `record_receipt.py` boot-prefix + NCST RAM-equivalence gate (`get_ram_range` byte-compare, abort on diff) | 2: READY; 3: ITERATE-light (boot-prefix equivalence needs empirical confirmation) | The frame_skip-held-buttons fact makes per-frame reconstruction exact (L10 linchpin); power-on prefix is the credibility move vs an NCST-start movie |
| 18:00–20:00 | **Chain prep 2-2→3-1**: rungs for 2-2 (water prior exists per L9), 2-3, 2-4 (standard castle weld), 3-1 (**pre-check hammer-projectile exposure** — L9's only MED-risk gap on the path to 3-1; night palette debunked). Overnight: seam consolidation (entropy decay) + chain robustification if green + Tier-A `replay_actions` verify on any green segment | ITERATE (standard weld loop) | L9: nothing between 2-1 and 3-1 needs a new system; 3-1 is the first Hammer Bros |

**Ready-to-implement today:** L3 fix, L1-b anchor, L4 routing, L10-1/2, L6 sampler core (if slack). **Need iteration:** L2 DAgger loop, L1-c JSRL (harvest prereq), L10-3+, L6 trainer integration (4 touch points), L9 builds A/B, L7 phase-1.

## 2. THIS WEEK vs PARK

**Implement this week, ranked:**
1. **L3 fix** (D0, gate) — everything demo-based depends on it.
2. **L1-b anchor** (D0) — P0 policy layer.
3. **L4 gx-routing** (D0) — P0 deployment layer + honest reporting surface.
4. **L10 receipts 1–5** (D0–D2) — Tier-B FCEUX verification of the W1 run D2; pin the verifying FCEUX/PPU version in the campaign log; `--render-mp4` D3.
5. **Harvest tool + L1-c JSRL** (D0–D1).
6. **L2 DAgger repair loop** (D1–D2, fork-triggered) — `replay_corridor` lands with the harvest tool regardless, since it is the permanently-safe demo-bank builder (obs regenerated from actions kills the corruption class).
7. **L6 PLR** (D2–D3): sampler + corpus midweek; ladder-vs-PLR A/B as the midweek overnight run (Bet 3 now, L8's coverage prerequisite later). Startup-assert exclusivity with ladder/consolidate.
8. **L9 build A** (maze obs + loopback detector, D3–D4 start): required before W4; the area-byte curriculum is provably blind to loopbacks (L9) — a landmine, cheap to defuse early.
9. **L7 decoupled phase-1** (D4+, only after L1 lands; rebase on the anchor loop; guards + selftest A/B mandatory).

**Park, with named triggers:**
1. **L5 returner** — the lane's own DEFER verdict is correct: teleport-free *discovery* is already solved by save/load state, teleport-free *deployment* by backward robustification. Triggers to build: Bet-5 one-net endgame, M3 ROM-hack gauntlet, genuinely branching deployment. Its two quick-wins are already landed; keep them exercised.
2. **L8 distillation build** — prereq chain unmet: sticky baseline unknown (measured tomorrow), W1 sticky-green unverified, farm sampler absent (arrives with L6). Trigger: all three green. Realistic slot: next week.
3. **L2 extractor change** (absolute-x + enemy-phase scalar) — contingency only; it forks the obs contract. If triggered, do it as **one tile-v2 schema** folding in L9-A's maze fields (`$0745` LoopCommand, page, Y-target distance) and L9's hammer/offscreen channels — one migration, not three; per-net encoders keep existing checkpoints valid via L4's legality guard.
4. **L9 build B** (dynamic platforms + offscreen lookahead) — trigger: chain reaches 3-2/3-3. Note 3-3 is HIGH risk and arrives days after 3-1, so this un-parks fast.
5. **L7** until the seam ships (see conflict 3).

## 3. CONFLICTS / SYNERGIES

1. **gx-routing vs demo-PPO: we need BOTH — they solve different halves of 2-1.** Routing reduces P0 to \"front half survives 0→1963\" and reuses the welded suffix (L4 §0); the anchor is what gets a policy through 1658 (L1; L2 locates the death precisely there). Routing without a 1658-surviving net routes into a death; anchor without routing re-learns an already-welded suffix. Zero file overlap — run as parallel lanes.
2. **HARD GATE: L1 depends on L3.** L1's design trusts npz obs that are provably garbage on disk. Fix, re-collect, and add the uniqueness assert to `DemoBank.from_npz` before the bank ever builds.
3. **L1 vs L7 edit the same trainer lines (~5827–5901).** L1 adds 8 lines inside the minibatch loop; L7 extracts that whole block into `_learner_update`. Strict sequence L1 → L7 rebased; never parallel. Worktree lanes, no git stash.
4. **L1 vs L2: same target, different mechanism.** L2's B-5 diagnostic (byte-identical obs, different action) is the arbiter between \"anchor will converge\" and \"policy class/obs must change.\" Run it at 08:30, not after a wasted day.
5. **One harvest tool, three consumers** — JSRL rungs (L1-c), PLR corpus (L6 §3), DAgger corridors (L2 B-2). Build once at 13:00.
6. **L4's capture-on-`active_key` fix feeds the weld loop and L6's corpus** — it is how `handoff_2-1@gx>=1963.state` gets banked for the post-barrier re-weld.
7. **L6 is L8's coverage prerequisite** (uniform sampler is the MVP fallback) — build PLR once, it serves Bet 3 and Bet 5.
8. **Extractor coordination**: L2's de-alias scalars, L9-A's maze fields, L9's hammer/offscreen channels are all extractor extensions. One coordinated tile-v2 migration, or none.
9. **L4 internal discrepancy** (§0 puts the 1658 compound before the 1963 handover; §4 says the compound is entered *through* 1963) — resolve empirically before setting `reset_on_entry`; it decides the weld contract and the stack-carry choice.
10. **L9's loopback detector vs the area-byte curriculum**: the curriculum trigger cannot see maze loopbacks (`$006D` −4 with no area-byte change) — must land before any W4 chain work.

## 4. HONEST-MODE IMPLICATIONS (sticky-eval is the reported metric)

- **Measure the denominator first.** The composite's sticky-0.25 W1 number is currently UNKNOWN (L8 prereq #1, roadmap L0). Tomorrow's background eval makes every subsequent claim gateable. If it collapses under noise, publish it anyway and schedule weld hardening — honesty over optics.
- **gx-routing weakens the \"single policy\" claim and must say so** (L4 §7): surface `intra_level_switches` + per-seam handoff gx in `summarize`; every gx segment passes sticky-0.25 + start-jitter before it counts; report seam count and commit to it trending down (the gx seam is scaffold, not destination — standing goal to re-consolidate into one net once the 1658 compound is learnable).
- **demo-PPO**: demos are self-harvested wins, so no external-input caveat — but the reported number is cold greedy (+ sticky) eval with `demo_coef` at floor/zero, never training telemetry (the 07-15 lesson: name the harness; training clears ≠ cold greedy clears). Entropy 0.01→0.002 is what converts a stochastic clear into a deployable greedy one (06-25 lesson) — it is in the D0 config.
- **JSRL/PLR warm-starts are training-time teleports only**; the reported chain is cold power-on, forward-playing, no teleports (L5's discovery-vs-deployment distinction is the honest framing).
- **Receipts**: the fm2 is a sticky-0 deterministic artifact by construction (L10); publish the sticky-0.25 chain rate beside it and state the composite-of-specialists caveat in the claims table (Machado et al. 2018 is the citation for both L4 and L10's protocol).
- **Distill gate (when unparked)**: λ_k exactly 0 and sticky-0.25 vs the measured baseline, or the number is teacher-propped (L8).

## 5. RISK REGISTER

| # | Risk | L×I | Mitigation | Lane |
|---|---|---|---|---|
| 1 | Re-collected demos corrupt via another no-copy idiom | M×H | Uniqueness assert in collector + DemoBank; repo-wide grep of `np.asarray(obs` | L3 |
| 2 | Anchor run plateaus at gx1658 (aliasing regime; ~19% conflict mass predicted, concentrated at the piranha wait, unfixable by feed-forward + x) | M×H | 08:30 diagnostic pre-positions the fork; timebox to 15:00; contingency = phase/timer scalar or recurrent front-half (B-5) | L2 |
| 3 | gx handover OOD first frame (wrong reset contract) | M×M | Manifest-declared weld contract + encoder legality guard; resolve the §0/§4 hazard-map discrepancy empirically first | L4 |
| 4 | Trainer-loop merge collisions (L1/L6/L7 touch the same function) | H×M | Strict sequence L1→L6→L7; worktree lanes; no git stash | coord |
| 5 | fm2 desyncs in FCEUX (PPU timing) | M×M | NewPPU 0↔1 retry protocol; SMB is the RNG-friendly case; pin the verifying build | L10 |
| 6 | Boot prefix lands on a different state than the NCST warm-start | L×H | `get_ram_range` byte-compare hard gate, abort with diff | L10 |
| 7 | Sticky-0.25 baseline reveals the composite collapses under noise | M×H | Publish regardless; weld hardening before headline claims; distill stays parked | L8/L4 |
| 8 | Area-byte curriculum silently blind to maze loopbacks stalls W4+ | H×M (later) | Loopback detector (`$006D` −4) lands with L9-A before W4 work | L9 |
| 9 | PLR self-reinforcing scores / duplicate corpus states split mass | M×M | ρ≥0.1 staleness, EMA-not-raw scores, content-hash dedup | L6 |
| 10 | Decoupled repeats the debunked async_pipeline failure | M×H (later) | Parked past mission day; when built: frame-exactness + lag-bound + KL-abort guards, selftest-learning A/B is the merge gate | L7 |
| 11 | Compute contention (seam run + evals + A/Bs on one M4) | M×L | Day = seam run + cheap evals; consolidation and PLR A/B run overnight | coord |
| 12 | 3-1 Hammer Bros projectile invisible to the tile grid stalls the chain at the mission target | M×M | Pre-check exposure in the 18:00 block; small enemy-projectile slot add if missing (fold into tile-v2 if that migration opens) | L9 |

**Bottom line:** tomorrow ships four things — clean demos (L3), a demo-anchored seam run (L1), the gx deployment rail with its honest-reporting surface (L4), and the first two receipt deliverables (L10) — with the L2 diagnostic standing by as the fork arbiter and the sticky baseline measured so every number this week has a denominator.",
