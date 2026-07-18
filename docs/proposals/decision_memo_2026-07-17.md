unt": 6,
  "logs": [],
  "result": {
    "memo": "# DECISION MEMO — SMB CAMPAIGN: CONTINUE OR PIVOT
**From:** Coordinator (synthesis of Lanes 1–5) · **To:** Owner · **Date:** Fri 2026-07-17

Yes — this is still worth it, and that is a statement about artifacts on disk, not optimism: 2-1's root cause (observation aliasing) is found, the fix is shipped, and it is proven — `robust_2_1_runway_v2.pt` greedy-plays at 1.0 the exact runway→flag compound that ate 16 machine-hours. Every hard asset for the next three levels already exists: a proven 2-1 back half, welded 2-2 and 2-3, and banked entry states for 2-4 and 3-1 — the chain is exactly one seam away from moving from \"dies in 2-1\" to \"dies in 2-4.\" We will prove that within 48 hours or I will call the pivot myself: go/no-go is **Sunday 2026-07-19, 18:00**, criterion in writing below.

---

## 1. THE CALL — Continue, under Plan SEAM-FORWARD

**The plan, named and ranked:**
1. **Ship Lane 1(A) — gx-keyed intra-level routing for 2-1** (`2-1@gx<1421` new front-half weld from the sticky trainer's iter_01940 net + `2-1@gx>=1421` the proven v2 runway net), gated by the arrival-phase verification, with two banked fallback boundaries (gx1811 / gx1971) pre-agreed. The running v2 full-level trainer stays alive as the background **seam-remover** whose eventual weld deletes the gx key — the standing \"seam count trends down\" commitment. Option C (cross-encoder kickstart) is rejected permanently: dominated, its payoff already banked.
2. **Run Lane 2's road:** 2-4 and 3-1 launch **tonight** in parallel worktree lanes (both entries banked, zero file overlap). v2 encoder (`smb_tiles_pos`) is the default for all new nets; **one** tile-v3 extractor migration (hammer misc-slots + platforms-as-surface + `$0745` LoopCommand) built during the 3-2 lull so it lands before 3-3 needs it. 3-4 via the 1-4 playbook. Estimated 27–46 machine-hours from 2-4 through 3-4.
3. **Adopt Lane 3's (e1) now** — search-as-teacher (oracle-labeled recovery demos through the shipped `demo_anchor_loss`) as the standing wall-breaker. It converts every honest-metric death into a labeled hard example at the causal frame, and it is the machine that moves the sticky number.

**Integrity compliance (owner's constraints):** nothing in this plan touches the environment — no state injection, no teleports, single life from power-on, every frame emulated (`frame_skip` is held-button action granularity, not skipped emulation). The searcher is never deployed as the agent; it labels training demos only. Every seam is disclosed in every published number.

**What \"soon\" means — GO/NO-GO, Sunday 2026-07-19, 18:00.** Hold us to all three:
- **(i)** `make world1-gate` green: the frozen World 1 chain replays seq_clear 1.0, warp_rate 0.0, on both gate seeds.
- **(ii)** One cold, single-session, single-life, warpless deterministic run of the live composite clears **1-1 through 2-3 consecutively** (first death no earlier than 2-4), on both gate seeds, with per-episode action receipts (`--record-actions` npy + fm2) and the 2-1 seam disclosed (`gx_switches` + per-seam gx in `summarize`).
- **(iii)** The sticky-0.25 + start-jitter pair published alongside — report-only, no threshold asserted (it is 0.0 today; asserting on it would freeze progress, per Lane 4).

**Miss (i) or (ii) ⇒ NO-GO:** Monday 07-20 runs the 1-day Tetris activation spike and Tetris becomes the primary lane; SMB demotes to unattended background trainers with zero new engineering. No renegotiation — the trigger is pre-made so the decision cannot be mood-made.

**Pace commitment (the \"and beyond\"):** by **Friday 2026-07-24 EOD**, the cold chain plays **1-1 → 3-4** (Worlds 1–3 complete) under the same protocol. Miss ⇒ same consequence. \"Soon\" therefore means: visible chain-depth progress in 2 days, Worlds 1–3 in 7.

---

## 2. THE 48-HOUR PLAN (Fri 18:00 → Sun 18:00)

**Tonight — Friday 07-17 (machine time starts accruing immediately):**
- 18:00–19:00 — Write the two v2 profile YAMLs (2-4, 3-1: clones with `encoder: smb_tiles_pos`). Launch **2-4** (backward rung ladder from `handoff_2-4_pretrain.state`, est. 4–8 machine-hours) and **3-1** (organic grind from `handoff_3-1_pretrain.state`, est. 6–10) in separate worktrees. Leave `mario_2_1_v2` and `mario_2_1_sticky` untouched. Exit: two runs writing checkpoints.
- Overnight — three trainers accrue.

**Day 1 — Saturday 07-18 (hour-granular):**

| Hours | Work | Exit criterion |
|---|---|---|
| 09:00–10:15 | **Integrity first** (Lane 4 a+b): recorder hook + `--record-actions` in `run_episode`/`eval_composite` incl. its 4 tests; `summarize()` gains sticky_prob/start_jitter/seed/git_commit; populate `rom_md5` | Every number produced after 10:15 is receipted and self-describing |
| 10:15–13:00 | **Lane 1 code** (steps 1–4): `ReachGxTracker`; `--success-gx` plumbing; gx routing + forward-latch + `gx_switches` in `composite_policy.py`; `tests/test_composite_gx_routing.py` (parse, precedence, latch monotonicity, single-commit, capture-on-key-change); `configs/smb_2_1_v2.yaml`; manifest gx pair; `_scoring_key` strip fix | All new + existing composite tests green; gx=None path byte-identical |
| 13:00–13:30 | **Launch front-half weld:** `robustify_level.py` from `mario_2_1_sticky/vanilla_ppo_iter_01940.pt`, `--success-gx 1471` (overshoot margin), `--sticky-prob 0.25` → `robust_2_1_front.pt` | Weld running (easy terrain; source crosses gx1421 routinely) |
| 13:30–15:30 | While weld runs — **Lane 4(c):** `configs/composite_world1_gate.yaml` frozen over `runs/world1_oneshot_20260716/` (closed under reference), `chain_gate.py`, make targets `world1-gate` / `chain-gate` / `chain-honest` / `receipts-check` + `chain-check` umbrella. Run `world1-gate` | Golden 1.0 pinned BEFORE any manifest promotion |
| 15:30–16:30 | **Verification gates (a)+(b):** deterministic `--capture-handoffs` banks `handoff_2-1_agx_ge1421.state`; single-life greedy-verify `runway_v2` **from the banked crossing state** — the arrival-phase check, the one real risk. Fallback ladder pre-agreed: `@gx>=1811` → `@gx>=1971` → re-weld v2 from banked state | Handover proven or fallback boundary selected |
| 16:30–17:30 | Promote pair into live manifest; **gate (c):** full-chain deterministic `--stop-after-worlds 2` with `--record-actions` (expect 2-1→2-2→2-3, die in 2-4); `make chain-gate`; raise highwater deliberately | The go/no-go (ii) artifact exists, receipted |
| 17:30–18:00 | Kick off the honest number: `eval_composite --sticky-prob 0.25 --start-jitter 16 --episodes 50 --record-actions` (background) | Pair-in-progress for (iii) |
| 18:00–19:00 | Check 2-4/3-1 progress; queue 2-4 entropy consolidation (0.01→0.002) if max_x is at the axe; bank the **chain-consistent** 3-1 entry re-capture from 2-4's exit (bank both kinds; weld to chain-arrived) | Overnight lanes re-aimed |
| Evening (opt.) | Start (e1) scaffolding: save/load_worker_state harness + backward binary-search skeleton (pure Python, zero trainer contention) | — |

**Day 2 — Sunday 07-19 (blocks):**
- 09:00–13:00 — **Build (e1) to first labels:** reachability blame-assignment (K≈16 × 64 frames, sticky-expectimax), bank winning suffixes into the DemoBank; fold (c) as reset-to-last-winnable into the warm-start partition. First customer: `mario_2_1_v2`'s ~1900px deaths — anchor-feed the live run same day.
- 13:00–16:00 — 2-4 weld work as its run matures; **Lane 4(d):** archive closure of `runs/world1_oneshot_20260716/` (profiles + states + one receipt set); per-level archives for `robust_2_1_front.pt` and the routing pair.
- 16:00–18:00 — Assemble go/no-go evidence; **18:00: call it** against the written criterion.
- Not in the critical path: **(d′) time-shift phase augmentation** ships Mon–Tue 07-20/21 (1 day, on the harvest tool) — adopted this week, after the checkpoint.

---

## 3. WHAT WE STOP DOING (sunk-cost traps, ranked by money currently burning)

1. **Waiting on full-level 2-1 convergence.** The v1 consolidation tail ate 16 machine-hours; never serialize the chain behind that class of run again. The v2 run is demoted to background seam-remover — useful if it welds, blocking nothing if it doesn't.
2. **Cross-encoder kickstart (Option C).** Dead. Its entire payoff is already banked as `runway_v2.pt`; it is the slowest path to a weld by construction.
3. **Re-encoding shipped v1 levels / piecemeal extractor migrations.** Mixed encoders per key are already supported; re-encode only on a failed weld gate. One tile-v3 migration, built in the 3-2 window — not three.
4. **Any 3-3 attempt on v1/v2.** The observation labels the landing surfaces lethal; hours spent there are wasted by construction. v3 first, then 3-3.
5. **Quoting deterministic greedy or training telemetry as clear rates.** The harness bug already quarantined every pre-single-life number once, and \"7/24\" turned out to be telemetry. The published claim is always the sticky pair with a named harness.
6. **RAM-edit data generation, and anything through `game_genie.rs` — ever.** Time-shift (d′) gets phase diversity with real inputs through real emulation; ROM-patch-class mechanisms contaminate provenance even for data gen.
7. **Deploy-time lookahead as the shipped agent.** Precedent (Baumgarten 2009, playfun, ALE's UCT category) makes it an already-conquered class, and it would render our fm2 receipts TAS-indistinguishable. The searcher is a teacher, period.
8. **Naive pre-death resets and IQL-lite.** Resets only to the last *winnable* frame, inside (e1); IQL parked until failure banking exists.

---

## 4. INTEGRITY GUARDRAILS SHIPPING THIS WEEK (Lane 4, in priority order)

1. **Recorder hook + `--record-actions` + sidecar meta** (Sat 09:00, ~1h): record the post-sticky *stepped* mask at the single choke point — even sticky episodes become exactly replayable. Receipts are Tier-A (self-replay, byte-compare) now; \"FCEUX-verified\" is claimed only after the boot-prefix RAM-equivalence gate lands (open Tier-B gap, flagged, not this week's critical path).
2. **Self-describing evals** (Sat, 15 min): `summarize()` records sticky_prob/start_jitter/seed/git_commit; `rom_md5` populated. A logged row must prove its own protocol.
3. **Pinned local regression gates** (Sat afternoon): frozen `composite_world1_gate.yaml` + `chain_gate.py` + `make world1-gate / chain-gate / chain-honest / receipts-check / chain-check`. Run discipline: before/after edits to `composite_policy.py`, `smb_sequential.py`, stackers, `rewards.rs`, or any manifest; after every `make build` (with the maturin dylib copy first, or the gate tests the old emulator); before any push touching the chain. Two deterministic seeds are a real gate, not a sample.
4. **Claims checklist enforced on every published number:** cold single session, single life; warpless by predicate; both numbers, sticky first; hierarchical disclosure including `gx_switches`, per-seam gx, and the seam-count-trends-down commitment; named harness + full provenance; receipts attached.
5. **Archive hygiene** (Sun, then standing): close the World 1 archive; every new weld gets the 9-item per-level archive (ckpt sha256, frozen profile, entry state + chain-captured-vs-training-native tag, weld verification row, receipts, training inputs incl. collection noise model, run_manifest, environment pins, `levels_index.json`); one canonical handoff naming.

---

## 5. TETRIS — the honest verdict: LATER, ON THE PRE-MADE TRIGGER

**Not now, and not a parallel training track this week.** It would compete for the same M4 the seam run and two new lanes need, and it advances breadth, not the flagship — Tetris uses almost none of the moat (piece RNG devalues save-state welding, demo banking, Go-Explore, backward robustification; we'd be down to vanilla PPO + RND, the toolkit any Atari rig has). Pivoting now would swap a root-caused problem for an uncharted-hard one: frame-level model-free NES Tetris has essentially no published successes, which is exactly why \"1 day to activate, 1–2 weeks to showable, real plateau risk\" is the honest estimate.

**Its correct role:** the pre-committed NO-GO contingency. The activation spike is written and cheap — curated level-9 start state (~0.5–1h), lockstep-verify the `0xEF` sentinel + `$0050` lines byte (~1h), add `tetris_board` to the `_is_tile_mode` tuple at `trainer.py:613` + tile-path smoke (2–4h) — and those are permanent assets worth banking in the first machine-idle lull **after** the 07-24 milestone regardless of outcome. If it ever activates, it is framed as breadth (\"16 games, one engine, semantic RAM observations\"; B-Type clears and level-up chains), never as a replacement flagship — not score max-out, not a level-29 claim StackRabbit already owns with search.

**Bottom line:** the position is the strongest it has been all week and the next three levels are one verified seam away. We continue — and you hold us to Sunday 18:00 and Friday 07-24, in writing, with the pivot consequence pre-signed.",
