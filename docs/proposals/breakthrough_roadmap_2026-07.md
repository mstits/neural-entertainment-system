# Breakthrough roadmap — 2026-07 research review

Synthesized from a six-lane literature review (model-based RL, exploration,
generalization, imitation/offline RL, sequence models, showcase benchmarks)
mapped against a full asset inventory of this repository.

inputs digested and grounded against the repo (go_explore.py archive internals, sticky-action support in trainer.py, robustify_level.py pipeline, eval script gaps, config knobs). Synthesis follows.

# NEXT-BREAKTHROUGH ROADMAP — NES-RL, post-World-1

**Strategic thesis (one paragraph).** Every lane converges on the same conclusion: the moat is the 30k-fps deterministic emulator + save-state machinery + demo banks, and the field's headline race (world models, sample efficiency) is *irrelevant to this regime* — real frames are cheaper than imagined ones here [MODEL-BASED]. In a deterministic environment, discovery is a search problem, not an RL problem [EXPLORATION]; the RL problem is turning found trajectories into policies that survive noise [GENERALIZATION][IMITATION]. The breakthrough is therefore not a new algorithm. It is industrializing the already-80%-built Go-Explore → robustify pipeline into a per-level *factory*, hardening its output against stochasticity, and pointing it at a claim nobody has published: a full-game SMB clear, single life, from power-on [SHOWCASE]. Everything below serves that.

---

## (A) TOP 5 BREAKTHROUGH BETS

Scored as (capability lift × asset fit) / effort, 1–5 each, higher is better.

### BET 1 — The Level Factory: full Go-Explore Phase 1 + Nature-spec backward robustification, systematized per level
**Score: lift 5 × fit 5 / effort 2 → 12.5. This is the engine for the flagship. Start today.**
- **What**: Convert ad-hoc harvesting into an automated pipeline that eats a level and emits a robust specialist: RAM-feature cell archive → frontier-weighted exploration → multi-demo harvest → backward-algorithm robustification with the exact Nature hyperparameters (160-frame start *window*, 0.1 move threshold, 50-frame allowed lag, 30% episodes from true start — the quartet that separates it from single-anchor welds) [EXPLORATION].
- **Papers**: Ecoffet et al., Nature 2021 (arXiv:2004.12919); Salimans & Chen (1812.03381); Backplay (1807.06919); RFCL (2405.03379) for the reverse-then-forward consolidation phase.
- **Builds on**: `src/training/go_explore.py` — already has domination/elite replacement (`record`, lines 102–137) and W=1/√(times_chosen+1) selection (`_selection_weight`, 141–151); `scripts/robustify_level.py` — already does GX→10-clear harvest→BC→robustify (`--clears 10 --explore-eps 0.15`); `checkpoints/harvested_seeds/` (3 `seed_2_1_*` states ready); `checkpoints/handoffs/handoff_2-*.state`; `configs/smb_2_2_water.yaml` already staged. The gaps are exactly: neighbor-aware W_location weights, keep-exploring-after-first-clear demo shortening, and the backward window/threshold/lag/retention quartet.
- **First experiment**: Point the factory at World 2 unmodified, then add the missing pieces and measure the delta:
  `python scripts/robustify_level.py --profile configs/mario_tiles.yaml --checkpoint runs/world1_oneshot_20260716/vanilla_ppo_iter_01580.pt --start-state checkpoints/handoffs/handoff_2-1_pretrain.state --out runs/factory_2_1 --clears 10 --explore-eps 0.15`
  Then 2-2 with `configs/smb_2_2_water.yaml` — water physics is the first real test that the factory generalizes past run-right.
- **Success metric**: levels cleared per week with zero per-level hand-tuning. Target ≥2/wk immediately, ≥4/wk after the quartet lands (28 levels remain).
- **Time-to-first-signal**: 2-1 greedy clear in 2–3 days; 2-2 water inside a week = factory validated.

### BET 2 — Honest-mode hardening: sticky-action robustification + DAgger seam repair
**Score: lift 4 × fit 5 / effort 1 → 20 nominal — the best ratio on the board, but smaller absolute lift than Bet 1. Run it in parallel; it is days, not weeks.**
- **What**: Make sticky-p=0.25 + start-jitter the *reported* eval everywhere, retrain welds with stochasticity ON, and DAgger-relabel seam divergences. This converts the composite from a lookup table into policies with recovery behavior — the \"understands vs memorizes\" keystone [GENERALIZATION], and the textbook cure for weld phase-curse (compounding covariate shift) [IMITATION].
- **Papers**: Machado 2018 (sticky actions, JAIR); Go-Explore Phase 2 (281k Montezuma *under sticky eval*); DART (Laskey 2017); DAgger (Ross 2011) + MEGA-DAgger (2303.00638) for conflicting multi-demo labels; SIL (1806.05635) / SILfD (2203.10905) as the continuous self-cloning replacement for offline BC passes.
- **Builds on**: Sticky actions are *already implemented with PPO-correct log-prob handling* — `src/training/trainer.py:397–422` (`sticky_action_prob`, line 422) and 1241–1253; `configs/mario_tiles.yaml:362` already trains at 0.5. The gap is embarrassing and cheap: **eval scripts have zero sticky/jitter support** (grep-confirmed in `eval_game.py`/`eval_composite.py`), and every weld/backward config runs sticky 0.0 (`smb_1_4_backward.yaml:54`, `smb_1_4_go_explore.yaml:69`, `smb_2_2_water.yaml:83`, `smb_oneshot_tiles.yaml:203`). DAgger expert queries are free: deterministic save-state teleport + banked replay as labeler.
- **First experiment**: Add `--sticky-prob`/`--start-jitter` to `scripts/eval_composite.py`; run the W1 composite (`runs/world1_oneshot_20260716/composite_world1.yaml`) at sticky 0.25 × 100 episodes. Publish that number whatever it is — it is the program's honesty baseline. Then flip sticky to 0.25 in the robustify configs and re-robustify 1-2.
- **Success metric**: sticky-0.25 single-life W1 chain clear rate: expect ~0 baseline → target >70% after retrain.
- **Time-to-first-signal**: baseline number in half a day; recovered 1-2 sticky clear in 2–3 days.

### BET 3 — PLR over the save-state farm + joint one-net training
**Score: lift 4 × fit 4 / effort 2 → 8.**
- **What**: \"Level = save state.\" Score every start state (stage anchors, GX cells, handoffs) by positive-value-loss EMA, sample starts ∝ score + staleness, train ONE tile net jointly across all of W1+W2. Add Florensa's [0.1, 0.9] learnability band (retire too-easy starts) and ACCEL-style RAM-perturbation edits (Mario x/y, enemy slots, timer byte) to spawn frontier variants. Replaces the hand-tuned rolling-mean advance gate with an automatic curriculum [GENERALIZATION].
- **Papers**: PLR (Jiang 2021); Robust PLR/DCD (facebookresearch/dcd); ACCEL (2203.01302); Florensa (1707.05300); Gotta Learn Fast (1804.03720) — joint-train + fine-tune ≈ 2x transfer on retro platformers.
- **Builds on**: `GoExploreArchive.select_return_states()` (go_explore.py:166–175) is *already the whole-pool warm-start hook* — PLR is \"swap the frontier weight for a value-loss-scored weight\" plus per-state bookkeeping in `trainer.py`; the save-state farm spans `checkpoints/{handoffs,harvested_seeds,super_mario_bros/smb_curriculum}`.
- **First experiment**: `src/training/plr_sampler.py` beside the archive; `configs/smb_plr_joint.yaml` (base mario_tiles, 60 envs); benchmark iters-to-W1-seq-clear against the 1680-iter ladder baseline (`checkpoints/super_mario_bros_one_shot_tiles/`).
- **Success metric**: (a) beat 1680 iters to W1 one-shot; (b) sticky-eval transfer to held-out start states beats the ladder-trained net.
- **Time-to-first-signal**: sampler coverage curves in 1–2 days; W1 head-to-head inside a week.

### BET 4 — Decoupled actor/learner (Sebulba/APPO pattern) — the throughput multiplier
**Score: lift 3–4 × fit 3 / effort 3 → ~4, gated by a one-hour measurement.**
- **What**: Overlap the PPO update with pool collection (train on batch N while collecting N+1 under the stale policy), V-trace/importance-corrected. Explicitly NOT the debunked per-step `async_pipeline` obs-lag knob — observations stay frame-exact; only *policy staleness* is introduced, which V-trace is designed to absorb [MODEL-BASED]. Opportunistic add-ons from the same lane: PPG aux value phase or BBF-style replay-ratio training over the deterministic banks; PQN as a cheap value-head experiment on grind levels.
- **Papers**: Podracer/Sebulba (2104.06272); Sample Factory APPO (2006.11751); cleanba; PufferLib huge-batch PPO (2406.12905); PQN (2407.04811); PPG (2009.04416); BBF (2305.19452).
- **Builds on**: Rust pool + pool-level pacing (3683fb2..e4b1210); MPS idles during collection in tile mode; `scripts/profile_training_buckets.py` and `scripts/bench_trainer_async.py` exist for before/after; `make selftest-learning` as the correctness guard.
- **First experiment**: `python scripts/profile_training_buckets.py --profile configs/mario_tiles.yaml` — measure the collect/update duty cycle FIRST. If update ≥25% of wall clock, implement double-buffered rollouts in the trainer loop with an IS-ratio clamp; A/B on 1-1 relearn.
- **Success metric**: ≥1.5x learner-consumed samples/s with unchanged clears-per-iter on the SMB selftest. If the duty cycle says update <15%, kill the bet without regret.
- **Time-to-first-signal**: duty-cycle profile in one hour (this gates the bet); full A/B ~1 week. Ranked #4 because it multiplies the others but is capped by the duty cycle and touches the one subsystem with a known correctness minefield.

### BET 5 — One-net endgame: kickstarted distillation of the composite
**Score: lift 4 × fit 4 / effort 3 → ~5.3, but sequenced after Bets 1–3 produce its inputs.**
- **What**: Distill all specialists into a single student trained under PPO with an annealed cross-entropy-to-teacher aux loss (multi-teacher kickstarting), PLR starts, sticky ON, DrAC-style augmentation consistency (tile-window crop/x-jitter). Removes the level-keyed switch; kickstarted students exceeded teachers by 42% on DMLab [IMITATION][GENERALIZATION]. Optional deployment artifact: a 1–3M-param DT/Decision Mamba over the demo bank — credible as \"one model plays the whole game,\" explicitly framed as distillation, not an optimizer [SEQUENCE].
- **Papers**: Kickstarting (1803.03835); Policy Distillation / Actor-Mimic; ITER (2006.05826); DrAC (2006.12862); DT (2106.01345) for the artifact only.
- **Builds on**: `src/training/composite_policy.py` (routing + hysteresis, lines 267–378) as the teacher; replay banks as perfect demo data; `src/training/behavior_cloning.py` build_dataset; `scripts/eval_composite.py` for teacher-vs-student tables.
- **First experiment**: `kickstart_ce_coef` aux loss in vanilla_ppo, teacher = routed composite, student = fresh tile MLP (2–3x wider than 14k params); train across W1 via PLR starts.
- **Success metric**: single student ≥ composite on W1 single-life *sticky* eval. This is the M1 second act: \"we replaced the committee with one tiny net.\"
- **Time-to-first-signal**: student clearing 1-1/1-2 via distill in 2–3 days.

---

## (B) QUICK WINS (<1 day each), in order

1. **Sticky+jitter eval flags** in `eval_composite.py`/`eval_game.py` + publish the W1 composite baseline number. The single highest-information day available [GENERALIZATION].
2. **Flip `sticky_action_prob` 0.0→0.25** in the weld/backward/GX configs (trainer support already shipped; mario_tiles already at 0.5) [GENERALIZATION][IMITATION].
3. **Neighbor-aware selection weight** in `go_explore.py:_selection_weight` (141–151): add W_location = (2−h)/10 for cells whose horizontal neighbors are missing from the archive. The paper credits this with a large Montezuma lift; it is a few lines [EXPLORATION].
4. **Keep exploring after first clear** — don't stop GX at first solve; trajectory length keeps dropping and shorter demos robustify far easier (Ecoffet Ext. Data Fig. 5). Loop-condition change in the harvest driver [EXPLORATION].
5. **Florensa [0.1, 0.9] band** on the curriculum gate — the advance gate is currently one-sided; add the lower bound to retire mastered starts [EXPLORATION][GENERALIZATION].
6. **Bank failure episodes** alongside clears (one flag in the banking path) — free now; feeds future IQL seam-quality probes and mixed-quality DT data [IMITATION][SEQUENCE].
7. **NovelD term on TileRND**: r = max(RND(s_t) − 0.5·RND(s_{t−1}), 0) × exact episodic first-visit gate (RAM hash makes the indicator exact, not approximate). One subtraction + a set; RND itself is already fixed per commit c057bbd [EXPLORATION].

---

## (C) WHAT TO STOP DOING

1. **Kill the DreamerV3 ambition.** `src/training/dreamer.py` + `src/models/world_model.py` + empty `checkpoints/dreamer/` + the `docs/world_model_rl.md` roadmap: shelve all of it. Every flagship world-model result optimizes sample scarcity this project does not have, at GPU costs MPS is worst-positioned to pay; BBF shows model-free wins that race anyway; MuZero's learned model is a workaround for lacking save-state access, which this project has perfectly [MODEL-BASED]. Wiring it in is negative-value work.
2. **Stop pursuing the recurrent tile-GRU as a weld/curriculum policy** (`mario_recurrent.yaml`, aborted `mario_1_4_recurrent` run). Recurrence welds strictly *worse* — save-state entry with no history is R2D2's stale-hidden-state problem in its worst case; SMB from tiles+stack is Markov [SEQUENCE]. Its one justified future is as the goal-conditioned returner in policy-based Go-Explore — and only with hidden state serialized into `.state` files plus 40–80-frame burn-in.
3. **Stop per-level from-scratch specialists as the default recipe.** It is the maximal-overfit configuration (Cobbe 2018); joint-train + fine-tune ≈ 2x transfer (Sonic contest) [GENERALIZATION]. Specialists remain a tool for boss levels; the default becomes Bet 3's joint PLR net.
4. **Stop reporting deterministic-greedy-only clear rates.** The program already got burned once (\"7/24 on 1-4\" was training telemetry). Deterministic eval structurally overstates trajectory-replay systems (Machado 2018). Sticky+jitter is the reported metric from now on [GENERALIZATION].
5. **Do not do event-driven PPU catch-up next.** A >9%-ceiling, high-risk emulator surgery is strictly dominated by Bet 4's potential 1.5–2.5x learner overlap — run the one-hour duty-cycle profile before touching the PPU. And do not resume the self-debunked micro-opt list (more ASM opcodes, torch.compile/fp16/MLX/ANE on collection, allocator flags) [MODEL-BASED, asset #6].
6. **Pause the pixel one-shot arm's duplicate ladder grind** (stuck at rung 9 of a ladder the tile arm already finished). Tile mode is the unfair advantage for every showcase on the board — even ROM hacks keep tile semantics [SHOWCASE]. Revisit pixels only for new-tileset hacks or cross-game goals. Also not starting: GAIL/AIRL, CQL/DT-as-primary, SPR/CURL in tile mode, Gato/MGDT-scale multi-game checkpoints — each lane explicitly killed one of these.

---

## (D) THE DEMONSTRATION

**\"First AI to beat Super Mario Bros. — warpless, single life, from power-on — with TAS-grade receipts.\"**

Why this and not the alternatives: no published cold single-session full-game clear exists; the de-facto SOTA citation (uvipen, 31/32) uses one net per level, never chains them, and fails 8-4 [SHOWCASE]. World 1 is already cleared cold, the factory (Bet 1) turns Worlds 2–8 into grind rather than research, and the unique assets map exactly: 30k-fps determinism makes verified resimulable runs trivial, the welding playbook *is* Go-Explore Phase 2, and FCEUX-playable input receipts are a credibility multiplier no RL demo has ever shipped. Timely hook: LLM labs now benchmark on SMB and fumble it. M2 (segment-time tables vs human WR splits) and M3 (ROM-hack gauntlet) inherit this run's credibility and reuse the same factory — do them after, not instead.

**Milestone ladder (each rung independently publishable):**
- **L0 (day 1)**: Honest baseline — W1 composite sticky-eval number + claims-table skeleton vs uvipen/SMBbot (QW1).
- **L1 (~wk 1)**: World 2 falls to the factory, including 2-2 water — first new mechanic, proves the factory generalizes (Bet 1).
- **L2 (~wk 2–3)**: Worlds 3–5; publish the levels/week burn-down. Easy levels should fall to the joint PLR net (Bet 3), hard ones get factory specialists.
- **L3 (~wk 3–4)**: **8-4 — the level PPO never beat.** The maze that kills pixel agents is disambiguated for free by RAM-cell archives. Standalone teaser release.
- **L4**: Chain integration — 31 seam welds, single-life power-on runs; DAgger seam repair (Bet 2) until deterministic chain success >50%, then >90% over 100 attempts; report the sticky-mode chain rate alongside as the honesty number.
- **L5 (ship)**: One unbroken ~25-min video; FCEUX-verifiable input file (`scripts/record_demo.py` / `scripts/convert_fm2.py` path); claims table (uvipen 31/32 per-level nets, SMBbot stitched bot, LLM benchmark failures); compute receipts. State the composite-of-specialists caveat explicitly and pre-announce the second act.
- **L6 (second act)**: Bet 5's single tiny net re-clears the game — \"the committee became one 20k-parameter net.\" Then M2 (retarget factory to time; \"AI finds a time-save\" is the Trackmania-proven viral branch) and M3 (viewer-submitted ROM-hack gauntlet) as the recurring formats.

Explicitly not in the demonstration: multi-game single checkpoint (big-lab-scale cost, weak completion story) and 24/7 ambient streaming (proven-niche; event/video formats win) [SHOWCASE].

**Bottom line**: Start Bets 1+2 in parallel this week (they share the robustify pipeline), run Bet 4's one-hour profile gate immediately, land Bet 3 as the W3+ training substrate, and hold Bet 5 for the second act. The breakthrough is the factory plus the claim no one else can make — and every piece of it is already 60–90% built in this repo.",
