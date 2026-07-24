# Driving prompt for the deep-research AI (v3, 2026-07-23)

Paste the following as the task prompt, attaching the four documents:
`FULL_DOSSIER_2026-07-22.md`, `LEARNED_STICKY_DISTILLATION_BRIEF_2026-07-22.md`,
`LEARNED_STICKY_DISTILLATION_BRIEF_v2_2026-07-22.md`, `DOSSIER_V3_2026-07-23.md`.

---

You are consulting on a reinforcement-learning system whose goal is a policy
that clears Super Mario Bros World 1-2 under the strict Machado evaluation
protocol: cold greedy (argmax) play, 25% sticky actions, 0–16 frame no-op
start jitter, from the level entrance, no state restoration at eval time.
Training-time state restoration and search-derived reward shaping ARE
permitted; game internals (disassembly, level maps, hand-authored inputs)
are NOT, in training or anywhere else.

Read the four attached documents in order (v1 brief → v2 brief → full
dossier → **Dossier v3, which supersedes where they conflict**). Dossier v3
reports: five verified reward-shaping exploits found and closed (transition
aliasing, x-only projection, Φ=0 sanctuary, etc.), a complete experimental
elimination of the imitation-learning solution space (capacity, features,
action history, recurrence, label conflicts — each with numbers), the
perfect-clone result (train acc 1.000 → sticky success 0.00), and the
gate-mirage result (3-episode sticky acceptance gates passing welds whose
true success is <0.25%). The system now has provably non-farmable shaping,
a validated measurement harness, and a PPO baseline that reaches ~36% of the
level and stalls at a specific gauntlet (gx 1600–2200) where per-attempt
sticky success for every constructible near-open-loop policy is below 1/400.

Your deliverable, in one response:

1. **Answer §5's five constrained questions directly**, with citations to
   published work where they exist. Do not repeat advice already refuted in
   the documents (the refuted list spans: dense-shaping-alone, recurrence,
   per-state DQfD anchoring, DART/DARS alone, backward ladders with
   small-sample gates, feature enrichment).
2. **Give ONE primary recipe** — the single algorithmic addition to
   PPO + Go-Explore restarts most likely to cross a <1/400-success gauntlet
   under the honest protocol on this exact stack (712-d tile obs, feedforward
   128/32, 60 envs, ~4.4k steps/s, CPU, 1400-iter budgets). Specify every
   hyperparameter, the restart-distribution schedule, the noise-curriculum
   schedule if you prescribe one, acceptance statistics, and 3 measurable
   signposts with iteration numbers at which we abandon the recipe if unmet.
3. **State the falsification condition** under which the correct conclusion
   becomes "SMB 1-2 under Machado sticky-0.25 exceeds what this policy class
   can express," and what the strongest field-defensible fallback claim is
   (e.g., sticky-0.10 clears, or per-segment robustness curves) — with
   precedent for how published work reports partial robustness.
4. **Cite any published agent that clears SMB 1-2 specifically under
   sticky-0.25 from the level start.** If none exists, say so plainly — that
   absence is itself decision-relevant information for this project.

Hard constraints: no game internals; all shaping must derive from the
system's own search output; the eval protocol is fixed and non-negotiable
for the primary claim; solutions must run on one M4 MacBook Pro (CPU
inference, ~4.4k env-steps/s total).
