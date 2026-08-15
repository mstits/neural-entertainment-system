"""Gate / kill / phase-assembly logic of the SMB 1-2 online campaign
controller (scripts/run_online_campaign.py).

Everything here runs on STUBBED metric streams — synthetic metrics.jsonl
rows and synthetic eval_game JSON — no emulator, no torch, no subprocess.
The controller module is imported for its pure pieces only; its heavy
imports (torch, the probe runner) are deferred inside functions, which
these tests also implicitly verify by importing the module at all.

Covered:
* deep_merge — phase overrides layer onto the base profile without
  mutating it;
* entropy_coef_at — the campaign's 0.01 -> 0.002 over 100M schedule;
* build_phase_profile — per-phase config assembly (sticky per phase,
  adversary armed ONLY in phase 4, same profile name across phases so
  every phase resumes the same checkpoint dir);
* KillMonitor — the three pre-registered kill criteria on stubbed rows;
* critic_cv + gate helpers — the phase-advance gates;
* probe_summary — median max-x / bottleneck-survival off a synthetic
  eval_game result;
* manifest mirror — every CONFIG threshold lands in the manifest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_online_campaign import (  # noqa: E402
    CONFIG, PHASES, KillMonitor, build_manifest, build_phase_profile,
    critic_cv, deep_merge, entropy_coef_at, probe_summary,
    read_backward_tau,
)

SPI = 92_160  # steps-per-iter used by the stub streams (1536 x 60)


def _row(gen: int, **kw) -> dict:
    row = {"generation": gen}
    row.update(kw)
    return row


def _base_profile() -> dict:
    return {
        "name": "Mario 1-2 online v2",
        "reinforce": {
            "sticky_action_prob": 0.0,
            "entropy_coef": 0.01,
            "rollout_steps": 1536,
            "num_envs": 60,
            "adversary": {"budget_penalty": 0.1, "epochs": 10},
            "backward_curriculum": {"states_dir": "x", "tau_init": -1},
        },
    }


# ---- deep_merge ------------------------------------------------------


def test_deep_merge_nests_and_does_not_mutate():
    base = _base_profile()
    out = deep_merge(base, {"reinforce": {"sticky_action_prob": 0.25}})
    assert out["reinforce"]["sticky_action_prob"] == 0.25
    assert out["reinforce"]["entropy_coef"] == 0.01
    assert base["reinforce"]["sticky_action_prob"] == 0.0


def test_deep_merge_adds_new_subkeys():
    out = deep_merge(_base_profile(),
                     {"reinforce": {"adversary": {"mode": "kernel_sticky"}}})
    adv = out["reinforce"]["adversary"]
    assert adv["mode"] == "kernel_sticky"
    assert adv["budget_penalty"] == 0.1  # base keys survive


# ---- entropy schedule ------------------------------------------------


def test_entropy_schedule_endpoints_and_midpoint():
    assert entropy_coef_at(0) == pytest.approx(CONFIG["entropy_start"])
    assert entropy_coef_at(CONFIG["entropy_decay_steps"]) == pytest.approx(
        CONFIG["entropy_end"])
    mid = entropy_coef_at(CONFIG["entropy_decay_steps"] / 2)
    assert mid == pytest.approx(
        (CONFIG["entropy_start"] + CONFIG["entropy_end"]) / 2)
    # Clamped past the end, never below entropy_end.
    assert entropy_coef_at(10 * CONFIG["entropy_decay_steps"]) == pytest.approx(
        CONFIG["entropy_end"])


# ---- phase profile assembly ------------------------------------------


def test_phases_are_the_registered_six():
    assert [p["name"] for p in PHASES] == [
        "critic_warmup", "local_clear", "sticky_local", "reverse_walk",
        "hardening", "consolidation",
    ]


def test_phase_profiles_keep_the_name_so_checkpoints_chain():
    base = _base_profile()
    for phase in PHASES:
        prof = build_phase_profile(base, phase, cum_env_steps=0)
        assert prof["name"] == base["name"]


def test_adversary_mode_armed_only_in_hardening():
    base = _base_profile()
    for phase in PHASES:
        prof = build_phase_profile(base, phase, cum_env_steps=0)
        mode = prof["reinforce"]["adversary"].get("mode")
        if phase["name"] == "hardening":
            assert mode == "kernel_sticky"
        else:
            assert mode is None


def test_sticky_zero_in_noise_free_phases_kernel_phase_included():
    base = _base_profile()
    sticky = {
        p["name"]: build_phase_profile(base, p, 0)["reinforce"][
            "sticky_action_prob"]
        for p in PHASES
    }
    assert sticky["critic_warmup"] == 0.0
    assert sticky["local_clear"] == 0.0
    assert sticky["sticky_local"] == 0.25
    assert sticky["reverse_walk"] == 0.25
    # Phase 4's perturbation is the kernel adversary, not house sticky.
    assert sticky["hardening"] == 0.0
    assert sticky["consolidation"] == 0.25


def test_loss_tether_coef_schedule_across_phases():
    """Attempt-3 loss-level KL tether: 1.0 at unfreeze, 0.3 through the
    sticky/reverse phases, 0.0 (anchor fully removed) from hardening on."""
    base = _base_profile()
    coefs = {
        p["name"]: build_phase_profile(base, p, 0)["reinforce"].get(
            "kl_anchor_loss_coef")
        for p in PHASES
    }
    # Phase 0 leaves the key absent (trainer default 0.0; the actor is
    # frozen for the whole critic warmup anyway).
    assert coefs["critic_warmup"] is None
    assert coefs["local_clear"] == 1.0
    assert coefs["sticky_local"] == 0.3
    assert coefs["reverse_walk"] == 0.3
    assert coefs["hardening"] == 0.0
    assert coefs["consolidation"] == 0.0


def test_kill_table_is_preregistered_and_unchanged():
    """The kill criteria are pre-registered per attempt. Attempt-4
    re-registration (rationale in the CONFIG comment; attempt-3 receipt
    runs/online_1_2_attempt3/): KL threshold 0.15 -> 0.60 — attempt 3's
    loss tether held a STABLE plateau at ~0.31 (vs attempt 2's monotone
    unspool to ~1.4) and 0.60 separates the two measured modes — plus a
    new immediate competence kill on probe median collapse, which guards
    the thing the KL kill was a proxy for. These pins exist so any later
    change must be equally deliberate."""
    assert CONFIG["kill_kl_threshold"] == 0.60
    assert CONFIG["kill_kl_sustain_steps"] == 2_000_000
    assert CONFIG["kill_probe_median_floor"] == 150.0
    assert CONFIG["kill_vloss_spike_ratio"] == 5.0
    assert CONFIG["kill_vloss_recovery_steps"] == 1_000_000
    assert CONFIG["kill_phase1_no_sil_clear_steps"] == 20_000_000


def test_base_profile_gammas_rescaled_and_equal():
    """Return scale (attempt 3): reinforce.gamma dropped 0.999 -> 0.99, and
    the wavefront shaping gamma must mirror it exactly — the monotone
    rule's PBRS invariance argument requires shaping gamma == RL gamma."""
    import yaml
    prof = yaml.safe_load((ROOT / CONFIG["base_profile"]).read_text())
    rl = prof["reinforce"]
    assert rl["gamma"] == 0.99
    assert rl["wavefront_reward"]["gamma"] == rl["gamma"]


def test_entropy_coef_follows_cumulative_steps():
    base = _base_profile()
    p0 = build_phase_profile(base, PHASES[0], cum_env_steps=0)
    assert p0["reinforce"]["entropy_coef"] == pytest.approx(
        CONFIG["entropy_start"])
    late = build_phase_profile(
        base, PHASES[5], cum_env_steps=CONFIG["entropy_decay_steps"])
    assert late["reinforce"]["entropy_coef"] == pytest.approx(
        CONFIG["entropy_end"])


# ---- kill criteria ----------------------------------------------------


def _monitor() -> KillMonitor:
    return KillMonitor(CONFIG, steps_per_iter=SPI)


def _steps_to_iters(steps: float) -> int:
    return int(steps // SPI) + 2


def test_kl_kill_fires_after_sustained_2m_in_phase1():
    m = _monitor()
    m.start_phase(1, env_steps=0, sil_clears_total=0)
    n = _steps_to_iters(CONFIG["kill_kl_sustain_steps"])
    reason = None
    for g in range(n):
        reason = m.observe(_row(g, kl_anchor_div=0.7, ppo_value_loss=1.0,
                                sil_clears_total=1))
        if reason:
            break
    assert reason is not None and "kl" in reason.lower()


def test_kl_run_resets_on_a_dip_below_threshold():
    m = _monitor()
    m.start_phase(1, env_steps=0, sil_clears_total=0)
    n = _steps_to_iters(CONFIG["kill_kl_sustain_steps"])
    for g in range(n):
        # Dip below the threshold every 5 iters — never 2M sustained.
        kl = 0.05 if g % 5 == 0 else 0.2
        assert m.observe(_row(g, kl_anchor_div=kl, ppo_value_loss=1.0,
                              sil_clears_total=1)) is None


def test_kl_kill_is_phase1_only():
    m = _monitor()
    m.start_phase(3, env_steps=0, sil_clears_total=5)
    n = _steps_to_iters(2 * CONFIG["kill_kl_sustain_steps"])
    for g in range(n):
        assert m.observe(_row(g, kl_anchor_div=0.5, ppo_value_loss=1.0,
                              sil_clears_total=5)) is None


def test_value_loss_spike_unrecovered_kills():
    m = _monitor()
    m.start_phase(2, env_steps=0, sil_clears_total=3)
    g = 0
    # Healthy baseline around 1.0.
    for _ in range(CONFIG["kill_vloss_baseline_rows"] + 5):
        assert m.observe(_row(g, ppo_value_loss=1.0, sil_clears_total=3)) is None
        g += 1
    # >500% spike that never recovers.
    reason = None
    for _ in range(_steps_to_iters(CONFIG["kill_vloss_recovery_steps"])):
        reason = m.observe(_row(g, ppo_value_loss=8.0, sil_clears_total=3))
        g += 1
        if reason:
            break
    assert reason is not None and "value" in reason.lower()


def test_value_loss_spike_that_recovers_does_not_kill():
    m = _monitor()
    m.start_phase(2, env_steps=0, sil_clears_total=3)
    g = 0
    for _ in range(CONFIG["kill_vloss_baseline_rows"] + 5):
        assert m.observe(_row(g, ppo_value_loss=1.0, sil_clears_total=3)) is None
        g += 1
    # Spike for a few iters (well under 1M steps), then recover.
    for _ in range(3):
        assert m.observe(_row(g, ppo_value_loss=9.0, sil_clears_total=3)) is None
        g += 1
    for _ in range(_steps_to_iters(2 * CONFIG["kill_vloss_recovery_steps"])):
        assert m.observe(_row(g, ppo_value_loss=1.1, sil_clears_total=3)) is None
        g += 1


def test_sil_drought_kills_at_20m_phase1_steps():
    m = _monitor()
    m.start_phase(1, env_steps=0, sil_clears_total=7)
    n = _steps_to_iters(CONFIG["kill_phase1_no_sil_clear_steps"])
    reason = None
    for g in range(n):
        reason = m.observe(_row(g, kl_anchor_div=0.01, ppo_value_loss=1.0,
                                sil_clears_total=7))
        if reason:
            break
    assert reason is not None and "sil" in reason.lower()


def test_a_single_sil_clear_disarms_the_drought_kill():
    m = _monitor()
    m.start_phase(1, env_steps=0, sil_clears_total=7)
    n = _steps_to_iters(2 * CONFIG["kill_phase1_no_sil_clear_steps"])
    for g in range(n):
        total = 7 if g < 5 else 8  # one clear early in the phase
        assert m.observe(_row(g, kl_anchor_div=0.01, ppo_value_loss=1.0,
                              sil_clears_total=total)) is None


# ---- advance gates ----------------------------------------------------


def test_critic_cv_flat_series_passes_noisy_fails():
    flat = [1.0 + 0.01 * (i % 3) for i in range(50)]
    noisy = [1.0 + (1.0 if i % 2 else -0.5) for i in range(50)]
    assert critic_cv(flat) < CONFIG["gate_critic_cv"]
    assert critic_cv(noisy) > CONFIG["gate_critic_cv"]


def test_critic_cv_empty_is_infinite():
    assert critic_cv([]) == float("inf")


# ---- probe summary ----------------------------------------------------


def test_probe_summary_median_and_bottleneck_survival():
    eval_json = {
        "status": "ok",
        "n_episodes": 6,
        "max_gx_per_episode": [100, 2600, 2700, 900, 2650, 3266],
        "seq_clear_rate": 0.5,
        "clear_rate": 0.0,
        "level_clear": True,
    }
    s = probe_summary(eval_json, survival_x=CONFIG["bottleneck_survival_x"])
    assert s["median_max_x"] == pytest.approx((2600 + 2650) / 2)
    assert s["bottleneck_survival"] == pytest.approx(4 / 6)
    assert s["clear_rate"] == pytest.approx(0.5)
    assert s["n_episodes"] == 6


def test_probe_summary_prefers_seq_rate_but_degrades():
    s = probe_summary({"n_episodes": 2, "max_gx_per_episode": [1, 2],
                       "clear_rate": 0.5}, survival_x=2600)
    assert s["clear_rate"] == pytest.approx(0.5)
    assert s["bottleneck_survival"] == 0.0


# ---- phase-3 gate: restart distribution reaches x=0 --------------------


def test_read_backward_tau_from_checkpoint_payload():
    # The trainer persists TauScheduler.state_dict() under this key
    # (checkpoint_manager); tau == 0 is the "reached the entrance" gate.
    assert read_backward_tau({"backward_curriculum": {"tau": 3}}) == 3
    assert read_backward_tau({"backward_curriculum": {"tau": 0}}) == 0
    assert read_backward_tau({"net_state_dict": {}}) is None
    assert read_backward_tau(None) is None


# ---- manifest mirror ---------------------------------------------------


def test_manifest_mirrors_every_config_threshold_and_phase():
    man = build_manifest(_base_profile())
    assert man["config"] == CONFIG
    assert [p["name"] for p in man["phases"]] == [p["name"] for p in PHASES]
    for key in ("kill_kl_threshold", "kill_kl_sustain_steps",
                "kill_vloss_spike_ratio", "kill_vloss_recovery_steps",
                "kill_phase1_no_sil_clear_steps", "gate_critic_cv",
                "gate_det_clears", "gate_sticky_clear"):
        assert key in man["config"]
