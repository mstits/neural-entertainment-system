"""experiment_preflight — the positive controls, pinned."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.experiment_preflight import (  # noqa: E402
    assess_learning, assess_mechanisms, assess_sentinels, param_deltas,
    steps_per_iter_from_profile)


def _sd(actor=0.0, critic=0.0):
    return {"trunk.fc1.weight": torch.zeros(2, 2) + actor,
            "pair_actor.weight": torch.zeros(2, 2) + actor,
            "critic.weight": torch.zeros(2) + critic}


def test_frozen_actor_signature_fails():
    ok, msg = assess_learning(param_deltas(_sd(), _sd(actor=0.0, critic=1.0)))
    assert not ok and "FROZEN-ACTOR" in msg


def test_live_actor_passes():
    ok, msg = assess_learning(param_deltas(_sd(), _sd(actor=0.01, critic=1.0)))
    assert ok


def test_nothing_moving_fails():
    ok, msg = assess_learning(param_deltas(_sd(), _sd()))
    assert not ok and "NOTHING moved" in msg


def test_multihead_critic_heads_prefix_classified_as_critic():
    # MultiHeadTilePolicy names its value heads "critic_heads.N.weight",
    # not "critic.weight" — CRITIC_PREFIXES must recognize both spellings
    # or a moving critic head gets folded into the actor delta.
    sd_before = {"trunk.fc1.weight": torch.zeros(2, 2),
                 "critic_heads.0.weight": torch.zeros(2, 2)}
    sd_after = {"trunk.fc1.weight": torch.zeros(2, 2),
                "critic_heads.0.weight": torch.zeros(2, 2) + 1.0}
    deltas = param_deltas(sd_before, sd_after)
    assert deltas["critic_max_delta"] == 1.0
    assert deltas["actor_max_delta"] == 0.0


def test_enabled_mechanism_without_armed_evidence_fails():
    prof = {"reinforce": {"commitment_options": {"enabled": True}}}
    ok, notes = assess_mechanisms(prof, "iter 1 throughput ...")
    assert not ok and any("NO ARMED EVIDENCE" in n for n in notes)
    ok2, _ = assess_mechanisms(prof, "[commitment] ARMED durations=(1, 2, 4)")
    assert ok2


def test_redo_armed_check_reads_overridden_tau_from_profile():
    prof = {"reinforce": {"redo_enabled": True, "redo_tau": 0.1}}
    ok, notes = assess_mechanisms(prof, "[redo] ENABLED tau=0.1 every_iters=1 "
                                        "scope=fc1,fc2 sample=4096 "
                                        "reset_moments=false")
    assert ok and any("armed evidence present" in n for n in notes)


def test_control_arm_with_no_mechanisms_passes():
    ok, notes = assess_mechanisms({"reinforce": {}}, "anything")
    assert ok and any("control arm" in n for n in notes)


def test_bottom_k_profile_requires_mode_and_k_on_the_armed_line():
    # V32_REDO_BOTTOM_K_2026-08-28.md §12 item 4b: a bottom_k profile's
    # armed check must not be satisfied by a threshold-mode line at the
    # same (provenance-only) tau — the exact vacuity a tau-only regex
    # would ship as the tenth.
    prof = {"reinforce": {
        "redo_enabled": True, "redo_mode": "bottom_k",
        "redo_bottom_k": 2, "redo_check_every_iters": 5,
        "redo_tau": 0.025,
    }}
    correct_line = (
        "[redo] ENABLED tau=0.025 every_iters=5 scope=fc1,fc2 sample=4096 "
        "reset_moments=true mode=bottom_k k=2 recycle_scope=fc2"
    )
    ok, notes = assess_mechanisms(prof, correct_line)
    assert ok and any("armed evidence present" in n for n in notes)


def test_bottom_k_profile_rejects_a_threshold_mode_log_at_the_same_tau():
    # The executed failure this gate exists to catch: a synthetic
    # threshold-mode log at the SAME tau=0.025 the bottom_k profile pins
    # for provenance must NOT satisfy a bottom_k profile's armed check.
    prof = {"reinforce": {
        "redo_enabled": True, "redo_mode": "bottom_k",
        "redo_bottom_k": 2, "redo_check_every_iters": 5,
        "redo_tau": 0.025,
    }}
    threshold_line = (
        "[redo] ENABLED tau=0.025 every_iters=1 scope=fc1,fc2 sample=4096 "
        "reset_moments=true mode=threshold k=0 recycle_scope=fc1,fc2"
    )
    ok, notes = assess_mechanisms(prof, threshold_line)
    assert not ok and any("NO ARMED EVIDENCE" in n for n in notes)


def test_threshold_mode_profile_is_unaffected_by_the_bottom_k_gate():
    # Byte-identical to the pre-v32 behaviour when redo_mode is absent
    # (the schema default) — every existing threshold-mode config must
    # keep passing preflight exactly as it did before this change.
    prof = {"reinforce": {"redo_enabled": True, "redo_tau": 0.025}}
    line = ("[redo] ENABLED tau=0.025 every_iters=1 scope=fc1,fc2 "
            "sample=4096 reset_moments=true")
    ok, notes = assess_mechanisms(prof, line)
    assert ok and any("armed evidence present" in n for n in notes)


def test_steps_per_iter_derived_from_profile_not_hardcoded_default():
    # mario_1_1_v27_seed0.yaml: rollout_steps=1024, num_envs=60 -> 61440,
    # NOT the old hardcoded 92160 default (tuned to a different config).
    prof = {"reinforce": {"num_envs": 60, "rollout_steps": 1024,
                          "actor_freeze_steps": 18e6}}
    assert steps_per_iter_from_profile(prof) == 61440
    # With the stale 92160 default, a sentinel of 18e6 would falsely
    # pass (18e6 < 250*92160=23.04e6); with the true per-config value
    # it correctly fails (18e6 > 250*61440=15.36e6).
    ok_wrong, _ = assess_sentinels(prof, 250, 92160, 0)
    ok_true, msg_true = assess_sentinels(
        prof, 250, steps_per_iter_from_profile(prof), 0)
    assert ok_wrong and not ok_true and "outlasts" in msg_true


def test_sentinel_outlasting_the_budget_fails():
    prof = {"reinforce": {"actor_freeze_steps": 1e12}}
    ok, msg = assess_sentinels(prof, planned_iters=200,
                               steps_per_iter=92160, resume_iter=1120)
    assert not ok and "outlasts" in msg
    prof2 = {"reinforce": {"actor_freeze_steps": 0}}
    ok2, _ = assess_sentinels(prof2, 200, 92160, 1120)
    assert ok2
