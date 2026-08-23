"""experiment_preflight — the positive controls, pinned."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.experiment_preflight import (  # noqa: E402
    assess_learning, assess_mechanisms, assess_sentinels, param_deltas)


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


def test_enabled_mechanism_without_armed_evidence_fails():
    prof = {"reinforce": {"commitment_options": {"enabled": True}}}
    ok, notes = assess_mechanisms(prof, "iter 1 throughput ...")
    assert not ok and any("NO ARMED EVIDENCE" in n for n in notes)
    ok2, _ = assess_mechanisms(prof, "[commitment] ARMED durations=(1, 2, 4)")
    assert ok2


def test_control_arm_with_no_mechanisms_passes():
    ok, notes = assess_mechanisms({"reinforce": {}}, "anything")
    assert ok and any("control arm" in n for n in notes)


def test_sentinel_outlasting_the_budget_fails():
    prof = {"reinforce": {"actor_freeze_steps": 1e12}}
    ok, msg = assess_sentinels(prof, planned_iters=200,
                               steps_per_iter=92160, resume_iter=1120)
    assert not ok and "outlasts" in msg
    prof2 = {"reinforce": {"actor_freeze_steps": 0}}
    ok2, _ = assess_sentinels(prof2, 200, 92160, 1120)
    assert ok2
