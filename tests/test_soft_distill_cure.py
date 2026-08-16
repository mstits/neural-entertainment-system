"""Third-generation greedy-cure (SMB 1-2) — soft-target self-distillation
of scripts/soft_distill_cure.py.

Everything here runs on SYNTHETIC payloads and tensors — tiny
TilePolicyNetwork state dicts for the cure, hand-built logits for the
loss/report math, tmp dirs for the payload/install plumbing. No emulator,
no rollout. The one live-ish test is the --dry-run subprocess, which is
explicitly allowed to load the real source checkpoint and the real
sil_pairs.npz (loads and CPU forwards, never a rollout).

Pre-registration under test (mirrors of the in-file CONFIG):
* the teacher is the SOURCE net, bit-frozen — a distillation that moves
  the teacher is invalid by construction;
* the student's loss uses the teacher's T=1 distribution as the target
  and tempers the STUDENT's logits at tau < 1 (default 0.7, forward KL
  default; 'rkl' is the registered mode-seeking alternative) — the
  executed-action labels in sil_pairs.npz are NEVER read (they carry
  ~25% sticky-forced repeats, the exact label noise that sank the two
  BC cures);
* razor-thin-wrong-argmax sanity: where the teacher splits 0.45/0.40 and
  the student's argmax sits on the 0.40 action, one gradient-descent
  step must move the student toward the 0.45 action (both objectives);
* critic ALWAYS frozen; out-of-scope params bit-identical; L2-to-teacher
  trunk anchor bounds drift;
* runs/night2/cured_v3.pt carries the exact night2 cured.pt payload
  shape (iter/net_state_dict/provenance/night2 — no optimizer state, no
  anticollapse);
* --install refuses an existing runs/night2/cured.pt without --force.
"""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.night2_runner import (  # noqa: E402
    CONFIG as NIGHT2_CONFIG, CURED_SEED_ITER,
)
from scripts.soft_distill_cure import (  # noqa: E402
    CONFIG, argmax_report, build_gate_commands, install_cured,
    load_success_obs, soft_distill_cure, soft_distill_loss, write_cured_v3,
)


# ---- synthetic fixtures --------------------------------------------------


def _tile_state_dict(num_actions=4, feature_dim=12, hidden=8, trunk=6):
    from src.models.tile_policy import TilePolicyNetwork
    torch.manual_seed(7)
    net = TilePolicyNetwork(num_actions=num_actions, feature_dim=feature_dim,
                            hidden_dim=hidden, trunk_dim=trunk)
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def _fixture(n=256, feature_dim=12, num_actions=4, seed=3):
    sd = _tile_state_dict(num_actions=num_actions, feature_dim=feature_dim)
    payload = {"iter": 910, "net_state_dict": sd}
    rng = np.random.default_rng(seed)
    obs = rng.integers(-4, 4, size=(n, feature_dim)).astype(np.int8)
    return payload, sd, obs


def _cure(payload, obs, **overrides):
    kw = dict(epochs=2, lr=1e-2, batch_size=64, seed=0,
              scope="actor+trunk", objective="fkl", tau_student=0.7,
              teacher_temp=1.0, l2_coef=1e-4)
    kw.update(overrides)
    return soft_distill_cure(payload, obs, **kw)


# ---- config pre-registration --------------------------------------------


def test_config_pins_match_night2():
    """The v3 cure distills the SAME pinned source the night-2 BC cures
    started from, off the SAME success-state npz, into the night2 run
    dir — one lineage, byte-comparable receipts."""
    assert CONFIG["source_checkpoint"] == NIGHT2_CONFIG["source_checkpoint"]
    assert (CONFIG["source_checkpoint_sha256"]
            == NIGHT2_CONFIG["source_checkpoint_sha256"])
    assert CONFIG["pairs_in"] == NIGHT2_CONFIG["sil_pairs_out"]
    assert CONFIG["install_target"] == NIGHT2_CONFIG["cured_out"]
    assert CONFIG["cured_v3_out"] == "runs/night2/cured_v3.pt"


def test_registered_defaults():
    # Teacher at T=1 (the classic KD T>1 SOFTENS — rejected, see module
    # docstring), student tempered BELOW 1, forward KL, width-safe scope,
    # drift insurance on.
    assert CONFIG["teacher_temp"] == 1.0
    assert CONFIG["tau_student"] == 0.7
    assert CONFIG["objective"] == "fkl"
    assert CONFIG["scope"] == "actor+trunk"
    assert CONFIG["l2_coef"] == 1e-4


# ---- the loss ------------------------------------------------------------


def _logits_from_probs(probs):
    return torch.log(torch.tensor([probs], dtype=torch.float64))


def test_loss_zero_at_tempered_match_both_objectives():
    """Both KL forms bottom out at exactly zero when the student's
    TEMPERED distribution equals the teacher's T=1 distribution, i.e.
    student logits = tau * teacher log-probs (+ any constant)."""
    torch.manual_seed(0)
    t = torch.randn(5, 4, dtype=torch.float64)
    s = 0.7 * F.log_softmax(t, dim=-1) + 3.0  # +3: shift-invariance too
    for obj in ("fkl", "rkl"):
        loss = soft_distill_loss(s, t, objective=obj, tau_student=0.7,
                                 teacher_temp=1.0)
        assert abs(float(loss)) < 1e-9, obj


def test_loss_positive_off_the_fixed_point():
    t = _logits_from_probs([0.45, 0.40, 0.15])
    s = _logits_from_probs([0.40, 0.45, 0.15])
    for obj in ("fkl", "rkl"):
        loss = soft_distill_loss(s, t, objective=obj, tau_student=0.7,
                                 teacher_temp=1.0)
        assert float(loss) > 0.0, obj


def test_gradient_moves_toward_teacher_preferred_action_fkl():
    """THE razor-thin case the cure exists for: teacher 0.45/0.40/0.15,
    student argmax on the 0.40 action. A gradient-descent step must
    raise the 0.45 action's logit and lower the 0.40 action's — i.e.
    grad[0] < 0 < grad[1]."""
    t = _logits_from_probs([0.45, 0.40, 0.15])
    s = _logits_from_probs([0.40, 0.45, 0.15]).requires_grad_(True)
    loss = soft_distill_loss(s, t, objective="fkl", tau_student=0.7,
                             teacher_temp=1.0)
    loss.backward()
    g = s.grad[0]
    assert float(g[0]) < 0.0  # descent raises the 0.45 action
    assert float(g[1]) > 0.0  # descent lowers the wrongly-argmaxed 0.40


def test_gradient_moves_toward_teacher_preferred_action_rkl():
    # Reverse KL is mode-seeking, but in the razor-thin flipped case the
    # step direction must agree: toward the teacher's preferred action.
    t = _logits_from_probs([0.45, 0.40, 0.15])
    s = _logits_from_probs([0.40, 0.45, 0.15]).requires_grad_(True)
    loss = soft_distill_loss(s, t, objective="rkl", tau_student=0.7,
                             teacher_temp=1.0)
    loss.backward()
    g = s.grad[0]
    assert float(g[0]) < 0.0
    assert float(g[1]) > 0.0


def test_objectives_differ_where_teacher_is_multimodal():
    # Covering vs mode-seeking must be two different numbers on a
    # bimodal teacher with a one-mode student.
    t = _logits_from_probs([0.48, 0.48, 0.04])
    s = _logits_from_probs([0.90, 0.05, 0.05])
    fkl = soft_distill_loss(s, t, objective="fkl", tau_student=0.7,
                            teacher_temp=1.0)
    rkl = soft_distill_loss(s, t, objective="rkl", tau_student=0.7,
                            teacher_temp=1.0)
    assert abs(float(fkl) - float(rkl)) > 1e-3


def test_loss_rejects_unknown_objective_and_bad_temps():
    t = torch.zeros(1, 3)
    with pytest.raises(ValueError):
        soft_distill_loss(t, t, objective="ce", tau_student=0.7,
                          teacher_temp=1.0)
    with pytest.raises(ValueError):
        soft_distill_loss(t, t, objective="fkl", tau_student=0.0,
                          teacher_temp=1.0)
    with pytest.raises(ValueError):
        soft_distill_loss(t, t, objective="fkl", tau_student=0.7,
                          teacher_temp=-1.0)


# ---- no-label discipline -------------------------------------------------


def test_cure_signature_cannot_take_action_labels():
    """The gen-2 failure was structured label noise (executed actions
    under sticky 0.25). The v3 cure is made incapable of repeating it:
    the training entrypoint accepts states only — there is no parameter
    an action array could even be passed through."""
    params = list(inspect.signature(soft_distill_cure).parameters)
    assert params[:2] == ["payload", "obs"]
    assert not any("act" in p or "label" in p for p in params)


def test_load_success_obs_ignores_action_labels(tmp_path):
    obs = np.arange(24, dtype=np.int8).reshape(4, 6)
    act = np.array([1, 2, 1, 0], dtype=np.int64)
    p = tmp_path / "pairs.npz"
    np.savez(p, n=1, obs_0=obs, act_0=act,
             traj_len=np.array([4]), label_max_gx=np.array([3400]),
             label_episode_success=np.array([True]),
             provenance=json.dumps({"mode": "collection"}))
    got, meta = load_success_obs(p)
    np.testing.assert_array_equal(got, obs)
    assert meta["n_states"] == 4
    assert meta["n_trajs"] == 1
    # The labels exist in the file and are DECLARED ignored — never
    # returned, never consumable by the caller.
    assert meta["has_action_labels"] is True
    assert meta["action_labels_used"] is False


# ---- the cure: freeze partition + teacher integrity ----------------------


def test_teacher_bit_frozen_and_source_payload_untouched():
    payload, sd, obs = _fixture()
    before = {k: v.clone() for k, v in sd.items()}
    cured, stats = _cure(payload, obs)
    assert stats["teacher_bit_frozen"] is True
    for k in before:  # the input payload is never mutated
        assert torch.equal(payload["net_state_dict"][k], before[k]), k


def test_scope_actor_freezes_trunk_and_critic_bit_identical():
    payload, sd, obs = _fixture()
    cured, _ = _cure(payload, obs, scope="actor")
    assert not torch.equal(cured["actor.weight"], sd["actor.weight"])
    for k in sd:
        if k.startswith("actor."):
            continue
        # Frozen means BIT-IDENTICAL, not merely close.
        assert torch.equal(cured[k], sd[k]), k


def test_scope_actor_trunk_moves_trunk_but_critic_stays_frozen():
    payload, sd, obs = _fixture()
    cured, _ = _cure(payload, obs, scope="actor+trunk")
    assert not torch.equal(cured["actor.weight"], sd["actor.weight"])
    assert not torch.equal(cured["fc1.weight"], sd["fc1.weight"])
    assert torch.equal(cured["critic.weight"], sd["critic.weight"])
    assert torch.equal(cured["critic.bias"], sd["critic.bias"])


def test_cure_is_deterministic():
    payload, _, obs = _fixture()
    a, _ = _cure(payload, obs, seed=11)
    b, _ = _cure(payload, obs, seed=11)
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_cure_rejects_empty_states_and_bad_scope():
    payload, _, obs = _fixture()
    with pytest.raises(ValueError):
        _cure(payload, obs[:0])
    with pytest.raises(ValueError):
        _cure(payload, obs, scope="everything")


# ---- L2-to-teacher trunk anchor -------------------------------------------


def test_l2_penalty_holds_trunk_near_teacher():
    payload, sd, obs = _fixture()
    free, _ = _cure(payload, obs, epochs=3, lr=1e-2, l2_coef=0.0)
    held, _ = _cure(payload, obs, epochs=3, lr=1e-2, l2_coef=50.0)

    def trunk_drift(cured):
        return sum(
            float((cured[k] - sd[k]).norm()) for k in sd
            if k.startswith(("fc1.", "norm1.", "fc2.", "norm2.")))

    assert trunk_drift(free) > 0.0
    assert trunk_drift(held) < trunk_drift(free)


# ---- report math ----------------------------------------------------------


def test_argmax_report_counts_changes_correctly():
    teacher = torch.tensor([[2.0, 1.0, 0.0],
                            [0.0, 3.0, 1.0],
                            [1.0, 0.0, 4.0],
                            [5.0, 4.5, 0.0]])
    pre = teacher.clone()          # the student starts as a bit-copy
    post = teacher.clone()
    post[3] = torch.tensor([4.5, 5.0, 0.0])  # one targeted flip
    r = argmax_report(teacher, pre, post)
    assert r["teacher_agreement_pre"] == 1.0
    assert r["teacher_agreement_post"] == pytest.approx(0.75)
    assert r["argmax_changed_frac"] == pytest.approx(0.25)
    assert r["n_changed"] == 1
    # Action-gap = mean (top1 - top2) of the T=1 logits.
    assert r["mean_gap_pre"] == pytest.approx((1.0 + 2.0 + 3.0 + 0.5) / 4)
    assert r["mean_gap_post"] == pytest.approx((1.0 + 2.0 + 3.0 + 0.5) / 4)
    assert r["mean_gap_delta"] == pytest.approx(0.0)
    assert r["argmax_counts_pre"] == [2, 1, 1]
    assert r["argmax_counts_post"] == [1, 2, 1]


def test_cure_stats_carry_the_report():
    payload, _, obs = _fixture()
    _, stats = _cure(payload, obs)
    assert stats["teacher_agreement_pre"] == 1.0  # student starts bit-equal
    assert 0.0 <= stats["argmax_changed_frac"] <= 1.0
    assert stats["n_states"] == obs.shape[0]
    assert stats["objective"] == "fkl"
    assert stats["tau_student"] == 0.7
    assert "mean_gap_pre" in stats and "mean_gap_post" in stats


# ---- payload format --------------------------------------------------------


def test_cured_v3_payload_matches_night2_shape(tmp_path):
    """Exactly the night2 cured.pt payload shape: iter (the consol2 seed
    iter) + net_state_dict + provenance + night2 metadata. Deliberately
    NO optimizer_state_dict (the cure invalidates the source's Adam
    moments) and NO anticollapse (its snapshot is the pre-cure net)."""
    sd = _tile_state_dict()
    out = tmp_path / "cured_v3.pt"
    write_cured_v3(out, sd, stats={"final_loss": 0.1},
                   provenance={"scope": "actor+trunk"})
    payload = torch.load(str(out), map_location="cpu", weights_only=False)
    assert set(payload.keys()) == {"iter", "net_state_dict", "provenance",
                                   "night2"}
    assert payload["iter"] == CURED_SEED_ITER
    assert "optimizer_state_dict" not in payload
    assert "anticollapse" not in payload
    assert set(payload["net_state_dict"]) == set(sd)
    for v in payload["net_state_dict"].values():
        assert v.device.type == "cpu"
    assert payload["provenance"] == "night2_soft_distill_cure"
    assert payload["night2"]["stats"]["final_loss"] == 0.1
    assert payload["night2"]["scope"] == "actor+trunk"


# ---- --install plumbing -----------------------------------------------------


def _v3_file(tmp_path):
    sd = _tile_state_dict()
    v3 = tmp_path / "cured_v3.pt"
    write_cured_v3(v3, sd, stats={}, provenance={})
    return v3


def test_install_writes_when_target_absent(tmp_path):
    v3 = _v3_file(tmp_path)
    target = tmp_path / "cured.pt"
    note = install_cured(v3, target, force=False)
    assert target.exists()
    assert target.read_bytes() == v3.read_bytes()
    assert "installed" in note


def test_install_refuses_existing_target_without_force(tmp_path):
    v3 = _v3_file(tmp_path)
    target = tmp_path / "cured.pt"
    target.write_bytes(b"the failed BC cure lineage")
    with pytest.raises(RuntimeError):
        install_cured(v3, target, force=False)
    # Refusal must leave the existing artifact untouched.
    assert target.read_bytes() == b"the failed BC cure lineage"


def test_install_force_overwrites_and_backs_up(tmp_path):
    v3 = _v3_file(tmp_path)
    target = tmp_path / "cured.pt"
    target.write_bytes(b"the failed BC cure lineage")
    note = install_cured(v3, target, force=True)
    assert target.read_bytes() == v3.read_bytes()
    backups = list(tmp_path.glob("cured.pt.pre_v3.*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"the failed BC cure lineage"
    assert "backed up" in note


# ---- gate command handoff ----------------------------------------------------


def test_gate_commands_are_night2s_with_cured_v3_swapped_in():
    """The emulator is owned by the consolidation run, so the gate is a
    HANDOFF: the exact two eval_game commands from night2's step 3, with
    the checkpoint swapped to cured_v3.pt — byte-comparable protocol."""
    ckpt = ROOT / CONFIG["cured_v3_out"]
    cmds = build_gate_commands(ckpt)
    honest = " ".join(cmds["honest"])
    det = " ".join(cmds["det"])
    assert str(ckpt) in honest and str(ckpt) in det
    assert "--episodes 50" in honest
    assert "--sticky-prob 0.25" in honest and "--start-jitter 16" in honest
    assert "per-episode" in honest and "sampled" not in honest
    assert "--episodes 10" in det
    assert "--sticky-prob 0.0" in det and "--start-jitter 0" in det
    assert "s_000559.state" in det


# ---- dry-run (live; loads + CPU forwards, no rollouts) -----------------------


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_dry_run_passes_live():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "soft_distill_cure.py"),
         "--dry-run"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=540)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ALL CHECKS PASSED" in proc.stdout
    assert "FAIL" not in proc.stdout
    # The no-label discipline must be REPORTED, not silent.
    assert "IGNORED" in proc.stdout
    # The handoff commands must be printed for the operator's window.
    assert "eval_game.py" in proc.stdout
