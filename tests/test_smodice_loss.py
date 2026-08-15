"""SMODICE loss math on a synthetic 2-state MDP (expert always goes right).

s0=[1,0], s1=[0,1]; action 1 (right) lands in s1, action 0 (left) lands in
s0. The expert's state occupancy concentrates on s1, so: the state
discriminator must separate s1 from s0; the chi-square Fenchel-dual value
objective must converge with higher occupancy weights on the rightward
transitions; and the weighted-BC extraction into the standard tile head
must prefer right. Plus the two terminal-handling contracts the IQ run
lacked: done=1 grounds V(s')=0 (v_next excluded), truncated=1 bootstraps
and is never folded into done.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.smodice_distill import (
    bellman_residual,
    clip_reward,
    extract_policy,
    initial_state_mask,
    load_expert_transitions,
    main,
    smodice_weights,
    terminal_mask,
    train_discriminator,
    train_value,
)

S0 = [1.0, 0.0]
S1 = [0.0, 1.0]
LEFT, RIGHT = 0, 1


def _mixed_offline(n_per=32):
    s, a, ns = [], [], []
    for _ in range(n_per):
        s += [S0, S1, S0, S1]
        a += [RIGHT, RIGHT, LEFT, LEFT]
        ns += [S1, S1, S0, S0]
    return (torch.tensor(s), torch.tensor(a, dtype=torch.int64),
            torch.tensor(ns), torch.zeros(len(a)))


def _expert_states(n=100):
    return torch.tensor([S1] * (9 * n // 10) + [S0] * (n // 10))


def test_discriminator_separates_expert_states():
    torch.manual_seed(0)
    S, _, _, _ = _mixed_offline()
    disc = train_discriminator(_expert_states(), S, hidden_dim=32,
                               epochs=200, lr=1e-2, batch=128)
    with torch.no_grad():
        r1 = float(disc(torch.tensor([S1])))
        r0 = float(disc(torch.tensor([S0])))
    assert r1 > r0 + 1.0


def test_done_grounds_next_value():
    r = torch.tensor([0.5])
    v_s = torch.tensor([2.0])
    done = torch.tensor([1.0])
    e_a = bellman_residual(r, v_s, torch.tensor([5.0]), done, 0.99)
    e_b = bellman_residual(r, v_s, torch.tensor([-7.0]), done, 0.99)
    assert torch.allclose(e_a, e_b)
    assert torch.allclose(e_a, r - v_s)


def test_truncated_bootstraps_next_value():
    r = torch.tensor([0.5])
    v_s = torch.tensor([2.0])
    v_ns = torch.tensor([3.0])
    done = terminal_mask(np.array([0], dtype=np.int8),
                         np.array([1], dtype=np.int8))
    assert done.tolist() == [0]
    e = bellman_residual(r, v_s, v_ns, torch.from_numpy(done).float(), 0.9)
    assert torch.allclose(e, r + 0.9 * v_ns - v_s)


def test_terminal_mask_rejects_done_and_truncated_overlap():
    with pytest.raises(ValueError):
        terminal_mask(np.array([1, 0], dtype=np.int8),
                      np.array([1, 0], dtype=np.int8))


def test_initial_state_mask_first_row_per_window():
    wid = np.array([0, 0, 0, 1, 1, 2])
    np.testing.assert_array_equal(
        initial_state_mask(wid),
        np.array([True, False, False, True, False, True]))


def test_value_converges_and_weights_prefer_expert_transitions():
    torch.manual_seed(0)
    S, A, NS, done = _mixed_offline()
    R = torch.where(S[:, 1] > 0.5, 2.0, -2.0)
    init = torch.tensor([S0] * 8)
    vnet, hist = train_value(init, S, NS, R, done, gamma=0.9, f="chi",
                             hidden_dim=32, epochs=300, lr=1e-2,
                             batch=len(S))
    assert hist[-1] < hist[0]
    w = smodice_weights(vnet, S, NS, R, done, 0.9, "chi")
    assert torch.all(w >= 0)
    right = A == RIGHT
    assert float(w[right].mean()) > float(w[~right].mean())


def test_kl_weights_are_normalized_importance_weights():
    torch.manual_seed(0)
    S, A, NS, done = _mixed_offline(n_per=4)
    R = torch.where(S[:, 1] > 0.5, 2.0, -2.0)
    init = torch.tensor([S0] * 4)
    vnet, _ = train_value(init, S, NS, R, done, gamma=0.9, f="kl",
                          hidden_dim=16, epochs=50, lr=1e-2, batch=len(S))
    w = smodice_weights(vnet, S, NS, R, done, 0.9, "kl")
    assert torch.all(w > 0)
    assert abs(float(w.sum()) - len(w)) < 1e-3


def test_extract_policy_prefers_weighted_action():
    torch.manual_seed(0)
    S, A, NS, done = _mixed_offline()
    w = torch.where(A == RIGHT, 2.0, 0.1)
    net, acc = extract_policy(S, A, w, num_actions=2, hidden_dim=16,
                              trunk_dim=8, epochs=100, lr=1e-2, batch=64)
    with torch.no_grad():
        logits, _ = net.forward_ac(torch.tensor([S0, S1]))
    assert logits.argmax(dim=1).tolist() == [RIGHT, RIGHT]
    assert acc > 0.0


def test_clip_reward_bounds_and_noop():
    r = torch.tensor([-80.0, -3.0, 0.0, 7.0])
    clipped = clip_reward(r, 5.0)
    assert clipped.tolist() == [-5.0, -3.0, 0.0, 5.0]
    assert clip_reward(r, None) is r


def test_clip_reward_rejects_nonpositive():
    with pytest.raises(ValueError):
        clip_reward(torch.tensor([1.0]), 0.0)


def test_disc_weight_decay_shrinks_logit_scale():
    S, _, _, _ = _mixed_offline()
    torch.manual_seed(0)
    free = train_discriminator(_expert_states(), S, hidden_dim=32,
                               epochs=200, lr=1e-2, batch=128)
    torch.manual_seed(0)
    decayed = train_discriminator(_expert_states(), S, hidden_dim=32,
                                  epochs=200, lr=1e-2, batch=128,
                                  weight_decay=1.0)
    with torch.no_grad():
        probe = torch.tensor([S0, S1])
        assert (float(decayed(probe).abs().max())
                < float(free(probe).abs().max()))


def test_load_expert_transitions_consecutive_within_windows(tmp_path):
    obs_0 = np.arange(4 * 3, dtype=np.int8).reshape(4, 3)
    act_0 = np.array([2, 0, 1, 3], dtype=np.int64)
    obs_1 = np.arange(100, 100 + 3 * 3, dtype=np.int8).reshape(3, 3)
    act_1 = np.array([1, 1, 0], dtype=np.int64)
    p = tmp_path / "demo.npz"
    np.savez(p, n=np.int64(2), obs_0=obs_0, act_0=act_0,
             obs_1=obs_1, act_1=act_1)

    ex = load_expert_transitions([p])
    assert ex["window_lengths"] == [3, 2]
    np.testing.assert_array_equal(ex["state"][:3], obs_0[:-1])
    np.testing.assert_array_equal(ex["next_state"][:3], obs_0[1:])
    np.testing.assert_array_equal(ex["action"], [2, 0, 1, 1, 1])
    # No window-crossing row: obs_0's last state never pairs with obs_1.
    np.testing.assert_array_equal(ex["state"][3:], obs_1[:-1])
    np.testing.assert_array_equal(ex["done"], np.zeros(5, dtype=np.int8))
    np.testing.assert_array_equal(ex["truncated"], [0, 0, 1, 0, 1])


def test_load_expert_transitions_rejects_transition_schema(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, state=np.zeros((2, 3), dtype=np.int8))
    with pytest.raises(ValueError):
        load_expert_transitions([p])


def _write_cli_fixtures(tmp_path, feature_dim=6, num_actions=2):
    rng = np.random.default_rng(0)
    n = 40
    data = tmp_path / "transitions.npz"
    trunc = np.zeros(n, dtype=np.int8)
    trunc[9::10] = 1
    np.savez(data,
             state=rng.integers(0, 2, (n, feature_dim)).astype(np.int8),
             action=rng.integers(0, num_actions, n).astype(np.int64),
             next_state=rng.integers(0, 2, (n, feature_dim)).astype(np.int8),
             done=np.zeros(n, dtype=np.int8),
             truncated=trunc,
             window_id=np.repeat(np.arange(4, dtype=np.int64), 10))
    demo = tmp_path / "demo.npz"
    np.savez(demo, n=np.int64(1),
             obs_0=rng.integers(0, 2, (8, feature_dim)).astype(np.int8),
             act_0=rng.integers(0, num_actions, 8).astype(np.int64))
    profile = tmp_path / "profile.yaml"
    profile.write_text("action_space:\n- [noop]\n- [right]\n")
    return data, demo, profile


def _run_cli(tmp_path, monkeypatch, extra):
    data, demo, profile = _write_cli_fixtures(tmp_path)
    out = tmp_path / "ckpt"
    monkeypatch.setattr("sys.argv", [
        "smodice_distill.py", "--data", str(data),
        "--expert-demos", str(demo), "--profile", str(profile),
        "--out", str(out), "--epochs", "2", "--hidden", "16",
        "--trunk-dim", "8", "--seed", "0"] + extra)
    assert main() == 0
    return out


def test_cli_flags_wire_through(tmp_path, monkeypatch, capsys):
    out = _run_cli(tmp_path, monkeypatch, [
        "--r-clip", "1.0", "--disc-epochs", "1",
        "--disc-weight-decay", "1e-3", "--union-expert",
        "--extract-epochs", "3", "--extract-lr", "1e-3"])
    logtext = capsys.readouterr().out
    # union: 40 offline + 7 expert-derived rows, truncated 4 + 1 windows.
    assert "union-expert: +7 transitions from 1 demo windows" in logtext
    assert "47 transitions" in logtext and "windows=5" in logtext
    assert "(1 epochs, wd=0.001)" in logtext
    assert "(clipped to ±1)" in logtext
    assert "extract epoch   2" in logtext  # --extract-epochs 3 ran
    assert (out / "vanilla_ppo_iter_00000.pt").exists()
    assert (out / "smodice_stage12.pt").exists()


def test_cli_unweighted_skips_stages_1_2(tmp_path, monkeypatch, capsys):
    out = _run_cli(tmp_path, monkeypatch, ["--unweighted"])
    logtext = capsys.readouterr().out
    assert "skipping stages 1-2" in logtext
    assert "stage 1" not in logtext and "stage 2" not in logtext
    assert (out / "vanilla_ppo_iter_00000.pt").exists()
    assert not (out / "smodice_stage12.pt").exists()


def test_cli_defaults_unchanged(tmp_path, monkeypatch, capsys):
    out = _run_cli(tmp_path, monkeypatch, [])
    logtext = capsys.readouterr().out
    assert "40 transitions" in logtext and "windows=4" in logtext
    assert "(2 epochs, wd=0.0)" in logtext
    assert "clipped" not in logtext
    assert (out / "vanilla_ppo_iter_00000.pt").exists()
    assert (out / "smodice_stage12.pt").exists()


def test_full_pipeline_prefers_right():
    torch.manual_seed(0)
    S, A, NS, done = _mixed_offline()
    disc = train_discriminator(_expert_states(), S, hidden_dim=32,
                               epochs=200, lr=1e-2, batch=128)
    with torch.no_grad():
        R = disc(S)
    init = torch.tensor([S0] * 8)
    vnet, _ = train_value(init, S, NS, R, done, gamma=0.9, f="chi",
                          hidden_dim=32, epochs=300, lr=1e-2, batch=len(S))
    w = smodice_weights(vnet, S, NS, R, done, 0.9, "chi")
    net, _ = extract_policy(S, A, w, num_actions=2, hidden_dim=16,
                            trunk_dim=8, epochs=100, lr=1e-2, batch=64)
    with torch.no_grad():
        logits, _ = net.forward_ac(torch.tensor([S0, S1]))
    assert logits.argmax(dim=1).tolist() == [RIGHT, RIGHT]
