"""Policy network tests — shape, checkpoint round-trip, MPS if available."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from src.models.policy_network import PolicyNetwork, get_best_device


def test_forward_shape() -> None:
    net = PolicyNetwork(num_actions=8)
    x = torch.zeros(2, 4, 84, 84)
    logits = net(x)
    assert logits.shape == (2, 8)


def test_act_returns_valid_action_index() -> None:
    net = PolicyNetwork(num_actions=8)
    obs = torch.zeros(4, 84, 84)
    a_det = net.act(obs, deterministic=True)
    a_sto = net.act(obs, deterministic=False)
    assert 0 <= a_det < 8
    assert 0 <= a_sto < 8


def test_num_params_reasonable() -> None:
    net = PolicyNetwork(num_actions=8)
    # Nature-DQN is ~1.7M params; our action head is tiny so count is dominated
    # by the conv stack + first fc layer.
    assert 500_000 < net.num_params < 5_000_000


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    net = PolicyNetwork(num_actions=7, frame_stack=4, frame_size=84)
    x = torch.randn(1, 4, 84, 84)
    out_before = net(x)

    ckpt = tmp_path / "net.pt"
    net.save(ckpt)
    restored = PolicyNetwork.load(ckpt)

    assert restored.num_actions == 7
    out_after = restored(x)
    assert torch.allclose(out_before, out_after)


def test_load_rejects_arch_version_mismatch(tmp_path: Path) -> None:
    """A checkpoint from an incompatible architecture must fail loud, not
    silently load a mismatched net. (Audit F62.)"""
    import pytest
    net = PolicyNetwork(num_actions=7)
    ckpt = tmp_path / "net.pt"
    net.save(ckpt)
    data = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    data["arch_version"] = PolicyNetwork.ARCH_VERSION + 99
    torch.save(data, str(ckpt))
    with pytest.raises(ValueError, match="arch_version"):
        PolicyNetwork.load(ckpt)


def test_load_rejects_missing_parameters(tmp_path: Path) -> None:
    """A checkpoint missing required params would load a partly-random net;
    load() must refuse rather than return a broken policy. (Audit F62.)"""
    import pytest
    net = PolicyNetwork(num_actions=7)
    ckpt = tmp_path / "net.pt"
    net.save(ckpt)
    data = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    # Drop a parameter the architecture requires.
    victim = next(k for k in data["state_dict"] if k.endswith("weight"))
    del data["state_dict"][victim]
    torch.save(data, str(ckpt))
    with pytest.raises(ValueError, match="missing"):
        PolicyNetwork.load(ckpt)


def test_runs_on_mps_if_available() -> None:
    device = get_best_device()
    net = PolicyNetwork(num_actions=8).to(device)
    x = torch.zeros(1, 4, 84, 84, device=device)
    logits = net(x)
    assert logits.shape == (1, 8)
    assert logits.device.type == device.type
