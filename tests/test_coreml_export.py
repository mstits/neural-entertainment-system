"""Core ML export smoke: convert + load + predict round-trip."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from src.models.policy_network import PolicyNetwork


@pytest.mark.timeout(120)
def test_coreml_export_and_inference(tmp_path):
    """Convert a fresh network, load back, run one inference, assert shape."""
    coremltools = pytest.importorskip("coremltools")
    from src.models.coreml_export import CoreMLPolicy, maybe_export

    net = PolicyNetwork(num_actions=8)
    out_path = tmp_path / "policy.mlpackage"
    result = maybe_export(net, out_path, num_actions=8)

    if result is None:
        pytest.skip("coremltools.convert failed on this platform")

    assert result.exists()
    loader = CoreMLPolicy(result)
    # Deterministic inference: argmax must be REPEATABLE (same input →
    # same output). The old `0 <= action < 8` bar passed even if the
    # exporter wired sampling instead of argmax. Take 5 samples and
    # require they all match.
    obs = torch.zeros(4, 84, 84)
    actions_det = [loader.act(obs, deterministic=True) for _ in range(5)]
    assert all(0 <= a < 8 for a in actions_det)
    assert len(set(actions_det)) == 1, (
        f"deterministic mode must be repeatable; got {actions_det}"
    )

    # Stochastic inference: same range. Don't require diversity (a
    # near-deterministic distribution can produce identical samples).
    action = loader.act(obs, deterministic=False)
    assert 0 <= action < 8
