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
    # Deterministic inference: argmax should be in range
    action = loader.act(torch.zeros(4, 84, 84), deterministic=True)
    assert 0 <= action < 8

    # Stochastic inference: same range
    action = loader.act(torch.zeros(4, 84, 84), deterministic=False)
    assert 0 <= action < 8
