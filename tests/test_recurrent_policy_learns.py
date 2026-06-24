"""TileRecurrentPolicyNetwork trainability guard (SMB-LSTM frontier #7).

The recurrent tile policy is the planned lever for SMB past 1-2. Before
wiring it into the trainer, this guards the core mechanics it depends
on: forward_ac_recurrent + hidden-state threading + gradient flow
through the unrolled GRU (truncated BPTT).

The task is a minimal memory test only a recurrent policy can solve: a
bit `b` appears in feature 0 at step 0, then only noise; the net must
output `b` at the last step. A stateless net cannot (the info is gone);
the GRU carries it. Supervised (not full PPO) so it's fast + stable —
it isolates the recurrent forward/BPTT mechanics, which are the part
the PPO integration reuses. Seeded for determinism.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.tile_policy import TileRecurrentPolicyNetwork  # noqa: E402

FEAT, T, B = 16, 8, 64


def _memory_batch(rng):
    bits = torch.from_numpy(rng.integers(0, 2, size=B)).long()
    obs = torch.from_numpy(rng.standard_normal((T, B, FEAT)).astype("float32")) * 0.1
    obs[0, :, 0] = bits.float()
    obs[1:, :, 0] = 0.0
    return obs, bits


def test_recurrent_policy_learns_a_memory_task():
    import numpy as np
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = TileRecurrentPolicyNetwork(num_actions=2, feature_dim=FEAT, hidden_dim=32, gru_dim=32)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)

    acc = 0.0
    for _ in range(80):
        obs, bits = _memory_batch(rng)
        # Replay the sequence step-by-step through the GRU (truncated
        # BPTT), supervise the LAST step's logits to predict the bit.
        h = net.initial_hidden(B, obs.device)
        last_logits = None
        for t in range(T):
            last_logits, _v, h = net.forward_ac_recurrent(obs[t], h)
        loss = F.cross_entropy(last_logits, bits)
        opt.zero_grad()
        loss.backward()
        opt.step()
        acc = (last_logits.argmax(-1) == bits).float().mean().item()

    # A stateless net is stuck at chance (~0.5) on this task; the GRU
    # must carry the bit forward, so a working recurrent path learns it.
    assert acc > 0.85, (
        f"recurrent policy failed to learn the memory task (acc={acc:.3f}) — "
        "hidden-state threading or BPTT through forward_ac_recurrent is broken"
    )
