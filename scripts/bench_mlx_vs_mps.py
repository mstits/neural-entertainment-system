#!/usr/bin/env python3
"""
MLX vs PyTorch-MPS — forward-pass bench on the policy net shape.

Gate for Batch 10. Builds the same Nature-DQN CNN in both frameworks
and measures per-step forward-pass time at batch_size=16 × 4 × 84 × 84
(the trainer's shape). Gate: MLX needs to be ≥1.2× faster to justify
the port effort.

The MLX port would touch src/models/policy_network.py,
src/training/trainer.py, src/training/behavior_cloning.py,
src/models/coreml_export.py, and the GA genome state-dict format.
Non-trivial. Gate it on real data.

Usage: .venv/bin/python scripts/bench_mlx_vs_mps.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import mlx.core as mx
    import mlx.nn as mlx_nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


NUM_ACTIONS = 6
FRAME_STACK = 4
FRAME_SIZE = 84
BATCH_SIZE = 16
WARMUP = 30
ITERS = 300


# --- PyTorch MPS --------------------------------------------------------

class TorchPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(FRAME_STACK, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, NUM_ACTIONS)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def bench_torch_mps():
    dev = torch.device("mps")
    net = TorchPolicy().to(dev).eval()
    # Warm-up: first dispatch compiles kernels.
    obs = torch.rand(BATCH_SIZE, FRAME_STACK, FRAME_SIZE, FRAME_SIZE, device=dev)
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = net(obs).argmax(dim=1)
    torch.mps.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(ITERS):
            acts = net(obs).argmax(dim=1)
            # Force sync to measure end-to-end latency per call (not just
            # kernel dispatch time).
            _ = acts.cpu().numpy()
    torch.mps.synchronize()
    dt = (time.perf_counter() - t0) * 1000.0
    return dt


# --- MLX ----------------------------------------------------------------

if MLX_AVAILABLE:
    class MlxPolicy(mlx_nn.Module):
        def __init__(self):
            super().__init__()
            # MLX conv layout is NHWC (unlike PyTorch NCHW). Channel
            # counts match the PyTorch arch.
            self.conv1 = mlx_nn.Conv2d(FRAME_STACK, 32, 8, 4)
            self.conv2 = mlx_nn.Conv2d(32, 64, 4, 2)
            self.conv3 = mlx_nn.Conv2d(64, 64, 3, 1)
            self.fc1 = mlx_nn.Linear(64 * 7 * 7, 512)
            self.fc2 = mlx_nn.Linear(512, NUM_ACTIONS)

        def __call__(self, x):
            x = mlx_nn.relu(self.conv1(x))
            x = mlx_nn.relu(self.conv2(x))
            x = mlx_nn.relu(self.conv3(x))
            # NHWC flatten.
            x = x.reshape((x.shape[0], -1))
            x = mlx_nn.relu(self.fc1(x))
            return self.fc2(x)
else:
    MlxPolicy = None  # type: ignore[assignment]


def bench_mlx():
    if not MLX_AVAILABLE:
        return None
    net = MlxPolicy()
    # MLX eval is lazy; force materialisation with `mx.eval`.
    obs = mx.random.uniform(
        shape=(BATCH_SIZE, FRAME_SIZE, FRAME_SIZE, FRAME_STACK),  # NHWC
    )
    # Warm-up.
    for _ in range(WARMUP):
        logits = net(obs)
        acts = mx.argmax(logits, axis=1)
        mx.eval(acts)

    t0 = time.perf_counter()
    for _ in range(ITERS):
        logits = net(obs)
        acts = mx.argmax(logits, axis=1)
        # Force host sync so we're measuring full end-to-end latency
        # equivalently to torch's .cpu().
        _ = np.asarray(acts)
    dt = (time.perf_counter() - t0) * 1000.0
    return dt


def main() -> int:
    print(f"== policy-net forward bench ==")
    print(f"  batch:      {BATCH_SIZE}")
    print(f"  input:      ({FRAME_STACK}, {FRAME_SIZE}, {FRAME_SIZE})")
    print(f"  arch:       Nature-DQN CNN (Conv×3 + Linear×2)")
    print(f"  warmup:     {WARMUP} iters")
    print(f"  measured:   {ITERS} iters")
    print(f"  MLX avail:  {MLX_AVAILABLE}")
    print()

    t_torch = bench_torch_mps()
    t_mlx = bench_mlx() if MLX_AVAILABLE else None

    print(f"{'framework':16}{'ms total':>14}{'ms/fwd':>12}{'fwd/sec':>14}")
    print("-" * 56)
    print(
        f"{'torch-mps':16}{t_torch:14.1f}{t_torch/ITERS:12.3f}"
        f"{ITERS/(t_torch/1000.0):14.0f}"
    )
    if t_mlx is not None:
        print(
            f"{'mlx-metal':16}{t_mlx:14.1f}{t_mlx/ITERS:12.3f}"
            f"{ITERS/(t_mlx/1000.0):14.0f}"
        )
        print()
        speedup = t_torch / t_mlx
        print(f"MLX vs PyTorch-MPS: {speedup:.3f}× "
              f"({(speedup - 1.0) * 100:+.1f}%)")
        if speedup >= 1.2:
            print("→ MEANINGFUL WIN — port warranted.")
        elif speedup >= 1.0:
            print("→ small win; port cost likely not worth it.")
        else:
            print("→ MLX slower — do not port.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
