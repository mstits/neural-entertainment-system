#!/usr/bin/env python3
"""
Core ML / ANE inference bench for the policy network.

Compares three compute-unit targets for the same trained-shape CNN:

  - PyTorch MPS (GPU)
  - Core ML with `compute_units=ALL` (ANE preferred, GPU/CPU fallback)
  - Core ML with `compute_units=CPU_ONLY` (baseline)

Use case: the replay viewer and future deployed-inference paths
could use ANE instead of MPS for 3-5× power efficiency (and
sometimes faster wall-time on Apple Silicon with a 16-core ANE).
Not a training-path optimization — training needs backprop which
ANE doesn't support.

Bench shape matches the trainer's: batch=16 × 4 × 84 × 84. Batch=1
is also measured since the replay viewer uses single-frame
inference.

Usage: .venv/bin/python scripts/bench_coreml_ane.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Silence the "Torch version x has not been tested" coremltools warning
# since we know 2.11 works fine for our simple Nature-DQN.
os.environ.setdefault("COREMLTOOLS_DISABLE_TORCH_VERSION_WARNINGS", "1")

import numpy as np
import torch


NUM_ACTIONS = 6


def bench_mps(net, batch: int, iters: int, warmup: int = 20) -> float:
    dev = torch.device("mps")
    net = net.to(dev).eval()
    obs = torch.rand(batch, 4, 84, 84, device=dev)
    with torch.no_grad():
        for _ in range(warmup):
            _ = net(obs).argmax(dim=1).cpu().numpy()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = net(obs).argmax(dim=1).cpu().numpy()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def bench_coreml(mlmodel_path: str, batch: int, iters: int, warmup: int = 20) -> float:
    import coremltools as ct
    m = ct.models.MLModel(mlmodel_path)
    obs = np.random.rand(batch, 4, 84, 84).astype(np.float32)
    spec = m.get_spec().description
    in_name = next(iter(spec.input)).name
    out_name = next(iter(spec.output)).name
    for _ in range(warmup):
        _ = m.predict({in_name: obs})[out_name]
    t0 = time.perf_counter()
    for _ in range(iters):
        out = m.predict({in_name: obs})[out_name]
        _ = out.argmax()
    return (time.perf_counter() - t0) * 1000.0


def main() -> int:
    import coremltools as ct  # raises if not installed

    from src.models.policy_network import PolicyNetwork

    batch_sizes = [1, 16]
    iters = 200
    results: list[tuple] = []

    for batch in batch_sizes:
        print(f"\n== batch_size = {batch} ==")
        net = PolicyNetwork(num_actions=NUM_ACTIONS).eval()

        # PyTorch MPS
        t_mps = bench_mps(net, batch, iters)
        results.append(("MPS", batch, t_mps))
        print(f"  MPS:              {t_mps:8.1f} ms  ({t_mps/iters:.3f} ms/fwd)")

        # Core ML export (all compute units — ANE preferred)
        example = torch.zeros(batch, 4, 84, 84)
        traced = torch.jit.trace(net.to("cpu"), example)
        for label, units in [
            ("CoreML(ANE)", ct.ComputeUnit.ALL),
            ("CoreML(CPU)", ct.ComputeUnit.CPU_ONLY),
        ]:
            mlmodel = ct.convert(
                traced,
                inputs=[ct.TensorType(
                    name="obs", shape=(batch, 4, 84, 84), dtype=np.float32,
                )],
                outputs=[ct.TensorType(name="logits", dtype=np.float32)],
                convert_to="mlprogram",
                compute_units=units,
                minimum_deployment_target=ct.target.macOS14,
            )
            tmpath = f"/tmp/policy_ane_bench_b{batch}.mlpackage"
            mlmodel.save(tmpath)
            t_cml = bench_coreml(tmpath, batch, iters)
            results.append((label, batch, t_cml))
            print(f"  {label:16}  {t_cml:8.1f} ms  ({t_cml/iters:.3f} ms/fwd)")

    print("\n== summary ==")
    print(f"{'target':20}{'batch':>8}{'ms/fwd':>12}{'vs MPS':>12}")
    print("-" * 52)
    for label, batch, t in results:
        if label == "MPS":
            ref = t
            print(f"{label:20}{batch:>8}{t/iters:>12.3f}{'1.000x':>12}")
            continue
        # Find MPS baseline for this batch.
        ref = next(x[2] for x in results if x[0] == "MPS" and x[1] == batch)
        ratio = t / ref
        print(f"{label:20}{batch:>8}{t/iters:>12.3f}{ratio:>11.3f}x")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
