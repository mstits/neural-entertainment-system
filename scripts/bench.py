"""
Performance benchmarks with regression thresholds.

Measures:
  * Frame-preprocessing throughput (ops/sec)
  * Policy forward pass latency (ms) on CPU and MPS
  * ParallelPool step throughput with FakeNESEnvironment
  * BC pretraining step time

Each benchmark prints its result and, if a baseline file exists, fails
with exit code 1 when any metric regresses by more than REGRESSION_TOL
(default 10%). First run creates the baseline.

Usage:
    python scripts/bench.py [--update-baseline]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


BASELINE_PATH = Path(__file__).resolve().parent.parent / "bench_baseline.json"
REGRESSION_TOL = 0.10  # fail if 10% slower than baseline


def _time(fn, *args, warmup: int = 2, trials: int = 5, **kwargs) -> float:
    """Return the median of `trials` wall-clock times in seconds."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def bench_preprocess_frame() -> float:
    from src.emulation.frame_utils import preprocess_frame
    frame = np.random.randint(0, 256, (240, 256, 3), dtype=np.uint8)
    def loop():
        for _ in range(200):
            preprocess_frame(frame)
    t = _time(loop)
    return 200.0 / t  # ops/sec


def bench_policy_forward(device: torch.device) -> float:
    from src.models.policy_network import PolicyNetwork
    net = PolicyNetwork(num_actions=12).to(device)
    net.eval()
    # Use random inputs (not zeros) so we measure real-world throughput.
    # Some conv kernels short-circuit on all-zero inputs, producing
    # over-optimistic numbers that don't match real training latency.
    g = torch.Generator(device="cpu").manual_seed(0)
    batch = torch.randint(
        0, 256, (8, 4, 84, 84), generator=g, dtype=torch.uint8
    ).to(device).float().div_(255.0)
    def loop():
        with torch.no_grad():
            for _ in range(50):
                net(batch)
        if device.type == "mps":
            torch.mps.synchronize()
    t = _time(loop)
    return (50.0 * 8) / t  # forward-pass-samples per second


def bench_parallel_pool() -> float:
    from src.emulation.parallel_pool import ParallelPool
    pool = ParallelPool(
        rom_path="/dev/null",
        num_workers=4,
        env_spec="src.emulation.fake_environment:FakeNESEnvironment",
    )
    pool.start()
    try:
        pool.reset_all()
        def loop():
            for _ in range(50):
                pool.step_all([0, 0, 0, 0])
        t = _time(loop, warmup=1, trials=3)
    finally:
        pool.shutdown()
    return (50.0 * 4) / t  # per-worker steps/sec


def bench_bc_pretrain_step() -> float:
    from src.models.policy_network import PolicyNetwork
    import torch.nn.functional as F
    net = PolicyNetwork(num_actions=12)
    g = torch.Generator(device="cpu").manual_seed(0)
    # Random states + varied labels — zero-init produces over-optimistic
    # numbers (cross-entropy on identical labels gives a degenerate loss
    # surface that some optimizers short-circuit on).
    states = torch.rand(128, 4, 84, 84, generator=g)
    actions = torch.randint(0, 12, (128,), generator=g, dtype=torch.long)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    def loop():
        for _ in range(20):
            logits = net(states)
            loss = F.cross_entropy(logits, actions)
            optim.zero_grad()
            loss.backward()
            optim.step()
    t = _time(loop, warmup=1, trials=3)
    return 20.0 / t  # steps/sec


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--update-baseline", action="store_true")
    args = p.parse_args()

    results = {}
    print("preprocess_frame ...")
    results["preprocess_ops_per_sec"] = bench_preprocess_frame()

    print("policy_forward_cpu ...")
    results["policy_forward_cpu_samples_per_sec"] = bench_policy_forward(torch.device("cpu"))

    if torch.backends.mps.is_available():
        print("policy_forward_mps ...")
        results["policy_forward_mps_samples_per_sec"] = bench_policy_forward(torch.device("mps"))

    print("parallel_pool_4workers ...")
    results["parallel_pool_steps_per_sec"] = bench_parallel_pool()

    print("bc_pretrain_step ...")
    results["bc_pretrain_steps_per_sec"] = bench_bc_pretrain_step()

    print()
    print("Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.1f}")

    # Compare to baseline
    if args.update_baseline or not BASELINE_PATH.exists():
        BASELINE_PATH.write_text(json.dumps(results, indent=2))
        print(f"\nBaseline written to {BASELINE_PATH.name}")
        return 0

    baseline = json.loads(BASELINE_PATH.read_text())
    regressions = []
    for k, v in results.items():
        base = baseline.get(k)
        if base is None:
            continue
        ratio = v / base
        arrow = "=" if abs(ratio - 1) < 0.02 else ("↑" if ratio > 1 else "↓")
        print(f"  vs baseline {arrow} {k}: {v:.1f} / {base:.1f} = {ratio:.2f}x")
        if ratio < (1 - REGRESSION_TOL):
            regressions.append((k, ratio))

    if regressions:
        print(f"\nREGRESSION: {len(regressions)} metrics regressed >{int(REGRESSION_TOL*100)}%")
        for k, r in regressions:
            print(f"  {k}: {r:.2f}x baseline")
        return 1

    print("\nAll metrics within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
