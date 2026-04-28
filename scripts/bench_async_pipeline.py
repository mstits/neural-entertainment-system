#!/usr/bin/env python3
"""
Prove-out bench for the async pipeline optimization.

Measures two loops that do the same work:

  1. **Sync** — stack → forward → step, serially. Current trainer.
  2. **Async** — step (in a thread) concurrent with forward. Overlap
     the 10.1% MPS forward-pass budget with the 89.3% emulator step.

If the async loop is ≥5% faster, the trainer integration is worth
doing. If the numbers are statistical noise, the GIL / MPS dispatch
synchronization defeats the overlap and we pivot to CPU bulk-stepping
instead.

Usage:
    .venv/bin/python scripts/bench_async_pipeline.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch
import yaml

import nes_core
from src.emulation.frame_utils import FrameStacker
from src.models.policy_network import PolicyNetwork, get_best_device


def _build_setup(workers: int, frame_skip: int, num_actions: int):
    rom = str(_REPO / "roms" / "zelda.nes")
    pool = nes_core.Pool(rom_path=rom, num_workers=workers, frame_skip=frame_skip)
    device = get_best_device()
    net = PolicyNetwork(num_actions=num_actions).to(device).eval()
    stackers = [FrameStacker(stack_size=4) for _ in range(workers)]
    init = pool.reset_all()
    for i, (frame, pp, _ram, _done) in enumerate(init):
        stackers[i].reset(frame, pp)
    # Warm up MPS (first forward compiles the graph).
    stacked = np.zeros((workers, 4, 84, 84), dtype=np.uint8)
    with torch.no_grad():
        obs_t = torch.from_numpy(stacked).to(device).float().div_(255.0)
        _ = net(obs_t).argmax(dim=1).cpu().numpy()
    if device == "mps":
        torch.mps.synchronize()
    return pool, net, stackers, device


def sync_loop(pool, net, stackers, device, steps: int, workers: int) -> float:
    stacked = np.zeros((workers, 4, 84, 84), dtype=np.uint8)
    for i in range(workers):
        stacked[i] = stackers[i]._view_oldest_to_newest()

    t0 = time.perf_counter()
    for _ in range(steps):
        # forward
        with torch.no_grad():
            obs_t = torch.from_numpy(stacked).to(device).float().div_(255.0)
            acts = net(obs_t).argmax(dim=1).cpu().numpy()
        if device == "mps":
            torch.mps.synchronize()

        # step
        results = pool.step_all([int(a) & 0xFF for a in acts])

        # stack
        for i, (frame, pp, _ram, _done) in enumerate(results):
            stacked[i] = stackers[i].push(frame, pp)
    return (time.perf_counter() - t0) * 1000.0


def mps_async_loop(pool, net, stackers, device, steps: int, workers: int) -> float:
    """MPS-native async overlap — no Python threads.

    PyTorch MPS dispatches kernels asynchronously; `net(obs)` returns a
    tensor handle immediately while the Metal kernel runs on the GPU.
    Only `.cpu()` / `.numpy()` / `mps.synchronize()` force a host-side
    wait.

    Strategy: kick off forward(t+1) IMMEDIATELY after we have stacked
    frames, BEFORE calling pool.step_all(t+1). MPS runs the kernel
    while Rust does emulator work (GIL-released). When we come back
    to materialise the acts after step_all returns, most of the
    forward's work is already done.

    This sidesteps the failed `threading.Thread` approach by exploiting
    async dispatch that's already there. No thread overhead.
    """
    stacked = np.zeros((workers, 4, 84, 84), dtype=np.uint8)
    for i in range(workers):
        stacked[i] = stackers[i]._view_oldest_to_newest()

    # Bootstrap forward — this one can't overlap with anything yet.
    with torch.no_grad():
        obs_t = torch.from_numpy(stacked).to(device).float().div_(255.0)
        logits_pending = net(obs_t)

    t0 = time.perf_counter()
    for _ in range(steps):
        # Materialise the prior forward pass's result. This blocks on
        # MPS iff the kernel hasn't finished yet. If the kernel has been
        # running during the prior iteration's step_all, we pay ~0 here.
        acts = logits_pending.argmax(dim=1).cpu().numpy()

        # Step the emulator — Rust releases the GIL, 20 ms.
        results = pool.step_all([int(a) & 0xFF for a in acts])

        # Stack this step's results.
        for i, (frame, pp, _ram, _done) in enumerate(results):
            stacked[i] = stackers[i].push(frame, pp)

        # Queue the NEXT forward pass. This returns immediately; the
        # Metal kernel runs on the GPU while we loop around to the
        # next step_all, which releases the GIL for 20 ms.
        with torch.no_grad():
            obs_t = torch.from_numpy(stacked).to(device).float().div_(255.0)
            logits_pending = net(obs_t)
    # Drain the final pending forward to keep timing honest.
    _ = logits_pending.argmax(dim=1).cpu().numpy()
    if device == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def async_loop(pool, net, stackers, device, steps: int, workers: int) -> float:
    """Overlap forward(t+1) with step_all(t).

    Thread model:
    - Main thread runs forward passes + stack/book-keeping.
    - Worker thread blocks on pool.step_all inside Rust's
      `py.allow_threads` — GIL-released, Python main thread can make
      progress on the next forward pass concurrently.
    """
    stacked = np.zeros((workers, 4, 84, 84), dtype=np.uint8)
    for i in range(workers):
        stacked[i] = stackers[i]._view_oldest_to_newest()

    # Bootstrap: first forward to seed the pipeline (no overlap yet).
    with torch.no_grad():
        obs_t = torch.from_numpy(stacked).to(device).float().div_(255.0)
        next_acts = net(obs_t).argmax(dim=1).cpu().numpy()
    if device == "mps":
        torch.mps.synchronize()

    result_holder: dict = {"results": None}

    def run_step(actions):
        result_holder["results"] = pool.step_all([int(a) & 0xFF for a in actions])

    t0 = time.perf_counter()
    pending = threading.Thread(target=run_step, args=(next_acts,))
    pending.start()

    for step_idx in range(steps):
        # While pool.step_all runs on the worker thread, start the
        # NEXT forward pass on the main thread. We can't actually
        # compute next_next_acts yet because we need the stacked frames
        # from THIS step's results — so the main thread idles on MPS
        # until the worker thread finishes.
        #
        # Optimal overlap means "prior forward" and "current step" run
        # concurrently; the forward for the NEXT step has to wait for
        # this step's results. So the overlap is only worthwhile if
        # forward takes ~same time as the step's "stack + process"
        # overhead, which is small. Measure and see.
        pending.join()
        results = result_holder["results"]

        # stack (needs results)
        for i, (frame, pp, _ram, _done) in enumerate(results):
            stacked[i] = stackers[i].push(frame, pp)

        # Kick off the NEXT forward for step t+1 AND next_next step.
        # The forward kicked off here runs CONCURRENTLY with the step
        # that's about to be spawned below.
        with torch.no_grad():
            obs_t = torch.from_numpy(stacked).to(device).float().div_(255.0)
            future_acts = net(obs_t)

        # Actions for the upcoming pool step are the ones we computed
        # on the previous iteration. Spawn the step now so it runs
        # concurrently with the MPS dispatch we just kicked off.
        if step_idx + 1 < steps:
            acts_for_this_step = next_acts
            pending = threading.Thread(
                target=run_step, args=(acts_for_this_step,)
            )
            pending.start()

        # Wait on the forward pass we started above. This materialises
        # CPU-side numpy values.
        next_acts = future_acts.argmax(dim=1).cpu().numpy()
        if device == "mps":
            torch.mps.synchronize()

    # Drain any straggler step (last iteration spawned nothing).
    return (time.perf_counter() - t0) * 1000.0


def main() -> int:
    workers = 16
    frame_skip = 16
    steps = 200
    num_actions = 6

    pool_s, net_s, stack_s, dev = _build_setup(workers, frame_skip, num_actions)
    print(f"device: {dev}, workers: {workers}, frame_skip: {frame_skip}, steps: {steps}")
    print()

    t_sync = sync_loop(pool_s, net_s, stack_s, dev, steps, workers)
    del pool_s, stack_s

    pool_m, net_m, stack_m, _ = _build_setup(workers, frame_skip, num_actions)
    t_mps = mps_async_loop(pool_m, net_m, stack_m, dev, steps, workers)
    del pool_m, stack_m

    pool_a, net_a, stack_a, _ = _build_setup(workers, frame_skip, num_actions)
    t_async = async_loop(pool_a, net_a, stack_a, dev, steps, workers)

    print(f"{'loop':14}{'ms total':>14}{'ms/step':>12}{'sps':>12}")
    print("-" * 54)
    print(f"{'sync':14}{t_sync:14.1f}{t_sync/steps:12.3f}{workers*steps/(t_sync/1000):12.0f}")
    print(f"{'mps-async':14}{t_mps:14.1f}{t_mps/steps:12.3f}{workers*steps/(t_mps/1000):12.0f}")
    print(f"{'thread-async':14}{t_async:14.1f}{t_async/steps:12.3f}{workers*steps/(t_async/1000):12.0f}")
    print()
    mps_pct = 100.0 * (t_sync - t_mps) / t_sync
    thr_pct = 100.0 * (t_sync - t_async) / t_sync
    print(f"mps-async:    {t_sync / t_mps:.3f}×  ({mps_pct:+.1f}% wall-time)")
    print(f"thread-async: {t_sync / t_async:.3f}×  ({thr_pct:+.1f}% wall-time)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
