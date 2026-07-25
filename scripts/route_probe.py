"""Supervised route-byte probe for silently-looping SMB castle mazes.

Implements the maze consultation's primary recipe (2026-07-24): discover the
hidden route-progress RAM byte from the system's OWN rollouts and validate it
causally — no disassembly, no maps. Protocol:

  1. LABELED COLLECTION: run right-biased rollouts from the maze root; every
     time a lineage crosses the pre-trigger checkpoint X (default gx 240,
     before the first observed loop trigger ~269), snapshot the full RAM and
     the emulator state blob. Label the snapshot by what happens next:
     loop-back (gx collapse) => Y=0, sustained traversal (gx >= X+80) => Y=1.
  2. SPARSE PROBE: one-hot encode the VARYING bytes (addr, value) and fit an
     L1-regularized logistic regression (torch); sweep lambda; keep the
     smallest feature set with perfect holdout accuracy.
  3. CAUSAL MUTATION TEST: take looping snapshots, overwrite ONLY the
     identified byte in the state blob (RAM sits verbatim at blob offset 13)
     with the right-route value, resume with rightward inputs — traversal
     without a loop proves the byte is upstream cause, not correlate.
  4. Output the validated (address, right-value) for the solver's cell key.

Methodological basis: State Representation Learning / active black-box
system identification (Atari-ARI, U-Tree, PSRs) — statistical inference over
the agent's own observations, explicitly not disassembly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nes_core import Pool  # noqa: E402
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402

RAM_BLOB_OFFSET = 13  # the 2KB RAM sits verbatim here in save_worker_state blobs


def _gx(ram) -> int:
    return (int(ram[0x6D]) << 8) | int(ram[0x86])


def collect(args, profile, rng):
    bm = action_space_to_bitmasks(profile["action_space"])
    root = Path(args.root_state).read_bytes()
    W = args.workers
    pool = Pool(rom_path=profile["rom_path"], num_workers=W,
                frame_skip=int(profile.get("frame_skip", 4)))
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    pool.reset_all()
    for i in range(W):
        pool.load_worker_state(i, root)
    r = pool.step_all(np.zeros(W, dtype=np.uint8))

    X = args.checkpoint_x
    XPASS = X + 80
    prev_gx = [0] * W
    pending: list = [None] * W   # (ram_bytes, blob) awaiting a label
    prev_a = [1] * W
    pos, neg = [], []            # lists of (ram_bytes, blob)
    steps = 0
    while (len(pos) < args.per_class or len(neg) < args.per_class) \
            and steps < args.max_steps:
        acts = np.zeros(W, dtype=np.uint8)
        for i in range(W):
            u = rng.random()
            a = 3 if u < 0.5 else (4 if u < 0.7 else (
                prev_a[i] if u < 0.85 else int(rng.integers(len(bm)))))
            prev_a[i] = a
            acts[i] = bm[a]
        r = pool.step_all(acts)
        steps += W
        for i in range(W):
            ram = bytes(r[i][2])
            gx = _gx(ram)
            done = bool(r[i][3])
            if gx < 3900:
                if prev_gx[i] - gx > 100:            # loop-back
                    if pending[i] is not None:
                        neg.append(pending[i])
                        pending[i] = None
                elif prev_gx[i] < X <= gx and pending[i] is None:
                    pending[i] = (ram, pool.save_worker_state(i))
                elif gx >= XPASS and pending[i] is not None:
                    pos.append(pending[i])
                    pending[i] = None
                prev_gx[i] = gx
            if done:                                  # death: discard pending
                pending[i] = None
                pool.load_worker_state(i, root)
                prev_gx[i] = 0
        if steps % 200000 < W:
            print(f"[collect] steps={steps} Y1={len(pos)} Y0={len(neg)}",
                  flush=True)
    pool.shutdown()
    print(f"[collect] DONE: steps={steps} Y1={len(pos)} Y0={len(neg)}")
    return pos, neg


def fit_probe(pos, neg, seed=0):
    """L1 logistic over one-hot (addr, value) features of VARYING bytes."""
    R = np.frombuffer(b"".join(p[0] for p in pos + neg),
                      dtype=np.uint8).reshape(len(pos) + len(neg), 2048)
    y = np.array([1.0] * len(pos) + [0.0] * len(neg), dtype=np.float32)
    varying = np.nonzero(R.std(axis=0) > 0)[0]
    print(f"[probe] {len(varying)} varying bytes of 2048")
    # one-hot only the observed values per varying byte
    feats, fmap = [], []
    for ad in varying:
        for v in np.unique(R[:, ad]):
            feats.append((R[:, ad] == v).astype(np.float32))
            fmap.append((int(ad), int(v)))
    Xf = torch.tensor(np.stack(feats, axis=1))
    yt = torch.tensor(y)
    n = len(yt)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    tr, ho = perm[: int(n * 0.8)], perm[int(n * 0.8):]
    results = []
    for lam in (0.3, 0.1, 0.03, 0.01):
        w = torch.zeros(Xf.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.05)
        for _ in range(400):
            z = Xf[tr] @ w + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                z, yt[tr]) + lam * w.abs().sum()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = (((Xf[ho] @ w + b) > 0).float() == yt[ho]).float().mean().item()
            nz = torch.nonzero(w.abs() > 0.05).flatten().tolist()
        results.append((lam, acc, nz))
        print(f"[probe] lambda={lam}: holdout acc {acc*100:.1f}%, "
              f"{len(nz)} active features: "
              f"{[(hex(fmap[j][0]), fmap[j][1], round(float(w[j]),2)) for j in nz[:6]]}")
    # pick: perfect (or best) accuracy with fewest features
    results.sort(key=lambda t: (-t[1], len(t[2])))
    lam, acc, nz = results[0]
    cands = sorted({fmap[j][0] for j in nz})
    print(f"[probe] SELECTED lambda={lam} acc={acc*100:.1f}% "
          f"candidate addrs: {[hex(a) for a in cands]}")
    return cands, R, y


def mutation_test(args, profile, cands, R, y, pos, neg):
    """Overwrite ONLY the candidate byte in looping blobs; does it traverse?"""
    bm = action_space_to_bitmasks(profile["action_space"])
    pool = Pool(rom_path=profile["rom_path"], num_workers=1,
                frame_skip=int(profile.get("frame_skip", 4)))
    pool.set_headless(True)
    X = args.checkpoint_x
    validated = []
    for ad in cands:
        # right-route value = majority value among Y=1 at this addr
        vals, counts = np.unique(R[y == 1.0][:, ad], return_counts=True)
        v1 = int(vals[np.argmax(counts)])
        okay = 0
        trials = min(3, len(neg))
        for k in range(trials):
            blob = bytearray(neg[k][1])
            blob[RAM_BLOB_OFFSET + ad] = v1
            pool.reset_all()
            pool.load_worker_state(0, bytes(blob))
            r = pool.step_all(np.zeros(1, dtype=np.uint8))
            prev, looped, reached = _gx(bytes(r[0][2])), False, False
            for _ in range(300):
                r = pool.step_all(np.array([bm[3]], dtype=np.uint8))
                ram = bytes(r[0][2]); gx = _gx(ram)
                if gx < 3900:
                    if prev - gx > 100:
                        looped = True; break
                    if gx >= X + 80:
                        reached = True; break
                    prev = gx
                if bool(r[0][3]):
                    break
            okay += int(reached and not looped)
        print(f"[mutate] ${ad:04X} <- {v1}: {okay}/{trials} traversals")
        if okay == trials and trials > 0:
            validated.append((ad, v1))
    pool.shutdown()
    return validated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-state", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--checkpoint-x", type=int, default=240)
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--max-steps", type=int, default=1_500_000)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="JSON result path")
    args = ap.parse_args()
    profile = yaml.safe_load(Path(args.profile).read_text())
    rng = np.random.default_rng(args.seed)
    pos, neg = collect(args, profile, rng)
    if len(pos) < 20 or len(neg) < 20:
        print(f"[probe] INSUFFICIENT classes (Y1={len(pos)} Y0={len(neg)}) — "
              "loosen exploration or move --checkpoint-x")
        return 1
    cands, R, y = fit_probe(pos, neg, args.seed)
    validated = mutation_test(args, profile, cands, R, y, pos, neg)
    print(f"[probe] CAUSALLY VALIDATED route bytes: "
          f"{[(hex(a), v) for a, v in validated]}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"checkpoint_x": args.checkpoint_x,
             "candidates": [hex(a) for a in cands],
             "validated": [[a, v] for a, v in validated]}))
    return 0 if validated else 2


if __name__ == "__main__":
    sys.exit(main())
