"""Convert the banked 6-action 1-2 policy into the 18-way options seed.

Both Phase-3 arms started from byte-identical weights; the options
experiment holds itself to the same standard. This takes the exact
checkpoint the control arm was seeded from and re-expresses it as a
CommitmentPolicy whose pair softmax MARGINALIZES to the same primitive
distribution (tested property of from_flat_policy), so at step zero the
treatment behaves like the control up to uniform duration choice.

    .venv/bin/python scripts/make_options_seed.py \\
        --src checkpoints/_preserved/consol2_40pct_strict_iter01120.pt \\
        --out checkpoints/mario_1_2_options/vanilla_ppo_iter_00000.pt
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-actions", type=int, default=6)
    ap.add_argument("--durations", default="1,2,4")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(REPO))
    import torch
    from src.models.tile_policy import build_tile_policy_from_checkpoint
    from src.training.commitment_policy import CommitmentPolicy

    blob = torch.load(REPO / args.src, map_location="cpu",
                      weights_only=False)
    sd = blob["net_state_dict"]
    flat, _ = build_tile_policy_from_checkpoint(
        blob, num_actions=args.num_actions,
        feature_dim=sd["fc1.weight"].shape[1])
    res = flat.load_state_dict(sd, strict=False)
    if res.missing_keys:
        raise SystemExit(f"seed missing core weights: {res.missing_keys}")

    durations = tuple(int(x) for x in args.durations.split(","))
    cp = CommitmentPolicy.from_flat_policy(
        flat, trunk_dim=flat.fc2.out_features,
        num_primitives=args.num_actions, durations=durations)

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net_state_dict": cp.state_dict(),
                "iter": int(blob.get("iter", 0)),
                "provenance": {
                    "converted_from": args.src,
                    "src_sha256": hashlib.sha256(
                        (REPO / args.src).read_bytes()).hexdigest(),
                    "durations": list(durations),
                    "marginalization": "pair softmax sums to the source "
                                       "primitive distribution exactly "
                                       "(tested in test_commitment_policy)",
                }}, out)
    print(f"wrote {out} (iter {int(blob.get('iter', 0))}, "
          f"pairs {cp.num_pairs})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
