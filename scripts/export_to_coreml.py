"""
CLI tool: load a gen_*.pt checkpoint, pick the best genome, export to
Core ML at checkpoints/best_policy.mlpackage. Run from the repo root.

Usage:
    python scripts/export_to_coreml.py checkpoints/gen_00050.pt
    # or
    python scripts/export_to_coreml.py  # auto-picks latest
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

from src.models.coreml_export import maybe_export
from src.models.policy_network import PolicyNetwork
from src.training.trainer import find_latest_checkpoint
from src.utils.logging_config import configure


def main() -> int:
    parser = argparse.ArgumentParser(description="Export best genome to Core ML")
    parser.add_argument("checkpoint", nargs="?", default=None, help="gen_*.pt path")
    parser.add_argument("--profile", default="configs/zelda.yaml")
    parser.add_argument("--output", default="checkpoints/best_policy.mlpackage")
    args = parser.parse_args()

    configure(verbose=False)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        latest = find_latest_checkpoint("checkpoints")
        if latest is None:
            print("no checkpoints found in ./checkpoints", file=sys.stderr)
            return 1
        ckpt_path = str(latest)

    with open(args.profile) as f:
        profile = yaml.safe_load(f)
    num_actions = len(profile["action_space"])

    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    genomes = sorted(
        data["population"],
        key=lambda g: g.get("fitness", float("-inf")),
        reverse=True,
    )
    best = genomes[0]
    fit = best.get("fitness", float("-inf"))
    fit_str = f"{fit:.2f}" if fit != float("-inf") else "unevaluated"
    print(f"Best genome id={best.get('genome_id', '?')} fitness={fit_str}")

    net = PolicyNetwork(num_actions=num_actions)
    net.load_state_dict(best["state_dict"])

    result = maybe_export(net, args.output, num_actions=num_actions)
    if result is None:
        print("Core ML export failed — see warnings above", file=sys.stderr)
        return 1
    print(f"Wrote {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
