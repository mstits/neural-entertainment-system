"""Does the argmax DECISION per fixed real state change between the peak
checkpoint and iter 240, or does the network just get more confident about
the SAME decisions? Directly separates "policy re-decides differently"
(a real behavioral change / possible overfitting/drift) from "policy just
saturates its existing decisions" (a pure confidence/logit-magnitude effect
that would leave honest-eval unaffected on these states).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.models.tile_policy import build_tile_policy_from_checkpoint  # noqa: E402

NUM_ACTIONS = 6
FEATURE_DIM = 712
DIR = Path(__file__).resolve().parent

RUNS = {
    "v27_seed0": REPO / "checkpoints/mario_1_1_v27_recovery_seed0",
    "v27_seed1": REPO / "checkpoints/mario_1_1_v27_recovery_seed1",
    "v27_seed2": REPO / "checkpoints/mario_1_1_v27_recovery_seed2",
    "v27_seed3": REPO / "checkpoints/mario_1_1_v27_recovery_seed3",
    "v28_seed0": REPO / "checkpoints/mario_1_1_v28_capacity_seed0",
    "v28_seed1": REPO / "checkpoints/mario_1_1_v28_capacity_seed1",
    "v28_seed2": REPO / "checkpoints/mario_1_1_v28_capacity_seed2",
    "v28_seed3": REPO / "checkpoints/mario_1_1_v28_capacity_seed3",
}


def load_net(ckpt_dir: Path, it: int):
    p = ckpt_dir / f"vanilla_ppo_iter_{it:05d}.pt"
    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    net, is_rec = build_tile_policy_from_checkpoint(
        ckpt, num_actions=NUM_ACTIONS, feature_dim=FEATURE_DIM,
    )
    net.eval()
    return net


def main():
    batch = np.load(DIR / "fixed_batch.npy")
    x = torch.from_numpy(batch.astype(np.float32))

    print(f"{'run':10s} {'peak':>5s} {'agree_peak_vs_240':>18s} "
          f"{'agree_peak_vs_peak+10':>22s} {'agree_10_vs_20':>15s}")
    for name, ckpt_dir in RUNS.items():
        best_json = json.loads((ckpt_dir / "winners" / "best.json").read_text())
        peak = best_json["source_iter"]

        net_peak = load_net(ckpt_dir, peak)
        net_240 = load_net(ckpt_dir, 240)
        # one step past peak, if it exists (peak+10 always <=240 here since
        # peaks are 50-120 and checkpoints run every 10 to 240)
        peak_plus_10 = peak + 10
        net_peak10 = load_net(ckpt_dir, peak_plus_10) if (ckpt_dir / f"vanilla_ppo_iter_{peak_plus_10:05d}.pt").exists() else None
        net_10 = load_net(ckpt_dir, 10)
        net_20 = load_net(ckpt_dir, 20)

        with torch.no_grad():
            a_peak = net_peak.forward_ac(x)[0].argmax(dim=-1).numpy()
            a_240 = net_240.forward_ac(x)[0].argmax(dim=-1).numpy()
            a_10 = net_10.forward_ac(x)[0].argmax(dim=-1).numpy()
            a_20 = net_20.forward_ac(x)[0].argmax(dim=-1).numpy()
            agree_peak10 = None
            if net_peak10 is not None:
                a_peak10 = net_peak10.forward_ac(x)[0].argmax(dim=-1).numpy()
                agree_peak10 = float((a_peak == a_peak10).mean())

        agree_240 = float((a_peak == a_240).mean())
        agree_10_20 = float((a_10 == a_20).mean())

        print(f"{name:10s} {peak:5d} {agree_240:18.3f} "
              f"{('%.3f' % agree_peak10) if agree_peak10 is not None else 'NA':>22s} "
              f"{agree_10_20:15.3f}")

        # confusion: which action does the peak's most-common-swap target
        # become at 240, restricted to disagreement rows
        disagree = a_peak != a_240
        if disagree.sum() > 0:
            from collections import Counter
            pairs = Counter(zip(a_peak[disagree].tolist(), a_240[disagree].tolist()))
            top = pairs.most_common(3)
            print(f"           top disagreement (peak_action->iter240_action, count): {top}")


if __name__ == "__main__":
    main()
