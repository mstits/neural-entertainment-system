"""Probe the NETWORK ITSELF across checkpoints of one run: action-distribution
entropy on a fixed real-observation batch, per-layer weight norms, weight
deltas between consecutive checkpoints, action-usage histogram, and logit
saturation. Loads checkpoints through the same dispatch point the eval
harness uses (build_tile_policy_from_checkpoint) rather than hand-parsing
state_dicts.

Usage:
    .venv/bin/python probe_checkpoints.py <checkpoint_dir> <out_json>

<checkpoint_dir> must contain vanilla_ppo_iter_NNNNN.pt files and (optionally)
winners/best.json (its source_iter is marked "peak" in the output).
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
FEATURE_DIM = 712  # 178 (smb_tiles_pos) * stack 4, verbatim across every v27/v28 seed


def softmax_entropy(logits: torch.Tensor) -> np.ndarray:
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    ent = -(p * logp).sum(dim=-1)
    return ent.numpy()


def layer_norms(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        out[k] = float(v.float().norm().item())
    return out


def flat_vector(sd: dict) -> torch.Tensor:
    return torch.cat([v.float().reshape(-1) for k, v in sorted(sd.items())])


def probe_run(ckpt_dir: Path, batch: np.ndarray) -> dict:
    iters = sorted(
        int(p.stem.rsplit("_", 1)[1])
        for p in ckpt_dir.glob("vanilla_ppo_iter_*.pt")
    )
    peak_iter = None
    best_json = ckpt_dir / "winners" / "best.json"
    if best_json.exists():
        peak_iter = json.loads(best_json.read_text()).get("source_iter")

    x = torch.from_numpy(batch.astype(np.float32))
    rows = []
    prev_sd = None
    prev_flat = None
    for it in iters:
        ckpt_path = ckpt_dir / f"vanilla_ppo_iter_{it:05d}.pt"
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        net, is_recurrent = build_tile_policy_from_checkpoint(
            ckpt, num_actions=NUM_ACTIONS, feature_dim=FEATURE_DIM,
        )
        assert not is_recurrent, f"{ckpt_path} unexpectedly recurrent"
        net.eval()
        sd = {k: v.detach().clone() for k, v in net.state_dict().items()}

        with torch.no_grad():
            logits, value = net.forward_ac(x)

        ent = softmax_entropy(logits)
        actions = logits.argmax(dim=-1).numpy()
        counts = np.bincount(actions, minlength=NUM_ACTIONS)
        fracs = (counts / counts.sum()).tolist()
        n_actions_used = int((counts > 0).sum())

        sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)
        margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).numpy()

        flat = flat_vector(sd)
        delta_l2 = None
        delta_cos = None
        delta_relative = None
        if prev_flat is not None:
            diff = flat - prev_flat
            delta_l2 = float(diff.norm().item())
            denom = float(flat.norm().item() * prev_flat.norm().item())
            delta_cos = float(
                torch.dot(flat, prev_flat).item() / denom
            ) if denom > 0 else None
            pn = float(prev_flat.norm().item())
            delta_relative = delta_l2 / pn if pn > 0 else None

        row = {
            "iter": it,
            "is_peak": bool(peak_iter is not None and it == peak_iter),
            "entropy_mean": float(ent.mean()),
            "entropy_std": float(ent.std()),
            "entropy_min": float(ent.min()),
            "entropy_max": float(ent.max()),
            "action_fracs": fracs,
            "n_actions_used": n_actions_used,
            "max_action_frac": float(max(fracs)),
            "logit_abs_mean": float(logits.abs().mean().item()),
            "logit_abs_max": float(logits.abs().max().item()),
            "margin_mean": float(margin.mean()),
            "margin_frac_gt5": float((margin > 5.0).mean()),
            "value_mean": float(value.mean().item()),
            "value_std": float(value.std().item()),
            "weight_norms": layer_norms(sd),
            "weight_total_norm": float(flat.norm().item()),
            "weight_delta_l2_from_prev": delta_l2,
            "weight_delta_cos_from_prev": delta_cos,
            "weight_delta_relative_from_prev": delta_relative,
        }
        rows.append(row)
        prev_sd = sd
        prev_flat = flat

    return {
        "checkpoint_dir": str(ckpt_dir),
        "peak_iter": peak_iter,
        "num_actions": NUM_ACTIONS,
        "feature_dim": FEATURE_DIM,
        "batch_size": int(batch.shape[0]),
        "rows": rows,
    }


if __name__ == "__main__":
    ckpt_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    batch = np.load(Path(__file__).resolve().parent / "fixed_batch.npy")
    result = probe_run(ckpt_dir, batch)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path} ({len(result['rows'])} checkpoints)")
