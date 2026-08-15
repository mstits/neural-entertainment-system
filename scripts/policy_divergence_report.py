"""Policy divergence / near-tie report over saved observation traces.

Pure forward passes on saved states (CPU) — no emulator, no rollouts.

Given one checkpoint and an npz of tile observations (e.g. the SMODICE-era
712-dim transition states in runs/smodice_1_2/transitions.npz), reports where
the greedy action is fragile: per-state policy entropy, action gap
(top1 - top2 logit margin), the probability mass that sampling disagrees with
argmax, and counts of near-ties under configurable gap thresholds.

Given a second checkpoint, additionally reports where the two policies
diverge most: argmax disagreement rate, mean KL(A||B), and the top-K most
divergent states.

Every report embeds obs provenance (npz meta_json, is_expert_window
fraction, presence of per-state position data) plus a distribution_scope
caveat: the statistics describe the policy ONLY on the supplied states.
Absent position data, nothing here supports a claim about a specific
x-corridor visited under some other rollout policy.

Usage:
    python scripts/policy_divergence_report.py \
        --checkpoint checkpoints/_preserved/online_v2_FINAL_consolidated.pt \
        --obs runs/smodice_1_2/transitions.npz --obs-key state \
        --out /tmp/report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.models.tile_policy import TilePolicyNetwork  # noqa: E402

DEFAULT_THRESHOLDS = (0.1, 0.25, 0.5)


# ------------------------------------------------------------- loading

def infer_dims(state_dict: dict) -> dict:
    """Read network dimensions straight off the weight shapes."""
    try:
        hidden_dim, feature_dim = state_dict["fc1.weight"].shape
        trunk_dim = state_dict["fc2.weight"].shape[0]
        num_actions = state_dict["actor.weight"].shape[0]
    except KeyError as e:
        raise ValueError(f"state dict is missing expected key: {e}") from e
    return {"feature_dim": int(feature_dim), "hidden_dim": int(hidden_dim),
            "trunk_dim": int(trunk_dim), "num_actions": int(num_actions)}


def load_policy_from_checkpoint(path: str | Path) -> tuple[TilePolicyNetwork, dict]:
    """Load a TilePolicyNetwork from any of the checkpoint formats in use.

    Accepts: trainer checkpoints ({'net_state_dict': ...}), the network's
    native save format ({'state_dict': ...}), or a bare state dict.
    Returns (net.eval(), meta) where meta carries 'iter' when present.
    """
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError(f"{path}: unsupported checkpoint object {type(blob)}")
    if "net_state_dict" in blob:
        sd = blob["net_state_dict"]
    elif "state_dict" in blob:
        sd = blob["state_dict"]
    elif "fc1.weight" in blob:
        sd = blob
    else:
        raise ValueError(
            f"{path}: no net_state_dict/state_dict/raw weights found "
            f"(keys: {sorted(blob.keys())[:10]})")
    dims = infer_dims(sd)
    net = TilePolicyNetwork(**dims)
    net.load_state_dict(sd)
    net.eval()
    meta = {"path": str(path), **dims}
    if "iter" in blob:
        meta["iter"] = int(blob["iter"])
    return net, meta


# ------------------------------------------------------------- metrics

def metrics_from_logits(logits: torch.Tensor) -> dict:
    """Per-state fragility metrics from raw actor logits (B, A)."""
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    entropy = -(p * logp).sum(-1)
    top2 = torch.topk(logits, k=2, dim=-1)
    action_gap = top2.values[:, 0] - top2.values[:, 1]
    p_top1 = p.max(dim=-1).values
    return {
        "argmax": logits.argmax(dim=-1).cpu().numpy(),
        "action_gap": action_gap.cpu().numpy(),
        "entropy": entropy.cpu().numpy(),
        "p_top1": p_top1.cpu().numpy(),
        "p_disagree": (1.0 - p_top1).cpu().numpy(),
    }


def _forward_logits(net: TilePolicyNetwork, obs: np.ndarray,
                    batch_size: int = 8192) -> torch.Tensor:
    outs = []
    with torch.no_grad():
        for i in range(0, len(obs), batch_size):
            x = torch.as_tensor(obs[i:i + batch_size],
                                dtype=torch.float32)
            outs.append(net.forward_ac(x)[0])
    return torch.cat(outs, dim=0)


def compute_state_metrics(net: TilePolicyNetwork, obs: np.ndarray,
                          batch_size: int = 8192) -> dict:
    return metrics_from_logits(_forward_logits(net, obs, batch_size))


def _stats(a: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(a)),
        "p10": float(np.percentile(a, 10)),
        "p50": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def near_tie_summary(metrics: dict,
                     thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS) -> dict:
    gaps = metrics["action_gap"]
    n = int(len(gaps))
    counts = {str(t): int((gaps < t).sum()) for t in thresholds}
    return {
        "n_states": n,
        "near_tie_counts": counts,
        "near_tie_frac": {k: v / n for k, v in counts.items()},
        "action_gap": _stats(gaps),
        "entropy": _stats(metrics["entropy"]),
        "p_top1": _stats(metrics["p_top1"]),
        "p_disagree": _stats(metrics["p_disagree"]),
    }


# ------------------------------------------------------------- comparison

def compare_policies(net_a: TilePolicyNetwork, net_b: TilePolicyNetwork,
                     obs: np.ndarray, top_k: int = 20,
                     batch_size: int = 8192) -> dict:
    la = _forward_logits(net_a, obs, batch_size)
    lb = _forward_logits(net_b, obs, batch_size)
    lpa, lpb = F.log_softmax(la, dim=-1), F.log_softmax(lb, dim=-1)
    kl = (lpa.exp() * (lpa - lpb)).sum(-1).cpu().numpy()
    am_a = la.argmax(dim=-1).cpu().numpy()
    am_b = lb.argmax(dim=-1).cpu().numpy()
    disagree = am_a != am_b
    order = np.argsort(-kl)[:top_k]
    top = [{"index": int(i), "kl": float(kl[i]),
            "argmax_a": int(am_a[i]), "argmax_b": int(am_b[i])}
           for i in order]
    return {
        "argmax_disagree_frac": float(disagree.mean()),
        "mean_kl_ab": float(kl.mean()),
        "p90_kl_ab": float(np.percentile(kl, 90)),
        "top_divergent_states": top,
    }


# ------------------------------------------------------------- provenance

POSITION_KEYS = ("x", "gx", "x_pos", "pos_x", "mario_x", "max_gx",
                 "max_gx_per_episode")

_SCOPE_WITH_POSITION = (
    "Metrics characterize the policy only on the states supplied in this "
    "file. Per-state position data is present, so results CAN be sliced by "
    "x; conclusions still do not extend to states outside this distribution.")

_SCOPE_NO_POSITION = (
    "Metrics characterize the policy only on the states supplied in this "
    "file. No per-state position data is present, so results cannot be "
    "sliced by x and support no claim about states outside this "
    "distribution (e.g. a specific x-corridor reached under a different "
    "rollout policy).")


def obs_provenance(path: str | Path) -> dict:
    """Describe where an obs npz came from, so every report is explicit
    about the state distribution its statistics were computed on.

    Reads (when present): 'meta_json' (parsed), 'is_expert_window'
    (fraction reported), and any per-state position key from
    POSITION_KEYS. Always emits a 'distribution_scope' caveat string.
    """
    prov: dict = {"meta_json": None, "is_expert_window_frac": None,
                  "has_position_data": False, "position_keys": []}
    z = np.load(str(path), allow_pickle=False)
    if isinstance(z, np.ndarray):  # bare .npy — no side metadata possible
        prov["npz_keys"] = []
        prov["distribution_scope"] = _SCOPE_NO_POSITION
        return prov
    keys = list(z.files)
    prov["npz_keys"] = keys
    if "meta_json" in keys:
        raw = z["meta_json"]
        raw = raw.item() if raw.shape == () else raw[0]
        try:
            prov["meta_json"] = json.loads(str(raw))
        except (json.JSONDecodeError, TypeError):
            prov["meta_json"] = str(raw)
    if "is_expert_window" in keys:
        prov["is_expert_window_frac"] = float(
            np.asarray(z["is_expert_window"]).mean())
    prov["position_keys"] = [k for k in keys if k in POSITION_KEYS]
    prov["has_position_data"] = bool(prov["position_keys"])
    prov["distribution_scope"] = (
        _SCOPE_WITH_POSITION if prov["has_position_data"]
        else _SCOPE_NO_POSITION)
    return prov


# ------------------------------------------------------------- CLI

def _load_obs(path: Path, key: Optional[str]) -> np.ndarray:
    z = np.load(str(path))
    if isinstance(z, np.ndarray):
        return z
    keys = list(z.files)
    if key is None:
        key = "state" if "state" in keys else keys[0]
    if key not in keys:
        raise ValueError(f"{path}: key '{key}' not in npz (has {keys})")
    return z[key]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--checkpoint-b", default=None,
                    help="optional second checkpoint for A-vs-B divergence")
    ap.add_argument("--obs", required=True, help="npz (or npy) of observations")
    ap.add_argument("--obs-key", default=None,
                    help="npz key holding the (N, feature_dim) states")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=list(DEFAULT_THRESHOLDS))
    args = ap.parse_args(argv)

    net_a, meta_a = load_policy_from_checkpoint(args.checkpoint)
    obs = _load_obs(Path(args.obs), args.obs_key)
    if obs.ndim != 2 or obs.shape[1] != net_a.feature_dim:
        raise ValueError(
            f"obs shape {obs.shape} does not match policy feature_dim "
            f"{net_a.feature_dim}")

    metrics = compute_state_metrics(net_a, obs, batch_size=args.batch_size)
    report = {
        "checkpoint": meta_a,
        "obs": {"path": str(args.obs), "n_states": int(len(obs)),
                "feature_dim": int(obs.shape[1]),
                "provenance": obs_provenance(Path(args.obs))},
        "n_states": int(len(obs)),
        **near_tie_summary(metrics, thresholds=tuple(args.thresholds)),
    }
    # Worst near-ties: smallest action gaps, ascending.
    order = np.argsort(metrics["action_gap"])[:args.top_k]
    report["worst_near_ties"] = [{
        "index": int(i),
        "action_gap": float(metrics["action_gap"][i]),
        "entropy": float(metrics["entropy"][i]),
        "p_top1": float(metrics["p_top1"][i]),
        "argmax": int(metrics["argmax"][i]),
    } for i in order]

    if args.checkpoint_b:
        net_b, meta_b = load_policy_from_checkpoint(args.checkpoint_b)
        report["checkpoint_b"] = meta_b
        report["comparison"] = compare_policies(
            net_a, net_b, obs, top_k=args.top_k, batch_size=args.batch_size)

    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
