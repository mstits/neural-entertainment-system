"""Shared-substrate offline BC trainer — one trunk, per-level heads.

THE QUESTION (docs/proposals/RESEARCH_SYNTHESIS_2026-08-17.md, v20):
naive pooling of 1-1 and 1-2 successes into one undifferentiated net was
already falsified in this repo (scripts/interference_falsifier.py):
1-1 was CAPTURED (52/100 strict, beating its own 43/100 specialist) while
1-2 was DESTROYED (4/100 vs 42/100). Trunk-plus-heads — a shared
representation with isolated per-level output heads — is the untested
form, ranked the #1 intervention of five by the same consultation round.
This script is the offline behavioural-cloning half of that experiment:
it consumes ALREADY-COLLECTED per-level success-trajectory npz files
(e.g. the ones scripts/interference_falsifier.py's collection step
writes) and trains one shared-trunk / multi-head net on the pool. It
never touches the emulator — no rollouts, no evals, no solver runs.

Pipeline:
  1. Data — load one npz per level (`--data LEVEL=PATH`, repeatable),
     balance them PAIR-WISE across levels (every level trimmed to the
     smallest level's pair count, leading pairs kept in collection
     order — the same reasoning interference_falsifier.balance_fifty_fifty
     documents: episode-level balance would let a longer level's
     trajectories dominate the pooled gradient). Counts are reported
     before and after balancing.
  2. Training — joint BC over the pooled, balanced set. Every sample
     routes through the shared trunk into ONLY its own level's head
     (MultiHeadTilePolicy.forward_ac(x, level_ids) does the per-sample
     routing), so the trunk's gradient sees every level every step while
     each head's gradient sees only its own. Per-level BC accuracy is
     reported every epoch, split into a fitting-set number
     (`per_level_train_acc`, measured on the exact pairs that produced
     the gradient) and, when `--val-frac` holds pairs out per level
     before training starts, a `per_level_val_acc` measured on pairs no
     gradient step ever saw. NEITHER number is the interference
     verdict. A shared trunk can drive both toward ~1.0 for every level
     — this is offline BC on a fixed, already-successful pair
     distribution, not a stochastic sticky/jitter rollout — while the
     resulting policy still collapses under the honest eval gate; this
     repo has not shown BC-accuracy divergence between levels to
     correlate with, let alone lead, that collapse. Per-level accuracy
     here is a cheap early-warning-only diagnostic (it can flag gross
     divergence worth investigating sooner) and must never substitute
     for the honest sticky-clear-rate verdict scripts/eval_shared_
     substrate.py produces against the pre-registered baseline sum —
     the same proxy-metric discipline CLAIMS.md already applies to
     3-episode gates and seq_clear vs strict clear_rate, one layer
     deeper.
  3. Export — the multi-head checkpoint, plus one single-head checkpoint
     per level via MultiHeadTilePolicy.export_head(level), written in
     the same net_state_dict layout eval_game.py shape-infers
     (fc1/norm1/fc2/norm2/actor/critic — see
     src/models/tile_policy.build_tile_policy_from_checkpoint), so the
     existing honest-eval harness can load them with `--checkpoint
     <path>` unchanged.
  4. --dry-run validates every data file, prints the balance plan and
     the export plan, and constructs the model — no training, no writes.

DEPENDENCY: `MultiHeadTilePolicy` (src/training/multihead_policy.py) was
being written concurrently by another lane. At the start of writing this
script it was NOT YET on disk; this script was drafted against the
interface below, and the module has since landed matching it exactly
(confirmed by reading it before finishing this file), so the import
below is real and unconditional at the call site, not a stub:

    class MultiHeadTilePolicy(nn.Module):
        def __init__(self, num_actions: int, num_levels: int,
                     feature_dim: int = 175, hidden_dim: int = 256,
                     trunk_dim: int = 64,
                     levels: Sequence[str] | None = None): ...
        def forward_ac(self, x: Tensor, level_ids
                        ) -> tuple[Tensor, Tensor]:
            # per-sample routed logits (B, num_actions) + value (B,);
            # gather-based routing gives EXACT (not approximate) zero
            # gradient at every head a batch doesn't touch.
        def export_head(self, level) -> dict[str, Tensor]:
            # TilePolicyNetwork-layout state dict (TRUNK_KEYS + HEAD_KEYS
            # from that same module) — exactly what
            # build_tile_policy_from_checkpoint shape-infers.
        def init_trunk_from(self, checkpoint) -> None: ...   # warm start
        def init_head_from(self, level, checkpoint) -> None: ...

  The import guard below (`MultiHeadTilePolicy is None`) and the
  fallback TRUNK_KEYS/HEAD_KEYS copies exist only so this script's own
  data/CLI/balancing logic stays importable and unit-testable even if a
  future edit to that module drops the import again — every call site
  that actually needs the network raises a clear, actionable error via
  `_require_model_class()` rather than failing at import time.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_IMPORT_ERROR: Optional[BaseException] = None
try:
    from src.training.multihead_policy import (  # noqa: E402
        HEAD_KEYS as _HEAD_KEYS,
        MultiHeadTilePolicy,
        TRUNK_KEYS as _TRUNK_KEYS,
    )
except ImportError as exc:  # module not on disk yet / signature drifted
    MultiHeadTilePolicy = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
    # Fallback copies (see the docstring's DEPENDENCY NOTE) so the
    # export-shape guard below still has something to check against even
    # before the dependency lands.
    _TRUNK_KEYS = (
        "fc1.weight", "fc1.bias", "norm1.weight", "norm1.bias",
        "fc2.weight", "fc2.bias", "norm2.weight", "norm2.bias",
    )
    _HEAD_KEYS = ("actor.weight", "actor.bias", "critic.weight", "critic.bias")

_STANDARD_HEAD_KEYS = tuple(_TRUNK_KEYS) + tuple(_HEAD_KEYS)


# ---------------------------------------------------------------------------
# CLI arg parsing helpers
# ---------------------------------------------------------------------------


def _parse_level_path_pairs(items: list[str], flag_name: str,
                             *, allow_empty: bool = False) -> dict[str, Path]:
    """Parse repeated `LEVEL=PATH` CLI args into an order-preserving dict.

    Shared by --data and --warm-heads. Raises ValueError on a missing
    '=', an empty level or path, or a level repeated within the same
    flag."""
    out: dict[str, Path] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"{flag_name} expects LEVEL=PATH, got {item!r}")
        level, _, raw_path = item.partition("=")
        level = level.strip()
        raw_path = raw_path.strip()
        if not level:
            raise ValueError(f"{flag_name} has an empty level name in {item!r}")
        if not raw_path:
            raise ValueError(f"{flag_name} {level} has an empty path")
        if level in out:
            raise ValueError(
                f"{flag_name} specifies level {level!r} more than once")
        out[level] = Path(raw_path)
    if not out and not allow_empty:
        raise ValueError(f"{flag_name} requires at least one LEVEL=PATH pair")
    return out


def parse_data_args(items: list[str]) -> dict[str, Path]:
    return _parse_level_path_pairs(items, "--data", allow_empty=False)


def parse_warm_heads_args(items: list[str]) -> dict[str, Path]:
    return _parse_level_path_pairs(items, "--warm-heads", allow_empty=True)


def validate_data_files(data_paths: dict[str, Path]) -> None:
    missing = {lvl: str(p) for lvl, p in data_paths.items() if not p.exists()}
    if missing:
        raise FileNotFoundError(
            f"--data file(s) not found: {missing}")


# ---------------------------------------------------------------------------
# Data loading + pairwise balancing
# ---------------------------------------------------------------------------


def load_level_pairs(path) -> tuple[np.ndarray, np.ndarray]:
    """Load one level's (obs, act) pairs from an already-collected npz.

    Accepts the house demo/per-level convention (`obs_0`/`act_0`, as
    written by scripts/interference_falsifier.write_success_pairs and
    scripts/bc_distill.py's demo glob) or the pooled convention
    (`obs`/`act`, as written by write_pooled_pairs). Never touches the
    emulator — this is a pure npz read."""
    with np.load(str(path), allow_pickle=False) as d:
        keys = set(d.files)
        if {"obs_0", "act_0"} <= keys:
            obs, act = d["obs_0"], d["act_0"]
        elif {"obs", "act"} <= keys:
            obs, act = d["obs"], d["act"]
        else:
            raise ValueError(
                f"{path}: expected obs_0/act_0 (per-level convention) or "
                f"obs/act (pooled convention) keys, found {sorted(keys)}")
        obs = np.asarray(obs, dtype=np.float32)
        act = np.asarray(act, dtype=np.int64)
    if obs.ndim != 2:
        raise ValueError(f"{path}: obs must be 2-D (N, feature_dim), got "
                          f"shape {obs.shape}")
    if act.ndim != 1 or act.shape[0] != obs.shape[0]:
        raise ValueError(f"{path}: act must be 1-D matching obs row count, "
                          f"got obs {obs.shape} act {act.shape}")
    return obs, act


def infer_feature_dim(by_level: dict[str, tuple[np.ndarray, np.ndarray]]) -> int:
    dims = {lvl: int(obs.shape[1]) for lvl, (obs, _act) in by_level.items()}
    uniq = set(dims.values())
    if len(uniq) != 1:
        raise ValueError(
            f"levels disagree on feature_dim (observation width): {dims} — "
            f"a shared trunk requires identical observation width across "
            f"every level in the pool.")
    return uniq.pop()


def infer_num_actions(by_level: dict[str, tuple[np.ndarray, np.ndarray]]) -> int:
    sizes = [int(act.max()) for _lvl, (_obs, act) in by_level.items()
             if act.size]
    if not sizes:
        raise ValueError("no pairs loaded for any level; cannot infer "
                          "num_actions")
    return max(sizes) + 1


def balance_pairwise(
    by_level: dict[str, tuple[np.ndarray, np.ndarray]],
    levels: list[str],
) -> dict:
    """The pooled set, balanced PAIR-WISE across `levels`.

    Every level is trimmed to the smallest level's pair count, keeping
    the LEADING pairs in collection order (deterministic; the trailing
    trim may cut mid-episode, which is harmless to a per-pair BC loss).
    This is the same reasoning and mechanics as
    scripts/interference_falsifier.balance_fifty_fifty, generalized from
    two levels to N: pair-level rather than episode-level balance
    because longer levels' trajectories would otherwise dominate a
    per-pair gradient. `level_ids` index into `levels` (0-based), so the
    training loop and the export step share one level ordering.

    Raises RuntimeError if any level has zero pairs — a set missing a
    whole level cannot exercise per-level head routing for it."""
    counts = {lvl: int(by_level[lvl][1].shape[0]) for lvl in levels}
    empty = [lvl for lvl, n in counts.items() if n == 0]
    if empty:
        raise RuntimeError(
            f"no pairs for level(s) {empty} — pairwise balancing needs "
            f"data on every level (counts: {counts})")
    target = min(counts.values())
    obs_parts, act_parts, id_parts = [], [], []
    for idx, lvl in enumerate(levels):
        obs, act = by_level[lvl]
        obs_parts.append(np.asarray(obs)[:target])
        act_parts.append(np.asarray(act, dtype=np.int64)[:target])
        id_parts.append(np.full(target, idx, dtype=np.int64))
    return {
        "obs": np.concatenate(obs_parts, axis=0),
        "act": np.concatenate(act_parts, axis=0),
        "level_ids": np.concatenate(id_parts, axis=0),
        "levels": list(levels),
        "counts": {lvl: target for lvl in levels},
        "counts_before_balance": counts,
    }


def format_balance_report(pooled: dict) -> str:
    lines = ["[train_shared_substrate] pairwise balance "
             "(before -> after, leading-order trim):"]
    for lvl in pooled["levels"]:
        before = pooled["counts_before_balance"][lvl]
        after = pooled["counts"][lvl]
        lines.append(f"  {lvl}: {before} -> {after}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def _require_model_class():
    if MultiHeadTilePolicy is None:
        raise RuntimeError(
            "src/training/multihead_policy.py is not importable "
            f"({_IMPORT_ERROR!r}). This script is written against the "
            "interface documented at the top of this file: "
            "MultiHeadTilePolicy(num_actions, num_levels, feature_dim, "
            "hidden_dim=.., trunk_dim=.., levels=..).forward_ac(x, "
            "level_ids) -> (logits, value); .export_head(level) -> a "
            "TilePolicyNetwork-layout state dict (fc1/norm1/fc2/norm2/"
            "actor/critic). Land that module (or adjust this script to "
            "its actual signature) before running anything beyond the "
            "data/CLI checks below.")
    return MultiHeadTilePolicy


def build_model(levels: list[str], num_actions: int, feature_dim: int,
                 hidden_dim: int, trunk_dim: int):
    cls = _require_model_class()
    return cls(num_actions=int(num_actions), num_levels=len(levels),
                feature_dim=int(feature_dim), hidden_dim=int(hidden_dim),
                trunk_dim=int(trunk_dim), levels=list(levels))


# ---------------------------------------------------------------------------
# Warm start — thin CLI-facing wrappers around the model's own
# init_trunk_from / init_head_from (they accept a path, an on-disk
# payload, or a bare state dict, and raise ValueError on a missing key
# or shape mismatch; this script only adds the --warm-* error framing).
# ---------------------------------------------------------------------------


def warm_start_trunk(net, ckpt_path) -> list[str]:
    net.init_trunk_from(str(ckpt_path))
    return list(_TRUNK_KEYS)


def warm_start_head(net, level: str, ckpt_path) -> list[str]:
    net.init_head_from(level, str(ckpt_path))
    return list(_HEAD_KEYS)


# ---------------------------------------------------------------------------
# Training — joint BC with per-sample head routing
# ---------------------------------------------------------------------------


def _split_train_val(L: "torch.Tensor", levels: list[str], val_frac: float,
                      gen: "torch.Generator") -> tuple["torch.Tensor", "torch.Tensor"]:
    """Per-level train/val index split, so the held-out set stays
    balanced across levels the same way the training set is.

    A level too small to spare at least one pair for validation (and
    still leave at least one for training) keeps ALL its pairs in
    train and contributes none to val for that level — callers must
    treat that level's val accuracy as unavailable (nan), not as a
    silently-empty-but-passing split."""
    train_idx: list[torch.Tensor] = []
    val_idx: list[torch.Tensor] = []
    for level_idx, _lvl in enumerate(levels):
        idx = (L == level_idx).nonzero(as_tuple=True)[0]
        n_lvl = int(idx.shape[0])
        if n_lvl == 0:
            continue
        perm = idx[torch.randperm(n_lvl, generator=gen)]
        n_val = int(round(n_lvl * float(val_frac))) if val_frac > 0 else 0
        n_val = min(n_val, n_lvl - 1) if n_lvl > 1 else 0  # keep >=1 train
        if n_val > 0:
            val_idx.append(perm[:n_val])
            train_idx.append(perm[n_val:])
        else:
            train_idx.append(perm)
    train_cat = (torch.cat(train_idx) if train_idx
                 else torch.zeros(0, dtype=torch.long))
    val_cat = (torch.cat(val_idx) if val_idx
               else torch.zeros(0, dtype=torch.long))
    return train_cat, val_cat


def _per_level_accuracy(net, X, Y, L, levels: list[str],
                         idx: "torch.Tensor") -> tuple[float, dict]:
    """No-grad accuracy over the samples named by `idx`, overall and
    broken out per level. `idx` selects the FULL-pool rows this call is
    scored on — callers decide whether that's the training rows (a
    fitting-set number) or a held-out split (never seen by a gradient
    step); this helper doesn't know or care which, so it must never be
    the sole place a caller consults for that distinction."""
    if int(idx.shape[0]) == 0:
        return float("nan"), {lvl: float("nan") for lvl in levels}
    net.eval()
    with torch.no_grad():
        logits, _value = net.forward_ac(X[idx], L[idx])
        correct = (logits.argmax(-1) == Y[idx])
        overall = float(correct.float().mean())
        lvl_ids = L[idx]
        per_level = {}
        for level_idx, lvl in enumerate(levels):
            mask = lvl_ids == level_idx
            per_level[lvl] = (float(correct[mask].float().mean())
                               if bool(mask.any()) else float("nan"))
    return overall, per_level


def train_joint_bc(net, pooled: dict, *, epochs: int, lr: float,
                    batch_size: int, seed: int, val_frac: float = 0.1,
                    log=print) -> list[dict]:
    """Joint BC over the pooled, balanced pairs.

    Every minibatch's forward pass is `net.forward_ac(x, level_ids)` —
    per-sample routing means each sample's loss only ever touches its
    own level's head plus the shared trunk, so the trunk's gradient sees
    every level every step while a level absent from a minibatch leaves
    its head's `.grad` untouched.

    Per-level BC accuracy is reported every epoch in TWO numbers, and
    they must not be conflated:
      - `per_level_train_acc` — no-grad accuracy on the exact rows that
        produced the epoch's gradient (the fitting set). A shared trunk
        can drive this to ~1.0 for every level while still memorizing
        rather than generalizing.
      - `per_level_val_acc` — no-grad accuracy on a `val_frac` slice of
        each level's balanced pairs that is carved out BEFORE the first
        training step and never appears in any minibatch. This is a
        held-out number, but it is still offline BC accuracy on the
        SAME collection-produced pair distribution as training — not a
        sticky/jitter rollout, so it is a strictly weaker signal than
        the honest-eval clear rate and can look fine while the rolled-
        out policy still collapses.
    Neither number is validated in this repo to predict, or even
    correlate with, honest sticky-eval collapse (scripts/eval_shared_
    substrate.py is the only harness that measures that). Both are
    printed every epoch as a cheap same-run diagnostic — a level whose
    accuracy (train or val) visibly diverges from the others is worth
    investigating sooner rather than waiting for eval — but the
    SUPERSEDES / MIXED / FAILED verdict comes only from eval_shared_
    substrate.py against the pre-registered baseline sum, never from
    this function's output.

    `val_frac` <= 0 disables the held-out split entirely: every
    level's val accuracy is reported as nan and training is unchanged
    from a plain full-pool fit (val_frac=0.1 is still the default
    because the split is nearly free to compute and materially more
    honest than fitting-set-only accuracy)."""
    levels = list(pooled["levels"])
    X = torch.from_numpy(np.asarray(pooled["obs"])).float()
    Y = torch.from_numpy(np.asarray(pooled["act"])).long()
    L = torch.from_numpy(np.asarray(pooled["level_ids"])).long()
    n = int(X.shape[0])
    if n == 0:
        raise ValueError("train_joint_bc on zero pairs")

    torch.manual_seed(int(seed))
    opt = torch.optim.Adam(net.parameters(), lr=float(lr))
    gen = torch.Generator()
    gen.manual_seed(int(seed))

    train_idx, val_idx = _split_train_val(L, levels, val_frac, gen)
    if int(train_idx.shape[0]) == 0:
        raise ValueError("train_joint_bc: val_frac left zero training pairs")
    n_train = int(train_idx.shape[0])

    history: list[dict] = []
    for epoch in range(int(epochs)):
        net.train()
        perm = train_idx[torch.randperm(n_train, generator=gen)]
        tot_loss = 0.0
        for i in range(0, n_train, int(batch_size)):
            idx = perm[i:i + int(batch_size)]
            xb, yb, lb = X[idx], Y[idx], L[idx]
            logits, _value = net.forward_ac(xb, lb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * int(idx.shape[0])

        overall_train_acc, per_level_train_acc = _per_level_accuracy(
            net, X, Y, L, levels, train_idx)
        overall_val_acc, per_level_val_acc = _per_level_accuracy(
            net, X, Y, L, levels, val_idx)
        row = {"epoch": epoch, "loss": tot_loss / n_train,
               "overall_train_acc": overall_train_acc,
               "overall_val_acc": overall_val_acc,
               "per_level_train_acc": per_level_train_acc,
               "per_level_val_acc": per_level_val_acc}
        history.append(row)
        log(f"[train_shared_substrate] epoch {epoch:3d}  "
            f"loss={row['loss']:.4f}  "
            f"train_acc={overall_train_acc:.3f}  "
            f"val_acc={overall_val_acc:.3f}  " +
            "  ".join(f"{lvl}(train={ta:.3f},val={per_level_val_acc[lvl]:.3f})"
                      for lvl, ta in per_level_train_acc.items()))
    return history


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _slug(level: str) -> str:
    return str(level).replace("/", "_")


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(tmp))
    os.replace(tmp, path)


def export_checkpoints(net, levels: list[str], out_dir,
                        *, epoch: int = 0,
                        extra_provenance: Optional[dict] = None,
                        ) -> dict[str, Path]:
    """One single-head checkpoint per level, in the standard
    net_state_dict payload eval_game.py loads (`--checkpoint <path>`),
    via `net.export_head(level)`. Verifies the returned keys are exactly
    the TilePolicyNetwork layout `build_tile_policy_from_checkpoint`
    shape-infers from — a mismatch there would write a checkpoint the
    eval harness cannot load, silently, so it fails loud here instead."""
    out_dir = Path(out_dir)
    written: dict[str, Path] = {}
    for level in levels:
        sd = {str(k): v for k, v in net.export_head(level).items()}
        missing = [k for k in _STANDARD_HEAD_KEYS if k not in sd]
        if missing:
            raise RuntimeError(
                f"export_head({level!r}) is missing standard "
                f"TilePolicyNetwork key(s) {missing} — eval_game's shape "
                f"inference requires fc1/norm1/fc2/norm2/actor/critic; "
                f"got keys {sorted(sd.keys())}")
        extra = [k for k in sd if k not in _STANDARD_HEAD_KEYS]
        if extra:
            raise RuntimeError(
                f"export_head({level!r}) returned unexpected key(s) "
                f"{extra} outside the standard layout {_STANDARD_HEAD_KEYS}")
        payload = {
            "iter": int(epoch),
            "net_state_dict": {k: v.detach().cpu().clone()
                                for k, v in sd.items()},
            "provenance": "shared_substrate_export",
            "shared_substrate": {"level": level, "levels": list(levels),
                                  **(extra_provenance or {})},
        }
        path = out_dir / f"{_slug(level)}.pt"
        _atomic_save(payload, path)
        written[level] = path
    return written


def export_multihead_checkpoint(net, out_dir, *, epoch: int = 0,
                                 extra_provenance: Optional[dict] = None,
                                 ) -> Path:
    payload = {
        "iter": int(epoch),
        "net_state_dict": {k: v.detach().cpu().clone()
                            for k, v in net.state_dict().items()},
        "provenance": "shared_substrate_multihead",
        "shared_substrate": dict(extra_provenance or {}),
    }
    path = Path(out_dir) / "multihead.pt"
    _atomic_save(payload, path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Shared-trunk / per-level-head offline BC trainer "
                     "(no emulator).")
    ap.add_argument("--data", action="append", default=[],
                     metavar="LEVEL=PATH",
                     help="Per-level success-pair npz, repeatable "
                          "(e.g. --data 1-1=runs/interference/"
                          "success_1_1.npz).")
    ap.add_argument("--out", required=True,
                     help="Output dir for the multi-head checkpoint and "
                          "the per-level exports.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--trunk-dim", type=int, default=64)
    ap.add_argument("--warm-trunk", default=None, metavar="CKPT",
                     help="Initialize the shared trunk from a "
                          "TilePolicyNetwork-layout checkpoint.")
    ap.add_argument("--warm-heads", action="append", default=[],
                     metavar="LEVEL=CKPT", help="Repeatable; initialize "
                     "one level's head from a TilePolicyNetwork-layout "
                     "checkpoint.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.1,
                     help="Fraction of each level's balanced pairs held "
                          "out (per level, before training starts) to "
                          "report per_level_val_acc — accuracy no "
                          "gradient step ever saw. 0 disables the split "
                          "(val accuracy reported as nan). This is still "
                          "offline BC accuracy, not the honest sticky-"
                          "eval clear rate: see scripts/eval_shared_"
                          "substrate.py for the actual interference "
                          "verdict.")
    ap.add_argument("--num-actions", type=int, default=None,
                     help="Override the inferred action-space size.")
    ap.add_argument("--dry-run", action="store_true",
                     help="Validate data, print the balance/export "
                          "plans, construct the model — no training, "
                          "no writes.")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    try:
        data_paths = parse_data_args(args.data)
        validate_data_files(data_paths)
        warm_head_paths = parse_warm_heads_args(args.warm_heads)
    except (ValueError, FileNotFoundError) as e:
        print(f"[train_shared_substrate] {e}", file=sys.stderr)
        return 2

    levels = list(data_paths.keys())
    by_level: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    try:
        for level, path in data_paths.items():
            by_level[level] = load_level_pairs(path)
        feature_dim = infer_feature_dim(by_level)
        num_actions = (int(args.num_actions) if args.num_actions
                       else infer_num_actions(by_level))
        pooled = balance_pairwise(by_level, levels)
    except (ValueError, RuntimeError) as e:
        print(f"[train_shared_substrate] {e}", file=sys.stderr)
        return 2

    print("[train_shared_substrate] pre-balance counts: " +
          ", ".join(f"{lvl}={pooled['counts_before_balance'][lvl]}"
                     for lvl in levels))
    print(format_balance_report(pooled))

    out_dir = Path(args.out)
    export_plan = {lvl: out_dir / f"{_slug(lvl)}.pt" for lvl in levels}
    print("[train_shared_substrate] export plan:")
    for lvl, path in export_plan.items():
        print(f"  {lvl} -> {path}")
    print(f"  multihead -> {out_dir / 'multihead.pt'}")

    try:
        net = build_model(levels, num_actions, feature_dim,
                           args.hidden_dim, args.trunk_dim)
    except RuntimeError as e:
        print(f"[train_shared_substrate] {e}", file=sys.stderr)
        return 3

    print(f"[train_shared_substrate] constructed MultiHeadTilePolicy: "
          f"levels={levels} num_actions={num_actions} "
          f"feature_dim={feature_dim} hidden_dim={args.hidden_dim} "
          f"trunk_dim={args.trunk_dim}")

    if args.dry_run:
        print("[train_shared_substrate] --dry-run: data validated, "
              "balance/export plans printed, model constructed — no "
              "training or writes performed.")
        return 0

    torch.manual_seed(int(args.seed))

    try:
        if args.warm_trunk:
            loaded = warm_start_trunk(net, args.warm_trunk)
            print(f"[train_shared_substrate] warm-started trunk from "
                  f"{args.warm_trunk}: {loaded}")
        for level, ckpt in warm_head_paths.items():
            if level not in levels:
                print(f"[train_shared_substrate] --warm-heads {level}="
                      f"{ckpt} ignored: {level!r} not among --data levels "
                      f"{levels}", file=sys.stderr)
                continue
            loaded = warm_start_head(net, level, ckpt)
            print(f"[train_shared_substrate] warm-started head {level} "
                  f"from {ckpt}: {loaded}")
    except (RuntimeError, ValueError, KeyError) as e:
        print(f"[train_shared_substrate] {e}", file=sys.stderr)
        return 3

    history = train_joint_bc(net, pooled, epochs=args.epochs, lr=args.lr,
                              batch_size=args.batch, seed=args.seed,
                              val_frac=args.val_frac)

    provenance = {
        "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
        "seed": args.seed, "hidden_dim": args.hidden_dim,
        "trunk_dim": args.trunk_dim, "val_frac": args.val_frac,
        "counts": pooled["counts"],
        "counts_before_balance": pooled["counts_before_balance"],
        "final_per_level_train_acc": (history[-1]["per_level_train_acc"]
                                       if history else {}),
        "final_per_level_val_acc": (history[-1]["per_level_val_acc"]
                                     if history else {}),
        "note": "per_level_*_acc is offline BC accuracy (fitting-set or "
                "held-out), NOT the honest sticky-eval clear rate; the "
                "interference verdict comes only from "
                "scripts/eval_shared_substrate.py against the "
                "pre-registered baseline sum.",
    }
    export_checkpoints(net, levels, out_dir, epoch=args.epochs,
                        extra_provenance=provenance)
    mh_path = export_multihead_checkpoint(net, out_dir, epoch=args.epochs,
                                           extra_provenance=provenance)
    print(f"[train_shared_substrate] wrote multi-head checkpoint -> "
          f"{mh_path}")
    print(f"[train_shared_substrate] wrote {len(levels)} single-head "
          f"export(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
