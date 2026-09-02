"""Night-2 orchestrator (SMB 1-2) — SIL acquisition, BC lock-in cure,
pre-registered gate, consol2 handoff.

Four gated steps, receipts to runs/night2/night2.jsonl:

STEP 1 — SIL trajectory acquisition.
  DETERMINATION (2026-08-15, torch.load key inspection of BOTH
  checkpoints/_preserved/online_v2_FINAL_consolidated.pt and its twin
  checkpoints/mario_1_2_online_v2/vanilla_ppo_iter_00910.pt): the
  payload keys are exactly {iter, net_state_dict, optimizer_state_dict,
  anticollapse, backward_curriculum, curriculum_resume} — NO SIL key.
  That matches the writer: CheckpointManager.save_iter
  (src/training/checkpoint_manager.py) never serializes the
  self-imitation buffer, and SelfImitationBuffer (src/training/sil.py)
  exposes no state_dict/persistence surface — the buffer lives and dies
  with the trainer process. Step 1 therefore runs in COLLECTION mode:
  roll the final checkpoint SAMPLED (eval_game's --action-select
  sampled semantics) under the house noise (sticky 0.25, jitter 16)
  from the 1-2 entrance and keep episodes whose max_gx >= 3267 or that
  scored episode_success — BOTH labels stored per trajectory (the two
  predicates disagreed 13:1 on real data; neither may masquerade as the
  other). eval_game.py was checked for a trace-saving flag and has
  none, so the collection loop lives here in-process — the
  gen_prevaction_demos.py pool-rollout pattern under eval_game's own
  honest-protocol mechanics (episode_rng_seeds / select_action are
  IMPORTED from eval_game, not re-derived). The persisted-extraction
  branch is still implemented and tested so a future payload that DOES
  carry SIL trajectories is extracted instead of re-collected.

STEP 2 — the BC lock-in cure: 15 epochs cross-entropy on the acquired
  pairs, ACTOR PARAMETERS ONLY (the DR spec's choice: fc1/norm1/fc2/
  norm2 trunk and critic head frozen BIT-IDENTICAL; only actor.* takes
  gradient), lr 1e-4 with per-step linear decay to 0, batch 256. If
  actor-only proves too weak because the trunk is shared with the
  critic, the registered alternative is --bc-scope actor+trunk (trunk
  unfrozen, critic still frozen) — a flag, never an edit. Saved to
  runs/night2/cured.pt in the standard vanilla_ppo_iter payload format
  (net_state_dict + iter). Deliberately absent from that payload:
  optimizer_state_dict (the cure invalidates the source's Adam moments;
  the trainer's resume logs a warning and continues with fresh Adam —
  the correct outcome) and anticollapse (its best_net_snapshot is the
  PRE-cure net; carrying it would arm a rollback that undoes the cure).

STEP 3 — the pre-registered gate, via eval_game subprocesses:
  * honest leg — 50-episode COLD ARGMAX probe, sticky 0.25, jitter 16,
    per-episode RNG (the honest protocol), from the 1-2 entrance:
    median max-x must EXCEED 2059 (the consolidated baseline);
  * det leg — 10-episode zero-noise argmax probe from the rung state
    checkpoints/online_1_2/restart_states/s_000559.state: median must
    EXCEED 2979.
  A failed/empty probe leg can never pass the gate. On PASS the cured
  net is seeded where run_consol2.py's flow resumes it (mechanism
  below). On FAIL: failure receipt, STOP — consol2 is NOT launched (the
  operator decides; `--skip-to 4` is the explicit override).

STEP 4 — exec scripts/run_consol2.py, console streamed to
  runs/night2/consol2_console.log.

CONSOL2 SEEDING MECHANISM (read from run_consol2.seed_checkpoint_dir):
  consol2 continues by copying its pinned source into
  checkpoints/mario_1_2_consol2/vanilla_ppo_iter_00910.pt and letting
  the trainer's tear-tolerant resume (newest-first) pick it up. When
  the dir already holds ANY vanilla_ppo_iter_*.pt, seed_checkpoint_dir
  returns "already seeded ... resume continues that lineage" — its only
  integrity check being that iter-00910, IF present, sha256-matches the
  pin. The cured net is therefore written at iter 911 (source_iter + 1,
  payload iter matching the filename): it sorts NEWEST, run_consol2's
  own seeding stands down, and the resume loads the cured policy while
  the campaign keeps its step axis. Seeding refuses a dir that has
  already progressed past iter 911 (the cured net would be silently
  ignored) and any non-matching file already at iter 911.

`--dry-run` validates the whole assembly without rollouts: source
checkpoint sha256 + load + net shape, the SIL persistence verdict
(REPORTED, on both the preserved checkpoint and its twin), step-1
collection assembly against real paths, the step-2 freeze partition on
the real net, both gate probe commands, the seeding plan, and
run_consol2.py --dry-run as a subprocess. `--skip-to N` resumes at step
N off the prior steps' on-disk artifacts. Every threshold lives in
CONFIG below and is mirrored into runs/night2/manifest.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Reused verbatim from the campaign/consol2 controllers (same jsonl
# plumbing, same probe JSON summary, same sha256 + checkpoint-dir
# derivation) so the receipts stay byte-comparable.
from scripts.run_online_campaign import (  # noqa: E402
    _append_jsonl, _load_yaml, probe_summary,
)
from scripts.run_consol2 import (  # noqa: E402
    CONFIG as CONSOL2_CONFIG, _sha256, consol2_checkpoint_dir,
)
from src.utils.disk_floor import disk_floor_breach  # noqa: E402

# ---------------------------------------------------------------------------
# CONFIG — every threshold night 2 runs on. Mirrored verbatim into
# runs/night2/manifest.json so pre-registration and run cannot drift.
# ---------------------------------------------------------------------------
CONFIG: dict[str, Any] = {
    "run_dir": "runs/night2",
    "receipt_log": "runs/night2/night2.jsonl",
    "consol2_profile": "configs/mario_1_2_consol2.yaml",

    # The final consolidated policy — same pin as run_consol2 (asserted
    # against CONSOL2_CONFIG in tests; sha verified before every step
    # that reads it).
    "source_checkpoint":
        "checkpoints/_preserved/online_v2_FINAL_consolidated.pt",
    "source_checkpoint_sha256":
        "daa34bbedd252e317d2a313224033574210231b22f15c15ce22a68fad383b091",
    "source_twin":
        "checkpoints/mario_1_2_online_v2/vanilla_ppo_iter_00910.pt",

    # Step 1 — SIL acquisition (collection-mode knobs; see module
    # docstring for the persistence determination).
    "sil_pairs_out": "runs/night2/sil_pairs.npz",
    "collect_episodes": 600,
    "collect_max_steps": 3000,
    "collect_sticky": 0.25,
    "collect_jitter": 16,
    "collect_temperature": 1.0,
    "collect_seed": 20260815,
    "collect_workers": 6,
    # Keep predicate: the ~14% sampled success regime — an episode whose
    # frontier reached x>=3267 carried the run through the 1-2 wall
    # (x~2674) into the exit zone. episode_success also keeps (both
    # labels stored regardless).
    "keep_min_max_gx": 3267,

    # Step 2 — the BC lock-in cure (DR spec: actor head only).
    "bc_epochs": 15,
    "bc_lr": 1e-4,
    "bc_batch_size": 256,
    "bc_seed": 20260815,
    "bc_train_scope": "actor",     # --bc-scope actor+trunk = the fallback
    "cured_out": "runs/night2/cured.pt",

    # Step 3 — the pre-registered gate. PASS requires BOTH:
    #   honest median_max_x  > 2059  (50 ep cold argmax, sticky 0.25,
    #                                 jitter 16, per-episode RNG), AND
    #   det rung median      > 2979  (10 ep zero-noise argmax from
    #                                 s_000559.state).
    "gate_honest_episodes": 50,
    "gate_honest_sticky": 0.25,
    "gate_honest_jitter": 16,
    "gate_honest_workers": 5,
    "gate_honest_median_baseline": 2059.0,
    "gate_det_episodes": 10,
    "gate_det_rung_state":
        "checkpoints/online_1_2/restart_states/s_000559.state",
    "gate_det_median_baseline": 2979.0,
    "gate_eval_seed": 20260815,
    "gate_max_steps": 3000,
    "probe_timeout_s": 7200.0,
    "bottleneck_survival_x": 2600,   # probe_summary survival mark (consol2's)

    # Step 4 — the handoff.
    "consol2_script": "scripts/run_consol2.py",
    "consol2_console_log": "runs/night2/consol2_console.log",
}

# The iter the cured checkpoint seeds at: one past the consol2
# continuation source, so it sorts newest in checkpoints/mario_1_2_consol2/
# and the trainer's resume picks it over the pinned source.
CURED_SEED_ITER: int = int(CONSOL2_CONFIG["source_iter"]) + 1

_BC_SCOPES: tuple[str, ...] = ("actor", "actor+trunk")
_TRUNK_PREFIXES: tuple[str, ...] = ("fc1.", "norm1.", "fc2.", "norm2.")


# ---------------------------------------------------------------------------
# Pure pieces (unit-tested on synthetic payloads/tensors —
# tests/test_night2_runner.py)
# ---------------------------------------------------------------------------


_SIL_KEY_TOKENS: tuple[str, ...] = ("sil", "self_imitation")


def _key_is_silish(key) -> bool:
    k = str(key).lower()
    return any(tok in k for tok in _SIL_KEY_TOKENS)


def _as_pair(item) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Coerce one stored trajectory into an (obs, actions) array pair.

    Accepts the SelfImitationBuffer element shape — a 2-sequence of
    (L, F) obs + (L,) actions — in ndarray/tensor/list form. Returns
    None for anything that is not trajectory-shaped; raises ValueError
    on a REAL pair whose lengths disagree (a corrupt trajectory must
    fail loud, mirroring SelfImitationBuffer.add)."""
    if not isinstance(item, (tuple, list)) or len(item) != 2:
        return None
    try:
        obs = np.asarray(item[0])
        act = np.asarray(item[1])
    except Exception:
        return None
    if obs.ndim < 1 or act.ndim != 1 or obs.dtype == object \
            or act.dtype == object or not np.issubdtype(act.dtype, np.number):
        return None
    if obs.shape[0] != act.shape[0]:
        raise ValueError(
            f"persisted SIL trajectory length mismatch: {obs.shape[0]} obs "
            f"vs {act.shape[0]} actions")
    return obs, act.astype(np.int64)


def _coerce_trajs(value) -> list[tuple[np.ndarray, np.ndarray]]:
    """All (obs, actions) trajectories recoverable from one SIL-named
    payload entry. Handles the plausible serializations: a list/tuple of
    per-episode pairs, a dict wrapping such a list under trajs/
    trajectories/_trajs, and a flat {obs, actions} array dict."""
    if isinstance(value, dict):
        for k in ("trajs", "trajectories", "_trajs"):
            if k in value:
                return _coerce_trajs(value[k])
        for ok, ak in (("obs", "actions"), ("obs", "act"), ("obs", "acts")):
            if ok in value and ak in value:
                pair = _as_pair((value[ok], value[ak]))
                return [pair] if pair is not None else []
        return []
    if isinstance(value, (list, tuple)) or type(value).__name__ == "deque":
        trajs = []
        for item in value:
            pair = _as_pair(item)
            if pair is None:
                return []
            trajs.append(pair)
        return trajs
    return []


def _find_sil_entries(payload) -> list[tuple[str, list]]:
    """(key_path, trajectories) for every SIL-named entry in the payload
    that actually holds trajectory data. Walks nested dicts only — never
    descends into tensors/arrays, so a 3 MB checkpoint scans in
    microseconds."""
    found: list[tuple[str, list]] = []

    def walk(node, path: str) -> None:
        if not isinstance(node, dict):
            return
        for k, v in node.items():
            p = f"{path}.{k}" if path else str(k)
            if _key_is_silish(k):
                trajs = _coerce_trajs(v)
                if trajs:
                    found.append((p, trajs))
                    continue
            if isinstance(v, dict):
                walk(v, p)

    walk(payload, "")
    return found


def sil_persistence_verdict(payload) -> dict:
    """THE step-1 determination: does this checkpoint payload carry SIL
    trajectories? Pure over the payload — reported in --dry-run and in
    the step-1 receipt either way, never silently branched on."""
    entries = _find_sil_entries(payload) if isinstance(payload, dict) else []
    return {
        "persisted": bool(entries),
        "sil_keys": [p for p, _ in entries],
        "n_trajs": sum(len(ts) for _, ts in entries),
        "n_steps": sum(int(a.shape[0]) for _, ts in entries for _, a in ts),
        "payload_keys": sorted(str(k) for k in payload.keys())
        if isinstance(payload, dict) else [],
    }


def extract_sil_pairs(
    payload,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Flat (obs, actions, traj_len) off the persisted SIL trajectories,
    episode order preserved; None when the payload carries none (which
    is what flips step 1 into collection mode)."""
    entries = _find_sil_entries(payload) if isinstance(payload, dict) else []
    if not entries:
        return None
    trajs = [t for _, ts in entries for t in ts]
    obs = np.concatenate([o for o, _ in trajs], axis=0)
    act = np.concatenate([a for _, a in trajs], axis=0).astype(np.int64)
    traj_len = np.array([a.shape[0] for _, a in trajs], dtype=np.int64)
    return obs, act, traj_len


def keep_episode(ep: dict, min_gx: int) -> bool:
    """The pre-registered keep predicate: frontier reached the exit zone
    (max_gx >= min_gx) OR strict episode_success. The two predicates
    disagreed 13:1 on real data, so neither vetoes the other; both are
    stored per trajectory either way."""
    return int(ep["max_gx"]) >= int(min_gx) or bool(ep["episode_success"])


def write_sil_pairs(
    path,
    *,
    obs: np.ndarray,
    act: np.ndarray,
    traj_len: np.ndarray,
    label_max_gx: np.ndarray,
    label_episode_success: np.ndarray,
    provenance: dict,
) -> None:
    """runs/night2/sil_pairs.npz in the house demo convention (n=1 +
    obs_0/act_0, consumable by the distill scripts unchanged) plus the
    night-2 per-trajectory labels and a provenance JSON blob."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    label_max_gx = np.asarray(label_max_gx, dtype=np.int64)
    np.savez(
        path,
        n=1,
        obs_0=np.asarray(obs),
        act_0=np.asarray(act, dtype=np.int64),
        traj_len=np.asarray(traj_len, dtype=np.int64),
        label_max_gx=label_max_gx,
        label_reached_gx=label_max_gx >= int(CONFIG["keep_min_max_gx"]),
        label_episode_success=np.asarray(label_episode_success, dtype=bool),
        provenance=json.dumps(provenance),
    )


def linear_decay_factor(step: int, total_steps: int) -> float:
    """The BC lr schedule: 1.0 at step 0, linearly to 0.0 at the final
    optimizer step (0.0 outright for a degenerate zero-step run)."""
    if total_steps <= 0:
        return 0.0
    return max(0.0, 1.0 - float(step) / float(total_steps))


def trainable_param_names(names, scope: str) -> set[str]:
    """The freeze partition for the cure. `actor` (the DR spec) trains
    actor.* only; `actor+trunk` is the registered fallback for the
    shared-trunk-too-weak case — trunk unfrozen, critic ALWAYS frozen
    (a BC loss has no value target; touching the critic would only
    corrupt it for the consol2 resume)."""
    if scope not in _BC_SCOPES:
        raise ValueError(f"scope must be one of {_BC_SCOPES}, got {scope!r}")
    keep: set[str] = set()
    for n in names:
        if n.startswith("actor."):
            keep.add(n)
        elif scope == "actor+trunk" and n.startswith(_TRUNK_PREFIXES):
            keep.add(n)
    return keep


def bc_cure(
    payload: dict,
    obs: np.ndarray,
    act: np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    scope: str,
) -> tuple[dict, dict]:
    """The lock-in cure: cross-entropy BC of the acquired (obs, action)
    pairs into the source policy, gradient restricted to the scope's
    parameters. Everything outside the scope stays BIT-IDENTICAL (no
    optimizer state, no grad, no touch). Deterministic from `seed` (own
    torch.Generator for the shuffle; CPU ops). Returns (cured
    state_dict on cpu, stats)."""
    from src.models.tile_policy import build_tile_policy_from_checkpoint

    sd = payload.get("net_state_dict", payload)
    num_actions = int(sd["actor.bias"].shape[0])
    feature_dim = int(sd["fc1.weight"].shape[1])
    net, is_recurrent = build_tile_policy_from_checkpoint(
        payload, num_actions=num_actions, feature_dim=feature_dim)
    if is_recurrent:
        raise ValueError("the BC cure targets the stateless tile policy; "
                         "got a recurrent checkpoint")
    net.load_state_dict(sd, strict=True)

    trainable = trainable_param_names(
        [n for n, _ in net.named_parameters()], scope)
    params = []
    for name, p in net.named_parameters():
        p.requires_grad = name in trainable
        if p.requires_grad:
            params.append(p)

    X = torch.from_numpy(np.asarray(obs)).float()
    Y = torch.from_numpy(np.asarray(act)).long()
    n = int(X.shape[0])
    if n == 0:
        raise ValueError("bc_cure on zero pairs")

    steps_per_epoch = (n + int(batch_size) - 1) // int(batch_size)
    total_steps = max(1, int(epochs) * steps_per_epoch)
    opt = torch.optim.Adam(params, lr=float(lr))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: linear_decay_factor(s, total_steps))
    gen = torch.Generator()
    gen.manual_seed(int(seed))

    net.eval()
    with torch.no_grad():
        logits, _ = net.forward_ac(X)
        initial_loss = float(F.cross_entropy(logits, Y))

    net.train()
    for _ in range(int(epochs)):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, int(batch_size)):
            idx = perm[i:i + int(batch_size)]
            logits, _ = net.forward_ac(X[idx])
            loss = F.cross_entropy(logits, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

    net.eval()
    with torch.no_grad():
        logits, _ = net.forward_ac(X)
        final_loss = float(F.cross_entropy(logits, Y))
        final_acc = float((logits.argmax(-1) == Y).float().mean())

    cured = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    stats = {
        "n_pairs": n,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "lr_schedule": "linear_decay_to_zero",
        "seed": int(seed),
        "scope": scope,
        "trainable_params": int(sum(p.numel() for p in params)),
        "frozen_params": int(sum(
            p.numel() for name, p in net.named_parameters()
            if name not in trainable)),
        "opt_steps": total_steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_acc": final_acc,
    }
    return cured, stats


def write_cured_checkpoint(path, cured_sd: dict, *, stats: dict,
                           provenance: dict) -> None:
    """runs/night2/cured.pt in the standard vanilla_ppo_iter payload
    format (net_state_dict + iter), iter = CURED_SEED_ITER so the file
    can be seeded byte-identically into the consol2 dir. Deliberately
    NO optimizer_state_dict (fresh Adam on resume — the cure invalidates
    the source's moments) and NO anticollapse (its snapshot is the
    pre-cure net; carrying it would arm a rollback that undoes the
    cure)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "iter": CURED_SEED_ITER,
        "net_state_dict": {
            k: torch.as_tensor(v).detach().cpu()
            for k, v in cured_sd.items()
        },
        "provenance": "night2_bc_cure",
        "night2": {"stats": dict(stats), **dict(provenance)},
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, str(tmp))
    os.replace(tmp, path)


def gate_verdict(honest_summ: dict, det_summ: dict,
                 cfg: Optional[dict] = None) -> dict:
    """The pre-registered PASS/FAIL: honest median STRICTLY above the
    2059 baseline AND det rung median STRICTLY above 2979. A failed or
    empty probe leg fails the gate outright — a broken harness is not
    evidence about the policy."""
    cfg = cfg or CONFIG
    reasons: list[str] = []

    def leg_ok(summ: dict) -> bool:
        return (summ.get("status") == "ok"
                and int(summ.get("n_episodes", 0) or 0) > 0)

    for name, summ, baseline_key in (
            ("honest", honest_summ, "gate_honest_median_baseline"),
            ("det", det_summ, "gate_det_median_baseline")):
        baseline = float(cfg[baseline_key])
        if not leg_ok(summ):
            reasons.append(
                f"{name} probe leg unusable (status="
                f"{summ.get('status')!r}, n_episodes="
                f"{summ.get('n_episodes')}) — a failed probe can never "
                f"pass the gate")
        elif not (float(summ["median_max_x"]) > baseline):
            reasons.append(
                f"{name} median_max_x "
                f"{float(summ['median_max_x']):.0f} <= baseline "
                f"{baseline:.0f} (must strictly exceed)")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "honest_median_max_x": honest_summ.get("median_max_x"),
        "honest_baseline": float(cfg["gate_honest_median_baseline"]),
        "det_median_max_x": det_summ.get("median_max_x"),
        "det_baseline": float(cfg["gate_det_median_baseline"]),
    }


def steps_from(skip_to: int) -> list[int]:
    """--skip-to N resumability: the ordered steps still to run."""
    if not 1 <= int(skip_to) <= 4:
        raise ValueError(f"--skip-to must be 1..4, got {skip_to!r}")
    return list(range(int(skip_to), 5))


# ---- eval command assembly ------------------------------------------------


def _eval_game_command(*, checkpoint, episodes: int, max_steps: int,
                       sticky: float, jitter: int, workers: int,
                       eval_rng: str, start_state, eval_seed: int,
                       action_select: str = "greedy",
                       temperature: Optional[float] = None) -> list[str]:
    """One eval_game argv under the consol2 profile. Greedy is
    eval_game's default; only a sampled run opts in explicitly (with its
    temperature on the row, per the receipt discipline)."""
    base = _load_yaml(REPO / CONFIG["consol2_profile"])
    cmd = [
        sys.executable, str(REPO / "scripts" / "eval_game.py"),
        "--game", "mario_1_2_consol2",
        "--profile", str(REPO / CONFIG["consol2_profile"]),
        "--rom", str(REPO / base["rom_path"]),
        "--checkpoint", str(checkpoint),
        "--episodes", str(int(episodes)),
        "--max-steps", str(int(max_steps)),
        "--sequential", "--level-clear",
        "--start-state", str(start_state),
        "--eval-seed", str(int(eval_seed)),
        "--sticky-prob", str(float(sticky)),
        "--start-jitter", str(int(jitter)),
        "--eval-workers", str(int(workers)),
        "--eval-rng", str(eval_rng),
    ]
    if action_select != "greedy":
        cmd += ["--action-select", str(action_select),
                "--temperature", str(float(temperature))]
    return cmd


def build_honest_probe_command(*, checkpoint) -> list[str]:
    """Gate leg 1: 50-episode COLD ARGMAX honest probe — sticky 0.25,
    jitter 16, per-episode RNG, from the 1-2 entrance (the consol2
    profile's start state). The protocol the 2059 baseline was measured
    on."""
    base = _load_yaml(REPO / CONFIG["consol2_profile"])
    return _eval_game_command(
        checkpoint=checkpoint,
        episodes=CONFIG["gate_honest_episodes"],
        max_steps=CONFIG["gate_max_steps"],
        sticky=CONFIG["gate_honest_sticky"],
        jitter=CONFIG["gate_honest_jitter"],
        workers=CONFIG["gate_honest_workers"],
        eval_rng="per-episode",
        start_state=REPO / base["start_state_path"],
        eval_seed=CONFIG["gate_eval_seed"],
    )


def build_det_probe_command(*, checkpoint) -> list[str]:
    """Gate leg 2: 10-episode ZERO-NOISE argmax probe from the rung
    state s_000559 — no sticky, no jitter, serial shared-stream (a
    deterministic replay, so the RNG mode is moot but pinned)."""
    return _eval_game_command(
        checkpoint=checkpoint,
        episodes=CONFIG["gate_det_episodes"],
        max_steps=CONFIG["gate_max_steps"],
        sticky=0.0,
        jitter=0,
        workers=1,
        eval_rng="shared-stream",
        start_state=REPO / CONFIG["gate_det_rung_state"],
        eval_seed=CONFIG["gate_eval_seed"],
    )


def build_collection_reference_command(*, checkpoint) -> list[str]:
    """The eval_game argv step-1 collection is protocol-equivalent to
    (sampled, house noise, per-episode RNG). eval_game has no
    trace-saving flag, so this command is the RECEIPT of the protocol —
    the traces themselves come from the in-process loop below."""
    base = _load_yaml(REPO / CONFIG["consol2_profile"])
    return _eval_game_command(
        checkpoint=checkpoint,
        episodes=CONFIG["collect_episodes"],
        max_steps=CONFIG["collect_max_steps"],
        sticky=CONFIG["collect_sticky"],
        jitter=CONFIG["collect_jitter"],
        workers=CONFIG["collect_workers"],
        eval_rng="per-episode",
        start_state=REPO / base["start_state_path"],
        eval_seed=CONFIG["collect_seed"],
        action_select="sampled",
        temperature=CONFIG["collect_temperature"],
    )


def collection_plan() -> dict:
    """The step-1 fallback plan, in one receipt-able dict."""
    base = _load_yaml(REPO / CONFIG["consol2_profile"])
    return {
        "mode": "collection",
        "why": ("no SIL trajectories persisted in the checkpoint payload "
                "(keys: iter/net_state_dict/optimizer_state_dict/"
                "anticollapse/backward_curriculum/curriculum_resume; "
                "CheckpointManager.save_iter never writes the buffer)"),
        # eval_game.py's argparse was checked for a trace/tape-saving
        # flag: none exists, so the collector is the in-process loop.
        "trace_flag": None,
        "collector": ("in-process pool rollouts (collect_trajectories) — "
                      "the gen_prevaction_demos.py pattern under "
                      "eval_game's episode_rng_seeds/select_action"),
        "checkpoint": str(REPO / CONFIG["source_checkpoint"]),
        "start_state": str(REPO / base["start_state_path"]),
        "episodes": CONFIG["collect_episodes"],
        "max_steps": CONFIG["collect_max_steps"],
        "sticky_prob": CONFIG["collect_sticky"],
        "start_jitter": CONFIG["collect_jitter"],
        "action_select": "sampled",
        "temperature": CONFIG["collect_temperature"],
        "eval_rng": "per-episode",
        "eval_seed": CONFIG["collect_seed"],
        "lanes": CONFIG["collect_workers"],
        "keep_min_max_gx": CONFIG["keep_min_max_gx"],
        "labels": ["max_gx", "episode_success"],
        "reference_command": build_collection_reference_command(
            checkpoint=REPO / CONFIG["source_checkpoint"]),
    }


# ---- step 1 fallback: the collection loop ---------------------------------


def collect_trajectories(
    pool,
    *,
    n_episodes: int,
    lanes: int,
    max_steps: int,
    bitmasks,
    make_policy: Callable[[], Any],
    make_reward_fn: Callable[[], Any],
    start_blob: bytes,
    sticky_prob: float,
    start_jitter: int,
    eval_seed: int,
    action_select: str,
    temperature: float,
) -> list[dict]:
    """Play `n_episodes` across `lanes` pool workers, recording the
    (obs, EXECUTED action) trace of every control step, plus max_gx and
    episode_success per episode.

    Protocol mechanics mirror eval_game's serial loop exactly — the
    post-restore flush step, Machado no-op starts (0..jitter, drawn
    before control), the sticky guard (no draw on control step 0,
    repeat the previous EXECUTED action), and per-episode RNG derivation
    via eval_game.episode_rng_seeds — so a collected trajectory is a
    draw of the SAME distribution the honest probes measure. Episode i
    is a pure function of (eval_seed, i), which makes the result
    invariant to the lane count (asserted by the stub suite).

    Rounds are batch-synchronous: all lanes start together, a finished
    lane holds NOOP until the round drains. Simpler than the eval
    executor's free-running lanes and costs only straggler steps."""
    from scripts.eval_game import episode_rng_seeds, select_action

    lanes = max(1, int(lanes))
    pols = [make_policy() for _ in range(lanes)]
    rfns = [make_reward_fn() for _ in range(lanes)]
    device = torch.device("cpu")
    episodes: list[dict] = []
    n_rounds = (int(n_episodes) + lanes - 1) // lanes
    for rnd in range(n_rounds):
        pool.reset_all()
        lane: list[Optional[dict]] = []
        for i in range(lanes):
            ep_idx = rnd * lanes + i
            if ep_idx >= int(n_episodes):
                lane.append(None)
                continue
            pool.load_worker_state(i, start_blob)
            np_seed, torch_seed = episode_rng_seeds(int(eval_seed), ep_idx)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(torch_seed)
            rfns[i].reset()
            lane.append({
                "ep": ep_idx,
                "rng": np.random.RandomState(np_seed),
                "gen": gen,
                "obs": None,
                "hidden": None,
                "prev_action": 0,
                "control_steps": 0,
                "jitter_left": 0,
                "max_gx": 0,
                "cleared": False,
                "done": False,
                "obs_rows": [],
                "act_rows": [],
            })
        # Post-restore flush (load_worker_state produces no frame until
        # the next step — same pattern as eval_game/the trainer).
        init = pool.step_all(np.zeros(lanes, dtype=np.uint8))
        for i, st in enumerate(lane):
            if st is None:
                continue
            st["jitter_left"] = (
                int(st["rng"].randint(0, int(start_jitter) + 1))
                if int(start_jitter) > 0 else 0)
            if st["jitter_left"] == 0:
                st["obs"] = pols[i].reset(init[i])
                st["hidden"] = pols[i].initial_hidden(device)

        budget = int(max_steps) + int(start_jitter) + 1
        for _ in range(budget):
            if all(st is None or st["done"] for st in lane):
                break
            acts = np.zeros(lanes, dtype=np.uint8)
            for i, st in enumerate(lane):
                if st is None or st["done"] or st["jitter_left"] > 0:
                    continue  # idle/jitter lanes hold NOOP
                with torch.no_grad():
                    logits, st["hidden"] = pols[i].logits(
                        st["obs"], st["hidden"], st["prev_action"])
                    a = select_action(
                        logits, action_select, temperature, st["gen"])
                # Machado sticky: repeat the previous EXECUTED action —
                # guard matches eval_game (no draw on control step 0).
                if (float(sticky_prob) > 0.0 and st["control_steps"] > 0
                        and st["rng"].random() < float(sticky_prob)):
                    a = st["prev_action"]
                st["prev_action"] = a
                # Record BEFORE the step: the obs this action was chosen
                # from, and the action actually EXECUTED (post-sticky) —
                # the SelfImitationBuffer semantics.
                st["obs_rows"].append(
                    np.asarray(st["obs"], dtype=np.int8).copy())
                st["act_rows"].append(int(a))
                acts[i] = bitmasks[a]
            r = pool.step_all(acts)
            for i, st in enumerate(lane):
                if st is None or st["done"]:
                    continue
                if st["jitter_left"] > 0:
                    st["jitter_left"] -= 1
                    if st["jitter_left"] == 0:
                        # Control begins next step; seed the stacker off
                        # the final jitter frame (eval_game's ordering).
                        st["obs"] = pols[i].reset(r[i])
                        st["hidden"] = pols[i].initial_hidden(device)
                    continue
                ram = r[i][2]
                st["control_steps"] += 1
                _, rew_done, _ = rfns[i].compute(ram, action=int(acts[i]))
                gx = (int(ram[0x006D]) << 8) | int(ram[0x0086])
                if gx > st["max_gx"]:
                    st["max_gx"] = gx
                st["obs"] = pols[i].push(r[i])
                if rfns[i].episode_success():
                    st["cleared"] = True
                if (rew_done or bool(r[i][3]) or st["cleared"]
                        or st["control_steps"] >= int(max_steps)):
                    st["done"] = True

        for st in lane:
            if st is None:
                continue
            episodes.append({
                "episode_index": st["ep"],
                "obs": (np.stack(st["obs_rows"]) if st["obs_rows"]
                        else np.zeros((0, 0), dtype=np.int8)),
                "actions": np.asarray(st["act_rows"], dtype=np.int64),
                "max_gx": int(st["max_gx"]),
                "episode_success": bool(st["cleared"]),
                "length": int(st["control_steps"]),
            })
    return episodes


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_manifest() -> dict:
    """CONFIG + the gate pre-registration, verbatim, for
    runs/night2/manifest.json."""
    return {
        "campaign": "smb_1_2_night2",
        "config": copy.deepcopy(CONFIG),
        "gate": {
            "honest_median_must_exceed":
                float(CONFIG["gate_honest_median_baseline"]),
            "det_rung_median_must_exceed":
                float(CONFIG["gate_det_median_baseline"]),
            "honest_protocol": ("50 ep cold argmax, sticky 0.25, jitter "
                                "16, per-episode RNG, 1-2 entrance"),
            "det_protocol": ("10 ep zero-noise argmax from rung "
                             "s_000559.state"),
            "on_fail": ("stop; write failure receipt; do NOT launch "
                        "consol2 (operator decides; --skip-to 4 is the "
                        "explicit override)"),
        },
        "steps": [
            {"step": 1, "name": "sil_acquisition"},
            {"step": 2, "name": "bc_cure"},
            {"step": 3, "name": "gate"},
            {"step": 4, "name": "consol2"},
        ],
        "consol2": {
            "source_iter": int(CONSOL2_CONFIG["source_iter"]),
            "cured_seed_iter": CURED_SEED_ITER,
            "script": CONFIG["consol2_script"],
        },
        "created_at": time.time(),
    }


# ---------------------------------------------------------------------------
# Runtime pieces (subprocess / filesystem / emulator; exercised by --dry-run
# and the real run, not by the unit tests — except seeding, which is pure
# enough to test on tmp dirs)
# ---------------------------------------------------------------------------


def _load_payload(path) -> dict:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _verify_source() -> Path:
    src = REPO / CONFIG["source_checkpoint"]
    got = _sha256(src)
    if got != CONFIG["source_checkpoint_sha256"]:
        raise RuntimeError(
            f"source checkpoint sha256 mismatch: {src} is {got}, pinned "
            f"{CONFIG['source_checkpoint_sha256']} — refusing to distill "
            f"an unverified policy")
    return src


def seed_cured_for_consol2(cured_path, *, ckpt_dir, source_iter: int,
                           plan_only: bool = False) -> str:
    """Seed the cured checkpoint where run_consol2.py's flow resumes it.

    Mechanism (read from run_consol2.seed_checkpoint_dir + the trainer's
    CheckpointManager.resume): the cured net lands at source_iter + 1 so
    it is the NEWEST vanilla_ppo_iter_*.pt in the consol2 dir —
    run_consol2's own seeding then reports "already seeded" (its only
    per-file check is that iter-00910, if present, matches its pin,
    which we never write) and the trainer's newest-first resume loads
    the cured policy. The missing optimizer_state_dict downgrades to a
    logged warning + fresh Adam, which is the intended outcome after a
    BC cure. Refuses a dir that already progressed past the seed iter
    (the cured net would be silently ignored) and any conflicting file
    already at the seed iter."""
    cured_path = Path(cured_path)
    ckpt_dir = Path(ckpt_dir)
    seed_iter = int(source_iter) + 1
    dst = ckpt_dir / f"vanilla_ppo_iter_{seed_iter:05d}.pt"

    if cured_path.exists():
        payload = _load_payload(cured_path)
        if int(payload.get("iter", -1)) != seed_iter:
            raise RuntimeError(
                f"{cured_path} carries iter {payload.get('iter')!r}, "
                f"expected {seed_iter} — resume numbering would break")
    elif not plan_only:
        raise RuntimeError(f"cured checkpoint missing: {cured_path}")

    existing = sorted(ckpt_dir.glob("vanilla_ppo_iter_*.pt")) \
        if ckpt_dir.exists() else []
    newer = [p for p in existing
             if int(p.stem.rsplit("_", 1)[-1]) > seed_iter]
    if newer:
        raise RuntimeError(
            f"{ckpt_dir} already holds {len(newer)} checkpoint(s) past "
            f"iter {seed_iter} (newest {newer[-1].name}) — seeding would "
            f"be silently ignored by the newest-first resume; refusing")
    if dst.exists():
        if cured_path.exists() and _sha256(dst) == _sha256(cured_path):
            return (f"already seeded ({dst.name} sha256-matches the cured "
                    f"checkpoint); resume picks it up")
        raise RuntimeError(
            f"{dst} exists but does not match the cured checkpoint — a "
            f"different lineage; refusing to overwrite")
    if plan_only:
        return (f"will seed {dst} from {cured_path} — newest in dir, so "
                f"run_consol2's seeding stands down and the trainer "
                f"resume loads the cured net")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".pt.tmp")
    shutil.copyfile(cured_path, tmp)
    if _sha256(tmp) != _sha256(cured_path):
        tmp.unlink()
        raise RuntimeError(f"copy of {cured_path} -> {tmp} failed "
                           f"verification")
    os.replace(tmp, dst)
    return f"seeded {dst} from {cured_path} (sha256 verified)"


def _run_eval(cmd: list[str]) -> dict:
    """Run one eval_game subprocess; return its JSON or an error dict
    (same contract as run_consol2.run_probe)."""
    try:
        proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True,
                              text=True, timeout=CONFIG["probe_timeout_s"])
    except subprocess.TimeoutExpired:
        return {"status": "probe_timeout"}
    if proc.returncode != 0:
        return {"status": "probe_failed",
                "detail": proc.stderr.strip()[-500:]}
    text = proc.stdout.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return {"status": "probe_failed", "detail": "no JSON on stdout"}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        return {"status": "probe_failed", "detail": f"bad JSON: {e}"}


def _collect_with_real_pool(plan: dict) -> list[dict]:
    """Assemble the real pool + policy + reward and run the collection
    loop. THE ONLY rollout surface in this module — never reached by
    --dry-run or the unit tests."""
    from nes_core import Pool
    from scripts.eval_game import _TilePolicy
    from src.emulation.frame_utils import TileFeatureStacker
    from src.models.tile_policy import build_tile_policy_from_checkpoint
    from src.training.profile_utils import (
        action_space_to_bitmasks, resolve_encoder,
    )
    from src.utils.reward_functions import build_reward_function

    profile = _load_yaml(REPO / CONFIG["consol2_profile"])
    bitmasks = action_space_to_bitmasks(profile["action_space"])
    extractor, feature_dim, stacked_dim = resolve_encoder(profile)
    stack_size = stacked_dim // feature_dim

    payload = _load_payload(plan["checkpoint"])
    net, is_recurrent = build_tile_policy_from_checkpoint(
        payload, num_actions=len(bitmasks), feature_dim=stacked_dim)
    missing, _ = net.load_state_dict(payload["net_state_dict"], strict=False)
    if missing:
        raise RuntimeError(f"checkpoint is missing policy weights: {missing}")
    net.eval()

    lanes = max(1, min(int(plan["lanes"]), int(plan["episodes"]),
                       int(os.cpu_count() or 1)))
    pool = Pool(rom_path=str(REPO / profile["rom_path"]),
                num_workers=lanes,
                frame_skip=int(profile.get("frame_skip", 4)))
    pool.set_headless(True)
    pool.set_skip_preprocess(True)  # tile obs read RAM, never pixels
    start_blob = Path(plan["start_state"]).read_bytes()
    try:
        return collect_trajectories(
            pool,
            n_episodes=int(plan["episodes"]),
            lanes=lanes,
            max_steps=int(plan["max_steps"]),
            bitmasks=bitmasks,
            make_policy=lambda: _TilePolicy(
                net,
                TileFeatureStacker(stack_size=stack_size,
                                   feature_dim=feature_dim),
                extractor, is_recurrent),
            make_reward_fn=lambda: build_reward_function(profile),
            start_blob=start_blob,
            sticky_prob=float(plan["sticky_prob"]),
            start_jitter=int(plan["start_jitter"]),
            eval_seed=int(plan["eval_seed"]),
            action_select=str(plan["action_select"]),
            temperature=float(plan["temperature"]),
        )
    finally:
        pool.shutdown()


# ---------------------------------------------------------------------------
# The four steps
# ---------------------------------------------------------------------------


def _receipt(row: dict) -> None:
    _append_jsonl(REPO / CONFIG["receipt_log"], row)


def _last_cure_receipt(receipt_log) -> Optional[dict]:
    """Most recent step-2 bc_cure row in the night2 receipt log (or
    None) — the recorded content binding for cured.pt, since (unlike
    the source checkpoint's static CONFIG-pinned sha256) the cured
    checkpoint's hash is per-run and can only be pinned dynamically, to
    the receipt step 2 wrote it with."""
    receipt_log = Path(receipt_log)
    if not receipt_log.exists():
        return None
    last = None
    with open(receipt_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") == "step" and row.get("step") == 2:
                last = row
    return last


def verify_cured_binding(cured_path, receipt_log) -> str:
    """Content-bind cured.pt to the step-2 receipt that wrote it before
    step 3 probes or seeds it. Without this, step 3 trusts whatever
    bytes currently sit at the path alone — any process with
    filesystem access (a second cure script's --install, a bare cp, a
    concurrent night2 invocation) can substitute a different, equally
    well-formed payload with zero trace in the receipt log. Raises
    RuntimeError on a missing receipt or a hash mismatch; returns the
    verified sha256 otherwise."""
    cured_path = Path(cured_path)
    receipt = _last_cure_receipt(receipt_log)
    if receipt is None or not receipt.get("cured_sha256"):
        raise RuntimeError(
            f"no step-2 bc_cure receipt with cured_sha256 in "
            f"{receipt_log} — refusing to gate {cured_path} unverified")
    got = _sha256(cured_path)
    expected = receipt["cured_sha256"]
    if got != expected:
        raise RuntimeError(
            f"{cured_path} sha256 {got} does not match the step-2 "
            f"receipt's {expected} — the file changed after step 2 "
            f"wrote it; refusing to gate an unverified checkpoint")
    return got


def step1_acquire() -> Path:
    """SIL acquisition: extract persisted trajectories if the payload
    carries any; otherwise collect fresh ones (see module docstring for
    the determination). Writes runs/night2/sil_pairs.npz either way."""
    src = _verify_source()
    payload = _load_payload(src)
    verdict = sil_persistence_verdict(payload)
    out = REPO / CONFIG["sil_pairs_out"]

    if verdict["persisted"]:
        obs, act, traj_len = extract_sil_pairs(payload)
        k = int(traj_len.shape[0])
        write_sil_pairs(
            out, obs=obs, act=act, traj_len=traj_len,
            # The buffer stores CLEARED episodes only (sil.py: add() is
            # gated on the completion-diff detector), so success is True
            # by construction; per-episode max_gx is not recorded in the
            # payload (-1 = unknown).
            label_max_gx=np.full(k, -1, dtype=np.int64),
            label_episode_success=np.ones(k, dtype=bool),
            provenance={
                "mode": "persisted_extraction",
                "source": str(src),
                "source_sha256": CONFIG["source_checkpoint_sha256"],
                "sil_keys": verdict["sil_keys"],
                "note": ("SIL buffer stores cleared episodes only; "
                         "max_gx -1 = not recorded in the payload"),
            })
        _receipt({"type": "step", "step": 1, "name": "sil_acquisition",
                  "mode": "persisted_extraction",
                  "persisted": True, "sil_keys": verdict["sil_keys"],
                  "trajs": k, "pairs": int(act.shape[0]),
                  "out": str(out)})
        print(f"[night2] step 1: extracted {k} persisted SIL "
              f"trajectories ({int(act.shape[0])} pairs) -> {out}",
              flush=True)
        return out

    plan = collection_plan()
    print(f"[night2] step 1: SIL absent from the payload "
          f"(keys: {verdict['payload_keys']}) — collecting "
          f"{plan['episodes']} sampled episodes", flush=True)
    episodes = _collect_with_real_pool(plan)
    kept = [e for e in episodes
            if keep_episode(e, CONFIG["keep_min_max_gx"])]
    if not kept:
        _receipt({"type": "step", "step": 1, "name": "sil_acquisition",
                  "mode": "collection", "persisted": False,
                  "episodes_run": len(episodes), "kept": 0,
                  "error": "no keepable episodes"})
        raise RuntimeError(
            f"collection kept 0 of {len(episodes)} episodes (predicate: "
            f"max_gx >= {CONFIG['keep_min_max_gx']} or episode_success) "
            f"— nothing to distill; not writing {out}")
    obs = np.concatenate([e["obs"] for e in kept], axis=0)
    act = np.concatenate([e["actions"] for e in kept], axis=0)
    traj_len = np.array([e["length"] for e in kept], dtype=np.int64)
    write_sil_pairs(
        out, obs=obs, act=act, traj_len=traj_len,
        label_max_gx=np.array([e["max_gx"] for e in kept],
                              dtype=np.int64),
        label_episode_success=np.array(
            [e["episode_success"] for e in kept], dtype=bool),
        provenance={k: v for k, v in plan.items()
                    if k != "reference_command"})
    n_success = sum(1 for e in kept if e["episode_success"])
    _receipt({"type": "step", "step": 1, "name": "sil_acquisition",
              "mode": "collection", "persisted": False,
              "episodes_run": len(episodes), "kept": len(kept),
              "kept_success": n_success,
              "kept_reached_gx": sum(
                  1 for e in kept
                  if e["max_gx"] >= CONFIG["keep_min_max_gx"]),
              "pairs": int(act.shape[0]), "out": str(out)})
    print(f"[night2] step 1: kept {len(kept)}/{len(episodes)} episodes "
          f"({n_success} strict clears, {int(act.shape[0])} pairs) -> "
          f"{out}", flush=True)
    return out


def step2_cure(scope: str) -> Path:
    pairs_path = REPO / CONFIG["sil_pairs_out"]
    if not pairs_path.exists():
        raise RuntimeError(
            f"{pairs_path} missing — run step 1 first (or drop --skip-to)")
    d = np.load(pairs_path)
    obs, act = d["obs_0"], d["act_0"]
    src = _verify_source()
    payload = _load_payload(src)
    cured, stats = bc_cure(
        payload, obs, act,
        epochs=CONFIG["bc_epochs"], lr=CONFIG["bc_lr"],
        batch_size=CONFIG["bc_batch_size"], seed=CONFIG["bc_seed"],
        scope=scope)
    out = REPO / CONFIG["cured_out"]
    write_cured_checkpoint(
        out, cured, stats=stats,
        provenance={"scope": scope, "pairs": str(pairs_path),
                    "source": str(src),
                    "source_sha256": CONFIG["source_checkpoint_sha256"]})
    cured_sha256 = _sha256(out)
    _receipt({"type": "step", "step": 2, "name": "bc_cure",
              "scope": scope, "stats": stats, "out": str(out),
              "cured_sha256": cured_sha256})
    print(f"[night2] step 2: cured (scope={scope}, "
          f"loss {stats['initial_loss']:.4f} -> {stats['final_loss']:.4f}, "
          f"acc {stats['final_acc']:.3f}) -> {out}", flush=True)
    return out


def step3_gate() -> dict:
    cured = REPO / CONFIG["cured_out"]
    if not cured.exists():
        raise RuntimeError(
            f"{cured} missing — run step 2 first (or drop --skip-to)")
    verify_cured_binding(cured, REPO / CONFIG["receipt_log"])
    summs: dict[str, dict] = {}
    for leg, cmd in (("honest", build_honest_probe_command(checkpoint=cured)),
                     ("det", build_det_probe_command(checkpoint=cured))):
        res = _run_eval(cmd)
        summ = probe_summary(res, CONFIG["bottleneck_survival_x"])
        summ["status"] = res.get("status")
        summs[leg] = summ
        _receipt({"type": "probe", "leg": leg, "checkpoint": str(cured),
                  "cmd": " ".join(cmd),
                  **{k: v for k, v in summ.items() if k != "status"},
                  "status": summ["status"]})
        print(f"[night2] step 3: {leg} probe status={summ['status']} "
              f"median_max_x={summ['median_max_x']}", flush=True)

    verdict = gate_verdict(summs["honest"], summs["det"])
    if verdict["passed"]:
        base = _load_yaml(REPO / CONFIG["consol2_profile"])
        note = seed_cured_for_consol2(
            cured, ckpt_dir=consol2_checkpoint_dir(base),
            source_iter=CONSOL2_CONFIG["source_iter"])
        _receipt({"type": "gate", **verdict, "seed_note": note})
        print(f"[night2] step 3: GATE PASSED — {note}", flush=True)
    else:
        _receipt({"type": "gate", **verdict,
                  "note": ("consol2 NOT launched — the operator decides "
                           "(--skip-to 4 is the explicit override)")})
        print("[night2] step 3: GATE FAILED — "
              + "; ".join(verdict["reasons"]), flush=True)
    return verdict


def step4_consol2() -> int:
    log_path = REPO / CONFIG["consol2_console_log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(REPO / CONFIG["consol2_script"])]
    _receipt({"type": "step", "step": 4, "name": "consol2_launch",
              "cmd": " ".join(cmd), "console_log": str(log_path)})
    print(f"[night2] step 4: exec {' '.join(cmd)} (console -> {log_path})",
          flush=True)
    with open(log_path, "ab") as lf:
        proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=lf,
                                stderr=subprocess.STDOUT)
        rc = proc.wait()
    _receipt({"type": "step", "step": 4, "name": "consol2_exit",
              "returncode": rc})
    print(f"[night2] step 4: consol2 exited {rc}", flush=True)
    return rc


# ---------------------------------------------------------------------------
# Dry run — checkpoint loads + SIL verdict + command assembly + consol2
# --dry-run. NOT rollouts.
# ---------------------------------------------------------------------------


def dry_run() -> int:
    ok = True

    def report(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"[dry-run] {'PASS' if good else 'FAIL'} {label}"
              + (f" — {detail}" if detail else ""), flush=True)

    # 1. Consol2 profile parses + strict schema (the profile every eval
    #    command and the collection loop assemble from).
    try:
        from src.training.config_schema import check_profile
        base = _load_yaml(REPO / CONFIG["consol2_profile"])
        warns = check_profile(base, strict=False)
        report("consol2 profile schema (strict)", not warns,
               "; ".join(warns) if warns else CONFIG["consol2_profile"])
    except Exception as e:
        report("consol2 profile schema (strict)", False, repr(e))
        return 1

    # 2. Source checkpoint sha256 matches the consol2 pin.
    try:
        src = _verify_source()
        report("source checkpoint sha256 matches pin", True,
               CONFIG["source_checkpoint_sha256"][:16] + "...")
    except Exception as e:
        report("source checkpoint sha256 matches pin", False, repr(e))
        return 1

    # 3. Source payload loads at the pinned iter and its net matches the
    #    profile's dims (torch.load + shape checks, no rollout).
    payload = None
    try:
        from src.training.profile_utils import resolve_encoder
        _, feature_dim, stacked_dim = resolve_encoder(base)
        payload = _load_payload(src)
        sd = payload["net_state_dict"]
        rl = base.get("reinforce", {})
        fc1 = tuple(sd["fc1.weight"].shape)
        n_act = int(sd["actor.bias"].shape[0])
        it = int(payload.get("iter", -1))
        good = (it == int(CONSOL2_CONFIG["source_iter"])
                and fc1 == (int(rl["tile_hidden_dim"]), stacked_dim)
                and n_act == len(base["action_space"]))
        report("source payload loads at pinned iter + net dims", good,
               f"iter={it} fc1={fc1} actions={n_act} obs={stacked_dim}")
    except Exception as e:
        report("source payload loads at pinned iter + net dims", False,
               repr(e))
        return 1

    # 4. THE step-1 determination, reported: does the payload carry SIL
    #    trajectories? Checked on the preserved checkpoint AND its twin.
    verdict = sil_persistence_verdict(payload)
    twin_note = ""
    twin = REPO / CONFIG["source_twin"]
    if twin.exists():
        try:
            twin_v = sil_persistence_verdict(_load_payload(twin))
            twin_note = (f"; twin agrees (persisted="
                         f"{twin_v['persisted']})")
        except Exception as e:
            twin_note = f"; twin unreadable ({e!r})"
    mode = "persisted_extraction" if verdict["persisted"] else "collection"
    report(
        "SIL persistence verdict determined",
        True,
        f"persisted={verdict['persisted']} -> step-1 mode={mode}; "
        f"payload keys={verdict['payload_keys']}{twin_note}")

    # 5. Step-1 collection assembly against real paths (plan + the
    #    protocol-equivalent eval_game command; eval_game has no
    #    trace-saving flag, so the collector is in-process).
    try:
        plan = collection_plan()
        missing_paths = [
            a for a in plan["reference_command"]
            if "/" in a and not a.startswith("-") and not Path(a).exists()]
        good = (not missing_paths
                and Path(plan["start_state"]).exists()
                and Path(plan["checkpoint"]).exists())
        report("step-1 collection assembles against real paths", good,
               f"episodes={plan['episodes']} sampled T="
               f"{plan['temperature']} sticky={plan['sticky_prob']} "
               f"jitter={plan['start_jitter']} keep_gx>="
               f"{plan['keep_min_max_gx']} lanes={plan['lanes']}"
               if good else f"missing paths: {missing_paths}")
    except Exception as e:
        report("step-1 collection assembles against real paths", False,
               repr(e))

    # 6. Step-2 freeze partition on the REAL net: exactly actor.* trains
    #    under the registered scope; trunk + critic frozen.
    try:
        sd = payload["net_state_dict"]
        trainable = trainable_param_names(list(sd.keys()),
                                          CONFIG["bc_train_scope"])
        n_train = sum(int(sd[k].numel()) for k in trainable)
        n_frozen = sum(int(v.numel()) for k, v in sd.items()
                       if k not in trainable)
        good = (trainable == {"actor.weight", "actor.bias"}
                if CONFIG["bc_train_scope"] == "actor"
                else bool(trainable))
        report("step-2 freeze partition (scope="
               + CONFIG["bc_train_scope"] + ")", good,
               f"trainable={sorted(trainable)} ({n_train} params), "
               f"frozen {n_frozen} params")
    except Exception as e:
        report("step-2 freeze partition (scope="
               + CONFIG["bc_train_scope"] + ")", False, repr(e))

    # 7 + 8. Both gate probe commands assemble against real paths. The
    #    cured checkpoint does not exist before step 2, so the source
    #    stands in for path validation (the real gate substitutes
    #    runs/night2/cured.pt).
    for label, builder in (
            ("honest gate probe command assembles",
             build_honest_probe_command),
            ("det rung probe command assembles", build_det_probe_command)):
        try:
            cmd = builder(checkpoint=src)
            missing_paths = [
                a for a in cmd
                if "/" in a and not a.startswith("-")
                and not Path(a).exists()]
            report(label, not missing_paths,
                   " ".join(cmd) if not missing_paths
                   else f"missing paths: {missing_paths}")
        except Exception as e:
            report(label, False, repr(e))

    # 9. Consol2 seeding plan is coherent (dir-state check only; the
    #    cured file appears after step 2).
    try:
        note = seed_cured_for_consol2(
            REPO / CONFIG["cured_out"],
            ckpt_dir=consol2_checkpoint_dir(base),
            source_iter=CONSOL2_CONFIG["source_iter"], plan_only=True)
        report("consol2 seeding plan (cured at iter "
               f"{CURED_SEED_ITER})", True, note)
    except Exception as e:
        report("consol2 seeding plan (cured at iter "
               f"{CURED_SEED_ITER})", False, repr(e))

    # 10. run_consol2.py --dry-run passes as a subprocess (step 4's own
    #     assembly check: profile, sha pin, seeding, probe commands).
    try:
        proc = subprocess.run(
            [sys.executable, str(REPO / CONFIG["consol2_script"]),
             "--dry-run"],
            cwd=str(REPO), capture_output=True, text=True, timeout=600)
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln]
        good = proc.returncode == 0 and any(
            "ALL CHECKS PASSED" in ln for ln in lines)
        report("run_consol2.py --dry-run passes", good,
               f"rc={proc.returncode}" if good
               else f"rc={proc.returncode}; tail={lines[-3:]}")
    except Exception as e:
        report("run_consol2.py --dry-run passes", False, repr(e))

    # 11. Manifest mirrors CONFIG; write it (pre-registration surface).
    try:
        man = build_manifest()
        good = man["config"] == CONFIG and len(man["steps"]) == 4
        run_dir = REPO / CONFIG["run_dir"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(
            json.dumps(man, indent=2) + "\n")
        report("manifest mirrors CONFIG", good,
               str(run_dir / "manifest.json"))
    except Exception as e:
        report("manifest mirrors CONFIG", False, repr(e))

    _receipt({"type": "dry_run", "ok": ok,
              "sil_persisted": verdict["persisted"],
              "sil_payload_keys": verdict["payload_keys"],
              "step1_mode": mode})
    print(f"[dry-run] {'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'}",
          flush=True)
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# The gated sequence
# ---------------------------------------------------------------------------


def run(skip_to: int = 1, bc_scope: Optional[str] = None) -> int:
    scope = bc_scope or CONFIG["bc_train_scope"]
    if scope not in _BC_SCOPES:
        raise ValueError(f"--bc-scope must be one of {_BC_SCOPES}")
    steps = steps_from(skip_to)

    run_dir = REPO / CONFIG["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(build_manifest(), indent=2) + "\n")
    _receipt({"type": "night2_start", "skip_to": int(skip_to),
              "steps": steps, "bc_scope": scope, "config": CONFIG})

    if 1 in steps:
        step1_acquire()
    if 2 in steps:
        step2_cure(scope)
    if 3 in steps:
        verdict = step3_gate()
        if not verdict["passed"]:
            # Pre-registered: a failed gate STOPS the night. The
            # receipts carry the numbers; relaunching consol2 anyway is
            # the operator's explicit `--skip-to 4`, never automatic.
            print("[night2] stopped at the gate; consol2 not launched.",
                  flush=True)
            return 2
    if 4 in steps:
        rc = step4_consol2()
        if rc != 0:
            return rc

    _receipt({"type": "night2_complete", "steps": steps})
    print("[night2] complete.", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate every step's assembly (checkpoint "
                         "loads, SIL persistence verdict, eval commands, "
                         "seeding plan, consol2 --dry-run) without "
                         "rollouts or training.")
    ap.add_argument("--skip-to", type=int, default=1, metavar="N",
                    help="Resume at step N (1=SIL acquisition, 2=BC "
                         "cure, 3=gate, 4=consol2) off the prior steps' "
                         "on-disk artifacts. --skip-to 4 is the explicit "
                         "operator override of a failed gate.")
    ap.add_argument("--bc-scope", type=str, default=None,
                    choices=list(_BC_SCOPES),
                    help="Cure scope: 'actor' (registered default — "
                         "actor head only) or 'actor+trunk' (the "
                         "fallback if actor-only is too weak; critic "
                         "stays frozen either way).")
    args = ap.parse_args()
    if args.dry_run:
        return dry_run()
    # Free-disk floor (src/utils/disk_floor.py, shared with
    # run_online_campaign.py and go_explore_chain.py): night 2 runs SIL
    # collection through consol2 hands-off for hours, so a volume that
    # fills mid-run must refuse before step 1 rather than degrade into
    # swallowed checkpoint-save warnings.
    _disk_reason = disk_floor_breach(REPO)
    if _disk_reason:
        raise SystemExit(f"[night2] REFUSING to start: {_disk_reason}")
    return run(skip_to=args.skip_to, bc_scope=args.bc_scope)


if __name__ == "__main__":
    os.chdir(REPO)
    sys.exit(main())
