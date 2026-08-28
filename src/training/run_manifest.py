"""Per-run reproducibility manifest.

Each training run writes a `run_manifest.json` into its checkpoint dir
capturing exactly what's needed to reproduce it: ROM + its MD5, start
state, seed, the git commit of the code, the load-bearing
hyperparameters, and the resolved dependency environment (torch/
torchvision/numpy + the nes_core extension's own build identity).
Combined with the code at that commit and the metrics.jsonl the run
emits, a result becomes reproducible and citable rather than "it
worked on my machine that one time."

Kept deliberately small — provenance + config only. The achieved
results live in metrics.jsonl / eval.jsonl (the scoreboard joins them).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Optional

# The hyperparameters that actually change learning behavior — worth
# pinning so a manifest fully specifies the run alongside the seed+commit.
_PINNED_HYPERPARAMS = (
    "lr", "gamma", "gae_lambda", "ppo_clip_eps", "value_coef",
    "entropy_coef", "grad_clip", "rollout_steps", "steps",
    "ppo_minibatch_size", "value_loss", "rnd_intrinsic_coef",
    "rnd_loss_coef", "tile_frame_stack",
)

# Packages whose resolved version can move a banked run's numbers even
# with the git commit + seed held fixed — a `pip install` months apart
# can silently resolve a different torch build (and therefore a
# different MPS backend/kernel set) against the same commit. A
# targeted list, not a full `pip freeze`: this manifest is meant to
# stay human-readable at a glance, and torch/torchvision/numpy are the
# only third-party packages with a direct line to numeric outcomes
# (loss values, action distributions, timing). Everything else in the
# venv is captured by the git commit's requirements.txt/lockfile, which
# already pins it exactly — duplicating that whole freeze here would
# just be a second, driftable copy of the same information.
_PINNED_PACKAGES = ("torch", "torchvision", "numpy")


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip() or None
    except Exception:
        return None


def _package_version(name: str) -> str:
    """Resolved version of an installed package, or `"unknown"`.

    Never raises: a missing package, an editable/odd distribution
    layout, or any other lookup hiccup must degrade to `"unknown"`
    rather than take down the training run that calls this at startup.
    """
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return "unknown"


def _nes_core_build_id() -> dict:
    """Identify the nes_core extension actually loaded, or `"unknown"`
    fields if it can't be determined.

    Mirrors `scripts/go_explore_solve.py`'s `_core_build_id()` (the
    `hw_provenance` precedent for solver runs) rather than inventing a
    second scheme: the PyPI-style dist version is static across local
    `maturin` rebuilds, so the compiled `.so`'s own digest is what
    actually pins the binary a run executed against — the same
    maturin-doesn't-update-the-venv-.so hazard that precedent guards
    against.
    """
    info = {"dist_version": "unknown", "module": "unknown",
            "sha256_16": "unknown"}
    try:
        import importlib.metadata as md
        info["dist_version"] = md.version("nes_core")
    except Exception:
        pass
    try:
        import nes_core
        so = Path(nes_core.__file__).with_name("nes_core.abi3.so")
        if not so.exists():
            so = Path(nes_core.nes_core.__file__)
        info["module"] = so.name
        info["sha256_16"] = hashlib.sha256(so.read_bytes()).hexdigest()[:16]
    except Exception:
        pass
    return info


def dependency_snapshot() -> dict:
    """Resolved dependency environment for this process, best-effort.

    The one training-run input the manifest didn't previously capture:
    ROM/seed/commit/hyperparams pin *what* ran, but not *which*
    torch/numpy build it ran against. Cheap (runs once at process
    start; no subprocess calls) and never raises — each lookup below
    is independently guarded so one missing/odd package degrades to
    `"unknown"` in its own field instead of losing the rest of the
    snapshot.
    """
    snapshot = {name: _package_version(name) for name in _PINNED_PACKAGES}
    try:
        snapshot["nes_core"] = _nes_core_build_id()
    except Exception:
        snapshot["nes_core"] = {"dist_version": "unknown", "module": "unknown",
                                 "sha256_16": "unknown"}
    return snapshot


def write_run_manifest(
    checkpoint_dir,
    *,
    game: Optional[str],
    rom_path: str,
    start_state_path: Optional[str],
    seed: Optional[int],
    profile: dict,
    num_envs: int,
    frame_skip: int,
    rom_md5: Optional[str] = None,
    created_at: Optional[float] = None,
) -> Path:
    """Write `<checkpoint_dir>/run_manifest.json` atomically; return its path.

    `seed` and `rom_md5` are the two provenance fields that make a run
    reproducible: the seed pins every RNG, and the MD5 pins the exact ROM
    dump (a re-dumped/patched ROM shifts RAM addresses and silently breaks
    reward functions). Both are recorded verbatim; a caller that leaves
    them as ``None`` records ``null`` (still a valid, if weaker, manifest).
    """
    rl = profile.get("reinforce", {}) or {}
    path = Path(checkpoint_dir) / "run_manifest.json"
    if rom_md5 is None:
        # A same-run second writer (e.g. vanilla_ppo's own
        # CheckpointManager.write_manifest, which has no rom_md5 kwarg to
        # forward) can call back into this function after the launcher
        # already wrote the correct MD5 to this same path, atomically
        # clobbering it with `None` moments later. Preserve the prior
        # value for the same ROM instead of losing it; a path with no
        # prior manifest (or one for a different ROM) still records
        # `null` as documented above.
        try:
            _prior = json.loads(path.read_text())
            if _prior.get("rom_path") == str(rom_path):
                rom_md5 = _prior.get("rom_md5")
        except Exception:
            pass
    manifest = {
        "game": game,
        "rom_path": str(rom_path),
        "rom_md5": rom_md5,
        "start_state_path": str(start_state_path) if start_state_path else None,
        "seed": seed,
        "git_commit": _git_commit(),
        "created_at": created_at if created_at is not None else time.time(),
        "trainer_mode": rl.get("trainer_mode", "vanilla_ppo"),
        "encoder": rl.get("encoder", "nature_dqn"),
        "device": rl.get("device"),
        "num_envs": int(num_envs),
        "frame_skip": int(frame_skip),
        "hyperparams": {k: rl[k] for k in _PINNED_HYPERPARAMS if k in rl},
        "dependencies": dependency_snapshot(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(path)
    return path


def update_run_manifest_redo(
    checkpoint_dir,
    *,
    redo_tau: float,
    cum_recycled: int,
    recycle_events: int,
    first_recycle_iter: Optional[int],
    median_agree: Optional[float],
    median_dose_frac: Optional[float],
    distinct_fc2_indices: int,
) -> Path:
    """Patch `run_manifest.json` with ReDo's own summary telemetry
    (V31_REDO_SURGICAL_2026-08-27.md §12 item 6), best-effort, after the
    run ends (success, VOID abort, or crash — the caller decides when).

    `write_run_manifest` runs BEFORE training starts and cannot know
    these; `scripts/redo_arm_gate.py` already derives the same numbers
    from `run.log` at verdict time — this patch is the convenience copy
    into the manifest a reader would check first, not the source of
    truth for the gate (the gate always re-derives from the log).
    Never raises: provenance is best-effort and must not mask whatever
    exception (if any) the trainer itself just raised.
    """
    path = Path(checkpoint_dir) / "run_manifest.json"
    try:
        manifest = json.loads(path.read_text()) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        manifest = {}
    manifest["redo_tau"] = redo_tau
    manifest["redo_cum_recycled"] = int(cum_recycled)
    manifest["redo_recycle_events"] = int(recycle_events)
    manifest["redo_first_recycle_iter"] = first_recycle_iter
    manifest["redo_median_agree"] = median_agree
    manifest["redo_median_dose_frac"] = median_dose_frac
    manifest["redo_distinct_fc2_indices"] = int(distinct_fc2_indices)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(path)
    return path
