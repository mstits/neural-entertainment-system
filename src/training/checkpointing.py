"""Checkpoint save / load / archive / rotate.

Pulled out of Trainer so the IO-sensitive surfaces (atomic
tmp+rename, fsync, mtime-based rotation, run archival) live in
their own module and can be exercised without spinning up a full
Trainer or any RL machinery. All functions are pure — they take
the GA, curriculum, etc. as arguments instead of `self.*`.

Atomicity guarantees that the trainer relies on:
  * `save_checkpoint_atomic` writes to `<path>.tmp`, fsyncs, then
    renames over the target. A mid-write Ctrl-C / OOM leaves
    either the old or the new file visible, never a truncated
    half-written one.
  * `archive_previous_run` is best-effort: any failure is logged
    and swallowed — archival must never block training startup.

Checkpoint rotation sorts by **mtime, not filename**: a fresh run
that restarts the gen counter at 0 must not have its checkpoints
pruned by older, higher-named cohorts (real data-loss case —
tile-mode runs writing gen_00010 were being deleted because a
pixel-era gen_01094 was alphabetically higher).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional


log = logging.getLogger(__name__)


def save_checkpoint_atomic(
    path: Path,
    ga: Any,
    curriculum: Any,
    checkpoint_dir: Path,
) -> None:
    """Atomic-write a GA checkpoint + curriculum.json.

    Uses `<path>.tmp` + fsync + rename for the GA file, same dance
    for `curriculum.json` in `checkpoint_dir`. Either both writes
    succeed atomically or both leave the old state intact.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    ga.save_checkpoint(str(tmp_path))
    try:
        fd = os.open(str(tmp_path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # fsync is best-effort — some filesystems error out, but the
        # rename alone still gives a consistent file.
        pass
    os.replace(str(tmp_path), str(path))

    curr_path = checkpoint_dir / "curriculum.json"
    curr_tmp = curr_path.with_suffix(curr_path.suffix + ".tmp")
    with open(curr_tmp, "w") as f:
        json.dump(curriculum.state_dict(), f)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(str(curr_tmp), str(curr_path))


def maybe_export_elite_to_coreml(
    checkpoint_path: Path,
    ga: Any,
    make_network: Callable[[], Any],
    num_actions: int,
    is_tile_mode: bool,
) -> None:
    """Best-effort CoreML export of the elite policy.

    Skipped when:
      * NES_DISABLE_COREML_EXPORT=1 in env
      * the run is in tile mode (14k-param MLP is fast enough on
        CPU; ANE acceleration is irrelevant, and the export was
        eating ~13% of trainer wall-time on tile runs)

    Output is `checkpoint_path.with_suffix(".mlpackage")`. The
    replay viewer prefers CoreML/ANE at batch=1 (8× faster than
    PyTorch MPS per scripts/bench_coreml_ane.py); exporting here
    amortizes the ~100ms JIT+convert cost out of the interactive
    replay-launch path.
    """
    if os.environ.get("NES_DISABLE_COREML_EXPORT") == "1":
        log.debug("CoreML export disabled via NES_DISABLE_COREML_EXPORT=1")
        return
    if is_tile_mode:
        log.debug("CoreML export skipped for tile mode (network too small to benefit)")
        return
    try:
        elite = ga.best_genome()
        if elite is None or elite.state_dict is None:
            return
        from src.models.coreml_export import maybe_export
        net = make_network()
        net.load_state_dict(elite.state_dict, strict=False)
        mlpath = checkpoint_path.with_suffix(".mlpackage")
        maybe_export(net, str(mlpath), num_actions=num_actions)
        del net
        import gc as _gc
        _gc.collect()
    except Exception as exc:
        log.debug("checkpoint CoreML export skipped: %s", exc)


def rotate_old_checkpoints(checkpoint_dir: Path, keep_last: int) -> None:
    """Delete oldest checkpoints + paired .mlpackages beyond `keep_last`.

    Sorted by **mtime, not filename**: a fresh run that restarts gen
    counters at 0 must not have its checkpoints pruned by older
    higher-named cohorts still in the dir. mtime is the only signal
    that tracks "most recent" across resumes.

    `keep_last <= 0` is a no-op (used by tests that want to inspect
    the full history).
    """
    if keep_last <= 0:
        return
    ckpts = sorted(
        Path(checkpoint_dir).glob("gen_*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if len(ckpts) <= keep_last:
        return
    for old in ckpts[:-keep_last]:
        try:
            old.unlink()
        except OSError:
            pass
        mlpath = old.with_suffix(".mlpackage")
        if mlpath.exists():
            try:
                shutil.rmtree(mlpath)
            except OSError:
                pass


def find_latest_checkpoint(checkpoint_dir) -> Optional[Path]:
    """Return the newest gen_*.pt in `checkpoint_dir`, or None if empty.

    Sorts by mtime so a manually-copied or misnamed file (e.g.
    `gen_archive.pt` from a different run) doesn't lexicographically
    sort past the actual newest checkpoint and get picked as the
    resume target. Matches the mtime ordering `rotate_old_checkpoints`
    uses.
    """
    d = Path(checkpoint_dir)
    if not d.exists():
        return None
    ckpts = sorted(d.glob("gen_*.pt"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def archive_previous_run(
    checkpoint_dir: Path,
    metrics_path: Path,
    game_profile: dict,
) -> Optional[Path]:
    """Snapshot a finished/abandoned run into `runs/<timestamp>/`.

    Captures: metrics.jsonl (moved — saves IO + ensures a fresh
    empty file next run), curriculum.json, bc_success_cache.npz,
    run.log, the latest gen checkpoint, and a serialized copy of
    the active game profile YAML so the archive is self-describing.

    No-op when the metrics file is empty or missing (first run of a
    fresh dir has nothing to archive). Returns the archive Path on
    success, None otherwise. Errors are logged but never propagate
    — archival must not block training startup.
    """
    try:
        if not metrics_path.exists() or metrics_path.stat().st_size == 0:
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = Path(checkpoint_dir) / "runs" / ts
        archive.mkdir(parents=True, exist_ok=True)
        # Move metrics (don't copy — saves IO and gives the next run
        # a fresh empty file).
        shutil.move(str(metrics_path), str(archive / "metrics.jsonl"))
        # Copy (not move) the rest — they're either still in use or
        # the user might want them in checkpoint_dir for resume.
        for src_name in ("curriculum.json", "bc_success_cache.npz", "run.log"):
            src = Path(checkpoint_dir) / src_name
            if src.exists():
                try:
                    shutil.copy2(str(src), str(archive / src_name))
                except Exception:
                    pass
        # Latest gen checkpoint (numerically highest gen_NNNNN.pt).
        gens = sorted(Path(checkpoint_dir).glob("gen_*.pt"))
        if gens:
            try:
                shutil.copy2(str(gens[-1]), str(archive / gens[-1].name))
            except Exception:
                pass
        # Serialize the active profile so the archive is self-
        # describing.
        try:
            import yaml as _yaml
            with open(archive / "game_profile.yaml", "w") as f:
                _yaml.safe_dump(game_profile, f, sort_keys=False)
        except Exception:
            pass
        log.info(
            "[archive] previous run snapshotted to %s "
            "(metrics + curriculum + bc cache + latest checkpoint + profile)",
            archive,
        )
        return archive
    except Exception as exc:
        log.warning("[archive] failed to archive previous run: %s", exc)
        return None
