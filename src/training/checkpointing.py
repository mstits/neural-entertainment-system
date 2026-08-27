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

Winner retention (the "watch it win" path): a curriculum run can
self-collapse (greedy clear rate 0.88 -> 0.00), leaving only failing
checkpoints under normal rotation. `save_winner` pins the best-ever
policy into `winners/best.pt` (with a self-describing sidecar), and
that subtree is deliberately excluded from rotation so a later
collapse can never prune the one checkpoint that actually wins.
`find_playable_checkpoint` prefers that winner, so demo/eval always
play a real win rather than whatever the latest (possibly drifted)
checkpoint happens to be.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional


log = logging.getLogger(__name__)

# Subdirectory (under a per-game checkpoint dir) that holds the
# best-ever policy. Kept out of the gen_*.pt / vanilla_ppo_iter_*.pt
# rotation namespace on purpose — see rotate_old_checkpoints.
WINNERS_SUBDIR = "winners"
_WINNER_FILENAME = "best.pt"
_WINNER_META_FILENAME = "best.json"


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
    # `winners/best.pt` is a retained best-ever policy — never a rotation
    # candidate. The glob is non-recursive so it can't reach into
    # winners/ today, but filter defensively so a future naming change
    # can't accidentally prune the one checkpoint that wins.
    ckpts = sorted(
        (
            p for p in Path(checkpoint_dir).glob("gen_*.pt")
            if WINNERS_SUBDIR not in p.parts
        ),
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


# --------------------------------------------------------------------------- #
# Winner retention + playable-checkpoint selection
# --------------------------------------------------------------------------- #


def _fsync_best_effort(path: Path) -> None:
    """fsync a just-written file, swallowing filesystems that refuse."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _git_sha() -> Optional[str]:
    """Best-effort short git SHA of the working tree, or None.

    Purely for self-describing winner sidecars ("which code produced this
    win"). Never raises — a detached/absent git is fine.
    """
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        sha = out.stdout.strip()
        return sha or None
    except Exception:
        return None


def winner_paths(out_dir: str | Path) -> tuple[Path, Path]:
    """Return `(best.pt, best.json)` paths under `<out_dir>/winners/`.

    Neither is guaranteed to exist — callers check. `out_dir` is a
    per-game checkpoint directory (e.g. `checkpoints/super_mario_bros`).
    """
    wdir = Path(out_dir) / WINNERS_SUBDIR
    return wdir / _WINNER_FILENAME, wdir / _WINNER_META_FILENAME


def load_winner_meta(out_dir: str | Path) -> Optional[dict]:
    """Return the retained winner's sidecar metadata dict, or None.

    Returns None when no winner has been saved or the sidecar is
    unreadable/corrupt (treated as "no recorded metric" so the next
    save_winner overwrites rather than refusing forever).
    """
    _, meta_path = winner_paths(out_dir)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return None


def _read_best_pt_metric_value(out_dir: str | Path) -> Optional[float]:
    """Return the `metric_value` embedded in `winners/best.pt`, or None.

    `save_winner` embeds the metric a checkpoint was saved with directly
    in the `.pt` blob (see below), which makes the blob its own ground
    truth independent of the `best.json` sidecar. Any read failure
    (file absent, corrupt torch blob, missing/non-numeric field, NaN)
    returns None so callers fall back to sidecar-only comparison.
    """
    best_path, _ = winner_paths(out_dir)
    if not best_path.exists():
        return None
    try:
        import torch

        # weights_only=True: `save_winner` (below) is the only writer of
        # `winners/best.pt` and puts nothing in it beyond plain tensors
        # and str/int/float primitives (net_state_dict, iter, metric_name,
        # metric_value) — no custom class instances, so the restricted
        # unpickler covers it. Verified by loading every winners/best.pt
        # under checkpoints/ on disk with this flag before it shipped.
        blob = torch.load(str(best_path), map_location="cpu", weights_only=True)
        val = float(blob.get("metric_value", float("-inf")))
    except Exception:
        return None
    return val if math.isfinite(val) else None


def save_winner(
    net_state: dict,
    game: str,
    metric_value: float,
    out_dir: str | Path,
    *,
    metric_name: str = "clear_rate",
    source_iter: Optional[int] = None,
) -> bool:
    """Pin `net_state` as the game's best-ever policy — only if it wins by more.

    Writes `<out_dir>/winners/best.pt` (a `{"net_state_dict": ...}` blob
    that loads identically to a `vanilla_ppo_iter_*.pt` checkpoint) plus a
    `best.json` sidecar (game, metric name+value, source iter, git SHA,
    timestamp) — but ONLY when `metric_value` strictly beats the stored
    best. This is what keeps the flagship "watch it win" path pointed at a
    real win even after a curriculum run self-collapses.

    Returns True if a new winner was written, False if the existing winner
    was as-good-or-better (or `metric_value` is not a finite number — a NaN
    from a degenerate eval must never dislodge a real win).

    Excluded from rotation by construction: the file lives in `winners/`,
    which `rotate_old_checkpoints` skips. Atomic tmp+fsync+rename for both
    files so an interrupted save leaves the previous winner intact.

    `best.pt` and `best.json` are written via two independent atomic
    renames (pt first, then the sidecar) rather than one transaction, so
    a kill/OOM between them can leave the sidecar reporting a stale,
    lower metric than what `best.pt` actually holds. Since `best.pt`
    embeds its own metric_value, the overwrite gate below reconciles
    the sidecar's value against the checkpoint's own embedded value
    (taking the max) so a lagging sidecar can never let a worse
    checkpoint pass the gate and clobber a genuinely better one. A
    sidecar file that exists but fails to parse is still treated as "no
    recorded metric" (see `load_winner_meta`) and is not reconciled
    against `best.pt` — that corrupt-file case is unrelated to the
    kill-window race and must stay overwritable or a genuinely corrupt
    sidecar could wedge winner retention forever.
    """
    try:
        metric_value = float(metric_value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(metric_value):
        return False

    _, meta_path = winner_paths(out_dir)
    sidecar_present = meta_path.exists()
    prev = load_winner_meta(out_dir)

    prev_val = float("-inf")
    if prev is not None:
        try:
            prev_val = float(prev.get("metric_value", float("-inf")))
        except (TypeError, ValueError):
            prev_val = float("-inf")  # corrupt value → allow overwrite

    # Reconcile against best.pt's own embedded metric, except when the
    # sidecar exists but failed to parse (prev is None with
    # sidecar_present True) — that corrupt-file path is intentionally
    # left alone so it stays overwritable rather than resurrecting a
    # stale/higher value from the checkpoint blob.
    if prev is not None or not sidecar_present:
        pt_val = _read_best_pt_metric_value(out_dir)
        if pt_val is not None and pt_val > prev_val:
            prev_val = pt_val

    if prev_val > metric_value:
        return False
    if prev_val == metric_value:
        # THE CEILING LOCK. `>=` here reads as harmless anti-churn until the
        # selection metric is one that SATURATES. `entrance_trailing_rate` is
        # successes/30, maximum exactly 1.0: once a run records 1.0 the gate
        # above became mathematically unsatisfiable and the winner froze for
        # the rest of training. Measured on the v27/v28 cohort: the selector's
        # last save lands at iter 50-120 of 250 (median 65), and two of eight
        # runs froze at a recorded 1.0000 at iter 90 with 160 iterations still
        # to come — deterministic, one-directional under-selection, not noise.
        #
        # The fix is the tie-break both v27 and v28 REGISTERED and neither
        # implemented: "ties -> later iter". A tie still never dislodges a
        # better metric, and a tie with no iteration information (or an older
        # one) still keeps the incumbent, so the anti-churn property holds
        # everywhere it was actually doing work.
        prev_iter = None
        if prev is not None:
            try:
                prev_iter = int(prev.get("source_iter"))
            except (TypeError, ValueError):
                prev_iter = None
        if source_iter is None or prev_iter is None or int(source_iter) <= prev_iter:
            return False

    import torch

    best_path, meta_path = winner_paths(out_dir)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    # Detach + CPU so the winner is portable and never pins a device
    # tensor (mirrors the trainer's own vanilla_ppo checkpoint format).
    cpu_state = {
        k: (v.detach().cpu() if hasattr(v, "detach") else v)
        for k, v in net_state.items()
    }
    tmp_pt = best_path.with_suffix(best_path.suffix + ".tmp")
    torch.save(
        {
            "net_state_dict": cpu_state,
            "iter": source_iter,
            "metric_name": metric_name,
            "metric_value": metric_value,
        },
        str(tmp_pt),
    )
    _fsync_best_effort(tmp_pt)
    os.replace(str(tmp_pt), str(best_path))

    # Sidecar last: it is the source of truth for "what metric best.pt
    # holds", so it must never claim a value the .pt doesn't back.
    meta = {
        "game": game,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "source_iter": source_iter,
        "git_sha": _git_sha(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    tmp_json = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with open(tmp_json, "w") as f:
        json.dump(meta, f, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(str(tmp_json), str(meta_path))

    log.info(
        "[winner] %s new best %s=%.4f (iter %s) -> %s",
        game, metric_name, metric_value, source_iter, best_path,
    )
    return True


def load_winner(game: str, out_dir: str | Path) -> Optional[dict]:
    """Return the retained winner checkpoint blob, or None if absent/unreadable.

    The blob is the same shape as a `vanilla_ppo_iter_*.pt` checkpoint
    (`{"net_state_dict": ..., "iter": ..., "metric_value": ...}`), so a
    caller loads its weights via `blob["net_state_dict"]`.
    """
    best_path, _ = winner_paths(out_dir)
    if not best_path.exists():
        return None
    try:
        import torch

        # weights_only=True: same blob format as `_read_best_pt_metric_value`
        # above — tensors + str/int/float only, no custom types — so the
        # restricted unpickler is sufficient here too.
        return torch.load(str(best_path), map_location="cpu", weights_only=True)
    except Exception as exc:  # corrupt/partial file — don't crash the caller
        log.warning("[winner] failed to load %s: %s", best_path, exc)
        return None


def _iter_num(p: Path) -> int:
    """Trailing integer of a `*_NNNNN.pt` stem, or -1 if unparseable."""
    try:
        return int(p.stem.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return -1


def find_latest_trained_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    """Newest trained-policy checkpoint in `checkpoint_dir`, or None.

    Prefers the highest-numbered `vanilla_ppo_iter_*.pt` (the current
    PPO/tile format), falling back to the newest `gen_*.pt` by mtime for
    legacy GA runs. This is the "latest" that demo/eval mean when a user
    passes `--latest` and the plain resume default.
    """
    d = Path(checkpoint_dir)
    if not d.exists():
        return None
    vppo = sorted(d.glob("vanilla_ppo_iter_*.pt"), key=_iter_num)
    if vppo:
        return vppo[-1]
    return find_latest_checkpoint(d)


def _row_is_honest(row: dict) -> bool:
    """True when an `eval.jsonl` row was drawn under a perturbed protocol.

    The ledger separates LEARNED (honest: sticky-actions + start-jitter) from
    EXHIBITION (deterministic replay). `eval_game.py` appends BOTH to the same
    `eval.jsonl`, so a selector that reads only `clear_rate` will happily rank
    a deterministic n=10 replay at 1.0 above an honest n=30 at 0.633 — which
    is the ledger's own EXHIBITION/LEARNED boundary being crossed in code
    rather than in prose. Measured across the checkpoint dirs in this repo,
    that is what the selector was doing in 8 of 62.

    "Honest" here means the ENVIRONMENT was perturbed — sticky-actions or
    start-jitter — which is what separates the two ledgers. It deliberately
    does NOT key on `stochastic`, which `run_consumes_randomness` also sets
    for a sampled action draw on an unperturbed environment: a sampled
    deterministic-env run is not a replay, but it is not the honest protocol
    either, and it must not be promoted as one. `stochastic` is consulted
    only as a legacy fallback for rows written before the perturbation
    fields existed.

    Rows carrying none of the three are treated as NOT honest — they are
    unlabelled, and an unlabelled number must never outrank a labelled one.
    """
    saw_field = False
    for key in ("sticky_prob", "start_jitter"):
        if key not in row:
            continue
        saw_field = True
        try:
            if float(row[key]) > 0.0:
                return True
        except (TypeError, ValueError):
            continue
    if saw_field:
        return False
    stoch = row.get("stochastic")
    return stoch if isinstance(stoch, bool) else False


def _best_from_eval_log(checkpoint_dir: Path) -> Optional[Path]:
    """Highest-clear_rate checkpoint recorded in `eval.jsonl`, or None.

    Scans the per-game eval log for the row with the greatest `clear_rate`
    whose `checkpoint` file still exists on disk. Returns None when the log
    is missing, holds only error rows, or nothing ever cleared (clear_rate
    0.0) — in which case a stale early iter is no more "playable" than the
    latest, so the caller should fall through to the freshest checkpoint
    instead of pinning it.

    PROTOCOL-AWARE: when the log holds any honest (perturbed-protocol) row,
    ONLY honest rows are eligible. A deterministic replay is EXHIBITION under
    the ledger and may not be promoted over a LEARNED measurement no matter
    how high its rate. When the log holds no honest row at all the whole log
    is eligible, so pre-protocol histories keep their old behaviour.

    Ordering within the eligible set is `(clear_rate, n_episodes, timestamp)`:
    the rate first, then the larger sample — a rate measured over 30 episodes
    is a better estimate than the same rate over 5 — and newest last.
    """
    log_path = Path(checkpoint_dir) / "eval.jsonl"
    if not log_path.exists():
        return None
    best_ckpt: Optional[Path] = None
    best_key: Optional[tuple[float, float, float]] = None
    honest_only = False
    try:
        # One cheap pre-pass to decide eligibility, so the honest/EXHIBITION
        # split is a property of the whole log rather than of row order.
        for line in log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("clear_rate") is None or row.get("checkpoint") is None:
                continue
            if _row_is_honest(row):
                honest_only = True
                break
    except OSError:
        return None
    try:
        for line in log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            cr, ckpt = row.get("clear_rate"), row.get("checkpoint")
            if cr is None or ckpt is None:
                continue
            if honest_only and not _row_is_honest(row):
                continue
            try:
                cr = float(cr)
            except (TypeError, ValueError):
                continue
            p = Path(ckpt)
            if not p.exists():
                continue
            # Timestamps are written as ISO strings (save_winner uses
            # datetime.isoformat()), so float(ts) raises ValueError — which is
            # NOT caught by the surrounding `except OSError` and crashed
            # make demo/eval's latest-checkpoint fallback. Coerce defensively;
            # clear_rate is the primary ordering key, so a 0.0 tiebreak on an
            # unparseable timestamp is harmless.
            ts_raw = row.get("timestamp", 0.0)
            try:
                ts = float(ts_raw)
            except (TypeError, ValueError):
                ts = 0.0
            # Prefer the better-sampled of two equal rates. `n_episodes` is
            # the requested count; `n_episodes_delivered` (newer receipts) is
            # what actually ran and wins when present.
            n_raw = row.get("n_episodes_delivered", row.get("n_episodes", 0))
            try:
                n_eps = float(n_raw)
            except (TypeError, ValueError):
                n_eps = 0.0
            key = (cr, n_eps, ts)
            if best_key is None or key > best_key:
                best_key, best_ckpt = key, p
    except OSError:
        return None
    if best_ckpt is not None and best_key is not None and best_key[0] > 0.0:
        return best_ckpt
    return None


def find_playable_checkpoint(
    game: str, checkpoint_dir: str | Path
) -> Optional[Path]:
    """Pick the checkpoint most likely to actually WIN, for demo/eval.

    Preference order:
      1. `winners/best.pt` — the retained best-ever policy (never rotated).
      2. The highest-`clear_rate` checkpoint from `eval.jsonl` history
         (only when something actually cleared).
      3. The latest trained checkpoint (freshest weights).

    Returns None only when the directory holds no usable checkpoint at all.
    `game` is accepted for symmetry/logging with the rest of the winner
    API; selection is driven entirely by `checkpoint_dir`.
    """
    d = Path(checkpoint_dir)
    best_path, _ = winner_paths(d)
    if best_path.exists():
        return best_path
    from_eval = _best_from_eval_log(d)
    if from_eval is not None:
        return from_eval
    return find_latest_trained_checkpoint(d)
