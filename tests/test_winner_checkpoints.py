"""Tests for winner retention + playable-checkpoint selection.

The flagship "watch it win" path depends on these invariants: a
better metric must pin a new winner, a worse one must not dislodge
it, the winner must survive a rotation that prunes everything else,
and find_playable_checkpoint must prefer the winner over the (possibly
collapsed) latest checkpoint. Each is locked down here without a
Trainer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from src.training.checkpointing import (
    find_playable_checkpoint,
    find_latest_trained_checkpoint,
    load_winner,
    load_winner_meta,
    rotate_old_checkpoints,
    save_winner,
    winner_paths,
)


def _state(bias: float = 0.0) -> dict:
    """A tiny distinguishable state_dict (one tensor) for round-tripping."""
    return {"w": torch.tensor([bias, bias + 1.0])}


def _write_iter_ckpt(d: Path, it: int, payload: dict | None = None) -> Path:
    p = d / f"vanilla_ppo_iter_{it:05d}.pt"
    torch.save({"iter": it, "net_state_dict": payload or _state(float(it))}, str(p))
    return p


# --------------------------------------------------------------------------- #
# save_winner: overwrite-only-on-improvement
# --------------------------------------------------------------------------- #


def test_save_winner_writes_first_winner(tmp_path: Path) -> None:
    wrote = save_winner(_state(1.0), "mario", 0.5, tmp_path, source_iter=10)
    assert wrote is True
    best_pt, best_json = winner_paths(tmp_path)
    assert best_pt.exists()
    assert best_json.exists()
    meta = load_winner_meta(tmp_path)
    assert meta["game"] == "mario"
    assert meta["metric_name"] == "clear_rate"
    assert meta["metric_value"] == 0.5
    assert meta["source_iter"] == 10


def test_save_winner_better_metric_overwrites(tmp_path: Path) -> None:
    assert save_winner(_state(1.0), "mario", 0.5, tmp_path, source_iter=10) is True
    assert save_winner(_state(2.0), "mario", 0.88, tmp_path, source_iter=42) is True
    meta = load_winner_meta(tmp_path)
    assert meta["metric_value"] == 0.88
    assert meta["source_iter"] == 42
    # The retained weights are the better run's, not the first.
    blob = load_winner("mario", tmp_path)
    assert torch.equal(blob["net_state_dict"]["w"], _state(2.0)["w"])


def test_save_winner_worse_metric_does_not_overwrite(tmp_path: Path) -> None:
    assert save_winner(_state(2.0), "mario", 0.88, tmp_path, source_iter=42) is True
    # A later, worse (collapsed) policy must NOT dislodge the win.
    assert save_winner(_state(9.0), "mario", 0.00, tmp_path, source_iter=193) is False
    meta = load_winner_meta(tmp_path)
    assert meta["metric_value"] == 0.88
    assert meta["source_iter"] == 42
    blob = load_winner("mario", tmp_path)
    assert torch.equal(blob["net_state_dict"]["w"], _state(2.0)["w"])


def test_save_winner_worse_metric_never_overwrites_even_from_a_later_iter(
    tmp_path: Path,
) -> None:
    """The primary guarantee is unchanged: metric first, always."""
    assert save_winner(_state(1.0), "mario", 0.88, tmp_path, source_iter=10) is True
    assert save_winner(_state(2.0), "mario", 0.87, tmp_path, source_iter=999) is False
    assert load_winner_meta(tmp_path)["source_iter"] == 10


def test_save_winner_equal_metric_prefers_the_later_iter(tmp_path: Path) -> None:
    """Ties go to the later iteration — the CEILING LOCK fix.

    `>=` here was harmless anti-churn right up until the selection metric
    saturated. `entrance_trailing_rate` is successes/30, max exactly 1.0: the
    first iteration to record 1.0 made the old gate unsatisfiable and froze
    the winner for the rest of the run (v27 seed 2 and v28 seed 3 both froze
    at iter 90 of 250). "Ties -> later iter" is the rule both campaigns
    registered and neither implemented.
    """
    assert save_winner(_state(1.0), "mario", 0.5, tmp_path, source_iter=10) is True
    assert save_winner(_state(2.0), "mario", 0.5, tmp_path, source_iter=20) is True
    assert load_winner_meta(tmp_path)["source_iter"] == 20


def test_save_winner_equal_metric_keeps_incumbent_without_a_later_iter(
    tmp_path: Path,
) -> None:
    """Anti-churn still holds wherever it was actually doing work: a tie with
    an equal iter, an earlier iter, or no iteration information at all keeps
    the incumbent."""
    assert save_winner(_state(1.0), "mario", 0.5, tmp_path, source_iter=10) is True
    assert save_winner(_state(2.0), "mario", 0.5, tmp_path, source_iter=10) is False
    assert save_winner(_state(3.0), "mario", 0.5, tmp_path, source_iter=5) is False
    assert save_winner(_state(4.0), "mario", 0.5, tmp_path) is False
    assert load_winner_meta(tmp_path)["source_iter"] == 10


def test_save_winner_rejects_nan(tmp_path: Path) -> None:
    assert save_winner(_state(2.0), "mario", 0.88, tmp_path, source_iter=42) is True
    # A NaN from a degenerate eval must never dislodge a real win.
    assert save_winner(_state(9.0), "mario", float("nan"), tmp_path) is False
    assert load_winner_meta(tmp_path)["metric_value"] == 0.88


def test_save_winner_rejects_non_numeric(tmp_path: Path) -> None:
    assert save_winner(_state(1.0), "mario", "not-a-number", tmp_path) is False
    assert not winner_paths(tmp_path)[0].exists()


def test_save_winner_records_git_sha_and_timestamp(tmp_path: Path) -> None:
    save_winner(_state(1.0), "contra", 0.7, tmp_path, source_iter=30)
    meta = load_winner_meta(tmp_path)
    # git_sha is best-effort (may be None off a repo) but the key must exist.
    assert "git_sha" in meta
    assert "timestamp" in meta and isinstance(meta["timestamp"], str)


def test_save_winner_custom_metric_name(tmp_path: Path) -> None:
    save_winner(_state(1.0), "mario", 3161.0, tmp_path,
                metric_name="max_depth", source_iter=7)
    meta = load_winner_meta(tmp_path)
    assert meta["metric_name"] == "max_depth"
    assert meta["metric_value"] == 3161.0


def test_load_winner_absent_returns_none(tmp_path: Path) -> None:
    assert load_winner("mario", tmp_path) is None
    assert load_winner_meta(tmp_path) is None


def test_save_winner_overwrites_corrupt_sidecar(tmp_path: Path) -> None:
    """A corrupt sidecar reads as 'no recorded metric' so the next save
    reclaims the winner rather than refusing forever."""
    save_winner(_state(1.0), "mario", 0.5, tmp_path, source_iter=10)
    _, best_json = winner_paths(tmp_path)
    best_json.write_text("{not valid json")
    assert save_winner(_state(2.0), "mario", 0.1, tmp_path, source_iter=20) is True
    assert load_winner_meta(tmp_path)["metric_value"] == 0.1


def test_save_winner_leaves_no_tmp_files(tmp_path: Path) -> None:
    save_winner(_state(1.0), "mario", 0.5, tmp_path, source_iter=10)
    wdir = tmp_path / "winners"
    assert not list(wdir.glob("*.tmp"))


def test_save_winner_stale_but_valid_sidecar_does_not_allow_downgrade(
    tmp_path: Path,
) -> None:
    """Regression for the two-file-transaction race: best.pt and
    best.json are two independently-atomic renames, so a kill between
    them can leave a *validly-parseable* sidecar reporting a stale,
    lower metric than what best.pt actually holds. A later call with a
    metric worse than the true best (0.9) but better than the stale
    sidecar (0.5) must NOT pass the gate and clobber the real winner —
    the checkpoint's own embedded metric_value is the reconciled
    source of truth, not the lagging sidecar alone."""
    assert save_winner(_state(9.0), "mario", 0.9, tmp_path, source_iter=42) is True
    best_pt, best_json = winner_paths(tmp_path)

    # Simulate the kill window: best.pt already embeds 0.9 (verified
    # below), but the sidecar never caught up and still shows the
    # previous, lower value.
    stale_meta = json.loads(best_json.read_text())
    stale_meta["metric_value"] = 0.5
    stale_meta["source_iter"] = 10
    best_json.write_text(json.dumps(stale_meta))
    assert torch.load(str(best_pt), weights_only=False)["metric_value"] == 0.9

    wrote = save_winner(_state(1.0), "mario", 0.7, tmp_path, source_iter=100)
    assert wrote is False

    # The true best (0.9, iter 42) must survive untouched.
    blob = load_winner("mario", tmp_path)
    assert blob["metric_value"] == 0.9
    assert torch.equal(blob["net_state_dict"]["w"], _state(9.0)["w"])


def test_save_winner_stale_sidecar_still_allows_genuine_improvement(
    tmp_path: Path,
) -> None:
    """A stale-low sidecar must not block a metric that genuinely beats
    what best.pt actually holds (not just what the sidecar claims)."""
    assert save_winner(_state(9.0), "mario", 0.9, tmp_path, source_iter=42) is True
    _, best_json = winner_paths(tmp_path)
    stale_meta = json.loads(best_json.read_text())
    stale_meta["metric_value"] = 0.5
    best_json.write_text(json.dumps(stale_meta))

    wrote = save_winner(_state(3.0), "mario", 0.95, tmp_path, source_iter=200)
    assert wrote is True
    meta = load_winner_meta(tmp_path)
    assert meta["metric_value"] == 0.95
    assert meta["source_iter"] == 200


def test_save_winner_missing_sidecar_reconciles_with_pt(tmp_path: Path) -> None:
    """Regression for the kill-before-first-sidecar-ever-exists case:
    best.pt was renamed into place (embedding metric_value=0.9) but the
    process died before best.json was ever created, so the sidecar is
    entirely absent rather than merely stale. A worse metric must still
    be refused by falling back to best.pt's own embedded value."""
    assert save_winner(_state(9.0), "mario", 0.9, tmp_path, source_iter=42) is True
    _, best_json = winner_paths(tmp_path)
    best_json.unlink()
    assert load_winner_meta(tmp_path) is None

    wrote = save_winner(_state(1.0), "mario", 0.7, tmp_path, source_iter=100)
    assert wrote is False
    blob = load_winner("mario", tmp_path)
    assert blob["metric_value"] == 0.9
    assert torch.equal(blob["net_state_dict"]["w"], _state(9.0)["w"])


# --------------------------------------------------------------------------- #
# rotation must never touch the winner
# --------------------------------------------------------------------------- #


def test_winner_survives_rotation_that_prunes_others(tmp_path: Path) -> None:
    """The whole point: a collapse-driven rotation that deletes every
    gen_*.pt must leave winners/best.pt standing."""
    import os
    # Save a real winner first.
    save_winner(_state(2.0), "mario", 0.88, tmp_path, source_iter=42)
    # Now a pile of gen checkpoints, oldest first.
    kept_last = None
    for i, name in enumerate(["gen_00001.pt", "gen_00002.pt", "gen_00003.pt"]):
        p = tmp_path / name
        p.write_bytes(b"x")
        ts = time.time() - (5 - i) * 10
        os.utime(p, (ts, ts))
        kept_last = p
    rotate_old_checkpoints(tmp_path, keep_last=1)
    # Only the newest gen survives...
    assert (tmp_path / "gen_00003.pt").exists()
    assert not (tmp_path / "gen_00001.pt").exists()
    assert not (tmp_path / "gen_00002.pt").exists()
    # ...and the winner is untouched.
    best_pt, best_json = winner_paths(tmp_path)
    assert best_pt.exists()
    assert best_json.exists()
    assert kept_last is not None


# --------------------------------------------------------------------------- #
# find_playable_checkpoint: winner > eval-history > latest
# --------------------------------------------------------------------------- #


def test_find_playable_prefers_winner(tmp_path: Path) -> None:
    _write_iter_ckpt(tmp_path, 10)
    _write_iter_ckpt(tmp_path, 20)  # latest
    save_winner(_state(2.0), "mario", 0.88, tmp_path, source_iter=15)
    chosen = find_playable_checkpoint("mario", tmp_path)
    assert chosen == winner_paths(tmp_path)[0]


def test_find_playable_falls_to_eval_history(tmp_path: Path) -> None:
    """No winner: pick the highest-clear_rate checkpoint from eval.jsonl,
    not the latest."""
    c10 = _write_iter_ckpt(tmp_path, 10)
    _write_iter_ckpt(tmp_path, 20)  # latest, but never cleared
    eval_log = tmp_path / "eval.jsonl"
    eval_log.write_text(
        json.dumps({"checkpoint": str(c10), "clear_rate": 0.8, "timestamp": 1.0}) + "\n"
        + json.dumps({"checkpoint": str(tmp_path / "vanilla_ppo_iter_00020.pt"),
                      "clear_rate": 0.0, "timestamp": 2.0}) + "\n"
    )
    assert find_playable_checkpoint("mario", tmp_path) == c10


def test_find_playable_ignores_zero_clear_history(tmp_path: Path) -> None:
    """If nothing ever cleared, eval history is no more playable than the
    latest — fall through to the freshest checkpoint."""
    _write_iter_ckpt(tmp_path, 10)
    c20 = _write_iter_ckpt(tmp_path, 20)
    eval_log = tmp_path / "eval.jsonl"
    eval_log.write_text(
        json.dumps({"checkpoint": str(tmp_path / "vanilla_ppo_iter_00010.pt"),
                    "clear_rate": 0.0, "timestamp": 1.0}) + "\n"
    )
    assert find_playable_checkpoint("mario", tmp_path) == c20


def test_find_playable_skips_missing_eval_checkpoint(tmp_path: Path) -> None:
    """An eval row whose checkpoint has since been rotated away is ignored;
    fall through to the latest that still exists."""
    c20 = _write_iter_ckpt(tmp_path, 20)
    eval_log = tmp_path / "eval.jsonl"
    eval_log.write_text(
        json.dumps({"checkpoint": str(tmp_path / "vanilla_ppo_iter_00005.pt"),
                    "clear_rate": 0.9, "timestamp": 1.0}) + "\n"
    )
    assert find_playable_checkpoint("mario", tmp_path) == c20


def _eval_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_find_playable_never_promotes_a_deterministic_replay_over_an_honest_row(
    tmp_path: Path,
) -> None:
    """EXHIBITION must not outrank LEARNED inside the selector.

    `eval_game.py` appends EVERY run to one `eval.jsonl`, deterministic ones
    included, and the selector keyed on `clear_rate` alone. Measured across
    62 checkpoint dirs, 8 were decided by a non-honest row while honest rows
    existed — `mario_1_2_online_v2` picked a `sticky 0.0, jitter 0, n=10`
    replay at 1.0 over its best honest `0.633, n=30`. That is this ledger's
    own EXHIBITION/LEARNED boundary being crossed in code.
    """
    honest = _write_iter_ckpt(tmp_path, 10)
    replay = _write_iter_ckpt(tmp_path, 20)
    _write_iter_ckpt(tmp_path, 30)  # latest, never evaluated
    _eval_rows(tmp_path / "eval.jsonl", [
        {"checkpoint": str(replay), "clear_rate": 1.0, "timestamp": 2.0,
         "sticky_prob": 0.0, "start_jitter": 0, "n_episodes": 10},
        {"checkpoint": str(honest), "clear_rate": 0.633, "timestamp": 1.0,
         "sticky_prob": 0.25, "start_jitter": 16, "n_episodes": 30},
    ])
    assert find_playable_checkpoint("mario", tmp_path) == honest


def test_find_playable_still_uses_the_whole_log_when_no_row_is_honest(
    tmp_path: Path,
) -> None:
    """Back-compat: a pre-protocol history keeps its old behaviour rather
    than being filtered down to nothing."""
    c10 = _write_iter_ckpt(tmp_path, 10)
    _write_iter_ckpt(tmp_path, 20)
    _eval_rows(tmp_path / "eval.jsonl", [
        {"checkpoint": str(c10), "clear_rate": 0.9, "timestamp": 1.0},
    ])
    assert find_playable_checkpoint("mario", tmp_path) == c10


def test_find_playable_prefers_the_better_sampled_of_two_equal_rates(
    tmp_path: Path,
) -> None:
    """0.5 over 30 episodes is a better estimate than 0.5 over 5."""
    small = _write_iter_ckpt(tmp_path, 10)
    big = _write_iter_ckpt(tmp_path, 20)
    _eval_rows(tmp_path / "eval.jsonl", [
        {"checkpoint": str(small), "clear_rate": 0.5, "timestamp": 9.0,
         "sticky_prob": 0.25, "start_jitter": 16, "n_episodes": 5},
        {"checkpoint": str(big), "clear_rate": 0.5, "timestamp": 1.0,
         "sticky_prob": 0.25, "start_jitter": 16, "n_episodes": 30},
    ])
    assert find_playable_checkpoint("mario", tmp_path) == big


def test_find_playable_does_not_treat_a_sampled_unperturbed_run_as_honest(
    tmp_path: Path,
) -> None:
    """`stochastic` is True for any sampled draw, including on an
    unperturbed environment. That is not a replay, but it is not the honest
    protocol either, and it must not be promoted as one."""
    sampled = _write_iter_ckpt(tmp_path, 10)
    honest = _write_iter_ckpt(tmp_path, 20)
    _eval_rows(tmp_path / "eval.jsonl", [
        {"checkpoint": str(sampled), "clear_rate": 1.0, "timestamp": 2.0,
         "sticky_prob": 0.0, "start_jitter": 0, "stochastic": True,
         "action_select": "sampled", "n_episodes": 10},
        {"checkpoint": str(honest), "clear_rate": 0.4, "timestamp": 1.0,
         "sticky_prob": 0.25, "start_jitter": 16, "stochastic": True,
         "n_episodes": 50},
    ])
    assert find_playable_checkpoint("mario", tmp_path) == honest


def test_find_playable_falls_to_latest(tmp_path: Path) -> None:
    _write_iter_ckpt(tmp_path, 10)
    c30 = _write_iter_ckpt(tmp_path, 30)
    _write_iter_ckpt(tmp_path, 20)
    assert find_playable_checkpoint("mario", tmp_path) == c30


def test_find_playable_none_when_empty(tmp_path: Path) -> None:
    assert find_playable_checkpoint("mario", tmp_path) is None
    assert find_playable_checkpoint("mario", tmp_path / "missing") is None


def test_find_playable_tolerates_garbage_eval_lines(tmp_path: Path) -> None:
    c10 = _write_iter_ckpt(tmp_path, 10)
    eval_log = tmp_path / "eval.jsonl"
    eval_log.write_text(
        "not json at all\n"
        + json.dumps({"game": "mario", "status": "no_checkpoint"}) + "\n"  # no clear_rate
        + json.dumps({"checkpoint": str(c10), "clear_rate": 0.6, "timestamp": 3.0}) + "\n"
        + "\n"
    )
    assert find_playable_checkpoint("mario", tmp_path) == c10


# --------------------------------------------------------------------------- #
# find_latest_trained_checkpoint: vanilla_ppo_iter by number, else gen by mtime
# --------------------------------------------------------------------------- #


def test_find_latest_trained_prefers_highest_iter(tmp_path: Path) -> None:
    _write_iter_ckpt(tmp_path, 5)
    c100 = _write_iter_ckpt(tmp_path, 100)
    _write_iter_ckpt(tmp_path, 40)
    assert find_latest_trained_checkpoint(tmp_path) == c100


def test_find_latest_trained_falls_back_to_gen(tmp_path: Path) -> None:
    import os
    old = tmp_path / "gen_00001.pt"
    old.write_bytes(b"x")
    new = tmp_path / "gen_00002.pt"
    new.write_bytes(b"x")
    os.utime(old, (time.time() - 100, time.time() - 100))
    assert find_latest_trained_checkpoint(tmp_path) == new


def test_find_latest_trained_none_when_empty(tmp_path: Path) -> None:
    assert find_latest_trained_checkpoint(tmp_path) is None
