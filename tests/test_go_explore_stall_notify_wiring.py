"""Source-anchor tests for the Go-Explore stall -> macOS-notification wiring
in `Trainer._run_vanilla_ppo`.

The actual "fires exactly once per stall episode" behavior is unit-tested
directly against `StallNotifier` in `tests/test_notifications.py` (fully
mocked, ROM-free). This file pins the trainer-side GLUE — the same style
as `tests/test_char_curriculum_glue.py` — since driving the real loop far
enough to observe a multi-hundred-iteration stall would need a live ROM
run lasting minutes, which the codebase's convention (see that file's own
"Determinism basis" note) is to avoid in favor of anchoring the exact
source clauses instead.

Pinned invariants:
  * The notify threshold is a multiple of the existing `ge_stall_patience`
    (NOT the unrelated, much smaller `SMB_ADVANCE_WINDOW` rolling-mean
    window a few hundred lines below) — see the reasoning comment in
    trainer.py for why.
  * The stall-check happens inside the SAME `if ge_burst_on:` guard as the
    existing counter increment, so it is exactly as inert as the counter
    itself when Go-Explore fallback is off.
  * The notify latch is reset ONLY on a genuine ladder advance, not on the
    burst arm/retract housekeeping resets of the raw counter.
  * The latch's state round-trips through the same `curriculum_resume`
    checkpoint blob the rest of the Go-Explore stall state already uses.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_TRAINER_SRC = (ROOT / "src" / "training" / "trainer.py").read_text()


def _vppo_body() -> str:
    start = _TRAINER_SRC.find("def _run_vanilla_ppo")
    end = _TRAINER_SRC.find("def _emit_metrics", start)
    assert start >= 0 and end > start, "could not bound _run_vanilla_ppo body"
    return _TRAINER_SRC[start:end]


def test_stall_notifier_imported_and_pure_addition_module_used() -> None:
    assert (
        "from src.training.notifications import StallNotifier" in _TRAINER_SRC
    ), "StallNotifier import missing/renamed"


def test_notify_threshold_is_a_multiple_of_stall_patience_not_advance_window() -> None:
    body = _vppo_body()
    assert (
        "ge_stall_notify_multiplier = max(\n            1, "
        'int(_geb_cfg.get("stall_notify_multiplier", 3))\n        )'
        in body
    ), "notify multiplier default/config-key changed"
    assert (
        "ge_stall_notify_threshold = ge_stall_patience * ge_stall_notify_multiplier"
        in body
    ), "notify threshold must be derived from ge_stall_patience"


def test_notify_check_lives_inside_the_ge_burst_on_increment_guard() -> None:
    """The check must be exactly as inert as the existing counter when the
    Go-Explore fallback is disabled (the common case) — i.e. nested in the
    same `if ge_burst_on:` block as `ge_iters_since_advance += 1`."""
    body = _vppo_body()
    anchor = (
        "if ge_burst_on:\n"
        "                ge_iters_since_advance += 1\n"
    )
    idx = body.find(anchor)
    assert idx >= 0, "ge_iters_since_advance increment site changed shape"
    # The notify call must appear shortly after, still inside the block
    # (before the next top-level statement resets the per-iter counters).
    window = body[idx: idx + 1500]
    assert "stall_notifier.maybe_notify(ge_iters_since_advance)" in window
    assert "n_clears_this_iter = 0" in window  # next sibling statement


def test_notify_call_is_try_except_wrapped_and_logs_a_warning() -> None:
    body = _vppo_body()
    idx = body.find("stall_notifier.maybe_notify(ge_iters_since_advance)")
    assert idx >= 0
    surrounding = body[max(0, idx - 200): idx + 300]
    assert "try:" in surrounding
    assert "except Exception" in surrounding
    assert "log.warning" in surrounding


def test_latch_resets_only_on_genuine_ladder_advance() -> None:
    body = _vppo_body()
    anchor = "ge_iters_since_advance = 0\n                    stall_notifier.reset()"
    assert anchor in body, (
        "stall_notifier.reset() must sit alongside the genuine-advance "
        "ge_iters_since_advance reset, not the burst arm/retract resets"
    )
    # Exactly one `stall_notifier.reset()` call in the whole method: the
    # genuine-advance site. The burst ARM and RETRACT sites also zero
    # `ge_iters_since_advance` (housekeeping, not real progress) and must
    # NOT also reset the notify latch, or one stall episode spanning
    # several burst cycles would be reported as several.
    assert body.count("stall_notifier.reset()") == 1, (
        "stall_notifier.reset() must be called from exactly one site "
        "(the genuine ladder-advance branch) — a call anywhere else "
        "(e.g. burst arm/retract) would re-arm the latch on internal "
        "housekeeping instead of real progress"
    )


def test_checkpoint_resume_round_trips_the_notify_latch() -> None:
    body = _vppo_body()
    assert '"ge_stall_notified": bool(stall_notifier.notified),' in body
    assert (
        'stall_notifier.notified = bool(_cr.get("ge_stall_notified", False))'
        in body
    )


def test_stall_metric_surfaced_alongside_existing_stall_iters_metric() -> None:
    body = _vppo_body()
    idx = body.find('progress_metrics["vanilla_ppo_ge_stall_iters"]')
    assert idx >= 0
    window = body[idx: idx + 200]
    assert 'progress_metrics["vanilla_ppo_ge_stall_notified"]' in window
