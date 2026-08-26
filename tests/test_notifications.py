"""Unit tests for src/training/notifications.py.

Covers two things, both without ever touching a real macOS notification:

  1. `notify_macos` shells out to `osascript` correctly and degrades to a
     logged warning (never an exception) on any failure — missing binary,
     non-zero exit, or timeout.
  2. `StallNotifier` fires at most once per stall episode: it latches on
     the first threshold-crossing call, stays silent for any number of
     further calls (even ones with a much higher counter, and even after
     the raw counter is reset back below threshold by unrelated
     bookkeeping), and only re-arms when `reset()` — the genuine-advance
     signal — is called.

Every `osascript` invocation is mocked via `unittest.mock.patch`; no
subprocess is ever actually spawned.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from src.training.notifications import StallNotifier, notify_macos


# ---------------------------------------------------------------------------
# notify_macos
# ---------------------------------------------------------------------------

def test_notify_macos_invokes_osascript_with_display_notification() -> None:
    with patch("src.training.notifications.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        ok = notify_macos("Training stalled", "stage 3 stuck for 200 iters")

    assert ok is True
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    cmd = args[0]
    assert cmd[0] == "osascript"
    assert cmd[1] == "-e"
    script = cmd[2]
    assert "display notification" in script
    assert "stage 3 stuck for 200 iters" in script
    assert "Training stalled" in script
    assert kwargs.get("check") is True


def test_notify_macos_escapes_quotes_in_message() -> None:
    with patch("src.training.notifications.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        notify_macos('title "with quotes"', 'message "with quotes" too')

    script = mock_run.call_args[0][0][2]
    assert '\\"with quotes\\"' in script


def test_notify_macos_returns_false_and_does_not_raise_when_binary_missing() -> None:
    with patch(
        "src.training.notifications.subprocess.run",
        side_effect=FileNotFoundError("no such file: osascript"),
    ):
        ok = notify_macos("t", "m")  # must not raise
    assert ok is False


def test_notify_macos_returns_false_on_nonzero_exit() -> None:
    with patch(
        "src.training.notifications.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, ["osascript"]),
    ):
        ok = notify_macos("t", "m")
    assert ok is False


def test_notify_macos_returns_false_on_timeout() -> None:
    with patch(
        "src.training.notifications.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["osascript"], timeout=10),
    ):
        ok = notify_macos("t", "m")
    assert ok is False


# ---------------------------------------------------------------------------
# StallNotifier — the at-most-once-per-stall-episode behavior
# ---------------------------------------------------------------------------

def test_stall_notifier_fires_exactly_once_when_crossing_threshold() -> None:
    with patch("src.training.notifications.notify_macos") as mock_notify:
        notifier = StallNotifier(
            threshold=60, title="Training stalled", message_fn=lambda n: f"n={n}"
        )

        # Below threshold: never fires.
        for n in range(0, 60):
            fired = notifier.maybe_notify(n)
            assert fired is False
        mock_notify.assert_not_called()

        # Crosses threshold: fires exactly this one time.
        fired = notifier.maybe_notify(60)
        assert fired is True
        mock_notify.assert_called_once()

        # Continuing to climb well past threshold for many more iters must
        # NOT spam additional notifications — this is the "at most once
        # per stall episode" guarantee the task is about.
        for n in range(61, 300):
            fired = notifier.maybe_notify(n)
            assert fired is False
        mock_notify.assert_called_once()


def test_stall_notifier_ignores_raw_counter_resets_within_same_episode() -> None:
    """The underlying stall counter can be reset by UNRELATED bookkeeping
    (e.g. a Go-Explore burst arming/retracting) without the run having
    genuinely advanced. The latch must not be fooled by the counter
    dropping back down and climbing past threshold again — only an
    explicit `reset()` (the genuine-advance signal) may re-arm it."""
    with patch("src.training.notifications.notify_macos") as mock_notify:
        notifier = StallNotifier(threshold=10, title="t", message_fn=lambda n: "m")

        assert notifier.maybe_notify(10) is True
        mock_notify.assert_called_once()

        # Counter housekeeping resets it to 0 and climbs past 10 again —
        # simulating a burst arm/retract cycle with no real advance.
        for n in list(range(0, 5)) + [10, 15, 20]:
            assert notifier.maybe_notify(n) is False
        mock_notify.assert_called_once()


def test_stall_notifier_fires_again_after_reset_new_stall_episode() -> None:
    """Exactly one notification PER stall episode: once the run genuinely
    advances (`reset()`), a fresh stall must be able to notify again."""
    with patch("src.training.notifications.notify_macos") as mock_notify:
        notifier = StallNotifier(threshold=10, title="t", message_fn=lambda n: "m")

        assert notifier.maybe_notify(10) is True
        assert notifier.maybe_notify(11) is False
        assert mock_notify.call_count == 1

        notifier.reset()  # genuine curriculum advance
        assert notifier.notified is False

        # Same underlying counter shape (resets to 0, climbs again) — but
        # this time via a real reset, so hitting threshold again SHOULD
        # notify: it's a new stall episode.
        assert notifier.maybe_notify(0) is False
        assert notifier.maybe_notify(9) is False
        assert notifier.maybe_notify(10) is True
        assert mock_notify.call_count == 2


def test_stall_notifier_survives_message_fn_exception() -> None:
    """A broken message formatter must not propagate — this mirrors the
    task's 'must not crash training' requirement one layer up."""

    def _boom(n: int) -> str:
        raise ValueError("boom")

    with patch("src.training.notifications.notify_macos") as mock_notify:
        notifier = StallNotifier(threshold=5, title="t", message_fn=_boom)
        fired = notifier.maybe_notify(5)  # must not raise

    assert fired is True
    mock_notify.assert_not_called()  # never reached notify_macos


def test_stall_notifier_restores_notified_flag_from_constructor() -> None:
    """Mirrors trainer.py's checkpoint-resume path, which restores
    `stall_notifier.notified` from a saved `ge_stall_notified` flag."""
    with patch("src.training.notifications.notify_macos") as mock_notify:
        notifier = StallNotifier(
            threshold=10, title="t", message_fn=lambda n: "m", notified=True
        )
        # A resumed run mid-stall-episode must not re-fire.
        assert notifier.maybe_notify(500) is False
        mock_notify.assert_not_called()
