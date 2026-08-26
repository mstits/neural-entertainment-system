"""Best-effort local (macOS) notifications for long-running training runs.

This module is a pure ADDITION to the training loop: it never influences
training behavior, only surfaces state that already exists (e.g. the
Go-Explore stall clock in `trainer.py`) to a human via Notification
Center. Every path here is designed to fail silently (logged, not
raised) so a missing `osascript`, a non-macOS host, or any subprocess
hiccup can never interrupt or alter a run.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Callable, Optional

log = logging.getLogger(__name__)


def _osascript_escape(text: str) -> str:
    """Escape a string for embedding in an AppleScript double-quoted
    literal. Backslashes first, then quotes, so an already-escaped
    backslash isn't double-escaped."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def notify_macos(title: str, message: str, *, subtitle: Optional[str] = None) -> bool:
    """Fire a macOS Notification Center banner via `osascript`.

    Returns True if `osascript` was invoked and exited zero, False on any
    failure. NEVER raises: a missing `osascript` binary (non-macOS CI, a
    stripped-down environment), a timeout, or a non-zero exit are all
    caught here and logged as a warning so the caller (the training loop)
    never has to guard against this being fatal.
    """
    script = (
        f'display notification "{_osascript_escape(message)}" '
        f'with title "{_osascript_escape(title)}"'
    )
    if subtitle:
        script += f' subtitle "{_osascript_escape(subtitle)}"'
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=True,
            capture_output=True,
            timeout=10,
        )
        return True
    except Exception as e:
        log.warning("[notifications] macOS notification failed (non-fatal): %s", e)
        return False


class StallNotifier:
    """At-most-once-per-stall-episode latch around `notify_macos`.

    Wraps a raw "iterations since last advance" counter so a caller can
    call `maybe_notify(iters_since_advance)` every iteration without
    worrying about spamming: it fires exactly once the first time the
    counter reaches `threshold`, then stays silent — no matter how many
    more iterations pass, and no matter how many times the counter itself
    gets reset by unrelated bookkeeping (e.g. a Go-Explore burst arming
    or retracting, which resets the raw stall clock without the run
    actually having advanced) — until `reset()` is called to re-arm it
    for the next stall episode. `reset()` should be called ONLY on a
    genuine curriculum advance, never on the burst housekeeping resets,
    or every stall episode would be reported as several.
    """

    def __init__(
        self,
        threshold: int,
        title: str,
        message_fn: Callable[[int], str],
        *,
        notified: bool = False,
    ) -> None:
        self.threshold = int(threshold)
        self.title = title
        self.message_fn = message_fn
        self.notified = bool(notified)

    def maybe_notify(self, iters_since_advance: int) -> bool:
        """Fire the notification if due. Returns True iff it fired THIS
        call (i.e. this is the transition edge, not just "past
        threshold")."""
        if self.notified or iters_since_advance < self.threshold:
            return False
        self.notified = True
        try:
            notify_macos(self.title, self.message_fn(iters_since_advance))
        except Exception as e:
            # notify_macos already catches its own failures; this is a
            # last-resort guard against a caller-supplied message_fn
            # raising, so a bad format string can't take down training.
            log.warning(
                "[notifications] stall notification raised unexpectedly "
                "(non-fatal): %s", e,
            )
        return True

    def reset(self) -> None:
        """Re-arm for the next stall episode. Call ONLY on a genuine
        advance — never on internal counter resets that aren't real
        progress (e.g. Go-Explore burst arm/retract)."""
        self.notified = False
