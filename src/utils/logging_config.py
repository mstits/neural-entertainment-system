"""
Structured logging for Neural Entertainment System.

Replaces `logging.basicConfig` with a setup that:
  * emits compact, grep-friendly single-line records,
  * writes to stderr AND to a rotating file next to the checkpoint dir,
  * sets sensible levels (INFO by default; DEBUG via --verbose).

Keep the format stable so downstream log parsing doesn't break.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


_FMT = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"


def configure(verbose: bool = False, log_dir: str | Path | None = None) -> None:
    """Initialize the root logger. Idempotent — safe to call twice."""
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    # Clear prior handlers so repeat calls don't duplicate output.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    stderr = logging.StreamHandler()
    stderr.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
    root.addHandler(stderr)

    if log_dir is not None:
        d = Path(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            d / "trainer.log", maxBytes=10 * 1024 * 1024, backupCount=5
        )
        fh.setFormatter(logging.Formatter(_FMT))
        root.addHandler(fh)
