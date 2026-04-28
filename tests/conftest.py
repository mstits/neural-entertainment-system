"""Put the repo root on sys.path so `import src...` works in pytest."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def pytest_addoption(parser):
    """Custom flags shared across test files."""
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Refresh golden files (backtest, etc.) from the current run.",
    )


def pytest_configure(config):
    """Register custom markers so they don't emit warnings."""
    config.addinivalue_line(
        "markers", "soak: long-running stability test — excluded by default."
    )
