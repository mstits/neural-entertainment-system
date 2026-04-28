from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "parity: reference-emulator diff harness (nes_core vs nes-py). "
        "Excluded from the default suite; run via `make parity`.",
    )
