"""The Forge: machinery that detects a stalled wall, diagnoses it from the
system's own telemetry, and proposes/gates/registers a mechanism to try
against it. See FORGE_SPEC_2026-09-01.md (docs/proposals/) for the full
architecture. This package holds pure functions only; nothing here writes
CLAIMS.md or touches the emulator directly.
"""
from __future__ import annotations
