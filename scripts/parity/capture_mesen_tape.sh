#!/usr/bin/env bash
# Capture a Mesen RAM tape for a ROM via Mesen --testRunner mode.
#
# Usage:
#   scripts/parity/capture_mesen_tape.sh <rom_basename> [frames]
#
# Example:
#   scripts/parity/capture_mesen_tape.sh "Super Mario Bros. (World).nes" 120
#
# Output: tests/parity/mesen_tapes/<rom_basename>.bin (frames * 2KB)
# Status:  tests/parity/mesen_tapes/<rom_basename>.status.txt
set -euo pipefail

ROM_BASENAME="${1:?ROM basename required}"
FRAMES="${2:-120}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROM_PATH="${REPO}/roms/${ROM_BASENAME}"
TAPES_DIR="${REPO}/tests/parity/mesen_tapes"
LUA="${REPO}/scripts/tracing/mesen_dump_generic.lua"

if [[ ! -f "$ROM_PATH" ]]; then
    echo "ROM not found: $ROM_PATH" >&2
    exit 1
fi

if [[ ! -x "/Applications/Mesen.app/Contents/MacOS/Mesen" ]]; then
    echo "Mesen.app not installed at /Applications/Mesen.app" >&2
    exit 2
fi

mkdir -p "$TAPES_DIR"
OUT="${TAPES_DIR}/${ROM_BASENAME}.bin"
rm -f "$OUT" "${OUT}.status.txt"

MESEN_DUMP_OUT="$OUT" \
MESEN_DUMP_FRAMES="$FRAMES" \
/Applications/Mesen.app/Contents/MacOS/Mesen \
    --testRunner --noaudio --novideo --noinput \
    "$LUA" \
    "$ROM_PATH" \
    --timeout=120 2>&1 | tail -3

STATUS=$(tail -1 "${OUT}.status.txt")
SIZE=$(wc -c < "$OUT")
echo "  $ROM_BASENAME: $STATUS ($SIZE bytes)"
