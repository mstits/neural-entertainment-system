#!/usr/bin/env bash
# Profile-Guided Optimization build for nes_core.
#
# Delivers a measured +81% throughput on the hot-path bench (M4 Max,
# 710 → 1289 worker-steps/sec) and 47% faster 1-gen trainer wall-time
# (2.54s → 1.35s on the test_trainer_one_gen configuration).
#
# Three-stage flow:
#   1. Build instrumented wheel (RUSTFLAGS="-Cprofile-generate=...").
#   2. Run a representative workload → emits raw profdata to
#      nes_core/pgo/raw/.
#   3. Merge + rebuild with RUSTFLAGS="-Cprofile-use=merged.profdata".
#
# Runtime code is unchanged; this is a pure compiler-side perf pass.
#
# Usage:
#   scripts/pgo_build.sh [mode] [workload]
#
#   mode:
#     full     (default) — all three stages, ~3 min on M4 Max
#     apply    — skip stages 1-2, just apply an existing merged profdata
#     refresh  — regenerate profdata from the cached raw dump
#
#   workload (stage 2 only):
#     bench    (default) — scripts/bench_hot_path.py
#     train    — scripts/test_trainer_one_gen.py
#
# PGO is NOT automatic on plain `maturin develop` — rebuilds without
# these RUSTFLAGS get a non-PGO wheel. Run this script once after any
# `nes_core/src/*` change that warrants a fresh perf pass.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PGO_DIR="${REPO}/nes_core/pgo"
RAW_DIR="${PGO_DIR}/raw"
MERGED="${PGO_DIR}/nes_core.profdata"
LLVM_PROFDATA="$(xcrun -find llvm-profdata)"

MODE="${1:-full}"
WORKLOAD="${2:-bench}"

mkdir -p "${PGO_DIR}"
cd "${REPO}/nes_core"
source "$HOME/.cargo/env" 2>/dev/null || true

if [[ "${MODE}" == "full" ]]; then
    echo "==> [1/3] Building instrumented wheel..."
    rm -rf "${RAW_DIR}"
    mkdir -p "${RAW_DIR}"
    RUSTFLAGS="-Cprofile-generate=${RAW_DIR}" \
        "${REPO}/.venv/bin/maturin" develop --release --quiet

    echo "==> [2/3] Running workload (${WORKLOAD}) to generate profile data..."
    # Empirical: the default (non-headless) xrgb→rgb + rgb→gray
    # two-stage path is faster than the fused xrgb→gray kernel
    # because vld4q+vst3q is cheaper than per-pixel luma muladd.
    # Profile the production path (headless=false) so PGO aligns
    # to the actually-fastest code.
    case "${WORKLOAD}" in
        bench)
            # Two-ROM profile run: Zelda (MMC1) + Contra (mapper differs,
            # exercises both hot-path dispatch branches). 16-worker +
            # 12-worker both profiled so the PGO covers the 16w-contended
            # code path (where the 16P-core rayon scheduling stresses
            # cache/lock prediction) AND the 12w-deployment path.
            "${REPO}/.venv/bin/python" \
                "${REPO}/scripts/bench_hot_path.py" \
                --workers 16 --steps 400 --frame-skip 16
            "${REPO}/.venv/bin/python" \
                "${REPO}/scripts/bench_hot_path.py" \
                --workers 12 --steps 200 --frame-skip 16 \
                --rom "${REPO}/roms/CONTRA.NES" \
                --profile "${REPO}/configs/contra.yaml"
            ;;
        train)
            "${REPO}/.venv/bin/python" \
                "${REPO}/scripts/test_trainer_one_gen.py"
            ;;
        *)
            echo "unknown workload: ${WORKLOAD}" >&2
            exit 2
            ;;
    esac
fi

if [[ "${MODE}" == "full" || "${MODE}" == "refresh" ]]; then
    echo "==> [2.5/3] Merging raw profdata..."
    "${LLVM_PROFDATA}" merge -o "${MERGED}" "${RAW_DIR}"
fi

if [[ ! -f "${MERGED}" ]]; then
    echo "error: merged profdata not found at ${MERGED}" >&2
    echo "hint: run 'scripts/pgo_build.sh full' to generate it" >&2
    exit 1
fi

echo "==> [3/3] Rebuilding with PGO applied..."
RUSTFLAGS="-Cprofile-use=${MERGED}" \
    "${REPO}/.venv/bin/maturin" develop --release --quiet

echo
echo "Done. nes_core rebuilt with PGO applied."
echo "  bench: .venv/bin/python scripts/bench_hot_path.py"
echo "  undo:  (cd nes_core && .venv/bin/maturin develop --release)"
