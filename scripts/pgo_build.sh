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
#   PGO_CORPUS env var (stage 2 only) — which profile corpus to emit.
#     Part of the event-driven-PPU catch-up campaign
#     (docs/proposals/ppu_event_driven_catchup.md, graft L4 "dual-corpus
#     PGO"): once a scanline-granular `advance` becomes the common
#     skip-render path, PGO could demote `Ppu::tick` to a cold section
#     and regress the full-render / spectator path that still uses it.
#     To keep BOTH `tick` (full render) and `advance` (skip render) hot,
#     stage 2 must profile both.
#       skip  (default) — current behaviour EXACTLY: the skip-render
#                         training frames only (frame-skip 16).
#       full            — only the full-render / spectator leg
#                         (frame-skip 1, skip_render OFF every frame).
#       dual            — skip-render legs PLUS the full-render leg, so
#                         the merged profile covers tick AND advance.
#     Default = skip, so an unset PGO_CORPUS reproduces today's profile
#     byte-for-byte.
#
#   PGO_DRY_RUN=1 — print every stage command (build, profile legs,
#     merge, apply) WITHOUT executing anything. Use to inspect the plan.
#
# PGO is NOT automatic on plain `maturin develop` — rebuilds without
# these RUSTFLAGS get a non-PGO wheel. Run this script once after any
# `nes_core/src/*` change that warrants a fresh perf pass.
#
#   scripts/pgo_build.sh help    — print this usage.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PGO_DIR="${REPO}/nes_core/pgo"
RAW_DIR="${PGO_DIR}/raw"
MERGED="${PGO_DIR}/nes_core.profdata"
LLVM_PROFDATA="$(xcrun -find llvm-profdata)"

MODE="${1:-full}"
WORKLOAD="${2:-bench}"
CORPUS="${PGO_CORPUS:-skip}"
DRY_RUN="${PGO_DRY_RUN:-0}"

if [[ "${MODE}" == "help" || "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
    sed -n '2,58p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

case "${CORPUS}" in
    skip|full|dual) ;;
    *) echo "unknown PGO_CORPUS: ${CORPUS} (want: skip | full | dual)" >&2; exit 2 ;;
esac

# Run (or, under PGO_DRY_RUN, just print) a labelled command.
run_step() {
    local label="$1"; shift
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '    [dry-run] %s\n              ' "${label}"
        printf '%q ' "$@"; printf '\n'
    else
        echo "    -> ${label}"
        "$@"
    fi
}

mkdir -p "${PGO_DIR}"
cd "${REPO}/nes_core"
source "$HOME/.cargo/env" 2>/dev/null || true

if [[ "${MODE}" == "full" ]]; then
    echo "==> [1/3] Building instrumented wheel..."
    if [[ "${DRY_RUN}" != "1" ]]; then
        rm -rf "${RAW_DIR}"
        mkdir -p "${RAW_DIR}"
    fi
    run_step "instrumented build (profile-generate)" \
        env RUSTFLAGS="-Cprofile-generate=${RAW_DIR} -Ctarget-cpu=apple-m4" \
        "${REPO}/.venv/bin/maturin" develop --release --quiet

    echo "==> [2/3] Running workload (${WORKLOAD}, corpus=${CORPUS}) to generate profile data..."
    # A full-render / spectator leg (frame-skip 1 => skip_render OFF
    # every frame) that keeps `Ppu::tick` and write_frame hot in the
    # profile. Emitted for PGO_CORPUS=full|dual (see the header). The
    # Zelda dominant matches the skip legs so block layout stays tuned
    # to the headline workload.
    full_render_leg() {
        run_step "full-render / spectator (fs=1, Zelda)" \
            "${REPO}/.venv/bin/python" \
            "${REPO}/scripts/bench_hot_path.py" \
            --workers 12 --steps 120 --frame-skip 1
    }
    # Empirical: the default (non-headless) xrgb→rgb + rgb→gray
    # two-stage path is faster than the fused xrgb→gray kernel
    # because vld4q+vst3q is cheaper than per-pixel luma muladd.
    # Profile the production path (headless=false) so PGO aligns
    # to the actually-fastest code.
    case "${WORKLOAD}" in
        bench)
            # Three-ROM profile run, one per deployed mapper family whose
            # dispatch is data-dependent hot Rust:
            #   * Zelda     (MMC1 / Mapper1) — the headline 16w training
            #                 workload; kept dominant so its shared PPU/CPU
            #                 block layout stays tuned to it.
            #   * Contra    (UNROM / Mapper2) — second dispatch branch.
            #   * Punch-Out (MMC2 / Mapper9) — its CHR bank-latch
            #                 (fetch_sprite_tile -> Mapper9::chr_read_byte)
            #                 is hot but data-dependent; when the profile
            #                 never sees it, PGO lays that function out COLD
            #                 (no inline, cold-section, pessimal branch
            #                 order), leaving Punch-Out training ~5-6%
            #                 slower. One short pass flips it hot.
            # 16-worker + 12-worker both profiled so the PGO covers the
            # 16w-contended path (where 16P-core rayon scheduling stresses
            # cache/lock prediction) AND the 12w-deployment path.
            #
            # The Punch-Out pass is deliberately SHORT (12w x 100 steps,
            # ~12% of the profile's frame mass) rather than an equal share:
            # the MMC2 win is a hot/cold-layout threshold effect (chr_read
            # only needs nonzero counts to leave the cold section), so a
            # short pass captures it, while a full-weight pass would pull
            # the shared PPU/CPU block layout toward Punch-Out's
            # sprite-heavy pattern and dilute Zelda. Measured (pure-Rust
            # emulator-codegen A/B, cargo-clean between profiles, min-of/
            # median of 7-10 interleaved P-core-pinned rounds vs the
            # two-ROM profile):
            #   Punch-Out (m9)  single-env +6.3% / 16w +5.7%
            #   Zelda     (m1)  16w denominator  -0.5% median (noise)
            #   Contra/SMB/Gradius              flat-to-positive
            # A naive all-games profile instead regresses the Zelda 16w
            # denominator -1.4%, so do NOT broaden past this short pass.
            #
            # These three skip-render legs are the historical corpus and
            # run for PGO_CORPUS=skip (default) or dual — byte-identical
            # to the pre-dual-corpus script.
            if [[ "${CORPUS}" == "skip" || "${CORPUS}" == "dual" ]]; then
                run_step "skip-render Zelda (16w, fs=16)" \
                    "${REPO}/.venv/bin/python" \
                    "${REPO}/scripts/bench_hot_path.py" \
                    --workers 16 --steps 400 --frame-skip 16
                run_step "skip-render Contra (12w, fs=16)" \
                    "${REPO}/.venv/bin/python" \
                    "${REPO}/scripts/bench_hot_path.py" \
                    --workers 12 --steps 200 --frame-skip 16 \
                    --rom "${REPO}/roms/CONTRA.NES" \
                    --profile "${REPO}/configs/contra.yaml"
                run_step "skip-render Punch-Out (12w, fs=16)" \
                    "${REPO}/.venv/bin/python" \
                    "${REPO}/scripts/bench_hot_path.py" \
                    --workers 12 --steps 100 --frame-skip 16 \
                    --rom "${REPO}/roms/Mike Tyson's Punch-Out!! (Japan, USA) (Rev A).nes" \
                    --profile "${REPO}/configs/punchout.yaml"
            fi
            if [[ "${CORPUS}" == "full" || "${CORPUS}" == "dual" ]]; then
                full_render_leg
            fi
            ;;
        train)
            if [[ "${CORPUS}" == "skip" || "${CORPUS}" == "dual" ]]; then
                run_step "trainer one-gen (skip-render)" \
                    "${REPO}/.venv/bin/python" \
                    "${REPO}/scripts/test_trainer_one_gen.py"
            fi
            if [[ "${CORPUS}" == "full" || "${CORPUS}" == "dual" ]]; then
                full_render_leg
            fi
            ;;
        *)
            echo "unknown workload: ${WORKLOAD}" >&2
            exit 2
            ;;
    esac
fi

if [[ "${MODE}" == "full" || "${MODE}" == "refresh" ]]; then
    echo "==> [2.5/3] Merging raw profdata..."
    run_step "merge raw profdata" \
        "${LLVM_PROFDATA}" merge -o "${MERGED}" "${RAW_DIR}"
fi

if [[ "${DRY_RUN}" != "1" && ! -f "${MERGED}" ]]; then
    echo "error: merged profdata not found at ${MERGED}" >&2
    echo "hint: run 'scripts/pgo_build.sh full' to generate it" >&2
    exit 1
fi

echo "==> [3/3] Rebuilding with PGO applied..."
run_step "PGO-applied build (profile-use)" \
    env RUSTFLAGS="-Cprofile-use=${MERGED} -Ctarget-cpu=apple-m4" \
    "${REPO}/.venv/bin/maturin" develop --release --quiet

if [[ "${DRY_RUN}" == "1" ]]; then
    echo
    echo "[dry-run] no commands were executed."
fi

echo
echo "Done. nes_core rebuilt with PGO applied."
echo "  bench: .venv/bin/python scripts/bench_hot_path.py"
echo "  undo:  (cd nes_core && .venv/bin/maturin develop --release)"
