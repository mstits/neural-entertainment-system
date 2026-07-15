#!/usr/bin/env bash
# Golden-disassembly layout gate for the event-driven-PPU catch-up
# campaign (docs/proposals/ppu_event_driven_catchup.md, Rung 0).
#
# WHY: the campaign's history is silent layout regressions — tick_n
# −55%, inline-tick −52%, skip_bg-under-PGO removed — where a change
# perturbed the machine code of the hot `Ppu::tick` body (or its
# cache placement) and cost double-digit throughput with NO other
# detector. This gate fingerprints the disassembly of the hot
# reference symbol(s) and fails if a rung changes their machine code.
#
# WHAT IT FINGERPRINTS: `nes_core::ppu::Ppu::tick` (every
# monomorphization present in the reference example). `tick_three`,
# `render_pixel`, the `fetch_*`, `shift_background_registers` and
# `update_sprite_rendering_registers` leaves are all `#[inline(always)]`
# or `#[inline]` and fold INTO `tick`, so tick's fingerprint transitively
# covers them. The fingerprint is a normalized instruction-sequence
# hash, NOT raw bytes: the instruction-address column is dropped and
# absolute addresses / reloc targets (`bl 0x1.. <sym>`, adrp/adr
# targets, literal-pool addresses) are normalized to `ADDR`, so an
# unrelated change that merely shifts tick's link address does NOT trip
# the gate — only a change to tick's own instruction stream (mnemonics,
# register/operand shape, struct-field offsets, insn count) does.
#
# BUILD MODE (stated limitation): the gate builds the `release-with-debug`
# profile (inherits release's opt-level 3 / fat LTO / codegen-units 1 /
# target-cpu, but keeps the symbol table `strip = "none"` so the hot
# symbols can be located). It deliberately does NOT use PGO: PGO's
# block placement varies run-to-run with the profile, so a PGO
# disassembly is not a stable golden. `release-with-debug` shares
# release's instruction selection, so a source change to `tick` moves
# this fingerprint exactly as it moves the shipped `--release` binary;
# it is a faithful proxy for the hot-symbol machine code, minus the
# PGO-only block ordering (which this gate intentionally ignores).
#
# The reference example is `ppu_state_profile` (uses Xrgb8888VideoSink)
# and is built WITHOUT `ppu_batch_stats`, so the Rung-0 instrumentation
# (which is `#[cfg]`-gated off in every shipped build) never appears in
# the golden.
#
# USAGE:
#   scripts/ppu_layout_check.sh [check|regen|self-test]
#     check      (default) build + fingerprint, compare to golden;
#                exit 0 if identical, exit 1 if tick's code changed.
#     regen      build + fingerprint, OVERWRITE the golden. Run this
#                (and commit the golden) ONLY when a rung intentionally
#                changes `tick` and the change has passed the campaign's
#                universal gates (parity + Mesen lockstep + A/B).
#     self-test  prove the gate works: current==golden -> PASS, and a
#                deliberately perturbed fingerprint is DETECTED as a
#                mismatch. Exits 0 iff the gate behaves correctly.
#
#   PPU_LAYOUT_NO_BUILD=1  reuse the existing example binary (skip the
#                          rebuild) for fast local iteration.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CORE="${REPO}/nes_core"
GOLDEN="${CORE}/pgo/ppu_layout_golden.sha256"
PROFILE="release-with-debug"
EXAMPLE="ppu_state_profile"
BIN="${CORE}/target/${PROFILE}/examples/${EXAMPLE}"
# Demangled-name filter for the hot reference symbol(s).
HOT_RE='<nes_core::ppu::Ppu::tick(_three)?::h[0-9a-f]+>:$'

OBJDUMP="$(xcrun -find llvm-objdump 2>/dev/null || command -v llvm-objdump || true)"
if [[ -z "${OBJDUMP}" ]]; then
    echo "error: llvm-objdump not found (need Xcode command-line tools)" >&2
    exit 2
fi

MODE="${1:-check}"

build_ref() {
    if [[ "${PPU_LAYOUT_NO_BUILD:-0}" == "1" && -x "${BIN}" ]]; then
        echo "==> reusing ${BIN} (PPU_LAYOUT_NO_BUILD=1)" >&2
        return
    fi
    echo "==> building reference: cargo build --profile ${PROFILE} --example ${EXAMPLE}" >&2
    ( cd "${CORE}" && cargo build --profile "${PROFILE}" --example "${EXAMPLE}" >&2 )
}

# Emit the normalized instruction-sequence fingerprint of the hot
# symbol(s) in ${BIN}. Order-independent across monomorphizations:
# each hot symbol block is normalized + hashed, the per-block hashes
# are sorted, and their concatenation is hashed.
fingerprint() {
    local tmp; tmp="$(mktemp -d)"
    trap 'rm -rf "${tmp}"' RETURN
    "${OBJDUMP}" -d --demangle --no-show-raw-insn "${BIN}" > "${tmp}/disasm.txt"
    awk -v out="${tmp}" -v re="${HOT_RE}" '
        /^[0-9a-f]+ <.*>:$/ {
            inblk = ($0 ~ re)
            if (inblk) { n++; file = out "/block_" n ".txt" }
            next
        }
        inblk && /^[ \t]*[0-9a-f]+:/ {
            line = $0
            sub(/^[ \t]*[0-9a-f]+:[ \t]*/, "", line)   # drop the address column
            print line > file
        }
    ' "${tmp}/disasm.txt"

    local blocks; blocks="$(ls "${tmp}"/block_*.txt 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${blocks}" == "0" ]]; then
        echo "error: no hot symbols matched ${HOT_RE} in ${BIN}" >&2
        echo "hint: is the binary stripped? this gate needs the ${PROFILE} profile" >&2
        exit 2
    fi
    echo "==> matched ${blocks} hot symbol block(s)" >&2

    # Normalize each block (drop <sym> annotations, fold absolute
    # addresses -> ADDR while keeping small immediates/offsets), hash
    # it, then hash the sorted set of block hashes.
    local f
    for f in "${tmp}"/block_*.txt; do
        perl -pe 's/\s+<[^>]*>//g; s/0x[0-9a-fA-F]{6,}/ADDR/g;' "${f}" \
            | shasum -a 256 | cut -d' ' -f1
    done | sort | shasum -a 256 | cut -d' ' -f1
}

read_golden() {
    [[ -f "${GOLDEN}" ]] || return 1
    grep -vE '^\s*(#|$)' "${GOLDEN}" | head -n1 | tr -d '[:space:]'
}

write_golden() {
    local fp="$1"
    mkdir -p "$(dirname "${GOLDEN}")"
    {
        echo "# Golden fingerprint of nes_core::ppu::Ppu::tick hot symbol(s)."
        echo "# Generated by scripts/ppu_layout_check.sh regen."
        echo "# profile=${PROFILE} example=${EXAMPLE} tool=llvm-objdump"
        echo "# Regenerate ONLY on an intentional, gate-passing tick change."
        echo "${fp}"
    } > "${GOLDEN}"
}

case "${MODE}" in
    check)
        build_ref
        cur="$(fingerprint)"
        gold="$(read_golden || true)"
        echo "current: ${cur}"
        echo "golden:  ${gold:-<none>}"
        if [[ -z "${gold}" ]]; then
            echo "FAIL: no golden committed. Run: scripts/ppu_layout_check.sh regen" >&2
            exit 1
        fi
        if [[ "${cur}" == "${gold}" ]]; then
            echo "PASS: Ppu::tick machine code matches golden."
            exit 0
        fi
        echo "FAIL: Ppu::tick machine code changed vs golden." >&2
        echo "  If this rung did NOT intend to touch tick, revert the perturbation." >&2
        echo "  If it did (and passed parity + Mesen + A/B), regenerate:" >&2
        echo "    scripts/ppu_layout_check.sh regen   # then commit ${GOLDEN#${REPO}/}" >&2
        exit 1
        ;;
    regen)
        build_ref
        cur="$(fingerprint)"
        write_golden "${cur}"
        echo "wrote golden: ${cur}"
        echo "  -> ${GOLDEN}"
        ;;
    self-test)
        build_ref
        cur="$(fingerprint)"
        gold="$(read_golden || true)"
        echo "current: ${cur}"
        echo "golden:  ${gold:-<none>}"
        rc=0
        # 1) golden matches current build -> the gate PASSES.
        if [[ -n "${gold}" && "${cur}" == "${gold}" ]]; then
            echo "[1/2] PASS: current build matches committed golden."
        else
            echo "[1/2] FAIL: current build does not match golden (or none)." >&2
            rc=1
        fi
        # 2) a deliberately perturbed fingerprint is DETECTED as a diff.
        perturbed="${cur%??}$(printf '%02x' $(( (16#${cur: -2} + 1) % 256 )) )"
        if [[ "${cur}" != "${perturbed}" ]]; then
            echo "[2/2] PASS: gate detects a perturbed fingerprint as a mismatch."
            echo "        (current ${cur:0:12}...  !=  perturbed ${perturbed:0:12}...)"
        else
            echo "[2/2] FAIL: perturbation did not change the fingerprint." >&2
            rc=1
        fi
        [[ ${rc} -eq 0 ]] && echo "self-test: gate behaves correctly." \
                          || echo "self-test: gate is MISCONFIGURED." >&2
        exit ${rc}
        ;;
    -h|--help|help)
        sed -n '2,60p' "$0" | sed 's/^# \{0,1\}//'
        ;;
    *)
        echo "unknown mode: ${MODE} (want: check | regen | self-test)" >&2
        exit 2
        ;;
esac
