#!/usr/bin/env bash
# Long-running ASM-vs-Rust 6502 differential fuzzer soak.
#
# Builds the diff-fuzz binary in release mode, then runs N iterations
# at the requested stream length. Logs per-chunk progress to
# logs/asm_diff_fuzz_soak_<timestamp>.log so the soak can run
# unattended overnight and the user can grep for "DIVERGENCE" to find
# any failures.
#
# Default sizing: 216M cases × 16 instructions/stream ≈ 3.5B
# instructions, which hits the README's "hours-scale soak" gate at
# roughly 6 hours wall on an M4 Max.
#
# Usage:
#   ./scripts/asm_diff_fuzz_soak.sh                    # 6h default
#   ITERATIONS=10000000 INSTRS=16 ./scripts/asm_diff_fuzz_soak.sh
#   SEED=42 ITERATIONS=1000000 ./scripts/asm_diff_fuzz_soak.sh   # reproducible

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

ITERATIONS=${ITERATIONS:-216000000}
INSTRS=${INSTRS:-16}
SEED=${SEED:-$(date +%s)}

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/asm_diff_fuzz_soak_${TS}.log"

echo "asm-diff-fuzz soak: iterations=$ITERATIONS instrs=$INSTRS seed=$SEED"
echo "log: $LOG"
echo "build:"
PATH="$HOME/.cargo/bin:$PATH" cargo build --release \
    --features asm_cpu \
    --example asm_diff_fuzz \
    --manifest-path nes_core/Cargo.toml 2>&1 | tail -3

echo "starting soak at $(date)"
"$REPO/nes_core/target/release/examples/asm_diff_fuzz" \
    "$ITERATIONS" "$INSTRS" "$SEED" 2>&1 | tee "$LOG"
exit_code=${PIPESTATUS[0]}

echo "soak ended at $(date), exit=$exit_code"
if [ "$exit_code" -eq 2 ]; then
    echo "DIVERGENCE(S) FOUND — grep '$LOG' for 'DIVERGENCE'"
elif [ "$exit_code" -ne 0 ]; then
    echo "harness errored (non-divergence)"
else
    echo "soak clean: 0 divergences"
fi
exit "$exit_code"
