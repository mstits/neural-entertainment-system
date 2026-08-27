#!/usr/bin/env bash
# Prove tests/test_rust_unwitnessed_semantics.py has teeth by ACTUAL REVERT.
#
# A guard that passes on the fixed tree proves nothing on its own — seven
# vacuous gates have shipped in this repo. The bar the predecessor sweep set
# is a real revert: restore the ORIGINAL rewards.rs and watch the assertions
# fail, then restore only ONE of a sibling pair and confirm a half-fix is
# still caught.
#
# Nothing is committed or left behind: the working file is saved to the
# scratch dir and restored on exit, including on error.
#
#   bash scripts/verify_unwitnessed_guard_by_revert.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

RS="nes_core/src/rewards.rs"
WORK="$(mktemp -d)"
TEST="tests/test_rust_unwitnessed_semantics.py"
PYTEST=".venv/bin/pytest"

cp "$RS" "$WORK/rewards.fixed.rs"
git show "HEAD:$RS" > "$WORK/rewards.orig.rs"

restore() { cp "$WORK/rewards.fixed.rs" "$RS"; }
trap 'restore; rm -rf "$WORK"' EXIT

# Source-lint assertions only. The compiled-binary checks need a rebuild,
# which a source revert cannot produce without a 2-minute cargo cycle; they
# are exercised separately by the normal suite run.
LINT_K="unwitnessed_constant or provenance_tag or retracted_clause or smb_"

count_failures() {
  $PYTEST "$TEST" -q --timeout=120 -p no:randomly -k "$LINT_K" 2>&1 \
    | grep -cE "^FAILED " || true
}

echo "=============================================================="
echo "BASELINE — the fixed tree must be green"
echo "=============================================================="
BASE=$(count_failures)
echo "failures on the fixed tree: $BASE"
if [ "$BASE" -ne 0 ]; then
  echo "ABORT: the guard is not green before the experiment starts."
  exit 1
fi

echo
echo "=============================================================="
echo "ARM 1 — FULL REVERT: restore the pre-annotation rewards.rs"
echo "=============================================================="
cp "$WORK/rewards.orig.rs" "$RS"
FULL=$(count_failures)
echo "failures on full revert: $FULL"
$PYTEST "$TEST" -q --timeout=120 -p no:randomly -k "$LINT_K" 2>&1 \
  | grep -E "^FAILED " | sed 's/^/    /'
restore

echo
echo "=============================================================="
echo "ARM 2 — HALF-FIX: restore ONLY Punch-Out's \$0001 sibling,"
echo "        leaving \$000A annotated. A guard that keys on the pair"
echo "        rather than on each member would pass this."
echo "=============================================================="
python3 - "$WORK" <<'PY'
import re, sys, pathlib
work = pathlib.Path(sys.argv[1])
rs = pathlib.Path("nes_core/src/rewards.rs")
fixed = work.joinpath("rewards.fixed.rs").read_text()

# Strip the RAM_MATCH_ID tag block only: everything from its PURITY marker
# down to the declaration, replaced by the original bare one-liner.
lines = fixed.splitlines()
decl = next(i for i, l in enumerate(lines)
            if "const RAM_MATCH_ID: usize = 0x0001;" in l)
start = decl
while start > 0 and "PURITY: UNWITNESSED-EXTERNAL" not in lines[start]:
    start -= 1
orig = ("    const RAM_MATCH_ID: usize = 0x0001; // win latch: 0 during "
        "bout, !=0 at the winning KO/TKO")
out = lines[:start] + [orig] + lines[decl + 1:]
rs.write_text("\n".join(out) + "\n")
print(f"    reverted RAM_MATCH_ID tag (source lines {start+1}..{decl+1})")
PY
HALF=$(count_failures)
echo "failures on the half-fix: $HALF"
$PYTEST "$TEST" -q --timeout=120 -p no:randomly -k "$LINT_K" 2>&1 \
  | grep -E "^FAILED " | sed 's/^/    /'
restore

echo
echo "=============================================================="
echo "ARM 2b — SIBLING GROUP: delete the ONE tag block shared by"
echo "         Metroid's four win bytes + the 32-hit threshold."
echo "         All five must fail; the rest of the file stays green."
echo "=============================================================="
python3 - <<'PY'
import pathlib
rs = pathlib.Path("nes_core/src/rewards.rs")
lines = rs.read_text().splitlines()
decl = next(i for i, l in enumerate(lines)
            if "const RAM_MB_STATE: usize = 0x0098;" in l)
start = decl
while start > 0 and "PURITY: UNWITNESSED-EXTERNAL" not in lines[start]:
    start -= 1
end = decl
while "EARNS IT:" not in lines[end]:
    end -= 1
# Drop the tag block, keep every declaration untouched.
out = lines[:start] + lines[decl:]
rs.write_text("\n".join(out) + "\n")
print(f"    deleted Metroid's shared tag block (lines {start+1}..{decl})")
PY
SIB=$(count_failures)
echo "failures on the sibling-group revert: $SIB"
$PYTEST "$TEST" -q --timeout=120 -p no:randomly -k "$LINT_K" 2>&1 \
  | grep -E "^FAILED " | sed 's/^/    /'
restore

echo
echo "=============================================================="
echo "ARM 3 — OVER-WITHDRAWAL: move an SMB constant."
echo "        SMB is earned by a completely solved game; the guard"
echo "        must fail in this direction too, not only on omission."
echo "=============================================================="
python3 - <<'PY'
import pathlib
rs = pathlib.Path("nes_core/src/rewards.rs")
s = rs.read_text()
assert "const RAM_FLOAT_STATE: usize = 0x001D;" in s
rs.write_text(s.replace("const RAM_FLOAT_STATE: usize = 0x001D;",
                        "const RAM_FLOAT_STATE: usize = 0x001E;", 1))
print("    moved SMB RAM_FLOAT_STATE 0x001D -> 0x001E")
PY
OVER=$(count_failures)
echo "failures on the SMB mutation: $OVER"
restore

echo
echo "=============================================================="
echo "VERDICT"
echo "=============================================================="
printf "  baseline (fixed tree) : %s failures\n" "$BASE"
printf "  full revert           : %s failures\n" "$FULL"
printf "  half-fix (1 sibling)  : %s failures\n" "$HALF"
printf "  sibling group (metroid): %s failures\n" "$SIB"
printf "  SMB over-withdrawal   : %s failures\n" "$OVER"
echo
RC=0
[ "$BASE" -eq 0 ] || { echo "FAIL: baseline not green"; RC=1; }
[ "$FULL" -ge 14 ] || { echo "FAIL: full revert must fail >=14 assertions"; RC=1; }
[ "$HALF" -ge 1 ]  || { echo "FAIL: a half-fix must still be caught"; RC=1; }
[ "$SIB" -ge 5 ]   || { echo "FAIL: a sibling-group revert must fail >=5"; RC=1; }
[ "$OVER" -ge 1 ]  || { echo "FAIL: over-withdrawal must be caught"; RC=1; }
[ "$FULL" -gt "$HALF" ] || { echo "FAIL: full revert must fail MORE than a half-fix"; RC=1; }
[ $RC -eq 0 ] && echo "PASS — the guard has teeth in all three directions."
exit $RC
