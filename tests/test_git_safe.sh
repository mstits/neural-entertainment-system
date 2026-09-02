#!/usr/bin/env bash
# Fixture test for git_safe.sh: two worktrees, one dirty+live, one clean.
# Exercises all four guarded ops and asserts refuse/allow correctly, then
# the revert-verify leg: prove the *unwrapped* real git would have gone
# through (i.e. the guard is the thing stopping it, not something else).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_SAFE="$HERE/../scripts/git_safe.sh"
# Build the fixture under a real mktemp, not the repo's own receipts/ dir
# (receipts/ is not gitignored - a fixture built there would leave stray
# files for git status to see if the trap ever failed to fire). `mktemp -d`
# on macOS lands under $TMPDIR (/var/folders/..., a symlink to
# /private/var/folders/...); resolve it with `pwd -P` (physical path, not
# `pwd`'s logical one) so it matches what `git rev-parse --show-toplevel`
# resolves to internally - otherwise the guard's path-derived lock slug for
# a worktree would silently differ from what `git worktree list` reports
# for the same directory, defeating the lookup.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/git_safe_test.XXXXXX")"
WORK="$(cd "$WORK" && pwd -P)"
trap 'rm -rf "$WORK"' EXIT

pass=0
fail=0
check() {
  local desc="$1" expect_rc="$2"; shift 2
  local rc=0
  "$@" >"$WORK/out.log" 2>&1 || rc=$?
  if [[ "$rc" -eq "$expect_rc" ]]; then
    echo "PASS: $desc"
    pass=$((pass+1))
  else
    echo "FAIL: $desc (expected rc=$expect_rc, got rc=$rc)"
    sed 's/^/    /' "$WORK/out.log"
    fail=$((fail+1))
  fi
}

# ---- build fixture repo -------------------------------------------------
REPO="$WORK/repo"
git init -q -b main "$REPO"
cd "$REPO"
git config user.email test@example.com
git config user.name "Test"
echo one > f.txt
git add f.txt
git commit -qm "initial"

WT="$WORK/wt_feature"
git worktree add -q -b feature "$WT" main
echo dirty >> "$WT/f.txt"   # worktree B is now dirty

# Fake a live session for worktree B: use this test process's own PID+lstart
# (guaranteed alive for the duration of the test).
TEST_PID=$$
TEST_START="$(ps -o lstart= -p "$TEST_PID")"

COMMON_DIR="$(git rev-parse --git-common-dir)"
case "$COMMON_DIR" in /*) : ;; *) COMMON_DIR="$REPO/$COMMON_DIR" ;; esac
LOCK_DIR="$COMMON_DIR/git_safe/locks"
mkdir -p "$LOCK_DIR"
slug() { printf '%s' "$1" | sed -e 's#[^A-Za-z0-9]#_#g'; }
{
  echo "$TEST_PID"
  echo "$TEST_START"
  echo "$WT"
  echo "2026-01-01T00:00:00Z"
} > "$LOCK_DIR/$(slug "$WT").lock"

export GIT_SAFE_SESSION_PID="$TEST_PID"
export GIT_SAFE_SESSION_START="$TEST_START"

echo "== fixture: main worktree=$REPO  dirty+live other worktree=$WT =="

# 1. update-ref against refs/heads/main -> refused (rc=1)
check "update-ref refs/heads/main refused while other WT dirty+live" 1 \
  "$GIT_SAFE" -C "$REPO" update-ref refs/heads/main HEAD

# 2. reset --hard on main -> refused
check "reset --hard on main refused while other WT dirty+live" 1 \
  "$GIT_SAFE" -C "$REPO" reset --hard HEAD

# 3. stash push on main -> refused
echo more >> "$REPO/f.txt"
check "stash push on main refused while other WT dirty+live" 1 \
  "$GIT_SAFE" -C "$REPO" stash push -m test

# 4. checkout main from elsewhere / off main -> refused (run from REPO, on main)
check "checkout away from main refused while other WT dirty+live" 1 \
  "$GIT_SAFE" -C "$REPO" checkout -b throwaway

# ---- clear the blocker: worktree B lock now points at a dead PID --------
{
  echo "999999999"
  echo "Mon Jan  1 00:00:00 2026"
  echo "$WT"
  echo "2026-01-01T00:00:00Z"
} > "$LOCK_DIR/$(slug "$WT").lock"

check "update-ref on main allowed once other WT's session is dead" 0 \
  "$GIT_SAFE" -C "$REPO" update-ref refs/heads/main HEAD

# ---- clear the blocker the other way: worktree B is clean ---------------
{
  echo "$TEST_PID"
  echo "$TEST_START"
  echo "$WT"
  echo "2026-01-01T00:00:00Z"
} > "$LOCK_DIR/$(slug "$WT").lock"
git -C "$WT" checkout -q -- f.txt   # clean it up

check "reset --hard on main allowed once other WT is clean (still live)" 0 \
  "$GIT_SAFE" -C "$REPO" reset --hard HEAD

# ---- non-guarded commands always pass through untouched -----------------
echo dirty2 >> "$WT/f.txt"  # re-dirty + still live, to prove selectivity
check "plain 'git status' on main passes through even while blocked-if-guarded" 0 \
  "$GIT_SAFE" -C "$REPO" status

check "'git log' passes through" 0 \
  "$GIT_SAFE" -C "$REPO" log --oneline -1

# ---- override escape hatch, logged ---------------------------------------
GIT_SAFE_OVERRIDE=1 check "override flag bypasses the refusal" 0 \
  "$GIT_SAFE" -C "$REPO" update-ref refs/heads/main HEAD
if grep -q OVERRIDE "$LOCK_DIR/override.log" 2>/dev/null; then
  echo "PASS: override was logged"
  pass=$((pass+1))
else
  echo "FAIL: override was not logged"
  fail=$((fail+1))
fi

# ---- fallback path: no session envs exported, wrapper uses $PPID --------
# This test script's own PID is what git_safe.sh sees as $PPID when it
# invokes the wrapper directly (no subshell in between), and that's the
# same PID the wt_feature lock above was written under - so the fallback
# should refuse exactly like the explicit-export path did.
echo dirty4 >> "$WT/f.txt"
rc=0
(
  unset GIT_SAFE_SESSION_PID GIT_SAFE_SESSION_START
  "$GIT_SAFE" -C "$REPO" update-ref refs/heads/main HEAD
) >"$WORK/out.log" 2>&1 || rc=$?
if [[ $rc -eq 1 ]]; then
  echo "PASS: no session envs exported - \$PPID fallback still refuses correctly"
  pass=$((pass+1))
else
  echo "FAIL: no session envs exported - \$PPID fallback (expected rc=1, got rc=$rc)"
  sed 's/^/    /' "$WORK/out.log"
  fail=$((fail+1))
fi

# ---- exact install form: no -C flag at all, cwd is the repo -------------
# The plan's install path is `alias git=git_safe.sh` in the operator's
# shell, so every real invocation has GLOBAL_ARGS empty (no leading -C,
# --git-dir, etc.) - every check above passes -C "$REPO" and so never
# exercises "${GLOBAL_ARGS[@]}" as a truly empty array under `set -u`.
# WT is still dirty+live from the fallback-path test above.
rc=0
( cd "$REPO" && "$GIT_SAFE" update-ref refs/heads/main HEAD ) \
  >"$WORK/out.log" 2>&1 || rc=$?
if [[ $rc -eq 1 ]]; then
  echo "PASS: no -C flag (alias install form) still refuses while other WT dirty+live"
  pass=$((pass+1))
else
  echo "FAIL: no -C flag (alias install form) (expected rc=1, got rc=$rc)"
  sed 's/^/    /' "$WORK/out.log"
  fail=$((fail+1))
fi

echo
echo "== revert-verify: unwrap and prove the guard, not something else, was blocking =="
git -C "$WT" checkout -q -- f.txt 2>/dev/null || true
echo dirty3 >> "$WT/f.txt"
REAL_GIT="$(command -v git)"
check "same op with the REAL git (unwrapped) succeeds - isolates the guard as cause" 0 \
  "$REAL_GIT" -C "$REPO" update-ref refs/heads/main HEAD

echo
echo "$pass passed, $fail failed"
[[ $fail -eq 0 ]]
