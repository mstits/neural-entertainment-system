#!/usr/bin/env bash
# git_safe.sh - refuse the four destructive ops that clobbered refs/heads/main
# once already (git update-ref race, 08-26) when another worktree's live
# session still has uncommitted work sitting in it.
#
# Guarded: update-ref, stash (push/save/pop/apply/drop/clear/branch), reset
# --hard, checkout <branch> - only when the operation touches refs/heads/main
# (moves it, or switches HEAD onto/off it) AND some OTHER worktree is both
# dirty (git status --porcelain is non-empty) and live (a session fingerprint
# for it is still running). Every other git command passes straight through.
#
# Liveness reuses run_lock.py's PID+lstart technique (run_lock.py:47-83):
# a bare PID isn't identity because macOS can reissue a dead PID to an
# unrelated process within seconds of boot; comparing `ps -o lstart=` catches
# that. The PID recorded is the *session* (the shell you're typing into), not
# this short-lived wrapper process, so export these once per shell before
# aliasing git:
#
#   export GIT_SAFE_SESSION_PID=$$
#   export GIT_SAFE_SESSION_START="$(ps -o lstart= -p $$)"
#   alias git=/path/to/git_safe.sh
#
# Without that export the wrapper falls back to $PPID (its parent shell) on
# a best-effort basis - works for a plain interactive shell (tests/
# test_git_safe.sh's "session envs unset" case proves that leg), degrades
# for nested subshells and some non-interactive callers, where the true
# session PID and $PPID diverge and coordination silently stops working.
#
# Install: alias it in the operator's shell profile (~/.zshrc). This never
# writes into the target repo, so it is safe to turn on even for a
# read-only-checkout repo. A git-native `reference-transaction` hook can
# also enforce part of this (it vetoes ref updates in its "prepared"
# state) but needs a write into .git/hooks/ and can't see git-stash's
# context cleanly - see the coordination-layer report for why the wrapper,
# not the hook, is the one to install first.
set -euo pipefail

REAL_GIT="$(command -v git)"
if [[ -z "$REAL_GIT" ]]; then
  echo "git_safe: no 'git' binary on PATH" >&2
  exit 1
fi
# If we've been aliased to ourselves (alias git=git_safe.sh puts our own
# dirname ahead of the real git on PATH), walk PATH for the next git.
SELF_REAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
if [[ "$(cd "$(dirname "$REAL_GIT")" && pwd)/$(basename "$REAL_GIT")" == "$SELF_REAL" ]]; then
  IFS=':' read -r -a _path_dirs <<< "$PATH"
  REAL_GIT=""
  for d in "${_path_dirs[@]}"; do
    if [[ -x "$d/git" ]]; then
      cand="$(cd "$d" && pwd)/git"
      [[ "$cand" == "$SELF_REAL" ]] && continue
      REAL_GIT="$cand"
      break
    fi
  done
  if [[ -z "$REAL_GIT" ]]; then
    echo "git_safe: could not find the real git binary past self on PATH" >&2
    exit 1
  fi
fi

# ---- session fingerprint -----------------------------------------------
SESSION_PID="${GIT_SAFE_SESSION_PID:-$PPID}"
SESSION_START="${GIT_SAFE_SESSION_START:-}"
if [[ -z "$SESSION_START" ]]; then
  SESSION_START="$(ps -o lstart= -p "$SESSION_PID" 2>/dev/null || true)"
fi

# ---- split leading global flags (esp. -C <dir>) from the subcommand ----
# Every internal lookup below must see the same -C the caller passed, or
# "current worktree" silently resolves to wherever this wrapper happens to
# be running from instead of the repo the caller named.
GLOBAL_ARGS=()
REST_ARGS=("$@")
i=0
while [[ $i -lt ${#REST_ARGS[@]} ]]; do
  a="${REST_ARGS[$i]}"
  case "$a" in
    -C|--git-dir|--work-tree|--namespace)
      GLOBAL_ARGS+=("$a" "${REST_ARGS[$((i+1))]:-}")
      i=$((i+2))
      ;;
    -c)
      GLOBAL_ARGS+=("$a" "${REST_ARGS[$((i+1))]:-}")
      i=$((i+2))
      ;;
    -*)
      GLOBAL_ARGS+=("$a")
      i=$((i+1))
      ;;
    *)
      break
      ;;
  esac
done
subcmd="${REST_ARGS[$i]:-}"
sub_i=$i

# ---- locate repo, bail out of the guard entirely if not in one ---------
if ! TOPLEVEL="$("$REAL_GIT" ${GLOBAL_ARGS[@]+"${GLOBAL_ARGS[@]}"} rev-parse --show-toplevel 2>/dev/null)"; then
  exec "$REAL_GIT" "$@"
fi
COMMON_DIR="$("$REAL_GIT" ${GLOBAL_ARGS[@]+"${GLOBAL_ARGS[@]}"} rev-parse --git-common-dir 2>/dev/null)"
case "$COMMON_DIR" in
  /*) : ;;
  *) COMMON_DIR="$TOPLEVEL/$COMMON_DIR" ;;
esac
LOCK_DIR="$COMMON_DIR/git_safe/locks"
mkdir -p "$LOCK_DIR" 2>/dev/null || true

slug_for() {
  # Stable, filesystem-safe id for a worktree path.
  printf '%s' "$1" | sed -e 's#[^A-Za-z0-9]#_#g'
}

# Refresh this worktree's activity lock unconditionally - every invocation
# (guarded or not) is proof the session is still around.
if [[ -n "$SESSION_START" ]]; then
  {
    printf '%s\n' "$SESSION_PID"
    printf '%s\n' "$SESSION_START"
    printf '%s\n' "$TOPLEVEL"
    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$LOCK_DIR/$(slug_for "$TOPLEVEL").lock" 2>/dev/null || true
fi

pid_is_live() {
  # Port of run_lock.lock_pid_is_live: a bare PID match isn't identity
  # (PID reuse); compare start-time fingerprints. Unknown => conservative
  # "live" (matches run_lock's stance: never let unknown read as safe).
  local pid="$1" recorded_start="$2"
  kill -0 "$pid" 2>/dev/null || return 1
  [[ -z "$recorded_start" ]] && return 0
  local current_start
  current_start="$(ps -o lstart= -p "$pid" 2>/dev/null || true)"
  [[ -z "$current_start" || "$current_start" == "$recorded_start" ]]
}

worktree_is_dirty() {
  local wt="$1"
  [[ -n "$("$REAL_GIT" -C "$wt" status --porcelain 2>/dev/null)" ]]
}

worktree_is_live() {
  local wt="$1"
  local lock="$LOCK_DIR/$(slug_for "$wt").lock"
  [[ -f "$lock" ]] || return 1
  local pid start
  pid="$(sed -n '1p' "$lock" 2>/dev/null)"
  start="$(sed -n '2p' "$lock" 2>/dev/null)"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  pid_is_live "$pid" "$start"
}

# ---- is this invocation one of the four guarded operations, aimed at main?
is_main_ref() {
  case "$1" in
    refs/heads/main|heads/main|main) return 0 ;;
    HEAD)
      local cur
      cur="$("$REAL_GIT" ${GLOBAL_ARGS[@]+"${GLOBAL_ARGS[@]}"} symbolic-ref -q --short HEAD 2>/dev/null || true)"
      [[ "$cur" == "main" ]]
      ;;
    *) return 1 ;;
  esac
}

current_branch() {
  "$REAL_GIT" ${GLOBAL_ARGS[@]+"${GLOBAL_ARGS[@]}"} symbolic-ref -q --short HEAD 2>/dev/null || true
}

needs_guard=0
if [[ -n "$subcmd" ]]; then
  args=("${REST_ARGS[@]}")
  case "$subcmd" in
    update-ref)
      # First non-flag positional arg after 'update-ref' is the ref name
      # (the -d form: update-ref -d <ref> is also covered - same scan).
      for ((j=sub_i+1; j<${#args[@]}; j++)); do
        arg="${args[$j]}"
        [[ "$arg" == -* ]] && continue
        if is_main_ref "$arg"; then needs_guard=1; fi
        break
      done
      ;;
    reset)
      for ((j=sub_i+1; j<${#args[@]}; j++)); do
        [[ "${args[$j]}" == "--hard" ]] && { [[ "$(current_branch)" == "main" ]] && needs_guard=1; break; }
      done
      ;;
    stash)
      sub="${args[$((sub_i+1))]:-push}"
      case "$sub" in
        push|save|pop|apply|drop|clear|branch)
          [[ "$(current_branch)" == "main" ]] && needs_guard=1
          ;;
      esac
      ;;
    checkout|switch)
      cur="$(current_branch)"
      target=""
      for ((j=sub_i+1; j<${#args[@]}; j++)); do
        arg="${args[$j]}"
        [[ "$arg" == -* ]] && continue
        [[ "$arg" == "--" ]] && break
        target="$arg"
        break
      done
      if [[ "$cur" == "main" ]] || is_main_ref "$target"; then
        needs_guard=1
      fi
      ;;
  esac
fi

if [[ $needs_guard -eq 1 ]]; then
  blockers=()
  while IFS= read -r wt; do
    [[ "$wt" == "$TOPLEVEL" ]] && continue
    [[ -d "$wt" ]] || continue
    if worktree_is_dirty "$wt" && worktree_is_live "$wt"; then
      blockers+=("$wt")
    fi
  done < <("$REAL_GIT" ${GLOBAL_ARGS[@]+"${GLOBAL_ARGS[@]}"} worktree list --porcelain 2>/dev/null | awk '/^worktree /{print substr($0,10)}')

  if [[ ${#blockers[@]} -gt 0 ]]; then
    {
      echo "git_safe: REFUSED - '$subcmd' would touch refs/heads/main while" \
           "another live worktree still has uncommitted work:"
      for b in "${blockers[@]}"; do
        echo "  - $b"
      done
      echo "Finish or stash-and-commit that work first, or override with"
      echo "  GIT_SAFE_OVERRIDE=1 git $*"
      echo "(the override is logged to $LOCK_DIR/override.log)"
    } >&2
    if [[ "${GIT_SAFE_OVERRIDE:-0}" != "1" ]]; then
      exit 1
    fi
    printf '%s\tOVERRIDE\t%s\t%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TOPLEVEL" "$*" >> "$LOCK_DIR/override.log" 2>/dev/null || true
  fi
fi

exec "$REAL_GIT" "$@"
