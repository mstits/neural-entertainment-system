#!/usr/bin/env bash
# Supplementary prior-art verification, run 2026-08-25 (second pass).
# Confirms three things search_prior_art.sh's grep-based pass left implicit:
#   1. entropy_guard's config coverage is EXACTLY {1-2, 1-3} backward — never
#      any 1-1 config, including the v4/v6/consol lineage v27/v28 descend from.
#   2. The unconditional "ANTI-COLLAPSE GUARD" in trainer.py was built (73bd244)
#      for the MIRROR-IMAGE failure (entropy rising toward ln(A) / melting-to-
#      random), not the sharpening/decay-to-zero direction v27/v28 exhibit.
#   3. The 1-2 OPTIONS control run that collapsed -74%/200iters had BOTH
#      kl_anchor (beta not yet decayed much at iter 200) AND sil.enabled:true
#      live throughout -- the closest existing "combat forgetting during
#      continued PPO, still failed" receipt.
# Read-only: git log/show, grep, sed on tracked files only.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

section() { printf '\n===== %s =====\n' "$*"; }

section "1. entropy_guard: full config census (every configs/*.yaml)"
grep -l "entropy_guard" configs/*.yaml 2>/dev/null || echo "(none found by this pattern)"
echo "--- confirm ABSENT from every 1-1 lineage config, incl. v27/v28 ---"
for f in configs/mario_1_1_backward.yaml configs/mario_1_1_backward_v4.yaml \
         configs/mario_1_1_backward_v6.yaml configs/mario_1_1_backward_consol.yaml \
         configs/mario_1_1_v27_seed0.yaml configs/mario_1_1_v28_seed0.yaml; do
  if grep -q "entropy_guard" "$f" 2>/dev/null; then
    echo "$f: HAS entropy_guard"
  else
    echo "$f: no entropy_guard"
  fi
done

section "2. Origin commit of the unconditional ANTI-COLLAPSE GUARD (trainer.py)"
git log -1 --format='%H %s' 73bd244
echo "--- its own stated trigger direction (entropy RISING, i.e. melting) ---"
git show 73bd244 --format='' | grep -A3 "entropy exceeds 90%" | head -6 || true
echo "--- v27/v28's actual entropy direction (falling), from v28 seed3 best run ---"
echo "iter 19: 1.4656 | 59: 0.7318 | 119: 0.2910 | 179: 0.0796 | 239: 0.0716"
echo "ln(6) = $(python3 -c 'import math; print(math.log(6))')  => 90% threshold = $(python3 -c 'import math; print(0.90*math.log(6))')"
echo "Every value above is below the 90%-of-ln(6) melting threshold at every logged iter -- the guard's fire condition is never approached in either direction that matters (it is a HIGH-entropy trigger; ours is a low-entropy trajectory throughout)."

section "3. KL-anchor decay arithmetic for the -74% (1-2 OPTIONS control) run"
grep -n "rollout_steps\|num_envs\|kl_beta_start\|kl_beta_end\|kl_beta_decay_steps\|sil:\|enabled: true\|bc_coef" configs/mario_1_2_options.yaml | sed -n '1,20p'
python3 - <<'PY'
rollout_steps = 1536
num_envs = 60
iters = 200
decay_steps = 50_000_000
beta_start, beta_end = 0.5, 0.01
steps = rollout_steps * num_envs * iters
frac = min(steps / decay_steps, 1.0)
beta_at_200 = beta_start - frac * (beta_start - beta_end)
print(f"steps consumed by iter 200: {steps:,} ({frac:.1%} of decay horizon)")
print(f"kl_beta at iter 200 (linear decay assumed): {beta_at_200:.3f} (start {beta_start}, end {beta_end})")
PY
echo "--- kl_anchor.py's own stated purpose (docstring) ---"
sed -n '1,17p' src/training/kl_anchor.py

section "4. Sanity check: was this the SAME run across Phase-3 and OPTIONS docs?"
grep -n "Control:.*Phase-3\|31/100 pooled" docs/proposals/OPTIONS_PREREG_2026-08-22.md || true

section "5. BackwardEntropyGuard's own docstring: what phenomenon it was scoped to"
sed -n '/class BackwardEntropyGuard/,/^class /p' src/training/trainer.py | head -40
