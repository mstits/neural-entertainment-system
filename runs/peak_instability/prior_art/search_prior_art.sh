#!/usr/bin/env bash
# Prior-art search for the "peak-then-collapse" / entropy-collapse phenomenon
# in v27/v28. Re-run from the repo root to reproduce every claim in
# search_output.txt. Read-only: git log, grep, cat/sed on tracked files only.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

section() { printf '\n===== %s =====\n' "$*"; }

section "1. Commit-message search: collapse / instability / degradation / entropy / plateau"
git log --oneline --all -i --grep="collapse" | head -80
echo "--- instability ---"
git log --oneline --all -i --grep="instability"
echo "--- entropy (all) ---"
git log --oneline --all -i --grep="entropy" | head -80

section "2. The -74% figure: every place it is quoted"
grep -rn "74%\|-74\b" docs/ CLAIMS.md CHANGELOG.md 2>/dev/null

section "3. Anti-collapse / KL-anchor / entropy-floor machinery: where it lives"
grep -rl "anticollapse\|anti_collapse\|anti-collapse" --include="*.py" --include="*.md" . | sort -u
echo "--- kl_anchor.py / entropy_config.py consumers ---"
grep -rl "kl_anchor\|entropy_floor" --include="*.py" --include="*.md" --include="*.yaml" . | sort -u

section "4. Was entropy_floor / entropy_guard / kl_anchor_checkpoint enabled in the v27/v28 configs?"
grep -n "entropy_floor\|entropy_coef\|entropy_guard\|kl_anchor_checkpoint" configs/mario_1_1_v27_seed0.yaml configs/mario_1_1_v28_seed0.yaml

section "5. Same knobs in the banked 1-1 control this was copied 'verbatim' from"
grep -n "entropy_floor\|entropy_coef\|entropy_guard\|kl_anchor_checkpoint" configs/mario_1_1_backward.yaml configs/mario_1_1_backward_v4.yaml configs/mario_1_1_backward_v6.yaml configs/mario_1_1_backward_consol.yaml

section "6. Which configs DO ship entropy_guard"
grep -rl "entropy_guard" configs/
echo "--- introducing commit ---"
git log --oneline --all -S"entropy_guard" -- src/training/trainer.py

section "7. trainer.py: the three distinct anti-collapse mechanisms and their scoping comments"
sed -n '318,347p' src/training/trainer.py
echo "---"
sed -n '9195,9240p' src/training/trainer.py

section "8. The 2026-07-20 entropy-floor-tried-and-degraded episode"
git log --oneline --since=2026-07-19 --until=2026-07-21
echo "--- memory receipt (point-in-time, verify against code) ---"
cat "/Users/stits/.claude/projects/-Users-stits-Documents-macos-emulation-and-training/memory/project_capability_attempt1_failed_2026-07-20.md" 2>/dev/null || echo "(memory file not found at run time)"

section "9. B4/B5/B6: the prior 1-1-backward-curriculum entropy-collapse campaign (2026-08-08..11)"
sed -n '1,70p' docs/research/B5_PREREG_2026-08-08.md
echo "..."
sed -n '400,445p' docs/research/B5_PREREG_2026-08-08.md

section "10. Was B4/B5/B6 (entropy_guard, CONFIDENT-WRONG NOOP SHARPENING) referenced when v27/v28 were registered?"
grep -in "confident-wrong\|noop sharpening\|entropy_guard\|consolidate_level" \
  docs/proposals/V27_FRESH_RECOVERY_2026-08-24.md \
  docs/proposals/V28_CAPACITY_2026-08-25.md \
  || echo "(no hits in either registration doc)"

section "11. The KL-anchor collapse receipt (1-2, options experiment) and whether entropy_guard rode along"
grep -n "31/100\|31 -> 8\|8/100" docs/research/OPTIONS_NEGATIVE_2026-08-23.md
echo "--- did the options control's backward_curriculum block carry entropy_guard? ---"
sed -n '75,116p' configs/mario_1_2_options.yaml

section "12. CLAIMS.md ledger entries carrying the -74% figure"
grep -n "74\|peak instability\|KL anchor active" CLAIMS.md

section "13. nes_core/KNOWN_ISSUES.md + CHANGELOG.md — direct hits"
grep -in "entropy\|collapse" nes_core/KNOWN_ISSUES.md CHANGELOG.md 2>/dev/null

section "14. analyze.py — what kind of tool it is (for the 'is the tool adequate' question)"
sed -n '1,20p' scripts/analyze.py
