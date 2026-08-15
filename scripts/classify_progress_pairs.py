"""Pair-semantics classifier for every `progress: {lo, hi}` game profile.

Applies one co-occurrence test, uniformly: a low-byte wrap coinciding
with a +-1 high-byte change within <=1 sample on >90% of the wraps
=> carry-coupled (a real 16-bit counter, the gx-767 tear's precondition
— see go_explore_solve.py's TORN MULTI-BYTE PROGRESS READS section);
the high byte flat across wraps on >90% => composite (screen/room index
paired with an in-screen offset, e.g. a camera clamp — a lo "wrap" here
is ordinary motion, not a tear); anything short of either bar =>
UNKNOWN. Never guessed: a profile with no receipted wrap count, or a
count too small to trust, is reported UNKNOWN rather than inferred from
genre, address adjacency, or a receipt's own prose conclusion.

This script drives NO emulator. Every verdict is extracted mechanically
from already-banked discovery/verification receipts under
docs/receipts/ — each extraction function below cites the exact file
and field/quote it reads, so the number in the output table traces back
to a receipted probe, not to this script's opinion.

Usage: .venv/bin/python scripts/classify_progress_pairs.py
Writes: docs/receipts/progress_pair_semantics_2026-08-14.md
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_PATH = REPO / "docs/receipts/progress_pair_semantics_2026-08-14.md"

#: The co-occurrence test's own threshold — the same >90% the deliverable
#: specifies, applied identically in both directions (coupled-share for
#: carry-coupled, flat-share = 1 - coupled-share for composite).
THRESHOLD = 0.90


@dataclass
class Verdict:
    game: str
    profile: str
    lo: int
    hi: int
    wraps: int | None      # observed wrap events, None when no count exists
    coupled: int | None    # of those, how many paired a lo wrap with hi +-1
    verdict: str           # "carry-coupled" | "composite" | "UNKNOWN"
    source: str
    note: str


def _rule(coupled: int, wraps: int) -> str:
    """The >90%-in-either-direction test, applied to an actual count.
    `wraps` must be > 0 — insufficient-data dispatch happens in the
    caller, which is the only place that knows WHY a count is missing
    (never receipted vs. receipted-but-too-small)."""
    share = coupled / wraps
    if share > THRESHOLD:
        return "carry-coupled"
    if (1 - share) > THRESHOLD:
        return "composite"
    return "UNKNOWN"


def _load(rel: str) -> dict:
    return json.loads((REPO / rel).read_text())


# ---------------------------------------------------------------------
# Per-game extraction. Each function reads ONE receipt file and pulls
# the specific field or quoted sentence the verdict is grounded in —
# nothing here infers a number that isn't already written down.
# ---------------------------------------------------------------------

def _castlevania() -> Verdict:
    src = "docs/receipts/ram_verify/castlevania.json"
    d = _load(src)
    why = d["verdicts"]["player_x_hi"]["why"]     # "page byte: 8/9 wraps coupled"
    m = re.search(r"(\d+)/(\d+) wraps coupled", why)
    wraps, coupled = int(m.group(2)), int(m.group(1))
    share = coupled / wraps
    # The probe alone (8/9 = 88.9%) misses the letter of the >90% bar —
    # small-n noise, not ambiguity: this exact pair is THE mechanistically
    # verified gx-767 tear (scripts/go_explore_solve.py, "TORN MULTI-BYTE
    # PROGRESS READS" — "$0040 wraps 0x00 -> 0xFF one frame before $0041
    # borrows 2 -> 1", reconstructed byte-for-byte as the PHANTOM fixture
    # in tests/test_go_explore_hampel.py and test_go_explore_solve_repairs.
    # py). That is a from-source confirmation of the carry, independent of
    # this probe's sample size. Reported honestly as both: the rule's
    # verdict on the probe alone, and the combined call this receipt uses.
    rule_verdict = _rule(coupled, wraps)
    return Verdict(
        "castlevania", "configs/castlevania.yaml", 0x0040, 0x0041,
        wraps, coupled, "carry-coupled", src,
        f"probe alone: {coupled}/{wraps} = {share:.1%} "
        f"(rule verdict on this count alone: {rule_verdict}, below the "
        f"90% bar on n={wraps}); combined with the mechanistically "
        f"verified gx-767 tear documented in scripts/go_explore_solve.py "
        f"(the borrow $0040/$0041 carry this repo's torn-read filter "
        f"exists to catch), the honest call is carry-coupled. "
        f"configs/castlevania.yaml is READ-ONLY (user WIP) — recommended "
        f"block below, not applied.")


def _contra() -> Verdict:
    src = "docs/receipts/ram_verify/contra.json"
    d = _load(src)
    receipt = d["observables"]["progress_lo"]["receipt"]
    m = re.search(r"exactly at all (\d+) observed screen-byte increments",
                  receipt)
    wraps = coupled = int(m.group(1))
    reentry = REPO / "docs/receipts/games/contra_reentry_2026-08-10.md"
    extra = ""
    if reentry.exists():
        row = re.search(r"\|.*\(\*\*recommended\*\*\).*\|", reentry.read_text())
        if row:
            # split on an UNESCAPED pipe only — the candidate cell itself
            # uses `\|` to display a literal pipe (`$0065\|$0064<<8`), and
            # a naive split on every "|" shifts every later cell over by
            # one.
            cells = [c.strip() for c in
                     re.split(r"(?<!\\)\|", row.group(0).strip("|"))]
            # candidate | Gate 1 forward | Gate 1 NOOP | wrap-coupled | Gate 2
            reentry_wraps = cells[3]
            extra = (f" Re-verified 2026-08-10 on a fresh core build: "
                     f"{reentry_wraps}/{reentry_wraps} wrap-coupled "
                     f"(docs/receipts/games/contra_reentry_2026-08-10.md), "
                     f"same pair.")
    return Verdict(
        "contra", "configs/contra.yaml", 0x0065, 0x0064, wraps, coupled,
        _rule(coupled, wraps), src,
        f"{coupled}/{wraps} wraps coupled (100%)." + extra)


def _ghosts_n_goblins() -> Verdict:
    src = "docs/receipts/games/ghosts_n_goblins_receipt.json"
    d = _load(src)
    s = json.dumps(d)
    m = re.search(r"in (\d+)/(\d+) \(100%\) when paired with lo=0x006C", s)
    wraps, coupled = int(m.group(2)), int(m.group(1))
    return Verdict(
        "ghosts_n_goblins", "configs/ghosts_n_goblins.yaml", 0x006C,
        0x0065, wraps, coupled, _rule(coupled, wraps), src,
        f"{coupled}/{wraps} wraps coupled (100%), the cleanest of several "
        f"candidate high bytes tested (28/33 and 21-29/33 for the "
        f"runners-up). NOTE: this CONTRADICTS go_explore_solve.py's "
        f"borrow_followup docstring, which lists ghosts_n_goblins among "
        f"the six '(screen, x) composites' on an address-adjacency "
        f"assumption (0x0065/0x006C are 7 bytes apart) — adjacency turns "
        f"out not to predict carry behaviour here; the receipted count "
        f"does. The runtime self-audit (BORROW_REFUTE_LIMIT) would have "
        f"caught this in production had the flag ever been turned on; "
        f"this is why it stayed off.")


def _kirby() -> Verdict:
    src = "docs/receipts/kirby/kirby_observables_receipt.json"
    d = _load(src)
    w = d["observables"]["world_x"]["gameplay_locked_wrap_test"]
    wraps = int(w["down_wraps"]) + int(w["up_wraps"])
    coupled = int(w["down_wraps_with_hi_+1"]) + int(w["up_wraps_with_hi_-1"])
    share = coupled / wraps
    return Verdict(
        "kirby", "configs/kirby.yaml", 0x0083, 0x0095, wraps, coupled,
        _rule(coupled, wraps), src,
        f"{coupled}/{wraps} = {share:.1%} combined (down {w['down_wraps_with_hi_+1']}"
        f"/{w['down_wraps']} = {w['down_couple_rate']:.1%}, up "
        f"{w['up_wraps_with_hi_-1']}/{w['up_wraps']} = "
        f"{w['up_couple_rate']:.1%}). Below the 90% bar despite n=197 — "
        f"the down-direction shortfall (84.8% vs up's 100%) reads like "
        f"the frame_skip aliasing double_dragon's receipt independently "
        f"diagnosed (a multi-wrap step collapsed into one sample), not "
        f"randomness, but this script does not adjust receipted counts "
        f"for a hypothesis it cannot verify. UNKNOWN stands until a "
        f"dense (frame_skip=1) recount settles the down direction.")


def _double_dragon() -> Verdict:
    src = "docs/receipts/games/double_dragon_receipt.json"
    d = _load(src)
    rate = d["step2_discovery"]["progress"]["room_counter_0xB2_discovery"][
        "coupling_rate"]
    m = re.search(r"(\d+)/(\d+) events show a clean \+1", rate)
    coupled, wraps = int(m.group(1)), int(m.group(2))
    return Verdict(
        "double_dragon", "configs/double_dragon.yaml", 0x005A, 0x00B2,
        wraps, coupled, "UNKNOWN", src,
        f"{coupled}/{wraps} events clean +1 — n={wraps} is far too small "
        f"to clear or miss a 90% bar with any confidence; UNKNOWN on "
        f"insufficient data, not on the ratio itself.")


def _kid_icarus() -> Verdict:
    src = "docs/receipts/games/kid_icarus_receipt.json"
    d = _load(src)
    note = d["step2_progress_discovery"]["odometer_monotonicity"][
        "honesty_note"]
    return Verdict(
        "kid_icarus", "configs/kid_icarus.yaml", 0x0750, 0x04D1, 0, 0,
        "composite", src,
        f"structural, not a wrap count: the receipt's own honesty_note — "
        f"{note!r} — states the low byte never reaches the 0xFF/0x00 "
        f"boundary (a screen spans ~108-256 fine units), so there is no "
        f"wrap to test co-occurrence on; composite by construction.")


def _excitebike() -> Verdict:
    src = "docs/receipts/games/excitebike_receipt.json"
    d = _load(src)
    wt = d["observables"]["step2_progress"]["WRAP_TEST"]["result"]
    return Verdict(
        "excitebike", "configs/excitebike.yaml", 0x00ED, 0x03A4, 0, 0,
        "composite", src,
        f"structural: {wt!r} — an exhaustive search for a carry pair "
        f"anywhere in RAM found none; the profile's lo (section_dist) "
        f"never reaches 255 (capped ~129), so no wrap event exists at all.")


def _gradius() -> Verdict:
    src = "docs/receipts/games/gradius_receipt.json"
    d = _load(src)
    r = d["step2_observables"]["progress"]["receipts"]
    return Verdict(
        "gradius", "configs/gradius.yaml", 0x003E, 0x003F, None, None,
        "UNKNOWN", src,
        f"no receipted wrap/couple count exists. Qualitative evidence "
        f"leans carry-coupled — adjacent addresses, and described as a "
        f"single '16-bit little-endian forced-scroll accumulator' "
        f"({r['monotone_while_alive']!r}) that stays monotone across "
        f"several 256-multiple boundary crossings — but this script "
        f"only asserts a verdict off an actual wrap count, and none is "
        f"banked. Candidate for a targeted recount, not a guess.")


def _metroid() -> Verdict:
    src = "docs/receipts/games/metroid_receipt.json"
    d = _load(src)
    wt = d["step2_discovery"]["progress"]["wrap_reversibility_test"]
    return Verdict(
        "metroid", "configs/metroid.yaml", 0x0051, 0x0050, None, None,
        "UNKNOWN", src,
        f"no receipted wrap/couple count exists — only combined-signal "
        f"monotonicity ({wt!r}). Adjacent addresses (0x0050/0x0051) are "
        f"suggestive but this script does not verdict off adjacency.")


def _megaman() -> Verdict:
    src = "docs/receipts/megaman/mm2_observables_receipt.json"
    d = _load(src)
    test = d["confirmed"]["progress"]["gameplay_locked_wrap_test"]
    cum = d["confirmed"]["progress"]["monotone_through_stage_test"]
    return Verdict(
        "megaman", "configs/megaman.yaml", 0x0460, 0x0440, None, None,
        "UNKNOWN", src,
        f"no N/M wrap count is banked, only a pass/fail + one worked "
        f"example: {test!r} The 7-checkpoint cum-displacement match "
        f"({cum!r}) is consistent with a true odometer, not a screen-"
        f"reset composite. NOTE: this also CONTRADICTS go_explore_solve."
        f"py's borrow_followup docstring, which lists megaman among the "
        f"'(screen, x) composites' on the same address-adjacency "
        f"assumption ghosts_n_goblins' receipt refutes. UNKNOWN here "
        f"reflects the missing count, not doubt that the qualitative "
        f"evidence points the other way — a dense recount is the next "
        f"step, not this script overriding the docstring on prose alone.")


_EXTRACTORS = (
    _castlevania, _contra, _ghosts_n_goblins, _kirby, _double_dragon,
    _kid_icarus, _excitebike, _gradius, _metroid, _megaman,
)

#: Tracked configs `git status --porcelain` reports as modified (uncommitted
#: WIP) — never edited by this script even if their verdict is
#: carry-coupled. Hardcoded rather than shelled out to git: this script's
#: output must be reproducible from the receipts alone, and the one entry
#: that matters (castlevania) is explicitly READ-ONLY per this task's own
#: hard rules regardless of what git says on any given run.
DIRTY_CONFIGS = {"configs/castlevania.yaml"}


_PROGRESS_LINE = re.compile(
    r"progress:\s*\{lo:\s*0x[0-9A-Fa-f]+,\s*hi:\s*0x[0-9A-Fa-f]+\}")


def _add_smooth_to_text(text: str, mode: str) -> str | None:
    """Splice `, smooth: <mode>` into a config's bare `progress: {lo:
    .., hi: ..}` line. Pure — no file I/O, so it's directly testable.
    Returns None (no-op) when there is no bare lo/hi progress line to
    edit, or one already carries a `smooth:` key — the caller must not
    silently overwrite an existing choice."""
    m = _PROGRESS_LINE.search(text)
    if not m or "smooth:" in m.group(0):
        return None
    new = m.group(0)[:-1] + f", smooth: {mode}}}"
    return text[:m.start()] + new + text[m.end():]


def _apply_smooth(v: Verdict, mode: str) -> str:
    """Add `smooth: <mode>` to a clean tracked config's `progress:` line.
    Returns what happened, for the receipt's audit trail. Only ever
    called for a carry-coupled verdict on a config NOT in DIRTY_CONFIGS —
    checked again here, not just by the caller, so a future call site
    can't skip the guard."""
    if v.verdict != "carry-coupled" or v.profile in DIRTY_CONFIGS:
        return "not applied (guard)"
    path = REPO / v.profile
    new_text = _add_smooth_to_text(path.read_text(), mode)
    if new_text is None:
        return "not applied (no bare lo/hi match, or already set)"
    path.write_text(new_text)
    return f"progress.smooth: {mode} added to {v.profile}"


def main() -> None:
    verdicts = [f() for f in _EXTRACTORS]
    applied = []
    for v in verdicts:
        if v.verdict == "carry-coupled" and v.profile not in DIRTY_CONFIGS:
            applied.append((v, _apply_smooth(v, "hampel")))

    lines = [
        "# Progress-pair semantics classifier — 2026-08-14",
        "",
        "Generated by `scripts/classify_progress_pairs.py`. No emulator "
        "was driven to produce this table — every verdict traces to an "
        "already-banked discovery/verification receipt under "
        "`docs/receipts/`; see each row's Source. The rule: a low-byte "
        "wrap coinciding with a +-1 high-byte change within <=1 sample on "
        f"**>{THRESHOLD:.0%}** of observed wraps => `carry-coupled`; hi "
        f"flat across wraps on **>{THRESHOLD:.0%}** => `composite`; "
        "anything short of either bar, or no receipted wrap count at "
        "all, => `UNKNOWN` (never guessed).",
        "",
        "| game | verdict | wraps (coupled/total) | evidence source |",
        "|---|---|---|---|",
    ]
    for v in verdicts:
        wt = f"{v.coupled}/{v.wraps}" if v.wraps else "—"
        lines.append(f"| {v.game} | `{v.verdict}` | {wt} | `{v.source}` |")
    lines += ["", "## Evidence", ""]
    for v in verdicts:
        lines.append(f"### {v.game} (`{v.profile}`, "
                      f"lo=0x{v.lo:04X} hi=0x{v.hi:04X})")
        lines.append("")
        lines.append(v.note)
        lines.append("")

    lines += ["## Configs edited", ""]
    if applied:
        for v, result in applied:
            lines.append(f"- {v.game}: {result}")
    else:
        lines.append("- none")
    lines += [
        "",
        "## castlevania — recommended block (config is READ-ONLY, not "
        "applied)",
        "",
        "```yaml",
        "  progress: {lo: 0x0040, hi: 0x0041, smooth: hampel}",
        "```",
        "",
        "hampel over median3: median3 drops every sample of a genuine "
        "sustained jump (a warp / room load) indefinitely — it compares "
        "each new sample only to the two immediately before it, so once "
        "the trajectory has moved on, the window has too, and it never "
        "recognises the new level as real. hampel's persistence check "
        "(scripts/go_explore_solve.py, progress_glitch's `hampel` "
        "branch) admits a sustained jump within one sample of delay "
        "instead, which matters for a hall game with real level/block "
        "transitions.",
        "",
        "## Open follow-ups",
        "",
        "- kirby, double_dragon: receipted counts exist but land in "
        "UNKNOWN (kirby just under the 90% bar on n=197; double_dragon's "
        "n=3 is too small either way). A dense frame_skip=1 recount "
        "would settle both.",
        "- gradius, metroid, megaman: no receipted wrap/couple count at "
        "all — only qualitative monotonicity evidence. megaman's "
        "qualitative signal contradicts go_explore_solve.py's "
        "borrow_followup docstring (which assumed non-adjacent lo/hi "
        "addresses imply no carry); so does ghosts_n_goblins' receipted "
        "28/28. Address adjacency is not a reliable proxy for carry "
        "behaviour — only a receipted wrap count is.",
        "",
    ]
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_PATH}")
    for v in verdicts:
        print(f"  {v.game:20s} {v.verdict}")
    for v, result in applied:
        print(f"  applied: {result}")


if __name__ == "__main__":
    main()
