"""Layer-1 gameplay test for Contra at idle. Contra has a 73-byte
total RAM divergence from nes-py at 600f idle (per
test_lockstep_baseline.py), but 56 of those bytes are in the stack
($0100-$01FF) — interrupt-push residue accumulated across NMI
handlers. The 14 zero-page bytes that diverge at 600f are slowly-
animating game-engine timers, not core gameplay state.

At 360 idle frames the gameplay-critical addresses (player position,
lives, weapon, level) all match nes-py byte-exact. This test passes
today and locks in that Contra's player + level state is correct in
nes_core; the extra bytes seen in the lockstep baseline don't
represent a gameplay logic bug.

LABELS ARE DECORATIVE, NOT A SEMANTICS CLAIM: this is a fidelity/parity
harness comparing raw byte agreement between nes_core and nes-py — the
assertion is "these two emulators compute the same byte," independent
of what the byte means. The dict keys (addresses) are what's actually
checked; the string values exist only so a failure message is readable.

That said, a decorative label that flatly CONTRADICTS a measured
constant elsewhere in the tree is a landmine for the next reader, not
a harmless nicety — it was found here 2026-08-27: this dict labeled
0x0032 "p1_x_position" and 0x0040 "p1_lives", while configs/contra.yaml
(`lives: 0x0032`, re-verified 2026-08-10 by an independent 3-probe
protocol: starts at 2, decrements on death, top-ranked of 4 candidates)
and nes_core/src/rewards.rs's `ContraReward::RAM_LIVES = 0x0032` both
independently agree the LIVES byte is 0x0032, not 0x0040. Contra has no
witnessed clear and its boss is separately falsified, so none of these
labels carry game-truth weight either way — but the measured one wins
over the unsourced one. 0x0018 ("level") has no supporting measurement
anywhere in the tree either (Contra's only progress-related address on
record is the quarantined, unverified `q_current_level` at 0x0030, a
different address) — relabeled to say so plainly rather than assert a
level semantic nobody measured.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ROM = REPO / "roms" / "Contra (USA).nes"
CONTRA_YAML = REPO / "configs" / "contra.yaml"

CONTRA_GAMEPLAY_BYTES = {
    0x0018: "unverified_0x0018",  # no measurement anywhere backs "level" here
    0x0032: "p1_lives",  # measured: configs/contra.yaml + rewards.rs RAM_LIVES
    0x0040: "unverified_0x0040",  # no measurement anywhere backs "p1_lives" here
    0x0080: "unverified_0x0080",
    0x00AA: "p1_weapon",  # matches rewards.rs RAM_WEAPON = 0x00AA
}


@pytest.fixture(autouse=True)
def _silence_gym():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_lives_label_does_not_contradict_the_measured_lives_address():
    """Regression for the 2026-08-27 contradiction: this dict must never
    label an address "lives" that isn't the one configs/contra.yaml
    measured and rewards.rs's RAM_LIVES constant agrees on. Reads the
    yaml's declared address directly so this can't drift out of sync
    with the config that carries the actual receipt."""
    text = CONTRA_YAML.read_text()
    m = re.search(r"^\s*lives:\s*(0x[0-9A-Fa-f]+)", text, re.MULTILINE)
    assert m, "configs/contra.yaml no longer declares a `lives:` address"
    measured_lives_addr = int(m.group(1), 16)

    for addr, label in CONTRA_GAMEPLAY_BYTES.items():
        if "lives" in label:
            assert addr == measured_lives_addr, (
                f"label {label!r} at ${addr:04X} contradicts the measured "
                f"lives address ${measured_lives_addr:04X} declared in "
                f"configs/contra.yaml"
            )


@pytest.mark.parity
def test_contra_gameplay_state_matches_nespy_after_360f_idle():
    if not ROM.exists():
        pytest.skip(f"ROM missing: {ROM}")
    from tests.parity.lockstep import _load_ours, _load_theirs
    ours = _load_ours(str(ROM))
    theirs = _load_theirs(str(ROM))
    for _ in range(360):
        ours.step(0)
        theirs.step(0)
    diffs = []
    for addr, name in CONTRA_GAMEPLAY_BYTES.items():
        o = int(ours.get_ram(addr))
        t = int(theirs.ram[addr])
        if o != t:
            diffs.append(f"${addr:04X} ({name}): ours=0x{o:02X} theirs=0x{t:02X}")
    if diffs:
        pytest.fail("Contra game-state diverges from nes-py:\n  " + "\n  ".join(diffs))
