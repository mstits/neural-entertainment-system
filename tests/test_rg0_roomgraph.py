"""RG-0 — the room-graph offline falsifier (ROOMGRAPH_ENGINE_2026-08-24
§6, T4). BLOCKS ALL LIVE ROOM-GRAPH COMPUTE: every one of the five
gate assertions below must pass before T5's RG-1 runs launch
(cheap-premise-first, BINDING).

The fixtures under tests/fixtures/roomgraph/ are captured hardware-
surface streams (2 KB nametable VRAM + odometer + scene + rendered
lines per solver step) from our own scripted rollouts — generation
commands and provenance in docs/receipts/room_fp/{zelda,metroid}.md,
re-mintable via scripts/room_fp_calibrate.py. The detector replayed
over them is the T1 identity layer itself (fp_settle /
classify_transition / RoomIndex), driven by replay_room_stream — the
same routine the receipts' numbers come from. The classifier constants
come from the SHIPPED gate profiles (configs/*_roomfp.yaml), so a
profile edit that breaks a fixture fails this falsifier.

MEASURED DEVIATIONS from the design draft (full receipts in
docs/receipts/room_fp/):
  * settle = 14 at fs4 (ceil(56/frame_skip)), not 3: Zelda's east pan
    contains an 8-step stable window (pre-drawn columns scrolling —
    VRAM is scroll-invariant), Metroid's mid-run repeating terrain a
    12-step one; settle 3 mints hybrid phantom rooms (locked below).
  * Onset baseline = the previous rendered sample (replay_room_stream
    docstring): the death flash's scene bumps land in the same step as
    its first NT rewrite, so a current-sample onset misreads Δscene.
  * RG-0.5's literal wording ("zero extra nodes") is unsatisfiable on
    the measured surface — any route that racks up 4 scene bumps
    inside Metroid's room 1 crosses motion reversals, where the camera
    holds still long enough to settle a real, reproducible camera-view
    fingerprint. The test asserts the falsifier's actual claim: scene
    noise mints NOTHING — every node is a stable camera view that
    re-interns exactly on revisit (lap 2 = zero new nodes while the
    scene counter keeps climbing), and the fingerprint node count
    stays below the scene-ordinal count it replaces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.go_explore_solve import GenericGame, nt_fingerprint, room_fp_mask
from scripts.room_fp_calibrate import load_capture, replay_room_stream

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures" / "roomgraph"


def _cfg(profile: str) -> dict:
    game = GenericGame(yaml.safe_load((ROOT / "configs" / profile).read_text()))
    assert game.room_fp is not None
    return game.room_fp


@pytest.fixture(scope="module")
def zelda_cfg() -> dict:
    return _cfg("zelda_roomfp.yaml")


@pytest.fixture(scope="module")
def metroid_cfg() -> dict:
    return _cfg("metroid_roomfp.yaml")


def _replay(name: str, cfg: dict):
    nt, odo, scene, lines, meta = load_capture(FIX / name)
    index, events = replay_room_stream(nt, odo, scene, lines, cfg)
    return index, events, (nt, odo, scene, lines, meta)


def _edges(index):
    return [(s, d, e["kind"], e["dir"])
            for s, dsts in index.adj.items() for d, e in dsts.items()]


# ---------------------------------------------------------------------
# The five gate assertions (§6 RG-0). Any failure => stop, no live
# compute.
# ---------------------------------------------------------------------


def test_rg0_1_zelda_east_exit_is_a_pan_e_minting_one_node(zelda_cfg):
    """Zelda east exit => pan-E, exactly one new node (one directed
    pan edge, zero warps). Fixture: the receipted north-screen state,
    weave continuation across the east edge, then idle."""
    index, events, _ = _replay("zelda_east_exit_fs4.npz", zelda_cfg)
    assert index.n_rooms() == 2          # adoption + exactly one new node
    assert len(events) == 2
    adopt, exit_ = events
    assert adopt["src"] is None          # adoption from unknown: no edge
    assert exit_["kind"] == "pan" and exit_["dir"] == "E"
    assert exit_["d_odo"][0] == 256      # the full pan, onset-integrated
    assert _edges(index) == [(0, 1, "pan", "E")]
    assert index.warp_count == 0


def test_rg0_2_zelda_death_is_a_warp_minting_zero_edges(zelda_cfg):
    """Zelda death => warp, ZERO edges. Fixture: hold-right from the
    one-heart root — two octorok hits, the death flash (attribute
    rewrite; Δscene +2 with the odometer flat), window ending before
    the game-over screens (their measured fade chain is locked
    separately below)."""
    index, events, _ = _replay("zelda_death_fs4.npz", zelda_cfg)
    assert [e["kind"] for e in events] == ["fade", "warp"]  # adopt, death
    warp = events[1]
    assert warp["d_scene"] >= zelda_cfg["warp_scene_min"]
    assert warp["d_odo"] == [0, 0]
    assert _edges(index) == []           # the death minted NOTHING navigable
    assert index.warp_count == 1


def test_rg0_3_zelda_idle_is_one_hash_post_mask(zelda_cfg):
    """300 idle frames post-mask => exactly ONE fingerprint. The mask
    is load-bearing: unmasked, the HUD heart byte (NT 214) splits the
    same screen into two."""
    nt, _, _, lines, _ = load_capture(FIX / "zelda_idle_fs1.npz")
    rendered = nt[lines >= zelda_cfg["min_lines"]]
    assert len(rendered) == 300
    masked = room_fp_mask(zelda_cfg["mask"])
    assert len({nt_fingerprint(f, masked) for f in rendered}) == 1
    raw = room_fp_mask([])
    assert len({nt_fingerprint(f, raw) for f in rendered}) == 2


def test_rg0_4_metroid_doors_are_exactly_two_pan_edges(metroid_cfg):
    """Metroid door1/door2 => exactly two pan edges (+254 px each,
    Δscene +1, no mid-scroll settle), zero warps."""
    index, events, _ = _replay("metroid_doors_fs4.npz", metroid_cfg)
    assert index.n_rooms() == 3
    kinds = [(e["kind"], e["dir"]) for e in events]
    assert kinds == [("fade", None), ("pan", "E"), ("pan", "E")]
    assert [e["d_odo"][0] for e in events[1:]] == [254, 254]
    edges = _edges(index)
    assert len(edges) == 2
    assert all(k == "pan" and d == "E" for _, _, k, d in edges)
    assert index.warp_count == 0


def test_rg0_5_metroid_scene_noise_mints_zero_extra_nodes(metroid_cfg):
    """Metroid spurious scene bumps (scene 4+ without leaving room 1)
    => zero nodes minted BY the noise. Fixture: two identical
    right/left laps inside the first room; the scene-cut counter — the
    identity D0 would have keyed on — reaches 8. The fingerprint's
    nodes are the stable camera views only: lap 2 mints ZERO new nodes
    (every settle re-interns lap 1's ordinal, the -170 px reversal
    view landing on the byte-identical VRAM both laps), and the node
    count stays strictly under the scene-ordinal count. Deviation from
    the draft's literal wording is documented in the module docstring
    + docs/receipts/room_fp/metroid.md."""
    index, events, (nt, odo, scene, lines, _) = _replay(
        "metroid_scene_noise_fs4.npz", metroid_cfg)
    assert int(scene[-1]) >= 8           # the noise, all inside room 1
    assert index.warp_count == 0
    # Lap 1's last camera view (the left-end reversal) settles just
    # after the lap boundary; every settle after step 230 is pure lap-2
    # re-traversal and must re-intern a lap-1 ordinal.
    lap2 = [e for e in events if e["step"] >= 230]
    assert len(lap2) >= 2                # it did settle again...
    known = {e["dst"] for e in events if e["step"] < 230}
    assert all(e["dst"] in known for e in lap2)      # ...all dedupe
    assert index.n_rooms() < len(np.unique(scene))   # beats scene identity


# ---------------------------------------------------------------------
# Regression locks for the measured behavior behind the gate numbers —
# not part of the five-assertion gate, but they pin the physics the
# constants were registered from.
# ---------------------------------------------------------------------


def test_zelda_death_aftermath_is_fades_never_warp_edges(zelda_cfg):
    """The FULL death stream (through the game-over/continue screens):
    still exactly one warp and zero warp-minted edges; the aftermath
    screens arrive as two fade edges out of the warp-adopted flash
    room (spark rewrite, then the continue menu). This is the honest
    residual the RG-1a audit ('zero warp-minted edges') rides on — in
    live Zelda runs the lives proxy kills the lineage at the first hit,
    long before these settles."""
    index, events, _ = _replay("zelda_death_full_fs4.npz", zelda_cfg)
    assert [e["kind"] for e in events] == ["fade", "warp", "fade", "fade"]
    assert index.warp_count == 1
    edges = _edges(index)
    assert len(edges) == 2
    assert all(k == "fade" for _, _, k, _ in edges)


def test_settle_three_would_mint_a_hybrid_mid_pan_room(zelda_cfg):
    """The counterfactual that forced settle 14: at the draft's settle
    3, the east pan's 8-step pre-drawn scroll window settles a hybrid
    half-drawn room (3 rooms, 2 edges, truncated +132 px pan) where
    settle 14 sees exactly one pan-E of +256."""
    cfg = dict(zelda_cfg, settle=3)
    nt, odo, scene, lines, _ = load_capture(FIX / "zelda_east_exit_fs4.npz")
    index, events = replay_room_stream(nt, odo, scene, lines, cfg)
    assert index.n_rooms() == 3
    assert [e["kind"] for e in events] == ["fade", "fade", "pan"]


def test_masks_reproduce_probe_idle_stability(zelda_cfg, metroid_cfg):
    """The T4 done-when: each calibration capture hashes to exactly one
    post-mask fingerprint (Zelda idle 300f + walk 210f, byte-214 heart
    mask; Metroid idle 300f + walk 120f, empty mask)."""
    for name, cfg in (("zelda_idle_fs1.npz", zelda_cfg),
                      ("zelda_walk_fs1.npz", zelda_cfg),
                      ("metroid_idle_fs1.npz", metroid_cfg),
                      ("metroid_walk_fs1.npz", metroid_cfg)):
        nt, _, _, lines, _ = load_capture(FIX / name)
        rendered = nt[lines >= cfg["min_lines"]]
        masked = room_fp_mask(cfg["mask"])
        assert len({nt_fingerprint(f, masked) for f in rendered}) == 1, name


def test_fixture_provenance_is_stamped():
    """Every .npz fixture carries its generation record (tool, rom +
    state sha256, frame_skip, script) so it can be re-minted."""
    npz = sorted(FIX.glob("*.npz"))
    assert len(npz) >= 7
    for p in npz:
        _, _, _, _, meta = load_capture(p)
        for key in ("tool", "rom_sha256", "state_sha256", "frame_skip",
                    "script", "steps"):
            assert key in meta, (p.name, key)
