"""Tests for the SMB one-shot sub-stage ladder
(`src/training/smb_substage_ladder.py`).

Covers the three behaviours the campaign spec names for this lane:
nearest-x seed binding, `region_of` monotonicity within a level, and
"a warp never raises the region".
"""

from __future__ import annotations

import json

from src.training.smb_substage_ladder import (
    AREA_TO_LEVEL_LABEL,
    LADDER_SIZE,
    OFF_LADDER,
    RAM_AREA,
    RAM_WORLD,
    RAM_X_HI,
    RAM_X_LO,
    SubStage,
    build_ladder,
    region_of,
    smb_sequential_cell,
)


def _ram(*, world: int = 0, area: int = 0, x: int = 0, display: int = 0) -> bytes:
    """A 2 KB RAM snapshot with the SMB progress bytes set.

    world/area are raw RAM bytes ($075F is 0 for World 1); x is the absolute
    forward position (page<<8 | fine); display is $075C.
    """
    buf = bytearray(2048)
    buf[RAM_WORLD] = world & 0xFF
    buf[RAM_AREA] = area & 0xFF
    buf[RAM_X_HI] = (x >> 8) & 0xFF
    buf[RAM_X_LO] = x & 0xFF
    buf[0x075C] = display & 0xFF
    return bytes(buf)


# ---- smb_sequential_cell admission ----------------------------------


def test_cell_rejects_world_two_and_off_allowlist_areas() -> None:
    # Warp landing in world 2 -> off-chain regardless of area byte.
    assert smb_sequential_cell(_ram(world=1, area=0, x=100)) is None
    # An area byte outside the sequential allowlist (e.g. a bonus/warp room).
    assert smb_sequential_cell(_ram(world=0, area=9, x=100)) is None


def test_cell_admits_world_one_chain_and_buckets_x() -> None:
    cell = smb_sequential_cell(_ram(world=0, area=3, x=800), x_bucket=8)
    assert cell is not None
    _world, area, x_bucket = cell
    assert area == 3
    assert x_bucket == 100  # 800 // 8

    # Different area bytes at the same x page are distinct cells (the cell
    # key carries the area, so it cannot collide across levels).
    a = smb_sequential_cell(_ram(world=0, area=2, x=800))
    b = smb_sequential_cell(_ram(world=0, area=3, x=800))
    assert a is not None and b is not None
    assert a != b


# ---- ladder structure -----------------------------------------------


def test_structural_ladder_is_18_ordered_rungs() -> None:
    ladder = build_ladder()
    assert LADDER_SIZE == 18
    assert len(ladder) == 18
    # Orders are 0..17 contiguous.
    assert [s.order for s in ladder] == list(range(18))
    # No disk seeds bound when no globs are passed.
    assert all(s.seed_blob is None for s in ladder)
    # Level partition matches the milestone table.
    by_level: dict[str, list[int]] = {}
    for s in ladder:
        by_level.setdefault(s.level, []).append(s.order)
    assert by_level["1-1"] == [0, 1, 2]
    assert by_level["1-2"] == [3, 4, 5, 6]
    assert by_level["1-3"] == [7, 8, 9]
    assert by_level["1-4"] == [10, 11, 12, 13, 14, 15, 16, 17]
    # All rungs are World 1 and carry the shared reward profile.
    assert all(s.world == 0 for s in ladder)
    assert len({s.reward_profile for s in ladder}) == 1


def test_substage_helpers() -> None:
    s = SubStage(order=5, world=0, area=2, x_lo=900, x_hi=1800)
    assert s.level == "1-2"
    assert s.x_center == 1350.0
    assert s.contains_x(900) and s.contains_x(1799)
    assert not s.contains_x(1800)
    assert s.anchor == (0 << 4) | 2


# ---- nearest-x seed binding -----------------------------------------


def _touch(path) -> None:
    path.write_bytes(b"\x00")


def test_build_ladder_binds_backward_states_by_nearest_x(tmp_path) -> None:
    # Four 1-4 backward states; each should attach to the nearest 1-4 rung.
    for x in (100, 700, 1300, 2400):
        _touch(tmp_path / f"stage_1_4_bw_x{x}.state")
    ladder = build_ladder(str(tmp_path / "*.state"))
    order_to_seed = {s.order: s.seed_blob for s in ladder}

    # 1-4 rungs are orders 10..17 with 256-wide buckets (last absorbs the
    # tail). Centers: 128, 384, 640, 896, 1152, 1408, 1664, 2396.
    def seed_x(order: int) -> int:
        blob = order_to_seed[order]
        assert blob is not None, f"rung {order} should have bound a seed"
        return int(blob.rsplit("_x", 1)[1].split(".")[0])

    assert seed_x(10) == 100   # nearest center 128
    assert seed_x(11) == 100   # 384 -> {100:284, 700:316} -> 100
    assert seed_x(12) == 700   # 640
    assert seed_x(13) == 700   # 896 -> {700:196,1300:404} -> 700
    assert seed_x(14) == 1300  # 1152
    assert seed_x(15) == 1300  # 1408
    assert seed_x(17) == 2400  # 2396

    # Non-1-4 rungs have no matching seed -> unbound.
    assert all(order_to_seed[o] is None for o in range(0, 10))


def test_build_ladder_binds_depth_snapshots_by_area_and_x(tmp_path) -> None:
    # depth_<world>_<area>_<x>_<ts>.state.bin — bind by area byte then x.
    _touch(tmp_path / "depth_0_2_400_1699999999.state.bin")   # 1-2 pipes
    _touch(tmp_path / "depth_0_2_1300_1699999999.state.bin")  # 1-2 gauntlet
    _touch(tmp_path / "depth_0_3_100_1699999999.state.bin")   # 1-3 entry
    # A world-2 depth snapshot must NOT bind to any World-1 rung.
    _touch(tmp_path / "depth_1_0_100_1699999999.state.bin")
    ladder = build_ladder([str(tmp_path / "depth_*.state.bin")])
    by_order = {s.order: s.seed_blob for s in ladder}

    # Area 2 rungs are orders 4 (x<900) / 5 (900-1800) / 6.
    assert by_order[4] is not None and "depth_0_2_400" in by_order[4]
    assert by_order[5] is not None and "depth_0_2_1300" in by_order[5]
    # Area 3 entry rung (order 7).
    assert by_order[7] is not None and "depth_0_3_100" in by_order[7]
    # The world-2 snapshot bound nowhere on the World-1 ladder.
    assert all(
        b is None or "depth_1_0" not in b for b in by_order.values()
    )


def test_build_ladder_binds_stage_files_via_meta_anchor(tmp_path) -> None:
    # stage_<NN>.state + sidecar anchor (world<<4|area); no x -> entry rung.
    _touch(tmp_path / "stage_00.state")
    (tmp_path / "stage_00.meta.json").write_text(json.dumps({"anchor": 0x00}))
    _touch(tmp_path / "stage_01.state")
    (tmp_path / "stage_01.meta.json").write_text(json.dumps({"anchor": 0x03}))
    ladder = build_ladder([str(tmp_path / "stage_*.state")])
    by_order = {s.order: s.seed_blob for s in ladder}

    # anchor 0x00 -> area 0 (1-1) entry rung, order 0.
    assert by_order[0] is not None and "stage_00.state" in by_order[0]
    # anchor 0x03 -> area 3 (1-3) entry rung, order 7.
    assert by_order[7] is not None and "stage_01.state" in by_order[7]


def test_build_ladder_missing_files_degrade_gracefully(tmp_path) -> None:
    # A glob that matches nothing, and the explicit no-seed path, both yield
    # the structural ladder with every rung live-fill (seed_blob None).
    for ladder in (
        build_ladder(str(tmp_path / "does_not_exist_*.state")),
        build_ladder(None),
        build_ladder([]),
    ):
        assert len(ladder) == 18
        assert all(s.seed_blob is None for s in ladder)


# ---- region_of on synthetic RAM -------------------------------------


def test_region_of_maps_each_level_slice() -> None:
    # 1-1 (area 0): three buckets.
    assert region_of(_ram(area=0, x=500)) == 0
    assert region_of(_ram(area=0, x=1500)) == 1
    assert region_of(_ram(area=0, x=2500)) == 2
    # x past the flagpole clamps to the level's final rung (no drop).
    assert region_of(_ram(area=0, x=9000)) == 2
    # 1-2 entrance (area 1): single rung.
    assert region_of(_ram(area=1, x=50)) == 3
    # 1-2 underground (area 2): three buckets.
    assert region_of(_ram(area=2, x=100)) == 4
    assert region_of(_ram(area=2, x=1000)) == 5
    assert region_of(_ram(area=2, x=2500)) == 6
    # 1-3 (area 3): three buckets.
    assert region_of(_ram(area=3, x=100)) == 7
    assert region_of(_ram(area=3, x=1000)) == 8
    assert region_of(_ram(area=3, x=2500)) == 9
    # 1-4 (area 4): eight buckets.
    assert region_of(_ram(area=4, x=100)) == 10
    assert region_of(_ram(area=4, x=300)) == 11
    assert region_of(_ram(area=4, x=2400)) == 17


def test_region_of_off_chain_states_are_off_ladder() -> None:
    # Warp into world 2.
    assert region_of(_ram(world=1, area=0, x=200)) == OFF_LADDER
    # Off-allowlist area byte (bonus/warp room).
    assert region_of(_ram(world=0, area=7, x=200)) == OFF_LADDER


# ---- region monotonicity --------------------------------------------


def test_region_monotone_along_a_low_road_playthrough() -> None:
    # A sequential run: 1-1 forward, into 1-2 entrance, 1-2 underground
    # (x resets on the area transition), 1-3, then deep into 1-4. The area
    # byte increases monotonically on the low road, so region must be
    # non-decreasing even though x resets at each area boundary.
    trace = [
        (0, 0), (0, 800), (0, 1600), (0, 3000),      # 1-1
        (1, 0), (1, 500),                            # 1-2 entrance
        (2, 0), (2, 500), (2, 1200), (2, 2400),      # 1-2 underground
        (3, 0), (3, 900), (3, 1900),                 # 1-3
        (4, 0), (4, 500), (4, 1500), (4, 2560),      # 1-4 to the axe
    ]
    regions = [region_of(_ram(area=a, x=x)) for (a, x) in trace]
    assert all(b >= a for a, b in zip(regions, regions[1:])), regions
    # It genuinely climbs the whole ladder, ending on the 1-4 tail.
    assert regions[0] == 0
    assert regions[-1] == 17


def test_region_monotone_in_x_within_each_level() -> None:
    for area in (0, 2, 3, 4):
        prev = OFF_LADDER
        for x in range(0, 3200, 64):
            r = region_of(_ram(area=area, x=x))
            assert r >= prev, f"area {area} x {x}: {r} < {prev}"
            prev = r


# ---- a warp never raises the region ---------------------------------


def test_warp_does_not_raise_region() -> None:
    # Deep in the 1-2 underground, at the warp-pipe approach.
    pre_warp = region_of(_ram(world=0, area=2, x=2500))
    assert pre_warp == 6
    # The 1-2 warp jumps the world byte to World 2 (area 0). Even though
    # area 0 is on the World-1 allowlist, the world-byte rise makes the cell
    # off-chain, so the region drops to OFF_LADDER — it never rises past the
    # sequential frontier the agent legitimately reached.
    post_warp = region_of(_ram(world=1, area=0, x=50))
    assert post_warp == OFF_LADDER
    assert post_warp < pre_warp


def test_warp_cannot_advance_the_frontier_gate() -> None:
    # The advance gate fires on `region_of(ram) >= k+1`. Sitting at rung 6
    # (deep 1-2), a warp state must never satisfy the gate for the next rung.
    k = 6
    warp = _ram(world=1, area=1, x=400)  # world 2, "2-1"-ish
    assert region_of(warp) < k + 1
    assert region_of(warp) == OFF_LADDER


def test_area_to_level_label_matches_sequential_chain() -> None:
    assert AREA_TO_LEVEL_LABEL == {0: "1-1", 1: "1-2", 2: "1-2", 3: "1-3", 4: "1-4"}
