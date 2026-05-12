"""Tests for BC demo-path resolution and seed cache-key hashing.

The colon-separator multi-demo escape hatch is the trickiest piece
— `Path(spec).exists()` on a long colon-joined string used to
raise ENAMETOOLONG on macOS and kill BC seeding before the trainer
even reached the first generation. Lock that path down.

Cache-key invalidation: any change to (demo contents, ROM, game
name, action_space, frame_skip, encoder_kind) must produce a
different file so a stale seed never warm-starts the wrong
distribution.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.training.bc_seed_cache import bc_seed_cache_path, resolve_bc_demo_paths


def _touch(path: Path, contents: bytes = b"x") -> Path:
    path.write_bytes(contents)
    return path


def test_resolve_empty_returns_empty_list() -> None:
    assert resolve_bc_demo_paths(None) == []
    assert resolve_bc_demo_paths("") == []


def test_resolve_single_path(tmp_path: Path) -> None:
    demo = _touch(tmp_path / "a.state.bin")
    assert resolve_bc_demo_paths(str(demo)) == [demo]


def test_resolve_directory_globs_state_bins(tmp_path: Path) -> None:
    _touch(tmp_path / "a.state.bin")
    _touch(tmp_path / "b.state.bin")
    _touch(tmp_path / "noise.txt")  # should be ignored
    out = resolve_bc_demo_paths(str(tmp_path))
    assert [p.name for p in out] == ["a.state.bin", "b.state.bin"]


def test_resolve_colon_separator_list(tmp_path: Path) -> None:
    a = _touch(tmp_path / "a.state.bin")
    b = _touch(tmp_path / "b.state.bin")
    spec = f"{a}:{b}"
    assert resolve_bc_demo_paths(spec) == [a, b]


def test_resolve_colon_separator_skips_missing(tmp_path: Path) -> None:
    """Existence filter must drop entries that don't exist on disk."""
    a = _touch(tmp_path / "a.state.bin")
    bogus = tmp_path / "missing.state.bin"
    assert resolve_bc_demo_paths(f"{a}:{bogus}") == [a]


def test_resolve_list_input_coerces_to_paths(tmp_path: Path) -> None:
    a = _touch(tmp_path / "a.state.bin")
    b = _touch(tmp_path / "b.state.bin")
    out = resolve_bc_demo_paths([str(a), str(b)])
    assert out == [a, b]


def test_resolve_long_colon_spec_does_not_raise_enametoolong(tmp_path: Path) -> None:
    """ENAMETOOLONG guard: a 30+ demo colon-joined spec must not crash
    on macOS even though Path(spec).exists() on the full string would
    raise OSError(63). This was the bug ENAMETOOLONG fix shipped for."""
    demos = [_touch(tmp_path / f"d{i:03d}.state.bin") for i in range(40)]
    spec = ":".join(str(p) for p in demos)
    assert len(spec) > 1000  # confirm we're past the guard threshold
    out = resolve_bc_demo_paths(spec)
    assert out == demos


def test_cache_path_is_none_when_no_demos(tmp_path: Path) -> None:
    assert bc_seed_cache_path(
        demos=[],
        rom_path="/dev/null",
        game_name="x",
        action_space=[[]],
        frame_skip=4,
        encoder_kind="nature_dqn",
        checkpoint_dir=tmp_path,
    ) is None


def test_cache_path_changes_when_demo_contents_change(tmp_path: Path) -> None:
    """Same path, different bytes → different cache file. Without this,
    re-recording a demo would silently load a stale seed."""
    demo = tmp_path / "a.state.bin"
    demo.write_bytes(b"version_one")
    p1 = bc_seed_cache_path(
        demos=[demo], rom_path="/r", game_name="g", action_space=[[]],
        frame_skip=4, encoder_kind="nature_dqn", checkpoint_dir=tmp_path,
    )
    demo.write_bytes(b"version_two")
    p2 = bc_seed_cache_path(
        demos=[demo], rom_path="/r", game_name="g", action_space=[[]],
        frame_skip=4, encoder_kind="nature_dqn", checkpoint_dir=tmp_path,
    )
    assert p1 != p2


@pytest.mark.parametrize("changed_kwarg,first,second", [
    ("rom_path",      "/rom-a", "/rom-b"),
    ("game_name",     "smb",    "zelda"),
    ("frame_skip",    4,        16),
    ("encoder_kind",  "nature_dqn", "tile_mlp"),
])
def test_cache_path_changes_with_each_load_bearing_input(
    tmp_path: Path, changed_kwarg: str, first, second,
) -> None:
    demo = _touch(tmp_path / "a.state.bin", b"fixed")
    base = dict(
        demos=[demo],
        rom_path="/rom",
        game_name="g",
        action_space=[[]],
        frame_skip=4,
        encoder_kind="nature_dqn",
        checkpoint_dir=tmp_path,
    )
    p1 = bc_seed_cache_path(**{**base, changed_kwarg: first})
    p2 = bc_seed_cache_path(**{**base, changed_kwarg: second})
    assert p1 != p2, f"{changed_kwarg} changed but cache key didn't"


def test_cache_path_prunes_stale_seeds(tmp_path: Path) -> None:
    """A bc_seed_*.pt file from a previous config combination must be
    deleted when the current call produces a different digest."""
    stale = tmp_path / "bc_seed_deadbeef0123.pt"
    stale.write_bytes(b"stale")
    demo = _touch(tmp_path / "a.state.bin")
    current = bc_seed_cache_path(
        demos=[demo], rom_path="/r", game_name="g", action_space=[[]],
        frame_skip=4, encoder_kind="nature_dqn", checkpoint_dir=tmp_path,
    )
    assert current is not None and current.name.startswith("bc_seed_")
    assert not stale.exists(), "stale BC seed not pruned"
    # The current path itself wasn't created — it's just the key.
    assert not current.exists()


def test_cache_path_does_not_delete_current(tmp_path: Path) -> None:
    """An existing file at the current cache path must NOT be unlinked
    by the prune step — that file IS the valid cache for this config."""
    demo = _touch(tmp_path / "a.state.bin", b"fixed")
    cur = bc_seed_cache_path(
        demos=[demo], rom_path="/r", game_name="g", action_space=[[]],
        frame_skip=4, encoder_kind="nature_dqn", checkpoint_dir=tmp_path,
    )
    assert cur is not None
    cur.write_bytes(b"valid_seed")
    cur2 = bc_seed_cache_path(
        demos=[demo], rom_path="/r", game_name="g", action_space=[[]],
        frame_skip=4, encoder_kind="nature_dqn", checkpoint_dir=tmp_path,
    )
    assert cur2 == cur
    assert cur.exists()
    assert cur.read_bytes() == b"valid_seed"
