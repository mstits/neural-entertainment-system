"""Behavior-cloning seed cache: demo-path resolution + cache-key hashing.

Lifted out of Trainer so the colon-separator multi-demo escape hatch
and the cache-key hash composition can be exercised in isolation.
The trainer methods that still bear the `_bc_demo_paths` /
`_bc_seed_cache_path` names are one-line wrappers over these.

Cache-key inputs are load-bearing: changing any of (demo file
contents, ROM, game name, action_space, frame_skip, encoder kind)
must produce a different cache file so a stale seed trained on a
mismatched observation distribution never warm-starts a new run.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional, Sequence


log = logging.getLogger(__name__)


def resolve_bc_demo_paths(bc_demo_path) -> list[Path]:
    """Normalize the trainer's `bc_demo_path` field to a list of files.

    Accepts:
      * None / empty string → []
      * a single .state.bin path → [path]
      * a directory → every `*.state.bin` inside (sorted by name)
      * a colon-separated string of paths → split and existence-filter
      * already a list/tuple of paths → coerced to Path objects

    Multi-demo BC is the only path that gives the policy enough state
    coverage for a long game like Zelda (~1k pairs per 5-min demo;
    six varied playthroughs ≈ 10k pairs).
    """
    if not bc_demo_path:
        return []
    paths: list[Path] = []
    spec = bc_demo_path
    if isinstance(spec, (list, tuple)):
        paths = [Path(p) for p in spec]
    elif isinstance(spec, str) and ":" in spec:
        # Colon-separated multi-demo spec from the GUI. The earlier
        # implementation called `Path(spec).exists()` to disambiguate
        # "single path that happens to contain ':'" from "joined
        # multi-demo list" — but on macOS that calls os.stat on the
        # full string, which raises ENAMETOOLONG (errno 63) at ~1024
        # chars. With 30+ demo files joined, the string easily
        # exceeds that. Guard the existence check by length first AND
        # swallow the OSError defensively so an exotic FS error never
        # crashes BC seeding.
        looks_single = len(spec) < 1000
        if looks_single:
            try:
                looks_single = Path(spec).exists()
            except OSError:
                looks_single = False
        if looks_single:
            paths = [Path(spec)]
        else:
            paths = [Path(p) for p in spec.split(":") if p]
    else:
        p = Path(spec)
        if p.is_dir():
            paths = sorted(p.glob("*.state.bin"))
        else:
            paths = [p]
    # Final existence filter, defensively wrapped — any individual
    # path that's too long for stat() shouldn't kill the rest.
    out: list[Path] = []
    for p in paths:
        try:
            if p.exists():
                out.append(p)
        except OSError:
            log.warning("BC demo path skipped (stat failed): %s", p)
    return out


def bc_seed_cache_path(
    demos: Sequence[Path],
    rom_path,
    game_name: str,
    action_space: Sequence,
    frame_skip: int,
    encoder_kind: str,
    checkpoint_dir: Path,
) -> Optional[Path]:
    """Return the cache file path for the BC seed of a given config.

    Hashes ALL demos when more than one is configured so a single
    demo being added/replaced invalidates the cache. `frame_skip`
    and `encoder_kind` are load-bearing:
      * fs=4 vs fs=16 produces different observation distributions —
        re-using the wrong seed silently warm-starts on the wrong one.
      * tile_mlp (175,) vs nature_dqn (4,84,84) state_dicts have
        incompatible shapes — loading one into the other crashes.

    Returns None when there are no demos (nothing to seed from).

    Side effect: prunes stale `bc_seed_*.pt` files in `checkpoint_dir`
    that no longer match the current cache key. Each (demos, ROM,
    action_space, frame_skip, encoder, ...) tuple produces a distinct
    file; the previous combo's file is never useful again.
    """
    if not demos:
        return None
    h = hashlib.sha1()
    for demo in demos:
        h.update(demo.read_bytes())
        h.update(b"||")
    h.update(str(rom_path).encode())
    h.update(b"|")
    h.update(game_name.encode())
    h.update(b"|")
    for entry in action_space:
        h.update(("+".join(entry) + ",").encode())
    h.update(b"|")
    h.update(f"fs={frame_skip}".encode())
    h.update(b"|")
    h.update(f"enc={encoder_kind}".encode())
    digest = h.hexdigest()[:12]
    current_path = Path(checkpoint_dir) / f"bc_seed_{digest}.pt"
    # Best-effort GC of seed files that no longer match the current
    # config. Without this, every config change leaves a 60-200 KB
    # corpse behind in checkpoints/.
    try:
        for stale in Path(checkpoint_dir).glob("bc_seed_*.pt"):
            if stale != current_path:
                try:
                    stale.unlink()
                except OSError as exc:
                    log.debug("could not prune stale BC seed %s: %s", stale, exc)
    except OSError:
        pass
    return current_path
