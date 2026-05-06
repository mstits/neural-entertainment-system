"""Game-specific tile observation extractors.

Tile-based observations dramatically shrink the policy's input space
compared to raw 4×84×84 pixels. Instead of asking the network to learn
"this brown rectangle is a brick" from scratch via convolutions, we
decode the game's RAM directly into a small grid of semantic tile
codes (empty / solid / enemy / etc.) plus a handful of game-state
scalars (velocity, power-up state, lives).

Why this matters for the training loop:

* **Search space shrinks ~100×.** A 175-int feature vector vs 28k
  pixel values means the policy network can be ~14k params instead of
  ~1.7M. GA mutation finds working policies; PPO gradients are large
  enough relative to the parameter count to actually steer learning.
* **Credit assignment is simpler.** "Mario was on a tile next to a
  pit and pressed nothing" is one feature; "the brown pixels at row
  47 were 4 pixels lower than usual" is dozens of features the conv
  stack has to integrate. Dense, semantic features make the value
  head's job tractable.
* **No more "MarI/O lua trick" envy.** SethBling's Mario neural net
  worked because tile-based input is the right abstraction for
  platformers. We're adopting the same recipe.

Each game implements the `TileObservation` protocol independently
(SMB's RAM layout differs from Zelda's, etc.). Game-agnostic
infrastructure (the policy MLP, the trainer's encoder dispatch) is
reused 100% across games.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class TileObservation(Protocol):
    """One game's RAM → feature-vector decoder.

    Implementations live in `src/emulation/tile_observations/<game>.py`.
    The trainer dispatches based on `reinforce.encoder` in the profile
    YAML — for example `encoder: smb_tiles` selects `SMBTileObservation`.
    """

    @property
    def feature_dim(self) -> int:
        """Total length of the feature vector returned by `extract`.

        Must be a constant — the policy network's input layer is sized
        from this at construction time.
        """
        ...

    def extract(self, ram_bytes: bytes) -> np.ndarray:
        """Decode a 2 KB CPU RAM snapshot into a flat int8 feature vector.

        Returns a `(feature_dim,)` int8 array. The network casts to
        float and normalizes via its own learned weights — no global
        normalization needed since features are already small ints.
        """
        ...


def get_extractor(name: str) -> TileObservation:
    """Factory: profile name → extractor instance.

    Adding a new game is `tile_observations/<game>.py` plus an entry
    here. No trainer code changes.
    """
    if name == "smb_tiles":
        from src.emulation.tile_observations.smb import SMBTileObservation
        return SMBTileObservation()
    raise ValueError(
        f"unknown tile encoder {name!r}; "
        "expected one of: smb_tiles"
    )
