"""
Genome name pool — stable, human-memorable identifiers for each
individual in the GA population. Names persist across mutation and
crossover via a monotonic counter: new children get fresh names, old
names die out as their genomes do. The elite carries its name forward
unchanged so viewers can track "Brutus" across dozens of generations.

Why: with anonymous genome IDs, the grid tile layout and streaming
captions become unreadable ("agent 7 just killed Aquamentus"). Names
turn the training into a cast of characters the audience can follow
and adopt favourites from.

Taxonomy leans toward punchy, single-word identifiers mixing martial
nicknames, mythology, animals, and internet-era personality tags. Mix
genders, cultures, registers. Avoid real public figures. ~150 names,
deterministic pick order when a seed is supplied so reproducible runs
produce the same genome-name mapping.
"""

from __future__ import annotations

import random
from typing import Optional


# Keep the pool >> population size so collisions are rare. Each name is
# a single token so it fits in a grid tile corner. Mix of archetypal /
# mythological / pop-cultural tags. NO real public figures.
_NAME_POOL: tuple[str, ...] = (
    # Martial / warrior-coded
    "Brutus", "Ajax", "Valkyrie", "Ronin", "Reaver", "Garrote",
    "Brawler", "Clobber", "Dagger", "Cyclone", "Havoc", "Rampart",
    # Scout / explorer-coded
    "Scout", "Trailblazer", "Nomad", "Wayfarer", "Pathfinder",
    "Ranger", "Drifter", "Voyager", "Vagabond", "Stryder",
    # Mythology
    "Odin", "Freya", "Loki", "Heimdall", "Circe", "Hermes",
    "Athena", "Kali", "Anubis", "Tyr", "Persephone", "Bastet",
    "Morrigan", "Thoth", "Fenrir", "Baldur", "Sif",
    # Pop-cultural / internet
    "Paperclip", "Chonker", "Goblin", "Pylon", "Zigzag", "Quirk",
    "Glitch", "Whisker", "Noodle", "Pickle", "Muffin", "Waffle",
    "Sprocket", "Cobbler", "Tinsel", "Fizz", "Bumble", "Dingo",
    # Nature / animal
    "Falcon", "Mantis", "Gecko", "Barnacle", "Otter", "Walrus",
    "Heron", "Kestrel", "Hyena", "Lynx", "Meerkat", "Pangolin",
    "Capybara", "Tarsier", "Manatee", "Stoat", "Vole",
    # Weather / celestial
    "Zephyr", "Tempest", "Nimbus", "Ember", "Vortex", "Cinder",
    "Flare", "Quasar", "Eclipse", "Solstice", "Aurora", "Draco",
    # Mechanical / inorganic
    "Cog", "Rivet", "Bolt", "Piston", "Gasket", "Flywheel",
    "Socket", "Grommet", "Lathe", "Caliper", "Wrench", "Clamp",
    # Sweet / cutesy contrast
    "Jellybean", "Gumdrop", "Taffy", "Fondue", "Sorbet", "Macaron",
    "Brioche", "Nougat", "Beignet", "Praline", "Cranberry", "Sultana",
    # Card-game / fantasy rogue
    "Ace", "Dealer", "Trickster", "Jinx", "Scoundrel", "Hustler",
    "Mischief", "Bandit", "Pilferer", "Grift", "Chancer",
    # Abstract / pattern
    "Echo", "Silhouette", "Mirage", "Paradox", "Ripple", "Fractal",
    "Drift", "Parallax", "Kismet", "Zeitgeist", "Epsilon", "Lambda",
)


def _base_names() -> tuple[str, ...]:
    """Exposed for tests — the pool is static but tests may want to
    assert pool size / uniqueness without importing the private name."""
    return _NAME_POOL


class GenomeNamer:
    """Assigns unique, stable names to each genome created in the GA.

    One instance lives on the GeneticAlgorithm; every call to
    `next_name()` returns a fresh name. If the pool runs out (which
    only happens after > `len(pool)` distinct genomes — on Zelda
    that's thousands of generations), we append a numeric suffix so
    names stay human-readable but remain unique.

    Seeding is per-GeneticAlgorithm so a `--seed N` run produces the
    same genome-name mapping twice.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        # Shuffle a private copy so we don't mutate the module-level
        # constant (pytest parallel runs would race).
        self._remaining: list[str] = list(_NAME_POOL)
        # Deterministic shuffle when a seed is provided.
        rng = random.Random(seed)
        rng.shuffle(self._remaining)
        # Suffix counter for when the pool is exhausted.
        self._used_count = 0

    def next_name(self) -> str:
        if self._remaining:
            name = self._remaining.pop()
            self._used_count += 1
            return name
        # Pool drained — synthesise unique names by round-robin with
        # numeric suffix. Keeps names readable ("Brutus-2") even after
        # thousands of generations. Numbering starts at -2 because the
        # FIRST occurrence of each name was the original pop-from-pool
        # (no suffix); the second occurrence is the first synthesised.
        idx = self._used_count % len(_NAME_POOL)
        cycle = self._used_count // len(_NAME_POOL)  # 1, 2, 3, ...
        self._used_count += 1
        return f"{_NAME_POOL[idx]}-{cycle + 1}"

    def __len__(self) -> int:
        return self._used_count
