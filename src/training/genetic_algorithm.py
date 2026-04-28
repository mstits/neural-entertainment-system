"""
Genetic algorithm over neural network weights.

Each "genome" is a PolicyNetwork's state_dict. Operations:
- Selection: tournament (pick K random, keep the best)
- Elitism: top N genomes survive unchanged to the next generation
- Crossover: per-parameter weighted average of two parents
- Mutation: Gaussian noise added to each weight with some probability

This is simpler than NEAT (no topology evolution) but works well when paired
with gradient-based updates on the best genomes. The GA handles exploration,
gradient descent handles exploitation.

Starting point: Uber's "Deep Neuroevolution" paper showed this approach beats
Nature-DQN on many Atari games.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from src.models.policy_network import PolicyNetwork
from src.training.genome_names import GenomeNamer


log = logging.getLogger(__name__)


@dataclass
class Genome:
    """One individual in the population.

    `name` is a human-memorable label assigned when the Genome is
    created ("Brutus", "Scout", …). Elites keep the same name across
    generations since they're the same underlying network object; new
    children from crossover/mutation get fresh names. Viewers can
    follow specific genomes across the stream — especially the elite,
    which often stays named for dozens of generations."""
    genome_id: int
    state_dict: dict
    fitness: float = -float("inf")
    generation: int = 0
    parent_ids: tuple[int, ...] = field(default_factory=tuple)
    name: str = ""


class GeneticAlgorithm:
    """
    Evolves a population of PolicyNetwork weights.

    Typical usage:
        ga = GeneticAlgorithm(population_size=16, network_factory=...)
        ga.initialize()
        for generation in range(num_generations):
            # Evaluate fitness on each genome via the emulation pool
            for genome in ga.population:
                genome.fitness = evaluate(genome)
            ga.evolve()
    """

    def __init__(
        self,
        population_size: int,
        network_factory,  # callable returning a fresh PolicyNetwork
        mutation_rate: float = 0.2,
        mutation_std: float = 0.05,
        elite_fraction: float = 0.2,
        tournament_size: int = 5,
        # Stale-best diversity injection (MarI/O-style): if best fitness
        # has not improved for this many consecutive generations, wipe
        # `restart_fraction` of the post-evolve population (from the
        # bottom up, never elites) and replace them with freshly
        # initialised networks. 0 disables the mechanism.
        stale_gens_before_restart: int = 0,
        restart_fraction: float = 0.5,
        fitness_improvement_epsilon: float = 1e-6,
        seed: Optional[int] = None,
    ) -> None:
        self.population_size = population_size
        self.network_factory = network_factory
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.elite_fraction = elite_fraction
        self.tournament_size = tournament_size
        self.stale_gens_before_restart = int(stale_gens_before_restart)
        self.restart_fraction = float(restart_fraction)
        self.fitness_improvement_epsilon = float(fitness_improvement_epsilon)

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self.population: list[Genome] = []
        self.generation = 0
        self._next_genome_id = 0
        self._best_fitness_seen: float = -float("inf")
        self._gens_since_improvement: int = 0
        # Deterministic when seeded so --seed N reproduces the same
        # genome-name mapping across runs.
        self._namer = GenomeNamer(seed=seed)

    def _new_genome_id(self) -> int:
        gid = self._next_genome_id
        self._next_genome_id += 1
        return gid

    def _fresh_name(self) -> str:
        """Issue a new unique name for a fresh genome."""
        return self._namer.next_name()

    def initialize(self) -> None:
        """Create the initial population of randomly-initialized networks."""
        self.population = []
        for _ in range(self.population_size):
            net = self.network_factory()
            self.population.append(Genome(
                genome_id=self._new_genome_id(),
                state_dict=copy.deepcopy(net.state_dict()),
                generation=0,
                name=self._fresh_name(),
            ))

    def evolve(self) -> None:
        """
        Run one generation of evolution. Assumes fitness has been set on every
        genome in self.population.
        """
        if not self.population:
            raise RuntimeError("Population is empty. Call initialize() first.")

        # Sort descending by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)

        # Stale-best bookkeeping — done on the CURRENT (just-evaluated)
        # population, before we bred the next one. Updated here so a
        # restart is triggered off freshly-scored fitness, not from
        # children that have no fitness yet.
        current_best = self.population[0].fitness
        if current_best > self._best_fitness_seen + self.fitness_improvement_epsilon:
            self._best_fitness_seen = current_best
            self._gens_since_improvement = 0
        else:
            self._gens_since_improvement += 1

        # Elitism: top N survive
        num_elite = max(1, int(self.elite_fraction * self.population_size))
        elite = self.population[:num_elite]

        # Fill the rest via tournament selection + crossover + mutation
        new_population = list(elite)  # elites carry over unchanged

        while len(new_population) < self.population_size:
            parent_a = self._tournament_select()
            parent_b = self._tournament_select()
            child_state = self._crossover(parent_a.state_dict, parent_b.state_dict)
            child_state = self._mutate(child_state)
            new_population.append(Genome(
                genome_id=self._new_genome_id(),
                state_dict=child_state,
                generation=self.generation + 1,
                parent_ids=(parent_a.genome_id, parent_b.genome_id),
                name=self._fresh_name(),
            ))

        # Stale-best restart: the best-seen fitness has been stuck for
        # `stale_gens_before_restart` generations, so replace the
        # bottom-ranked non-elite children with freshly-initialised
        # networks to inject diversity. Elites are preserved so the
        # current best solution isn't lost.
        if (
            self.stale_gens_before_restart > 0
            and self._gens_since_improvement >= self.stale_gens_before_restart
        ):
            n_restart = int(self.restart_fraction * self.population_size)
            n_restart = min(n_restart, self.population_size - num_elite)
            if n_restart > 0:
                log.info(
                    "GA stale-best restart: best fitness %.3f stuck for %d gens; "
                    "replacing %d bottom genomes with fresh random weights.",
                    self._best_fitness_seen,
                    self._gens_since_improvement,
                    n_restart,
                )
                for i in range(n_restart):
                    net = self.network_factory()
                    new_population[-(i + 1)] = Genome(
                        genome_id=self._new_genome_id(),
                        state_dict=copy.deepcopy(net.state_dict()),
                        generation=self.generation + 1,
                        name=self._fresh_name(),
                    )
                # Reset the staleness counter so we don't restart every
                # subsequent gen until fitness happens to climb.
                self._gens_since_improvement = 0

        self.population = new_population
        self.generation += 1

    def _tournament_select(self) -> Genome:
        """Pick K random genomes, return the fittest. K is clamped to the
        current population size so tiny populations still work."""
        k = min(self.tournament_size, len(self.population))
        candidates = random.sample(self.population, k)
        return max(candidates, key=lambda g: g.fitness)

    def _crossover(self, parent_a: dict, parent_b: dict) -> dict:
        """Uniform crossover: per-parameter coin flip between parents."""
        child = {}
        for key in parent_a:
            tensor = parent_a[key]
            # `torch.rand_like` requires a float dtype. PolicyNetwork has
            # only float weights today, but BatchNorm-style integer
            # buffers (e.g. `num_batches_tracked`) sneak in as soon as
            # someone adds a normalisation layer. Build the mask from a
            # float tensor of matching shape so the call doesn't crash
            # in that scenario; integer state survives unchanged from
            # parent_a (deterministic — identical between parents anyway
            # since they were both seeded from the same architecture).
            if tensor.is_floating_point():
                # Parents may disagree on keys when one was modified in
                # _reinforce_update (which writes back the CURRENT
                # PolicyNetwork's full state_dict — including any keys
                # added since the checkpoint was saved, e.g. value_head
                # added with PPO+GAE on 2026-04-21) and the other was
                # loaded from a pre-add checkpoint. Fall back to
                # parent_a's value when parent_b is missing the key so
                # population evolution continues cleanly. The diverged
                # key will reconcile: every elite promoted through
                # REINFORCE carries forward the full state_dict, so
                # within a few generations all genomes have it.
                b_val = parent_b.get(key, tensor)
                mask = torch.rand_like(tensor) < 0.5
                child[key] = torch.where(mask, tensor, b_val)
            else:
                child[key] = tensor.clone()
        return child

    def _mutate(self, state_dict: dict) -> dict:
        """Add Gaussian noise to a fraction of weights.

        The previous implementation chose mutation per-tensor: with
        probability `mutation_rate` an ENTIRE layer's weights got
        Gaussian noise added; otherwise the whole layer carried over
        unchanged. That's coarse — at `mutation_std=0.05` it's either
        a hammer to a whole conv kernel or nothing at all.

        Per-element mutation gives finer-grained exploration: each
        weight is independently perturbed with probability
        `mutation_rate`, with the same standard deviation. Average
        wall-clock cost is unchanged (same Gaussian draw count); the
        gradient of evolutionary search is just smoother.
        """
        mutated = {}
        for key, tensor in state_dict.items():
            if not tensor.is_floating_point():
                # Integer buffers (BN counters etc.) don't accept noise.
                mutated[key] = tensor.clone()
                continue
            mask = (torch.rand_like(tensor) < self.mutation_rate).to(tensor.dtype)
            noise = torch.randn_like(tensor) * self.mutation_std * mask
            mutated[key] = tensor + noise
        return mutated

    def best_genome(self) -> Genome:
        """Return the fittest genome in the current population."""
        return max(self.population, key=lambda g: g.fitness)

    def save_checkpoint(self, path: str) -> None:
        """Pickle the entire population to disk."""
        torch.save({
            "generation": self.generation,
            "population": [
                {
                    "genome_id": g.genome_id,
                    "state_dict": g.state_dict,
                    "fitness": g.fitness,
                    "generation": g.generation,
                    "parent_ids": g.parent_ids,
                    "name": g.name,
                }
                for g in self.population
            ],
            "next_genome_id": self._next_genome_id,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Restore population from disk."""
        # weights_only=False required because the checkpoint stores a dict
        # with nested state_dicts; torch>=2.6 defaults it to True and rejects.
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.generation = data["generation"]
        self._next_genome_id = data["next_genome_id"]
        self.population = [
            Genome(
                genome_id=g["genome_id"],
                state_dict=g["state_dict"],
                fitness=g["fitness"],
                generation=g["generation"],
                parent_ids=tuple(g["parent_ids"]),
                # Back-compat: checkpoints from before names existed have
                # no `name` key. Synthesise a fresh one so the rest of the
                # pipeline never has to handle empty-string names.
                name=g.get("name") or self._namer.next_name(),
            )
            for g in data["population"]
        ]
