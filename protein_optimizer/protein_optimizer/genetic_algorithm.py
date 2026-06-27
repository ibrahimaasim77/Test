"""
Genetic Algorithm Module

A clean, biology-agnostic GA engine. It operates entirely on:
  - sequences: List[str]       (the population)
  - scores:    List[float]     (fitness values, higher = better)

It has no knowledge of BioEmu, ESM-2, or amino acids. The scoring
callable is injected — the GA only knows it takes a sequence and returns
a float.

Components:
  - Population initialisation (from seed sequence or explicit list)
  - Evaluation via injected fitness callable
  - Selection: tournament or top-k
  - Crossover via CrossoverOperator
  - Mutation via BaseMutator
  - Elitism: best-N sequences always survive
  - Termination: max generations or convergence patience
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .config import GAConfig
from .mutation import BaseMutator, CrossoverOperator

logger = logging.getLogger(__name__)

# Type alias for a fitness function
FitnessCallable = Callable[[List[str]], List[float]]


# ---------------------------------------------------------------------------
# GA state snapshot
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """Snapshot of a single GA generation."""

    generation: int
    population: List[str]
    scores: List[float]
    best_sequence: str
    best_score: float
    mean_score: float
    diversity: float    # fraction of unique sequences in population

    @classmethod
    def from_evaluation(
        cls,
        generation: int,
        population: List[str],
        scores: List[float],
    ) -> "GenerationResult":
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return cls(
            generation=generation,
            population=list(population),
            scores=list(scores),
            best_sequence=population[best_idx],
            best_score=scores[best_idx],
            mean_score=sum(scores) / len(scores) if scores else 0.0,
            diversity=len(set(population)) / len(population) if population else 0.0,
        )


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------


class TournamentSelector:
    """
    Tournament selection: repeatedly sample k individuals, keep the best.

    Produces exactly `n_select` parents.
    """

    def __init__(self, tournament_size: int, rng: random.Random) -> None:
        self.tournament_size = tournament_size
        self.rng = rng

    def select(
        self, population: List[str], scores: List[float], n_select: int
    ) -> List[str]:
        selected = []
        for _ in range(n_select):
            contestants_idx = self.rng.sample(range(len(population)), self.tournament_size)
            winner_idx = max(contestants_idx, key=lambda i: scores[i])
            selected.append(population[winner_idx])
        return selected


class TopKSelector:
    """
    Deterministic top-k selection. Useful for greedy benchmarking.
    Repeats top sequences to fill the required count.
    """

    def select(
        self, population: List[str], scores: List[float], n_select: int
    ) -> List[str]:
        paired = sorted(zip(scores, population), reverse=True)
        top_seqs = [seq for _, seq in paired[:n_select]]
        # Repeat cyclically to fill n_select if needed
        while len(top_seqs) < n_select:
            top_seqs.extend(top_seqs[: n_select - len(top_seqs)])
        return top_seqs[:n_select]


# ---------------------------------------------------------------------------
# Convergence tracker
# ---------------------------------------------------------------------------


@dataclass
class ConvergenceTracker:
    """Tracks whether the GA has stalled."""

    threshold: float
    patience: int
    _best: float = field(default=float("-inf"), init=False)
    _stale_count: int = field(default=0, init=False)

    def update(self, best_score: float) -> bool:
        """
        Call once per generation.
        Returns True if the GA has converged (stale_count >= patience).
        """
        improvement = best_score - self._best
        if improvement > self.threshold:
            self._best = best_score
            self._stale_count = 0
        else:
            self._stale_count += 1

        return self._stale_count >= self.patience

    @property
    def stale_generations(self) -> int:
        return self._stale_count


# ---------------------------------------------------------------------------
# Main GA class
# ---------------------------------------------------------------------------


class GeneticAlgorithm:
    """
    Evolutionary optimiser for sequences.

    The GA has no knowledge of biology. It receives:
      - An initial population (list of strings)
      - A fitness callable: List[str] → List[float]
      - Mutator and CrossoverOperator instances
      - A GAConfig

    Example::

        ga = GeneticAlgorithm(
            config=cfg.ga,
            fitness_fn=pipeline.evaluate,
            mutator=mutator,
            crossover=crossover_op,
        )
        history = ga.run(initial_population)
        best = ga.best_sequence

    Args:
        config: GAConfig with population size, selection strategy, etc.
        fitness_fn: Callable that scores a batch of sequences.
        mutator: BaseMutator instance (random or ESM-guided).
        crossover: CrossoverOperator instance.
        callbacks: Optional list of callables(GenerationResult) for hooks.
    """

    def __init__(
        self,
        config: GAConfig,
        fitness_fn: FitnessCallable,
        mutator: BaseMutator,
        crossover: CrossoverOperator,
        callbacks: Optional[List[Callable[[GenerationResult], None]]] = None,
    ) -> None:
        self.config = config
        self.fitness_fn = fitness_fn
        self.mutator = mutator
        self.crossover = crossover
        self.callbacks = callbacks or []

        self._rng = random.Random(config.seed)

        self._selector = self._build_selector()
        self._convergence = ConvergenceTracker(
            threshold=config.convergence_threshold,
            patience=config.convergence_patience,
        )

        # Populated during run()
        self.history: List[GenerationResult] = []
        self.best_sequence: Optional[str] = None
        self.best_score: float = float("-inf")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, initial_population: List[str]) -> List[GenerationResult]:
        """
        Execute the full GA loop.

        Args:
            initial_population: Starting sequences. If fewer than
                population_size, will be padded with mutated copies.

        Returns:
            history: One GenerationResult per generation.
        """
        population = self._initialise_population(initial_population)
        self.history.clear()

        logger.info(
            "GA start | pop=%d | max_gen=%d | strategy=%s",
            self.config.population_size,
            self.config.max_generations,
            self.config.selection_strategy,
        )

        for gen in range(self.config.max_generations):
            scores = self.fitness_fn(population)
            result = GenerationResult.from_evaluation(gen, population, scores)
            self.history.append(result)

            # Track global best
            if result.best_score > self.best_score:
                self.best_score = result.best_score
                self.best_sequence = result.best_sequence

            self._log_generation(result)

            for cb in self.callbacks:
                cb(result)

            # Check termination
            converged = self._convergence.update(result.best_score)
            if converged:
                logger.info(
                    "Converged at generation %d (no improvement for %d gens).",
                    gen,
                    self.config.convergence_patience,
                )
                break

            # Evolve
            population = self._evolve(population, scores)

        logger.info(
            "GA complete | best_score=%.4f | generations=%d",
            self.best_score,
            len(self.history),
        )
        return self.history

    # ------------------------------------------------------------------
    # Evolution steps
    # ------------------------------------------------------------------

    def _evolve(self, population: List[str], scores: List[float]) -> List[str]:
        """Produce the next generation from the current population + scores."""
        elite = self._extract_elite(population, scores)

        # How many new individuals we need
        n_offspring = self.config.population_size - len(elite)

        parents = self._selector.select(population, scores, n_offspring)
        offspring = self.crossover.crossover_population(parents)
        offspring = self.mutator.mutate_population(offspring)

        return elite + offspring[: n_offspring]

    def _extract_elite(
        self, population: List[str], scores: List[float]
    ) -> List[str]:
        """Return the top elite_size sequences (always survive)."""
        paired = sorted(zip(scores, population), reverse=True)
        return [seq for _, seq in paired[: self.config.elite_size]]

    def _initialise_population(self, seed_population: List[str]) -> List[str]:
        """
        Build a population of exactly population_size from the seed.
        If seed is larger, subsample. If smaller, pad with mutations.
        """
        target = self.config.population_size
        pop = list(seed_population)

        if len(pop) > target:
            pop = self._rng.sample(pop, target)
        elif len(pop) < target:
            while len(pop) < target:
                parent = self._rng.choice(seed_population)
                pop.append(self.mutator.mutate(parent))

        return pop

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_selector(self):
        if self.config.selection_strategy == "tournament":
            return TournamentSelector(
                tournament_size=self.config.tournament_size,
                rng=self._rng,
            )
        elif self.config.selection_strategy == "top_k":
            return TopKSelector()
        else:
            raise ValueError(
                f"Unknown selection strategy: {self.config.selection_strategy!r}. "
                "Valid options: 'tournament', 'top_k'."
            )

    @staticmethod
    def _log_generation(result: GenerationResult) -> None:
        logger.info(
            "Gen %3d | best=%.4f | mean=%.4f | diversity=%.2f | "
            "stale_gens=%s",
            result.generation,
            result.best_score,
            result.mean_score,
            result.diversity,
            "—",
        )
