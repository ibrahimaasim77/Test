"""
Mutation and Crossover Module

Provides:
  - BaseMutator: abstract contract for all mutators
  - RandomMutator: uniform random amino acid substitutions (baseline)
  - ESMGuidedMutator: ESM-2 log-prob weighted substitutions (preferred)
  - CrossoverOperator: single-point, two-point, and uniform crossover

The GA only interacts with BaseMutator and CrossoverOperator.
BioEmu/ESM2 are never imported here directly — only ESM2MutationProposer
is injected at construction time, keeping modules decoupled.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from .config import MutationConfig
from .esm import AMINO_ACIDS, ESM2MutationProposer


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseMutator(ABC):
    """
    Abstract mutation operator.

    All mutators take a sequence string and return a new (possibly mutated)
    sequence string. The GA calls `mutate_population` for batch efficiency.
    """

    def __init__(self, config: MutationConfig, rng: random.Random) -> None:
        self.config = config
        self.rng = rng

    @abstractmethod
    def mutate(self, sequence: str) -> str:
        """Return a mutated copy of *sequence*."""

    def mutate_population(self, sequences: List[str]) -> List[str]:
        """Batch mutation. Override for vectorised implementations."""
        return [self.mutate(seq) for seq in sequences]

    # ------------------------------------------------------------------

    def _candidate_positions(self, sequence: str) -> List[int]:
        """Return positions eligible for mutation."""
        if self.config.allowed_positions is not None:
            return [p for p in self.config.allowed_positions if p < len(sequence)]
        return list(range(len(sequence)))

    def _apply_substitution(self, sequence: str, position: int, new_aa: str) -> str:
        seq = list(sequence)
        seq[position] = new_aa
        return "".join(seq)


# ---------------------------------------------------------------------------
# Random mutator
# ---------------------------------------------------------------------------


class RandomMutator(BaseMutator):
    """
    Baseline mutator: choose a random position, substitute a random AA.

    Number of mutations drawn uniformly from [1, max_mutations_per_sequence].
    """

    def mutate(self, sequence: str) -> str:
        n_mutations = self.rng.randint(1, self.config.max_mutations_per_sequence)
        positions = self._candidate_positions(sequence)
        if not positions:
            return sequence

        chosen = self.rng.sample(positions, min(n_mutations, len(positions)))
        result = sequence
        for pos in chosen:
            current_aa = result[pos]
            candidates = [aa for aa in AMINO_ACIDS if aa != current_aa]
            result = self._apply_substitution(result, pos, self.rng.choice(candidates))
        return result


# ---------------------------------------------------------------------------
# ESM-2 guided mutator
# ---------------------------------------------------------------------------


class ESMGuidedMutator(BaseMutator):
    """
    Mutation operator that uses ESM-2 log-probabilities to bias substitutions
    toward biologically plausible replacements.

    Sampling strategy:
      - For each selected position, obtain top-k candidates from ESM-2.
      - Sample from those candidates using a softmax over their log-probs
        (so higher-probability AAs are preferred but not deterministic).

    Falls back to RandomMutator behaviour when ESM-2 proposals are unavailable.

    Args:
        config: MutationConfig.
        esm_proposer: A fully-initialised ESM2MutationProposer.
        rng: shared Random instance for reproducibility.
    """

    def __init__(
        self,
        config: MutationConfig,
        esm_proposer: ESM2MutationProposer,
        rng: random.Random,
    ) -> None:
        super().__init__(config, rng)
        self._esm = esm_proposer
        self._fallback = RandomMutator(config, rng)

    def mutate(self, sequence: str) -> str:
        n_mutations = self.rng.randint(1, self.config.max_mutations_per_sequence)
        positions = self._candidate_positions(sequence)
        if not positions:
            return sequence

        chosen_positions = self.rng.sample(positions, min(n_mutations, len(positions)))

        try:
            proposals = self._esm.propose(sequence, chosen_positions)
        except Exception:
            # Graceful fallback when ESM-2 is unavailable
            return self._fallback.mutate(sequence)

        result = sequence
        for pos in chosen_positions:
            candidates = [
                c for c in proposals.get(pos, []) if c.proposed_aa != result[pos]
            ]
            if not candidates:
                result = self._fallback.mutate(result)
                continue
            chosen_aa = self._weighted_sample(candidates)
            result = self._apply_substitution(result, pos, chosen_aa)
        return result

    @staticmethod
    def _weighted_sample(candidates) -> str:  # type: ignore[override]
        """Softmax-weighted sampling over MutationCandidate log-probs."""
        import math

        # Numerical stability: subtract max before exp
        log_probs = [c.log_prob for c in candidates]
        max_lp = max(log_probs)
        weights = [math.exp(lp - max_lp) for lp in log_probs]
        total = sum(weights)
        normalised = [w / total for w in weights]

        r = random.random()
        cumulative = 0.0
        for candidate, w in zip(candidates, normalised):
            cumulative += w
            if r <= cumulative:
                return candidate.proposed_aa
        return candidates[-1].proposed_aa


# ---------------------------------------------------------------------------
# Crossover operator
# ---------------------------------------------------------------------------


class CrossoverOperator:
    """
    Recombines two parent sequences to produce two offspring.

    Supported strategies:
      - "single_point": one crossover point
      - "two_point": two crossover points
      - "uniform": each position independently swapped with probability 0.5

    Sequences must be equal length. If they differ (shouldn't happen in a
    well-formed population), the shorter parent's length is used as the
    safe boundary.
    """

    def __init__(self, config: MutationConfig, rng: random.Random) -> None:
        self.config = config
        self.rng = rng
        self._strategies = {
            "single_point": self._single_point,
            "two_point": self._two_point,
            "uniform": self._uniform,
        }

    def crossover(self, parent_a: str, parent_b: str) -> Tuple[str, str]:
        """
        Return two offspring. If crossover rate not triggered, return clones.
        """
        if self.rng.random() > self.config.crossover_rate:
            return parent_a, parent_b

        strategy = self._strategies.get(self.config.crossover_strategy)
        if strategy is None:
            raise ValueError(
                f"Unknown crossover strategy: {self.config.crossover_strategy!r}. "
                f"Valid options: {list(self._strategies)}"
            )
        return strategy(parent_a, parent_b)

    def crossover_population(
        self, parents: List[str]
    ) -> List[str]:
        """
        Pair up parents sequentially, apply crossover, flatten results.
        If population size is odd, the last parent is cloned unchanged.
        """
        offspring: List[str] = []
        for i in range(0, len(parents) - 1, 2):
            child_a, child_b = self.crossover(parents[i], parents[i + 1])
            offspring.extend([child_a, child_b])
        if len(parents) % 2 == 1:
            offspring.append(parents[-1])
        return offspring

    # ------------------------------------------------------------------

    def _single_point(self, a: str, b: str) -> Tuple[str, str]:
        length = min(len(a), len(b))
        cut = self.rng.randint(1, length - 1)
        child_a = a[:cut] + b[cut:]
        child_b = b[:cut] + a[cut:]
        return child_a, child_b

    def _two_point(self, a: str, b: str) -> Tuple[str, str]:
        length = min(len(a), len(b))
        cut1, cut2 = sorted(self.rng.sample(range(1, length), 2))
        child_a = a[:cut1] + b[cut1:cut2] + a[cut2:]
        child_b = b[:cut1] + a[cut1:cut2] + b[cut2:]
        return child_a, child_b

    def _uniform(self, a: str, b: str) -> Tuple[str, str]:
        child_a, child_b = [], []
        for aa, ab in zip(a, b):
            if self.rng.random() < 0.5:
                child_a.append(aa)
                child_b.append(ab)
            else:
                child_a.append(ab)
                child_b.append(aa)
        # Preserve any tail if sequences differ in length (defensive)
        if len(a) > len(b):
            child_a.extend(list(a[len(b):]))
            child_b.extend(list(a[len(b):]))
        elif len(b) > len(a):
            child_a.extend(list(b[len(a):]))
            child_b.extend(list(b[len(a):]))
        return "".join(child_a), "".join(child_b)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_mutator(
    config: MutationConfig,
    rng: random.Random,
    esm_proposer: Optional[ESM2MutationProposer] = None,
) -> BaseMutator:
    """
    Construct the appropriate mutator given the strategy in config.

    Pass esm_proposer=None to force RandomMutator regardless of config.
    """
    if config.strategy == "esm_guided":
        if esm_proposer is None:
            raise ValueError(
                "ESM-2 guided mutation requires an ESM2MutationProposer instance. "
                "Either pass one or set mutation.strategy = 'random'."
            )
        return ESMGuidedMutator(config, esm_proposer, rng)
    elif config.strategy == "random":
        return RandomMutator(config, rng)
    else:
        raise ValueError(
            f"Unknown mutation strategy: {config.strategy!r}. "
            "Valid options: 'random', 'esm_guided'."
        )
