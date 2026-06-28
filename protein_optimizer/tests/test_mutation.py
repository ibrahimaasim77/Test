"""Tests for mutation and crossover operators."""

import random
import pytest

from protein_optimizer.config import MutationConfig
from protein_optimizer.mutation import (
    CrossoverOperator,
    RandomMutator,
    build_mutator,
)
from protein_optimizer.esm import AMINO_ACIDS

SEQ = "MKTLLILAVLCLGFAQAS"


@pytest.fixture
def rng():
    return random.Random(0)


@pytest.fixture
def random_mutator(rng):
    cfg = MutationConfig(strategy="random", max_mutations_per_sequence=3)
    return RandomMutator(cfg, rng)


@pytest.fixture
def crossover_op(rng):
    cfg = MutationConfig(crossover_strategy="two_point", crossover_rate=1.0)
    return CrossoverOperator(cfg, rng)


class TestRandomMutator:
    def test_output_same_length(self, random_mutator):
        result = random_mutator.mutate(SEQ)
        assert len(result) == len(SEQ)

    def test_output_valid_amino_acids(self, random_mutator):
        result = random_mutator.mutate(SEQ)
        for aa in result:
            assert aa in AMINO_ACIDS

    def test_at_least_one_mutation(self, random_mutator):
        # With max_mutations=3 and rate driven by sample, should differ from original
        results = {random_mutator.mutate(SEQ) for _ in range(50)}
        assert any(r != SEQ for r in results)

    def test_allowed_positions_respected(self, rng):
        cfg = MutationConfig(
            strategy="random",
            max_mutations_per_sequence=5,
            allowed_positions=[0, 1, 2],
        )
        mutator = RandomMutator(cfg, rng)
        for _ in range(30):
            result = mutator.mutate(SEQ)
            # Positions 3+ must be unchanged
            assert result[3:] == SEQ[3:]

    def test_population_mutation(self, random_mutator):
        population = [SEQ] * 10
        results = random_mutator.mutate_population(population)
        assert len(results) == 10
        assert all(len(r) == len(SEQ) for r in results)


class TestCrossoverOperator:
    def test_single_point_length(self, rng):
        cfg = MutationConfig(crossover_strategy="single_point", crossover_rate=1.0)
        op = CrossoverOperator(cfg, rng)
        c1, c2 = op.crossover(SEQ, SEQ[::-1])
        assert len(c1) == len(SEQ)
        assert len(c2) == len(SEQ)

    def test_two_point_preserves_characters(self, crossover_op):
        parent_a = "AAAAAAAAAAAAAAAAAAAA"[:len(SEQ)]
        parent_b = "CCCCCCCCCCCCCCCCCCCC"[:len(SEQ)]
        c1, c2 = crossover_op.crossover(parent_a, parent_b)
        # All characters must be either A or C
        for aa in c1 + c2:
            assert aa in ("A", "C")

    def test_uniform_crossover(self, rng):
        cfg = MutationConfig(crossover_strategy="uniform", crossover_rate=1.0)
        op = CrossoverOperator(cfg, rng)
        c1, c2 = op.crossover(SEQ, SEQ[::-1])
        assert len(c1) == len(SEQ)
        assert len(c2) == len(SEQ)

    def test_crossover_rate_zero_returns_clones(self, rng):
        cfg = MutationConfig(crossover_strategy="two_point", crossover_rate=0.0)
        op = CrossoverOperator(cfg, rng)
        c1, c2 = op.crossover(SEQ, SEQ[::-1])
        assert c1 == SEQ
        assert c2 == SEQ[::-1]

    def test_crossover_population_size(self, crossover_op):
        parents = [SEQ] * 8
        offspring = crossover_op.crossover_population(parents)
        assert len(offspring) == 8

    def test_invalid_strategy_raises(self, rng):
        cfg = MutationConfig(crossover_strategy="invalid_xyz", crossover_rate=1.0)
        op = CrossoverOperator(cfg, rng)
        with pytest.raises(ValueError, match="Unknown crossover strategy"):
            op.crossover(SEQ, SEQ)


class TestBuildMutator:
    def test_random_strategy(self, rng):
        cfg = MutationConfig(strategy="random")
        mutator = build_mutator(cfg, rng, esm_proposer=None)
        assert isinstance(mutator, RandomMutator)

    def test_esm_strategy_without_proposer_raises(self, rng):
        cfg = MutationConfig(strategy="esm_guided")
        with pytest.raises(ValueError, match="ESM-2 guided mutation requires"):
            build_mutator(cfg, rng, esm_proposer=None)

    def test_unknown_strategy_raises(self, rng):
        cfg = MutationConfig(strategy="unknown_xyz")
        with pytest.raises(ValueError, match="Unknown mutation strategy"):
            build_mutator(cfg, rng, esm_proposer=None)
