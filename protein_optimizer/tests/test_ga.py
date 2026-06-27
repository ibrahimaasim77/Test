"""Tests for GeneticAlgorithm and the full pipeline (mock mode)."""

import random
import pytest

from protein_optimizer.config import GAConfig, MutationConfig, OptimizationConfig
from protein_optimizer.genetic_algorithm import (
    ConvergenceTracker,
    GeneticAlgorithm,
    GenerationResult,
    TopKSelector,
    TournamentSelector,
)
from protein_optimizer.mutation import CrossoverOperator, RandomMutator
from protein_optimizer.pipeline import ProteinOptimizationPipeline

SEQ = "MKTLLILAVLCLGFAQAS"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return random.Random(42)


@pytest.fixture
def mutator(rng):
    cfg = MutationConfig(strategy="random", max_mutations_per_sequence=2)
    return RandomMutator(cfg, rng)


@pytest.fixture
def crossover(rng):
    cfg = MutationConfig(crossover_strategy="two_point", crossover_rate=0.8)
    return CrossoverOperator(cfg, rng)


def dummy_fitness(sequences):
    """Fitness = fraction of 'M' characters — stable and deterministic."""
    return [seq.count("M") / len(seq) for seq in sequences]


# ---------------------------------------------------------------------------
# ConvergenceTracker
# ---------------------------------------------------------------------------


class TestConvergenceTracker:
    def test_no_convergence_with_steady_improvement(self):
        tracker = ConvergenceTracker(threshold=0.001, patience=3)
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        results = [tracker.update(s) for s in scores]
        assert not any(results)

    def test_convergence_triggered_after_patience(self):
        tracker = ConvergenceTracker(threshold=0.001, patience=3)
        tracker.update(0.9)     # improvement
        tracker.update(0.9)     # stale 1
        tracker.update(0.9)     # stale 2
        converged = tracker.update(0.9)   # stale 3 — triggers
        assert converged

    def test_reset_on_improvement(self):
        tracker = ConvergenceTracker(threshold=0.001, patience=3)
        tracker.update(0.5)
        tracker.update(0.5)     # stale 2
        tracker.update(0.6)     # improvement — resets
        assert tracker.stale_generations == 0


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------


class TestSelectors:
    def test_tournament_returns_correct_count(self, rng):
        sel = TournamentSelector(tournament_size=3, rng=rng)
        pop = [SEQ] * 20
        scores = list(range(20))
        selected = sel.select(pop, scores, n_select=10)
        assert len(selected) == 10

    def test_top_k_returns_best_sequences(self):
        sel = TopKSelector()
        pop = ["AAA", "CCC", "MMM"]
        scores = [0.1, 0.9, 0.5]
        selected = sel.select(pop, scores, n_select=2)
        assert selected[0] == "CCC"  # highest score

    def test_top_k_fills_to_n_select(self):
        sel = TopKSelector()
        pop = ["AAA", "CCC"]
        scores = [0.1, 0.9]
        selected = sel.select(pop, scores, n_select=5)
        assert len(selected) == 5


# ---------------------------------------------------------------------------
# GeneticAlgorithm
# ---------------------------------------------------------------------------


class TestGeneticAlgorithm:
    def _make_ga(self, mutator, crossover, max_gen=5, pop_size=10):
        cfg = GAConfig(
            population_size=pop_size,
            max_generations=max_gen,
            elite_fraction=0.2,
            selection_strategy="tournament",
            tournament_size=3,
            seed=42,
        )
        return GeneticAlgorithm(
            config=cfg,
            fitness_fn=dummy_fitness,
            mutator=mutator,
            crossover=crossover,
        )

    def test_run_returns_history(self, mutator, crossover):
        ga = self._make_ga(mutator, crossover)
        history = ga.run([SEQ] * 10)
        assert len(history) > 0
        assert all(isinstance(r, GenerationResult) for r in history)

    def test_population_size_maintained(self, mutator, crossover):
        ga = self._make_ga(mutator, crossover, pop_size=20)
        history = ga.run([SEQ] * 5)
        for result in history:
            assert len(result.population) == 20

    def test_best_score_monotonically_non_decreasing(self, mutator, crossover):
        ga = self._make_ga(mutator, crossover, max_gen=10)
        history = ga.run([SEQ])
        best_scores = [r.best_score for r in history]
        for i in range(1, len(best_scores)):
            assert best_scores[i] >= best_scores[i - 1] - 1e-9

    def test_best_sequence_set_after_run(self, mutator, crossover):
        ga = self._make_ga(mutator, crossover)
        ga.run([SEQ])
        assert ga.best_sequence is not None
        assert isinstance(ga.best_score, float)

    def test_callback_called_each_generation(self, mutator, crossover):
        calls = []
        ga = self._make_ga(mutator, crossover, max_gen=3)
        ga.callbacks.append(lambda r: calls.append(r.generation))
        ga.run([SEQ])
        assert calls == [0, 1, 2]

    def test_initial_population_padded_from_seed(self, mutator, crossover):
        ga = self._make_ga(mutator, crossover, pop_size=15)
        history = ga.run([SEQ])  # seed has only 1 sequence
        assert len(history[0].population) == 15


# ---------------------------------------------------------------------------
# Full pipeline integration (mock mode)
# ---------------------------------------------------------------------------


class TestPipelineMock:
    def _make_cfg(self, sequence=SEQ):
        cfg = OptimizationConfig()
        cfg.original_sequence = sequence
        cfg.bioemu.mock = True
        cfg.mutation.strategy = "random"
        cfg.mutation.max_mutations_per_sequence = 2
        cfg.ga.population_size = 10
        cfg.ga.max_generations = 3
        cfg.logging.output_dir = "/tmp/test_results"
        cfg.logging.export_format = "json"
        return cfg

    def test_pipeline_runs_end_to_end(self):
        pipeline = ProteinOptimizationPipeline(self._make_cfg())
        result = pipeline.run()
        assert result.best_sequence is not None
        assert 0.0 <= result.best_score <= 1.0
        assert result.generations_run <= 3

    def test_pipeline_rejects_empty_sequence(self):
        cfg = self._make_cfg(sequence="")
        with pytest.raises(ValueError, match="original_sequence must be set"):
            ProteinOptimizationPipeline(cfg)

    def test_pipeline_rejects_invalid_aa(self):
        cfg = self._make_cfg(sequence="MKTB!!X")
        with pytest.raises(ValueError, match="non-standard characters"):
            ProteinOptimizationPipeline(cfg)

    def test_result_best_score_is_float(self):
        pipeline = ProteinOptimizationPipeline(self._make_cfg())
        result = pipeline.run()
        assert isinstance(result.best_score, float)

    def test_tracker_score_trajectory_length(self):
        pipeline = ProteinOptimizationPipeline(self._make_cfg())
        result = pipeline.run()
        assert len(result.tracker.score_trajectory) == result.generations_run
