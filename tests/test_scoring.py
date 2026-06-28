"""Tests for scoring module."""

import numpy as np
import pytest

from protein_optimizer.bioemu import BioEmuConfig, BioEmuOutput, ConformationSample, MockBioEmuBackend
from protein_optimizer.config import ScoringConfig
from protein_optimizer.scoring import (
    CompactnessScorer,
    ComponentScorer,
    ConformationalLandscapeScorer,
    ConsistencyScorer,
    EnergyScorer,
    ScoringFunction,
    StabilityScorer,
)


SEQ = "MKTLLILAVLCLGFAQAS"


def make_landscape_output(sequence=SEQ, states=(5.0, 12.0, 5.0, 12.0)):
    """Build a small synthetic ensemble with distance-matrix states."""
    output = BioEmuOutput(sequence=sequence)
    for distance in states:
        matrix = np.full((6, 6), distance, dtype=float)
        np.fill_diagonal(matrix, 0.0)
        output.samples.append(ConformationSample(distance_matrix=matrix))
    return output


@pytest.fixture
def mock_output():
    """A realistic BioEmuOutput via the mock backend."""
    backend = MockBioEmuBackend(BioEmuConfig(mock=True, num_samples=5))
    outputs = backend.infer_batch([SEQ])
    return outputs[0]


@pytest.fixture
def empty_output():
    return BioEmuOutput(sequence=SEQ)


@pytest.fixture
def scoring_fn():
    return ScoringFunction(ScoringConfig())


class TestIndividualScorers:
    def test_stability_scorer_range(self, mock_output):
        scorer = StabilityScorer()
        s = scorer.score(mock_output)
        assert 0.0 <= s <= 1.0

    def test_stability_returns_neutral_on_missing(self, empty_output):
        scorer = StabilityScorer()
        assert scorer.score(empty_output) == 0.5

    def test_consistency_scorer_range(self, mock_output):
        scorer = ConsistencyScorer()
        s = scorer.score(mock_output)
        assert 0.0 <= s <= 1.0

    def test_energy_scorer_range(self, mock_output):
        scorer = EnergyScorer()
        s = scorer.score(mock_output)
        assert 0.0 <= s <= 1.0

    def test_compactness_scorer_range(self, mock_output):
        scorer = CompactnessScorer()
        s = scorer.score(mock_output)
        assert 0.0 <= s <= 1.0

    def test_very_stable_output_scores_high(self):
        """Manually construct a near-perfect output and verify high score."""
        output = BioEmuOutput(sequence=SEQ)
        L = len(SEQ)
        for _ in range(10):
            output.samples.append(
                ConformationSample(
                    per_residue_confidence=np.full(L, 95.0),
                    energy_proxy=-250.0,
                    radius_of_gyration=15.0,
                    distance_matrix=np.ones((L, L)) * 8.0,
                )
            )
        from protein_optimizer.bioemu import BaseStructuralBackend
        aggregated = BaseStructuralBackend._aggregate(output)

        scorer = StabilityScorer()
        assert scorer.score(aggregated) > 0.85


class TestScoringFunction:
    def test_score_returns_float_in_range(self, scoring_fn, mock_output):
        s = scoring_fn.score(mock_output)
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0

    def test_score_batch_returns_correct_length(self, scoring_fn):
        backend = MockBioEmuBackend(BioEmuConfig(mock=True, num_samples=3))
        sequences = ["MKTLLILAVLCL", "ACDEFGHIKLMN", "PQRSTVWYACDE"]
        outputs = backend.infer_batch(sequences)
        scores = scoring_fn.score_batch(outputs)
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_score_with_breakdown_has_fitness_key(self, scoring_fn, mock_output):
        breakdown = scoring_fn.score_with_breakdown(mock_output)
        assert "fitness" in breakdown
        assert "stability" in breakdown
        assert "consistency" in breakdown
        assert "energy" in breakdown

    def test_register_custom_scorer(self, mock_output):
        class DummyScorer(ComponentScorer):
            name = "dummy"
            def score(self, output):
                return 0.77

        fn = ScoringFunction(ScoringConfig())
        fn.register("dummy", DummyScorer, weight=0.1, renormalize=True)
        assert "dummy" in fn.component_names()
        # Weights must remain normalised
        assert abs(sum(fn._weights) - 1.0) < 1e-6

    def test_add_preconfigured_component_scorer(self):
        target = make_landscape_output()
        fn = ScoringFunction(ScoringConfig())
        fn.add_component(
            ConformationalLandscapeScorer(target, max_states=2),
            weight=0.5,
            renormalize=True,
        )
        assert "landscape" in fn.component_names()
        assert abs(sum(fn._weights) - 1.0) < 1e-6

    def test_invalid_weights_raise_on_config(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ScoringConfig(
                stability_weight=0.5,
                consistency_weight=0.5,
                energy_weight=0.5,
                diversity_penalty_weight=0.5,
            )

    def test_different_sequences_produce_different_scores(self, scoring_fn):
        backend = MockBioEmuBackend(BioEmuConfig(mock=True, num_samples=5))
        seqs = ["MKTLLILAVLCLGFAQAS", "ACDEFGHIKLMNPQRSTV"]
        outputs = backend.infer_batch(seqs)
        scores = scoring_fn.score_batch(outputs)
        # Mock backend hashes the sequence — scores should differ
        assert scores[0] != scores[1]


class TestConformationalLandscapeScorer:
    def test_identical_ensemble_scores_near_one(self):
        target = make_landscape_output()
        scorer = ConformationalLandscapeScorer(target, max_states=2)

        score = scorer.score(target)

        assert score > 0.99

    def test_off_target_ensemble_scores_low(self):
        target = make_landscape_output(states=(5.0, 12.0, 5.0, 12.0))
        candidate = make_landscape_output(states=(30.0, 30.0, 30.0, 30.0))
        scorer = ConformationalLandscapeScorer(
            target,
            max_states=2,
            rmsd_scale=2.0,
            outlier_threshold=4.0,
        )

        score = scorer.score(candidate)

        assert score < 0.2

    def test_target_distribution_includes_outlier_bin(self):
        target = make_landscape_output(states=(5.0, 12.0, 5.0, 12.0))
        scorer = ConformationalLandscapeScorer(target, max_states=2)

        distribution = scorer.target_distribution()

        assert len(distribution) == 3
        assert sum(distribution) == pytest.approx(1.0)
        assert distribution[-1] == 0.0

    def test_missing_distance_matrices_scores_zero(self, empty_output):
        target = make_landscape_output()
        scorer = ConformationalLandscapeScorer(target, max_states=2)

        assert scorer.score(empty_output) == 0.0
