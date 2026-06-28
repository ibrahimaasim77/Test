"""
Tests for wildtype-guided recovery mode:
  - WildtypeProximityScorer
  - StageReporter warmth logic
  - Full pipeline with wildtype target
"""

import pytest

from protein_optimizer.analysis import StageReporter, warmth_label
from protein_optimizer.bioemu import BioEmuConfig, BioEmuOutput, MockBioEmuBackend
from protein_optimizer.config import GAConfig, OptimizationConfig
from protein_optimizer.pipeline import ProteinOptimizationPipeline
from protein_optimizer.scoring import WildtypeProximityScorer

WILDTYPE = "MKTLLILAVLCLGFAQAS"
BAD_SEQ  = "ACDEFGHIKLMNPQRST V"[:18]   # almost nothing in common


# ---------------------------------------------------------------------------
# WildtypeProximityScorer
# ---------------------------------------------------------------------------


class TestWildtypeProximityScorer:
    def _make_output(self, seq: str) -> BioEmuOutput:
        out = BioEmuOutput(sequence=seq)
        return out

    def test_identical_sequence_scores_one(self):
        scorer = WildtypeProximityScorer(WILDTYPE)
        output = self._make_output(WILDTYPE)
        assert scorer.score(output) == pytest.approx(1.0)

    def test_completely_different_scores_zero(self):
        scorer = WildtypeProximityScorer("A" * len(WILDTYPE))
        output = self._make_output("C" * len(WILDTYPE))
        assert scorer.score(output) == pytest.approx(0.0)

    def test_partial_match(self):
        # First half matches, second half doesn't
        half = len(WILDTYPE) // 2
        mixed = WILDTYPE[:half] + "A" * (len(WILDTYPE) - half)
        scorer = WildtypeProximityScorer(WILDTYPE)
        output = self._make_output(mixed)
        score = scorer.score(output)
        assert 0.0 < score < 1.0

    def test_empty_sequence_returns_zero(self):
        scorer = WildtypeProximityScorer(WILDTYPE)
        output = self._make_output("")
        assert scorer.score(output) == 0.0

    def test_hamming_distance(self):
        dist = WildtypeProximityScorer.hamming_distance("AAAA", "AACG")
        assert dist == 2

    def test_sequence_identity(self):
        identity = WildtypeProximityScorer.sequence_identity("AAAA", "AACG")
        assert identity == pytest.approx(0.5)

    def test_score_in_range(self):
        backend = MockBioEmuBackend(BioEmuConfig(mock=True, num_samples=3))
        scorer = WildtypeProximityScorer(WILDTYPE)
        outputs = backend.infer_batch([WILDTYPE, "ACDEFGHIKLMNPQRST"])
        for o in outputs:
            s = scorer.score(o)
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# warmth_label
# ---------------------------------------------------------------------------


class TestWarmthLabel:
    @pytest.mark.parametrize("identity,expected_label", [
        (0.05, "FREEZING"),
        (0.20, "ICY"),
        (0.35, "COLD"),
        (0.52, "COOL"),
        (0.65, "WARM"),
        (0.78, "HOT"),
        (0.88, "VERY HOT"),
        (0.97, "SCORCHING"),
        (1.00, "SCORCHING"),
    ])
    def test_label_at_identity(self, identity, expected_label):
        label, _ = warmth_label(identity)
        assert label == expected_label

    def test_returns_tuple_of_two_strings(self):
        result = warmth_label(0.5)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(x, str) for x in result)


# ---------------------------------------------------------------------------
# StageReporter
# ---------------------------------------------------------------------------


class TestStageReporter:
    def _make_ga_config(self, max_gen=10, n_stages=5):
        cfg = GAConfig(max_generations=max_gen, n_stages=n_stages, seed=0)
        return cfg

    def _make_result(self, generation, sequence, score=0.5):
        from protein_optimizer.genetic_algorithm import GenerationResult
        return GenerationResult(
            generation=generation,
            population=[sequence],
            scores=[score],
            best_sequence=sequence,
            best_score=score,
            mean_score=score,
            diversity=1.0,
        )

    def test_stage_boundaries_correct(self):
        reporter = StageReporter(WILDTYPE, self._make_ga_config(max_gen=10, n_stages=5))
        # With 10 gens / 5 stages: boundaries should be [1, 3, 5, 7, 9]
        assert len(reporter._stage_boundaries) == 5
        assert reporter._stage_boundaries[-1] == 9

    def test_snapshots_collected_per_stage(self):
        reporter = StageReporter(WILDTYPE, self._make_ga_config(max_gen=10, n_stages=5))
        for gen in range(10):
            reporter.on_generation(self._make_result(gen, WILDTYPE))
        assert len(reporter.snapshots) == 5

    def test_identical_sequence_identity_is_one(self):
        reporter = StageReporter(WILDTYPE, self._make_ga_config(max_gen=2, n_stages=1))
        reporter.on_generation(self._make_result(1, WILDTYPE))
        assert reporter.snapshots[0].sequence_identity == pytest.approx(1.0)
        assert reporter.snapshots[0].warmth_label == "SCORCHING"

    def test_stage_report_returns_string(self):
        reporter = StageReporter(WILDTYPE, self._make_ga_config(max_gen=4, n_stages=2))
        for gen in range(4):
            reporter.on_generation(self._make_result(gen, WILDTYPE))
        report = reporter.stage_report()
        assert isinstance(report, str)
        assert "Stage" in report
        assert "SCORCHING" in report

    def test_export_stages_writes_json(self, tmp_path):
        reporter = StageReporter(WILDTYPE, self._make_ga_config(max_gen=2, n_stages=1))
        reporter.on_generation(self._make_result(1, WILDTYPE))
        path = reporter.export_stages(tmp_path)
        assert path.exists()
        import json
        data = json.loads(path.read_text())
        assert data["wildtype_sequence"] == WILDTYPE
        assert len(data["stages"]) == 1


# ---------------------------------------------------------------------------
# Full pipeline — wildtype recovery mode
# ---------------------------------------------------------------------------


class TestWildtypePipeline:
    def _make_cfg(self):
        cfg = OptimizationConfig()
        cfg.original_sequence = "ACDEFGHIKLMNPQRSTVWY"  # nothing like wildtype
        cfg.wildtype_sequence = WILDTYPE
        cfg.bioemu.mock = True
        cfg.mutation.strategy = "random"
        cfg.mutation.max_mutations_per_sequence = 2
        cfg.ga.population_size = 10
        cfg.ga.max_generations = 10
        cfg.ga.n_stages = 5
        cfg.logging.output_dir = "/tmp/test_wildtype_results"
        cfg.logging.export_format = "json"
        return cfg

    def test_pipeline_runs_with_wildtype(self):
        pipeline = ProteinOptimizationPipeline(self._make_cfg())
        result = pipeline.run()
        assert result.best_sequence is not None
        assert 0.0 <= result.best_score <= 1.0

    def test_stage_reporter_attached_when_wildtype_set(self):
        pipeline = ProteinOptimizationPipeline(self._make_cfg())
        assert pipeline._stage_reporter is not None

    def test_no_stage_reporter_when_wildtype_empty(self):
        cfg = self._make_cfg()
        cfg.wildtype_sequence = ""
        pipeline = ProteinOptimizationPipeline(cfg)
        assert pipeline._stage_reporter is None

    def test_wildtype_scorer_in_scoring_fn(self):
        pipeline = ProteinOptimizationPipeline(self._make_cfg())
        assert "wildtype_proximity" in pipeline._scorer.component_names()

    def test_scoring_weights_still_normalised(self):
        pipeline = ProteinOptimizationPipeline(self._make_cfg())
        total = sum(pipeline._scorer._weights)
        assert abs(total - 1.0) < 1e-6

    def test_stages_populated_after_run(self):
        pipeline = ProteinOptimizationPipeline(self._make_cfg())
        pipeline.run()
        assert pipeline._stage_reporter is not None
        assert len(pipeline._stage_reporter.snapshots) > 0
