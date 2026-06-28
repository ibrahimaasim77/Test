"""
Orchestration Layer — ProteinOptimizationPipeline

The single entry point that wires every module together and runs the GA loop.

Responsibility split:
  - Pipeline: builds all components, owns the fitness evaluation loop,
    coordinates BioEmu inference + scoring, feeds results to GA.
  - GeneticAlgorithm: knows nothing about biology; receives fitness scores.
  - OptimizationTracker: passive observer attached as a GA callback.

Usage::

    from protein_optimizer.config import OptimizationConfig
    from protein_optimizer.pipeline import ProteinOptimizationPipeline

    cfg = OptimizationConfig.from_yaml("config/default.yaml")
    pipeline = ProteinOptimizationPipeline(cfg)
    result = pipeline.run()
    print(result.best_sequence, result.best_score)
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .analysis import OptimizationTracker, StageReporter
from .bioemu import BaseStructuralBackend, build_bioemu_backend
from .config import OptimizationConfig
from .esm import ESM2MutationProposer
from .genetic_algorithm import GeneticAlgorithm
from .mutation import CrossoverOperator, build_mutator
from .scoring import ScoringFunction, WildtypeProximityScorer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Final result object
# ---------------------------------------------------------------------------


@dataclass
class OptimizationResult:
    """Returned by ProteinOptimizationPipeline.run()."""

    best_sequence: str
    best_score: float
    original_sequence: str
    generations_run: int
    total_wall_time_s: float
    export_paths: dict
    tracker: OptimizationTracker


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ProteinOptimizationPipeline:
    """
    Orchestrates the full protein sequence optimization run.

    Construction builds all sub-components lazily where possible:
      - ESM-2 is not loaded until the first mutation call
      - BioEmu is not loaded until the first fitness evaluation
      - Both respect GPU device settings from config

    Args:
        config: Root OptimizationConfig.
        scoring_fn: Optional custom ScoringFunction (default: built from config).
        bioemu_backend: Optional custom structural backend (default: from config).
    """

    def __init__(
        self,
        config: OptimizationConfig,
        scoring_fn: Optional[ScoringFunction] = None,
        bioemu_backend: Optional[BaseStructuralBackend] = None,
    ) -> None:
        self.config = config
        self._validate_config()

        self._rng = random.Random(config.ga.seed)

        # Structural backend (BioEmu or mock)
        self._bioemu: BaseStructuralBackend = (
            bioemu_backend or build_bioemu_backend(config.bioemu)
        )

        # Scoring function
        self._scorer: ScoringFunction = scoring_fn or ScoringFunction(config.scoring)

        # ESM-2 proposer (lazy — only built if strategy requires it)
        self._esm_proposer: Optional[ESM2MutationProposer] = None
        if config.mutation.strategy == "esm_guided":
            self._esm_proposer = ESM2MutationProposer(config.esm2)

        # Mutation operator
        self._mutator = build_mutator(
            config=config.mutation,
            rng=self._rng,
            esm_proposer=self._esm_proposer,
        )

        # Crossover operator
        self._crossover = CrossoverOperator(config.mutation, self._rng)

        # Tracker / logger
        self._tracker = OptimizationTracker(
            config=config.logging,
            original_sequence=config.original_sequence,
        )

        # Wildtype proximity scorer — injected into scoring fn when target is set
        self._stage_reporter: Optional[StageReporter] = None
        if config.wildtype_sequence:
            self._attach_wildtype_scorer()
            self._stage_reporter = StageReporter(
                wildtype_sequence=config.wildtype_sequence,
                ga_config=config.ga,
            )

        # GA (constructed in run() so callbacks are set once tracker is ready)
        self._ga: Optional[GeneticAlgorithm] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> OptimizationResult:
        """
        Execute the full optimization. Returns OptimizationResult.

        Steps:
          1. Build initial population from the original sequence
          2. Run GA loop (evaluate → select → crossover → mutate)
          3. Export results via tracker
        """
        self._configure_logging()
        logger.info(
            "Starting optimization: experiment=%s | seq_len=%d",
            self.config.experiment_name,
            len(self.config.original_sequence),
        )

        callbacks = [self._tracker.on_generation]
        if self._stage_reporter is not None:
            callbacks.append(self._stage_reporter.on_generation)

        self._ga = GeneticAlgorithm(
            config=self.config.ga,
            fitness_fn=self._evaluate_population,
            mutator=self._mutator,
            crossover=self._crossover,
            callbacks=callbacks,
        )

        start_time = time.time()
        initial_population = self._build_initial_population()
        self._ga.run(initial_population)
        elapsed = time.time() - start_time

        export_paths = self._tracker.export()

        if self._stage_reporter is not None:
            print(self._stage_reporter.stage_report())
            stage_path = self._stage_reporter.export_stages(
                self.config.logging.output_dir
            )
            export_paths["stages"] = stage_path

        print(self._tracker.summary_report())

        logger.info(
            "Optimization complete in %.1fs | best_score=%.4f",
            elapsed,
            self._tracker.best_score,
        )

        return OptimizationResult(
            best_sequence=self._tracker.best_sequence,
            best_score=self._tracker.best_score,
            original_sequence=self.config.original_sequence,
            generations_run=len(self._tracker.score_trajectory),
            total_wall_time_s=elapsed,
            export_paths=export_paths,
            tracker=self._tracker,
        )

    # ------------------------------------------------------------------
    # Fitness evaluation (called by GA each generation)
    # ------------------------------------------------------------------

    def _evaluate_population(self, sequences: List[str]) -> List[float]:
        """
        Full evaluation pipeline for one generation's population:
          sequences → BioEmu inference → ScoringFunction → fitness scores

        This is the only place BioEmu and scoring are called.
        The GA receives only the float scores.
        """
        logger.debug("Evaluating %d sequences via BioEmu...", len(sequences))
        bioemu_outputs = self._bioemu.infer_batch(sequences)
        scores = self._scorer.score_batch(bioemu_outputs)
        return scores

    # ------------------------------------------------------------------
    # Population initialisation
    # ------------------------------------------------------------------

    def _build_initial_population(self) -> List[str]:
        """
        Seed population:
          - The original sequence is always included (index 0)
          - Remaining slots filled with single-mutation variants
        """
        original = self.config.original_sequence
        pop_size = self.config.ga.population_size
        population = [original]

        while len(population) < pop_size:
            population.append(self._mutator.mutate(original))

        logger.info(
            "Initial population: %d sequences (1 original + %d mutants)",
            pop_size,
            pop_size - 1,
        )
        return population

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _attach_wildtype_scorer(self) -> None:
        """
        Directly append WildtypeProximityScorer into the scoring function and
        renormalise all weights so they still sum to 1.0.

        The wildtype_proximity_weight from ScoringConfig determines how much
        the GA is pulled toward the target vs. purely structural fitness.
        """
        wt_weight = self.config.scoring.wildtype_proximity_weight
        scorer = WildtypeProximityScorer(self.config.wildtype_sequence)

        self._scorer._components.append(scorer)
        self._scorer._weights.append(wt_weight)

        # Renormalise all weights
        total = sum(self._scorer._weights)
        self._scorer._weights = [w / total for w in self._scorer._weights]

        logger.info(
            "Wildtype proximity scorer active (weight=%.2f) | target: %s...",
            wt_weight,
            self.config.wildtype_sequence[:20],
        )

    def _validate_config(self) -> None:
        if not self.config.original_sequence:
            raise ValueError(
                "original_sequence must be set in config before running. "
                "Set cfg.original_sequence = 'MKTLL...'"
            )
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        invalid = set(self.config.original_sequence.upper()) - valid_aas
        if invalid:
            raise ValueError(
                f"original_sequence contains non-standard characters: {invalid}. "
                "Only the 20 canonical amino acids are supported."
            )

    def _configure_logging(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.config.logging.log_level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
