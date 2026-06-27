"""
Logging and Analysis Module

Responsibilities:
  - Attach to GA as a callback to record generation statistics
  - Track best sequence, score trajectory, and population diversity
  - Optionally record per-sequence mutation history
  - Export results to JSON and/or CSV

Usage::

    tracker = OptimizationTracker(cfg.logging)
    ga = GeneticAlgorithm(..., callbacks=[tracker.on_generation])
    ga.run(population)
    tracker.export()
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .config import LoggingConfig
from .genetic_algorithm import GenerationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-generation summary record
# ---------------------------------------------------------------------------


@dataclass
class GenerationSummary:
    """Serialisable summary of one generation (subset of GenerationResult)."""

    generation: int
    best_score: float
    mean_score: float
    worst_score: float
    diversity: float
    best_sequence: str
    population_size: int


# ---------------------------------------------------------------------------
# Mutation history entry
# ---------------------------------------------------------------------------


@dataclass
class MutationRecord:
    """Records what changed between parent and child sequences."""

    generation: int
    parent_sequence: str
    child_sequence: str
    mutations: List[Dict]   # [{"position": int, "from": str, "to": str}]

    @classmethod
    def compute(
        cls,
        generation: int,
        parent: str,
        child: str,
    ) -> "MutationRecord":
        diffs = [
            {"position": i, "from": a, "to": b}
            for i, (a, b) in enumerate(zip(parent, child))
            if a != b
        ]
        return cls(
            generation=generation,
            parent_sequence=parent,
            child_sequence=child,
            mutations=diffs,
        )


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------


class OptimizationTracker:
    """
    Collects GA progress data and exports to disk.

    Attach to a GeneticAlgorithm via callbacks::

        ga = GeneticAlgorithm(..., callbacks=[tracker.on_generation])

    Then after the run::

        tracker.export()
    """

    def __init__(self, config: LoggingConfig, original_sequence: str = "") -> None:
        self.config = config
        self.original_sequence = original_sequence

        self._summaries: List[GenerationSummary] = []
        self._mutation_records: List[MutationRecord] = []
        self._global_best_score: float = float("-inf")
        self._global_best_sequence: str = ""
        self._global_best_generation: int = -1

        self._output_dir = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # GA callback
    # ------------------------------------------------------------------

    def on_generation(self, result: GenerationResult) -> None:
        """Called by GeneticAlgorithm after each generation is evaluated."""
        summary = GenerationSummary(
            generation=result.generation,
            best_score=result.best_score,
            mean_score=result.mean_score,
            worst_score=min(result.scores),
            diversity=result.diversity,
            best_sequence=result.best_sequence,
            population_size=len(result.population),
        )
        self._summaries.append(summary)

        if result.best_score > self._global_best_score:
            self._global_best_score = result.best_score
            self._global_best_sequence = result.best_sequence
            self._global_best_generation = result.generation
            logger.info(
                "New best at generation %d: %.4f | %s...",
                result.generation,
                result.best_score,
                result.best_sequence[:20],
            )

        # Record mutations vs. original if tracking enabled
        if self.config.track_mutation_history and self.original_sequence:
            record = MutationRecord.compute(
                generation=result.generation,
                parent=self.original_sequence,
                child=result.best_sequence,
            )
            self._mutation_records.append(record)

        # Periodic checkpoint save
        if (
            result.generation > 0
            and result.generation % self.config.save_every_n_generations == 0
        ):
            self._save_checkpoint(result.generation)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self) -> Dict[str, Path]:
        """
        Write final results to disk.

        Returns:
            dict mapping format name to output file path.
        """
        paths: Dict[str, Path] = {}

        if self.config.export_format in ("json", "both"):
            paths["json"] = self._export_json()
        if self.config.export_format in ("csv", "both"):
            paths["csv"] = self._export_csv()

        logger.info("Results exported to: %s", self._output_dir)
        return paths

    def _export_json(self) -> Path:
        path = self._output_dir / "optimization_results.json"
        payload = {
            "global_best": {
                "sequence": self._global_best_sequence,
                "score": self._global_best_score,
                "generation": self._global_best_generation,
            },
            "generations": [asdict(s) for s in self._summaries],
            "mutation_history": [asdict(r) for r in self._mutation_records],
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        return path

    def _export_csv(self) -> Path:
        path = self._output_dir / "generation_summary.csv"
        if not self._summaries:
            return path
        fieldnames = list(asdict(self._summaries[0]).keys())
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for summary in self._summaries:
                writer.writerow(asdict(summary))
        return path

    def _save_checkpoint(self, generation: int) -> None:
        path = self._output_dir / f"checkpoint_gen_{generation:04d}.json"
        payload = {
            "generation": generation,
            "best_sequence": self._global_best_sequence,
            "best_score": self._global_best_score,
            "summaries_so_far": [asdict(s) for s in self._summaries],
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        logger.debug("Checkpoint saved: %s", path)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def best_sequence(self) -> str:
        return self._global_best_sequence

    @property
    def best_score(self) -> float:
        return self._global_best_score

    @property
    def score_trajectory(self) -> List[float]:
        """Best-score per generation (useful for plotting)."""
        return [s.best_score for s in self._summaries]

    @property
    def mean_trajectory(self) -> List[float]:
        return [s.mean_score for s in self._summaries]

    @property
    def diversity_trajectory(self) -> List[float]:
        return [s.diversity for s in self._summaries]

    def summary_report(self) -> str:
        """Return a human-readable one-page report string."""
        lines = [
            "=" * 60,
            "Protein Optimization — Run Summary",
            "=" * 60,
            f"Generations completed : {len(self._summaries)}",
            f"Best score            : {self._global_best_score:.4f}",
            f"Best found at gen     : {self._global_best_generation}",
            f"Best sequence         : {self._global_best_sequence[:40]}...",
        ]
        if self._summaries:
            initial = self._summaries[0].best_score
            final = self._summaries[-1].best_score
            lines.append(f"Score improvement     : {initial:.4f} → {final:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)
