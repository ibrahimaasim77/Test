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
from typing import Dict, List, Optional, Tuple

from .config import GAConfig, LoggingConfig
from .genetic_algorithm import GenerationResult

logger = logging.getLogger(__name__)

# Warmth labels keyed by (lower_bound, upper_bound] sequence identity
_WARMTH_SCALE = [
    (0.00, 0.15, "FREEZING",   "❄❄❄❄❄"),
    (0.15, 0.30, "ICY",        "❄❄❄❄ "),
    (0.30, 0.45, "COLD",       "❄❄❄  "),
    (0.45, 0.60, "COOL",       "❄❄   "),
    (0.60, 0.72, "WARM",       "🔥   "),
    (0.72, 0.84, "HOT",        "🔥🔥  "),
    (0.84, 0.93, "VERY HOT",   "🔥🔥🔥 "),
    (0.93, 1.01, "SCORCHING",  "🔥🔥🔥🔥"),
]


def warmth_label(identity: float) -> tuple[str, str]:
    """Return (label, icon_bar) for a given sequence identity value."""
    for lo, hi, label, icons in _WARMTH_SCALE:
        if lo <= identity < hi:
            return label, icons
    return "SCORCHING", "🔥🔥🔥🔥"


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


# ---------------------------------------------------------------------------
# Stage-based warmth reporter
# ---------------------------------------------------------------------------


@dataclass
class StageSnapshot:
    """Warmth reading at the end of one stage."""

    stage: int           # 1-indexed (1 = first fifth, 5 = final fifth)
    gen_start: int
    gen_end: int
    best_sequence: str
    best_score: float
    sequence_identity: float    # vs wildtype (0–1)
    warmth_label: str
    warmth_icons: str
    mean_score: float
    diversity: float


class StageReporter:
    """
    Divides a GA run into N equal stages and reports how "warm" (close to
    wildtype) the best sequence is at the end of each stage.

    Attach as a GA callback::

        reporter = StageReporter(
            wildtype_sequence="MKTLL...",
            ga_config=cfg.ga,
        )
        ga = GeneticAlgorithm(..., callbacks=[tracker.on_generation, reporter.on_generation])

    After the run::

        print(reporter.stage_report())
        reporter.export_stages(output_dir)

    The warm/cold scale (sequence identity vs wildtype):
        0–15%   FREEZING  ❄❄❄❄❄
        15–30%  ICY       ❄❄❄❄
        30–45%  COLD      ❄❄❄
        45–60%  COOL      ❄❄
        60–72%  WARM      🔥
        72–84%  HOT       🔥🔥
        84–93%  VERY HOT  🔥🔥🔥
        93–100% SCORCHING 🔥🔥🔥🔥
    """

    def __init__(
        self,
        wildtype_sequence: str,
        ga_config: GAConfig,
    ) -> None:
        self.wildtype = wildtype_sequence
        self.n_stages = ga_config.n_stages
        self.max_generations = ga_config.max_generations

        # Compute which generation marks the end of each stage
        self._stage_boundaries: List[int] = self._compute_boundaries()
        self._stage_index: int = 0   # which stage we're currently in

        self.snapshots: List[StageSnapshot] = []
        self._gen_start_of_current_stage: int = 0

    def _compute_boundaries(self) -> List[int]:
        """
        Return the last generation (inclusive) for each stage.
        e.g. max_gen=100, n_stages=5 → [19, 39, 59, 79, 99]
        """
        size = self.max_generations / self.n_stages
        return [int((i + 1) * size) - 1 for i in range(self.n_stages)]

    def on_generation(self, result: GenerationResult) -> None:
        """GA callback — called after each generation is evaluated."""
        if self._stage_index >= self.n_stages:
            return

        boundary = self._stage_boundaries[self._stage_index]
        if result.generation >= boundary:
            identity = self._compute_identity(result.best_sequence)
            label, icons = warmth_label(identity)

            snap = StageSnapshot(
                stage=self._stage_index + 1,
                gen_start=self._gen_start_of_current_stage,
                gen_end=result.generation,
                best_sequence=result.best_sequence,
                best_score=result.best_score,
                sequence_identity=identity,
                warmth_label=label,
                warmth_icons=icons,
                mean_score=result.mean_score,
                diversity=result.diversity,
            )
            self.snapshots.append(snap)
            self._print_stage(snap)

            self._gen_start_of_current_stage = result.generation + 1
            self._stage_index += 1

    def _compute_identity(self, sequence: str) -> float:
        if not self.wildtype or not sequence:
            return 0.0
        length = min(len(sequence), len(self.wildtype))
        matches = sum(a == b for a, b in zip(sequence, self.wildtype))
        return matches / length if length > 0 else 0.0

    @staticmethod
    def _progress_bar(identity: float, width: int = 30) -> str:
        filled = round(identity * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {identity * 100:.1f}%"

    def _print_stage(self, snap: StageSnapshot) -> None:
        bar = self._progress_bar(snap.sequence_identity)
        print(
            f"\n  Stage {snap.stage}/{self.n_stages}  "
            f"(gen {snap.gen_start}–{snap.gen_end})  "
            f"{snap.warmth_icons}  {snap.warmth_label}\n"
            f"  Proximity to wildtype : {bar}\n"
            f"  Best fitness score    : {snap.best_score:.4f}  "
            f"(mean: {snap.mean_score:.4f})\n"
            f"  Best sequence         : {snap.best_sequence[:40]}...\n"
        )

    # ------------------------------------------------------------------

    def stage_report(self) -> str:
        """Full multi-stage summary as a printable string."""
        if not self.snapshots:
            return "No stage snapshots recorded yet."

        lines = [
            "",
            "=" * 64,
            "  WILDTYPE RECOVERY — Stage-by-Stage Warmth Report",
            "=" * 64,
            f"  Wildtype target : {self.wildtype[:40]}{'...' if len(self.wildtype) > 40 else ''}",
            "",
        ]

        for snap in self.snapshots:
            bar = self._progress_bar(snap.sequence_identity, width=24)
            lines += [
                f"  Stage {snap.stage}/{self.n_stages}  "
                f"Gen {snap.gen_start:3d}–{snap.gen_end:3d}  "
                f"{snap.warmth_icons}  {snap.warmth_label}",
                f"    {bar}",
                f"    Fitness: {snap.best_score:.4f}  |  "
                f"Diversity: {snap.diversity:.2f}",
                "",
            ]

        if self.snapshots:
            initial_id = self.snapshots[0].sequence_identity
            final_id = self.snapshots[-1].sequence_identity
            lines += [
                f"  Overall identity improvement: "
                f"{initial_id * 100:.1f}% → {final_id * 100:.1f}%",
            ]
        lines.append("=" * 64)
        return "\n".join(lines)

    def export_stages(self, output_dir: str | Path) -> Path:
        """Write stage snapshots to JSON."""
        path = Path(output_dir) / "stage_warmth_report.json"
        payload = {
            "wildtype_sequence": self.wildtype,
            "n_stages": self.n_stages,
            "stages": [asdict(s) for s in self.snapshots],
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        return path
