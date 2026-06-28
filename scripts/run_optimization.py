"""
Example usage script — demonstrates the full Python API
without going through the CLI.

Run from the repo root:
    python scripts/run_optimization.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from protein_optimizer import (
    ConformationalLandscapeScorer,
    GAConfig,
    MutationConfig,
    OptimizationConfig,
    ProteinOptimizationPipeline,
    ScoringConfig,
)
from protein_optimizer.scoring import ComponentScorer, ScoringFunction
from protein_optimizer.bioemu import BioEmuOutput


# ---------------------------------------------------------------------------
# 1. Minimal quick-start (mock mode, random mutations — no GPU needed)
# ---------------------------------------------------------------------------

def quick_start() -> None:
    print("=" * 50)
    print("Quick-start: mock BioEmu + random mutations")
    print("=" * 50)

    cfg = OptimizationConfig()
    cfg.original_sequence = "MKTLLILAVLCLGFAQASGNIERPIDGFHGDLQ"
    cfg.bioemu.mock = True
    cfg.mutation.strategy = "random"
    cfg.ga.population_size = 20
    cfg.ga.max_generations = 10
    cfg.logging.output_dir = "results/quick_start"
    cfg.logging.export_format = "json"

    pipeline = ProteinOptimizationPipeline(cfg)
    result = pipeline.run()

    print(f"\nBest: {result.best_sequence[:30]}... | score={result.best_score:.4f}")


# ---------------------------------------------------------------------------
# 2. Full config from YAML
# ---------------------------------------------------------------------------

def from_yaml_config() -> None:
    print("\n" + "=" * 50)
    print("From YAML config")
    print("=" * 50)

    cfg = OptimizationConfig.from_yaml("config/default.yaml")
    cfg.bioemu.mock = True          # Remove for real GPU run
    cfg.mutation.strategy = "random"
    cfg.ga.max_generations = 5

    pipeline = ProteinOptimizationPipeline(cfg)
    result = pipeline.run()
    print(result.tracker.summary_report())


# ---------------------------------------------------------------------------
# 3. Custom scoring component — demonstrates extension point
# ---------------------------------------------------------------------------

class SASAScorer(ComponentScorer):
    """
    Custom component: penalises sequences with very high average SASA
    (over-exposed residues suggest an unstable fold).
    """

    name = "sasa"

    def score(self, output: BioEmuOutput) -> float:
        import numpy as np
        sasa_values = []
        for sample in output.samples:
            if sample.sasa is not None:
                sasa_values.append(float(np.mean(sample.sasa)))
        if not sasa_values:
            return 0.5
        mean_sasa = sum(sasa_values) / len(sasa_values)
        # Penalise mean SASA > 100 Å²/residue
        return float(max(0.0, 1.0 - (mean_sasa / 200.0)))


def custom_scorer_demo() -> None:
    print("\n" + "=" * 50)
    print("Custom ScoringFunction with SASA component")
    print("=" * 50)

    cfg = OptimizationConfig()
    cfg.original_sequence = "MKTLLILAVLCLGFAQASGNIERPIDGFHGDLQ"
    cfg.bioemu.mock = True
    cfg.mutation.strategy = "random"
    cfg.ga.population_size = 10
    cfg.ga.max_generations = 5
    cfg.logging.output_dir = "results/custom_scorer"

    # Build a custom scoring function and register the SASA scorer
    # with 0.15 weight (existing weights auto-renormalized)
    scoring_fn = ScoringFunction(cfg.scoring)
    scoring_fn.register("sasa", SASAScorer, weight=0.15, renormalize=True)

    print(f"Active scoring components: {scoring_fn.component_names()}")

    pipeline = ProteinOptimizationPipeline(cfg, scoring_fn=scoring_fn)
    result = pipeline.run()
    print(f"Best score: {result.best_score:.4f}")


# ---------------------------------------------------------------------------
# 4. Score breakdown — inspect individual component contributions
# ---------------------------------------------------------------------------

def score_breakdown_demo() -> None:
    print("\n" + "=" * 50)
    print("Score breakdown for a single sequence")
    print("=" * 50)

    from protein_optimizer.bioemu import MockBioEmuBackend, BioEmuConfig
    from protein_optimizer.scoring import ScoringFunction
    from protein_optimizer.config import ScoringConfig

    backend = MockBioEmuBackend(BioEmuConfig(mock=True, num_samples=5))
    scorer = ScoringFunction(ScoringConfig())

    sequence = "MKTLLILAVLCLGFAQASGNIERPIDGFHGDLQ"
    outputs = backend.infer_batch([sequence])
    breakdown = scorer.score_with_breakdown(outputs[0])

    for component, value in breakdown.items():
        print(f"  {component:20s}: {value:.4f}")


# ---------------------------------------------------------------------------
# 5. Target conformational landscape scorer
# ---------------------------------------------------------------------------

def landscape_scorer_demo() -> None:
    print("\n" + "=" * 50)
    print("Target conformational landscape scorer")
    print("=" * 50)

    from protein_optimizer.bioemu import BioEmuConfig, MockBioEmuBackend

    cfg = OptimizationConfig()
    cfg.original_sequence = "MKTLLILAVLCLGFAQASGNIERPIDGFHGDLQ"
    cfg.bioemu.mock = True
    cfg.mutation.strategy = "random"
    cfg.ga.population_size = 10
    cfg.ga.max_generations = 5
    cfg.logging.output_dir = "results/landscape_scorer"

    # In a planted-mutation proof of concept, this target would be generated
    # from the hidden mutant sequence, then the GA tries to recover it.
    hidden_target_sequence = "MKTLLILAVLCLGFAQASGNIERPIDGFHFDLQ"
    backend = MockBioEmuBackend(BioEmuConfig(mock=True, num_samples=8))
    target_output = backend.infer_batch([hidden_target_sequence])[0]

    scoring_fn = ScoringFunction(cfg.scoring)
    scoring_fn.add_component(
        ConformationalLandscapeScorer(target_output, max_states=4),
        weight=0.6,
        renormalize=True,
    )

    print(f"Active scoring components: {scoring_fn.component_names()}")

    pipeline = ProteinOptimizationPipeline(cfg, scoring_fn=scoring_fn)
    result = pipeline.run()
    print(f"Best score: {result.best_score:.4f}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    quick_start()
    from_yaml_config()
    custom_scorer_demo()
    score_breakdown_demo()
    landscape_scorer_demo()
