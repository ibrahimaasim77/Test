"""
protein_optimizer — Evolutionary protein sequence optimization framework.

Public API surface::

    from protein_optimizer import ProteinOptimizationPipeline, OptimizationConfig

    cfg = OptimizationConfig.from_yaml("config/default.yaml")
    cfg.original_sequence = "MKTLLILAVLCLGFAQAS"
    pipeline = ProteinOptimizationPipeline(cfg)
    result = pipeline.run()
"""

from .config import (
    BioEmuConfig,
    ESM2Config,
    GAConfig,
    LoggingConfig,
    MutationConfig,
    OptimizationConfig,
    ScoringConfig,
)
from .pipeline import OptimizationResult, ProteinOptimizationPipeline
from .scoring import ConformationalLandscapeScorer, ScoringFunction

__all__ = [
    "ProteinOptimizationPipeline",
    "OptimizationResult",
    "OptimizationConfig",
    "GAConfig",
    "ScoringConfig",
    "MutationConfig",
    "BioEmuConfig",
    "ESM2Config",
    "LoggingConfig",
    "ScoringFunction",
    "ConformationalLandscapeScorer",
]
