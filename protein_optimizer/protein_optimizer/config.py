"""
Unified configuration for the protein optimization framework.

All modules are driven exclusively by this config — no magic numbers in logic files.
Load from YAML with OptimizationConfig.from_yaml("config/default.yaml").
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Sub-configs (one per module)
# ---------------------------------------------------------------------------


@dataclass
class ESM2Config:
    """Controls ESM-2 mutation proposal behaviour."""

    model_name: str = "facebook/esm2_t33_650M_UR50D"
    device: str = "cuda"
    batch_size: int = 8
    top_k_candidates: int = 5
    temperature: float = 1.0
    cache_dir: Optional[str] = None


@dataclass
class BioEmuConfig:
    """Controls BioEmu structural inference."""

    # Model version: "bioemu-v1.0", "bioemu-v1.1" (default), or "bioemu-v1.2"
    # Also accepts a local checkpoint path.
    model_path: Optional[str] = "bioemu-v1.1"
    device: str = "cuda"
    batch_size: int = 10        # batch_size_100 parameter passed to bioemu.sample.main
    num_samples: int = 10       # ensemble size per sequence
    inference_steps: int = 50
    # Set mock=True to run with synthetic outputs (CI / no-GPU environments)
    mock: bool = False


@dataclass
class ScoringConfig:
    """Weights for the composite fitness function."""

    stability_weight: float = 0.4
    consistency_weight: float = 0.3
    energy_weight: float = 0.2
    diversity_penalty_weight: float = 0.1
    normalize: bool = True

    # When wildtype_sequence is set in OptimizationConfig, this weight is
    # automatically added and all other weights are renormalised to compensate.
    wildtype_proximity_weight: float = 0.3

    def __post_init__(self) -> None:
        total = (
            self.stability_weight
            + self.consistency_weight
            + self.energy_weight
            + self.diversity_penalty_weight
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total:.4f}. "
                "Adjust stability/consistency/energy/diversity_penalty weights."
            )


@dataclass
class MutationConfig:
    """Controls how sequences are mutated during evolution."""

    strategy: str = "esm_guided"          # "random" | "esm_guided"
    max_mutations_per_sequence: int = 3
    allowed_positions: Optional[List[int]] = None   # None = all positions
    mutation_rate: float = 0.1             # per-residue probability (random mode)
    crossover_strategy: str = "two_point"  # "single_point" | "two_point" | "uniform"
    crossover_rate: float = 0.7


@dataclass
class GAConfig:
    """Genetic algorithm hyper-parameters."""

    population_size: int = 50
    max_generations: int = 100
    elite_fraction: float = 0.1
    selection_strategy: str = "tournament"  # "top_k" | "tournament"
    tournament_size: int = 5
    convergence_threshold: float = 1e-4    # min improvement to reset patience
    convergence_patience: int = 10         # generations without improvement → stop
    seed: Optional[int] = 42
    n_stages: int = 5                      # number of progress checkpoints

    @property
    def elite_size(self) -> int:
        return max(1, int(self.population_size * self.elite_fraction))


@dataclass
class LoggingConfig:
    """Controls logging, checkpointing, and result export."""

    output_dir: str = "results"
    log_level: str = "INFO"
    save_every_n_generations: int = 10
    track_diversity: bool = True
    track_mutation_history: bool = True
    export_format: str = "both"   # "json" | "csv" | "both"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


@dataclass
class OptimizationConfig:
    """
    Root configuration object.

    Usage::

        cfg = OptimizationConfig.from_yaml("config/default.yaml")
        # or override programmatically
        cfg.ga.population_size = 100
    """

    experiment_name: str = "protein_opt"
    original_sequence: str = ""       # the "bad" starting sequence
    wildtype_sequence: str = ""       # the target to recover toward (optional)

    esm2: ESM2Config = field(default_factory=ESM2Config)
    bioemu: BioEmuConfig = field(default_factory=BioEmuConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OptimizationConfig":
        """Load from a YAML file. Missing keys fall back to dataclass defaults."""
        with open(path) as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "OptimizationConfig":
        cfg = cls()
        scalar_fields = {"experiment_name", "original_sequence", "wildtype_sequence"}
        sub_map = {
            "esm2": ESM2Config,
            "bioemu": BioEmuConfig,
            "scoring": ScoringConfig,
            "mutation": MutationConfig,
            "ga": GAConfig,
            "logging": LoggingConfig,
        }
        for key, value in data.items():
            if key in scalar_fields:
                setattr(cfg, key, value)
            elif key in sub_map:
                setattr(cfg, key, sub_map[key](**value))
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (useful for logging)."""
        import dataclasses
        return dataclasses.asdict(self)
