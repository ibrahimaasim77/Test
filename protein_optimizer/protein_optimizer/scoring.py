"""
Scoring Function Module

Converts BioEmuOutput objects into a single scalar fitness value used by the GA.

Design:
  - Four independent component scorers, each returning a value in [0, 1].
  - A weighted sum combines them into one fitness score.
  - Weights come from ScoringConfig — fully config-driven, no hard-coded constants.
  - All component scorers are pluggable: subclass ComponentScorer and register.

The GA only ever sees a float. It has no knowledge of BioEmu, ESM-2, or biology.

Scoring components
------------------
1. StabilityScorer       — mean pLDDT confidence across ensemble (higher = better)
2. ConsistencyScorer     — inverse variance of structural ensemble (lower var = better)
3. EnergyScorer          — normalised mean energy proxy (lower energy = better)
4. CompactnessScorer     — radius of gyration stability (tighter fold = better)
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type

from .bioemu import BioEmuOutput
from .config import ScoringConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Component scorer base
# ---------------------------------------------------------------------------


class ComponentScorer(ABC):
    """
    A single scoring axis. Returns a float in [0, 1].

    Subclass and implement `score`. Register via ScoringFunction.register().
    """

    name: str = "base"

    @abstractmethod
    def score(self, output: BioEmuOutput) -> float:
        """Compute a score in [0, 1] from a BioEmuOutput. Higher = better."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Built-in component scorers
# ---------------------------------------------------------------------------


class StabilityScorer(ComponentScorer):
    """
    Structural stability score from per-residue confidence (pLDDT proxy).

    pLDDT values range 0–100. We normalise to [0, 1].
    Low confidence_std across the ensemble further boosts the score.
    """

    name = "stability"

    def score(self, output: BioEmuOutput) -> float:
        if output.mean_confidence is None:
            return 0.5  # neutral default when data is missing

        normalised_confidence = output.mean_confidence / 100.0   # [0, 1]

        # Penalise high variance across ensemble members
        if output.confidence_std is not None and output.confidence_std > 0:
            variance_penalty = min(output.confidence_std / 20.0, 0.3)
            return max(0.0, normalised_confidence - variance_penalty)

        return float(normalised_confidence)


class ConsistencyScorer(ComponentScorer):
    """
    Conformational consistency: how similar are structures across the ensemble?

    Uses pairwise distance variance: lower variance → more consistent → higher score.
    We map variance to [0, 1] with an exponential decay.
    """

    name = "consistency"

    _DECAY_RATE: float = 0.005   # controls how fast score drops with variance

    def score(self, output: BioEmuOutput) -> float:
        if output.pairwise_distance_variance is None:
            return 0.5

        # Exponential decay: score → 1 as variance → 0
        return float(math.exp(-self._DECAY_RATE * output.pairwise_distance_variance))


class EnergyScorer(ComponentScorer):
    """
    Energy-based stability proxy.

    Assumes energy_proxy is negative (lower = more stable). We compute a
    normalised score within the range typically seen for protein energies.
    Uses a soft sigmoid mapping to keep the score well-behaved for outliers.

    Reference range: -300 (very stable) to +50 (unstable).
    """

    name = "energy"

    _E_MIN: float = -300.0
    _E_MAX: float = 50.0

    def score(self, output: BioEmuOutput) -> float:
        if output.mean_energy is None:
            return 0.5

        # Penalise high energy variance (unstable ensemble)
        energy_std_penalty = 0.0
        if output.energy_std is not None:
            energy_std_penalty = min(output.energy_std / 100.0, 0.2)

        # Linear normalisation clamped to [0, 1]: lower energy = higher score
        span = self._E_MAX - self._E_MIN
        raw = (self._E_MAX - output.mean_energy) / span
        return float(max(0.0, min(1.0, raw - energy_std_penalty)))


class CompactnessScorer(ComponentScorer):
    """
    Radius of gyration stability proxy.

    A compact, consistent fold (low Rg with low std) scores higher.
    Very large or very variable Rg suggests unfolded or disordered regions.

    Expected range: 10–30 Å for typical globular proteins.
    """

    name = "compactness"

    _RG_IDEAL: float = 15.0   # Å, ideal compact globular protein
    _RG_SIGMA: float = 8.0    # tolerance

    def score(self, output: BioEmuOutput) -> float:
        if output.mean_rg is None:
            return 0.5

        # Gaussian-shaped reward around ideal Rg
        diff = output.mean_rg - self._RG_IDEAL
        compactness_score = math.exp(-(diff ** 2) / (2 * self._RG_SIGMA ** 2))

        # Penalise high Rg variance (conformational disorder)
        if output.rg_std is not None:
            std_penalty = min(output.rg_std / 10.0, 0.3)
            return float(max(0.0, compactness_score - std_penalty))

        return float(compactness_score)


# ---------------------------------------------------------------------------
# Composite scoring function
# ---------------------------------------------------------------------------


class ScoringFunction:
    """
    Weighted combination of component scorers → single fitness scalar.

    Usage::

        fn = ScoringFunction(cfg.scoring)
        scores = fn.score_batch(bioemu_outputs)   # List[float]

    Extension::

        class MySASAScorer(ComponentScorer):
            name = "sasa"
            def score(self, output): ...

        fn.register("sasa", MySASAScorer, weight=0.15)
        # Existing weights are automatically renormalised.
    """

    # Default component registry
    _BUILTIN: Dict[str, Type[ComponentScorer]] = {
        "stability": StabilityScorer,
        "consistency": ConsistencyScorer,
        "energy": EnergyScorer,
        "compactness": CompactnessScorer,
    }

    def __init__(self, config: ScoringConfig) -> None:
        self.config = config
        self._components: List[ComponentScorer] = []
        self._weights: List[float] = []
        self._build_default_components()

    # ------------------------------------------------------------------

    def _build_default_components(self) -> None:
        weight_map = {
            "stability": self.config.stability_weight,
            "consistency": self.config.consistency_weight,
            "energy": self.config.energy_weight,
            "compactness": self.config.diversity_penalty_weight,
        }
        for name, cls in self._BUILTIN.items():
            self._components.append(cls())
            self._weights.append(weight_map[name])

    # ------------------------------------------------------------------

    def score(self, output: BioEmuOutput) -> float:
        """Score a single BioEmuOutput. Returns fitness in [0, 1]."""
        component_scores = [c.score(output) for c in self._components]

        if self.config.normalize:
            total_weight = sum(self._weights)
            if total_weight == 0:
                return 0.0
            weights = [w / total_weight for w in self._weights]
        else:
            weights = self._weights

        fitness = sum(w * s for w, s in zip(weights, component_scores))

        logger.debug(
            "Scored sequence [len=%d]: fitness=%.4f | components=%s",
            len(output.sequence),
            fitness,
            {c.name: f"{s:.3f}" for c, s in zip(self._components, component_scores)},
        )
        return float(fitness)

    def score_batch(self, outputs: List[BioEmuOutput]) -> List[float]:
        """Score a list of BioEmuOutputs. Returns List[float] in same order."""
        return [self.score(o) for o in outputs]

    def score_with_breakdown(self, output: BioEmuOutput) -> Dict[str, float]:
        """
        Return both the aggregate fitness and per-component scores.
        Useful for analysis and debugging.
        """
        breakdown = {c.name: c.score(output) for c in self._components}
        breakdown["fitness"] = self.score(output)
        return breakdown

    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        scorer_cls: Type[ComponentScorer],
        weight: float,
        renormalize: bool = True,
    ) -> None:
        """
        Add a custom component scorer and optionally renormalize all weights
        so they sum to 1.
        """
        instance = scorer_cls()
        self._components.append(instance)
        self._weights.append(weight)

        if renormalize:
            total = sum(self._weights)
            self._weights = [w / total for w in self._weights]

        logger.info("Registered scorer '%s' with weight %.3f", name, weight)

    def component_names(self) -> List[str]:
        return [c.name for c in self._components]
