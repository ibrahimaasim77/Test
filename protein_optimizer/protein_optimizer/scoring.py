"""
Scoring Function Module

Converts BioEmuOutput objects into a single scalar fitness value used by the GA.

Design:
  - Independent component scorers, each returning a value in [0, 1].
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
5. ConformationalLandscapeScorer — ensemble distribution similarity to a target
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Type

import numpy as np

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


class ConformationalLandscapeScorer(ComponentScorer):
    """
    Compare a candidate ensemble against a target conformational landscape.

    The scorer turns each BioEmu distance matrix into a structural fingerprint,
    clusters the target ensemble into a small set of reference states, then
    compares target vs candidate occupancy probabilities over those states.

    Final score = weighted blend of:
      - state-distribution similarity via Jensen-Shannon similarity
      - mean structural proximity to the nearest target state

    This is intentionally model-agnostic: it only needs distance matrices from
    BioEmuOutput.samples, so the same scorer works for real BioEmu output and
    for the mock backend used in tests.
    """

    name = "landscape"

    def __init__(
        self,
        target_output: BioEmuOutput,
        max_states: int = 5,
        outlier_threshold: Optional[float] = None,
        rmsd_scale: float = 2.0,
        distribution_weight: float = 0.7,
        structural_weight: float = 0.3,
        smoothing: float = 1e-8,
    ) -> None:
        if max_states < 1:
            raise ValueError("max_states must be >= 1")
        if rmsd_scale <= 0:
            raise ValueError("rmsd_scale must be > 0")
        if distribution_weight < 0 or structural_weight < 0:
            raise ValueError("landscape scorer weights must be non-negative")

        total_weight = distribution_weight + structural_weight
        if total_weight == 0:
            raise ValueError("at least one landscape scorer weight must be positive")

        self.target_output = target_output
        self.max_states = max_states
        self.outlier_threshold = outlier_threshold or (2.0 * rmsd_scale)
        self.rmsd_scale = rmsd_scale
        self.distribution_weight = distribution_weight / total_weight
        self.structural_weight = structural_weight / total_weight
        self.smoothing = smoothing

        self._target_features = self._extract_features(target_output)
        self._prototypes = self._select_target_prototypes(self._target_features)
        self._target_distribution = self._build_target_distribution()

    def score(self, output: BioEmuOutput) -> float:
        if self._prototypes.size == 0:
            return 0.0

        candidate_features = self._extract_features(output)
        if candidate_features.size == 0:
            return 0.0
        if candidate_features.shape[1] != self._prototypes.shape[1]:
            logger.warning(
                "Landscape scorer dimension mismatch: target=%d candidate=%d",
                self._prototypes.shape[1],
                candidate_features.shape[1],
            )
            return 0.0

        min_distances, assignments = self._assign_to_prototypes(candidate_features)
        candidate_distribution = self._state_distribution(assignments, min_distances)

        js_distance = _jensen_shannon_divergence(
            self._target_distribution,
            candidate_distribution,
            smoothing=self.smoothing,
        )
        distribution_score = _clamp01(1.0 - (js_distance / math.log(2.0)))

        mean_distance = float(np.mean(min_distances))
        structural_score = _clamp01(math.exp(-mean_distance / self.rmsd_scale))

        return float(
            self.distribution_weight * distribution_score
            + self.structural_weight * structural_score
        )

    def target_distribution(self) -> List[float]:
        """Return the target occupancy distribution, including the outlier bin."""
        return self._target_distribution.tolist()

    def _build_target_distribution(self) -> np.ndarray:
        if self._prototypes.size == 0:
            return np.array([], dtype=float)

        min_distances, assignments = self._assign_to_prototypes(self._target_features)
        return self._state_distribution(assignments, min_distances, allow_outliers=False)

    def _state_distribution(
        self,
        assignments: np.ndarray,
        min_distances: np.ndarray,
        allow_outliers: bool = True,
    ) -> np.ndarray:
        n_states = len(self._prototypes)
        counts = np.zeros(n_states + 1, dtype=float)  # final bin = off-target state

        for state_idx, distance in zip(assignments, min_distances):
            if allow_outliers and distance > self.outlier_threshold:
                counts[-1] += 1.0
            else:
                counts[int(state_idx)] += 1.0

        total = float(np.sum(counts))
        if total == 0:
            return np.full(n_states + 1, 1.0 / (n_states + 1))
        return counts / total

    @staticmethod
    def _extract_features(output: BioEmuOutput) -> np.ndarray:
        features: List[np.ndarray] = []
        for sample in output.samples:
            if sample.distance_matrix is None:
                continue
            matrix = np.asarray(sample.distance_matrix, dtype=float)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                continue
            upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
            if upper.size > 0:
                features.append(upper)

        if not features:
            return np.empty((0, 0), dtype=float)

        first_dim = features[0].shape[0]
        compatible = [feature for feature in features if feature.shape[0] == first_dim]
        return np.vstack(compatible) if compatible else np.empty((0, 0), dtype=float)

    def _select_target_prototypes(self, target_features: np.ndarray) -> np.ndarray:
        if target_features.size == 0:
            return np.empty((0, 0), dtype=float)

        n_samples = target_features.shape[0]
        n_states = min(self.max_states, n_samples)
        selected = [0]

        while len(selected) < n_states:
            selected_features = target_features[selected]
            distances = _pairwise_drmsd(target_features, selected_features)
            min_distances = np.min(distances, axis=1)
            next_idx = int(np.argmax(min_distances))
            if next_idx in selected:
                break
            selected.append(next_idx)

        return target_features[selected]

    def _assign_to_prototypes(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distances = _pairwise_drmsd(features, self._prototypes)
        assignments = np.argmin(distances, axis=1)
        min_distances = distances[np.arange(distances.shape[0]), assignments]
        return min_distances, assignments


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
        fn.add_component(
            ConformationalLandscapeScorer(target_output),
            weight=0.5,
        )
        # Preconfigured scorers can hold target ensembles or other state.
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

    def add_component(
        self,
        scorer: ComponentScorer,
        weight: float,
        renormalize: bool = True,
    ) -> None:
        """
        Add a preconfigured component scorer instance.

        Use this for scorers that require constructor data, such as
        ConformationalLandscapeScorer(target_output=...).
        """
        self._components.append(scorer)
        self._weights.append(weight)

        if renormalize:
            total = sum(self._weights)
            self._weights = [w / total for w in self._weights]

        logger.info("Added scorer '%s' with weight %.3f", scorer.name, weight)

    def component_names(self) -> List[str]:
        return [c.name for c in self._components]


def _pairwise_drmsd(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pairwise distance RMSD between rows of two feature matrices."""
    deltas = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.mean(deltas * deltas, axis=2))


def _jensen_shannon_divergence(
    p: Sequence[float],
    q: Sequence[float],
    smoothing: float = 1e-8,
) -> float:
    """Jensen-Shannon divergence using natural logs; range is [0, ln(2)]."""
    p_arr = np.asarray(p, dtype=float) + smoothing
    q_arr = np.asarray(q, dtype=float) + smoothing
    p_arr = p_arr / np.sum(p_arr)
    q_arr = q_arr / np.sum(q_arr)
    midpoint = 0.5 * (p_arr + q_arr)
    return float(
        0.5 * np.sum(p_arr * np.log(p_arr / midpoint))
        + 0.5 * np.sum(q_arr * np.log(q_arr / midpoint))
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
