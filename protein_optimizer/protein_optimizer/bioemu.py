"""
BioEmu Interface Module

Abstracts all structural inference behind a single contract: given a list
of protein sequences, return a list of BioEmuOutput objects.

Design principles:
  - BioEmu is treated as a black-box feature extractor, never a scorer.
  - The concrete BioEmuWrapper can be swapped for any other structural model
    (AlphaFold, RoseTTAFold, etc.) by subclassing BaseStructuralBackend.
  - MockBioEmuBackend provides deterministic synthetic outputs for CI / GPU-free
    development — toggle with BioEmuConfig(mock=True).
  - Batch inference is the default and only public interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import BioEmuConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------


@dataclass
class ConformationSample:
    """
    A single structural sample from BioEmu's ensemble.

    Fields reflect what BioEmu typically provides; absent values are None.
    The scoring module consumes these fields — it never calls BioEmu directly.
    """

    # Per-residue pLDDT-style confidence (0–100), shape (L,)
    per_residue_confidence: Optional[np.ndarray] = None

    # Predicted energy proxy (lower = more stable). Scalar.
    energy_proxy: Optional[float] = None

    # Pairwise distance matrix, shape (L, L). Used for consistency scoring.
    distance_matrix: Optional[np.ndarray] = None

    # Radius of gyration. Proxy for compactness / fold integrity.
    radius_of_gyration: Optional[float] = None

    # Solvent accessible surface area per residue, shape (L,)
    sasa: Optional[np.ndarray] = None


@dataclass
class BioEmuOutput:
    """
    Aggregated structural output for one protein sequence.

    `samples` holds the full ensemble; summary statistics are pre-computed
    here so the scoring module can work without looping over samples itself.
    """

    sequence: str
    samples: List[ConformationSample] = field(default_factory=list)

    # Pre-aggregated across ensemble (populated by BioEmuWrapper.infer)
    mean_confidence: Optional[float] = None          # mean pLDDT across residues & samples
    confidence_std: Optional[float] = None           # std of per-sample mean pLDDT
    mean_energy: Optional[float] = None
    energy_std: Optional[float] = None
    mean_rg: Optional[float] = None
    rg_std: Optional[float] = None
    pairwise_distance_variance: Optional[float] = None   # structural consistency proxy

    @property
    def ensemble_size(self) -> int:
        return len(self.samples)


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class BaseStructuralBackend(ABC):
    """
    Minimal contract that any structural inference engine must satisfy.

    Implement `_run_inference` to plug in a new model without touching
    the GA, scoring, or pipeline layers.
    """

    def __init__(self, config: BioEmuConfig) -> None:
        self.config = config

    def infer_batch(self, sequences: List[str]) -> List[BioEmuOutput]:
        """
        Run inference on a batch of sequences and return aggregated outputs.

        This method handles batching and aggregation; subclasses only need to
        implement `_run_inference` for a single sequence.
        """
        outputs: List[BioEmuOutput] = []
        for batch_start in range(0, len(sequences), self.config.batch_size):
            batch = sequences[batch_start : batch_start + self.config.batch_size]
            logger.debug(
                "BioEmu inference: batch %d–%d of %d",
                batch_start,
                batch_start + len(batch),
                len(sequences),
            )
            for seq in batch:
                raw_output = self._run_inference(seq)
                aggregated = self._aggregate(raw_output)
                outputs.append(aggregated)
        return outputs

    @abstractmethod
    def _run_inference(self, sequence: str) -> BioEmuOutput:
        """Run inference for a single sequence. Return un-aggregated output."""

    @staticmethod
    def _aggregate(output: BioEmuOutput) -> BioEmuOutput:
        """
        Compute ensemble-level summary statistics in-place and return the object.
        Called automatically by infer_batch after _run_inference.
        """
        if not output.samples:
            return output

        confidences, energies, rgs = [], [], []
        distance_mats = []

        for sample in output.samples:
            if sample.per_residue_confidence is not None:
                confidences.append(float(np.mean(sample.per_residue_confidence)))
            if sample.energy_proxy is not None:
                energies.append(sample.energy_proxy)
            if sample.radius_of_gyration is not None:
                rgs.append(sample.radius_of_gyration)
            if sample.distance_matrix is not None:
                distance_mats.append(sample.distance_matrix.flatten())

        if confidences:
            output.mean_confidence = float(np.mean(confidences))
            output.confidence_std = float(np.std(confidences))
        if energies:
            output.mean_energy = float(np.mean(energies))
            output.energy_std = float(np.std(energies))
        if rgs:
            output.mean_rg = float(np.mean(rgs))
            output.rg_std = float(np.std(rgs))
        if distance_mats:
            stacked = np.stack(distance_mats, axis=0)   # (n_samples, L*L)
            output.pairwise_distance_variance = float(np.mean(np.var(stacked, axis=0)))

        return output


# ---------------------------------------------------------------------------
# Concrete BioEmu backend
# ---------------------------------------------------------------------------


class BioEmuWrapper(BaseStructuralBackend):
    """
    Wraps the real BioEmu model (microsoft/bioemu).

    BioEmu is a diffusion-based structural ensemble model. This wrapper
    calls its Python API; the exact interface may need to be adjusted to
    the installed version.

    Install BioEmu separately — it is not listed in pyproject.toml because
    it requires CUDA and has large weights.
    """

    def __init__(self, config: BioEmuConfig) -> None:
        super().__init__(config)
        self._model = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            # BioEmu's public Python API (adjust import path to match your install)
            from bioemu.inference import BioEmuModel  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "BioEmu is not installed or not importable. "
                "Follow the installation guide at "
                "https://github.com/microsoft/bioemu"
            ) from exc

        logger.info("Loading BioEmu from: %s", self.config.model_path or "default")
        self._model = BioEmuModel.from_pretrained(
            self.config.model_path,
            device=self.config.device,
        )
        self._model.eval()
        self._loaded = True
        logger.info("BioEmu ready on device: %s", self.config.device)

    def _run_inference(self, sequence: str) -> BioEmuOutput:
        self._ensure_loaded()
        assert self._model is not None

        raw_samples = self._model.sample(
            sequence=sequence,
            num_samples=self.config.num_samples,
            num_steps=self.config.inference_steps,
        )
        output = BioEmuOutput(sequence=sequence)
        for raw in raw_samples:
            sample = ConformationSample(
                per_residue_confidence=np.array(raw.get("plddt", [])) if raw.get("plddt") else None,
                energy_proxy=raw.get("energy"),
                distance_matrix=np.array(raw["distance_matrix"]) if "distance_matrix" in raw else None,
                radius_of_gyration=raw.get("radius_of_gyration"),
                sasa=np.array(raw["sasa"]) if "sasa" in raw else None,
            )
            output.samples.append(sample)
        return output


# ---------------------------------------------------------------------------
# Mock backend (testing / GPU-free environments)
# ---------------------------------------------------------------------------


class MockBioEmuBackend(BaseStructuralBackend):
    """
    Deterministic synthetic backend for unit tests and GPU-free development.

    Scores are a simple hash of the sequence so mutations produce measurably
    different values, enabling end-to-end GA testing without real models.
    """

    def _run_inference(self, sequence: str) -> BioEmuOutput:
        rng = np.random.default_rng(seed=abs(hash(sequence)) % (2**31))

        output = BioEmuOutput(sequence=sequence)
        for _ in range(self.config.num_samples):
            L = len(sequence)
            output.samples.append(
                ConformationSample(
                    per_residue_confidence=rng.uniform(60, 95, size=L),
                    energy_proxy=float(rng.uniform(-200, -50)),
                    distance_matrix=rng.uniform(3.5, 20.0, size=(L, L)),
                    radius_of_gyration=float(rng.uniform(10, 30)),
                    sasa=rng.uniform(0, 150, size=L),
                )
            )
        return output


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_bioemu_backend(config: BioEmuConfig) -> BaseStructuralBackend:
    """Return the appropriate backend based on config.mock."""
    if config.mock:
        logger.warning(
            "BioEmu running in MOCK mode — outputs are synthetic. "
            "Set bioemu.mock=false for real inference."
        )
        return MockBioEmuBackend(config)
    return BioEmuWrapper(config)
