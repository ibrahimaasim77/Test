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
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import BioEmuConfig

logger = logging.getLogger(__name__)

# Supported BioEmu model versions
BIOEMU_MODELS = ("bioemu-v1.0", "bioemu-v1.1", "bioemu-v1.2")


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
    Wraps the real BioEmu model (microsoft/bioemu v1.4+).

    Uses bioemu.sample.main() which writes NPZ files to a temp directory.
    We parse those files and compute structural features (Rg, distance matrix,
    energy proxy) from the Cα coordinates.

    Note: BioEmu does not output pLDDT — StabilityScorer will return its
    neutral default (0.5). The other three scorers (consistency, energy,
    compactness) all work from coordinates and will be fully active.

    Install BioEmu separately:
        pip install git+https://github.com/microsoft/bioemu.git
    """

    def _run_inference(self, sequence: str) -> BioEmuOutput:
        try:
            from bioemu.sample import main as bioemu_sample
        except ImportError as exc:
            raise ImportError(
                "BioEmu is not installed. "
                "Run: pip install git+https://github.com/microsoft/bioemu.git"
            ) from exc

        # model_path doubles as model_name if it's a version string
        model_name = self.config.model_path or "bioemu-v1.1"
        if model_name not in BIOEMU_MODELS:
            model_name = "bioemu-v1.1"

        logger.info(
            "BioEmu inference: seq_len=%d | model=%s | samples=%d",
            len(sequence), model_name, self.config.num_samples,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bioemu_sample(
                sequence=sequence,
                num_samples=self.config.num_samples,
                output_dir=tmpdir,
                batch_size_100=self.config.batch_size,
                model_name=model_name,
            )
            return self._parse_output_dir(sequence, Path(tmpdir))

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_output_dir(self, sequence: str, output_dir: Path) -> BioEmuOutput:
        output = BioEmuOutput(sequence=sequence)

        # BioEmu writes: batch_0000000_0000010.npz, batch_0000010_0000020.npz ...
        npz_files = sorted(output_dir.glob("batch_*.npz"))
        for npz_file in npz_files:
            output.samples.extend(self._parse_npz(npz_file))

        # Fallback: individual PDB files (written if convert_chemgraph ran)
        if not output.samples:
            for pdb_file in sorted(output_dir.glob("*.pdb")):
                sample = self._parse_pdb(pdb_file)
                if sample is not None:
                    output.samples.append(sample)

        if not output.samples:
            logger.warning(
                "BioEmu produced no parseable samples for sequence %s...",
                sequence[:10],
            )
        else:
            logger.info("Parsed %d BioEmu samples.", len(output.samples))

        return output

    def _parse_npz(self, path: Path) -> List[ConformationSample]:
        """
        Load one BioEmu NPZ file and return one ConformationSample per frame.

        BioEmu stores Cα (and backbone) coordinates under keys like 'pos' or
        'positions'. Shape is typically (n_samples, L, 3) for Cα-only or
        (n_samples, L, n_atoms, 3) for full backbone — we extract Cα (index 1).
        """
        samples: List[ConformationSample] = []
        try:
            data = np.load(path, allow_pickle=False)
        except Exception as exc:
            logger.warning("Could not load NPZ %s: %s", path, exc)
            return samples

        # Log available keys on first file for debugging
        logger.debug("NPZ keys in %s: %s", path.name, list(data.files))

        coords = self._extract_coords(data)
        if coords is None:
            logger.warning("No coordinate array found in %s", path.name)
            return samples

        # Normalise to (n_samples, L, 3)
        if coords.ndim == 2:                        # (L, 3) — single sample
            coords = coords[np.newaxis]
        elif coords.ndim == 4:                      # (N, L, atoms, 3) — take Cα
            coords = coords[:, :, 1, :]

        for i in range(len(coords)):
            sample = self._coords_to_sample(coords[i])
            if sample is not None:
                samples.append(sample)

        return samples

    @staticmethod
    def _extract_coords(data) -> Optional[np.ndarray]:
        """Try common key names for coordinate arrays in BioEmu NPZ files."""
        for key in ("pos", "positions", "coords", "ca_coords", "backbone_pos", "x"):
            if key in data.files:
                arr = data[key]
                if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                    return arr

        # Last resort: first array whose last dimension is 3
        for key in data.files:
            arr = data[key]
            if isinstance(arr, np.ndarray) and arr.ndim >= 2 and arr.shape[-1] == 3:
                return arr

        return None

    def _parse_pdb(self, path: Path) -> Optional[ConformationSample]:
        """Parse Cα coordinates from a PDB file using Biopython."""
        try:
            from Bio.PDB import PDBParser
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("prot", str(path))
            ca_coords: List[np.ndarray] = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.get_id()[0] == " " and "CA" in residue:
                            ca_coords.append(residue["CA"].get_coord())
                break  # first model only
            if ca_coords:
                return self._coords_to_sample(np.array(ca_coords))
        except Exception as exc:
            logger.warning("Could not parse PDB %s: %s", path, exc)
        return None

    @staticmethod
    def _coords_to_sample(ca_coords: np.ndarray) -> Optional[ConformationSample]:
        """
        Compute structural features from a (L, 3) Cα coordinate array.

        Features computed:
          - distance_matrix: pairwise Cα distances (Å)
          - radius_of_gyration: Cα Rg (Å)
          - energy_proxy: negative mean pairwise distance
            (more compact structure = lower value = more stable proxy)
        """
        if ca_coords.ndim != 2 or ca_coords.shape[1] != 3 or len(ca_coords) < 2:
            return None

        L = len(ca_coords)

        # Pairwise Cα distance matrix (Å)
        diff = ca_coords[:, None, :] - ca_coords[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Radius of gyration
        center = ca_coords.mean(axis=0)
        rg = float(np.sqrt(np.mean(np.sum((ca_coords - center) ** 2, axis=1))))

        # Energy proxy: negative mean pairwise distance (lower = more compact)
        upper = dist_matrix[np.triu_indices(L, k=1)]
        energy_proxy = -float(np.mean(upper))

        return ConformationSample(
            per_residue_confidence=None,   # BioEmu v1.x does not output pLDDT
            energy_proxy=energy_proxy,
            distance_matrix=dist_matrix,
            radius_of_gyration=rg,
            sasa=None,
        )


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
