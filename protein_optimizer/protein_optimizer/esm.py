"""
ESM-2 Mutation Proposal Module

Wraps Meta's ESM-2 (via HuggingFace) as a biologically-guided mutation oracle.

Workflow per call:
  1. Receive a protein sequence + target position
  2. Mask that position → "<mask>"
  3. Run ESM-2 masked-language-model inference
  4. Return top-k ranked amino acid substitutions with log-probabilities

The module is deliberately stateless per call and GPU-batch-aware.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import ESM2Config

logger = logging.getLogger(__name__)

# Canonical single-letter amino acid alphabet (20 standard residues)
AMINO_ACIDS: List[str] = list("ACDEFGHIKLMNPQRSTVWY")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MutationCandidate:
    """A single ranked substitution at one sequence position."""

    position: int
    original_aa: str
    proposed_aa: str
    log_prob: float

    def __repr__(self) -> str:
        return (
            f"MutationCandidate(pos={self.position}, "
            f"{self.original_aa}→{self.proposed_aa}, "
            f"log_prob={self.log_prob:.3f})"
        )


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class ESM2MutationProposer:
    """
    Wraps ESM-2 to suggest biologically plausible amino acid substitutions.

    Args:
        config: ESM2Config instance controlling model name, device, batch size.

    Example::

        proposer = ESM2MutationProposer(cfg.esm2)
        candidates = proposer.propose(sequence="MKTLLLT...", positions=[5, 12])
        # candidates: Dict[int, List[MutationCandidate]]
    """

    _MASK_TOKEN: str = "<mask>"

    def __init__(self, config: ESM2Config) -> None:
        self.config = config
        self._tokenizer = None
        self._model = None
        self._aa_token_ids: Optional[List[int]] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Lazy loading — only pay the cost when first used
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        try:
            import torch  # noqa: F401
            from transformers import AutoTokenizer, EsmForMaskedLM
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required for ESM-2 integration. "
                "Install with: pip install torch transformers"
            ) from exc

        logger.info("Loading ESM-2 model: %s", self.config.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
        )
        self._model = EsmForMaskedLM.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
        )
        self._model.eval()
        self._model = self._model.to(self.config.device)

        # Pre-compute token IDs for the 20 canonical amino acids
        self._aa_token_ids = [
            self._tokenizer.convert_tokens_to_ids(aa) for aa in AMINO_ACIDS
        ]
        logger.info("ESM-2 ready on device: %s", self.config.device)
        self._loaded = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose(
        self,
        sequence: str,
        positions: Optional[List[int]] = None,
    ) -> Dict[int, List[MutationCandidate]]:
        """
        Propose top-k substitutions for each target position.

        Args:
            sequence: Protein sequence string (single-letter AA codes).
            positions: Positions to evaluate. None = all positions.

        Returns:
            Mapping of position → ranked MutationCandidate list (best first).
        """
        self._ensure_loaded()
        if positions is None:
            positions = list(range(len(sequence)))

        results: Dict[int, List[MutationCandidate]] = {}

        # Process in batches to respect GPU memory limits
        for batch_start in range(0, len(positions), self.config.batch_size):
            batch_positions = positions[batch_start : batch_start + self.config.batch_size]
            batch_results = self._score_batch(sequence, batch_positions)
            results.update(batch_results)

        return results

    def propose_single(
        self, sequence: str, position: int
    ) -> List[MutationCandidate]:
        """Convenience wrapper for a single position."""
        return self.propose(sequence, [position])[position]

    # ------------------------------------------------------------------
    # Internal inference
    # ------------------------------------------------------------------

    def _score_batch(
        self, sequence: str, positions: List[int]
    ) -> Dict[int, List[MutationCandidate]]:
        """
        Build one masked sequence per position, batch-tokenise, run forward pass,
        and extract log-probabilities for the 20 canonical AAs.
        """
        import torch

        assert self._tokenizer is not None
        assert self._model is not None
        assert self._aa_token_ids is not None

        masked_sequences = []
        for pos in positions:
            seq_list = list(sequence)
            seq_list[pos] = self._MASK_TOKEN
            masked_sequences.append(" ".join(seq_list))

        encodings = self._tokenizer(
            masked_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = encodings["input_ids"].to(self.config.device)
        attention_mask = encodings["attention_mask"].to(self.config.device)

        with torch.no_grad():
            logits = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits  # (batch, seq_len, vocab)

        results: Dict[int, List[MutationCandidate]] = {}
        for batch_idx, pos in enumerate(positions):
            # +1 because tokenizer prepends <cls> token
            mask_token_pos = pos + 1
            token_logits = logits[batch_idx, mask_token_pos, :]  # (vocab,)

            log_probs = torch.log_softmax(
                token_logits / self.config.temperature, dim=-1
            )
            aa_log_probs = log_probs[self._aa_token_ids]  # (20,)

            # Sort descending by log-prob
            sorted_indices = torch.argsort(aa_log_probs, descending=True)
            candidates = []
            for rank_idx in sorted_indices[: self.config.top_k_candidates]:
                aa = AMINO_ACIDS[rank_idx.item()]
                lp = aa_log_probs[rank_idx].item()
                candidates.append(
                    MutationCandidate(
                        position=pos,
                        original_aa=sequence[pos],
                        proposed_aa=aa,
                        log_prob=lp,
                    )
                )
            results[pos] = candidates

        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_top_substitution(
        self, sequence: str, position: int, exclude_original: bool = True
    ) -> MutationCandidate:
        """Return the single highest-probability substitution at a position."""
        candidates = self.propose_single(sequence, position)
        if exclude_original:
            candidates = [c for c in candidates if c.proposed_aa != sequence[position]]
        if not candidates:
            raise ValueError(
                f"No valid candidates at position {position} "
                f"(original: {sequence[position]})."
            )
        return candidates[0]
