"""
Evolutionary Search Pipeline

Implements the team's target algorithm:

  1. Score the reference sequence with BioEmu → reference_llr  (e.g. -2)
  2. Use ESM-2 to generate a batch of 100 candidate mutations
  3. Score all 100 with BioEmu → pick the top 20 by LLR
  4. Find positions that are IDENTICAL across all 20 (the "common ancestor")
  5. Crossover: fix those positions, randomly recombine the rest → 100 new sequences
  6. Repeat from step 3 until 500 total unique mutations have been evaluated

The LLR here is BioEmu's mean_energy (lower = more stable; we rank higher = better
so internally we negate it so the best sequences sort to the top).  The reference LLR
lets you calibrate: a mutant is "better" when its LLR beats the reference.

Usage::

    from protein_optimizer.config import OptimizationConfig
    from protein_optimizer.evolutionary_search import BudgetedEvolutionarySearch

    cfg = OptimizationConfig.from_yaml("config/default.yaml")
    search = BudgetedEvolutionarySearch(cfg)
    result = search.run()
    print(result.best_sequence, result.best_llr, "vs reference", result.reference_llr)
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .bioemu import BaseStructuralBackend, BioEmuOutput, build_bioemu_backend, write_trajectory_files
from .config import OptimizationConfig
from .esm import ESM2MutationProposer
from .mutation import CommonAncestorCrossover, ESMGuidedMutator, RandomMutator, MutationConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass
class EvolutionarySearchResult:
    """Final result returned by BudgetedEvolutionarySearch.run()."""

    reference_llr: float                    # BioEmu LLR of the original (defective) sequence
    best_sequence: str                       # best mutant found
    best_llr: float                          # its BioEmu LLR
    rounds_run: int                          # how many crossover rounds were completed
    total_evaluated: int                     # total unique sequences scored
    all_scores: Dict[str, float]             # {sequence: llr} for every scored mutant
    top20_per_round: List[List[str]]         # the elite-20 snapshot after each round
    total_wall_time_s: float
    # Goal LLR when running in fit-to-target mode; None = maximise mode
    target_parameter: Optional[float] = None
    # BioEmuOutput objects kept for trajectory file export after the search
    reference_output: Optional[BioEmuOutput] = None
    best_output: Optional[BioEmuOutput] = None

    def _fitness(self, llr: float) -> float:
        """Higher = better in both modes. Maximise mode: the LLR itself.
        Target mode: negative distance to the goal (closest = highest)."""
        if self.target_parameter is None:
            return llr
        return -abs(llr - self.target_parameter)

    @property
    def improved(self) -> bool:
        """Maximise mode: best LLR beats the reference.
        Target mode: best LLR is closer to the goal than the reference is."""
        return self._fitness(self.best_llr) > self._fitness(self.reference_llr)

    def ranked(self, n: int = 20) -> List[Tuple[str, float]]:
        """Return the top-n (sequence, llr) pairs, best first (by fitness)."""
        return sorted(
            self.all_scores.items(),
            key=lambda x: self._fitness(x[1]),
            reverse=True,
        )[:n]


# ---------------------------------------------------------------------------
# Core search class
# ---------------------------------------------------------------------------


class BudgetedEvolutionarySearch:
    """
    Budget-constrained evolutionary sequence search.

    Parameters (from OptimizationConfig):
      - original_sequence     : the reference / starting sequence
      - ga.population_size    : mutations per round (default: 100)
      - ga.elite_size         : elite pool size used for crossover (default: 20)
      - ga.max_generations    : max rounds before stopping (default: 5)
      - bioemu.*              : BioEmu inference settings
      - esm2.*                : ESM-2 settings for round-1 mutation proposals
      - mutation.*            : max_mutations_per_sequence, allowed_positions, etc.

    Total mutations scored = population_size × max_generations (≤ 500 by default).
    """

    def __init__(
        self,
        config: OptimizationConfig,
        bioemu_backend: Optional[BaseStructuralBackend] = None,
        verbose: bool = False,
    ) -> None:
        self.config = config
        self._rng = random.Random(config.ga.seed)
        self._verbose = verbose

        self._bioemu = bioemu_backend or build_bioemu_backend(config.bioemu)
        self._crossover = CommonAncestorCrossover(self._rng)
        self._use_esm = config.mutation.strategy == "esm_guided"
        # Only instantiate ESM-2 if actually needed — avoids any HF download otherwise
        self._esm: Optional[ESM2MutationProposer] = (
            ESM2MutationProposer(config.esm2) if self._use_esm else None
        )

        self._batch_size = config.ga.population_size   # 100
        self._n_elite = config.ga.elite_size           # 20
        self._max_rounds = config.ga.max_generations   # 5  → 500 total

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _fitness(self, llr: float) -> float:
        """Higher = better in both modes. Maximise mode: the LLR itself.
        Target mode: negative distance to the goal (closest to goal = highest)."""
        if self._target is None:
            return llr
        return -abs(llr - self._target)

    def run(self, progress_callback=None) -> EvolutionarySearchResult:
        """Execute the full search. Returns EvolutionarySearchResult."""

        def emit(event: dict) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(event)
                except Exception:
                    pass

        start = time.time()
        original = self.config.original_sequence

        # Score the reference (defective) sequence first
        logger.info("Scoring reference sequence (len=%d) ...", len(original))
        emit({"type": "reference_scoring", "seq_len": len(original)})
        ref_output = self._bioemu.infer_batch([original])[0]
        reference_llr = self._extract_llr(ref_output)
        logger.info("Reference LLR = %.4f", reference_llr)

        # Determine the optimisation target (goal LLR), if any.
        #   healthy_sequence (scored once)  >  target_parameter  >  None (maximise)
        self._target: Optional[float] = None
        if self.config.healthy_sequence:
            logger.info("Scoring healthy reference sequence for target parameter ...")
            healthy_output = self._bioemu.infer_batch([self.config.healthy_sequence])[0]
            self._target = self._extract_llr(healthy_output)
            logger.info("Target parameter (from healthy protein) = %.4f", self._target)
        elif self.config.target_parameter is not None:
            self._target = float(self.config.target_parameter)
            logger.info("Target parameter (user-defined) = %.4f", self._target)
        else:
            logger.info("No target set — maximising LLR.")

        emit({
            "type": "reference_scored",
            "llr": reference_llr,
            "seq_len": len(original),
            "target": self._target,
            "mode": "maximize" if self._target is None else "fit",
        })

        if self._verbose:
            self._print_header("REFERENCE SEQUENCE")
            print(f"  Sequence : {original}", flush=True)
            print(f"  Length   : {len(original)} residues", flush=True)
            print(f"  LLR      : {reference_llr:.4f}", flush=True)
            if self._target is not None:
                print(f"  Target   : {self._target:.4f}  (optimising toward this)", flush=True)
            else:
                print(f"  Goal     : maximise LLR (no target set)", flush=True)

        all_scores: Dict[str, float] = {}
        top20_per_round: List[List[str]] = []
        current_parents: List[str] = [original]
        best_output: Optional[BioEmuOutput] = None
        best_fitness_tracked: float = float("-inf")

        for round_idx in range(self._max_rounds):
            remaining = self._batch_size

            if round_idx == 0:
                src = "ESM-2 guided mutations" if self._use_esm else "random mutations"
                if self._verbose:
                    self._print_round_header(round_idx + 1, self._max_rounds,
                                             f"generating {remaining} candidates via {src}")
                emit({
                    "type": "generating",
                    "round": round_idx + 1,
                    "total_rounds": self._max_rounds,
                    "method": "esm2" if self._use_esm else "random",
                    "count": remaining,
                })
                if self._use_esm:
                    logger.info("Round 1 — ESM-2 generating %d candidates ...", remaining)
                    candidates = self._esm2_generate(current_parents, remaining)
                else:
                    logger.info("Round 1 — random mutations generating %d candidates ...", remaining)
                    candidates = self._random_generate(current_parents, remaining)
            else:
                n_variable = self._crossover.variable_position_count(current_parents)
                if self._verbose:
                    self._print_round_header(
                        round_idx + 1, self._max_rounds,
                        f"crossover of top-{self._n_elite} parents "
                        f"({n_variable} variable positions) → {remaining} new candidates",
                    )
                logger.info(
                    "Round %d — crossover of %d parents (%d variable positions) → %d candidates",
                    round_idx + 1, len(current_parents), n_variable, remaining,
                )
                emit({
                    "type": "generating",
                    "round": round_idx + 1,
                    "total_rounds": self._max_rounds,
                    "method": "crossover",
                    "count": remaining,
                    "variable_positions": n_variable,
                    "elite_size": self._n_elite,
                })
                candidates = self._crossover.generate_offspring(current_parents, remaining)

            new_candidates = [s for s in candidates if s not in all_scores and s != original]
            if not new_candidates:
                logger.warning("Round %d produced no new unique candidates — stopping.", round_idx + 1)
                if self._verbose:
                    print(f"  [!] No new unique candidates — stopping early.", flush=True)
                break

            if self._verbose:
                print(f"  Scoring {len(new_candidates)} candidates with BioEmu...", flush=True)

            emit({"type": "scoring", "round": round_idx + 1, "count": len(new_candidates)})
            logger.info("Round %d — scoring %d sequences with BioEmu ...", round_idx + 1, len(new_candidates))
            outputs = self._bioemu.infer_batch(new_candidates)
            round_scores: Dict[str, float] = {}
            for seq, out in zip(new_candidates, outputs):
                llr = self._extract_llr(out)
                all_scores[seq] = llr
                round_scores[seq] = llr
                if self._fitness(llr) > best_fitness_tracked:
                    best_fitness_tracked = self._fitness(llr)
                    best_output = out

            # Show every candidate scored this round
            if self._verbose:
                col2 = "dist→goal" if self._target is not None else "δ vs ref"
                round_ranked = sorted(round_scores.items(), key=lambda x: self._fitness(x[1]), reverse=True)
                print(f"\n  All {len(round_ranked)} candidates scored this round:", flush=True)
                print(f"  {'Rank':>4}  {'LLR':>8}  {col2:>9}  Sequence", flush=True)
                print(f"  {'-'*4}  {'-'*8}  {'-'*9}  {'-'*40}", flush=True)
                for rank, (seq, llr) in enumerate(round_ranked, 1):
                    secondary = abs(llr - self._target) if self._target is not None else (llr - reference_llr)
                    marker = " ◄ best" if rank == 1 else ""
                    print(
                        f"  {rank:>4}  {llr:>8.4f}  {secondary:>+9.4f}  {seq[:50]}{marker}",
                        flush=True,
                    )

            # Select global top-20 (closest to target, or highest LLR if maximising)
            sorted_all = sorted(all_scores.items(), key=lambda x: self._fitness(x[1]), reverse=True)
            top20 = [seq for seq, _ in sorted_all[: self._n_elite]]
            top20_per_round.append(top20)

            best_this_round = sorted_all[0]
            logger.info(
                "Round %d complete | scored=%d | best_so_far=%.4f (%s...)",
                round_idx + 1, len(all_scores), best_this_round[1], best_this_round[0][:15],
            )

            emit({
                "type": "round_scored",
                "round": round_idx + 1,
                "total_rounds": self._max_rounds,
                "scores": [{"seq": s, "llr": l} for s, l in round_scores.items()],
                "top20": [{"seq": s, "llr": all_scores[s]} for s in top20],
                "best_llr": best_this_round[1],
                "total_evaluated": len(all_scores),
            })

            if self._verbose:
                col2 = "dist→goal" if self._target is not None else "δ vs ref"
                print(f"\n  Top {self._n_elite} selected (global best so far — seed next round):", flush=True)
                print(f"  {'Rank':>4}  {'LLR':>8}  {col2:>9}  Sequence", flush=True)
                print(f"  {'-'*4}  {'-'*8}  {'-'*9}  {'-'*40}", flush=True)
                for rank, seq in enumerate(top20, 1):
                    llr = all_scores[seq]
                    secondary = abs(llr - self._target) if self._target is not None else (llr - reference_llr)
                    print(
                        f"  {rank:>4}  {llr:>8.4f}  {secondary:>+9.4f}  {seq[:50]}",
                        flush=True,
                    )
                print(flush=True)

            current_parents = top20

        elapsed = time.time() - start
        best_seq, best_llr = max(all_scores.items(), key=lambda x: self._fitness(x[1]))
        improved = self._fitness(best_llr) > self._fitness(reference_llr)

        logger.info(
            "Search complete | rounds=%d | total_evaluated=%d | target=%s | "
            "best_llr=%.4f | reference_llr=%.4f | improved=%s | wall_time=%.1fs",
            len(top20_per_round), len(all_scores),
            f"{self._target:.4f}" if self._target is not None else "none",
            best_llr, reference_llr, improved, elapsed,
        )

        return EvolutionarySearchResult(
            reference_llr=reference_llr,
            best_sequence=best_seq,
            best_llr=best_llr,
            rounds_run=len(top20_per_round),
            total_evaluated=len(all_scores),
            all_scores=all_scores,
            top20_per_round=top20_per_round,
            total_wall_time_s=elapsed,
            target_parameter=self._target,
            reference_output=ref_output,
            best_output=best_output,
        )

    # ------------------------------------------------------------------
    # Verbose display helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_header(title: str) -> None:
        bar = "=" * 65
        print(f"\n{bar}", flush=True)
        print(f"  {title}", flush=True)
        print(f"{bar}", flush=True)

    @staticmethod
    def _print_round_header(round_num: int, total: int, description: str) -> None:
        bar = "-" * 65
        print(f"\n{bar}", flush=True)
        print(f"  Round {round_num} / {total}  —  {description}", flush=True)
        print(f"{bar}", flush=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_llr(self, output: BioEmuOutput) -> float:
        """
        Extract a single LLR scalar from a BioEmuOutput.

        Priority:
          1. log_partition — actual BioEmu importance-weighted free energy estimate,
             available when the NPZ output contains 'log_weights' (BioEmu v1.4+).
             Higher = more favourable free energy = better sequence.
          2. mean_energy proxy — negative mean pairwise Cα distance (structural
             compactness proxy). Used when log_weights are absent.
          3. mean_rg — radius of gyration fallback.
        """
        if output.log_partition is not None:
            return output.log_partition
        if output.mean_energy is not None:
            return -output.mean_energy   # negate: lower raw energy → higher score
        if output.mean_rg is not None:
            return -output.mean_rg
        return 0.0

    def _random_generate(self, parents: List[str], n: int) -> List[str]:
        """Generate `n` candidates via random amino acid substitutions (no HF download)."""
        mutator = RandomMutator(config=self.config.mutation, rng=self._rng)
        return [mutator.mutate(self._rng.choice(parents)) for _ in range(n)]

    def _esm2_generate(self, parents: List[str], n: int) -> List[str]:
        """
        Generate `n` mutation candidates via ESM-2.

        For each slot, picks a random parent and proposes a mutation using the
        ESM-2 masked-LM model (biologically guided substitution).
        """
        mut_cfg = self.config.mutation
        # Build a one-off ESMGuidedMutator for this call
        mutator = ESMGuidedMutator(
            config=mut_cfg,
            esm_proposer=self._esm,
            rng=self._rng,
        )

        candidates: List[str] = []
        for _ in range(n):
            parent = self._rng.choice(parents)
            mutant = mutator.mutate(parent)
            candidates.append(mutant)
        return candidates
