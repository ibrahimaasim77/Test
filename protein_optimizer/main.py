"""
Protein Optimization — Entry Point

Default usage (runs evolutionary search with ESM-2 + BioEmu):

    python main.py --config config/evolutionary.yaml

Override the sequence on the command line:

    python main.py --config config/evolutionary.yaml \
        --sequence MKTLLILAVLCLGFAQASG...

GPU-free development (synthetic BioEmu outputs):

    python main.py --config config/evolutionary.yaml --mock

Verbose round-by-round scoring:

    python main.py --config config/evolutionary.yaml --verbose
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Silence HuggingFace Hub's "set a HF_TOKEN" warning — we intentionally use
# unauthenticated public model downloads (no account/token needed). Must run
# before transformers / huggingface_hub are imported.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))

from protein_optimizer import OptimizationConfig
from protein_optimizer.bioemu import write_trajectory_files
from protein_optimizer.evolutionary_search import BudgetedEvolutionarySearch

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Protein Sequence Optimization — ESM-2 mutations scored by BioEmu LLR",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/evolutionary.yaml",
        help="Path to YAML config file (default: config/evolutionary.yaml)",
    )
    parser.add_argument(
        "--sequence",
        default=None,
        help="Protein sequence to optimise (single-letter AA codes). Overrides config.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use synthetic BioEmu outputs — no GPU required (for testing)",
    )
    parser.add_argument(
        "--random-mutations",
        action="store_true",
        help="Use random mutations instead of ESM-2 (no HuggingFace download needed)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print live scoring table after each round",
    )
    parser.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help="Override any config field, e.g. --set ga.population_size=100",
    )
    return parser.parse_args()


def apply_overrides(cfg: OptimizationConfig, overrides: list[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: {override!r}. Expected KEY=VALUE")
        key_path, _, raw_value = override.partition("=")
        parts = key_path.strip().split(".")
        if raw_value.lower() in ("true", "false"):
            value: object = raw_value.lower() == "true"
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    cfg = OptimizationConfig.from_yaml(config_path)
    apply_overrides(cfg, args.set)

    if args.sequence:
        cfg.original_sequence = args.sequence.upper().strip()
    if args.mock:
        cfg.bioemu.mock = True
    if args.random_mutations:
        cfg.mutation.strategy = "random"
    else:
        cfg.mutation.strategy = "esm_guided"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.log_level, logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\nStarting evolutionary search")
    print(f"  Sequence  : {cfg.original_sequence[:60]}{'...' if len(cfg.original_sequence) > 60 else ''}")
    print(f"  Length    : {len(cfg.original_sequence)} residues")
    print(f"  Strategy  : {'ESM-2 guided' if cfg.mutation.strategy == 'esm_guided' else 'random'} mutations")
    print(f"  Rounds    : {cfg.ga.max_generations}  ×  {cfg.ga.population_size} candidates  "
          f"= {cfg.ga.max_generations * cfg.ga.population_size} total")
    print(f"  BioEmu    : {'MOCK (synthetic)' if cfg.bioemu.mock else 'REAL'}\n")

    search = BudgetedEvolutionarySearch(cfg, verbose=args.verbose)
    result = search.run()

    # ------------------------------------------------------------------
    # Save trajectory files for reference + best sequences
    # ------------------------------------------------------------------
    traj_base = Path(cfg.bioemu.trajectory_dir)
    ref_traj_paths = {}
    best_traj_paths = {}

    if result.reference_output is not None:
        ref_dir = traj_base / "reference"
        ref_traj_paths = write_trajectory_files(result.reference_output, ref_dir)

    if result.best_output is not None:
        best_dir = traj_base / "best_mutant"
        best_traj_paths = write_trajectory_files(result.best_output, best_dir)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)

    print(f"\n  When we ran the original sequence through BioEmu,")
    print(f"  it received an LLR of  {result.reference_llr:.4f}")
    print()
    print(f"  The best mutant sequence our search found")
    print(f"  achieved an LLR of     {result.best_llr:.4f}"
          + ("  [improved]" if result.improved else "  [no improvement]"))

    change = result.best_llr - result.reference_llr
    print()
    print(f"  LLR change             {change:+.4f}   (higher = more favourable)")

    print(f"\n  Best sequence:")
    print(f"    {result.best_sequence}")

    print(f"\n  Search stats:")
    print(f"    Rounds run     : {result.rounds_run}")
    print(f"    Total scored   : {result.total_evaluated}")
    print(f"    Wall time      : {result.total_wall_time_s:.1f}s")

    # Trajectory file locations
    if ref_traj_paths or best_traj_paths:
        print(f"\n  Trajectory files saved:")
        for label, paths in [("Reference", ref_traj_paths), ("Best mutant", best_traj_paths)]:
            for kind, p in paths.items():
                print(f"    {label:12s} .{kind:3s} : {p}")

    # Top 10 mutants
    print(f"\n  Top 10 mutant sequences (by LLR):")
    print(f"  {'Rank':>4}  {'LLR':>8}  Sequence")
    print(f"  {'-'*4}  {'-'*8}  {'-'*45}")
    for rank, (seq, llr) in enumerate(result.ranked(10), 1):
        print(f"  {rank:>4}  {llr:>8.4f}  {seq}")

    print()


if __name__ == "__main__":
    main()
