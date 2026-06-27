"""
Entry point for the Protein Optimization Framework.

Quick start (no GPU — uses mock BioEmu):

    python main.py --config config/default.yaml --mock

Full run:

    python main.py --config config/default.yaml --sequence MKTLLILAVLCLGFAQ

Override any top-level config field via --set:

    python main.py --config config/default.yaml \
        --set ga.population_size=100 \
        --set ga.max_generations=200 \
        --set bioemu.mock=true
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from protein_optimizer import OptimizationConfig, ProteinOptimizationPipeline

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Protein Sequence Optimization via Genetic Algorithm + BioEmu + ESM-2",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--sequence",
        default=None,
        help="Override original_sequence from config (single-letter AA codes)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run with mock BioEmu backend (no GPU required)",
    )
    parser.add_argument(
        "--random-mutations",
        action="store_true",
        help="Use random mutations instead of ESM-2 guided (faster, lower quality)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override results output directory",
    )
    parser.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help="Override config fields (e.g. --set ga.population_size=100)",
    )
    return parser.parse_args()


def apply_overrides(cfg: OptimizationConfig, overrides: list[str]) -> None:
    """Apply dot-notation overrides: 'ga.population_size=100'"""
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: {override!r}. Expected KEY=VALUE")
        key_path, _, raw_value = override.partition("=")
        parts = key_path.strip().split(".")

        # Coerce value type
        value: object
        if raw_value.lower() in ("true", "false"):
            value = raw_value.lower() == "true"
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value

        # Navigate to target sub-config
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
        logger.debug("Config override: %s = %r", key_path, value)


def main() -> None:
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    cfg = OptimizationConfig.from_yaml(config_path)

    # Apply CLI overrides (in order of precedence)
    apply_overrides(cfg, args.set)

    if args.sequence:
        cfg.original_sequence = args.sequence.upper().strip()
    if args.mock:
        cfg.bioemu.mock = True
    if args.random_mutations:
        cfg.mutation.strategy = "random"
    if args.output_dir:
        cfg.logging.output_dir = args.output_dir

    # Run
    pipeline = ProteinOptimizationPipeline(cfg)
    result = pipeline.run()

    print(f"\nBest sequence : {result.best_sequence}")
    print(f"Best score    : {result.best_score:.4f}")
    print(f"Generations   : {result.generations_run}")
    print(f"Wall time     : {result.total_wall_time_s:.1f}s")
    print(f"Results saved : {result.export_paths}")


if __name__ == "__main__":
    main()
