#!/usr/bin/env bash
#
# Practice run: in silico directed evolution toward a healthy protein.
#
# We take a HEALTHY protein, a DEFECTIVE version of it (5 point mutations),
# and ask the pipeline to evolve the defective sequence so its BioEmu LLR
# moves back toward the healthy protein's LLR.
#
# Usage:
#   bash practice.sh         # REAL BioEmu (run this on the GPU VM)
#   bash practice.sh mock    # synthetic — runs anywhere, no GPU, for rehearsing
#
set -euo pipefail

# Healthy reference protein (the goal).
HEALTHY="MKTLLILAVLCLGFAQASGNIERPIDGFHGDLQSLIKAMFESRHITAYIEQLEELRQRQTAATGGMRQ"

# Defective version: same protein with 5 residues mutated (positions 6,21,36,51,66).
DEFECTIVE="MKTLLGLAVLCLGFAQASGNPERPIDGFHGDLQSLDKAMFESRHITAYIEWLEELRQRQTAATGGKRQ"

# Small/fast settings so the practice run finishes quickly.
EXTRA=""
if [ "${1:-}" = "mock" ]; then
    echo ">>> Running in MOCK mode (synthetic BioEmu, no GPU)"
    EXTRA="--mock --random-mutations --set esm2.device=cpu"
else
    echo ">>> Running with REAL BioEmu (needs the GPU VM)"
fi

set -x
python main.py --config config/evolutionary.yaml \
    --sequence "$DEFECTIVE" \
    --healthy-sequence "$HEALTHY" \
    --set bioemu.num_samples=100 \
    --set ga.population_size=20 \
    --set ga.max_generations=3 \
    $EXTRA
