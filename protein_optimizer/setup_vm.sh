#!/usr/bin/env bash
#
# One-shot BioEmu + protein_optimizer setup for a GPU rental VM
# (Lambda Labs / RunPod / Vast.ai — Ubuntu with NVIDIA drivers preinstalled).
#
# Usage (from inside the repo on the VM):
#     bash setup_vm.sh
#
# It creates a Python venv, installs BioEmu + project deps, and runs a tiny
# mock smoke test. No conda needed — we skip BioEmu's optional sidechain step
# (the code passes filter_samples=False).
#
set -euo pipefail

echo "=== 1/5  Checking GPU ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. This needs a GPU VM with NVIDIA drivers." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== 2/5  Checking Python (need 3.10+) ==="
PY=python3
$PY -c 'import sys; assert sys.version_info >= (3,10), "Python 3.10+ required, found %s" % sys.version' \
    || { echo "ERROR: Python 3.10+ required." >&2; exit 1; }
$PY --version

echo "=== 3/5  Creating virtual environment (.venv) ==="
if [ ! -d .venv ]; then
    $PY -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel

echo "=== 4/5  Installing BioEmu + dependencies ==="
# BioEmu core (MSA tooling is bundled; no separate install needed).
pip install "bioemu[cuda]"
# Project dependencies.
pip install torch transformers numpy pyyaml biopython

echo "=== 5/5  Mock smoke test (no GPU work, verifies wiring) ==="
python main.py --config config/evolutionary.yaml --mock --random-mutations \
    --set ga.population_size=6 --set ga.max_generations=1 --set esm2.device=cpu

cat <<'EOF'

============================================================
 Setup complete.

 To run a REAL BioEmu job with your partner's mutation:

   source .venv/bin/activate
   python main.py --config config/evolutionary.yaml \
       --sequence <THE_MUTATION_SEQUENCE> \
       --set bioemu.num_samples=100

 The output prints the reference LLR, the best LLR found,
 and the LLR change. Trajectory files (.xtc/.pdb) are saved
 under results/trajectories/.

 NOTE: 100 samples x many candidates is slow. For a first
 real test, shrink the search:
       --set ga.population_size=20 --set ga.max_generations=2

 HuggingFace: model weights download once with NO token needed
 (the "set a HF_TOKEN" warning is already suppressed). After the
 first run has cached the weights, you can go fully offline with:
       export HF_HUB_OFFLINE=1
============================================================
EOF
