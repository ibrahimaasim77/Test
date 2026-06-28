Protein Optimizer
An industrial-grade evolutionary optimization framework for protein sequence design.

Combines a Genetic Algorithm (GA) with BioEmu structural-ensemble inference and Meta ESM-2 guided mutations to evolve protein sequences toward a desired conformational landscape, not just a single static fold.

For AI assistants reading this: This is a pure inference + optimization system. We are NOT training any models. ESM-2 is used only to propose biologically plausible mutations. BioEmu is used only to evaluate structural quality. The GA drives the search. Every component is modular and swappable.
Table of Contents

Quick Start: GPU VM (Directed Evolution)
Problem Statement
System Architecture
Component Breakdown
Data Flow
Project Structure
Installation
Configuration
Running the Optimizer
Extension Points
Testing
Design Principles
Quick Start: GPU VM (Directed Evolution)

This is the end-to-end workflow for running the real BioEmu pipeline on a GPU rental VM (Lambda / RunPod / Vast). It evolves a defective protein sequence so its BioEmu parameter (LLR) approaches a goal — either a number you choose or the LLR of a healthy protein.

1. Set up the VM (one time)

# On the VM, in a fresh terminal:
git clone https://github.com/ibrahimaasim77/Test.git
cd Test/protein_optimizer
bash setup_vm.sh
setup_vm.sh checks the GPU, creates a .venv, installs BioEmu + dependencies (pip install "bioemu[cuda]"), and runs a mock smoke test. No conda and no HuggingFace token are needed — model weights are public and download once. The "set a HF_TOKEN" warning is already suppressed.

2. Activate the environment (every new terminal)

source .venv/bin/activate
3. Run it

Fit-to-target toward a healthy protein (recommended — the directed-evolution story):

python main.py --config config/evolutionary.yaml \
    --sequence   DEFECTIVE_SEQUENCE \
    --healthy-sequence HEALTHY_SEQUENCE \
    --set bioemu.num_samples=100 \
    --set ga.population_size=20 \
    --set ga.max_generations=3
Fit-to-target toward a chosen LLR value:

python main.py --config config/evolutionary.yaml \
    --sequence DEFECTIVE_SEQUENCE \
    --target -1.5 \
    --set bioemu.num_samples=100
Maximise mode (no goal — just find the highest LLR): omit --target and --healthy-sequence.

4. Practice sequences

A ready-made recovery demo (the healthy protein with 5 point mutations):

# DEFECTIVE (goes after --sequence):
MKTLLGLAVLCLGFAQASGNPERPIDGFHGDLQSLDKAMFESRHITAYIEWLEELRQRQTAATGGKRQ

# HEALTHY (goes after --healthy-sequence):
MKTLLILAVLCLGFAQASGNIERPIDGFHGDLQSLIKAMFESRHITAYIEQLEELRQRQTAATGGMRQ
Or run the bundled script: bash practice.sh (real BioEmu) or bash practice.sh mock (synthetic, no GPU, for rehearsing).

5. How long it takes

Total time ≈ (number of sequences) × (time for one BioEmu run). Sequences scored = 2 (reference + goal) + population_size × max_generations. num_samples is the heavy knob. Time the first sequence and multiply. For a ~5-minute smoke test:

python main.py --config config/evolutionary.yaml \
    --sequence MKTLLGLAVLCLGFAQASGNPERPIDGFHGDLQSLDKAMFESRHITAYIEWLEELRQRQTAATGGKRQ \
    --healthy-sequence MKTLLILAVLCLGFAQASGNIERPIDGFHGDLQSLIKAMFESRHITAYIEQLEELRQRQTAATGGMRQ \
    --set bioemu.num_samples=5 --set ga.population_size=4 --set ga.max_generations=1
6. Output — what you get and where

Printed to screen (save it with ... 2>&1 | tee my_results.txt):

Defective sequence parameter :  -3.9915   (start)
Target (goal) parameter      :  -2.0000   (what we optimise toward)
Best engineered parameter    :  -2.3803   [closer to goal]
Distance to goal: 1.9915  →  0.3803   (closed +1.6111)
Best sequence: ...
Trajectory files (BioEmu's real .xtc + .pdb, saved automatically) under results/trajectories/ — full path on the VM /workspace/Test/protein_optimizer/results/trajectories/:

results/trajectories/reference/samples.xtc    + topology.pdb
results/trajectories/best_mutant/samples.xtc  + topology.pdb
List them: ls -R results/trajectories/

7. Download files to your laptop

RunPod / Vast: use the dashboard's web file browser → navigate to /workspace/Test/protein_optimizer/results/.
SCP (run from your laptop, not the VM):
scp -P <port> root@<vm-ip>:/workspace/Test/protein_optimizer/results/trajectories/best_mutant/samples.xtc .
The provider's "Connect" page gives <vm-ip> and <port>.
Useful flags

Flag	Meaning
--sequence SEQ	The defective / starting sequence (single-letter AA codes)
--healthy-sequence SEQ	Healthy protein; its LLR becomes the goal
--target LLR	Goal LLR as a number (used if no --healthy-sequence)
--set bioemu.num_samples=N	Conformations sampled per sequence (heavy)
--set ga.population_size=N	Candidates generated per round
--set ga.max_generations=N	Number of rounds
--mock	Synthetic BioEmu (no GPU) for testing
--random-mutations	Skip ESM-2; use random mutations
--verbose	Print the per-round scoring tables
After the first run caches the weights, you can go fully offline: export HF_HUB_OFFLINE=1.

Wildtype Recovery Mode

In addition to open-ended structural optimization, the system supports a target-guided recovery experiment:

Provide a known wildtype sequence (the target)
Provide a degraded / mutated starting sequence (the "bad" protein)
The GA evolves the bad sequence back toward the wildtype, guided by both structural fitness and sequence identity
Progress is visualised in N equal stages (default: fifths) with a warm/cold proximity indicator at each checkpoint
This is useful for validating that the system works — if you know the answer, you can watch the GA find it.

  Stage 1/5  (gen 0–19)   ❄❄❄   COLD
  Proximity to wildtype : [████████░░░░░░░░░░░░░░░░] 31.2%

  Stage 2/5  (gen 20–39)  ❄❄    COOL
  Proximity to wildtype : [████████████░░░░░░░░░░░░] 48.6%

  Stage 3/5  (gen 40–59)  🔥    WARM
  Proximity to wildtype : [████████████████░░░░░░░░] 65.3%

  Stage 4/5  (gen 60–79)  🔥🔥   HOT
  Proximity to wildtype : [████████████████████░░░░] 80.1%

  Stage 5/5  (gen 80–99)  🔥🔥🔥  VERY HOT
  Proximity to wildtype : [███████████████████████░] 92.7%
Warmth scale:

Identity	Label	Icons
0–15%	FREEZING	❄❄❄❄❄
15–30%	ICY	❄❄❄❄
30–45%	COLD	❄❄❄
45–60%	COOL	❄❄
60–72%	WARM	🔥
72–84%	HOT	🔥🔥
84–93%	VERY HOT	🔥🔥🔥
93–100%	SCORCHING	🔥🔥🔥🔥
Problem Statement

Given:

An original protein sequence (single-letter amino acid codes)
A target conformational landscape represented by a BioEmu structural ensemble
A BioEmu model that generates structural ensembles for candidate sequences
A pretrained ESM-2 model that predicts biologically plausible amino acid substitutions
Goal: Find a mutated sequence whose equilibrium ensemble matches the target landscape, using evolutionary search.

We are NOT training models. We are doing inference-guided optimization.

System Architecture

┌─────────────────────────────────────────────────────────────────────┐
│                    ProteinOptimizationPipeline                      │
│                         (pipeline.py)                               │
│                                                                     │
│  original_sequence ──► build_initial_population()                  │
│                                  │                                  │
│                                  ▼                                  │
│                    ┌─────────────────────────┐                     │
│                    │    GeneticAlgorithm      │ ◄── GAConfig        │
│                    │   (genetic_algorithm.py) │                     │
│                    │                         │                     │
│                    │  ┌──────────────────┐   │                     │
│                    │  │ evaluate(pop)    │   │                     │
│                    │  │                 │   │                     │
│                    │  │ BioEmu.infer()  │   │ ◄── BioEmuConfig    │
│                    │  │      │          │   │                     │
│                    │  │ Scorer.score()  │   │ ◄── ScoringConfig   │
│                    │  └──────────────────┘   │                     │
│                    │                         │                     │
│                    │  select → crossover      │                     │
│                    │       → mutate           │ ◄── MutationConfig  │
│                    │           │              │                     │
│                    │      ESM-2 proposals     │ ◄── ESM2Config      │
│                    └─────────────────────────┘                     │
│                                  │                                  │
│                    OptimizationTracker (analysis.py)                │
│                    ── per-gen stats, JSON/CSV export                │
└─────────────────────────────────────────────────────────────────────┘
Component Breakdown

config.py — Unified Configuration

All modules are driven by a single OptimizationConfig dataclass. No magic numbers anywhere else.

cfg = OptimizationConfig.from_yaml("config/default.yaml")
# Or build programmatically:
cfg = OptimizationConfig()
cfg.original_sequence = "MKTLLILAVLCLGFAQ"
cfg.bioemu.mock = True          # synthetic outputs, no GPU
cfg.ga.population_size = 50
Sub-configs: ESM2Config, BioEmuConfig, ScoringConfig, MutationConfig, GAConfig, LoggingConfig.

esm.py — ESM-2 Mutation Proposal Engine

Wraps Meta's ESM-2 (via HuggingFace) as a mutation oracle. Given a sequence and a target position, it masks that position and uses ESM-2's masked language model to return the top-k most biologically plausible amino acid substitutions, ranked by log-probability.

Key class: ESM2MutationProposer

proposer = ESM2MutationProposer(cfg.esm2)
candidates = proposer.propose(sequence="MKTLL...", positions=[5, 12])
# candidates: {5: [MutationCandidate(pos=5, A→G, log_prob=-0.3), ...], ...}
Important: ESM-2 is only ever called inside the mutation step. It never touches scoring or the GA directly. It is lazy-loaded — no GPU cost unless strategy="esm_guided".

mutation.py — Mutation and Crossover Operators

Three classes, one abstract base:

Class	Description
BaseMutator	Abstract contract. Implement mutate(sequence) → str.
RandomMutator	Picks random positions, substitutes random amino acids. Baseline.
ESMGuidedMutator	Uses ESM-2 log-prob weighted sampling. Biologically smarter.
CrossoverOperator	Recombines two parent sequences. Supports single_point, two_point, uniform.
Use build_mutator(config, rng, esm_proposer) as the factory — it returns the right mutator based on config.strategy.

Mutation constraints:

max_mutations_per_sequence: hard cap on substitutions per sequence
allowed_positions: restrict mutations to specific residue indices
crossover_rate: probability that crossover actually fires (else clones returned)
bioemu.py — BioEmu Structural Inference

Abstracts all structural inference behind a single interface: infer_batch(sequences) → List[BioEmuOutput].

BioEmu is a black box. It takes a sequence and returns structural samples. It is never a scorer.

Class	Description
BaseStructuralBackend	Abstract base. Implement _run_inference(sequence). Aggregation is handled automatically.
BioEmuWrapper	Real BioEmu inference (requires GPU + BioEmu install).
MockBioEmuBackend	Deterministic synthetic outputs for testing. Activated by bioemu.mock=true.
BioEmuOutput fields (pre-aggregated across ensemble):

mean_confidence — mean pLDDT-style confidence (0–100)
confidence_std — variance across ensemble members
mean_energy / energy_std — energy proxy
mean_rg / rg_std — radius of gyration
pairwise_distance_variance — structural consistency proxy
To swap BioEmu for another model (AlphaFold, RoseTTAFold, etc.), subclass BaseStructuralBackend and implement _run_inference. Nothing else needs to change.

scoring.py — Fitness Function (including Wildtype Proximity)

Converts a BioEmuOutput into a single scalar fitness value in [0, 1]. This is the central abstraction — the GA sees only this number.

Four default component scorers, each returning [0, 1]:

Scorer	What it measures	Higher =
StabilityScorer	Mean pLDDT confidence across ensemble	More stable fold
ConsistencyScorer	Inverse pairwise distance variance	More consistent ensemble
EnergyScorer	Normalised energy proxy	Lower energy (more stable)
CompactnessScorer	Radius of gyration vs. ideal compact fold	More compact structure
WildtypeProximityScorer	Normalised Hamming identity vs. wildtype target	Closer to wildtype
The WildtypeProximityScorer is automatically injected by the pipeline when wildtype_sequence is set in config. Its weight (wildtype_proximity_weight) is appended and all other weights are renormalised — so the rest of the scoring config does not need to change.

For the landscape-optimization proof of concept, add a target-aware scorer:

from protein_optimizer import ConformationalLandscapeScorer, ScoringFunction

target_output = bioemu_backend.infer_batch([target_sequence])[0]
scoring_fn = ScoringFunction(cfg.scoring)
scoring_fn.add_component(
    ConformationalLandscapeScorer(target_output, max_states=5),
    weight=0.6,
    renormalize=True,
)
ConformationalLandscapeScorer converts each ensemble sample's distance matrix into a structural fingerprint, clusters the target ensemble into reference states, compares state occupancy distributions with Jensen-Shannon similarity, and adds a structural proximity term. This gives the GA a bounded scalar objective for "how target-like is this mutant's landscape?"

The composite fitness is a weighted sum of all components. Weights come from ScoringConfig and must sum to 1.0 (enforced at construction).

scoring_fn = ScoringFunction(cfg.scoring)

# Score one sequence
score = scoring_fn.score(bioemu_output)           # float in [0, 1]

# Score a whole population
scores = scoring_fn.score_batch(bioemu_outputs)   # List[float]

# Debug breakdown
breakdown = scoring_fn.score_with_breakdown(output)
# → {"stability": 0.81, "consistency": 0.74, "energy": 0.65, ..., "fitness": 0.75}
Adding a custom scorer (extension point):

class SASAScorer(ComponentScorer):
    name = "sasa"
    def score(self, output: BioEmuOutput) -> float:
        # your logic → return float in [0, 1]

scoring_fn.register("sasa", SASAScorer, weight=0.15, renormalize=True)
genetic_algorithm.py — GA Engine

A biology-agnostic evolutionary optimiser. It operates on:

population: List[str] — the sequences (strings)
scores: List[float] — fitness values from the scoring function
It has zero knowledge of amino acids, BioEmu, or ESM-2.

GA loop per generation:

Evaluate population via injected fitness_fn
Extract elite sequences (always survive)
Select parents (tournament or top-k)
Apply crossover to produce offspring
Apply mutation to offspring
Check convergence → stop if stale for convergence_patience generations
Key classes:

GeneticAlgorithm — main engine
TournamentSelector — sample k, keep best (default)
TopKSelector — deterministic, greedy
ConvergenceTracker — stops early if improvement < threshold for N generations
GenerationResult — snapshot of one generation (population, scores, best, diversity)
pipeline.py — Orchestration Layer

ProteinOptimizationPipeline is the single entry point. It:

Validates the config and original sequence
Constructs all sub-components (ESM-2, BioEmu, scorer, mutator, crossover, GA, tracker)
Builds the initial population (original + mutant copies)
Runs GeneticAlgorithm.run(), injecting _evaluate_population as the fitness callable
Exports results via OptimizationTracker
_evaluate_population is the only place BioEmu and scoring are ever called:

sequences → BioEmu.infer_batch() → ScoringFunction.score_batch() → List[float] → GA
analysis.py — Tracking, Export, and Stage Warmth Reporting

OptimizationTracker attaches to the GA as a callback and records every generation.

Tracks:

Best score and sequence per generation
Mean score and population diversity trajectory
Mutation history vs. original sequence (positions changed, from/to)
Periodic checkpoints every N generations
Exports:

results/optimization_results.json — full run data
results/generation_summary.csv — one row per generation
results/stage_warmth_report.json — per-stage warmth snapshots (wildtype mode only)
StageReporter is the second major class here. It fires at the end of each stage (fifth by default) and prints the warm/cold progress bar to stdout. It also exports the full stage breakdown to JSON.

tracker.score_trajectory    # List[float] — plot convergence
tracker.diversity_trajectory
tracker.summary_report()    # human-readable one-page summary
Data Flow

original_sequence
       │
       ▼
build_initial_population()
  [original] + [mutant_1, mutant_2, ..., mutant_N-1]
       │
       ▼ (each generation)
┌──────────────────────────────────────┐
│ 1. fitness_fn(population)            │
│      BioEmu.infer_batch(sequences)   │
│          → List[BioEmuOutput]        │
│      ScoringFunction.score_batch()   │
│          → List[float]               │
│                                      │
│ 2. Extract elite (top elite_size)    │
│                                      │
│ 3. Select parents                    │
│      TournamentSelector              │
│                                      │
│ 4. CrossoverOperator                 │
│      two_point / single_point        │
│                                      │
│ 5. Mutator.mutate_population()       │
│      ESM-2 proposes substitutions    │
│      Weighted sampling from top-k    │
│                                      │
│ 6. OptimizationTracker.on_gen()      │
│      Record stats, checkpoint        │
│                                      │
│ 7. ConvergenceTracker.update()       │
│      Stop if stale N generations     │
└──────────────────────────────────────┘
       │
       ▼
OptimizationResult
  .best_sequence
  .best_score
  .tracker  (full history, export paths)
Project Structure

protein_optimizer/
├── protein_optimizer/          # Main package
│   ├── __init__.py             # Public API surface
│   ├── config.py               # All config dataclasses
│   ├── esm.py                  # ESM-2 mutation proposals
│   ├── mutation.py             # Random + ESM-guided mutators, crossover
│   ├── bioemu.py               # BioEmu wrapper + mock backend
│   ├── scoring.py              # Composite fitness scoring function
│   ├── genetic_algorithm.py    # GA engine (biology-agnostic)
│   ├── pipeline.py             # Orchestration layer
│   └── analysis.py             # Tracking, logging, export
│
├── config/
│   └── default.yaml            # All tunable parameters
│
├── scripts/
│   └── run_optimization.py     # Python API usage examples
│
├── tests/
│   ├── test_mutation.py        # Mutator + crossover tests
│   ├── test_scoring.py         # Scoring component tests
│   └── test_ga.py              # GA + full pipeline integration tests
│
├── main.py                     # CLI entry point
└── pyproject.toml
Installation

Core dependencies (no BioEmu):

pip install -e ".[dev]"
This installs: numpy, pyyaml, torch, transformers (for ESM-2), pytest.

BioEmu must be installed separately — it requires CUDA and large model weights:

pip install "bioemu[cuda]"
# Reference: https://github.com/microsoft/bioemu
On a GPU VM, just run the setup script — it does all of the above plus a smoke test (see Quick Start):

bash setup_vm.sh
No GPU / testing without models:

Set bioemu.mock=true in your config or pass --mock on the CLI. Set mutation.strategy=random to skip ESM-2 loading. The full GA loop runs entirely on CPU with no model weights.

Configuration

All parameters live in config/default.yaml. Every field maps directly to a typed dataclass in config.py.

original_sequence: "MKTLLILAVLCLGFAQASGNIERPIDGFHGDLQ"

bioemu:
  mock: false          # true = synthetic outputs, no GPU
  num_samples: 10      # ensemble size per sequence
  batch_size: 4

scoring:
  stability_weight: 0.40
  consistency_weight: 0.30
  energy_weight: 0.20
  diversity_penalty_weight: 0.10
  # Weights must sum to 1.0

mutation:
  strategy: "esm_guided"   # "random" or "esm_guided"
  max_mutations_per_sequence: 3
  allowed_positions: null   # null = all positions

ga:
  population_size: 50
  max_generations: 100
  elite_fraction: 0.10
  selection_strategy: "tournament"
  convergence_patience: 10

logging:
  output_dir: "results"
  export_format: "both"    # "json", "csv", or "both"
Running the Optimizer

CLI

For the main directed-evolution workflow (BioEmu LLR, fit-to-target, output files), see Quick Start: GPU VM. The examples below cover the GA pipeline and no-GPU testing.
# Directed evolution toward a healthy protein (real BioEmu)
python main.py --config config/evolutionary.yaml \
    --sequence DEFECTIVE_SEQUENCE \
    --healthy-sequence HEALTHY_SEQUENCE \
    --set bioemu.num_samples=100

# Directed evolution toward a target LLR value
python main.py --config config/evolutionary.yaml \
    --sequence DEFECTIVE_SEQUENCE --target -1.5

# No GPU (mock BioEmu + random mutations)
python main.py --config config/default.yaml --mock --random-mutations

# Wildtype recovery experiment (no GPU)
python main.py --config config/default.yaml --mock --random-mutations \
    --set wildtype_sequence=MKTLLILAVLCLGFAQAS \
    --set original_sequence=ACDEFGHIKLMNPQRSTV

# Full run
python main.py --config config/default.yaml

# Override any config field at runtime
python main.py --config config/default.yaml \
    --set ga.population_size=100 \
    --set ga.max_generations=200 \
    --set scoring.stability_weight=0.5 \
    --set scoring.consistency_weight=0.2 \
    --set scoring.energy_weight=0.2 \
    --set scoring.diversity_penalty_weight=0.1

# Provide sequence directly
python main.py --sequence MKTLLILAVLCLGFAQAS --mock
Python API

from protein_optimizer import OptimizationConfig, ProteinOptimizationPipeline

cfg = OptimizationConfig.from_yaml("config/default.yaml")
cfg.original_sequence = "MKTLLILAVLCLGFAQAS"
cfg.bioemu.mock = True      # remove for real inference

pipeline = ProteinOptimizationPipeline(cfg)
result = pipeline.run()

print(result.best_sequence)
print(result.best_score)
print(result.tracker.summary_report())
Wildtype recovery mode:

cfg = OptimizationConfig()
cfg.original_sequence = "ACDEFGHIKLMNPQRSTVWY"  # the "bad" starting protein
cfg.wildtype_sequence  = "MKTLLILAVLCLGFAQAS"   # the known target
cfg.bioemu.mock = True
cfg.ga.n_stages = 5    # fifths by default

pipeline = ProteinOptimizationPipeline(cfg)
result = pipeline.run()
# Automatically prints warm/cold stage report during the run.
# Stage snapshots saved to results/stage_warmth_report.json
See scripts/run_optimization.py for more examples including custom scorers and score breakdowns.

Extension Points

Swap BioEmu for another structural model

from protein_optimizer.bioemu import BaseStructuralBackend, BioEmuOutput

class AlphaFoldBackend(BaseStructuralBackend):
    def _run_inference(self, sequence: str) -> BioEmuOutput:
        # call AlphaFold, populate BioEmuOutput fields
        ...

pipeline = ProteinOptimizationPipeline(cfg, bioemu_backend=AlphaFoldBackend(cfg.bioemu))
Add a custom scoring component

from protein_optimizer.scoring import ComponentScorer, ScoringFunction
from protein_optimizer.bioemu import BioEmuOutput

class SASAScorer(ComponentScorer):
    name = "sasa"
    def score(self, output: BioEmuOutput) -> float:
        # compute from output.samples[i].sasa
        return my_sasa_score

scoring_fn = ScoringFunction(cfg.scoring)
scoring_fn.register("sasa", SASAScorer, weight=0.15, renormalize=True)

pipeline = ProteinOptimizationPipeline(cfg, scoring_fn=scoring_fn)
Plug in a custom mutator

from protein_optimizer.mutation import BaseMutator

class ConservativeMutator(BaseMutator):
    """Only allow mutations at surface-exposed residues."""
    def mutate(self, sequence: str) -> str:
        ...

pipeline._mutator = ConservativeMutator(cfg.mutation, rng)
Attach GA callbacks

def log_to_wandb(result):
    wandb.log({"best_score": result.best_score, "gen": result.generation})

ga = GeneticAlgorithm(..., callbacks=[tracker.on_generation, log_to_wandb])
Testing

# Run all 71 tests
python3 -m pytest tests/ -v

# With coverage
python3 -m pytest tests/ --cov=protein_optimizer --cov-report=term-missing
All tests run fully offline (no GPU, no model weights) using MockBioEmuBackend and RandomMutator. The mock backend hashes each sequence to produce deterministic but distinct outputs, so the full GA loop is exercised end-to-end.

Test coverage:

test_mutation.py — RandomMutator, CrossoverOperator, allowed_positions, build_mutator factory
test_scoring.py — All 4 component scorers, composite ScoringFunction, custom scorer registration, weight validation
test_ga.py — ConvergenceTracker, selectors, full GA loop, population sizing, pipeline integration, input validation
test_wildtype.py — WildtypeProximityScorer, warmth_label scale, StageReporter boundaries + snapshots + export, full wildtype pipeline integration
Design Principles

Principle	Implementation
GA is biology-agnostic	GeneticAlgorithm only imports from mutation.py and config.py. No amino acid knowledge.
BioEmu is a black box	BaseStructuralBackend abstracts all inference. Scoring never calls BioEmu directly.
Scoring is pluggable	ComponentScorer ABC + ScoringFunction.register() — add axes without modifying core code.
Config-driven	Every tunable value lives in OptimizationConfig. No hard-coded constants in logic files.
Lazy model loading	ESM-2 and BioEmu load on first use. Importing the package has zero GPU cost.
Modular imports	Each module can be imported independently. No circular dependencies.
Mock-first testability	Full pipeline is testable on CPU with no model weights via MockBioEmuBackend.
