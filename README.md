# Protein Optimizer

An industrial-grade evolutionary optimization framework for protein sequence design.

Combines an **evolutionary search** with **BioEmu** structural-ensemble inference and **Meta ESM-2** guided mutations to evolve protein sequences toward a target. Progress is measured by comparing each sequence's **BioEmu LLR** (a stability/plausibility parameter) against a reference and a goal — see [How We Measure Progress](#how-we-measure-progress-llr-comparison).

> **For AI assistants reading this:** This is a pure inference + optimization system. We are NOT training any models. ESM-2 is used only to propose biologically plausible mutations. BioEmu is used only to evaluate structural quality. The GA drives the search. Every component is modular and swappable.

---

## Table of Contents

- [Quick Start: GPU VM (Directed Evolution)](#quick-start-gpu-vm-directed-evolution)
- [How We Measure Progress: LLR Comparison](#how-we-measure-progress-llr-comparison)
- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [Component Breakdown](#component-breakdown)
- [Data Flow](#data-flow)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Optimizer](#running-the-optimizer)
- [Extension Points](#extension-points)
- [Testing](#testing)
- [Design Principles](#design-principles)

---

## Quick Start: GPU VM (Directed Evolution)

This is the end-to-end workflow for running the real BioEmu pipeline on a GPU
rental VM (Lambda / RunPod / Vast). It evolves a **defective** protein sequence
so its BioEmu parameter (LLR) approaches a **goal** — either a number you choose
or the LLR of a **healthy** protein.

### 1. Set up the VM (one time)

```bash
# On the VM, in a fresh terminal:
git clone https://github.com/ibrahimaasim77/Test.git
cd Test
bash setup_vm.sh
```

`setup_vm.sh` checks the GPU, creates a `.venv`, installs BioEmu + dependencies
(`pip install "bioemu[cuda]"`), and runs a mock smoke test. No conda and **no
HuggingFace token** are needed — model weights are public and download once.
The "set a HF_TOKEN" warning is already suppressed.

### 2. Activate the environment (every new terminal)

```bash
source .venv/bin/activate
```

### 3. Run it

**Fit-to-target toward a healthy protein** (recommended — the directed-evolution story):

```bash
python main.py --config config/evolutionary.yaml \
    --sequence   DEFECTIVE_SEQUENCE \
    --healthy-sequence HEALTHY_SEQUENCE \
    --set bioemu.num_samples=100 \
    --set ga.population_size=20 \
    --set ga.max_generations=3
```

**Fit-to-target toward a chosen LLR value:**

```bash
python main.py --config config/evolutionary.yaml \
    --sequence DEFECTIVE_SEQUENCE \
    --target -1.5 \
    --set bioemu.num_samples=100
```

**Maximise mode** (no goal — just find the highest LLR): omit `--target` and
`--healthy-sequence`.

### 4. Practice sequences

A ready-made recovery demo (the healthy protein with 5 point mutations):

```bash
# DEFECTIVE (goes after --sequence):
MKTLLGLAVLCLGFAQASGNPERPIDGFHGDLQSLDKAMFESRHITAYIEWLEELRQRQTAATGGKRQ

# HEALTHY (goes after --healthy-sequence):
MKTLLILAVLCLGFAQASGNIERPIDGFHGDLQSLIKAMFESRHITAYIEQLEELRQRQTAATGGMRQ
```

Or run the bundled script: `bash practice.sh` (real BioEmu) or
`bash practice.sh mock` (synthetic, no GPU, for rehearsing).

### 5. How long it takes

Total time ≈ **(number of sequences) × (time for one BioEmu run)**.
Sequences scored = `2 (reference + goal) + population_size × max_generations`.
`num_samples` is the heavy knob. **Time the first sequence and multiply.**
For a ~5-minute smoke test:

```bash
python main.py --config config/evolutionary.yaml \
    --sequence MKTLLGLAVLCLGFAQASGNPERPIDGFHGDLQSLDKAMFESRHITAYIEWLEELRQRQTAATGGKRQ \
    --healthy-sequence MKTLLILAVLCLGFAQASGNIERPIDGFHGDLQSLIKAMFESRHITAYIEQLEELRQRQTAATGGMRQ \
    --set bioemu.num_samples=5 --set ga.population_size=4 --set ga.max_generations=1
```

### 6. Output — what you get and where

**Printed to screen** (save it with `... 2>&1 | tee my_results.txt`):

```
Defective sequence parameter :  -3.9915   (start)
Target (goal) parameter      :  -2.0000   (what we optimise toward)
Best engineered parameter    :  -2.3803   [closer to goal]
Distance to goal: 1.9915  →  0.3803   (closed +1.6111)
Best sequence: ...
```

**Trajectory files** (BioEmu's real `.xtc` + `.pdb`, saved automatically) under
`results/trajectories/` — full path on the VM
`/workspace/Test/results/trajectories/`:

```
results/trajectories/reference/samples.xtc    + topology.pdb
results/trajectories/best_mutant/samples.xtc  + topology.pdb
```

List them: `ls -R results/trajectories/`

### 7. Download files to your laptop

- **RunPod / Vast:** use the dashboard's web file browser → navigate to
  `/workspace/Test/results/`.
- **SCP** (run from your laptop, not the VM):
  ```bash
  scp -P <port> root@<vm-ip>:/workspace/Test/results/trajectories/best_mutant/samples.xtc .
  ```
  The provider's "Connect" page gives `<vm-ip>` and `<port>`.

### Useful flags

| Flag | Meaning |
|---|---|
| `--sequence SEQ` | The defective / starting sequence (single-letter AA codes) |
| `--healthy-sequence SEQ` | Healthy protein; its LLR becomes the goal |
| `--target LLR` | Goal LLR as a number (used if no `--healthy-sequence`) |
| `--set bioemu.num_samples=N` | Conformations sampled per sequence (heavy) |
| `--set ga.population_size=N` | Candidates generated per round |
| `--set ga.max_generations=N` | Number of rounds |
| `--mock` | Synthetic BioEmu (no GPU) for testing |
| `--random-mutations` | Skip ESM-2; use random mutations |
| `--verbose` | Print the per-round scoring tables |

After the first run caches the weights, you can go fully offline:
`export HF_HUB_OFFLINE=1`.

---

## How We Measure Progress: LLR Comparison

**This is the core of the project, and it replaces the old "warmer / colder"
sequence-identity indicator.** We no longer score a mutant by how many residues
it shares with a known wildtype. Instead we judge every sequence by a single
physical number that BioEmu gives us: its **LLR (log-likelihood ratio)** — a
stability/plausibility parameter derived from the structural ensemble (internally
BioEmu's energy proxy, negated so that **higher = more favourable**).

Why this is better:

- The wildtype warm/cold scale only worked if you already knew the "right"
  answer. LLR needs no answer key — it scores any sequence on its own merits.
- It's a continuous physical quantity, not a string-matching percentage, so the
  search can find a *different* sequence that behaves like the goal.

### The three numbers we compare

| Number | What it is |
|---|---|
| **Reference LLR** | The LLR of the starting (defective) sequence — the baseline. |
| **Goal LLR** | The LLR we aim for: either a value you pass with `--target`, or the LLR of a **healthy** protein you pass with `--healthy-sequence`. |
| **Best engineered LLR** | The LLR of the best sequence the search found. |

### Two ways to run

**Fit-to-target (directed evolution).** Give the search a goal LLR. A mutant is
"better" when its LLR is **closer to the goal** than the reference is. We report
the *distance to the goal* shrinking:

```
Defective sequence parameter : -3.9915   (start — reference LLR)
Target (goal) parameter      : -2.0000   (the goal LLR)
Best engineered parameter    : -2.3803   [closer to goal]
Distance to goal: 1.9915  →  0.3803   (closed +1.6111)
```

**Maximise (no goal).** Omit `--target` and `--healthy-sequence`. A mutant is
"better" when its LLR simply **beats the reference**:

```
Reference LLR : -3.9915
Best LLR      : -1.8402   [improved]
LLR change    : +2.1513   (higher = more favourable)
```

Every round prints a table ranked by LLR, with a second column showing each
candidate's distance-to-goal (fit-to-target) or change-vs-reference (maximise).
Run with `--verbose` to see those tables live.

> **Legacy note:** a sequence-identity `WildtypeProximityScorer` and the warm/cold
> `StageReporter` still ship in `scoring.py` / `analysis.py` and are covered by
> tests, but they belong to the older GA pipeline. The LLR comparison above is
> the approach we present and run.

---

## Problem Statement

Given:
- An **original protein sequence** (single-letter amino acid codes)
- A **target conformational landscape** represented by a BioEmu structural ensemble
- A **BioEmu model** that generates structural ensembles for candidate sequences
- A **pretrained ESM-2 model** that predicts biologically plausible amino acid substitutions

Goal: Find a mutated sequence whose **equilibrium ensemble** matches the target landscape, using evolutionary search.

We are NOT training models. We are doing inference-guided optimization.

---

## System Architecture

```
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
```

---

## Component Breakdown

### `config.py` — Unified Configuration

All modules are driven by a single `OptimizationConfig` dataclass. No magic numbers anywhere else.

```python
cfg = OptimizationConfig.from_yaml("config/default.yaml")
# Or build programmatically:
cfg = OptimizationConfig()
cfg.original_sequence = "MKTLLILAVLCLGFAQ"
cfg.bioemu.mock = True          # synthetic outputs, no GPU
cfg.ga.population_size = 50
```

Sub-configs: `ESM2Config`, `BioEmuConfig`, `ScoringConfig`, `MutationConfig`, `GAConfig`, `LoggingConfig`.

---

### `esm.py` — ESM-2 Mutation Proposal Engine

Wraps Meta's ESM-2 (via HuggingFace) as a **mutation oracle**. Given a sequence and a target position, it masks that position and uses ESM-2's masked language model to return the top-k most biologically plausible amino acid substitutions, ranked by log-probability.

**Key class:** `ESM2MutationProposer`

```python
proposer = ESM2MutationProposer(cfg.esm2)
candidates = proposer.propose(sequence="MKTLL...", positions=[5, 12])
# candidates: {5: [MutationCandidate(pos=5, A→G, log_prob=-0.3), ...], ...}
```

**Important:** ESM-2 is only ever called inside the mutation step. It never touches scoring or the GA directly. It is lazy-loaded — no GPU cost unless `strategy="esm_guided"`.

---

### `mutation.py` — Mutation and Crossover Operators

Three classes, one abstract base:

| Class | Description |
|---|---|
| `BaseMutator` | Abstract contract. Implement `mutate(sequence) → str`. |
| `RandomMutator` | Picks random positions, substitutes random amino acids. Baseline. |
| `ESMGuidedMutator` | Uses ESM-2 log-prob weighted sampling. Biologically smarter. |
| `CrossoverOperator` | Recombines two parent sequences. Supports `single_point`, `two_point`, `uniform`. |

Use `build_mutator(config, rng, esm_proposer)` as the factory — it returns the right mutator based on `config.strategy`.

**Mutation constraints:**
- `max_mutations_per_sequence`: hard cap on substitutions per sequence
- `allowed_positions`: restrict mutations to specific residue indices
- `crossover_rate`: probability that crossover actually fires (else clones returned)

---

### `bioemu.py` — BioEmu Structural Inference

Abstracts all structural inference behind a single interface: `infer_batch(sequences) → List[BioEmuOutput]`.

**BioEmu is a black box.** It takes a sequence and returns structural samples. It is never a scorer.

| Class | Description |
|---|---|
| `BaseStructuralBackend` | Abstract base. Implement `_run_inference(sequence)`. Aggregation is handled automatically. |
| `BioEmuWrapper` | Real BioEmu inference (requires GPU + BioEmu install). |
| `MockBioEmuBackend` | Deterministic synthetic outputs for testing. Activated by `bioemu.mock=true`. |

**`BioEmuOutput` fields** (pre-aggregated across ensemble):
- `mean_confidence` — mean pLDDT-style confidence (0–100)
- `confidence_std` — variance across ensemble members
- `mean_energy` / `energy_std` — energy proxy
- `mean_rg` / `rg_std` — radius of gyration
- `pairwise_distance_variance` — structural consistency proxy

To swap BioEmu for another model (AlphaFold, RoseTTAFold, etc.), subclass `BaseStructuralBackend` and implement `_run_inference`. Nothing else needs to change.

---

### `scoring.py` — Fitness Function (including Wildtype Proximity)

Converts a `BioEmuOutput` into a single scalar fitness value in `[0, 1]`. This is the **central abstraction** — the GA sees only this number.

Four default component scorers, each returning `[0, 1]`:

| Scorer | What it measures | Higher = |
|---|---|---|
| `StabilityScorer` | Mean pLDDT confidence across ensemble | More stable fold |
| `ConsistencyScorer` | Inverse pairwise distance variance | More consistent ensemble |
| `EnergyScorer` | Normalised energy proxy | Lower energy (more stable) |
| `CompactnessScorer` | Radius of gyration vs. ideal compact fold | More compact structure |
| `WildtypeProximityScorer` | Normalised Hamming identity vs. wildtype target | Closer to wildtype |

The `WildtypeProximityScorer` is **automatically injected** by the pipeline when `wildtype_sequence` is set in config. Its weight (`wildtype_proximity_weight`) is appended and all other weights are renormalised — so the rest of the scoring config does not need to change.

For the landscape-optimization proof of concept, add a target-aware scorer:

```python
from protein_optimizer import ConformationalLandscapeScorer, ScoringFunction

target_output = bioemu_backend.infer_batch([target_sequence])[0]
scoring_fn = ScoringFunction(cfg.scoring)
scoring_fn.add_component(
    ConformationalLandscapeScorer(target_output, max_states=5),
    weight=0.6,
    renormalize=True,
)
```

`ConformationalLandscapeScorer` converts each ensemble sample's distance matrix into a structural fingerprint, clusters the target ensemble into reference states, compares state occupancy distributions with Jensen-Shannon similarity, and adds a structural proximity term. This gives the GA a bounded scalar objective for "how target-like is this mutant's landscape?"

The composite fitness is a **weighted sum** of all components. Weights come from `ScoringConfig` and must sum to 1.0 (enforced at construction).

```python
scoring_fn = ScoringFunction(cfg.scoring)

# Score one sequence
score = scoring_fn.score(bioemu_output)           # float in [0, 1]

# Score a whole population
scores = scoring_fn.score_batch(bioemu_outputs)   # List[float]

# Debug breakdown
breakdown = scoring_fn.score_with_breakdown(output)
# → {"stability": 0.81, "consistency": 0.74, "energy": 0.65, ..., "fitness": 0.75}
```

**Adding a custom scorer** (extension point):
```python
class SASAScorer(ComponentScorer):
    name = "sasa"
    def score(self, output: BioEmuOutput) -> float:
        # your logic → return float in [0, 1]

scoring_fn.register("sasa", SASAScorer, weight=0.15, renormalize=True)
```

---

### `evolutionary_search.py` — LLR-Driven Directed Evolution (primary run path)

`BudgetedEvolutionarySearch` is what `main.py` actually runs. It works directly
in **LLR space** instead of the composite GA fitness:

1. Score the reference (defective) sequence with BioEmu → **reference LLR**.
2. If `--healthy-sequence` / `--target` is set, score it once → **goal LLR**.
3. Each round: propose `population_size` mutants (ESM-2 or random), score them
   with BioEmu, rank by LLR, keep the best — in fit-to-target mode "best" means
   closest to the goal LLR; in maximise mode it means highest LLR.
4. Repeat for `max_generations` rounds, then report the comparison and save
   trajectory files for the reference and the best mutant.

This is the module that implements the [LLR comparison](#how-we-measure-progress-llr-comparison)
described above. `scoring.py`, `genetic_algorithm.py`, and `pipeline.py` below
are the older composite-fitness GA path, kept for reference and tests.

---

### `genetic_algorithm.py` — GA Engine

A **biology-agnostic** evolutionary optimiser. It operates on:
- `population: List[str]` — the sequences (strings)
- `scores: List[float]` — fitness values from the scoring function

It has zero knowledge of amino acids, BioEmu, or ESM-2.

**GA loop per generation:**
1. Evaluate population via injected `fitness_fn`
2. Extract elite sequences (always survive)
3. Select parents (tournament or top-k)
4. Apply crossover to produce offspring
5. Apply mutation to offspring
6. Check convergence → stop if stale for `convergence_patience` generations

**Key classes:**
- `GeneticAlgorithm` — main engine
- `TournamentSelector` — sample k, keep best (default)
- `TopKSelector` — deterministic, greedy
- `ConvergenceTracker` — stops early if improvement < threshold for N generations
- `GenerationResult` — snapshot of one generation (population, scores, best, diversity)

---

### `pipeline.py` — Orchestration Layer

`ProteinOptimizationPipeline` is the single entry point. It:
1. Validates the config and original sequence
2. Constructs all sub-components (ESM-2, BioEmu, scorer, mutator, crossover, GA, tracker)
3. Builds the initial population (original + mutant copies)
4. Runs `GeneticAlgorithm.run()`, injecting `_evaluate_population` as the fitness callable
5. Exports results via `OptimizationTracker`

`_evaluate_population` is the only place BioEmu and scoring are ever called:
```
sequences → BioEmu.infer_batch() → ScoringFunction.score_batch() → List[float] → GA
```

---

### `analysis.py` — Tracking and Export

`OptimizationTracker` attaches to the GA as a callback and records every generation.

Tracks:
- Best score and sequence per generation
- Mean score and population diversity trajectory
- Mutation history vs. original sequence (positions changed, from/to)
- Periodic checkpoints every N generations

Exports:
- `results/optimization_results.json` — full run data
- `results/generation_summary.csv` — one row per generation
- `results/stage_warmth_report.json` — per-stage snapshots (legacy wildtype mode only)

`StageReporter` is a **legacy** class kept for the old wildtype-recovery experiment:
it prints the warm/cold sequence-identity progress bar. The current workflow does
not use it — progress is reported as the [LLR comparison](#how-we-measure-progress-llr-comparison)
by `evolutionary_search.py`.

```python
tracker.score_trajectory    # List[float] — plot convergence
tracker.diversity_trajectory
tracker.summary_report()    # human-readable one-page summary
```

---

## Data Flow

```
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
```

---

## Project Structure

The project now lives at the **repository root** (`Test/`) so the README and all
files are visible on the GitHub landing page — no more digging into a subfolder.

```
Test/                              # repo root (this README renders here)
├── protein_optimizer/             # Main Python package
│   ├── __init__.py                # Public API surface
│   ├── config.py                  # All config dataclasses
│   ├── esm.py                     # ESM-2 mutation proposals
│   ├── mutation.py                # Random + ESM-guided mutators, crossover
│   ├── bioemu.py                  # BioEmu wrapper + mock backend (LLR source)
│   ├── scoring.py                 # Composite fitness scoring (legacy GA path)
│   ├── genetic_algorithm.py       # GA engine (biology-agnostic)
│   ├── evolutionary_search.py     # BudgetedEvolutionarySearch — the LLR-driven run
│   ├── pipeline.py                # Orchestration layer (legacy GA path)
│   └── analysis.py                # Tracking, logging, export
│
├── config/
│   ├── default.yaml               # GA-pipeline parameters
│   └── evolutionary.yaml          # Directed-evolution / LLR-search parameters
│
├── scripts/
│   ├── run_optimization.py        # Python API usage examples
│   └── bioemu_gif_maker.py        # Render a GIF from BioEmu trajectories
│
├── tests/
│   ├── test_mutation.py           # Mutator + crossover tests
│   ├── test_scoring.py            # Scoring component tests
│   ├── test_ga.py                 # GA + full pipeline integration tests
│   └── test_wildtype.py           # Legacy wildtype-proximity / stage tests
│
├── main.py                        # CLI entry point (LLR evolutionary search)
├── practice.sh                    # Ready-made recovery demo
├── setup_vm.sh                    # One-time GPU VM setup
└── pyproject.toml
```

> `AI-Harness/` also sits at the repo root as a separate git submodule — it is
> unrelated to the protein optimizer.

---

## Installation

**Core dependencies (no BioEmu):**
```bash
pip install -e ".[dev]"
```

This installs: `numpy`, `pyyaml`, `torch`, `transformers` (for ESM-2), `pytest`.

**BioEmu** must be installed separately — it requires CUDA and large model weights:
```bash
pip install "bioemu[cuda]"
# Reference: https://github.com/microsoft/bioemu
```

**On a GPU VM, just run the setup script** — it does all of the above plus a
smoke test (see [Quick Start](#quick-start-gpu-vm-directed-evolution)):
```bash
bash setup_vm.sh
```

**No GPU / testing without models:**

Set `bioemu.mock=true` in your config or pass `--mock` on the CLI. Set `mutation.strategy=random` to skip ESM-2 loading. The full GA loop runs entirely on CPU with no model weights.

---

## Configuration

All parameters live in `config/default.yaml`. Every field maps directly to a typed dataclass in `config.py`.

```yaml
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
```

---

## Running the Optimizer

### CLI

> For the main directed-evolution workflow (BioEmu LLR, fit-to-target, output
> files), see [Quick Start: GPU VM](#quick-start-gpu-vm-directed-evolution).
> The examples below cover the GA pipeline and no-GPU testing.

```bash
# Directed evolution toward a healthy protein (real BioEmu)
python main.py --config config/evolutionary.yaml \
    --sequence DEFECTIVE_SEQUENCE \
    --healthy-sequence HEALTHY_SEQUENCE \
    --set bioemu.num_samples=100

# Directed evolution toward a target LLR value
python main.py --config config/evolutionary.yaml \
    --sequence DEFECTIVE_SEQUENCE --target -1.5

# No GPU (mock BioEmu + random mutations) — rehearse the LLR search on CPU
python main.py --config config/evolutionary.yaml --mock --random-mutations

# Legacy GA pipeline (composite fitness, not LLR comparison)
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
```

### Python API

```python
from protein_optimizer import OptimizationConfig, ProteinOptimizationPipeline

cfg = OptimizationConfig.from_yaml("config/default.yaml")
cfg.original_sequence = "MKTLLILAVLCLGFAQAS"
cfg.bioemu.mock = True      # remove for real inference

pipeline = ProteinOptimizationPipeline(cfg)
result = pipeline.run()

print(result.best_sequence)
print(result.best_score)
print(result.tracker.summary_report())
```

**Legacy wildtype recovery mode** (sequence-identity warm/cold — superseded by the
[LLR comparison](#how-we-measure-progress-llr-comparison), kept only for reference):

```python
cfg = OptimizationConfig()
cfg.original_sequence = "ACDEFGHIKLMNPQRSTVWY"  # the "bad" starting protein
cfg.wildtype_sequence  = "MKTLLILAVLCLGFAQAS"   # the known target
cfg.bioemu.mock = True
cfg.ga.n_stages = 5    # fifths by default

pipeline = ProteinOptimizationPipeline(cfg)
result = pipeline.run()
# Prints the old warm/cold stage report; snapshots → results/stage_warmth_report.json
```

See `scripts/run_optimization.py` for more examples including custom scorers and score breakdowns.

---

## Extension Points

### Swap BioEmu for another structural model

```python
from protein_optimizer.bioemu import BaseStructuralBackend, BioEmuOutput

class AlphaFoldBackend(BaseStructuralBackend):
    def _run_inference(self, sequence: str) -> BioEmuOutput:
        # call AlphaFold, populate BioEmuOutput fields
        ...

pipeline = ProteinOptimizationPipeline(cfg, bioemu_backend=AlphaFoldBackend(cfg.bioemu))
```

### Add a custom scoring component

```python
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
```

### Plug in a custom mutator

```python
from protein_optimizer.mutation import BaseMutator

class ConservativeMutator(BaseMutator):
    """Only allow mutations at surface-exposed residues."""
    def mutate(self, sequence: str) -> str:
        ...

pipeline._mutator = ConservativeMutator(cfg.mutation, rng)
```

### Attach GA callbacks

```python
def log_to_wandb(result):
    wandb.log({"best_score": result.best_score, "gen": result.generation})

ga = GeneticAlgorithm(..., callbacks=[tracker.on_generation, log_to_wandb])
```

---

## Testing

```bash
# Run all 76 tests
python3 -m pytest tests/ -v

# With coverage
python3 -m pytest tests/ --cov=protein_optimizer --cov-report=term-missing
```

All tests run fully offline (no GPU, no model weights) using `MockBioEmuBackend` and `RandomMutator`. The mock backend hashes each sequence to produce deterministic but distinct outputs, so the full GA loop is exercised end-to-end.

Test coverage:
- `test_mutation.py` — RandomMutator, CrossoverOperator, allowed_positions, build_mutator factory
- `test_scoring.py` — All 4 component scorers, composite ScoringFunction, custom scorer registration, weight validation
- `test_ga.py` — ConvergenceTracker, selectors, full GA loop, population sizing, pipeline integration, input validation
- `test_wildtype.py` — WildtypeProximityScorer, warmth_label scale, StageReporter boundaries + snapshots + export, full wildtype pipeline integration

---

## Design Principles

| Principle | Implementation |
|---|---|
| GA is biology-agnostic | `GeneticAlgorithm` only imports from `mutation.py` and `config.py`. No amino acid knowledge. |
| BioEmu is a black box | `BaseStructuralBackend` abstracts all inference. Scoring never calls BioEmu directly. |
| Scoring is pluggable | `ComponentScorer` ABC + `ScoringFunction.register()` — add axes without modifying core code. |
| Config-driven | Every tunable value lives in `OptimizationConfig`. No hard-coded constants in logic files. |
| Lazy model loading | ESM-2 and BioEmu load on first use. Importing the package has zero GPU cost. |
| Modular imports | Each module can be imported independently. No circular dependencies. |
| Mock-first testability | Full pipeline is testable on CPU with no model weights via `MockBioEmuBackend`. |
