# EvoEmu

**In-silico directed evolution for proteins.** EvoEmu evolves a protein sequence
toward a target *conformational landscape* — not just a single static fold — by
pairing a **genetic algorithm** with **Meta ESM-2** (to propose biologically
plausible mutations) and **Microsoft BioEmu** (to score each candidate's
structural ensemble).

It ships with two ways to drive it:

- **A web app** — a clean dashboard with live 3D structure prediction (ESMFold),
  a real-time search feed, and ranked results.
- **A command-line tool** — for scripted / GPU-VM runs.

> This is a pure **inference + optimization** system. No models are trained.
> ESM-2 only proposes mutations, BioEmu only evaluates structures, and the
> genetic algorithm drives the search.

---

## What it does

You give EvoEmu a starting protein sequence and a **target**. It runs a budgeted
genetic search:

1. **Generate** ~N candidate mutants from the current population (ESM-2 guided).
2. **Score** each candidate with BioEmu — a log-likelihood ratio (LLR) between
   two structural states.
3. **Select** the top-20 elite.
4. **Recombine** the elite into the next generation, and repeat.

Over successive rounds the population's BioEmu score is driven toward your goal.

### Two optimization modes

| Mode | You provide | EvoEmu optimizes toward |
|------|-------------|-------------------------|
| **Fit to LLR** | Two state populations `p₁` / `p₂` (%) | the target `LLR = ln(p₁/p₂)` |
| **Fit to Structure** | A target `.pdb` structure | the sequence extracted from that structure |

---

## Quick start (web app)

The web app runs everywhere, including without a GPU (it uses synthetic BioEmu
scoring by default for fast, reproducible demos).

```bash
cd protein_optimizer

# install the web dependencies
pip install fastapi uvicorn numpy pyyaml torch transformers

# start the server
python3 server.py
```

Then open **http://localhost:8000**.

In the UI you can:

- Paste a sequence and watch it fold live (ESMFold via the ESM Atlas API).
- Pick a mode, set your target, and tune population size / generations.
- Hit **Run Optimization** and follow the genetic search in real time.
- Inspect the best sequence, the gap-closed metric, and the top-10 mutants.

---

## Quick start (command line)

```bash
cd protein_optimizer

# evolve toward a target LLR / healthy reference (synthetic scoring, no GPU)
python3 main.py --config config/evolutionary.yaml --mock --random-mutations

# override the starting sequence
python3 main.py --config config/evolutionary.yaml \
    --sequence MKTLLILAVLCLGFAQASG... --mock

# fit-to-target: optimize toward a healthy protein's BioEmu parameter
python3 main.py --config config/evolutionary.yaml \
    --sequence <DEFECTIVE> --healthy-sequence <HEALTHY>
```

A ready-made demo run is in `protein_optimizer/practice.sh`:

```bash
cd protein_optimizer
bash practice.sh mock   # synthetic, runs anywhere
bash practice.sh        # real BioEmu (run this on a GPU VM)
```

For the full GPU-VM / real-BioEmu workflow, see
[`protein_optimizer/README.md`](protein_optimizer/README.md).

---

## How it's built

```
EvoEmu
├── Web UI            frontend/index.html   — dashboard, 3D viewers, live feed
├── Web server        server.py             — FastAPI; /api/run, /api/fold, SSE stream
├── CLI               main.py               — argparse entry point
├── Search engine     protein_optimizer/    — genetic algorithm, BioEmu, ESM-2 mutation
└── Config            config/*.yaml         — population size, generations, samples, target
```

### Web API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serves the dashboard |
| `/api/run` | POST | Starts an optimization job, returns a `job_id` |
| `/api/job/{id}` | GET | Job status / result |
| `/api/job/{id}/stream` | GET | Server-sent events — live round-by-round progress |
| `/api/fold` | POST | ESMFold structure prediction (proxied to ESM Atlas) |
| `/api/saved-trajectory/{run}/{which}` | GET | A saved real-BioEmu ensemble for animation |

---

## Configuration

Runs are driven by YAML files in `protein_optimizer/config/`. Any field can be
overridden on the CLI with `--set key.path=value`, e.g.:

```bash
python3 main.py --config config/evolutionary.yaml \
    --set ga.population_size=100 \
    --set ga.max_generations=10 \
    --set bioemu.num_samples=200
```

Key knobs: `ga.population_size`, `ga.max_generations`, `bioemu.num_samples`,
`mutation.strategy` (`esm_guided` | `random`), and `bioemu.mock`.

---

## Tech stack

- **Python 3.10+**, FastAPI + Uvicorn (web server)
- **PyTorch** + HuggingFace **Transformers** (ESM-2)
- **BioEmu** (structural-ensemble scoring; CUDA + model weights for real runs)
- **3Dmol.js** and **ESMFold** (in-browser structure visualization)
- No build step — the frontend is a single static `index.html`

---

## Development

```bash
cd protein_optimizer
pip install -e ".[dev]"   # editable install + pytest
pytest                    # run the test suite
```

---

## Documentation

- **[protein_optimizer/README.md](protein_optimizer/README.md)** — deep dive:
  architecture, component breakdown, data flow, GPU-VM setup, extension points,
  and design principles.

---

## License

MIT
