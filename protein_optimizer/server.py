"""
Protein Optimizer — Web Server

Usage:
    pip install fastapi uvicorn
    python server.py

Then open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from protein_optimizer import OptimizationConfig
from protein_optimizer.evolutionary_search import BudgetedEvolutionarySearch

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Protein Optimizer")

STATIC_DIR = Path(__file__).parent / "frontend"
CONFIG_DIR = Path(__file__).parent / "config"
SHARED_RUNS_DIR = Path(__file__).parent / "shared_runs"

jobs: Dict[str, Dict[str, Any]] = {}
job_queues: Dict[str, queue.Queue] = {}


class RunRequest(BaseModel):
    sequence: str
    mode: str  # "maximize" | "fit_to_llr" | "fit_to_healthy"
    target_llr: Optional[float] = None
    healthy_sequence: Optional[str] = None
    mock: bool = True
    random_mutations: bool = False
    population_size: int = 20
    max_generations: int = 3
    num_samples: int = 10


@app.get("/")
def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/saved-trajectory/{run}/{which}")
def saved_trajectory(run: str, which: str):
    """Serve a pre-converted multi-frame PDB of a saved BioEmu run (real .xtc
    ensemble), for animation in the 3D viewer. Read-only, path-sanitised."""
    if which not in ("reference", "best_mutant") or "/" in run or ".." in run:
        return JSONResponse({"error": "Invalid trajectory"}, status_code=400)
    path = SHARED_RUNS_DIR / run / which / "ensemble.pdb"
    if not path.is_file():
        return JSONResponse({"error": "Trajectory not found"}, status_code=404)
    return FileResponse(str(path), media_type="text/plain")


@app.post("/api/run")
def start_run(req: RunRequest):
    job_id = str(uuid.uuid4())[:8]
    q: queue.Queue = queue.Queue()
    jobs[job_id] = {
        "status": "running",
        "result": None,
        "error": None,
        "started_at": time.time(),
    }
    job_queues[job_id] = q

    def progress_callback(event: dict) -> None:
        q.put(event)

    def run_job() -> None:
        try:
            cfg = OptimizationConfig.from_yaml(str(CONFIG_DIR / "evolutionary.yaml"))
            cfg.original_sequence = req.sequence.upper().strip()
            cfg.bioemu.mock = req.mock
            cfg.bioemu.num_samples = req.num_samples
            cfg.ga.population_size = req.population_size
            cfg.ga.max_generations = req.max_generations
            cfg.mutation.strategy = "random" if req.random_mutations else "esm_guided"

            if req.mode == "fit_to_llr" and req.target_llr is not None:
                cfg.target_parameter = req.target_llr
                cfg.healthy_sequence = ""
            elif req.mode == "fit_to_healthy" and req.healthy_sequence:
                cfg.healthy_sequence = req.healthy_sequence.upper().strip()
                cfg.target_parameter = None
            else:
                cfg.target_parameter = None
                cfg.healthy_sequence = ""

            search = BudgetedEvolutionarySearch(cfg, verbose=False)
            result = search.run(progress_callback=progress_callback)

            result_data = {
                "reference_llr": result.reference_llr,
                "best_llr": result.best_llr,
                "best_sequence": result.best_sequence,
                "target_parameter": result.target_parameter,
                "improved": result.improved,
                "rounds_run": result.rounds_run,
                "total_evaluated": result.total_evaluated,
                "wall_time_s": result.total_wall_time_s,
                "top10": [
                    {"sequence": seq, "llr": llr} for seq, llr in result.ranked(10)
                ],
                "top20": [
                    {"sequence": seq, "llr": llr} for seq, llr in result.ranked(20)
                ],
            }
            jobs[job_id]["result"] = result_data
            jobs[job_id]["status"] = "done"
            q.put({"type": "done", "result": result_data})
            logger.info("Job %s completed in %.1fs", job_id, result.total_wall_time_s)
        except Exception as exc:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(exc)
            q.put({"type": "error", "message": str(exc)})
            logger.exception("Job %s failed", job_id)

    threading.Thread(target=run_job, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return jobs[job_id]


def _queue_get(q: queue.Queue, timeout: float = 1.0):
    try:
        return q.get(block=True, timeout=timeout)
    except queue.Empty:
        return None


@app.get("/api/job/{job_id}/stream")
async def stream_job(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    job = jobs[job_id]

    # Already finished — serve result immediately (queue may be drained)
    if job["status"] in ("done", "error"):
        async def immediate():
            if job["status"] == "done":
                yield f"data: {json.dumps({'type': 'done', 'result': job['result']})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': job.get('error', '')})}\n\n"

        return StreamingResponse(
            immediate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Job still running — stream live from queue
    q = job_queues.get(job_id)
    if q is None:
        return JSONResponse({"error": "Queue not available"}, status_code=500)

    loop = asyncio.get_running_loop()

    async def generate():
        while True:
            event = await loop.run_in_executor(None, _queue_get, q, 1.0)
            if event is None:
                yield ": keepalive\n\n"
                continue
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") in ("done", "error"):
                break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


@app.post("/api/fold")
async def fold_sequence(request: Request):
    """Proxy ESMFold structure prediction through the ESM Atlas public API."""
    seq = (await request.body()).decode().strip().upper()
    if not seq or not all(c in VALID_AA for c in seq):
        return JSONResponse({"error": "Invalid sequence"}, status_code=400)
    if len(seq) > 400:
        return JSONResponse({"error": "Sequence too long (max 400)"}, status_code=400)

    def _call_esm():
        req = urllib.request.Request(
            "https://api.esmatlas.com/foldSequence/v1/pdb/",
            data=seq.encode(),
            headers={"Content-Type": "text/plain"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()

    loop = asyncio.get_running_loop()
    try:
        pdb = await loop.run_in_executor(None, _call_esm)
        return Response(content=pdb, media_type="text/plain")
    except Exception as exc:
        logger.warning("ESMFold failed (len=%d): %s", len(seq), exc)
        return JSONResponse({"error": "Structure prediction unavailable"}, status_code=503)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
