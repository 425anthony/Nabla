import json
import os
import asyncio
import uuid
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
import threading
from .network import gradient_check, NeuralNetwork
from .data import load_mnist
from .trainer import Trainer, one_hot
import numpy as np

app = FastAPI(title="Nabla API")

# Allowed CORS origins are configurable so the deployed frontend's URL can be
# permitted in production. Comma-separated list via ALLOWED_ORIGINS; defaults to
# local dev origins (Vite dev server + the docker-compose nginx port).
_default_origins = "http://localhost:5173,http://localhost:3000,http://localhost:8080"
allow_origins = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", _default_origins).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store  { job_id: { "status": ..., "snapshots": [...], "config": ... } }
jobs: dict[str, dict] = {}

# Resource guards for the public deployment (Render free tier ~512 MB).
MAX_LAYERS = 8          # total layers including input/output
MAX_LAYER_SIZE = 2048   # neurons per layer
MAX_ACTIVE_JOBS = 2     # concurrent training jobs


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    layer_sizes: list[int] = Field(default=[784, 128, 64, 10])
    hidden_activation: str = Field(default="relu")
    learning_rate: float = Field(default=0.01, ge=1e-5, le=1.0)
    momentum: float = Field(default=0.9, ge=0.0, le=0.999)
    batch_size: int = Field(default=256, ge=16, le=1024)
    epochs: int = Field(default=20, ge=1, le=100)
    # Capped low so a raw API call can't trigger a full 60k MNIST run and OOM.
    data_fraction: float = Field(default=0.1, gt=0.0, le=0.2)
    data_dir: str = Field(default="./data")

    @field_validator("layer_sizes")
    @classmethod
    def _validate_layer_sizes(cls, v: list[int]) -> list[int]:
        if len(v) < 2:
            raise ValueError("layer_sizes must have at least 2 layers (input and output)")
        if len(v) > MAX_LAYERS:
            raise ValueError(f"layer_sizes may have at most {MAX_LAYERS} layers")
        if any(n < 1 or n > MAX_LAYER_SIZE for n in v):
            raise ValueError(f"each layer size must be between 1 and {MAX_LAYER_SIZE}")
        return v


# ---------------------------------------------------------------------------
# Background training thread
# ---------------------------------------------------------------------------

def _run_training(job_id: str, config: dict):
    jobs[job_id]["status"] = "running"

    def on_epoch(snapshot):
        jobs[job_id]["snapshots"].append(snapshot)

    try:
        # Build inside the try: a bad config (e.g. unknown activation) raises
        # here, and we must record it as an error rather than let the thread
        # die silently and leave the job stuck "running" (SSE would hang).
        trainer = Trainer(config)
        trainer.train(progress_callback=on_epoch)
        jobs[job_id]["status"] = "complete"
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/train")
async def start_training(req: TrainRequest):
    """Kick off a training job and return a job_id immediately."""
    active = sum(1 for j in jobs.values() if j["status"] in ("pending", "running"))
    if active >= MAX_ACTIVE_JOBS:
        raise HTTPException(
            status_code=429,
            detail=f"Too many active training jobs (max {MAX_ACTIVE_JOBS}). Try again shortly.",
        )

    job_id = str(uuid.uuid4())
    config = req.model_dump()
    jobs[job_id] = {"status": "pending", "snapshots": [], "config": config}
    thread = threading.Thread(target=_run_training, args=(job_id, config), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/train/{job_id}/stream")
async def stream_training(job_id: str):
    """
    Server-Sent Events stream. Emits each epoch snapshot as it completes.
    The frontend connects here and receives live updates.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        sent = 0
        while True:
            job = jobs[job_id]
            snapshots = job["snapshots"]

            while sent < len(snapshots):
                data = json.dumps(snapshots[sent])
                yield f"data: {data}\n\n"
                sent += 1

            if job["status"] in ("complete", "error"):
                payload = {"event": job["status"]}
                if job["status"] == "error":
                    payload["detail"] = job.get("error", "Training failed")
                yield f"data: {json.dumps(payload)}\n\n"
                break

            await asyncio.sleep(0.2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/train/{job_id}/status")
async def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return {
        "status": job["status"],
        "epochs_complete": len(job["snapshots"]),
        "config": job["config"],
    }


@app.get("/train/{job_id}/snapshots")
async def all_snapshots(job_id: str):
    """Return all snapshots at once (for epoch scrubber after training completes)."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"snapshots": jobs[job_id]["snapshots"]}


@app.get("/train/{job_id}/snapshots/{epoch}")
async def snapshot_at_epoch(job_id: str, epoch: int):
    """Return the snapshot for a specific epoch (1-indexed)."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    snaps = jobs[job_id]["snapshots"]
    if epoch < 1 or epoch > len(snaps):
        raise HTTPException(status_code=404, detail=f"Epoch {epoch} not yet available")
    return snaps[epoch - 1]


@app.post("/gradient-check")
async def run_gradient_check():
    """
    Runs finite-difference gradient check on a small network.
    Useful for the README / demo to prove the backprop is correct.
    """
    np.random.seed(42)
    net = NeuralNetwork([784, 32, 10])
    x = np.random.randn(8, 784)
    y = np.random.randint(0, 10, 8)
    result = gradient_check(net, x, one_hot(y))
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}
