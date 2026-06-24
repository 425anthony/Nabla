# Nabla

A from-scratch neural network trained on MNIST with a live interactive visualizer. No PyTorch, no Keras — pure NumPy backpropagation with a React frontend that animates weight changes and gradient magnitudes epoch by epoch.


---

## What it does

- Trains a configurable feedforward network on MNIST (60k images, 10 classes)
- Streams training progress live to the browser via Server-Sent Events
- Visualizes the network as an SVG graph where edge color encodes weight sign/magnitude
- **Gradient heatmap mode** — colors each neuron by its ∂L/∂z value so you can watch vanishing gradients in real time
- Epoch scrubber to play back training frame by frame
- Click any neuron to inspect its weights, bias, incoming weight distribution, and gradient
- Gradient check endpoint: verifies backprop correctness via finite differences

---

## Stack

| Layer | Tech |
|---|---|
| Training engine | Pure NumPy (no autograd) |
| API | FastAPI + Server-Sent Events |
| Frontend | React + TypeScript + Vite |
| Charts | Recharts |
| Deployment | Docker + docker-compose, Render (blueprint) |

---

## Math

### Forward pass

For each dense layer $l$:

$$z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}$$
$$a^{(l)} = \sigma(z^{(l)})$$

where $\sigma$ is ReLU for hidden layers and softmax for the output layer.

### Loss

Cross-entropy over the softmax output:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log \hat{p}_{ik}$$

### Backpropagation

Output layer — softmax + cross-entropy gradient fuses cleanly:

$$\frac{\partial \mathcal{L}}{\partial z^{(L)}} = \frac{\hat{p} - y}{N}$$

Hidden layers — chain rule:

$$\delta^{(l)} = \left(\delta^{(l+1)} {W^{(l+1)}}^T\right) \odot \sigma'\!\left(z^{(l)}\right)$$

Weight gradients:

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = {a^{(l-1)}}^T \delta^{(l)}$$

### SGD with momentum

$$v_W \leftarrow \mu v_W - \eta \frac{\partial \mathcal{L}}{\partial W}$$
$$W \leftarrow W + v_W$$

### Gradient check

Correctness is verified via finite differences on a random mini-batch:

$$\frac{\partial \mathcal{L}}{\partial W_{ij}} \approx \frac{\mathcal{L}(W_{ij} + \varepsilon) - \mathcal{L}(W_{ij} - \varepsilon)}{2\varepsilon}$$

Max relative error is typically $< 10^{-5}$, confirming the analytical gradients are correct.

---

## Running locally

**With Docker (recommended):**
```bash
git clone <your-repo-url>   # e.g. https://github.com/<you>/nabla
cd nabla
docker-compose up --build
```
Then open **http://localhost:8080** (frontend); the backend runs on http://localhost:8000.

**Without Docker:**
```bash
# Backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev   # serves on http://localhost:5173
```

MNIST data (~11MB) downloads automatically on first run.

---

## Deployment


### Local (Docker Compose)

```bash
docker-compose up --build
```

| Service | URL | Notes |
|---|---|---|
| Frontend (nginx) | http://localhost:8080 | serves the built Vite app |
| Backend (FastAPI) | http://localhost:8000 | `/health`, `/train`, … |

The frontend's API base URL is baked in at build time via the `VITE_API_URL`
build arg (set in `docker-compose.yml`). The backend's permitted CORS origins
come from `ALLOWED_ORIGINS`. MNIST data persists in the `mnist-cache` volume.

### Render (infrastructure-as-code)

`render.yaml` defines both services — backend as a Dockerized **web service**,
frontend as a **static site**.

1. Push the repo to GitHub.
2. In Render: **New + → Blueprint**, select the repo (Render reads `render.yaml`).
3. After the first deploy, set these env vars (Render prompts — they're `sync: false`):
   - **nabla-backend** → `ALLOWED_ORIGINS` = the frontend URL (e.g. `https://nabla-frontend.onrender.com`)
   - **nabla-frontend** → `VITE_API_URL` = the backend URL (e.g. `https://nabla-backend.onrender.com`)
4. `VITE_API_URL` is **build-time**, so after setting it redeploy the frontend
   ("Clear build cache & deploy"). `ALLOWED_ORIGINS` is **runtime** — the backend
   just restarts.

> On Render's free plan, services sleep when idle (first request ~50s) and have
> ephemeral disk, so MNIST re-downloads after a restart — handled automatically
> on the first training run.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| POST | `/train` | Start a training job, returns `job_id` |
| GET | `/train/{job_id}/stream` | SSE stream of epoch snapshots |
| GET | `/train/{job_id}/status` | Job status + epochs completed |
| GET | `/train/{job_id}/snapshots` | All snapshots after training |
| GET | `/train/{job_id}/snapshots/{epoch}` | Snapshot at specific epoch |
| POST | `/gradient-check` | Finite-difference gradient verification |
| GET | `/health` | Health check |

---

## Project structure

```
nabla/
├── docker-compose.yml          # Brings up backend + frontend together
├── render.yaml                 # Render blueprint (backend web service + static frontend)
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── __init__.py
│       ├── data.py             # MNIST download + cache + batching
│       ├── network.py          # DenseLayer, SoftmaxOutputLayer, NeuralNetwork, gradient_check
│       ├── trainer.py          # Training loop, one-hot, snapshot export
│       └── main.py             # FastAPI app, SSE streaming, job store, validation
└── frontend/
    ├── Dockerfile
    ├── nginx.conf              # SPA serving for the production image
    ├── index.html
    └── src/
        ├── main.tsx
        ├── index.css           # Dark theme tokens + component styles
        ├── App.tsx             # Layout, config panel, controls, loss/accuracy chart, scrubber
        ├── hooks/
        │   └── useTraining.ts  # POST /train → SSE stream, snapshot normalization
        └── components/
            └── NetworkDiagram.tsx   # SVG network: glowing neurons, weight/gradient coloring, inspect panel
```

> The loss curve, epoch scrubber, config form, and neuron inspector are
> implemented inline within `App.tsx` and `NetworkDiagram.tsx` rather than as
> separate component files.
