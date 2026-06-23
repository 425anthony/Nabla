# Nabla

A from-scratch neural network trained on MNIST with a live interactive visualizer. No PyTorch, no Keras — pure NumPy backpropagation with a React frontend that animates weight changes and gradient magnitudes epoch by epoch.

**Live demo:** 

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
| Frontend | React + Vite |
| Charts | Recharts |
| Deployment | Docker + docker-compose |

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
git clone https://github.com/425anthony/neural-net-visualizer
cd neural-net-visualizer
docker-compose up --build
```
Open http://localhost:5173

**Without Docker:**
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

MNIST data (~11MB) downloads automatically on first run.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| POST | `/train` | Start a training job, returns `job_id` |
| GET | `/train/{job_id}/stream` | SSE stream of epoch snapshots |
| GET | `/train/{job_id}/snapshots` | All snapshots after training |
| GET | `/train/{job_id}/snapshots/{epoch}` | Snapshot at specific epoch |
| POST | `/gradient-check` | Finite-difference gradient verification |
| GET | `/health` | Health check |

---

## Project structure

```
neural-net-visualizer/
├── backend/
│   ├── network.py      # DenseLayer, SoftmaxOutputLayer, NeuralNetwork, gradient_check
│   ├── trainer.py      # MNIST loader, training loop, snapshot export
│   ├── main.py         # FastAPI app, SSE streaming, job store
│   └── requirements.txt
└── frontend/
    └── src/
        ├── App.jsx
        └── components/
            ├── NetworkDiagram.jsx   # SVG network with weight/gradient coloring
            ├── LossCurve.jsx        # Recharts loss + accuracy plots
            ├── EpochScrubber.jsx    # Play/pause epoch animation
            ├── NeuronInspector.jsx  # Click-to-inspect panel
            └── TrainingConfig.jsx   # Architecture + hyperparameter form
```