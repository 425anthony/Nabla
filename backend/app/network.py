import numpy as np


# Largest gradient magnitude applied per update — clamps extreme learning rates
# so a single step can't blow the weights up to inf/NaN.
GRAD_CLIP = 5.0


def _finite(x):
    """Return x if finite, else None (so JSON serializes null instead of NaN)."""
    return x if np.isfinite(x) else None


def _clean(arr: np.ndarray) -> list:
    """ndarray → nested list with non-finite entries replaced by None (JSON-safe)."""
    return np.where(np.isfinite(arr), arr, None).tolist()


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)

def softmax(z):
    # Numerically stable: subtract row max before exp
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

ACTIVATIONS = {
    "relu":    (relu,    relu_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
}


# ---------------------------------------------------------------------------
# Dense layer
# ---------------------------------------------------------------------------

class DenseLayer:
    def __init__(self, n_in: int, n_out: int, activation: str = "relu"):
        # He initialisation (good default for ReLU networks)
        scale = np.sqrt(2.0 / n_in)
        self.W = np.random.randn(n_in, n_out) * scale
        self.b = np.zeros((1, n_out))
        self.activation = activation
        self.act_fn, self.act_grad = ACTIVATIONS[activation]

        # Populated during forward / backward passes
        self.x = None   # input to this layer
        self.z = None   # pre-activation  z = xW + b
        self.a = None   # post-activation a = act(z)
        self.dW = None
        self.db = None
        self.grad_magnitude = 0.0  # scalar summary for heatmap

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.z = x @ self.W + self.b
        self.a = self.act_fn(self.z)
        return self.a

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        delta : upstream gradient  dL/da  shape (batch, n_out)
        returns downstream gradient dL/dx shape (batch, n_in)

        dL/dz  = delta * act'(z)       (element-wise)
        dL/dW  = x^T @ dL/dz           (outer product summed over batch)
        dL/db  = sum(dL/dz, axis=0)
        dL/dx  = dL/dz @ W^T           (propagate upstream)
        """
        dz = delta * self.act_grad(self.z)         # (batch, n_out)
        # Upstream delta already carries the 1/batch from the output layer's
        # (probs - y) / batch, so do NOT divide again here.
        self.dW = self.x.T @ dz                     # (n_in, n_out)
        self.db = np.sum(dz, axis=0, keepdims=True)
        self.grad_magnitude = float(np.mean(np.abs(dz)))
        return dz @ self.W.T                        # (batch, n_in)

    def snapshot(self) -> dict:
        """Serialisable state for one training epoch."""
        return {
            "W": _clean(self.W),
            "b": _clean(self.b),
            "dW": _clean(self.dW) if self.dW is not None else None,
            "db": _clean(self.db) if self.db is not None else None,
            "grad_magnitude": _finite(self.grad_magnitude),
            "activation": self.activation,
            "shape": [int(self.W.shape[0]), int(self.W.shape[1])],
        }


# ---------------------------------------------------------------------------
# Output layer (softmax + cross-entropy fused for numerical stability)
# ---------------------------------------------------------------------------

class SoftmaxOutputLayer:
    def __init__(self, n_in: int, n_out: int):
        scale = np.sqrt(1.0 / n_in)
        self.W = np.random.randn(n_in, n_out) * scale
        self.b = np.zeros((1, n_out))
        self.x = None
        self.probs = None
        self.dW = None
        self.db = None
        self.grad_magnitude = 0.0
        self.activation = "softmax"

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        z = x @ self.W + self.b
        self.probs = softmax(z)
        return self.probs

    def backward_from_loss(self, y_onehot: np.ndarray) -> np.ndarray:
        """
        For softmax + cross-entropy the fused gradient simplifies to:
            dL/dz = (probs - y) / batch_size
        This is the most numerically stable form.
        """
        batch = self.x.shape[0]
        dz = (self.probs - y_onehot) / batch        # (batch, n_out)
        self.dW = self.x.T @ dz
        self.db = np.sum(dz, axis=0, keepdims=True)
        self.grad_magnitude = float(np.mean(np.abs(dz)))
        return dz @ self.W.T                         # (batch, n_in)

    def snapshot(self) -> dict:
        return {
            "W": _clean(self.W),
            "b": _clean(self.b),
            "dW": _clean(self.dW) if self.dW is not None else None,
            "db": _clean(self.db) if self.db is not None else None,
            "grad_magnitude": _finite(self.grad_magnitude),
            "activation": self.activation,
            "shape": [int(self.W.shape[0]), int(self.W.shape[1])],
        }


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class NeuralNetwork:
    def __init__(self, layer_sizes: list[int], hidden_activation: str = "relu"):
        """
        layer_sizes : e.g. [784, 128, 64, 10]
        Creates len(layer_sizes)-2 hidden layers + 1 softmax output layer.
        """
        self.layers: list[DenseLayer | SoftmaxOutputLayer] = []
        for i in range(len(layer_sizes) - 2):
            self.layers.append(
                DenseLayer(layer_sizes[i], layer_sizes[i + 1], activation=hidden_activation)
            )
        self.layers.append(
            SoftmaxOutputLayer(layer_sizes[-2], layer_sizes[-1])
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_onehot: np.ndarray):
        # Output layer has fused softmax-cross-entropy backward
        delta = self.layers[-1].backward_from_loss(y_onehot)
        for layer in reversed(self.layers[:-1]):
            delta = layer.backward(delta)

    def update_weights(self, lr: float, momentum: float = 0.9,
                       optimizer: str = "sgd",
                       beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        """
        Update every layer's weights. Per-layer optimizer state is stored on the
        layer objects. Supports:
          - "sgd":  SGD with momentum (velocity vW/vb)
          - "adam": Adam with bias-corrected first/second moments
        """
        for layer in self.layers:
            # Clip gradients so an extreme learning rate can't diverge to inf/NaN.
            dW = np.clip(layer.dW, -GRAD_CLIP, GRAD_CLIP)
            db = np.clip(layer.db, -GRAD_CLIP, GRAD_CLIP)

            if optimizer == "adam":
                if not hasattr(layer, "adam_t"):
                    layer.adam_mW = np.zeros_like(layer.W)
                    layer.adam_vW = np.zeros_like(layer.W)
                    layer.adam_mb = np.zeros_like(layer.b)
                    layer.adam_vb = np.zeros_like(layer.b)
                    layer.adam_t = 0
                layer.adam_t += 1
                t = layer.adam_t
                # First/second moment estimates
                layer.adam_mW = beta1 * layer.adam_mW + (1 - beta1) * dW
                layer.adam_vW = beta2 * layer.adam_vW + (1 - beta2) * (dW * dW)
                layer.adam_mb = beta1 * layer.adam_mb + (1 - beta1) * db
                layer.adam_vb = beta2 * layer.adam_vb + (1 - beta2) * (db * db)
                # Bias-corrected estimates
                mW_hat = layer.adam_mW / (1 - beta1 ** t)
                vW_hat = layer.adam_vW / (1 - beta2 ** t)
                mb_hat = layer.adam_mb / (1 - beta1 ** t)
                vb_hat = layer.adam_vb / (1 - beta2 ** t)
                layer.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
                layer.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
            else:  # sgd with momentum
                if not hasattr(layer, "vW"):
                    layer.vW = np.zeros_like(layer.W)
                    layer.vb = np.zeros_like(layer.b)
                layer.vW = momentum * layer.vW - lr * dW
                layer.vb = momentum * layer.vb - lr * db
                layer.W += layer.vW
                layer.b += layer.vb

    def cross_entropy_loss(self, probs: np.ndarray, y_onehot: np.ndarray) -> float:
        eps = 1e-12
        return float(-np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1)))

    def accuracy(self, probs: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(np.argmax(probs, axis=1) == y))

    def snapshot(self) -> list[dict]:
        return [layer.snapshot() for layer in self.layers]


# ---------------------------------------------------------------------------
# Gradient check (finite differences) — used in tests / debugging
# ---------------------------------------------------------------------------

def gradient_check(network: NeuralNetwork, x: np.ndarray, y_onehot: np.ndarray,
                   eps: float = 1e-5, sample_size: int = 20,
                   tolerance: float = 1e-5) -> dict:
    """
    Verifies dW against finite differences for EVERY layer.
    For each layer, samples `sample_size` weight entries, compares the
    analytical gradient to a central finite-difference estimate, and reports
    the max relative error. Backprop is correct iff every layer is < tolerance.
    """
    # Analytical gradients for all layers (one forward + backward pass).
    network.forward(x)
    network.backward(y_onehot)
    analytical = [layer.dW.copy() for layer in network.layers]

    per_layer = []
    overall_max = 0.0

    for li, layer in enumerate(network.layers):
        W = layer.W
        a_dW = analytical[li]
        flat_idx = np.random.choice(W.size, min(sample_size, W.size), replace=False)
        indices = np.array(np.unravel_index(flat_idx, W.shape)).T

        rel_errors = []
        for i, j in indices:
            orig = W[i, j]
            W[i, j] = orig + eps
            loss_plus = network.cross_entropy_loss(network.forward(x), y_onehot)
            W[i, j] = orig - eps
            loss_minus = network.cross_entropy_loss(network.forward(x), y_onehot)
            W[i, j] = orig  # restore exactly to avoid drift
            numerical = (loss_plus - loss_minus) / (2 * eps)
            # Standard relative error, robust when both grads are tiny.
            denom = max(abs(numerical), abs(a_dW[i, j]), 1e-8)
            rel_errors.append(abs(a_dW[i, j] - numerical) / denom)

        layer_max = float(np.max(rel_errors)) if rel_errors else 0.0
        overall_max = max(overall_max, layer_max)
        per_layer.append({
            "layer": li,
            "type": type(layer).__name__,
            "shape": [int(W.shape[0]), int(W.shape[1])],
            "max_relative_error": layer_max,
            "mean_relative_error": float(np.mean(rel_errors)) if rel_errors else 0.0,
            "passed": bool(layer_max < tolerance),
        })

    return {
        "max_relative_error": overall_max,
        "mean_relative_error": float(np.mean([p["mean_relative_error"] for p in per_layer])),
        "tolerance": tolerance,
        "passed": bool(overall_max < tolerance),
        "layers": per_layer,
    }
