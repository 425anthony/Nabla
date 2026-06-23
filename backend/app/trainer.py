import numpy as np

from .network import NeuralNetwork
from .data import load_mnist, make_batches


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    Convert integer class labels to one-hot rows.
    y : (n,) integer labels.  Returns (n, num_classes) float matrix.
    """
    y = np.asarray(y).astype(int).ravel()
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


# ---------------------------------------------------------------------------
# Trainer — wraps NeuralNetwork with an SGD/momentum loop over MNIST
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, config: dict):
        """
        config keys (see TrainRequest in main.py):
            layer_sizes, hidden_activation, learning_rate,
            momentum, batch_size, epochs, data_dir
        """
        self.layer_sizes = config["layer_sizes"]
        self.hidden_activation = config.get("hidden_activation", "relu")
        self.lr = config["learning_rate"]
        self.momentum = config.get("momentum", 0.9)
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.data_fraction = config.get("data_fraction", 1.0)
        self.num_classes = self.layer_sizes[-1]

        self.net = NeuralNetwork(self.layer_sizes, self.hidden_activation)

    def _epoch_snapshot(self, epoch: int,
                        X_train, y_train, y_train_oh,
                        X_test, y_test) -> dict:
        """Evaluate on train/test and bundle metrics + layer state for one epoch."""
        train_probs = self.net.forward(X_train)
        train_loss = self.net.cross_entropy_loss(train_probs, y_train_oh)
        train_acc = self.net.accuracy(train_probs, y_train)

        test_probs = self.net.forward(X_test)
        test_loss = self.net.cross_entropy_loss(test_probs, one_hot(y_test, self.num_classes))
        test_acc = self.net.accuracy(test_probs, y_test)

        return {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "layers": self.net.snapshot(),
        }

    def train(self, progress_callback=None) -> NeuralNetwork:
        """
        Run the full training loop. After each epoch, build a snapshot and
        hand it to progress_callback (used by main.py to feed the SSE stream).
        """
        X_train, y_train, X_test, y_test = load_mnist()

        # Optionally train on a random subset (data_fraction in (0, 1]).
        if self.data_fraction < 1.0:
            n = max(self.batch_size, int(X_train.shape[0] * self.data_fraction))
            idx = np.random.permutation(X_train.shape[0])[:n]
            X_train, y_train = X_train[idx], y_train[idx]

        y_train_oh = one_hot(y_train, self.num_classes)

        for epoch in range(1, self.epochs + 1):
            for x_batch, y_batch in make_batches(X_train, y_train, self.batch_size):
                self.net.forward(x_batch)
                self.net.backward(one_hot(y_batch, self.num_classes))
                self.net.update_weights(self.lr, self.momentum)

            if progress_callback is not None:
                snapshot = self._epoch_snapshot(
                    epoch, X_train, y_train, y_train_oh, X_test, y_test
                )
                progress_callback(snapshot)

        return self.net
