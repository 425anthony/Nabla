import numpy as np
import urllib.request
import gzip
import os
import struct

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".mnist_cache")

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}


def _download(filename: str) -> bytes:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(MNIST_URL + filename, path)
    with gzip.open(path, "rb") as f:
        return f.read()


def _load_images(filename: str) -> np.ndarray:
    data = _download(filename)
    _, n, rows, cols = struct.unpack(">IIII", data[:16])
    pixels = np.frombuffer(data[16:], dtype=np.uint8).reshape(n, rows * cols)
    return pixels.astype(np.float32) / 255.0  # normalise to [0, 1]


def _load_labels(filename: str) -> np.ndarray:
    data = _download(filename)
    return np.frombuffer(data[8:], dtype=np.uint8).astype(np.int32)


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X_train, y_train, X_test, y_test).
    Images are flattened to 784-dim vectors, normalised to [0, 1].
    Labels are integer class indices 0-9.
    """
    X_train = _load_images(FILES["train_images"])
    y_train = _load_labels(FILES["train_labels"])
    X_test  = _load_images(FILES["test_images"])
    y_test  = _load_labels(FILES["test_labels"])
    return X_train, y_train, X_test, y_test


def make_batches(X: np.ndarray, y: np.ndarray,
                 batch_size: int, shuffle: bool = True):
    """Generator that yields (X_batch, y_batch) mini-batches."""
    n = X.shape[0]
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        b = idx[start:start + batch_size]
        yield X[b], y[b]
