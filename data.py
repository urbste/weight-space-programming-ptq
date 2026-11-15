"""CIFAR-10 loading (numpy only, no torch needed).

Split: fixed 45k/5k train/val (permutation from torch.randperm(50000, seed 0), generated in train.py
and stored in the exported .npz) + the 10k test set.
"""
import pickle
from pathlib import Path

import numpy as np

MEAN = np.array((0.4914, 0.4822, 0.4465), dtype=np.float32)
STD = np.array((0.2470, 0.2435, 0.2616), dtype=np.float32)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def _load_batches(root: Path, names):
    xs, ys = [], []
    for n in names:
        with open(root / n, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        xs.append(d[b"data"])
        ys.extend(d[b"labels"])
    x = np.concatenate(xs).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    return x, np.array(ys, dtype=np.int64)


def normalize(x: np.ndarray) -> np.ndarray:
    return ((x - MEAN.reshape(1, 3, 1, 1)) / STD.reshape(1, 3, 1, 1)).astype(np.float32)


def load_raw(root: str = "./data"):
    """Raw [0,1] images: (x_train50k, y_train50k), (x_test, y_test)."""
    root = Path(root) / "cifar-10-batches-py"
    return _load_batches(root, [f"data_batch_{i}" for i in range(1, 6)]), _load_batches(root, ["test_batch"])


def load_eval(val_idx: np.ndarray, root: str = "./data") -> dict:
    """Normalized val and test sets."""
    (x_tr, y_tr), (x_te, y_te) = load_raw(root)
    return {"val": (normalize(x_tr[val_idx]), y_tr[val_idx]), "test": (normalize(x_te), y_te)}
