"""Shared evaluation: val/test accuracy, per-class test accuracy, agreement with FP32 test predictions."""
import numpy as np

from data import CLASSES, load_eval
from quant import forward_folded

_DATA = {}


def get_data(val_idx: np.ndarray) -> dict:
    key = val_idx.tobytes()
    if key not in _DATA:
        _DATA[key] = load_eval(val_idx)
    return _DATA[key]


def predict(float_layers: dict, x: np.ndarray, bs: int = 500) -> np.ndarray:
    return np.concatenate([forward_folded(float_layers, x[i:i + bs]).argmax(1) for i in range(0, len(x), bs)])


def evaluate(float_layers: dict, val_idx: np.ndarray, ref_test_preds: np.ndarray | None = None):
    d = get_data(val_idx)
    (xv, yv), (xt, yt) = d["val"], d["test"]
    pv, pt = predict(float_layers, xv), predict(float_layers, xt)
    res = {
        "val": round(100 * float((pv == yv).mean()), 2),
        "test": round(100 * float((pt == yt).mean()), 2),
        "per_class_test": {c: round(100 * float((pt[yt == i] == i).mean()), 1) for i, c in enumerate(CLASSES)},
    }
    if ref_test_preds is not None:
        res["agree_fp32"] = round(100 * float((pt == ref_test_preds).mean()), 2)
    return res, pt
