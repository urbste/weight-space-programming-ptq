"""Experiment 1: standard post-training quantization of the FP32 network at 8 / 4 / 2 bits.

python ptq_eval.py   -> results/ptq.json, exports/float32_folded.npz, exports/rtn_int{8,4,2}.npz
(numpy only; reads checkpoints/small_e500.npz written by train.py)
"""
import json
import sys
from pathlib import Path

import numpy as np

from evaluate import evaluate
from quant import LAYERS, dequantize, fold_bn, load_npz, rtn_quantize

BITS = (8, 4, 2)


def save_layers(path, layers):
    np.savez(path, **{f"{n}.{k}": v for n, d in layers.items() for k, v in d.items()})


def load_layers(path):
    z = np.load(path)
    out = {}
    for key in z.files:
        n, k = key.split(".", 1)
        out.setdefault(n, {})[k] = z[key]
    return {n: out[n] for n in LAYERS}


def main(ckpt="checkpoints/small_e500.npz"):
    sd, meta = load_npz(ckpt)
    val_idx = meta["val_idx"]
    fl = fold_bn(sd)
    Path("exports").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    save_layers("exports/float32_folded.npz", fl)
    np.save("exports/val_idx.npy", val_idx)

    res = {"checkpoint": ckpt, "epoch": int(meta["epoch"]),
           "torch_val": float(meta["val_acc_torch"]), "torch_test": float(meta["test_acc_torch"])}
    r, ref = evaluate(fl, val_idx)
    r["agree_fp32"] = 100.0
    res["fp32"] = r
    np.save("exports/fp32_test_preds.npy", ref)
    print(f"fp32       val {r['val']:.2f}  test {r['test']:.2f}   (torch: val {100*res['torch_val']:.2f} "
          f"test {100*res['torch_test']:.2f})")
    for b in BITS:
        q = rtn_quantize(fl, b)
        save_layers(f"exports/rtn_int{b}.npz", q)
        r, _ = evaluate(dequantize(q), val_idx, ref)
        res[f"rtn_int{b}"] = r
        print(f"rtn int{b}   val {r['val']:.2f}  test {r['test']:.2f}  agree {r['agree_fp32']:.2f}")
    Path("results/ptq.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
