"""Ablation: RTN PTQ on the UNFOLDED network (BN stays float) vs. the folded one used everywhere else.

python analysis/unfolded_rtn.py -> results/unfolded_rtn.json
Unfolded: per-channel RTN on the raw conv weights; BN (gain k = gamma/sqrt(var+eps), shift) applied in float.
Since BN is a per-channel affine map after the conv, this equals a folded forward pass with weights k * (q*scale).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
from evaluate import evaluate  # noqa: E402
from quant import dequantize, fold_bn, load_npz, rtn_quantize  # noqa: E402

sd, meta = load_npz("checkpoints/small_e500.npz")
val_idx = np.load("exports/val_idx.npy")
ref = np.load("exports/fp32_test_preds.npy")
eps = 1e-5

raw = {f"conv{i}": {"weight": sd[f"conv{i}.weight"].astype(np.float32),
                    "bias": np.zeros(sd[f"conv{i}.weight"].shape[0], np.float32)} for i in range(1, 5)}
raw["fc"] = {"weight": sd["fc.weight"].astype(np.float32), "bias": sd["fc.bias"].astype(np.float32)}
folded = fold_bn(sd)

def bn_params(i):
    g, b = sd[f"bn{i}.weight"].astype(np.float64), sd[f"bn{i}.bias"].astype(np.float64)
    mu, var = sd[f"bn{i}.running_mean"].astype(np.float64), sd[f"bn{i}.running_var"].astype(np.float64)
    k = g / np.sqrt(var + eps)
    return k, b - mu * k

out = {}
for bits in (8, 6, 5, 4, 3, 2):
    # folded (reference)
    rf, _ = evaluate(dequantize(rtn_quantize(folded, bits)), val_idx, ref)
    # unfolded
    dq = dequantize(rtn_quantize(raw, bits))
    L = {}
    for i in range(1, 5):
        k, c = bn_params(i)
        L[f"conv{i}"] = {"weight": (dq[f"conv{i}"]["weight"] * k[:, None, None, None]).astype(np.float32),
                         "bias": c.astype(np.float32)}
    L["fc"] = dq["fc"]
    ru, _ = evaluate(L, val_idx, ref)
    out[f"int{bits}"] = {"folded_val": rf["val"], "folded_test": rf["test"], "folded_agree": rf["agree_fp32"],
                         "unfolded_val": ru["val"], "unfolded_test": ru["test"], "unfolded_agree": ru["agree_fp32"]}
    print(f"int{bits}  folded test {rf['test']:.2f} (agree {rf['agree_fp32']:.1f})   "
          f"unfolded test {ru['test']:.2f} (agree {ru['agree_fp32']:.1f})")

# spread of per-row max|w| (what per-channel scales have to absorb)
spread = {}
for i in range(1, 5):
    mf = np.abs(folded[f"conv{i}"]["weight"].reshape(folded[f"conv{i}"]["weight"].shape[0], -1)).max(1)
    mr = np.abs(raw[f"conv{i}"]["weight"].reshape(raw[f"conv{i}"]["weight"].shape[0], -1)).max(1)
    spread[f"conv{i}"] = {"folded_max_over_min": round(float(mf.max() / mf.min()), 2),
                          "unfolded_max_over_min": round(float(mr.max() / mr.min()), 2)}
out["row_range_spread"] = spread
print(spread)
Path("results/unfolded_rtn.json").write_text(json.dumps(out, indent=1))

# per-tensor scale (one scale per layer): here folding does matter
def rtn_tensor(layers, bits):
    qmax = 2 ** (bits - 1) - 1
    o = {}
    for n, d in layers.items():
        w = d["weight"]; s = np.float32(np.abs(w).max() / qmax)
        q = np.clip(np.rint(w / s), -qmax, qmax).astype(np.int32)
        o[n] = {"q": q, "scale": np.full(w.shape[0], s, np.float32), "bias": d["bias"].copy()}
    return o

pt = {}
for bits in (8, 6, 4):
    rf, _ = evaluate(dequantize(rtn_tensor(folded, bits)), val_idx, ref)
    dq = dequantize(rtn_tensor(raw, bits)); L = {}
    for i in range(1, 5):
        k, c = bn_params(i)
        L[f"conv{i}"] = {"weight": (dq[f"conv{i}"]["weight"] * k[:, None, None, None]).astype(np.float32), "bias": c.astype(np.float32)}
    L["fc"] = dq["fc"]
    ru, _ = evaluate(L, val_idx, ref)
    pt[f"int{bits}"] = {"folded_test": rf["test"], "unfolded_test": ru["test"]}
    print(f"per-tensor int{bits}  folded test {rf['test']:.2f}   unfolded test {ru['test']:.2f}")
out["per_tensor"] = pt
Path("results/unfolded_rtn.json").write_text(json.dumps(out, indent=1))
