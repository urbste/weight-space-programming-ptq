"""Diagnostics for Experiment 6 (P4: float weights + BN statistics), Opus and Fable runs.

For each run: exact re-rounding at the model's own scales, per-layer swap into RTN, scale ratio vs. outlier
ratio (largest / second-largest |w| in the row), kernel sums. Plus an MSE-optimal per-row scale search, and two
controls for conv2 (random per-row scale factors, the model's ratios shuffled across rows).
-> results/fp32_bn_diag.json
"""
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "llm")]
os.chdir(ROOT)
from eval_fp32_bn import build, parse_rows  # noqa: E402
from evaluate import evaluate  # noqa: E402
from ptq_eval import load_layers  # noqa: E402
from quant import LAYERS, dequantize  # noqa: E402

QMAX = 7
q0 = load_layers("exports/rtn_int4.npz")
fl = load_layers("exports/float32_folded.npz")
ref = np.load("exports/fp32_test_preds.npy")
val_idx = np.load("exports/val_idx.npy")


def acc(q):
    r, _ = evaluate(dequantize(q), val_idx, ref)
    return {"val": r["val"], "test": r["test"]}


def requant(q, layers=LAYERS):
    """Round the float weights exactly at the scales in q."""
    out = copy.deepcopy(q)
    for n in layers:
        w = fl[n]["weight"]
        s = out[n]["scale"].reshape((-1,) + (1,) * (w.ndim - 1))
        out[n]["q"] = np.clip(np.rint(w / s), -QMAX, QMAX).astype(np.int32)
    return out


def with_scales(n, scales, base=q0):
    out = copy.deepcopy(base)
    out[n]["scale"] = scales.astype(np.float32)
    return requant(out, [n])


def outlier_ratio(w):
    a = np.sort(np.abs(w.reshape(w.shape[0], -1)), 1)
    return a[:, -1] / a[:, -2]


def mean_abs_E(n, q):
    w = fl[n]["weight"]
    s = q[n]["scale"].reshape((-1,) + (1,) * (w.ndim - 1))
    e = q[n]["q"] - w / s
    E = e.sum((2, 3)) if w.ndim == 4 else e.sum(1)
    return float(np.abs(E).mean())


def mse_scales(n, grid=np.linspace(0.5, 1.0, 201)):
    w = fl[n]["weight"].reshape(fl[n]["weight"].shape[0], -1).astype(np.float64)
    smax = np.abs(w).max(1) / QMAX
    best = np.empty(len(w))
    for r in range(len(w)):
        errs = [((w[r] - a * smax[r] * np.clip(np.rint(w[r] / (a * smax[r])), -QMAX, QMAX)) ** 2).sum() for a in grid]
        best[r] = grid[int(np.argmin(errs))] * smax[r]
    return best


out = {"rtn": acc(q0)}

# MSE-optimal per-row scale search
mse = copy.deepcopy(q0)
for n in LAYERS:
    mse[n]["scale"] = mse_scales(n).astype(np.float32)
mse = requant(mse)
mse_c4fc = requant({n: (copy.deepcopy(mse[n]) if n in ("conv4", "fc") else copy.deepcopy(q0[n])) for n in LAYERS})
out["mse_search"] = {
    "all_layers": acc(mse), "conv4_fc_only": acc(mse_c4fc),
    "median_scale_ratio": {n: round(float(np.median(mse[n]["scale"] / q0[n]["scale"])), 4) for n in LAYERS}}
print("MSE search", out["mse_search"])

runs = {}
for tag in ("opus", "fable"):
    _, rows, bad = parse_rows(Path(f"llm/fp32_bn/reply_{tag}.txt").read_text())
    q, used, fb = build(q0, rows, bad)
    d = {"model": acc(q), "exact_rounding_at_own_scales": acc(requant(q)), "swap": {}, "corr_scale_vs_outlier": {},
         "median_scale_ratio": {}, "mean_abs_E": {}, "mean_abs_E_rtn": {}}
    for n in LAYERS:
        mix = copy.deepcopy(q0)
        mix[n] = copy.deepcopy(q[n])
        a = acc(mix)
        d["swap"][n] = {**a, "delta_test": round(a["test"] - out["rtn"]["test"], 2)}
        ratio = q[n]["scale"] / q0[n]["scale"]
        d["median_scale_ratio"][n] = round(float(np.median(ratio)), 4)
        d["corr_scale_vs_outlier"][n] = round(float(np.corrcoef(ratio, outlier_ratio(fl[n]["weight"]))[0, 1]), 3)
        d["mean_abs_E"][n] = round(mean_abs_E(n, q), 3)
        d["mean_abs_E_rtn"][n] = round(mean_abs_E(n, q0), 3)
    no_c1 = copy.deepcopy(q)
    no_c1["conv1"] = copy.deepcopy(q0["conv1"])
    d["without_conv1_rows"] = acc(no_c1)
    runs[tag] = (q, d)
    out[tag] = d
    print(tag, json.dumps(d))

# conv2 controls against Fable's conv2 scales
qf = runs["fable"][0]
ratio_f = qf["conv2"]["scale"] / q0["conv2"]["scale"]
rng = np.random.default_rng(0)
draws = [acc(with_scales("conv2", q0["conv2"]["scale"] * rng.uniform(0.88, 1.0, len(ratio_f))))["test"]
         for _ in range(12)]
shuf = [acc(with_scales("conv2", q0["conv2"]["scale"] * rng.permutation(ratio_f)))["test"] for _ in range(12)]
out["conv2_controls"] = {
    "fable_conv2_exact": acc(with_scales("conv2", qf["conv2"]["scale"])),
    "random_factors_0.88_1": {"mean": round(float(np.mean(draws)), 2), "min": min(draws), "max": max(draws), "draws": draws},
    "fable_ratios_shuffled": {"mean": round(float(np.mean(shuf)), 2), "min": min(shuf), "max": max(shuf), "draws": shuf}}
print("conv2 controls", out["conv2_controls"])
Path("results/fp32_bn_diag.json").write_text(json.dumps(out, indent=1))
