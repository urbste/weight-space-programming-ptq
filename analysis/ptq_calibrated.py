"""Calibrated weight-only PTQ baselines: AdaRound (Nagel et al. 2020) and GPTQ (Frantar et al. 2023).

Both use 1,024 calibration images from the 45k training split (disjoint from val/test), RTN per-row max scales
(only the rounding is optimized), and quantize layer by layer with inputs from the already quantized layers
(sequential / asymmetric reconstruction). Target: float pre-activation output of each layer.
python analysis/ptq_calibrated.py   -> results/ptq_calibrated.json  (GPU, ~minutes)
"""
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "llm")]
os.chdir(ROOT)
from data import load_raw, normalize  # noqa: E402
from evaluate import evaluate  # noqa: E402
from ptq_eval import load_layers  # noqa: E402
from quant import LAYERS, dequantize, qmax_for, rtn_quantize  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_CALIB = 1024
fl = load_layers("exports/float32_folded.npz")
ref = np.load("exports/fp32_test_preds.npy")
val_idx = np.load("exports/val_idx.npy")
(x_tr, _), _ = load_raw()
train_idx = np.setdiff1d(np.arange(50000), val_idx)
rng = np.random.default_rng(0)
calib = torch.tensor(normalize(x_tr[rng.choice(train_idx, N_CALIB, replace=False)]), device=DEV)


def T(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32, device=DEV)


def layer_op(n, x, w, b):
    return F.conv2d(x, w, b, padding=1) if n != "fc" else x @ w.T + b


def post(n, y):
    """Activation/pooling after layer n, producing the input of the next layer."""
    if n == "fc":
        return y
    y = F.relu(y)
    if n in ("conv2", "conv3"):
        y = F.max_pool2d(y, 2)
    if n == "conv4":
        y = y.mean((2, 3))
    return y


@torch.no_grad()
def float_targets():
    """Float pre-activation output of every layer on float inputs."""
    x, out = calib, {}
    for n in LAYERS:
        y = layer_op(n, x, T(fl[n]["weight"]), T(fl[n]["bias"]))
        out[n] = y
        x = post(n, y)
    return out


def adaround_layer(n, x_in, y_t, w, s, b, qmax, iters=10000, lam=0.01, bs=64, lr=1e-3):
    shp = (-1,) + (1,) * (w.dim() - 1)
    u = w / s.view(shp)
    fl_u = torch.floor(u)
    rest = (u - fl_u).clamp(1e-4, 1 - 1e-4)
    zeta, gamma = 1.1, -0.1
    V = (-torch.log((zeta - gamma) / (rest - gamma) - 1)).clone().requires_grad_(True)  # h(V) = rest at start
    opt = torch.optim.Adam([V], lr=lr)

    def h():
        return torch.clamp(torch.sigmoid(V) * (zeta - gamma) + gamma, 0, 1)

    warm = int(0.2 * iters)
    for it in range(iters):
        idx = torch.randint(0, len(x_in), (bs,), device=DEV)
        wq = s.view(shp) * torch.clamp(fl_u + h(), -qmax, qmax)
        rec = F.mse_loss(layer_op(n, x_in[idx], wq, b), y_t[idx])
        if it < warm:
            reg = torch.zeros((), device=DEV)
        else:
            beta = 20 + (2 - 20) * (it - warm) / (iters - warm)
            reg = (1 - (2 * h() - 1).abs().pow(beta)).sum()
        loss = rec + lam * reg
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.clamp(fl_u + (h() >= 0.5).float(), -qmax, qmax)


@torch.no_grad()
def gptq_layer(n, x_in, w, s, qmax, damp=0.01):
    if n == "fc":
        X = x_in
        W = w.clone()
    else:
        X = F.unfold(x_in, 3, padding=1).transpose(1, 2).reshape(-1, w.shape[1] * 9)
        W = w.reshape(w.shape[0], -1).clone()
    H = 2 * X.T.double() @ X.double() / len(X)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(len(H), device=DEV, dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True).float()
    Q = torch.zeros_like(W)
    sc = s.view(-1)
    for j in range(W.shape[1]):
        wj = W[:, j]
        qj = torch.clamp(torch.round(wj / sc), -qmax, qmax)
        Q[:, j] = qj
        err = (wj - qj * sc) / Hinv[j, j]
        W[:, j:] -= err[:, None] * Hinv[j, j:][None, :]
    return Q.reshape(w.shape)


def run(method, bits):
    qmax = qmax_for(bits)
    q0 = rtn_quantize(fl, bits)
    q = copy.deepcopy(q0)
    yt = float_targets()
    x = calib
    for n in LAYERS:
        w, b, s = T(fl[n]["weight"]), T(fl[n]["bias"]), T(q0[n]["scale"])
        if method == "adaround":
            codes = adaround_layer(n, x, yt[n], w, s, b, qmax)
        else:
            codes = gptq_layer(n, x, w, s, qmax)
        q[n]["q"] = codes.cpu().numpy().astype(np.int32)
        with torch.no_grad():
            shp = (-1,) + (1,) * (w.dim() - 1)
            x = post(n, layer_op(n, x, codes * s.view(shp), b))
    r, _ = evaluate(dequantize(q), val_idx, ref)
    changed = int(sum((q[n]["q"] != q0[n]["q"]).sum() for n in LAYERS))
    return {"val": r["val"], "test": r["test"], "agree_fp32": r["agree_fp32"], "codes_changed": changed}


torch.manual_seed(0)
out = {"n_calib": N_CALIB, "scales": "RTN per-row max"}
for bits in (4, 3):
    for m in ("gptq", "adaround"):
        out[f"{m}_int{bits}"] = run(m, bits)
        print(m, bits, out[f"{m}_int{bits}"], flush=True)
Path("results/ptq_calibrated.json").write_text(json.dumps(out, indent=1))
