"""SQuant (Guo et al., ICLR 2022) on our network, as a literature reference for kernel balancing.

SQuant-K: per 3x3 kernel, flip the k = round(|sum e|) elements whose rounding error has the sign of the sum and the
largest magnitude (identical to our kernel balancing). SQuant-C: per output channel, flip k = round(|sum e|) more
codes, one in each of the k kernels whose own sum has the channel's sign and is largest in magnitude (the element
with the largest same-sign error in that kernel). For fc (1x1 kernels) SQuant-C flips single elements.
-> results/squant.json
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
from evaluate import evaluate  # noqa: E402
from ptq_eval import load_layers  # noqa: E402
from quant import LAYERS, dequantize, qmax_for, rtn_quantize  # noqa: E402

fl = load_layers("exports/float32_folded.npz")
ref = np.load("exports/fp32_test_preds.npy")
val_idx = np.load("exports/val_idx.npy")


def flip(q, u, idx_mask, qmax):
    """Flip one code per row of q (2-D), choosing among idx_mask the largest error with the sign of the row sum."""
    e = q - u
    s = np.sign(e.sum(1, keepdims=True))
    cand = (e * s > 0) & idx_mask & (np.abs(q - s) <= qmax)
    score = np.where(cand, np.abs(e), -np.inf)
    j = score.argmax(1)
    rows = np.where(np.isfinite(score[np.arange(len(q)), j]))[0]
    q[rows, j[rows]] -= s[rows, 0].astype(q.dtype)


def squant_layer(w, scale, qmax, kernel_step=True, channel_step=True):
    u = (w / scale.reshape((-1,) + (1,) * (w.ndim - 1))).astype(np.float64)
    q = np.clip(np.rint(u), -qmax, qmax).astype(np.int64)
    if w.ndim == 4:
        o, c = q.shape[:2]
        qk, uk = q.reshape(o * c, 9), u.reshape(o * c, 9)
        if kernel_step:
            for _ in range(9):  # k flips per kernel, one per pass
                need = np.abs(np.rint((qk - uk).sum(1))) >= 1
                if not need.any():
                    break
                sub_q = qk[need]
                flip(sub_q, uk[need], np.ones_like(sub_q, bool), qmax)
                qk[need] = sub_q
        if channel_step:
            qc, uc = qk.reshape(o, c, 9), uk.reshape(o, c, 9)
            for m in range(o):
                ks = (qc[m] - uc[m]).sum(1)
                tot = ks.sum()
                k = int(abs(np.rint(tot)))
                same = np.where(np.sign(ks) == np.sign(tot))[0]
                for n in same[np.argsort(-np.abs(ks[same]))][:k]:
                    row_q = qc[m, n][None]
                    e = row_q - uc[m, n][None]
                    s = np.sign(tot)
                    cand = (e * s > 0) & (np.abs(row_q - s) <= qmax)
                    if cand.any():
                        j = np.where(cand[0], np.abs(e[0]), -np.inf).argmax()
                        qc[m, n, j] -= int(s)
            qk = qc.reshape(o * c, 9)
        return qk.reshape(q.shape)
    if channel_step:  # fc: channel step on single elements
        for _ in range(q.shape[1]):
            need = np.abs(np.rint((q - u).sum(1))) >= 1
            if not need.any():
                break
            sub_q = q[need]
            flip(sub_q, u[need], np.ones_like(sub_q, bool), qmax)
            q[need] = sub_q
    return q


def run(bits, **kw):
    q0 = rtn_quantize(fl, bits)
    q = copy.deepcopy(q0)
    for n in LAYERS:
        q[n]["q"] = squant_layer(fl[n]["weight"], q0[n]["scale"], qmax_for(bits), **kw).astype(np.int32)
    r, _ = evaluate(dequantize(q), val_idx, ref)
    changed = int(sum((q[n]["q"] != q0[n]["q"]).sum() for n in LAYERS))
    return {"val": r["val"], "test": r["test"], "codes_changed": changed}


out = {}
for bits in (4, 3):
    out[f"int{bits}"] = {"squant_K": run(bits, channel_step=False),
                         "squant_KC": run(bits),
                         "squant_C": run(bits, kernel_step=False)}
    print(bits, out[f"int{bits}"])
Path("results/squant.json").write_text(json.dumps(out, indent=1))
