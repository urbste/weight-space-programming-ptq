"""Exact code implementations of the methods the tool-free LLMs described in their replies.

Separates "wrong method" from "imprecise hand execution": same idea, computed exactly, no data, no forward passes
in the method itself (forward passes only for the final evaluation).

python llm/reference_methods.py   -> results/reference.json
"""
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from evaluate import evaluate  # noqa: E402
from ptq_eval import load_layers  # noqa: E402
from quant import dequantize, qmax_for  # noqa: E402

CONVS = ("conv1", "conv2", "conv3", "conv4")


def u_of(fl, ql, name):
    w, s = fl[name]["weight"].astype(np.float64), ql[name]["scale"].astype(np.float64)
    return w / s.reshape((-1,) + (1,) * (w.ndim - 1))


def balance_groups(u, q, qmax, tol=0.5):
    """Each group = one row of u/q (flattened). Re-round the weights closest to their rounding boundary
    (in the direction that reduces the error sum) until |sum(q - u)| <= tol. Codes stay in [-qmax, qmax]."""
    u2, q2 = u.reshape(len(u), -1), q.reshape(len(q), -1).copy()
    for g in range(len(u2)):
        while True:
            e = q2[g] - u2[g]
            d = e.sum()
            if abs(d) <= tol:
                break
            step = -np.sign(d)                     # move codes down if sum too large
            cand = q2[g] + step
            ok = (np.abs(cand) <= qmax)
            cost = np.abs(cand - u2[g]) - np.abs(e)  # extra per-weight error of flipping
            cost[~ok] = np.inf
            j = int(np.argmin(cost))
            if not np.isfinite(cost[j]) or abs(d + step) >= abs(d):
                break
            q2[g, j] = cand[j]
    return q2.reshape(q.shape)


def kernel_balance(fl, ql, layers, bits):
    """LLM int4 method (Opus/Sonnet): per 3x3 kernel, make sum of codes match sum of float u."""
    ql = copy.deepcopy(ql)
    for n in layers:
        u, q = u_of(fl, ql, n), ql[n]["q"]
        o, c = q.shape[:2]
        ql[n]["q"] = balance_groups(u.reshape(o * c, 9), q.reshape(o * c, 9), qmax_for(bits)).reshape(q.shape).astype(np.int32)
    return ql


def fc_row_balance(fl, ql, bits):
    """LLM int4/int8 method (Haiku/Sonnet): per fc row, make sum of rounding errors ~0."""
    ql = copy.deepcopy(ql)
    ql["fc"]["q"] = balance_groups(u_of(fl, ql, "fc"), ql["fc"]["q"], qmax_for(bits)).astype(np.int32)
    return ql


def ternary(fl, ql, thr: dict, scale_mode="ls", scale_mult: dict | None = None, keep_codes=()):
    """LLM int2 method. u is in units of max|w_row| (RTN 2-bit scale = max|w|).
    Opus: q = sign(u) if |u| >= t, scale = old * mean|u_kept| (least squares for fixed codes).
    Sonnet: q = sign(u) if |u| >= t, scale = old * fixed multiplier; conv4 codes kept, only rescaled."""
    ql = copy.deepcopy(ql)
    for n in ql:
        u = u_of(fl, ql, n)
        rows = u.reshape(len(u), -1)
        if n in keep_codes:
            q = ql[n]["q"].reshape(len(u), -1)
        else:
            q = (np.sign(rows) * (np.abs(rows) >= thr[n])).astype(np.int32)
        if scale_mode == "ls":
            num = (np.abs(rows) * np.abs(q)).sum(1)
            den = np.maximum(np.abs(q).sum(1), 1)
            mult = num / den
        else:
            mult = np.full(len(u), scale_mult[n])
        ql[n]["q"] = q.reshape(ql[n]["q"].shape).astype(np.int32)
        ql[n]["scale"] = (ql[n]["scale"] * mult).astype(np.float32)
    return ql


def twn_classic(fl, ql):
    """Textbook TWN (Li et al. 2016): threshold 0.7 * mean|w| per row, LS scale."""
    thr = {}
    out = copy.deepcopy(ql)
    for n in ql:
        u = np.abs(u_of(fl, ql, n)).reshape(len(ql[n]["scale"]), -1)
        rows = u_of(fl, ql, n).reshape(len(u), -1)
        t = 0.7 * u.mean(1, keepdims=True)
        q = (np.sign(rows) * (u >= t)).astype(np.int32)
        mult = (u * np.abs(q)).sum(1) / np.maximum(np.abs(q).sum(1), 1)
        out[n]["q"] = q.reshape(ql[n]["q"].shape)
        out[n]["scale"] = (ql[n]["scale"] * mult).astype(np.float32)
    return out


def main():
    fl = load_layers("exports/float32_folded.npz")
    ref = np.load("exports/fp32_test_preds.npy")
    val_idx = np.load("exports/val_idx.npy")
    llm = json.loads(Path("results/llm.json").read_text())
    res = {}

    def run(key, ql, q0, note):
        r, _ = evaluate(dequantize(ql), val_idx, ref)
        r["codes_changed"] = int(sum((ql[n]["q"] != q0[n]["q"]).sum() for n in ql))
        r["note"] = note
        res[key] = r
        print(f"{key:34s} val {r['val']:6.2f}  test {r['test']:6.2f}  agree {r['agree_fp32']:6.2f}  "
              f"codes changed {r['codes_changed']:5d}   {note}")

    for bits in (8, 4, 2):
        q0 = load_layers(f"exports/rtn_int{bits}.npz")
        run(f"int{bits}_rtn", q0, q0, "baseline")
        for m in ("opus", "sonnet", "haiku"):
            k = f"{m}_int{bits}"
            if k in llm and "test" in llm[k]:
                print(f"{'  LLM ' + k:34s} val {llm[k]['val']:6.2f}  test {llm[k]['test']:6.2f}  (by hand)")
        if bits in (8, 4):
            run(f"int{bits}_kernelbal_conv1", kernel_balance(fl, q0, ["conv1"], bits), q0,
                "Opus/Sonnet int4 method, exact (conv1 only)")
            run(f"int{bits}_kernelbal_all", kernel_balance(fl, q0, CONVS, bits), q0,
                "same method on all conv layers")
            run(f"int{bits}_fcbal", fc_row_balance(fl, q0, bits), q0, "Haiku int4 / Sonnet int8 method, exact")
            run(f"int{bits}_kernelbal_all+fcbal", fc_row_balance(fl, kernel_balance(fl, q0, CONVS, bits), bits), q0,
                "all conv kernels + fc rows")
        if bits == 2:
            thr_opus = {"conv1": 0.30, "conv2": 0.25, "conv3": 0.25, "conv4": 0.25, "fc": 0.25}
            run("int2_opus_twn", ternary(fl, q0, thr_opus, "ls"), q0, "Opus int2 method, exact (t=.30/.25, LS scale)")
            thr_son = {n: 0.32 for n in q0}
            mult_son = {n: 0.55 for n in q0} | {"conv4": 0.72}
            run("int2_sonnet_ternary", ternary(fl, q0, thr_son, "fixed", mult_son, keep_codes=("conv4",)), q0,
                "Sonnet int2 method, exact (t=.32, scale x0.55, conv4 codes kept x0.72)")
            run("int2_sonnet_ternary_conv4too", ternary(fl, q0, thr_son, "fixed", {n: 0.55 for n in q0}), q0,
                "Sonnet method incl. conv4 codes (what it skipped)")
            hk = ternary(fl, q0, {n: 0.25 for n in q0}, "ls")
            hk = {n: (hk[n] if n == "fc" else copy.deepcopy(q0[n])) for n in q0}  # Haiku changed fc only
            run("int2_haiku_fc_only", hk, q0, "Haiku int2 method, exact (fc only, t=.25, LS scale)")
            run("int2_twn_classic", twn_classic(fl, q0), q0, "textbook TWN, 0.7*mean|w|, LS scale")
    Path("results/reference.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
