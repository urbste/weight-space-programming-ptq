"""Token budget: how compactly can the network be handed to a tool-free LLM, and what precision is needed?

python llm/compression_study.py   -> results/compression.json

(1) Size of alternative text encodings of the int4 network (characters; tokens scale roughly with characters
    for digit-heavy text).
(2) Information needed: run the exact kernel-balancing method (all conv layers + fc rows) on a degraded view of
    the float weights, i.e. what the LLM would see with a coarser encoding.
"""
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
from quant import LAYERS, dequantize  # noqa: E402
from reference_methods import CONVS, balance_groups, u_of  # noqa: E402

BITS, QMAX = 4, 7


def encodings(fl, ql):
    """Characters needed for the weight payload (rows only, no header) in several encodings."""
    n_w = sum(fl[n]["weight"].size for n in LAYERS)
    U = {n: u_of(fl, ql, n) for n in LAYERS}
    Q = {n: ql[n]["q"] for n in LAYERS}
    R = {n: U[n] - Q[n] for n in LAYERS}
    sep = 1  # one space per value
    sizes = {}
    sizes["current: u (2 dec) + q"] = sum(sum(len(f"{v:.2f}") + sep for v in U[n].ravel()) +
                                         sum(len(str(int(v))) + sep for v in Q[n].ravel()) for n in LAYERS)
    sizes["u only (2 dec), q = round(u)"] = sum(sum(len(f"{v:.2f}") + sep for v in U[n].ravel()) for n in LAYERS)
    sizes["u only (1 dec)"] = sum(sum(len(f"{v:.1f}") + sep for v in U[n].ravel()) for n in LAYERS)
    # q plus residual digit: one char for q in [-7,7] (A..O) + one digit for round(10*r) in [-5,5] (a..k)
    sizes["packed: 2 chars per weight (code letter + residual letter), no separators"] = 2 * n_w
    # sparse: only weights whose residual is near the rounding boundary (|r| >= 0.3) can be worth flipping
    cand = sum(int((np.abs(R[n]) >= 0.3).sum()) for n in LAYERS)
    sizes[f"sparse: only |r|>=0.3 candidates ({cand} of {n_w}) as 'pos:r', + per-kernel sum"] = \
        cand * 6 + sum(R[n].reshape(R[n].shape[0], -1).shape[0] * (R[n].shape[1] if R[n].ndim == 4 else 1) * 5
                       for n in LAYERS)
    return {k: {"chars": int(v), "relative": round(v / sizes["current: u (2 dec) + q"], 3)} for k, v in sizes.items()}, n_w


def balanced_from_view(fl, ql, view_fn):
    """Exact kernel balancing (all convs) + fc row balancing, but computed from a degraded view of u."""
    out = {n: {k: v.copy() for k, v in ql[n].items()} for n in ql}
    for n in LAYERS:
        u = view_fn(u_of(fl, ql, n))
        q = ql[n]["q"]
        if n in CONVS:
            o, c = q.shape[:2]
            out[n]["q"] = balance_groups(u.reshape(o * c, 9), q.reshape(o * c, 9), QMAX).reshape(q.shape)
        else:
            out[n]["q"] = balance_groups(u, q, QMAX)
        out[n]["q"] = out[n]["q"].astype(np.int32)
    return out


def main():
    fl = load_layers("exports/float32_folded.npz")
    ql = load_layers(f"exports/rtn_int{BITS}.npz")
    ref = np.load("exports/fp32_test_preds.npy")
    val_idx = np.load("exports/val_idx.npy")
    enc, n_w = encodings(fl, ql)
    print(f"weights: {n_w}")
    for k, v in enc.items():
        print(f"  {v['chars']:8,d} chars  ({v['relative']:.2f}x)  {k}")

    views = {
        "exact float": lambda u: u,
        "u to 2 decimals (current file)": lambda u: np.round(u, 2),
        "u to 1 decimal": lambda u: np.round(u, 1),
        "residual in 3 levels (q-0.33/q/q+0.33)": lambda u: np.rint(u) + np.round((u - np.rint(u)) * 3) / 3,
        "residual sign only (q +- 0.25)": lambda u: np.rint(u) + 0.25 * np.sign(np.round(u - np.rint(u), 6)),
        "no float info (u = q)": lambda u: np.rint(u),
    }
    acc = {}
    for k, f in views.items():
        r, _ = evaluate(dequantize(balanced_from_view(fl, ql, f)), val_idx, ref)
        acc[k] = {"val": r["val"], "test": r["test"]}
        print(f"  balance all layers from view '{k}': val {r['val']:.2f} test {r['test']:.2f}")
    Path("results/compression.json").write_text(json.dumps({"encodings": enc, "method_from_view": acc}, indent=2))


if __name__ == "__main__":
    main()
