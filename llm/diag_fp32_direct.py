"""Diagnostics for Experiment 5: arithmetic accuracy of the written codes and per-layer swap-in effect."""
import copy, json, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "llm")]
os.chdir(ROOT)
from eval_fp32_direct import REPLY, build, parse_rows  # noqa: E402
from evaluate import evaluate  # noqa: E402
from ptq_eval import load_layers  # noqa: E402
from quant import LAYERS, dequantize  # noqa: E402

q0 = load_layers("exports/rtn_int4.npz")
fl = load_layers("exports/float32_folded.npz")
ref = np.load("exports/fp32_test_preds.npy"); val_idx = np.load("exports/val_idx.npy")
_, rows, bad = parse_rows(REPLY.read_text())
q, used, fb = build(q0, rows, bad)
fbset = {(l, r) for l, r, _ in fb}
out = {"arith": {}, "swap": {}}
for n in LAYERS:
    w = fl[n]["weight"]; ok = tot = 0; rtn_ok = 0; changed = []
    for r in range(w.shape[0]):
        if (n, r) in fbset: continue
        s = q[n]["scale"][r]
        exp = np.clip(np.rint(w[r] / s), -7, 7).astype(int)
        got = q[n]["q"][r]
        ok += int((exp == got).sum()); tot += got.size
        diff = np.argwhere(q[n]["q"][r] != q0[n]["q"][r])
        for idx in diff: changed.append((r, [int(i) for i in idx], int(q0[n]["q"][r][tuple(idx)]), int(got[tuple(idx)]), round(float(w[r][tuple(idx)] / s), 2)))
    out["arith"][n] = {"codes_equal_round_w_over_scale": ok / tot, "n_changed_vs_rtn": len(changed), "changed_examples": changed[:12]}
    print(n, f"{ok / tot:.4f} equal round(w/scale);", len(changed), "codes differ from RTN", changed[:6])
for n in LAYERS:
    mix = copy.deepcopy(q0); mix[n] = copy.deepcopy(q[n])
    r, _ = evaluate(dequantize(mix), val_idx, ref)
    out["swap"][n] = {"val": r["val"], "test": r["test"]}
    print("swap in", n, r["val"], r["test"])
Path("results/fp32_direct_diag.json").write_text(json.dumps(out, indent=1))
