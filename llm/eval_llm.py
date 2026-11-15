"""Experiment 2 evaluation: apply the edit lines from each tool-free LLM reply and evaluate.

python llm/eval_llm.py   -> reads llm/replies/<model>_int<bits>.txt, writes results/llm.json
"""
import copy
import os
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from evaluate import evaluate  # noqa: E402
from ptq_eval import load_layers  # noqa: E402
from quant import LAYERS, dequantize, qmax_for  # noqa: E402

MODELS = ("opus", "sonnet", "haiku")
BITS = (8, 4, 2)


def parse_edits(text: str) -> list[str]:
    m = re.search(r"^BEGIN\s*$(.*?)^END\s*$", text, re.S | re.M)
    if not m:
        raise ValueError("no BEGIN/END block")
    return [ln.strip() for ln in m.group(1).splitlines() if ln.strip() and not ln.strip().startswith("#")]


def apply_edits(ql: dict, lines: list[str], bits: int) -> tuple[dict, dict]:
    """Returns edited network + stats. Invalid lines are skipped and counted, never 'repaired'."""
    qmax = qmax_for(bits)
    ql = copy.deepcopy(ql)
    st = {"lines": len(lines), "applied": 0, "invalid": [], "by_kind": {}}

    def code(s):
        v = int(s)
        if not -qmax <= v <= qmax:
            raise ValueError(f"code {v} out of range")
        return v

    for ln in lines:
        t = ln.replace(",", " ").split()
        try:
            kind, layer, row = t[0].upper(), t[1], int(t[2])
            if layer not in LAYERS:
                raise ValueError("unknown layer")
            d = ql[layer]
            if not 0 <= row < d["q"].shape[0]:
                raise ValueError("row out of range")
            if kind == "Q" and layer != "fc":
                if len(t) != 6: raise ValueError("arity")
                ic, pos = int(t[3]), int(t[4])
                if not (0 <= ic < d["q"].shape[1] and 0 <= pos < 9): raise ValueError("index")
                d["q"][row, ic, pos // 3, pos % 3] = code(t[5])
            elif kind == "Q":
                if len(t) != 5: raise ValueError("arity")
                i = int(t[3])
                if not 0 <= i < d["q"].shape[1]: raise ValueError("index")
                d["q"][row, i] = code(t[4])
            elif kind == "K" and layer != "fc":
                ic = int(t[3])
                if len(t) != 13 or not 0 <= ic < d["q"].shape[1]: raise ValueError("arity/index")
                d["q"][row, ic] = np.array([code(x) for x in t[4:13]], dtype=np.int32).reshape(3, 3)
            elif kind == "R" and layer == "fc":
                if len(t) != 3 + d["q"].shape[1]: raise ValueError("arity")
                d["q"][row] = np.array([code(x) for x in t[3:]], dtype=np.int32)
            elif kind == "S":
                s = float(t[3])
                if not s > 0: raise ValueError("scale must be > 0")
                d["scale"][row] = s
            elif kind == "B":
                d["bias"][row] = float(t[3])
            else:
                raise ValueError("unknown edit")
            st["applied"] += 1
            st["by_kind"][kind] = st["by_kind"].get(kind, 0) + 1
        except (ValueError, IndexError) as e:
            st["invalid"].append(f"{ln}  <- {e}")
    return ql, st


def diff_stats(q0: dict, q1: dict) -> dict:
    out = {}
    for n in LAYERS:
        out[n] = {"codes_changed": int((q0[n]["q"] != q1[n]["q"]).sum()),
                  "mean_abs_code_change": round(float(np.abs(q0[n]["q"] - q1[n]["q"]).sum()) / max(1, int((q0[n]["q"] != q1[n]["q"]).sum())), 3),
                  "codes_total": int(q0[n]["q"].size),
                  "scales_changed": int((q0[n]["scale"] != q1[n]["scale"]).sum()),
                  "biases_changed": int((q0[n]["bias"] != q1[n]["bias"]).sum())}
    return out


def main():
    ref = np.load("exports/fp32_test_preds.npy")
    val_idx = np.load("exports/val_idx.npy")
    ptq = json.loads(Path("results/ptq.json").read_text())
    results = {}
    for bits in BITS:
        q0 = load_layers(f"exports/rtn_int{bits}.npz")
        for m in MODELS:
            p = Path(f"llm/replies/{m}_int{bits}.txt")
            if not p.exists():
                continue
            key = f"{m}_int{bits}"
            try:
                lines = parse_edits(p.read_text())
            except ValueError as e:
                results[key] = {"error": str(e)}
                print(f"{key}: {e}")
                continue
            q1, st = apply_edits(q0, lines, bits)
            r, _ = evaluate(dequantize(q1), val_idx, ref)
            r["delta_vs_rtn_test"] = round(r["test"] - ptq[f"rtn_int{bits}"]["test"], 2)
            r["edits"] = st
            r["diff"] = diff_stats(q0, q1)
            results[key] = r
            print(f"{key:12s} val {r['val']:.2f}  test {r['test']:.2f}  (RTN {ptq[f'rtn_int{bits}']['test']:.2f}, "
                  f"Δ {r['delta_vs_rtn_test']:+.2f})  agree {r['agree_fp32']:.2f}  "
                  f"lines {st['lines']} applied {st['applied']} invalid {len(st['invalid'])}")
    Path("results/llm.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
