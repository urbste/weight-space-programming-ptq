"""Experiment 4 evaluation: apply the per-chunk edit blocks of each layer-wise reply, cumulatively.

python llm/eval_layerwise.py   -> results/layerwise.json
Reads llm/layerwise/replies/<model>_int4.txt (verbatim assistant text of the whole run). For each chunk k the
last "BEGIN CHUNK k ... END CHUNK k" block counts. Edits outside a chunk's rows are still applied (and counted).
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "llm")]
os.chdir(ROOT)
from eval_llm import apply_edits, diff_stats  # noqa: E402
from evaluate import evaluate  # noqa: E402
from ptq_eval import load_layers  # noqa: E402
from quant import dequantize  # noqa: E402

BITS, N_CHUNKS = 4, 11


def chunk_blocks(text):
    blocks = {}
    for m in re.finditer(r"^BEGIN CHUNK (\d+)\s*$(.*?)^END CHUNK \1\s*$", text, re.S | re.M):
        blocks[int(m.group(1))] = [l.strip() for l in m.group(2).splitlines()
                                   if l.strip() and not l.strip().startswith(("#", "```"))]
    return blocks


def main():
    ref = np.load("exports/fp32_test_preds.npy")
    val_idx = np.load("exports/val_idx.npy")
    q0 = load_layers(f"exports/rtn_int{BITS}.npz")
    res = {}
    for p in sorted(Path("llm/layerwise/replies").glob(f"*_int{BITS}.txt")):
        m = p.name[: -len(f"_int{BITS}.txt")]  # e.g. opus, sonnet, sonnet_run2, haiku
        blocks = chunk_blocks(p.read_text())
        lines, curve, q = [], [], q0
        for k in range(1, N_CHUNKS + 1):
            lines += blocks.get(k, [])
            if k in (1, 2, 4, 10, 11):  # after conv1, conv2, conv3, conv4, fc
                q, st = apply_edits(q0, lines, BITS)
                r, _ = evaluate(dequantize(q), val_idx, ref)
                curve.append({"after_chunk": k, "val": r["val"], "test": r["test"]})
        r, _ = evaluate(dequantize(q), val_idx, ref)
        r.update({"chunks_answered": sorted(blocks), "edits": st, "diff": diff_stats(q0, q), "curve": curve})
        res[m] = r
        print(f"{m:7s} val {r['val']:.2f} test {r['test']:.2f} agree {r['agree_fp32']:.2f}  lines {st['lines']} "
              f"invalid {len(st['invalid'])}  codes/layer {[d['codes_changed'] for d in r['diff'].values()]}  "
              f"curve {[c['test'] for c in curve]}")
    Path("results/layerwise.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
