"""Score one LLM reply for any protocol of the paper, with the same parsing rules as the paper's evaluation.

python llm/score.py <p1|p2|p3|p4> <reply.txt>

p1  edit lines between BEGIN / END                  (eval_llm.py rules)
p2  per-chunk edit blocks BEGIN CHUNK k / END CHUNK k  (eval_layerwise.py rules: the last block per chunk counts)
p3/p4  ROW lines; missing or invalid rows fall back to RTN (eval_fp32_direct.py / eval_fp32_bn.py rules)
Invalid lines are skipped and counted, never repaired. Prints val/test accuracy and agreement with float.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "llm")]
import os  # noqa: E402
os.chdir(ROOT)  # data/ and exports/ are relative to the repository root
from evaluate import evaluate  # noqa: E402
from ptq_eval import load_layers  # noqa: E402
from quant import LAYERS, dequantize  # noqa: E402


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("p1", "p2", "p3", "p4"):
        sys.exit(__doc__)
    proto, text = sys.argv[1], Path(sys.argv[2]).read_text()
    q0 = load_layers(str(ROOT / "exports/rtn_int4.npz"))
    ref = np.load(ROOT / "exports/fp32_test_preds.npy")
    val_idx = np.load(ROOT / "exports/val_idx.npy")
    if proto in ("p1", "p2"):
        from eval_llm import apply_edits, parse_edits
        if proto == "p1":
            lines = parse_edits(text)
        else:
            from eval_layerwise import chunk_blocks
            blocks = chunk_blocks(text)
            lines = [l for k in sorted(blocks) for l in blocks[k]]
        q, st = apply_edits(q0, lines, 4)
        info = {"edit_lines": st["lines"], "applied": st["applied"], "invalid": len(st["invalid"])}
    else:
        from eval_fp32_direct import build, parse_rows
        blocks, rows, bad = parse_rows(text)
        q, used, fallback = build(q0, rows, bad)
        info = {"rows_valid": used, "rows_total": sum(q0[n]["q"].shape[0] for n in LAYERS),
                "rows_fallback_rtn": len(fallback), "unparsed_lines": len(bad)}
    info["integer_weights_changed_vs_rtn"] = int(sum((q[n]["q"] != q0[n]["q"]).sum() for n in LAYERS))
    r, _ = evaluate(dequantize(q), val_idx, ref)
    print(json.dumps({"protocol": proto, "val": r["val"], "test": r["test"], "agree_fp32": r["agree_fp32"], **info},
                     indent=1))
    print("reference (test %): float 81.42, RTN int4 60.39, exact kernel balancing 77.80, AdaRound 80.71")


if __name__ == "__main__":
    main()
