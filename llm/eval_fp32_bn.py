"""Experiment 6 evaluation: the model wrote the int4 network itself from the float32 weights.

python llm/eval_fp32_bn.py <subagent transcript .jsonl>   (first call: extracts the visible assistant text
    and a tool audit, then evaluates)
python llm/eval_fp32_bn.py llm/fp32_bn/reply_<tag>.txt          (CLI-driver run: scores that reply file -> results/fp32_bn_<tag>.json)
python llm/eval_fp32_bn.py                                (re-evaluates llm/fp32_bn/reply_opus.txt)

Valid ROW lines replace the row; missing / invalid rows fall back to plain RTN and are counted. Never repaired.
-> results/fp32_bn.json, results/tool_audit_fp32_bn.json
"""
import copy
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "llm")]
os.chdir(ROOT)
from evaluate import evaluate  # noqa: E402
from ptq_eval import load_layers  # noqa: E402
from quant import LAYERS, dequantize, qmax_for  # noqa: E402

REPLY = ROOT / "llm/fp32_bn/reply_opus.txt"
BITS = 4


def extract(transcript: Path):
    seen, texts, calls, out_tokens, responses = set(), [], [], 0, 0
    for line in transcript.read_text().splitlines():
        d = json.loads(line)
        if d.get("type") != "assistant":
            continue
        m = d["message"]
        for b in m.get("content", []):
            if b.get("type") == "text":
                texts.append((m.get("id"), b["text"]))
            elif b.get("type") == "tool_use":
                calls.append({"tool": b["name"], "input": b["input"]})
        mid = m.get("id")
        if mid not in seen:
            seen.add(mid)
            responses += 1
        # usage is repeated per streamed block of one message; keep the last value per message id
    last = {}
    for line in transcript.read_text().splitlines():
        d = json.loads(line)
        if d.get("type") == "assistant":
            last[d["message"].get("id")] = d["message"].get("usage", {}).get("output_tokens", 0)
    out_tokens = sum(last.values())
    uniq, seen_t = [], set()
    for mid, t in texts:
        if (mid, t) not in seen_t:
            seen_t.add((mid, t))
            uniq.append(t)
    return "\n".join(uniq), calls, {"responses": responses, "output_tokens": out_tokens}


def audit(calls):
    allowed = re.compile(r".*/llm/fp32_bn/(task|chunk\d\d)\.txt$")
    viol = [c for c in calls if c["tool"] != "Read" or not allowed.match(str(c["input"].get("file_path", "")))]
    return {"tool_calls": len(calls), "reads": [Path(c["input"].get("file_path", "?")).name for c in calls],
            "violations": viol}


ROW = re.compile(r"^ROW\s+(\w+)\s+(\d+)\s+SCALE\s+([-+0-9.eE]+)\s+Q\s+(.*)$")


def parse_rows(text):
    blocks = {}
    for m in re.finditer(r"^BEGIN CHUNK (\d+)\s*$(.*?)^END CHUNK \1\s*$", text, re.S | re.M):
        blocks[int(m.group(1))] = m.group(2)
    rows, bad = {}, []
    for k in sorted(blocks):
        for ln in blocks[k].splitlines():
            ln = ln.strip().replace("`", "")
            if not ln.startswith("ROW"):
                continue
            m = ROW.match(ln)
            if not m:
                bad.append((k, ln[:80], "format"))
                continue
            layer, r, s, codes = m.group(1), int(m.group(2)), m.group(3), m.group(4)
            rows[(layer, r)] = (k, s, codes)  # later block wins
    return blocks, rows, bad


def build(q0, rows, bad):
    qmax = qmax_for(BITS)
    q = copy.deepcopy(q0)
    used, fallback = 0, []
    for layer in LAYERS:
        n_rows, row_len = q0[layer]["q"].shape[0], int(np.prod(q0[layer]["q"].shape[1:]))
        for r in range(n_rows):
            item = rows.get((layer, r))
            why = None
            if item is None:
                why = "missing"
            else:
                _, s, codes = item
                try:
                    s = float(s)
                    c = [int(x) for x in codes.replace(",", " ").split()]
                    if not s > 0:
                        why = "scale<=0"
                    elif len(c) != row_len:
                        why = f"length {len(c)} != {row_len}"
                    elif max(map(abs, c)) > qmax:
                        why = "code out of range"
                except ValueError:
                    why = "not numeric"
            if why:
                fallback.append((layer, r, why))
                continue
            q[layer]["q"][r] = np.array(c, dtype=np.int32).reshape(q0[layer]["q"].shape[1:])
            q[layer]["scale"][r] = np.float32(s)
            used += 1
    return q, used, fallback


def main():
    global REPLY
    tag = "opus"
    if len(sys.argv) > 1 and sys.argv[1].endswith(".txt"):
        REPLY = Path(sys.argv[1]).resolve()
        tag = REPLY.stem.replace("reply_", "")
        sys.argv = sys.argv[:1]  # a reply text file: no transcript to extract, no tool audit
    if len(sys.argv) > 1:
        text, calls, usage = extract(Path(sys.argv[1]))
        REPLY.write_text(text)
        a = audit(calls) | usage
        Path("results/tool_audit_fp32_bn.json").write_text(json.dumps(a, indent=1))
        print("tool calls", a["tool_calls"], "violations", len(a["violations"]), usage)
    text = REPLY.read_text()
    q0 = load_layers(f"exports/rtn_int{BITS}.npz")
    fl = load_layers("exports/float32_folded.npz")
    ref = np.load("exports/fp32_test_preds.npy")
    val_idx = np.load("exports/val_idx.npy")
    blocks, rows, bad = parse_rows(text)
    q, used, fallback = build(q0, rows, bad)
    total = sum(q0[n]["q"].shape[0] for n in LAYERS)
    r, _ = evaluate(dequantize(q), val_idx, ref)
    # how different from RTN?
    cmp = {}
    for n in LAYERS:
        same = float((q[n]["q"] == q0[n]["q"]).mean())
        ratio = q[n]["scale"] / q0[n]["scale"]
        cmp[n] = {"codes_equal_rtn": round(same, 4), "scale_ratio_median": round(float(np.median(ratio)), 4),
                  "scale_ratio_min": round(float(ratio.min()), 4), "scale_ratio_max": round(float(ratio.max()), 4),
                  "zero_frac": round(float((q[n]["q"] == 0).mean()), 4),
                  "zero_frac_rtn": round(float((q0[n]["q"] == 0).mean()), 4)}
    res = {**r, "rows_total": total, "rows_used": used, "rows_fallback_rtn": len(fallback),
           "fallback": fallback[:60], "unparsed_lines": bad[:20], "chunks_answered": sorted(blocks),
           "vs_rtn": cmp}
    Path(f"results/fp32_bn_{tag}.json").write_text(json.dumps(res, indent=1))
    print(f"rows used {used}/{total}, fallback to RTN {len(fallback)}, unparsed {len(bad)}")
    print(f"val {r['val']:.2f}  test {r['test']:.2f}  agree {r['agree_fp32']:.2f}")
    for n, c in cmp.items():
        print(n, c)


if __name__ == "__main__":
    main()
