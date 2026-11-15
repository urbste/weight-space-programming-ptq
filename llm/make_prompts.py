"""Experiment 2 input: one self-contained task file per bit width with the WHOLE network.

python llm/make_prompts.py   -> llm/prompts/int{8,4,2}.txt

Each file contains the task, the rules, and for every layer/row: the RTN scale, the float bias,
the float32 weights (expressed in units of the row's step, w = u * scale) and the RTN integer codes.
Lines are kept < 1000 chars so file readers do not truncate them.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quant import LAYERS, qmax_for  # noqa: E402
from ptq_eval import load_layers  # noqa: E402

IC_PER_LINE = 8   # conv: input channels per dump line
FC_PER_LINE = 16  # fc: inputs per dump line

HEADER = """\
You are given the complete trained float32 weights of a small convolutional network for CIFAR-10
(10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), together with its
standard {bits}-bit post-training-quantized version. Your task: edit the {bits}-bit network so that it
classifies as accurately as possible.

RULES
- No tools: no code execution, no calculator, no files, no internet. Do every computation in your own
  reasoning and write the numbers directly.
- You cannot run the network. There is no data and there are no BatchNorm statistics: the weights below
  are all you get.
- You may only change the quantized network's parameters: integer codes, per-row scales and per-row biases.
  The result must remain a valid {bits}-bit network: integer codes in [-{qmax}, {qmax}], one positive float
  scale per output row, one float bias per output row. The architecture stays as it is.

NETWORK
Input: 3x32x32 RGB image, normalized per channel with mean (0.4914, 0.4822, 0.4465), std (0.2470, 0.2435, 0.2616).
  conv1: Conv2d 3->8,   3x3, padding 1, bias, then ReLU
  conv2: Conv2d 8->16,  3x3, padding 1, bias, then ReLU, then MaxPool 2x2
  conv3: Conv2d 16->32, 3x3, padding 1, bias, then ReLU, then MaxPool 2x2
  conv4: Conv2d 32->48, 3x3, padding 1, bias, then ReLU, then global average pooling
  fc:    Linear 48->10 (logits; prediction = argmax)
The network was trained with BatchNorm after every conv; BatchNorm has been folded into the conv weights and
biases, so the network above is complete as written.

{bits}-BIT FORMAT AND STARTING POINT
Every layer is stored as integer codes q in [-{qmax}, {qmax}], one positive float scale per output row and one
float bias per row. The effective weight is w = q * scale[row].
The given {bits}-bit network is standard per-channel symmetric round-to-nearest PTQ:
scale[row] = max|w_float[row]| / {qmax}, q = round(w_float / scale[row]), biases copied unchanged.
On a held-out split the float32 network scores {fp32_val:.2f}% and this {bits}-bit network scores {rtn_val:.2f}%.

DUMP FORMAT (below)
For every layer and output row r:
  "<layer> r<r> scale <s> bias <b>"                                   the RTN scale and the float bias
  "<layer> r<r> u ...", one or more lines                              FLOAT32 weights in units of the row's
                                                                        step: w_float = u * scale
  "<layer> r<r> q ...", one or more lines                              the RTN INTEGER codes (q = round(u))
conv layers: values are grouped per input channel; each group is "c<i>:" followed by the 3x3 kernel in
row-major order (position p = 3*kh + kw, 0..8). Long rows are split over several lines (c0-c7, c8-c15, ...).
fc: values are input features 0..47 in order, split over lines of 16 ("i0:" marks the first index on the line).

EDIT LANGUAGE (your answer)
  Q <layer> <row> <in_ch> <pos> <code>     set one conv code (pos 0..8)
  Q fc <row> <in> <code>                   set one fc code
  K <layer> <row> <in_ch> <9 codes>        replace a whole 3x3 kernel (conv only)
  R fc <row> <48 codes>                    replace a whole fc row
  S <layer> <row> <scale>                  set a row's scale (positive float)
  B <layer> <row> <bias>                   set a row's bias (float)
Rows/channels/positions are 0-based. Anything you do not edit stays as in the RTN network. If a parameter
is edited twice, the later line wins. There is no fixed limit on the number of edits, but your whole answer
must fit in a single response, so spend your effort on the edits you expect to matter most.

ANSWER FORMAT
A short explanation (a few sentences) of what you changed and why, then
BEGIN
<edit lines, one per line>
END
TOOLS USED: <every tool call you made>

"""


def fmt_u(v: float, bits: int) -> str:
    s = f"{v:.2f}"
    return "0.00" if s == "-0.00" else s


def dump(fl: dict, ql: dict, bits: int, layers=LAYERS, row_offset: int = 0) -> list[str]:
    lines = []
    for name in layers:
        w, q, scale, bias = fl[name]["weight"], ql[name]["q"], ql[name]["scale"], ql[name]["bias"]
        if len(layers) > 1:
            lines.append(f"\n## {name} weight shape {list(w.shape)}")
        else:  # one chunk of one layer (layer-wise delivery)
            lines.append(f"## {name} rows {row_offset}-{row_offset + w.shape[0] - 1}")
        for r in range(w.shape[0]):  # r indexes the given arrays; printed row id is r + row_offset
            u = (w[r].astype(np.float64) / float(scale[r])).tolist()
            qq = q[r].tolist()
            lines.append(f"{name} r{r + row_offset} scale {float(scale[r]):.6g} bias {float(bias[r]):.4f}")
            if w.ndim == 4:
                nin = w.shape[1]
                for tag, vals, f in (("u", u, lambda v: fmt_u(v, bits)), ("q", qq, str)):
                    for c0 in range(0, nin, IC_PER_LINE):
                        grp = [f"c{c}: " + " ".join(f(x) for x in [v for row in vals[c] for v in row])
                               for c in range(c0, min(nin, c0 + IC_PER_LINE))]
                        lines.append(f"{name} r{r + row_offset} {tag}  " + " | ".join(grp))
            else:
                for tag, vals, f in (("u", u, lambda v: fmt_u(v, bits)), ("q", qq, str)):
                    for i0 in range(0, len(vals), FC_PER_LINE):
                        lines.append(f"{name} r{r + row_offset} {tag}  i{i0}: " + " ".join(f(x) for x in vals[i0:i0 + FC_PER_LINE]))
    return lines


def main():
    fl = load_layers(ROOT / "exports/float32_folded.npz")
    ptq = json.loads((ROOT / "results/ptq.json").read_text())
    out = ROOT / "llm/prompts"
    out.mkdir(parents=True, exist_ok=True)
    for bits in (8, 4, 2):
        ql = load_layers(ROOT / f"exports/rtn_int{bits}.npz")
        head = HEADER.format(bits=bits, qmax=qmax_for(bits), fp32_val=ptq["fp32"]["val"],
                             rtn_val=ptq[f"rtn_int{bits}"]["val"])
        text = head + "\n".join(dump(fl, ql, bits)) + "\n"
        p = out / f"int{bits}.txt"
        p.write_text(text)
        n = text.count("\n")
        print(f"{p}: {len(text):,} chars, {n} lines, longest line {max(map(len, text.splitlines()))}")


if __name__ == "__main__":
    main()
