"""Experiment 5 input: the model gets ONLY the float32 weights (no RTN codes, no scales) and has to write the
4-bit network itself, row by row, chunk by chunk.

python llm/make_fp32_direct.py   -> llm/fp32_direct/{task.txt, chunk01.txt ...}

Chunks (at most 1,152 weights = at most 1,152 codes to write each): conv1 | conv2 | conv3 in 4x8 rows |
conv4 in 12x4 rows | fc  -> 19 chunks.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptq_eval import load_layers  # noqa: E402

ROWS = {"conv1": 8, "conv2": 16, "conv3": 8, "conv4": 4, "fc": 10}
OUT = ROOT / "llm/fp32_direct"

TASK = """\
You are given the complete trained float32 weights of a small convolutional network for CIFAR-10
(10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). Your task: turn it into a
4-bit weight-quantized network that classifies as accurately as possible, by writing the 4-bit network yourself.

RULES
- No tools: no code execution, no calculator, no files, no internet. Do every computation in your own
  reasoning and write the numbers directly.
- You cannot run the network. There is no data and there are no BatchNorm statistics: the weights in the chunk
  files are all you get.
- The result must be a valid 4-bit network: every layer is stored as integer codes q in [-7, 7] and one positive
  float scale per output row, and the effective weight is w = q * scale[row]. Biases stay float32 exactly as given
  (you may not change them). The architecture stays as it is. You choose the scale of every row and every code.

NETWORK
Input: 3x32x32 RGB image, normalized per channel with mean (0.4914, 0.4822, 0.4465), std (0.2470, 0.2435, 0.2616).
  conv1: Conv2d 3->8,   3x3, padding 1, bias, then ReLU
  conv2: Conv2d 8->16,  3x3, padding 1, bias, then ReLU, then MaxPool 2x2
  conv3: Conv2d 16->32, 3x3, padding 1, bias, then ReLU, then MaxPool 2x2
  conv4: Conv2d 32->48, 3x3, padding 1, bias, then ReLU, then global average pooling
  fc:    Linear 48->10 (logits; prediction = argmax)
The network was trained with BatchNorm after every conv; BatchNorm has been folded into the conv weights and
biases, so the network above is complete as written. On a held-out split the float32 network scores {fp32_val:.2f}%.

DUMP FORMAT (in the chunk files)
For every layer and output row r:
  "<layer> r<r> bias <b>"                                       the float bias (stays as given)
  "<layer> r<r> w ...", one or more lines                       the FLOAT32 weights of that row (3 decimals)
conv layers: values are grouped per input channel; each group is "c<i>:" followed by the 3x3 kernel in
row-major order (position p = 3*kh + kw, 0..8). Long rows are split over several lines.
fc: values are input features 0..47 in order, split over lines of 16 ("i0:" marks the first index on the line).

YOUR ANSWER: ONE LINE PER OUTPUT ROW
  ROW <layer> <r> SCALE <positive float> Q <codes>
<codes> are the integer codes of that row, separated by single spaces, in exactly the order of the weights in the
dump (conv rows: input channel 0 kernel p=0..8, then input channel 1 kernel, ...; no "c<i>:" labels, no "|";
fc rows: inputs 0..47). A conv row has 9 * (number of input channels) codes (conv1: 27, conv2: 72, conv3: 144,
conv4: 288), an fc row has 48. Example for a conv1 row:  ROW conv1 0 SCALE 0.0635 Q 4 -1 0 -4 2 ... (27 codes).
Write EVERY row of the chunk. A row that is missing, has a wrong number of codes, a code outside [-7, 7] or a
non-positive scale cannot be used and is replaced by plain round-to-nearest quantization of that row.

DELIVERY IN CHUNKS
The dump is NOT in this file. It is delivered in {n} chunk files, read one after another:
{listing}
Work strictly chunk by chunk. In EACH response:
  1. read exactly one chunk file (the next one in order),
  2. think about that chunk only (you may use everything you have seen so far),
  3. write the ROW lines for ALL rows of that chunk as plain text, between a line "BEGIN CHUNK <k>" and a line
     "END CHUNK <k>",
  4. in the same response, read the next chunk file.
Because every chunk gets its own response, you have the budget to write every row of every layer.
After the last chunk, send a final message with a short summary (how you chose the scales and codes, and why) and
"TOOLS USED: <every tool call>". Do not repeat the ROW lines in the final message.
"""


def wline(layer, r, w_row):
    if layer == "fc":
        parts = []
        for i in range(0, len(w_row), 16):
            parts.append(f"{layer} r{r} w i{i}: " + " ".join(f"{v:.3f}" for v in w_row[i:i + 16]))
        return parts
    ks = w_row.reshape(-1, 9)
    groups = [f"c{c}: " + " ".join(f"{v:.3f}" for v in k) for c, k in enumerate(ks)]
    return [f"{layer} r{r} w " + " | ".join(groups[i:i + 8]) for i in range(0, len(groups), 8)]


def main():
    fl = load_layers(ROOT / "exports/float32_folded.npz")
    ptq = __import__("json").loads((ROOT / "results/ptq.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    chunks = []
    for name, rpc in ROWS.items():
        w, b = fl[name]["weight"], fl[name]["bias"]
        for r0 in range(0, w.shape[0], rpc):
            lines = []
            for r in range(r0, min(w.shape[0], r0 + rpc)):
                lines.append(f"{name} r{r} bias {b[r]:.4f}")
                lines += wline(name, r, w[r].reshape(-1))
            chunks.append((name, r0, min(w.shape[0], r0 + rpc), lines))
    listing = "\n".join(f"  chunk{k + 1:02d}.txt: {n} rows {a}-{b - 1}" for k, (n, a, b, _) in enumerate(chunks))
    (OUT / "task.txt").write_text(TASK.format(fp32_val=ptq["fp32"]["val"], n=len(chunks), listing=listing))
    for k, (n, a, b, lines) in enumerate(chunks):
        txt = f"CHUNK {k + 1}/{len(chunks)}: {n} rows {a}-{b - 1}\n" + "\n".join(lines) + "\n"
        (OUT / f"chunk{k + 1:02d}.txt").write_text(txt)
        print(f"chunk{k + 1:02d}: {n:5s} rows {a:2d}-{b - 1:2d}  {len(txt):7,d} chars")
    print("task.txt", len((OUT / "task.txt").read_text()), "chars")


if __name__ == "__main__":
    main()
