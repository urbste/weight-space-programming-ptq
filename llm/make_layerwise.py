"""Experiment 4 input: the same 4-bit task, but the network is delivered in chunks (layer by layer, large layers
split into row groups). The agent handles ONE chunk per response, so the output/reasoning budget of a single
response no longer limits how much of the network it can edit.

python llm/make_layerwise.py [bits]   -> llm/layerwise/int<bits>/{task.txt, chunk01.txt ...}

Chunks (int4 model 8-16-32-48): conv1 | conv2 | conv3 rows 0-15 | conv3 rows 16-31 | conv4 rows 0-7 ... 40-47 | fc
-> 11 chunks, at most 256 3x3 kernels each.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "llm"))
from make_prompts import HEADER, dump  # noqa: E402
from ptq_eval import load_layers  # noqa: E402
from quant import qmax_for  # noqa: E402

ROWS_PER_CHUNK = {"conv1": 8, "conv2": 16, "conv3": 16, "conv4": 8, "fc": 10}

PROTOCOL = """\
DELIVERY IN CHUNKS (this experiment)
The dump is NOT in this file. It is delivered in {n} chunk files, read one after another:
{listing}
Work strictly chunk by chunk. In EACH response:
  1. read exactly one chunk file (the next one in order),
  2. think about that chunk only (you may use everything you have seen so far),
  3. write the edit lines for the rows of that chunk as plain text, between a line "BEGIN CHUNK <k>" and a
     line "END CHUNK <k>" (write "BEGIN CHUNK <k>" / "END CHUNK <k>" with nothing between if you make no edits),
  4. in the same response, read the next chunk file.
Because every chunk gets its own response, you have the budget to treat EVERY row of every layer; you do not need
to restrict yourself to a few rows or layers. After the last chunk, send a final message with a short summary
(what you changed and why) and "TOOLS USED: <every tool call>". Do not repeat the edit lines in the final message.
"""


def main(bits=4):
    fl = load_layers(ROOT / "exports/float32_folded.npz")
    ql = load_layers(ROOT / f"exports/rtn_int{bits}.npz")
    ptq = json.loads((ROOT / "results/ptq.json").read_text())
    out = ROOT / f"llm/layerwise/int{bits}"
    out.mkdir(parents=True, exist_ok=True)

    chunks = []
    for name, rpc in ROWS_PER_CHUNK.items():
        nrows = ql[name]["q"].shape[0]
        for r0 in range(0, nrows, rpc):
            r1 = min(nrows, r0 + rpc)
            sub_f = {name: {k: v[r0:r1] for k, v in fl[name].items()}}
            sub_q = {name: {k: v[r0:r1] for k, v in ql[name].items()}}
            lines = dump(sub_f, sub_q, bits, layers=(name,), row_offset=r0)
            chunks.append((name, r0, r1, lines))

    listing = "\n".join(f"  chunk{k + 1:02d}.txt: {n} rows {a}-{b - 1}" for k, (n, a, b, _) in enumerate(chunks))
    head = HEADER.format(bits=bits, qmax=qmax_for(bits), fp32_val=ptq["fp32"]["val"],
                         rtn_val=ptq[f"rtn_int{bits}"]["val"])
    head = head.replace("DUMP FORMAT (below)", "DUMP FORMAT (in the chunk files)")
    head = head.replace("must fit in a single response, so spend your effort on the edits you expect to matter most.",
                        "is spread over one response per chunk (see below).")
    head = head.replace("A short explanation (a few sentences) of what you changed and why, then\nBEGIN\n"
                        "<edit lines, one per line>\nEND\nTOOLS USED: <every tool call you made>\n",
                        "Per chunk, as described under DELIVERY IN CHUNKS.\n")
    (out / "task.txt").write_text(head + PROTOCOL.format(n=len(chunks), listing=listing))
    for k, (n, a, b, lines) in enumerate(chunks):
        txt = f"CHUNK {k + 1}/{len(chunks)}: {n} rows {a}-{b - 1}\n" + "\n".join(lines) + "\n"
        (out / f"chunk{k + 1:02d}.txt").write_text(txt)
        print(f"chunk{k + 1:02d}: {n:5s} rows {a:2d}-{b - 1:2d}  {len(txt):7,d} chars  {txt.count(chr(10))} lines")
    print(out / "task.txt", len((out / "task.txt").read_text()), "chars")


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
