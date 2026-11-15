"""Follow-up to a finished run_fp32_bn_cli.sh run: ask the model, in the same session, to resend the rows
that were missing or unusable. This is a protocol deviation (the main runs never get feedback); its result is
scored and reported separately.

python llm/request_missing_rows.py --tag haiku_high --sid <session id> --model claude-haiku-5-5 --effort high

One message per chunk that has missing/invalid rows (keeps each answer short). The message names the rows and
the expected code count, nothing about scales or values. Answers go to llm/fp32_bn/cli_log_<tag>/fix<k>.txt.
Output: llm/fp32_bn/reply_<tag>_fixed.txt = the original reply plus the fix blocks, renumbered as chunk 100+k so
the scorer reads them last (a later row replaces an earlier one; nothing else is changed).
Score: python llm/eval_fp32_bn.py llm/fp32_bn/reply_<tag>_fixed.txt
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "llm")]
import eval_fp32_bn as E  # noqa: E402  (chdirs to ROOT)
from ptq_eval import load_layers  # noqa: E402

D = Path("llm/fp32_bn")
LEN = {"conv1": 27, "conv2": 72, "conv3": 144, "conv4": 288, "fc": 48}


def chunk_rows():
    out = {}
    for f in sorted(D.glob("chunk*.txt")):
        m = re.match(r"CHUNK (\d+)/19: (\w+) rows (\d+)-(\d+)", f.read_text().splitlines()[0])
        k, layer, a, b = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        out[k] = [(layer, r) for r in range(a, b + 1)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--limit", type=int, default=3600)
    a = ap.parse_args()
    reply = D / f"reply_{a.tag}.txt"
    log = D / f"cli_log_{a.tag}"
    q0 = load_layers("exports/rtn_int4.npz")
    _, rows, bad = E.parse_rows(reply.read_text())
    _, _, fallback = E.build(q0, rows, bad)
    why = {(layer, r): w for layer, r, w in fallback}
    fixed_text = reply.read_text()
    for k, rr in chunk_rows().items():
        todo = [(layer, r) for layer, r in rr if (layer, r) in why]
        if not todo:
            continue
        lines = [f"{layer} r{r}: {why[(layer, r)].replace('length', 'had').replace(' != ', ' codes, needs ')}"
                 for layer, r in todo]
        msg = (f"Corrections for chunk {k}. These rows of chunk {k} are missing or not usable:\n" + "\n".join(lines)
               + f"\nEvery {todo[0][0]} row needs exactly {LEN[todo[0][0]]} codes. Write ONLY these rows, as ROW lines, "
               f"between a line \"BEGIN CHUNK {k}\" and a line \"END CHUNK {k}\", and nothing else.")
        print(f"== chunk {k}: {len(todo)} rows", flush=True)
        cmd = ["timeout", str(a.limit), "claude", "--model", a.model, "--effort", a.effort, "--tools", "",
               "--strict-mcp-config", "-p", "--resume", a.sid, msg]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        (log / f"fix{k:02d}.txt").write_text(out)
        if not out.strip() or "hit your" in out[:200] or out.startswith("API Error"):
            print("!! stopped:", out[:300]); break
        print(f"   rows in answer: {sum(1 for x in out.splitlines() if x.startswith('ROW'))}", flush=True)
        fixed_text += "\n" + re.sub(r"^(BEGIN|END) CHUNK (\d+)\s*$", lambda m: f"{m.group(1)} CHUNK {100 + int(m.group(2))}",
                                    out, flags=re.M) + "\n"
    out_path = D / f"reply_{a.tag}_fixed.txt"
    out_path.write_text(fixed_text)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
