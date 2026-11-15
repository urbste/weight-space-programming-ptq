#!/usr/bin/env bash
# Run one protocol (P1-P4) of the paper on a Claude model from the command line, with all tools disabled.
#
#   PROTOCOL=p2 MODEL=claude-opus-5-5 ./llm/run_cli.sh
#   PROTOCOL=p4 MODEL=claude-fable-5-2 EFFORT=high ./llm/run_cli.sh
#   continue after an interruption:  PROTOCOL=p3 SID=<session id> START=12 ./llm/run_cli.sh
#
#   p1  float (in steps) + RTN int4 network in ONE message, answer = sparse edits      (llm/prompts/int4.txt)
#   p2  as p1, task message + 11 chunk messages, one edit block per chunk               (llm/layerwise/int4/)
#   p3  float weights only, task message + 19 chunks, full int4 network                  (llm/fp32_direct/)
#   p4  as p3 plus BatchNorm statistics                                                  (llm/fp32_bn/)
#
# All chunks go into ONE session (--session-id / --resume), so the model keeps everything it has seen.
# The "No tools" rule and the file-reading delivery wording of the original task files are replaced by
# message delivery; the CLI itself enforces the tool restriction (--tools "").
#
# Output:  runs/<protocol>_<tag>/reply.txt   (all assistant answers, in order)
#          runs/<protocol>_<tag>/turnNN.txt   (one file per turn), task.txt (the exact first message), summary.txt
# Score:   python llm/score.py <protocol> runs/<protocol>_<tag>/reply.txt
set -euo pipefail
cd "$(dirname "$0")/.."
PROTOCOL="${PROTOCOL:?set PROTOCOL=p1|p2|p3|p4}"
MODEL="${MODEL:-claude-opus-5-5}"
EFFORT="${EFFORT:-high}"            # claude --effort: low|medium|high|xhigh|max
LIMIT="${LIMIT:-3600}"              # seconds per turn; a stalled request stops the run instead of hanging
TAG="${TAG:-$(echo "$MODEL" | sed -E 's/^claude-//')_${EFFORT}}"
START="${START:-1}"
SID="${SID:-$(python -c 'import uuid;print(uuid.uuid4())')}"
case "$PROTOCOL" in
  p1) TASK=llm/prompts/int4.txt;        CHUNKS=();  N=0  ;;
  p2) TASK=llm/layerwise/int4/task.txt; D=llm/layerwise/int4; N=11 ;;
  p3) TASK=llm/fp32_direct/task.txt;    D=llm/fp32_direct;    N=19 ;;
  p4) TASK=llm/fp32_bn/task.txt;        D=llm/fp32_bn;        N=19 ;;
  *) echo "unknown PROTOCOL $PROTOCOL"; exit 1 ;;
esac
OUT=runs/${PROTOCOL}_${TAG}
mkdir -p "$OUT"
CL=(timeout "$LIMIT" claude --model "$MODEL" --effort "$EFFORT" --tools "" --strict-mcp-config -p)
echo "protocol $PROTOCOL  model $MODEL  effort $EFFORT  session $SID  -> $OUT"

# the exact first message
python - "$PROTOCOL" "$TASK" "$N" <<'PY' > "$OUT/task.txt"
import re, sys
proto, path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
t = open(path).read()
t = re.sub(r"- No tools:.*?\n(?:  .*\n)*", "", t, count=1)
if proto == "p1":
    t = re.sub(r"\nTOOLS USED:.*\n?", "\n", t)
    print(t)
    sys.exit()
t = t.split("\nDELIVERY IN CHUNKS")[0] + "\n"
what = ("the edit lines for the rows of that chunk (nothing between the two lines if you make no edits)"
        if proto == "p2" else "the ROW lines for ALL rows of that chunk")
t += f"""DELIVERY IN CHUNKS
The weight dump is delivered in {n} messages, one chunk per message, after this one. For EACH chunk message, write
{what} as plain text between a line "BEGIN CHUNK <k>" and a line "END CHUNK <k>", and nothing else. You may use
everything you have seen so far. After the last chunk I will ask for a short summary. Now reply only with the word
READY.
"""
print(t)
PY

turn() {  # turn <label> <logfile> <message file> <--session-id|--resume>   (message goes in on stdin)
  local label="$1" file="$2" msg="$3" mode="$4"
  echo "== $label"
  if ! "${CL[@]}" "$mode" "$SID" < "$msg" > "$file"; then
    echo "!! $label failed or timed out (limit ${LIMIT}s):"; head -c 1500 "$file"; echo; exit 1
  fi
  [ -s "$file" ] || { echo "!! $label returned nothing"; exit 1; }
  if head -c 300 "$file" | grep -q '^API Error'; then echo "!! $label: API error:"; head -c 1500 "$file"; echo; exit 1; fi
}

if [ "$START" -le 1 ]; then
  turn "task" "$OUT/turn00.txt" "$OUT/task.txt" --session-id
  cat "$OUT/turn00.txt"
  if [ "$N" -eq 0 ]; then cp "$OUT/turn00.txt" "$OUT/reply.txt"; fi
  [ "$N" -eq 0 ] || : > "$OUT/reply.txt"
else
  echo "continuing session $SID at chunk $START"
fi
for k in $(seq -w "$START" "$N"); do
  [ "$N" -eq 0 ] && break
  turn "chunk $k/$N" "$OUT/turn$k.txt" "$D/chunk$k.txt" --resume
  tee -a "$OUT/reply.txt" < "$OUT/turn$k.txt"
  echo >> "$OUT/reply.txt"
done
if [ "$N" -gt 0 ]; then
  echo "All chunks are done. Give a short summary of what you did and why. Do not repeat the edit or ROW lines." > "$OUT/summary_prompt.txt"
  turn "summary" "$OUT/summary.txt" "$OUT/summary_prompt.txt" --resume
fi
[ -f "$OUT/summary.txt" ] && cat "$OUT/summary.txt"
echo "session $SID" > "$OUT/session.txt"
echo "done. score: python llm/score.py $PROTOCOL $OUT/reply.txt"
