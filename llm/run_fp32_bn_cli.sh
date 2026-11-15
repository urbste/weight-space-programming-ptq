#!/usr/bin/env bash
# Experiment 6 driver: float weights + BatchNorm stats, chunked, all tools disabled from the command line.
# The model gets task.txt (without its "No tools" rule) and then one chunk per message in ONE resumed session.
#
#   EFFORT=high MODEL=claude-opus-5-5 ./llm/run_fp32_bn_cli.sh  (EFFORT default high) 
#   continue after an interruption:  SID=<session id> START=12 EFFORT=medium ./llm/run_fp32_bn_cli.sh
#   MODEL=claude-haiku-5-5 ./llm/run_fp32_bn_cli.sh      (default MODEL=claude-opus-5-5; TAG is derived: opus/sonnet/haiku)
#
# Output per model:  llm/fp32_bn/reply_<tag>.txt   (the chunk replies, printed to the terminal as well)
#                    llm/fp32_bn/cli_log_<tag>/    (task text, one file per turn, summary)
# Score with:        python llm/eval_fp32_bn.py llm/fp32_bn/reply_<tag>.txt   -> results/fp32_bn_<tag>.json
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-claude-opus-5-5}"
TAG="${TAG:-$(echo "$MODEL" | sed -E 's/^claude-([a-z]+)-.*/\1/')}"
EFFORT="${EFFORT:-high}"   # claude --effort: low|medium|high|xhigh|max
LIMIT="${LIMIT:-3600}"   # seconds per turn; a stalled request stops the run instead of hanging
D=llm/fp32_bn
LOG=$D/cli_log_$TAG
REPLY=$D/reply_$TAG.txt
mkdir -p "$LOG"
START="${START:-1}"     # first chunk to send; START>1 continues an earlier run (needs SID=<session id>)
SID="${SID:-$(python -c 'import uuid;print(uuid.uuid4())')}"
CL=(timeout "$LIMIT" claude --model "$MODEL" --effort "$EFFORT" --tools "" --strict-mcp-config -p)
echo "model $MODEL  effort $EFFORT  tag $TAG  session $SID"

# task text without the "No tools" bullet and without the "read a chunk file" delivery wording
python - <<'PY' > "$LOG/task_cli.txt"
import re
t = open("llm/fp32_bn/task.txt").read()
t = re.sub(r"- No tools:.*?\n(?:  .*\n)*", "", t, count=1)
t = t.split("DELIVERY IN CHUNKS")[0]
t += """DELIVERY IN CHUNKS
The weight dump is delivered in 19 messages, one chunk per message, after this one. For EACH chunk message, write the
ROW lines for ALL rows of that chunk as plain text between a line "BEGIN CHUNK <k>" and a line "END CHUNK <k>", and
nothing else. You may use everything you have seen so far. After the last chunk I will ask for a short summary of how
you chose the scales and codes. Now reply only with the word READY.
"""
print(t)
PY

turn() {  # turn <label> <logfile> <message> [--session-id|--resume]
  local label="$1" file="$2" msg="$3" mode="$4"
  echo "== $label"
  if ! "${CL[@]}" "$mode" "$SID" "$msg" > "$file"; then
    echo "!! $label failed or timed out (limit ${LIMIT}s). Output of the call:"; head -c 1500 "$file"; echo; exit 1
  fi
  [ -s "$file" ] || { echo "!! $label returned nothing"; exit 1; }
  if head -c 300 "$file" | grep -q '^API Error'; then echo "!! $label: API error, not an answer:"; head -c 1500 "$file"; echo; exit 1; fi
}

if [ "$START" -le 1 ]; then
  turn "task" "$LOG/turn00.txt" "$(cat "$LOG/task_cli.txt")" --session-id
  cat "$LOG/turn00.txt"
  : > "$REPLY"
else
  echo "continuing session $SID at chunk $START (keeping $REPLY)"
fi
for k in $(seq -w "$START" 19); do
  turn "chunk $k" "$LOG/turn$k.txt" "$(cat "$D/chunk$k.txt")" --resume
  tee -a "$REPLY" < "$LOG/turn$k.txt"
  echo "   rows in reply: $(grep -c '^ROW' "$LOG/turn$k.txt" || true)"
  echo >> "$REPLY"
done
turn "summary" "$LOG/summary.txt" "All chunks are done. Give a short summary of how you chose the scales and codes, and why. Do not repeat the ROW lines." --resume
cat "$LOG/summary.txt"
echo "done. rows: $(grep -c '^ROW' "$REPLY") / 114   score: python llm/eval_fp32_bn.py $REPLY"
