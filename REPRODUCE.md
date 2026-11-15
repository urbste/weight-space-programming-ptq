# Reproducing the paper

There are three levels of reproduction. The first two are deterministic and cost nothing; the third calls Claude
models and gives a new sample, not the same numbers.

| Level | What it checks | Needs | Time / cost |
|---|---|---|---|
| 1. Score the saved replies | every LLM number in the paper | CIFAR-10, numpy | seconds, free |
| 2. Rebuild network, prompts, baselines and analysis | RTN, kernel balancing, SQuant, AdaRound/GPTQ, error model, figures | + torch (GPU for training and AdaRound) | ~15 min, free |
| 3. Prompt your own Claude models with P1–P4 | the LLM behaviour itself | Claude Code CLI or API access | hours, about $1–25 per run |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .              # numpy, scipy, matplotlib: enough for levels 1 and 3
pip install -e ".[torch]"     # adds torch for level 2 (training, AdaRound/GPTQ)
mkdir -p data && wget -qO- https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | tar xz -C data
```

All scripts run from the repository root. The trained network (`checkpoints/small_e500.*`), the BN-folded float
weights, the RTN int4 network, the validation split and the float test predictions (`exports/`) are in the
repository, so levels 1 and 3 need no training.

## Level 1: score the saved replies

The verbatim LLM answers are in the repository. `llm/score.py` applies them with the paper's rules (invalid lines are
skipped and counted, never repaired; invalid rows fall back to RTN):

```bash
python llm/score.py p1 llm/replies/opus_int4.txt              # 64.07
python llm/score.py p2 llm/layerwise/replies/opus_int4.txt    # 77.95
python llm/score.py p3 llm/fp32_direct/reply_opus.txt         # 60.58
python llm/score.py p4 llm/fp32_bn/reply_fable.txt            # 69.97
```

| Protocol | Saved replies |
|---|---|
| P1 whole network | `llm/replies/<model>_int4.txt` |
| P2 layer by layer | `llm/layerwise/replies/<model>_int4.txt` |
| P3 float only | `llm/fp32_direct/reply_opus*.txt` |
| P4 float + BN | `llm/fp32_bn/reply_<model>.txt`, with every turn and the final summary in `llm/fp32_bn/cli_log_<model>/` |

The per-protocol scripts `llm/eval_llm.py`, `llm/eval_layerwise.py`, `llm/eval_fp32_direct.py` and
`llm/eval_fp32_bn.py` write the full result files in `results/`, including per-layer statistics.

## Level 2: rebuild everything else

```bash
python train.py                     # optional: retrains SmallCNN (500 epochs, GPU); the checkpoint is included
python ptq_eval.py                  # float + RTN -> exports/, results/ptq.json
python llm/make_prompts.py          # P1 task file  -> llm/prompts/int4.txt
python llm/make_layerwise.py 4      # P2 task + 11 chunks -> llm/layerwise/int4/
python llm/make_fp32_direct.py      # P3 task + 19 chunks -> llm/fp32_direct/
python llm/make_fp32_bn.py          # P4 task + 19 chunks -> llm/fp32_bn/
python llm/reference_methods.py     # exact kernel balancing and bias correction -> results/reference.json
python llm/squant_ref.py            # SQuant-K / -C / -K+C -> results/squant.json
python analysis/ptq_calibrated.py   # AdaRound and GPTQ, 1,024 calibration images (GPU) -> results/ptq_calibrated.json
python analysis/analyze.py          # bit sweep, SNR, error decomposition, figures -> results/analysis.json, paper/figs/
python analysis/edge_split.py       # patch-mean vs. detail error term -> results/edge_split.json, paper/figs/edge_split.svg
typst compile paper/main.typ        # the paper
```

Retraining changes the weights slightly (GPU non-determinism), so every downstream number moves a little. Keep the
included checkpoint to reproduce the paper exactly.

## Level 3: prompt your own Claude models

`llm/run_cli.sh` sends a protocol to a model through the Claude Code CLI with **all tools disabled**
(`claude -p --tools ""`). It sends the task as the first message, then one chunk per message in the same session,
and saves every answer.

```bash
PROTOCOL=p1 MODEL=claude-opus-5-5   ./llm/run_cli.sh     # one message, one answer
PROTOCOL=p2 MODEL=claude-opus-5-5   ./llm/run_cli.sh     # task + 11 chunks
PROTOCOL=p3 MODEL=claude-sonnet-5-5 ./llm/run_cli.sh     # task + 19 chunks
PROTOCOL=p4 MODEL=claude-fable-5-2  ./llm/run_cli.sh     # task + 19 chunks with BN statistics

python llm/score.py p2 runs/p2_opus-5-5_high/reply.txt
```

Options (environment variables): `EFFORT` (`low`…`max`, default `high`; the paper used `high`, except `medium` for
the best Sonnet 5.5 P4 run), `LIMIT` (seconds per turn, default 3600), `TAG` (output folder name). If a run is
interrupted, continue it with `SID=<session id> START=<next chunk>`; the session id is printed at the start.

Output in `runs/<protocol>_<tag>/`: `task.txt` (the exact first message), `turnNN.txt` (one answer per turn),
`reply.txt` (all chunk answers in order, the input to `score.py`) and `summary.txt`.

The exact task texts are `llm/prompts/int4.txt` (P1), `llm/layerwise/int4/task.txt` (P2),
`llm/fp32_direct/task.txt` (P3) and `llm/fp32_bn/task.txt` (P4); the runner only replaces their "No tools" rule
and their file-reading delivery instructions with message delivery. To use another client (the web app, the API),
send `task.txt` and then each `chunkNN.txt` of the protocol as separate messages in one conversation, paste the
answers into one text file, and score it with `llm/score.py`.

**Things to know before you spend money.**

- **Cost.** At list prices a P1 run costs about $1–3, P2 about $5–15 (the context grows to ~700k tokens), P3 about
  $2–7, and P4 up to $25 for Fable 5.2, because the LLM writes all 20k integer weights. See the Inference Cost section of the paper.
- **Harness.** In the paper, P1–P3 ran as Claude Code subagents whose only tool was reading the task file, and P4 ran
  with this script. The runner delivers the same text as messages; this should not matter, but it is not the
  identical setup.
- **Variance.** Every run is a new sample. Single flipped integer weights in conv1 or conv2 move accuracy by about
  1 pp, and some LLMs return rows with the wrong number of integer weights, which fall back to RTN.
- **Refusals.** During our experiments some requests were stopped by API safeguards at the first message (paper,
  Limitations). If that happens, the script stops with the error text; rerunning or changing `EFFORT` helped us.
- **Hidden reasoning.** If an answer contains only reasoning and no `BEGIN CHUNK` block, ask the model in the same
  session (`claude -p --resume <session id>`) to re-emit the block and append it to `reply.txt`.
