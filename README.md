# Tool-free weight-space programming for PTQ

Code, prompts and verbatim LLM replies for the paper
**"Can a Frontier LLM Repair a Quantized Network by Tool-Free Weight-Space Programming?"** ([paper/main.pdf](paper/main.pdf)).

We give Claude models the complete weights of a 20k-weight CIFAR-10 CNN as text and ask them to produce a better
4-bit network, without tools, data or the ability to run anything. All arithmetic happens in the model's own
reasoning. Per-channel round-to-nearest (RTN) at 4 bit lowers test accuracy from 81.42% to 60.39%.

## Protocols

| | The LLM receives | The LLM writes |
|---|---|---|
| P1 | float weights (in units of the step), RTN int4 network, accuracy drop | sparse edits, one answer |
| P2 | as P1, in 11 chunks | sparse edits, one answer per chunk |
| P3 | float weights only, in 19 chunks | the full int4 network |
| P4 | as P3, plus BatchNorm statistics | the full int4 network |

## Main results (test accuracy, %)

| | |
|---|---|
| float | 81.42 |
| RTN int4 | 60.39 |
| P1, Opus 5.5 / Sonnet 5.5 (kernel balancing on conv1 only) | 64.07 / 64.37 |
| **P2, Opus 5.5 (kernel balancing on all 2,200 kernels, by hand)** | **77.95** |
| P3, Opus 5.5 | 60.58 |
| P4, Fable 5.2 (row-wise outlier clipping) / Opus 5.5 | 69.97 / 63.26 |
| exact kernel balancing in code (= SQuant kernel step), no data | 77.80 |
| AdaRound, 1,024 calibration images | 80.71 |

When the LLM sees the RTN network and is told that it fails, Opus 5.5 and Sonnet 5.5 balance the rounding errors
within each 3×3 kernel, the kernel step of the data-free method SQuant. Given only the float weights, the LLMs write
plain RTN, at most with mildly clipped scales.

## Installation

The scripts run from the repository root; `pyproject.toml` only installs their dependencies.

```bash
git clone https://github.com/urbste/weight-space-programming-ptq
cd weight-space-programming-ptq
python -m venv .venv && source .venv/bin/activate      # or: conda create -n wsp python=3.11 && conda activate wsp
pip install -e .                                       # numpy, scipy, matplotlib
pip install -e ".[torch]"                              # adds torch, only for training and AdaRound/GPTQ
mkdir -p data && wget -qO- https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | tar xz -C data
```

With [uv](https://docs.astral.sh/uv/): `uv sync` (or `uv sync --extra torch`), then prefix commands with `uv run`.

## Quick start

```bash
python llm/score.py p2 llm/layerwise/replies/opus_int4.txt    # scores the saved Opus 5.5 P2 reply: 77.95
PROTOCOL=p2 MODEL=claude-opus-5-5 ./llm/run_cli.sh             # a new run with the Claude Code CLI (costs API usage)
```

[REPRODUCE.md](REPRODUCE.md) describes all steps: scoring the saved replies, rebuilding baselines and analysis, and
prompting your own Claude models with P1–P4.

## Repository

| Path | Content |
|---|---|
| `model.py`, `data.py`, `train.py` | `SmallCNN`, CIFAR-10 loading (fixed 45k/5k split), training |
| `quant.py`, `ptq_eval.py`, `evaluate.py` | BN folding, per-channel RTN, numpy evaluation |
| `checkpoints/`, `exports/` | trained network, folded float and RTN weights, validation split, float predictions |
| `llm/make_*.py` | build the task and chunk files of P1–P4 |
| `llm/prompts/`, `llm/layerwise/int4/`, `llm/fp32_direct/`, `llm/fp32_bn/` | exact inputs the LLMs saw (P3 and P4 folders also hold the replies) |
| `llm/replies/`, `llm/layerwise/replies/` | verbatim replies for P1 and P2 |
| `llm/run_cli.sh`, `llm/score.py` | run any protocol on a Claude model with tools disabled; score a reply |
| `llm/eval_*.py`, `llm/reference_methods.py`, `llm/squant_ref.py` | evaluation, exact kernel balancing, bias correction, SQuant |
| `analysis/` | AdaRound/GPTQ, error model, error decomposition, figures |
| `results/` | all result files used in the paper |
| `paper/` | Typst source, figures and PDF |

Some folders also contain replies from runs that are not reported in the paper (repeated runs, other effort
settings, int8/int2 runs of P1, a run blocked by API safeguards).

## License

MIT, see [LICENSE](LICENSE).
