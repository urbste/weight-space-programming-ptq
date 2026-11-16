#set document(title: "Can a Frontier LLM Repair a Quantized Network by Tool-Free Weight-Space Programming?", author: "Steffen Urban")
#set page(paper: "a4", margin: (x: 1.75cm, top: 2.0cm, bottom: 2.2cm), columns: 2, numbering: "1")
#set columns(gutter: 0.65cm)
#set text(font: "New Computer Modern", size: 9.5pt, lang: "en")
#set par(justify: true, leading: 0.55em, spacing: 0.75em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#show heading.where(level: 1): set text(size: 11pt)
#show heading.where(level: 1): set block(above: 1.2em, below: 0.65em)
#show heading.where(level: 2): set text(size: 10pt)
#show heading.where(level: 2): set block(above: 1.0em, below: 0.55em)
#show heading.where(level: 3): set text(size: 9.5pt, style: "italic", weight: "regular")
#show heading.where(level: 3): set block(above: 0.8em, below: 0.45em)
#show figure.caption: set text(size: 8.5pt)
#show figure.where(kind: table): set figure.caption(position: top)
#set figure(gap: 0.5em)
#show raw: set text(size: 7.5pt)
#show raw.where(block: true): block.with(fill: luma(246), inset: 5pt, radius: 2pt, width: 100%)
#set table(stroke: none, inset: (x: 3.5pt, y: 2.6pt))
#show table: set text(size: 8pt)
#set list(indent: 0.4em, spacing: 0.6em)
#set enum(indent: 0.4em, spacing: 0.6em)

#let hl = table.hline(stroke: 0.6pt)
#let hl2 = table.hline(stroke: 0.3pt)
#let best(x) = underline(x)
#let up(x) = text(fill: rgb("#1a7f37"), x)
#let down(x) = text(fill: rgb("#c0392b"), x)

#place(top + center, float: true, scope: "parent", clearance: 1.6em)[
  #text(size: 15pt, weight: "bold")[Can a Frontier LLM Repair a Quantized Network \ by Tool-Free Weight-Space Programming?]
  #v(0.4em)
  Steffen Urban \
  #text(size: 9pt)[Independent Researcher]
  #v(0.9em)
  #block(width: 88%)[
    #set par(justify: true)
    #set text(size: 9pt)
    We give four large language models (LLMs) of the Claude family the complete weights of a 20k-parameter
    CIFAR-10 CNN as text and ask them to produce a 4-bit network without tools, data or the ability to run
    inference. Standard round-to-nearest (RTN) quantization to 4 bit lowers the test accuracy from 81.4% to 60.4%.
    We design four evaluation protocols that differ in the input and task description given to the LLMs and analyze
    their responses. In the first two protocols the LLMs are given the float and the 4-bit RTN network and are told
    that the accuracy has dropped significantly. In the last two they receive only the float network and the task
    to quantize it, which tests whether they apply an optimization on their own or fall back to plain RTN. Our
    analysis shows that a frontier LLM can recover most of the lost accuracy. Given one answer per layer, Opus 5.5
    raises the accuracy of the 4-bit network to 77.95% by balancing the rounding errors within each 3×3 kernel,
    reasoning over the integer weights without tools. This is 3.5 pp below the float network and 2.8 pp below
    AdaRound, which learns the rounding from calibration data. Given only the float network, the LLMs name
    error-cancelling corrections but do not apply them and fall back to plain RTN, at most with mildly clipped
    scales, which still lifts Fable 5.2's quantized network to 70.0%. We analyze in detail what the LLMs did and
    why it works.
  ]
]

= Introduction <sec-intro>

Post-training quantization (PTQ) converts a trained float network to low-bit integers without retraining
@jacob2018 @krishnamoorthi2018 @nagel2021whitepaper. Its simplest form, per-channel symmetric round-to-nearest
(RTN), is close to lossless at 8 bit for most CNNs, while at 4 bit its accuracy loss varies widely between
networks and can be large @krishnamoorthi2018 @nagel2021whitepaper. Better PTQ methods fix this with calibration data and
layer-wise reconstruction @adaround @li2021brecq @gptq. Data-free methods @nagel2019 @cai2020zeroq @guo2022squant work from the
checkpoint alone, but they too are fixed numerical procedures.

We ask whether a large language model (LLM) can find and execute such a repair by _reading_ the
weights. The LLM gets the weights of a small CNN as text, either the float weights alone or together with the RTN integer
weights, and writes integer weights and scales. It has no code interpreter, no calculator, no data and cannot evaluate the network. All arithmetic happens
in its own reasoning. The setting is deliberately extreme. Of course, a short Python script quantizes better
and faster. The question is what the LLM notices in the numbers, how accurately it executes what it decides, and what limits it.

To this end, we train a simple 20k-weight CNN on CIFAR-10 @cifar that will fit into the context of current LLMs, which makes the question testable. We evaluate *four protocols* (@sec-protocols) that differ in whether the LLM sees the RTN network and is told about the accuracy drop, whether it gets BatchNorm statistics, and whether it answers in one response or in one response per chunk.

We find that frontier LLMs without tools arrive at a published data-free PTQ correction by reading raw weights.
Opus 5.5 and Sonnet 5.5 independently balance the per-kernel rounding error sums (kernel balancing, @sec-whole), the
kernel-wise step of SQuant @guo2022squant. With one answer per chunk, Opus 5.5 applies it to all 2,200 kernels by
hand (96% of edits identical to exact kernel balancing) and reaches 77.95% (@sec-layerwise), so the one-answer result is limited by
the amount of arithmetic per answer. A first-order error model shows that the per-kernel error
sum sets the mean shift of each output, which explains why the rule works (@sec-model). Given only the float weights, the LLMs write RTN, at most with clipped
outlier rows, and none uses the BatchNorm statistics to cancel the mean shift (@sec-direct, @sec-clipping).

= Related work <sec-related>

_Post-training quantization._ Per-channel RTN with BatchNorm folding is the standard baseline
@jacob2018 @krishnamoorthi2018. Clipping the range trades clipping error for resolution, analytically in ACIQ
@banner2019aciq and by MSE search in most toolchains @nagel2021whitepaper. Outlier channel splitting
@zhao2019ocs avoids clipping by duplicating channels. Rounding to nearest is not optimal for the layer output.
AdaRound @adaround learns the rounding direction per weight from calibration data, BRECQ @li2021brecq
reconstructs blocks, and GPTQ @gptq uses second-order information to compensate errors column by column. Without
data, DFQ @nagel2019 equalizes ranges across layers and corrects the bias shift caused by rounding, estimating
input statistics from BatchNorm parameters. ZeroQ @cai2020zeroq synthesizes calibration data from BatchNorm
statistics. SQuant @guo2022squant approximates the Hessian by element-, kernel- and channel-wise terms and flips
rounding decisions until the absolute rounding-error sum of every kernel is at most half a step, then does the
same per output channel while allowing one step per kernel. Vendor toolchains for edge accelerators differ in
scaling and clipping, so the same checkpoint can reach different accuracy on different backends
@dhahri2025quanttrim. The correction our LLMs find, constraining the sum of rounding errors per kernel, is
the kernel step of SQuant. AdaRound learns rounding decisions of this kind from data.

_LLMs over numbers and weights._ LLMs can act as optimizers that propose solutions from a history of scored
attempts @yang2024opro, or drive program search with an evaluator in the loop @romera2024funsearch. Both rely on
feedback, which we remove. Weight-space learning trains networks that take other networks' weights as input
@schurholt2021 @navon2023. We instead ask a general LLM to reason over weights in context. Bills et al.
@bills2023 use LLMs to explain individual neurons from activations, while here the LLM sees only parameters.
Weight-space programming @takehi lets LLM agents edit network weights with tools and evaluation. Our setting is
its tool-free limit, where every operation must be carried out in chain-of-thought @wei2022cot.

= Method <sec-method>

== Network, baseline and notation <sec-baseline>

`SmallCNN` has four conv(3×3)–BN–ReLU blocks with 8, 16, 32 and 48 channels, 2×2 max-pooling after blocks 2 and
3, global average pooling and a 48→10 linear layer (20,280 weights). It is trained on CIFAR-10 for 500 epochs
(SGD, Nesterov, one-cycle, weight decay $5 dot 10^(-4)$, crop and flip) on a fixed 45k/5k train/val split and
reaches 82.14% val and 81.42% test accuracy (@fig-arch).

#figure(
  scope: "parent", placement: top,
  image("figs/architecture.svg", width: 92%),
  caption: [`SmallCNN`. One CIFAR-10 test image per class. The highlighted one is passed through the float
  network. Each stack shows the first five channels after ReLU (and pooling), the bar the 48 pooled features,
  and the bars at the right the ten logits.],
) <fig-arch>

BatchNorm is folded into the convolutions. We call the weights that compute one output channel a _row_, which is one conv filter
($C_"in" times 3 times 3$ weights) or one row of the fc matrix (48 weights). Hence the network has 114 rows (8+16+32+48+10).
The row is also the unit in which the LLMs read and write the network. The input lists each row with its scale and
bias (@app-protocol), edits address weights by layer and row, and in P3/P4 the LLM writes one line per row.
Per-channel RTN gives each row $r$ with weights $w_(r j)$ its own scale:

$ s_r = max_j |w_(r j)| \/ q_max, quad q_(r j) = "round"(w_(r j) \/ s_r), $ <eq-rtn>

where $q_(r j)$ is the quantized integer weight for $w_(r j)$, $s_r$ the per-row scale, and $s_r q_(r j)$ the
dequantized weight used at inference. With $b$ bits, $q_max = 2^(b-1) - 1$ ($q in {-7, dots, 7}$ at int4). Biases
and activations stay float (BN folding does not change per-channel RTN).

For a row with scale $s$ we write $u = w\/s$ for a float weight in units of the quantization step, $q =
"round"(u)$ for its integer weight and $e = q - u in [-1\/2, 1\/2]$ for its rounding error. The _kernel sum_ $E = sum_k e_k$ adds the rounding errors over the nine positions $k$ of
one 3×3 kernel (for fc layers, over a whole row).

== Protocols <sec-protocols>

In every experiment, a frontier LLM (@sec-setup lists them) receives a task
description and `SmallCNN` as text, and returns a quantized version of `SmallCNN`. Appendices @app-prompt[], @app-prompt-p2[], @app-prompt-p3[] and
@app-prompt-p4[] give the task descriptions of P1–P4. The four protocols differ in what the LLM sees and what it writes.

*P1: whole network, one answer.* The LLM sees the float weights in units of the step, $u = w\/s$ (two
decimals), and the RTN integer weights $q$ for every row, plus RTN scales and biases (format in @app-protocol). It is told
the architecture, input normalization, and that the float network scores 82.14% on a held-out split while the
quantized one scores 60.38% (int4). It answers with sparse edits that set one integer weight, replace a 3×3 kernel or an fc row,
or set a scale or a bias. Unmentioned values keep their RTN value. Invalid lines are skipped, never repaired. Writing $u$ instead of $w$ saves one division per weight and exposes the rounding residual $u - q$
directly, which is a hint.

*P2: layer by layer.* Same input, but delivered in eleven chunks (conv1, conv2, conv3
in two, conv4 in six and fc, with at most 256 kernels each). The LLM answers each chunk in its own response, inside one
conversation, so every chunk gets a fresh output budget while the context keeps everything so far.

*P3: from float weights.* The LLM gets only the float weights (three decimals), in 19 chunks of at most 1,152
weights, and writes the 4-bit network itself as one line `ROW <layer> <r> SCALE <s> Q <codes>` per output row, with
free scale and integer weights in $[-7, 7]$. The prompt does not mention RTN or the accuracy gap. A missing row, a wrong
number of integer weights or an out-of-range integer weight falls back to RTN for that row.

*P4: from float weights with BN statistics.* As P3, plus $gamma$, $beta$, running mean and variance for every
conv channel. Published data-free methods use these statistics in place of data. DFQ @nagel2019 estimates the
input means from them for bias correction, and ZeroQ @cai2020zeroq synthesizes calibration data from them. P4
tests whether the LLM uses them in a similar way. Biases stay fixed, so the statistics can only act through
integer weights and scales.

== LLMs and evaluation <sec-setup>

_LLMs and tool restriction._ We use Claude Opus 5.5, Sonnet 5.5 and Haiku 5.5 in P1, P2 and P4, Opus 5.5 alone in
P3, and Fable 5.2 only in P4 (API model IDs `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`,
`claude-fable-5-2`). P1–P3
ran as subagents that could call only the file-reading tool on their task files, which only delivers the input. We audited every transcript and
found no other calls. P4 ran from the command line with all tools disabled (`claude -p --tools ""`), one chunk per
message in one session, because API safeguards blocked the subagent route for Opus 5.5 and Sonnet 5.5.

_Evaluation._ We report test accuracy unless stated otherwise. For analysis only, we
compute on the 5k validation images per-layer output SNR $10 log_10 (sum y_"fp"^2 \/ sum (y_q - y_"fp")^2)$,
_isolated_ (float input to each layer) and _accumulated_ (quantized network end to end), input means and patch
covariances. The LLMs never see these.

_Reference corrections._ To judge the LLM edits, we also run two data-free corrections in Python. _Kernel
balancing_ re-rounds integer weights, cheapest first, until every kernel sum satisfies $|E| <= 1\/2$. This is the rule Opus 5.5 and
Sonnet 5.5 describe in P1 (@sec-whole). We call this
implementation, run on every kernel, _exact kernel balancing_. _Bias correction_ @nagel2019 keeps the integer
weights and cancels the mean output shift through the bias, using input means measured on validation images.
In section @sec-two-corrections we provide more theoretical details about those two methods.

= Results <sec-exp>

== What RTN does to the network <sec-damage>

For our simple network, RTN is basically lossless at 8 bit (81.45%, 98.6% agreement with float) and drops by 21 percentage points (pp) to 60.39% at 4 bit (@tab-bits). Below
4 bit RTN gives 28.7% at 3 bit and chance at 2 bit. Below 4 bit no data-free correction comes near float accuracy, hence in the remainder of the paper *we study the 4-bit case, where corrections can recover most of the loss*.

#figure(
  placement: top,
  table(columns: 7, align: (left, ..range(6).map(_ => right)),
    fill: (x, _) => if x == 4 { luma(225) },
    hl, [Test %], [8 bit], [6 bit], [5 bit], [4 bit], [3 bit], [2 bit], hl,
    [RTN],                [#best[81.45]], [80.86], [77.91], [60.39], [28.70], [8.60],
    [\+ bias corr.#super[†]], [81.38], [81.33], [#best[80.80]], [75.29], [50.75], [12.03],
    [kernel bal.],        [81.41], [#best[81.55]], [80.10], [77.80], [62.47], [#best[13.26]],
    [both#super[†]],      [81.40], [81.43], [80.54], [#best[79.68]], [#best[64.23]], [12.64], hl,
  ),
  caption: [Test accuracy (%) of RTN and of the two corrections defined in @sec-setup, from 8 to 2 bit calculated using a Python script. The float
  network reaches 81.42%. Kernel balancing is applied to every conv kernel and fc row. Underlined: best value per
  column. Shaded: 4 bit, the setting we study in the rest of the paper. #super[†]Uses input means measured on
  validation images, so it is not data-free.],
) <tab-bits>

@tab-ptq compares kernel balancing (what the frontier LLMs propose in P1/P2) with published PTQ methods. Kernel balancing on the conv layers is basically SQuant's @guo2022squant
kernel step and leads to 77.63%, and adding SQuant's channel step gives 79.06%. With 1,024 calibration images
from the training split, AdaRound @adaround and GPTQ @gptq reach 80.7%, 0.7 pp below float. All methods keep the RTN scales, and all except bias correction change only the rounding.

#figure(
  table(columns: 3, align: (left, left, right),
    hl, [Method], [Data], [int4], hl,
    [RTN],                                [none],        [60.39],
    [kernel bal. = SQuant-K (conv)],      [none],        [77.63],
    [kernel bal. (conv + fc rows)],       [none],        [77.80],
    [SQuant-K+C @guo2022squant],          [none],        [79.06],
    [bias corr.],                         [val means],   [75.29],
    [kernel bal. + bias corr.],           [val means],   [79.68],
    [GPTQ @gptq],                         [1k images],   [80.66],
    [AdaRound @adaround],                 [1k images],   [#best[80.71]], hl,
  ),
  caption: [Test accuracy (%) at int4 of PTQ methods on our network (FP32: 81.42), our implementations, RTN
  per-row scales throughout. Calibration images come from the training split.],
) <tab-ptq>

The total accuracy loss is not caused by a single layer. The isolated SNR of all five int4 layers lies between 14 and 19 dB, while the
accumulated SNR falls from 18.6 dB after conv1 to 3.5 dB at the logits (@fig-snr). Every layer contributes a
similar error, and the errors compound.

#figure(
  table(columns: 7, align: (left, ..range(6).map(_ => right)),
    hl, table.header([layer], [weights], [peak/RMS], [zeros], [SNR acc.], [SNR iso.], [mean $|E|$]), hl,
    [conv1], [216],    [2.28], [.18], [18.6], [18.6], [0.59],
    [conv2], [1,152],  [3.03], [.22], [12.8], [17.4], [0.71],
    [conv3], [4,608],  [3.32], [.21], [8.0],  [13.7], [0.68],
    [conv4], [13,824], [3.23], [.19], [6.8],  [16.8], [0.73],
    [fc],    [480],    [3.03], [.18], [3.5],  [17.7], [1.29], hl,
  ),
  caption: [RTN int4 layer statistics. Peak/RMS: ratio of largest to RMS float weight. Zeros: fraction of zero integer weights.
  SNR (dB): accumulated and isolated. Mean $|E|$: mean absolute rounding-error sum per 3×3 kernel (fc: per
  row), in steps.],
) <tab-layers>

#figure(
  scope: "parent", placement: top,
  image("figs/snr.svg", width: 80%),
  caption: [Output SNR per layer at int4. Left: accumulated. Right: isolated (float input to each layer).
  RTN, exact kernel balancing, Opus 5.5 P1 (conv1 only), Opus 5.5 P2.],
) <fig-snr>

== LLM results <sec-overview>

@tab-all lists the int4 LLM results with the mechanism behind each result. We discuss them by protocol below.

#figure(
  scope: "parent", placement: top,
  { set par(justify: false); table(columns: (3.2cm, auto, auto, auto, auto, 1fr), align: (left, left, right, right, right, left),
    hl, [Protocol], [LLM], [Changed / valid], [Test %], [Δ RTN (pp)], [What the LLM did], hl,
    [*Reference*\ #text(size: 7pt)[computed in code]], [RTN], [–], [60.39], [–], [],
    [], [exact kernel bal.], [1,495 weights], [77.80], up[+17.41], [code, all layers, no data (reference)],
    [], [AdaRound], [2,718 weights], [80.71], up[+20.32], [code, 1,024 calibration images (reference)], hl2,
    [*P1 whole network*\ #text(size: 7pt)[in: float + RTN int4,\ held-out 82.1% → 60.4%\ out: edits, one answer]], [Opus 5.5], [11 weights], [64.07], up[+3.68], [kernel balancing, conv1 only],
    [], [Sonnet 5.5], [13 weights], [64.37], up[+3.98], [kernel balancing, conv1 only],
    [], [Haiku 5.5], [fc rows], [62.18], up[+1.79], [balances fc row sums], hl2,
    [*P2 layer by layer*\ #text(size: 7pt)[in: as P1, in 11 chunks\ out: edits, one answer per chunk]], [Opus 5.5], [1,364 weights], [#best[77.95]], up[+17.56], [kernel balancing through all 2,200 kernels, 96% identical to exact kernel bal.],
    [], [Sonnet 5.5], [0], [60.39], [±0.00], [declines with "no basis for changing any code"],
    [], [Haiku 5.5], [74 weights, 11 scales], [59.90], down[−0.49], [scales 11 rows by 0.9 on an untested weight-MSE argument], hl2,
    [*P3 float only*\ #text(size: 7pt)[in: float weights only\ out: full int4 network]], [Opus 5.5], [111 / 114 rows], [60.58], up[+0.19], [writes RTN, clips conv4 outliers (+1.5 pp), loses it on conv1 ties], hl2,
    [*P4 float + BN*\ #text(size: 7pt)[in: float weights + BN stats\ out: full int4 network]], [Fable 5.2], [114 / 114 rows], [#best[69.97]], up[+9.58], [row-wise clipping of isolated outliers in all layers, BN unused],
    [], [Opus 5.5], [108 / 114 rows], [63.26], up[+2.87], [clips conv4 and fc outliers, BN unused],
    [], [Sonnet 5.5], [39 / 114], [59.23], down[−1.16], [RTN with wrong number of weights and shifted integer weights in valid rows],
    [], [Haiku 5.5], [19 / 114], [59.96], down[−0.43], [wrong number of weights, otherwise copies RTN], hl,
  ) },
  caption: [Int4 LLM results, grouped by protocol with what the LLM receives (in) and returns (out). In P1 and P2 the LLM is told the float and RTN accuracies and asked to improve the RTN network. In P3 and P4 it is asked to quantize the float network and is told neither. Δ RTN: change in test accuracy relative to RTN, green if better, red if worse. Invalid rows (P3, P4) fall back to RTN.],
) <tab-all>

=== P1: Editing the whole network <sec-whole>

In P1 (@tab-all), Opus 5.5 and Sonnet 5.5 arrive at the same idea.
Both argue that natural images are smooth, so a 3×3 kernel responds mainly to the sum of its weights. They compare float and integer
sums per conv1 kernel and re-round the weight closest to a boundary where they differ by more than about half a
step. This is kernel balancing and we analyze it in @sec-model. They change 11 and 13 integer weights (0.05% of the weights, 8 in common) and
gain 3.7–4.0 pp. Haiku 5.5 balances the fc row sums instead. None of the LLMs touches
conv2–conv4. All state that the work does not fit into one answer and stop. The limit is the length of one
answer, which must hold the reasoning and every edit. Their hand execution is accurate. Sonnet 5.5's 13 edits match exact kernel balancing in 11 positions, and the edited conv1
layers reach the same mean $|E|$ as exact kernel balancing (0.27 vs. 0.59 for RTN). Applied to all conv layers, the same rule
gives 77.63% (@tab-ptq). P2 lifts the output limit for the LLM itself (@sec-layerwise).

=== P2: One answer per layer <sec-layerwise>

With one response per chunk, Opus 5.5 works through every kernel of every conv layer (13, 77, 300 and 974 edits) and
reaches 77.95% test (78.62% val), which closes 17.6 of the 21 pp gap to float and matches exact kernel balancing within noise (@fig-recovery). Opus 5.5 did not edit fc, and exact kernel balancing on the conv layers alone gives 77.63%. The run was interrupted and its conv1 and conv2 edits were re-emitted after resumption (@sec-limits). 1,314 of its
1,364 edits coincide with exact kernel balancing in position and value, no integer weight moves by more than one step, and 118 of
2,200 kernels keep $|E| > 1\/2$. Isolated SNRs match exact kernel balancing within 0.4 dB in every layer, and the accumulated SNR
at the logits rises from 3.5 to 12.8 dB. Ahead of exact
kernel balancing by 2.3 pp after conv2 and a few tenths at the end, Opus 5.5's curve mainly shows that the choice among
nearly equivalent flips moves accuracy by about 1 pp.

#figure(
  scope: "parent", placement: top,
  image("figs/weights.svg", width: 78%),
  caption: [Top: float vs. dequantized int4 RTN weights (log count). Bottom: distribution of the per-kernel
  rounding-error sum $E$ (fc: per row) for RTN, exact kernel balancing and Opus 5.5 P2.],
) <fig-weights>

Opus 5.5's kernel sums have the same distribution as those of exact kernel balancing (@fig-weights).
Sonnet 5.5 reads all chunks and edits nothing, citing missing activation statistics, although it found
kernel balancing in P1. Haiku 5.5 shrinks eleven scales by 0.9 and loses 0.5 pp.

#figure(
  image("figs/recovery.svg", width: 82%),
  caption: [P2: cumulative test accuracy after editing up to a layer, Opus 5.5 vs. exact kernel balancing.],
) <fig-recovery>

=== P3/P4: Quantizing from float weights alone <sec-direct>

_Without the RTN result (P3)_, Opus 5.5 states the textbook recipe ($s = max|w|\/7$, "wastes none of the 15 levels")
and executes it accurately. In each layer, 98.1–100% of integer weights equal round($w\/s$) for its own scale, and 47 of the 48
exceptions lie within 0.02 of a tie. Its one departure is a lower scale (median 0.96× RTN) for conv4 rows with one
or two outliers. It scores 60.58%, +0.2 pp over RTN. The conv4 clipping is worth +1.5 pp, while three conv1 integer weights
rounded the other way at exact ties cost 1.1 pp. Three rows have the wrong length (143, 290, 287 integer weights) and fall
back to RTN. Without the RTN integer weights and without being told that RTN fails, the LLM does not look for a problem.

_With BN statistics (P4)_, no LLM uses them (@tab-all). Opus 5.5 and Fable 5.2 state that the statistics were only
context. Sonnet 5.5 argues that they are already folded into the weights, which is true for the scales and misses
that the running means describe activations. The statistics would have allowed one useful correction. They give the input means $mu_c$ (@sec-why), and with
them each row can be rounded so that its mean shift in @eq-shift cancels. None of the LLMs does this.

_Opus_ (63.26%) repeats its P3 strategy with more clipping, with median scale ratios of 0.96 (conv4) and 0.97 (fc).
Per-layer swaps into RTN give +2.0 pp (conv4) and +1.5 pp (fc). Re-rounding the float weights exactly at its scales gives 63.07%,
so the gain is in the scales. An MSE-optimal per-row scale search on conv4 and fc gives 64.39%, which Opus 5.5 approximates by eye.

_Fable 5.2_ (69.97%, all 114 rows valid) is the best LLM result without the RTN integer weights. Its summary describes
"per-row round-to-nearest with a mildly clipped scale" and adds "when one weight was an isolated outlier, I picked a
smaller scale and let that weight clip to ±7". The data confirm this. In conv2, conv4 and fc the scale ratio
correlates with the ratio of largest to second-largest weight at −0.87, −0.85 and −0.90, exact rounding at its
scales gives 69.94%, and kernel sums are not reduced. @sec-clipping analyzes this result.

_Sonnet and Haiku_ choose a workable method and fail at execution. Both write plain max/7 RTN, but most rows have the wrong number
of integer weights or are cut off (19–39 of 114 valid). Some valid Sonnet 5.5 rows contain shifted integer weights, and its valid conv4 rows
have an RMS error of 1.7 steps against 0.29 for RTN. Haiku 5.5 often does not return the
correct number of integer weights, and otherwise more or less copies the input. Its valid rows
equal RTN in 98–100% of the integer weights per layer.

== Inference cost <sec-tokens>

For comparison, kernel balancing in Python runs in milliseconds on a CPU, and AdaRound
needs a few minutes on one GPU. @tab-cost lists the approximate price of the LLM runs. We used a Claude Pro subscription, so those are estimates based on the number of tokens. All API calls of the experiments together produced 3.6M output tokens and would probably cost about \$81 at list API prices. The P4
runs account for two thirds, because each LLM writes every integer weight of the network. Output tokens dominate
wherever an LLM computes by hand, because summing nine residuals per kernel and writing the result costs reasoning
tokens for every kernel.

The two best runs differ in price per gain. Fable 5.2 in P4 wrote 272k output tokens over 21 responses, about 90% of them hidden reasoning, and read
4.2M input tokens, almost all from the prompt cache. At \$24.7 (\$10.9 output, \$13.2 cache writes) it gains
9.6 pp. Opus 5.5 in P2 wrote 447k output tokens over 15 responses with a context of up to 677k tokens. At about
\$13 it gains 17.6 pp. The P1 run of Opus 5.5 wrote 134k output tokens for 3.7 pp.

#figure(
  table(columns: 3, align: (left, right, right),
    hl, [Protocol], [Output tokens], [Cost (USD)], hl,
    [P1 whole network],  [178k],  [5#super[a]],
    [P2 layer by layer], [525k],  [15#super[a]],
    [P3 float only],     [235k],  [7],
    [P4 float + BN],     [2,642k], [53], hl,
    [Total],             [3,580k], [81], hl,
  ),
  caption: [Output tokens and cost at list API prices per 1M tokens (input/output): Opus 5.5 \$4/\$20, Sonnet 5.5
  \$2/\$10, Haiku 5.5 \$0.50/\$2.50 (context above 100k), Fable 5.2 \$8/\$40. Cache writes cost 1.25× and cache
  reads 0.05× the input price. P3 and P4 from the logged usage of all API calls.
  #super[a]Estimated from logged output tokens and maximum context, assuming linear context growth.],
) <tab-cost>

_Input encoding._ The int4 payload is 161k characters. Kernel balancing needs only the integer weight and a
coarse residual. Running it on the residual quantized to three levels per step gives 78.68%, and on its sign alone
66.45%. A packed encoding (one letter for $q$, one for round($10(u - q)$)) needs a quarter of the characters. We
kept the neutral format, since a compressed one would hint at the method.

= What the LLMs did and why it works <sec-analysis>

The LLMs that gain over RTN use one of two mechanisms. In P1/P2 they keep the RTN scales and change the rounding
of single weights (kernel balancing). In P3/P4 they keep plain rounding and change the scales (outlier clipping).
This section explains why each works and why the LLMs probably chose them.

== Kernel balancing <sec-model>

Two observations call for an explanation. First, the per-weight error of int4 RTN is small (about 13% relative
RMS, @tab-layers), yet accuracy drops by 21 pp. Second, kernel balancing makes individual weights _less_ accurate,
so per-weight error cannot be what it optimizes. A first-order model of the rounding error explains both and makes
predictions that we test on validation data. A conv output is $y_o = sum_c sum_k W_(o c k) x_(c k)$ over input
channels $c$ and kernel positions $k$. With the weight error $Delta W = s q - w = s e$, the output error is $Delta y_o = sum_c sum_k Delta W_(o c k) x_(c k)$.

=== Why the kernel sum matters <sec-constant>

The output error of one kernel is $Delta y = s sum_k e_k x_k$, so each rounding error
is weighted by the input it multiplies. Consider a constant input patch, where the nine inputs under the kernel (pixels
or feature-map activations) are equal, $x_k = x$. Then $x$ factors out, and $Delta y = s x sum_k e_k = s x E$ with
$E = sum_k e_k$ the rounding-error sum of the kernel. Which weight carries which error no longer matters, only
the total. @fig-kernel(a) shows two error patterns. In kernel A the individual errors are large ($norm(e)^2 =
0.50$) but cancel, $E = 0$, so a constant patch produces no output error. In kernel B every error is small ($+0.1$,
$norm(e)^2 = 0.09$) but all have the same sign, so they add up to $E = 0.9$ and the output is off by $0.9 s x$.
Per-weight error and output error rank the two kernels in opposite order. RTN minimizes every $|e_k|$ separately
and never checks whether the errors cancel.

Of course real patches are not always close to constant (e.g. at edges). Writing each input as the patch mean plus a deviation,
$x_k = macron(x) + d_k$ with $sum_k d_k = 0$, splits the output error exactly into

$ Delta y = s macron(x) E + s sum_k e_k d_k . $ <eq-split>

The first term depends only on the kernel sum. The second comes from edges and texture inside the patch and
depends on the individual errors. Kernel balancing will not reduce it and might even increase it slightly. Natural
images and feature maps are mostly smooth, so the first term dominates, except at edges. @sec-why measures
both terms. @eq-equicorr below quantifies this balance through the input correlation $rho$. That a kernel
responds mainly to the sum of its weights is also the argument Opus 5.5 and Sonnet 5.5 give for their edits in P1
(@sec-whole).

=== Mean shift and fluctuating error

The constant-patch argument has two consequences on real data. Averaged over positions and images, each input channel $c$ has a mean input
$mu_c = bb(E)[x_(c k)]$ (about 0 for the normalized image, positive after ReLU), which takes the place of $x_c$
and gives an expected output shift

$ bb(E)[Delta y_o] = sum_c (sum_(k in 3 times 3) Delta W_(o c k)) mu_c = s_o sum_c E_(o c) mu_c , $ <eq-shift>

a bias that is the same at every position and in every image, so it does not average out. What remains,
$Delta y_o - bb(E)[Delta y_o]$, is a fluctuating part. It changes with the content of each patch and averages to
zero, like noise. A bias correction can cancel the mean but not this part. Its size depends on how the inputs vary
together. Assume the nine inputs of a kernel have variance $sigma^2$ and common correlation $rho$. Then the
variance of the fluctuating part, in units of $s^2$, is

$ e^top Sigma e = sigma^2 [(1 - rho) norm(e)_2^2 + rho E^2], $ <eq-equicorr>

which interpolates between the per-weight error $norm(e)_2^2$ (uncorrelated inputs, $rho = 0$) and the kernel sum
$E^2$ (constant patches, $rho = 1$). @app-rho relates $sigma^2$ and $rho$ to the two terms of @eq-split. RTN
minimizes every $|e_k|$ but leaves $E$ unconstrained. With nine i.i.d.
uniform errors, $"std"(E) approx 0.87$ steps.

#figure(
  {
    let cell(v, err: none, hl: false) = table.cell(fill: if hl { rgb("#fde2c8") } else { none },
      align(center, stack(spacing: 2pt, text(size: 7.5pt, v),
        if err != none { text(size: 6.5pt, fill: luma(90), err) })))
    let kern(title, cells, foot) = stack(spacing: 4pt,
      align(center, text(size: 8pt, title)),
      table(columns: (0.8cm,) * 3, rows: 0.62cm, stroke: 0.4pt + luma(150), inset: 1.5pt, ..cells),
      align(center, text(size: 8pt, foot)))
    let A = ("+.4", "−.4", "0", "+.3", "−.3", "0", "0", "0", "0")
    let B = ("+.1",) * 9
    let U = ("2.58", "−0.62", "1.55", "0.57", "3.40", "−1.62", "−2.44", "0.61", "1.62")
    let Q = ("3", "−1", "2", "1", "3", "−2", "−2", "1", "2")
    let Ee = ("+.42", "−.38", "+.45", "+.43", "−.40", "−.38", "+.44", "+.39", "+.38")
    let Q2 = ("3", "−1", "1", "1", "3", "−2", "−2", "1", "2")
    let E2 = ("+.42", "−.38", "−.55", "+.43", "−.40", "−.38", "+.44", "+.39", "+.38")
    stack(spacing: 0.5cm,
      stack(spacing: 4pt, align(center, text(size: 8pt)[(a) Two error patterns $e$ on a constant patch]),
        align(center, grid(columns: 2, column-gutter: 0.8cm,
          kern([kernel A], A.map(v => cell(v)), [$norm(e)^2 = 0.50$ \ $E = 0$, $Delta y = 0$]),
          kern([kernel B], B.map(v => cell(v)), [$norm(e)^2 = 0.09$ \ $E = 0.9$, $Delta y = 0.9 s x$]),
        ))),
      stack(spacing: 4pt, align(center, text(size: 8pt)[(b) Kernel balancing on one kernel]),
        align(center, grid(columns: 3, column-gutter: 0.2cm,
          kern([float $u = w\/s$], U.map(v => cell(v)), [ ]),
          kern([RTN $q$ (error $e$)], Q.zip(Ee).map(((a, b)) => cell(a, err: b)),
            [$norm(e)^2 = 1.50$ \ $E = +1.35$]),
          kern([balanced (error $e$)], Q2.zip(E2).enumerate().map(((i, (a, b))) => cell(a, err: b, hl: i == 2)),
            [$norm(e)^2 = 1.60$ \ $E = +0.35$]),
        ))),
    )
  },
  kind: image,
  caption: [(a) On a constant input patch ($x_k = x$) the output error is $s x E$. Kernel A has the larger per-weight
  error but its errors cancel. Kernel B has small errors of one sign that add up. (b) RTN errors that happen to be
  mostly positive sum to $E = +1.35$ steps. Moving one weight down by one step lowers $E$ by 1. Balancing
  picks the weight with the largest positive error, i.e. the one closest to its rounding boundary (1.55, error
  +.45), and rounds it down from 2 to 1 instead of up. The squared error
  grows by $1 - 2 dot 0.45 = 0.10$, but on a constant patch the output error falls from $1.35 s x$ to $0.35 s x$.],
) <fig-kernel>

=== Two corrections <sec-two-corrections>

_Bias correction_ @nagel2019 keeps the integer weights and changes the bias. It adds
$delta b_o = -bb(E)[Delta y_o]$ from @eq-shift, which removes the mean shift. It needs the input means $mu_c$, from
data or from the BN parameters, and leaves the fluctuating part of the error unchanged.

_Kernel balancing_ (@fig-kernel\(b)) keeps the bias and changes integer weights until every kernel has
$|E| <= 1\/2$. Rounding a weight to its other neighboring integer changes its error, and therefore $E$, by
exactly 1. A kernel with $E = +1.35$, for example, reaches $E = +0.35$ when one weight that was rounded up is
rounded down instead. The weights to change are chosen by cost. Changing a weight with error $e_k$ raises its
squared error by $1 - 2|e_k|$. The cheapest weights are those with $|e_k|$ close to $1\/2$, whose float value lay
almost halfway between two integers. Balancing therefore changes the weights whose error has the same sign as $E$
and the largest magnitude. It needs no data and changes few integer weights. Because $E$ appears in both
@eq-shift and @eq-equicorr, a smaller $E$ lowers the mean shift for any input means and lowers the correlated part
of the error whenever $rho > 0$. For fc layers we apply the same rule to the sum over the whole row.

Kernel balancing is the kernel-wise step of SQuant @guo2022squant, and our implementation produces the same
integer weights (_exact kernel balancing_ in the tables). SQuant motivates this step with a Hessian approximation that is constant within a kernel, which
corresponds to $rho > 0$ in @eq-equicorr. @eq-shift gives a second reason for it, because the same sum sets the mean shift.

=== Validation on data <sec-why>

We test the error model of @sec-model on validation data. The mean shift predicted by @eq-shift from measured input
means matches the measured pre-activation shift with correlation ≥ 0.994 in every layer. At int4 it is 7–11% of
the output standard deviation. For the unpooled inputs of conv2 and fc the input means can also be predicted from BN parameters alone (a clipped Gaussian with mean $beta$
and std $|gamma|$, correlation 0.95 and 0.92), as in DFQ @nagel2019. We did not test the max-pooled inputs of
conv3 and conv4. Since biases are fixed in P4, the LLMs could use these means only through the rounding (@sec-direct).

@tab-decomp splits the expected squared layer error, using measured patch covariances. The mean term is about 40%
of the RTN error. Bias correction removes it and nothing else. Kernel balancing removes about 90% of it and makes
the off-diagonal term slightly more negative, so both corrections reach nearly the same total error (.014 and
.013). A relative error of only 0.024 thus costs 21 pp, because a small systematic error matters more than a larger
random one.

#figure(
  table(columns: 5, align: (left, ..range(4).map(_ => right)),
    hl, [int4], [mean], [diag], [off], [total], hl,
    [RTN],           [.010], [.018], [−.004], [.024],
    [\+ bias corr.], [.000], [.018], [−.004], [.014],
    [kernel bal.],   [.001], [.019], [−.007], [.013],
    [both],          [.000], [.019], [−.007], [.012], hl,
  ),
  caption: [Expected squared layer error at int4 relative to the float pre-activation variance (mean over layers,
  float inputs), split into mean, diagonal and off-diagonal parts.],
) <tab-decomp>

The split of @eq-split shows where balancing helps (@fig-edge). Under RTN the patch-mean term, which holds the mean shift of @tab-decomp plus the fluctuation of the patch mean, carries 53–76% of the
squared output error in conv1–conv4. Balancing cuts this term to 9–30% of its RTN value, while the detail term
grows by 21–46%. On the smoothest tenth of patches it removes 68–87% of the error in conv1–conv3, while on the sharpest
edges in conv1 it increases the error by half, because there the individual errors matter.

#figure(
  image("figs/edge_split.svg", width: 100%),
  caption: [Exact split of the int4 output error (@eq-split) on validation patches. (a) Mean-square error relative to
  the float pre-activation variance, split into the patch-mean term (solid) and the detail term (light), for RTN
  and kernel balancing. (b) Error of kernel balancing relative to RTN, with patches sorted into deciles by their
  contrast, from smooth (1) to edges (10).],
) <fig-edge>

== Outlier clipping <sec-clipping>

Fable 5.2's P4 result (@sec-direct) comes from the scales alone. Exact rounding at its scales gives 69.94%, and its
kernel sums are not reduced. It clips less than MSE search (median ratios in conv2–fc
0.95–0.98 vs. 0.80–0.92, both about 0.99 in conv1), yet MSE search on all layers gives only 64.6%. A possible explanation, which we did not test, is that clipping one isolated outlier gives the bulk of a
heavy-tailed row a finer grid at small cost, while clipping a group of large weights, which MSE does, puts the
error on the weights that matter most.

The gain is uneven across layers, with +8.8 pp in conv2, +2.8 in conv4, +0.3 in fc, −2.7 in conv1 and −2.0 in conv3 (72.1% without its
conv1 rows). Conv2 is unusually sensitive. Fable 5.2's conv2 rows alone give 69.2%.
Random per-row scale factors in $[0.88, 1]$ give 66.1% on average (61.4–69.0%, 12 draws), and Fable 5.2's ratios
shuffled across rows 64.3% (60.1–68.0%, 12 shuffles). Its assignment beats all 24 controls, but the best random
draw comes within 0.3 pp, and with one run we cannot quantify how much luck it holds. Random factors already
recover 5.7 of the 8.8 pp, so most of the conv2 gain comes from re-rounding at a slightly smaller scale, not from
the choice of rows.

Opus 5.5 does the same more weakly in P4, in conv4 and fc only (63.26%, @sec-direct). Outlier clipping is a standard
refinement of RTN, applied by MSE search in most toolchains @nagel2021whitepaper. Fable 5.2's choice of rows adds about
3 pp over random factors in conv2 and costs accuracy in conv1 and conv3.

== Why did the LLMs use these methods? <sec-rules>

_What the input allows._ In P1/P2 the LLM may change integer weights, scales and biases, but it sees no data and no
BN statistics. Bias correction needs the input means $mu_c$ of @eq-shift, so it would require guessing them. Scale
changes need an error criterion that the LLM would have to evaluate by hand for every candidate scale. The kernel
sum is the one quantity that predicts the output error without knowing the inputs, under the single assumption
that inputs are smooth, and the P1/P2 format shows every residual $u - q$ next to its integer weight. Kernel
balancing is therefore close to the cheapest correction that the input supports. Haiku 5.5 states the mean-shift
argument of @eq-shift directly for the fc layer. Its inputs "are non-negative after ReLU and global average
pooling, so the sum of rounding errors in each row acts as a systematic shift on the logits".

_Recall or derivation._ No LLM names SQuant, but Opus 5.5 places its rule in the literature, writing "The idea follows
the bias-correction and adaptive-rounding view of post-training quantization." Our runs therefore cannot tell
whether the rule is derived from the weights or recalled and then recognized as applicable. A prompt that
describes the setting without any weights would separate the two.

_Known and declined._ In P3/P4 the LLMs list the corrections and decline them. Opus 5.5 writes "Biases are fixed and
there is no data, so I couldn't do bias correction, activation-aware rounding, or cross-layer equalization."
Sonnet 5.5 writes "I didn't use bias correction, cross-layer equalization, or any rounding that cancels error across a
row." Sonnet 5.5 thus names error-cancelling rounding in P4 and does not apply it. Both treat plain rounding as an
adequate answer when nothing says that it fails. @sec-conclusion discusses which differences between the prompts
may cause this.

= Limitations <sec-limits>

_Statistics._ Every LLM number comes from one run of a particular LLM version. The binomial standard error is about
0.5 pp at 60%, and single flips in conv1 or conv2 move accuracy by about 1 pp. Differences below about 1 pp, such
as the P3 result and the Haiku 5.5 results, are therefore within noise. We ran no paired test and no control with random near-tie flips for P1.

_Scope._ All four LLMs are Claude models. The network is small, and other architectures will have different rounding statistics. The input format of
P1/P2 exposes the rounding residual, which hints at rounding-aware corrections. P4 ran in a different harness, and
we have no P3 runs for Sonnet 5.5, Haiku 5.5 and Fable 5.2, so the effect of the BN statistics on their behavior
is not isolated.

_Execution._ The tool restriction in P1–P3 rests on instructions and audited transcripts, and hidden reasoning is not
visible. Some runs were interrupted by usage limits and resumed, and in two runs the LLM's first output was only in
hidden reasoning and had to be re-emitted. The Opus 5.5 P2 run stopped after chunk 10. Its conv1 and conv2 edits
had been written only in hidden reasoning and were re-emitted as text after resumption (without them it scores
69.58%), and it did not redo the fc chunk.

_Proprietary APIs._ Midway through the experiments the API safety classifiers began to reject our runs. Requests
of the same kind that had run before were stopped at the first message, for P4 with Opus 5.5 and Sonnet 5.5 as
subagents and for a P3 rerun. We had to change the harness and try several prompt variants until the runs were
accepted again, which is why P4 ran from the command line (@sec-setup). We do not know why the classifier fired.
Such changes are invisible to the user, can stop an experiment at any point and make it hard to reproduce. This is
a general problem of research run against proprietary APIs.

_Next steps._ Natural next steps are repeated runs with paired tests, a random-flip control for P1, a bias-correction reference from BN statistics alone, prompts that state the RTN gap in P3/P4, a P1 prompt with raw float weights, a prompt that describes the task without weights (to separate recall from derivation), a small
evaluation budget, and a compressed delivery format.

= Conclusion <sec-conclusion>

*Without tools, an LLM can repair part of a quantized CNN by only reading its weights*. At 4 bit two LLMs use kernel balancing (the kernel step of SQuant), a data-free rule that removes most of the mean shift our
measurements show in every layer. Delivered one layer per answer, Opus 5.5 works through all 2,200 kernels by hand
and reaches 77.95%, 3.5 pp below float. Tasked to output only a single answer the LLMs found the method but could apply it only to conv1,
because the arithmetic for the other layers does not fit. When the
LLM must quantize float weights itself, the gains come from row-wise outlier clipping, and Fable 5.2 scores
above an MSE scale search (70.0% vs. 64.6%), partly by luck (@sec-clipping). Rounding that cancels the mean shift, for which the BatchNorm statistics given in P4 suffice at least in conv2 and fc, appears in no run. Calibrated methods (AdaRound, GPTQ) reach 80.7% on the same network, so the
best LLM result is 2.8 pp short of methods that use calibration data.

_Why P3 and P4 did not perform rounding correction._ Kernel balancing appears only when the LLM sees the RTN network and
is told that it fails, and even then not in every run (Sonnet 5.5 and Haiku 5.5 in P2). In P3 and P4 every LLM writes plain rounding, at most with clipped scales.

- _Stated defect._ P1/P2 report that rounding lowers accuracy on a held-out split from 82.1% to 60.4%, which turns the task into the
  search for a defect. P3/P4 report only the float accuracy, and the LLMs treat plain rounding as the expected
  answer (Fable 5.2 writes "I expect it to behave like a reasonable 4-bit per-channel post-training quantization").
- _Visible residual._ P1/P2 list every weight in units of the step, $u = w\/s$, next to its integer $q$, so the
  rounding error $u - q$ can be read off and summed per kernel. In P3/P4 each residual first requires a division by
  a scale that the LLM chooses itself.
- _Output volume._ P1/P2 accept sparse edits. P3/P4 require all 20k integer weights, so most of the output goes
  into writing integers.
- _Reference rounding._ The RTN integers show what plain rounding produces and invite a comparison of integer and
  float sums, which is how Opus 5.5 and Sonnet 5.5 describe their P1 idea.

Fable 5.2 came close in P4, writing that it was "sometimes alternating direction within a row so the row sum would
not drift one way". A P3 prompt that states the RTN gap would test
the first factor, and a P1 prompt with raw float weights the second.

Code, prompts, replies and result files are available at #link("https://github.com/urbste/weight-space-programming-ptq")[`github.com/urbste/weight-space-programming-ptq`].

_Use of AI tools._ The idea of the study, the four protocols and the experiments are the author's. AI tools were
used to write the protocol prompts and the evaluation code, to carry out the experiments and to write the paper.
The author read and corrected the paper and takes responsibility for its content.

#set text(size: 8.5pt)
#bibliography("refs.bib", style: "ieee")

#set text(size: 9.5pt)
#heading(numbering: none)[Appendix]
#set heading(numbering: (..n) => "A." + n.pos().slice(1).map(str).join("."))
#counter(heading).update(0)

== Input format and protocol <app-protocol>

P1/P2 input (excerpt, int4):
```
conv1 r0 scale 0.0635392 bias 0.3771
conv1 r0 u  c0: 3.83 -1.47 0.37 -4.49 2.26 -0.49 0.82 ...
conv1 r0 q  c0: 4 -1 0 -4 2 0 1 -1 -1 | c1: 0 0 0 -7 ...
```
Edit language: `Q <layer> <row> <in_ch> <pos> <code>`, `K <layer> <row> <in_ch> <9 codes>`,
`R fc <row> <48 codes>`, `S <layer> <row> <scale>`, `B <layer> <row> <bias>`. P2 instruction: "In EACH response
read exactly one chunk file, write the edit lines for its rows between `BEGIN CHUNK k` and `END CHUNK k`, then
read the next chunk." P3/P4 answer: `ROW <layer> <r> SCALE <s> Q <codes>`. Tool rule (P1–P3): "The only tool you
may call is the file-reading tool, and only on this single file. [...] no terminal, no code execution, no Python,
no calculator, no file writing, no web, no subagents."

== Input variance and the two error terms <app-rho>

In @eq-equicorr, $sigma^2$ is the variance of a single input across patches, measured from the channel mean
$mu_c$, not the within-patch deviation $d_k$ of @eq-split. The two are linked by $rho$. The patch mean has
variance $"Var"(macron(x)) = sigma^2 (1 + 8 rho)\/9$ and each deviation $bb(E)[d_k^2] = sigma^2 dot 8(1 - rho)\/9$.
Hence $rho$ sets how much of the input variation reaches the kernel-sum term of @eq-split and how much reaches the
detail term.

== Token usage <app-tokens>

#figure(
  table(columns: 4, align: (left, right, right, right),
    hl, [Run (int4)], [Resp.], [Output], [Max. ctx], hl,
    [Opus 5.5 P1],   [9],  [134k], [239k],
    [Sonnet 5.5 P1], [6],  [11k],  [239k],
    [Haiku 5.5 P1],  [15], [33k],  [238k],
    [Opus 5.5 P2],   [15], [447k], [677k],
    [Sonnet 5.5 P2], [13], [2k],   [238k],
    [Haiku 5.5 P2],  [13], [74k],  [260k],
    [Opus 5.5 P3],   [22], [235k], [–],
    [Fable 5.2 P4],  [21], [272k], [409k], hl,
  ),
  caption: [Responses, output tokens and maximum context per run. Not listed: the P4 runs of Opus 5.5, Sonnet 5.5
  and Haiku 5.5, which together wrote 2,370k output tokens.],
) <tab-tokens>


#set page(columns: 1)
== Task description (P1, int4) <app-prompt>

@fig-prompt reproduces the P1 task description at int4 verbatim, followed by the start of the network dump. The
P2 description is the same with the chunk instructions added. P3 and P4 replace the RTN part with the float
weights and the `ROW` answer format. All prompts are in the project repository.

#figure(
  block(width: 100%, fill: luma(246), inset: 6pt, radius: 2pt,
    align(left, {
      show raw: set text(size: 6.3pt)
      set par(leading: 0.42em)
      raw(read("prompt_p1_int4.txt"), block: true)
    })),
  kind: image,
  caption: [P1 task description at int4 as sent to the LLM. The dump lines of conv1 row 0 are cut here; the
    LLM receives them in full.],
) <fig-prompt>

#pagebreak()
== Task description (P2) <app-prompt-p2>

P2 uses the P1 task description up to the edit language and replaces the answer format with chunked delivery
(@fig-prompt-p2). Each chunk file holds the same dump format as P1.

#figure(
  block(width: 100%, fill: luma(246), inset: 6pt, radius: 2pt,
    align(left, {
      show raw: set text(size: 6.3pt)
      set par(leading: 0.42em)
      raw(read("prompt_p2_int4.txt"), block: true)
    })),
  kind: image,
  caption: [P2 task description at int4: the parts that differ from P1, and the start of the first chunk. Long lines are cut here.],
) <fig-prompt-p2>

#pagebreak()
== Task description (P3) <app-prompt-p3>

The P3 task asks the LLM to quantize the float network itself (@fig-prompt-p3). It names neither RTN nor the
accuracy of a quantized network; it gives only the float accuracy.

#figure(
  block(width: 100%, fill: luma(246), inset: 6pt, radius: 2pt,
    align(left, {
      show raw: set text(size: 6.3pt)
      set par(leading: 0.42em)
      raw(read("prompt_p3.txt"), block: true)
    })),
  kind: image,
  caption: [P3 task description, as read by the subagent, and the start of the first chunk file. Long lines are cut here.],
) <fig-prompt-p3>

#pagebreak()
== Task description (P4) <app-prompt-p4>

P4 adds the BN statistics to the P3 task and runs from the command line with all tools disabled, so the "No
tools" rule is dropped and the chunks arrive as messages (@fig-prompt-p4). The same text was sent to every LLM.

#figure(
  block(width: 100%, fill: luma(246), inset: 6pt, radius: 2pt,
    align(left, {
      show raw: set text(size: 6.3pt)
      set par(leading: 0.42em)
      raw(read("prompt_p4.txt"), block: true)
    })),
  kind: image,
  caption: [P4 task description as sent from the command line, and the start of the first chunk message. Long lines are cut here.],
) <fig-prompt-p4>
