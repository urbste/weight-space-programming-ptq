"""Layer-level analysis of the small network: where does int4 lose accuracy, why does kernel balancing help,
how good are the tool-free LLM edits layer by layer.

python analysis/analyze.py   -> results/analysis.json, paper/figs/*.svg   (numpy + matplotlib only)

Everything is computed on the 5k validation images (no augmentation); the test set is used only for accuracy.
"Calibration statistics" below (input means, patch covariances) come from FP32 activations on those images and
are used for diagnosis only, never by the LLM-edited networks.
"""
import copy
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "llm")]
os.chdir(ROOT)
from eval_layerwise import chunk_blocks  # noqa: E402
from eval_llm import apply_edits, parse_edits  # noqa: E402
from evaluate import evaluate, get_data  # noqa: E402
from ptq_eval import load_layers  # noqa: E402
from quant import LAYERS, conv3x3, dequantize, fold_bn, load_npz, maxpool2, qmax_for, relu, rtn_quantize  # noqa: E402
from reference_methods import balance_groups, kernel_balance, ternary, twn_classic  # noqa: E402

CONVS = LAYERS[:4]
FIG = ROOT / "paper/figs"
FIG.mkdir(exist_ok=True, parents=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "figure.dpi": 120})
COL = {"fp32": "0.55", "rtn": "#c0392b", "bc": "#e08e0b", "kb": "#2a9d8f", "kbbc": "#1b6f66",
       "opus_lw": "#1f5fbf", "opus_whole": "#7aa5e8", "haiku": "#8e44ad", "sonnet": "#d98cb3"}
NAME = {"rtn": "RTN", "bc": "RTN + bias corr.", "kb": "kernel balancing (code)", "kbbc": "kernel bal. + bias corr.",
        "opus_lw": "Opus, layer by layer", "opus_whole": "Opus, whole network", "haiku": "Haiku, whole network",
        "sonnet": "Sonnet, whole network"}

# ----------------------------------------------------------------------------------------------------------
sd, meta = load_npz("checkpoints/small_e500.npz")
VAL_IDX = meta["val_idx"]
FL = fold_bn(sd)
DATA = get_data(VAL_IDX)
XV, YV = DATA["val"]
REF = np.load("exports/fp32_test_preds.npy")


def forward_acts(L, x, bs=500):
    """Per-layer outputs: a1..a4 are post-ReLU conv outputs (before pooling), 'fc' the logits."""
    outs = {k: [] for k in LAYERS}
    for i in range(0, len(x), bs):
        h = x[i:i + bs]
        a1 = relu(conv3x3(h, L["conv1"]["weight"], L["conv1"]["bias"]))
        a2 = relu(conv3x3(a1, L["conv2"]["weight"], L["conv2"]["bias"]))
        a3 = relu(conv3x3(maxpool2(a2), L["conv3"]["weight"], L["conv3"]["bias"]))
        a4 = relu(conv3x3(maxpool2(a3), L["conv4"]["weight"], L["conv4"]["bias"]))
        lg = a4.mean((2, 3)) @ L["fc"]["weight"].T + L["fc"]["bias"]
        for k, v in zip(LAYERS, (a1, a2, a3, a4, lg)):
            outs[k].append(v)
    return {k: np.concatenate(v) for k, v in outs.items()}


def layer_inputs(acts, x):
    """FP32 input of every layer, given FP32 activations."""
    return {"conv1": x, "conv2": acts["conv1"], "conv3": maxpool2(acts["conv2"]),
            "conv4": maxpool2(acts["conv3"]), "fc": acts["conv4"].mean((2, 3))}


def apply_layer(name, w, b, inp):
    if name == "fc":
        return inp @ w.T + b
    return relu(conv3x3(inp, w, b))


def batched(f, x, bs=500):
    return np.concatenate([f(x[i:i + bs]) for i in range(0, len(x), bs)])


def snr_db(y, ref):
    return float(10 * np.log10((ref.astype(np.float64) ** 2).sum() / max(((y - ref).astype(np.float64) ** 2).sum(), 1e-30)))


def layer_snr(L, acts_fp, ins_fp):
    acc = forward_acts(L, XV)
    out = {}
    for n in LAYERS:
        iso = batched(lambda a: apply_layer(n, L[n]["weight"], L[n]["bias"], a), ins_fp[n])
        out[n] = {"accum": snr_db(acc[n], acts_fp[n]), "iso": snr_db(iso, acts_fp[n])}
    return out


def acc_of(L):
    r, _ = evaluate(L, VAL_IDX, REF)
    return r


# ---------------------------------------------------------------------------------------------- variants
def u_of(name, q):
    w = FL[name]["weight"].astype(np.float64)
    return w / q[name]["scale"].reshape((-1,) + (1,) * (w.ndim - 1)).astype(np.float64)


def kb_all(q0, bits):
    return kernel_balance(FL, q0, CONVS, bits)


def llm_variants():
    q0 = load_layers("exports/rtn_int4.npz")
    out = {}
    for m, key in (("opus", "opus_whole"), ("sonnet", "sonnet"), ("haiku", "haiku")):
        q, _ = apply_edits(q0, parse_edits(Path(f"llm/replies/{m}_int4.txt").read_text()), 4)
        out[key] = q
    blocks = chunk_blocks(Path("llm/layerwise/replies/opus_int4.txt").read_text())
    lines = [l for k in sorted(blocks) for l in blocks[k]]
    out["opus_lw"], _ = apply_edits(q0, lines, 4)
    return out


def mean_inputs(acts_fp, ins_fp):
    """Per-input-channel mean of each layer's FP32 input (over images and positions)."""
    mu = {}
    for n in LAYERS:
        a = ins_fp[n]
        mu[n] = a.mean(0) if n == "fc" else a.mean((0, 2, 3))
    return mu


def bias_correct(Lf, mu):
    """b <- b - sum_c (sum_k dW_ock) mu_c, with dW = quantized - FP32 weights and mu the FP32 input means."""
    out = copy.deepcopy(Lf)
    for n in LAYERS:
        dw = Lf[n]["weight"].astype(np.float64) - FL[n]["weight"].astype(np.float64)
        shift = (dw.sum((2, 3)) if dw.ndim == 4 else dw) @ mu[n]
        out[n]["bias"] = (Lf[n]["bias"] - shift).astype(np.float32)
    return out


# ------------------------------------------------------------------------------------------------ analysis
def weight_table():
    res = {}
    for n in LAYERS:
        w = FL[n]["weight"].reshape(FL[n]["weight"].shape[0], -1).astype(np.float64)
        rms = np.sqrt((w ** 2).mean(1))
        r = (np.abs(w).max(1) / rms).mean()
        z = (w - w.mean(1, keepdims=True)) / w.std(1, keepdims=True)
        res[n] = {"w_min": float(w.min()), "w_max": float(w.max()), "r": float(r), "kurt": float((z ** 4).mean() - 3),
                  "n_weights": int(w.size)}
        for bits in (8, 4, 2):
            Q = qmax_for(bits)
            q = rtn_quantize({n: FL[n]}, bits)[n]
            deq = q["q"].reshape(w.shape) * q["scale"][:, None]
            from math import erf, sqrt
            Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
            res[n][f"zeros_int{bits}"] = float((q["q"] == 0).mean())
            res[n][f"zeros_int{bits}_pred"] = 2 * Phi(r / (2 * Q)) - 1
            res[n][f"relerr_int{bits}"] = float(np.sqrt(((deq - w) ** 2).mean(1)).mean() / rms.mean())
            res[n][f"relerr_int{bits}_pred"] = float(r / (np.sqrt(12) * Q))
    return res


def kernel_sums(q, n):
    e = q[n]["q"].astype(np.float64) - u_of(n, q)
    return e.reshape(e.shape[0], -1).sum(1) if n == "fc" else e.reshape(e.shape[0] * e.shape[1], -1).sum(1)


def neighbour_corr(ins_fp):
    out = {}
    for n in CONVS:
        a = ins_fp[n]
        x, y = a[..., :-1], a[..., 1:]
        xm, ym = x - x.mean((0, 2, 3), keepdims=True), y - y.mean((0, 2, 3), keepdims=True)
        c = (xm * ym).mean((0, 2, 3)) / (xm.std((0, 2, 3)) * ym.std((0, 2, 3)) + 1e-9)
        out[n] = float(np.nanmean(c))
    return out


def mean_shift(Lq, ins_fp, acts_fp, mu, lname):
    """Predicted (dW.mu) versus measured mean shift of the pre-ReLU output, FP32 input, per output channel."""
    res = {}
    for n in LAYERS[1:]:
        dw = Lq[n]["weight"].astype(np.float64) - FL[n]["weight"].astype(np.float64)
        pred = (dw.sum((2, 3)) if dw.ndim == 4 else dw) @ mu[n]
        if n == "fc":
            yq = ins_fp[n] @ Lq[n]["weight"].T + Lq[n]["bias"]
            yf = ins_fp[n] @ FL[n]["weight"].T + FL[n]["bias"]
            meas = (yq - yf).mean(0)
            sd_ = yf.std(0)
        else:
            def f(a):
                return conv3x3(a, Lq[n]["weight"], Lq[n]["bias"]) - conv3x3(a, FL[n]["weight"], FL[n]["bias"])
            meas = batched(f, ins_fp[n]).mean((0, 2, 3))
            sd_ = batched(lambda a: conv3x3(a, FL[n]["weight"], FL[n]["bias"]), ins_fp[n]).std((0, 2, 3))
        resid = meas - pred
        res[n] = {"pred": pred.tolist(), "meas": meas.tolist(), "r": float(np.corrcoef(pred, meas)[0, 1]),
                  "explained": float(1 - (resid ** 2).sum() / (meas ** 2).sum()),
                  "shift_over_sd": float(np.mean(np.abs(meas) / sd_))}
    return res


def patches(a, n, maxn=500):
    a = a[:maxn]
    if n == "fc":
        return a.astype(np.float64)
    xp = np.pad(a, ((0, 0), (0, 0), (1, 1), (1, 1)))
    N, C, H, W = a.shape
    cols = np.stack([xp[:, :, i:i + H, j:j + W] for i in range(3) for j in range(3)], axis=2)  # N,C,9,H,W
    return cols.transpose(0, 3, 4, 1, 2).reshape(-1, C * 9).astype(np.float64)


def decomposition(Lq, ins_fp, mu_bc=False, patch_cache={}):
    """Per layer, averaged over output channels: error terms of E[(e.x + db)^2] / Var(z), e = dW row."""
    out = {}
    for n in LAYERS:
        if n not in patch_cache:
            P = patches(ins_fp[n], n)
            patch_cache[n] = (P.mean(0), np.cov(P.T))
        m, S = patch_cache[n]
        dw = (Lq[n]["weight"].astype(np.float64) - FL[n]["weight"].astype(np.float64))
        dw = dw.reshape(dw.shape[0], -1)
        wf = FL[n]["weight"].astype(np.float64).reshape(dw.shape[0], -1)
        db = Lq[n]["bias"].astype(np.float64) - FL[n]["bias"].astype(np.float64)
        var = np.einsum("oi,ij,oj->o", wf, S, wf)
        diag = (dw ** 2 * np.diag(S)).sum(1)
        full = np.einsum("oi,ij,oj->o", dw, S, dw)
        mean = (dw @ m + db) ** 2
        out[n] = {"mean": float((mean / var).mean()), "diag": float((diag / var).mean()),
                  "off": float(((full - diag) / var).mean()), "total": float(((mean + full) / var).mean())}
    return out


# ------------------------------------------------------------------------------------------------ figures
def fig_snr(snr):
    keys = ["rtn", "kb", "opus_whole", "opus_lw"]
    lab = {"rtn": "RTN", "kb": "exact kernel balancing", "opus_whole": "Opus 5.5 P1", "opus_lw": "Opus 5.5 P2"}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    w = 0.2
    for a, key, title in ((ax[0], "accum", "accumulated (whole int4 network)"), (ax[1], "iso", "isolated (FP32 input to the layer)")):
        for i, k in enumerate(keys):
            a.bar(np.arange(5) + (i - 1.5) * w, [snr[k][n][key] for n in LAYERS], w, color=COL[k], label=lab[k])
        a.set_xticks(range(5), LAYERS)
        a.set_title(title, fontsize=8)
    ax[0].set_ylabel("output SNR (dB)")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "snr.svg")
    plt.close(fig)


def fig_weights(qs, wtab):
    WLAB = {"rtn": "RTN (int4)", "kb": "exact kernel balancing", "opus_lw": "Opus 5.5 P2"}
    fig, ax = plt.subplots(2, 5, figsize=(9, 3.5))
    shown = ["rtn", "kb", "opus_lw"]
    for j, n in enumerate(LAYERS):
        w = FL[n]["weight"].ravel()
        lim = np.abs(w).max() * 1.05
        bins = np.linspace(-lim, lim, 101)
        ax[0, j].hist(w, bins=bins, color=COL["fp32"], alpha=0.5, label="FP32")
        ax[0, j].hist((qs["rtn"][n]["q"] * qs["rtn"][n]["scale"].reshape((-1,) + (1,) * (qs["rtn"][n]["q"].ndim - 1))).ravel(),
                      bins=bins, histtype="step", color=COL["rtn"], label="RTN (int4)")
        ax[0, j].set_yscale("log")
        ax[0, j].set_title(f"{n}  (r={wtab[n]['r']:.1f})", fontsize=8)
        for k in shown:
            ax[1, j].hist(kernel_sums(qs[k], n), bins=np.linspace(-3, 3, 49), histtype="step", color=COL[k], label=WLAB[k])
        ax[1, j].set_xlabel("kernel sum of rounding error" if n != "fc" else "row sum of rounding error")
    ax[0, 0].set_ylabel("weights")
    ax[1, 0].set_ylabel("kernels / rows")
    h, l = ax[0, 0].get_legend_handles_labels()
    h2, l2 = ax[1, 0].get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h + h2, l + l2):
        seen.setdefault(ll, hh)
    fig.legend(list(seen.values()), list(seen), loc="upper center", ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "weights.svg")
    plt.close(fig)


def fig_acts(acts_fp, acts_q):
    fig, ax = plt.subplots(2, 5, figsize=(9, 3.5))
    for j, n in enumerate(LAYERS):
        y = acts_fp[n].ravel()[::7]
        bins = np.linspace(y.min(), np.quantile(y, 0.9995), 80)
        ax[0, j].hist(y, bins=bins, color=COL["fp32"], alpha=0.5, label="FP32")
        err_all = []
        for k in ("rtn", "opus_lw"):
            yq = acts_q[k][n].ravel()[::7]
            ax[0, j].hist(yq, bins=bins, histtype="step", color=COL[k], label=NAME[k])
            err_all.append(yq - y)
        lim = np.quantile(np.abs(np.concatenate(err_all)), 0.995)
        for k, e in zip(("rtn", "opus_lw"), err_all):
            ax[1, j].hist(e, bins=np.linspace(-lim, lim, 70), histtype="step", color=COL[k])
        ax[0, j].set_yscale("log")
        ax[1, j].set_yscale("log")
        ax[0, j].set_title(n + (" (logits)" if n == "fc" else " (post-ReLU)"), fontsize=8)
        ax[1, j].set_xlabel("error vs. FP32")
    ax[0, 0].set_ylabel("count")
    ax[1, 0].set_ylabel("count")
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "acts.svg")
    plt.close(fig)


def fig_shift(shift):
    fig, ax = plt.subplots(1, 4, figsize=(8.5, 2.3))
    for a, n in zip(ax, LAYERS[1:]):
        for b, c in (("int2", "#999999"), ("int4", COL["rtn"])):
            d = shift[b][n]
            a.scatter(d["pred"], d["meas"], s=7, color=c, label=f"{b}", alpha=0.8)
        lim = max(np.abs(shift["int2"][n]["pred"]).max(), np.abs(shift["int2"][n]["meas"]).max()) * 1.05
        a.plot([-lim, lim], [-lim, lim], color="k", lw=0.6)
        a.set_title(f"{n}, r={shift['int4'][n]['r']:.3f} (int4)", fontsize=8)
        a.set_xlabel("predicted  $\\Delta W\\,\\mu$")
    ax[0].set_ylabel("measured mean shift")
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "shift.svg")
    plt.close(fig)


def fig_recovery(rec):
    fig, ax = plt.subplots(figsize=(4.3, 2.7))
    xs = ["RTN", "conv1", "conv2", "conv3", "conv4", "fc"]
    ax.plot(xs, rec["kb_exact"], "s--", color=COL["kb"], label="kernel balancing, exact")
    ax.plot(xs, rec["opus_lw"], "o-", color=COL["opus_lw"], label="Opus, layer by layer")
    ax.plot(xs, rec["haiku_lw"], "^-", color=COL["haiku"], label="Haiku, layer by layer")
    ax.plot(xs, rec["sonnet_lw"], "v-", color=COL["sonnet"], label="Sonnet, layer by layer (no edits)")
    ax.axhline(rec["fp32"], color="0.5", ls=":", lw=1)
    ax.text(0.1, rec["fp32"] + 0.8, f"FP32 {rec['fp32']:.1f}", color="0.4")
    ax.set_ylim(40, 86)
    ax.set_ylabel("test accuracy (%)")
    ax.set_xlabel("edits applied up to and including layer")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "recovery.svg")
    plt.close(fig)


def fig_bits(sweep, fp):
    fig, ax = plt.subplots(figsize=(4.3, 2.8))
    bits = sorted(sweep, reverse=True)
    for k, mk in (("rtn", "o"), ("bc", "s"), ("kb", "D"), ("kbbc", "^")):
        ax.plot(bits, [sweep[b][k]["test"] for b in bits], marker=mk, color=COL[k], label=NAME[k])
    ax.axhline(fp, color="0.5", ls="--", lw=1)
    ax.axhline(10, color="0.6", ls=":", lw=1)
    ax.set_xlabel("weight bits")
    ax.set_ylabel("test accuracy (%)")
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "bits.svg")
    plt.close(fig)


def fig_decomp(dec):
    fig, ax = plt.subplots(1, 3, figsize=(8, 2.5), sharey=False)
    keys = ["rtn", "bc", "kb", "kbbc"]
    for a, b in zip(ax, ("int4", "int3", "int2")):
        x = np.arange(len(keys))
        tot = lambda t: [np.mean([dec[b][k][n][t] for n in LAYERS]) for k in keys]
        a.bar(x, tot("diag"), color="#9ecae1", label="diagonal")
        a.bar(x, tot("mean"), bottom=tot("diag"), color="#fdae6b", label="mean")
        a.bar(x, tot("off"), color="#a1d99b", label="off-diagonal")
        a.plot(x, tot("total"), "k_", ms=12, mew=2, label="total")
        a.axhline(0, color="k", lw=0.5)
        a.set_xticks(x, ["RTN", "+BC", "KB", "KB+BC"])
        a.set_title(b, fontsize=8)
    ax[0].set_ylabel("rel. layer error (avg. over layers)")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "decomp.svg")
    plt.close(fig)


def fig_classes(pc):
    from data import CLASSES
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ks = ["fp32", "rtn", "opus_lw"]
    lab = {"fp32": "FP32", "rtn": "int4 RTN", "opus_lw": "Opus 5.5, layer by layer (P2)"}
    for i, k in enumerate(ks):
        ax.bar(np.arange(10) + (i - 1) * 0.26, [pc[k][c] for c in CLASSES], 0.26, color=COL[k], label=lab[k])
    ax.set_xticks(range(10), CLASSES, rotation=25)
    ax.set_ylabel("test accuracy (%)")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.tight_layout()
    fig.savefig(FIG / "classes.svg")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------- main
def main():
    res = {}
    acts_fp = forward_acts(FL, XV)
    ins_fp = layer_inputs(acts_fp, XV)
    mu = mean_inputs(acts_fp, ins_fp)
    res["fp32"] = acc_of(FL)
    res["weights"] = weight_table()
    res["input_neighbour_corr"] = neighbour_corr(ins_fp)
    res["fp32_layer"] = {n: {"out_min": float(acts_fp[n].min()), "out_max": float(acts_fp[n].max()),
                             "zero_frac": float((acts_fp[n] == 0).mean())} for n in LAYERS}

    # ---- int4 variants
    q0 = load_layers("exports/rtn_int4.npz")
    qs = {"rtn": q0, "kb": kb_all(q0, 4)}
    qs.update(llm_variants())
    Ls = {k: dequantize(q) for k, q in qs.items()}
    Ls["bc"] = bias_correct(Ls["rtn"], mu)
    Ls["kbbc"] = bias_correct(Ls["kb"], mu)
    res["int4"] = {}
    acts_q = {}
    for k, L in Ls.items():
        acts_q[k] = forward_acts(L, XV)
        d = {"acc": acc_of(L), "snr": layer_snr(L, acts_fp, ins_fp)}
        if k in qs:
            d["kernel_sum"] = {n: {"mean_abs": float(np.abs(kernel_sums(qs[k], n)).mean()),
                                   "sd": float(kernel_sums(qs[k], n).std()),
                                   "frac_gt_half": float((np.abs(kernel_sums(qs[k], n)) > 0.5).mean()),
                                   "codes_changed": int((qs[k][n]["q"] != q0[n]["q"]).sum())} for n in LAYERS}
        res["int4"][k] = d
        print(f"int4 {k:11s} test {d['acc']['test']:.2f}  snr_iso {[round(d['snr'][n]['iso'], 1) for n in LAYERS]}")
    snr = {k: res["int4"][k]["snr"] for k in Ls}

    # ---- mean shift (RTN at int4 and int2)
    shift = {"int4": mean_shift(Ls["rtn"], ins_fp, acts_fp, mu, "rtn"),
             "int2": mean_shift(dequantize(load_layers("exports/rtn_int2.npz")), ins_fp, acts_fp, mu, "rtn2")}
    shift["int8"] = mean_shift(dequantize(load_layers("exports/rtn_int8.npz")), ins_fp, acts_fp, mu, "rtn8")
    res["mean_shift"] = {b: {n: {k: v for k, v in d.items() if k not in ("pred", "meas")} for n, d in s.items()} for b, s in shift.items()}

    # BN closed form for the input means of unpooled inputs (conv2 input = a1, fc input = GAP of a4)
    from math import erf, exp, pi, sqrt
    bn_mu = {}
    for i, n in ((1, "conv2"), (4, "fc")):
        g, b = np.abs(sd[f"bn{i}.weight"].astype(np.float64)), sd[f"bn{i}.bias"].astype(np.float64)
        sgn = np.sign(sd[f"bn{i}.weight"].astype(np.float64))
        t = b / g
        Phi = 0.5 * (1 + np.vectorize(erf)(t / sqrt(2)))
        phi = np.exp(-t ** 2 / 2) / sqrt(2 * pi)
        pred = b * Phi + g * phi
        bn_mu[n] = {"r": float(np.corrcoef(pred, mu[n])[0, 1]), "ratio_median": float(np.median(mu[n] / pred))}
    res["bn_input_mean"] = bn_mu

    # ---- error decomposition and bit sweep
    res["decomp"], sweep = {}, {}
    for bits in (8, 6, 5, 4, 3, 2):
        qr = rtn_quantize(FL, bits)
        qk = kernel_balance(FL, qr, CONVS, bits)
        from reference_methods import fc_row_balance
        qk = fc_row_balance(FL, qk, bits)
        v = {"rtn": dequantize(qr), "kb": dequantize(qk)}
        v["bc"], v["kbbc"] = bias_correct(v["rtn"], mu), bias_correct(v["kb"], mu)
        sweep[bits] = {k: acc_of(L) for k, L in v.items()}
        if bits in (4, 3, 2):
            res["decomp"][f"int{bits}"] = {k: decomposition(L, ins_fp) for k, L in v.items()}
        print(f"bits {bits}: " + "  ".join(f"{k} {sweep[bits][k]['test']:.2f}" for k in v))
    res["bit_sweep"] = {str(b): {k: {"val": r["val"], "test": r["test"], "agree": r["agree_fp32"]} for k, r in d.items()}
                        for b, d in sweep.items()}

    # ---- recovery curve: exact kernel balancing applied up to layer k vs. LLM runs
    rec = {"fp32": res["fp32"]["test"]}
    curve = [res["int4"]["rtn"]["acc"]["test"]]
    for k in range(1, 5):
        curve.append(acc_of(dequantize(kernel_balance(FL, q0, CONVS[:k], 4)))["test"])
    curve.append(acc_of(dequantize(fc_row_balance_(FL, kb_all(q0, 4))))["test"])
    rec["kb_exact"] = curve
    lw = json.loads(Path("results/layerwise.json").read_text())
    for key, src in (("opus_lw", "opus"), ("haiku_lw", "haiku"), ("sonnet_lw", "sonnet")):
        rec[key] = [curve[0]] + [c["test"] for c in lw[src]["curve"]]
    res["recovery"] = rec

    # ---- int2 diagnosis
    q2 = load_layers("exports/rtn_int2.npz")
    thr = {n: 0.25 for n in q2}
    v2 = {"rtn": q2, "twn": twn_classic(FL, q2), "ternary_ls": ternary(FL, q2, thr, "ls")}
    res["int2"] = {}
    for k, q in v2.items():
        Lq = dequantize(q)
        Lb = bias_correct(Lq, mu)
        res["int2"][k] = {"acc": acc_of(Lq)["test"], "acc_bc": acc_of(Lb)["test"],
                          "zeros": {n: float((q[n]["q"] == 0).mean()) for n in LAYERS},
                          "snr_iso": layer_snr(Lq, acts_fp, ins_fp), "snr_iso_bc": layer_snr(Lb, acts_fp, ins_fp)}
        print(f"int2 {k:10s} test {res['int2'][k]['acc']:.2f} +bc {res['int2'][k]['acc_bc']:.2f}  "
              f"zeros {[round(res['int2'][k]['zeros'][n], 2) for n in LAYERS]}")

    # ---- per-class
    pc = {"fp32": res["fp32"]["per_class_test"], "rtn": res["int4"]["rtn"]["acc"]["per_class_test"],
          "opus_lw": res["int4"]["opus_lw"]["acc"]["per_class_test"], "haiku": res["int4"]["haiku"]["acc"]["per_class_test"]}

    # ---- figures
    fig_snr(snr)
    fig_weights(qs, res["weights"])
    fig_acts(acts_fp, acts_q)
    fig_shift(shift)
    fig_recovery(rec)
    fig_bits(sweep, res["fp32"]["test"])
    fig_decomp(res["decomp"])
    fig_classes(pc)
    Path("results/analysis.json").write_text(json.dumps(res, indent=1, default=float))
    print("wrote results/analysis.json and paper/figs/*.svg")


def fc_row_balance_(fl, ql):
    from reference_methods import fc_row_balance
    return fc_row_balance(fl, ql, 4)


if __name__ == "__main__":
    main()
