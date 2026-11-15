"""Test the exact split of the output error (Eq. 2 in Sec. 3.2) on validation data, conv1-conv4, int4.

For every 3x3 patch write the input as patch mean plus deviation, x_ck = xbar_c + d_ck. The output error splits
exactly into
    dy_o = s_o sum_c xbar_c E_oc  +  s_o sum_c sum_k e_ock d_ck      (patch-mean term, detail term)
1. Per layer: mean-square of both terms (and their cross term) relative to the float pre-activation variance,
   for RTN and kernel balancing.
2. Per patch: output error vs. local contrast (RMS of d over the patch), in deciles; RTN vs. kernel balancing.
-> results/edge_split.json, paper/figs/edge_split.svg
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import CONVS, COL, FIG, FL, XV, forward_acts, kb_all, layer_inputs, patches, plt  # noqa: E402
from quant import rtn_quantize  # noqa: E402

N_IMG = 500
N_BINS = 10


def errors_in_steps(q, n):
    w = FL[n]["weight"].astype(np.float64)
    s = q[n]["scale"].astype(np.float64)
    e = q[n]["q"] - w / s[:, None, None, None]
    return e.reshape(len(s), -1), s  # (O, C*9), (O,)


def main():
    acts = forward_acts(FL, XV[:N_IMG])
    ins = layer_inputs(acts, XV[:N_IMG])
    q_rtn = rtn_quantize(FL, 4)
    q_kb = kb_all(q_rtn, 4)
    res, curves = {}, {}
    for n in CONVS:
        P = patches(ins[n], n, maxn=N_IMG)                         # (patches, C*9)
        C = P.shape[1] // 9
        P3 = P.reshape(-1, C, 9)
        xbar = P3.mean(2)                                           # (patches, C)
        D = (P3 - xbar[:, :, None]).reshape(len(P), -1)             # deviations, (patches, C*9)
        contrast = np.sqrt((D ** 2).mean(1))
        wf = FL[n]["weight"].astype(np.float64).reshape(FL[n]["weight"].shape[0], -1)
        z = P @ wf.T
        var = z.var(0)                                              # float pre-activation variance per output
        edges = np.quantile(contrast, np.linspace(0, 1, N_BINS + 1))
        bin_id = np.clip(np.searchsorted(edges, contrast, side="right") - 1, 0, N_BINS - 1)
        res[n] = {"rho_note": "see analysis.json input_neighbour_corr"}
        curves[n] = {}
        for tag, q in (("rtn", q_rtn), ("kb", q_kb)):
            e, s = errors_in_steps(q, n)
            E = e.reshape(len(s), C, 9).sum(2)                      # (O, C) kernel sums
            t1 = (xbar @ E.T) * s                                   # patch-mean term
            t2 = (D @ e.T) * s                                      # detail term
            dy = t1 + t2
            assert np.allclose(dy, (P @ e.T) * s)                   # the split is exact
            rel = lambda a: float(np.mean(a / var))                 # noqa: E731  mean over outputs of E[.]/Var(z)
            res[n][tag] = {"total": rel((dy ** 2).mean(0)), "patch_mean_term": rel((t1 ** 2).mean(0)),
                           "detail_term": rel((t2 ** 2).mean(0)), "cross": rel(2 * (t1 * t2).mean(0))}
            per_patch = ((dy ** 2) / var).mean(1)
            curves[n][tag] = [float(per_patch[bin_id == b].mean()) for b in range(N_BINS)]
        r, k = res[n]["rtn"], res[n]["kb"]
        res[n]["share_patch_mean_rtn"] = r["patch_mean_term"] / (r["patch_mean_term"] + r["detail_term"])
        res[n]["kb_over_rtn"] = {t: k[t] / r[t] for t in ("total", "patch_mean_term", "detail_term")}
        res[n]["kb_over_rtn_by_contrast_decile"] = [a / b for a, b in zip(curves[n]["kb"], curves[n]["rtn"])]
        print(n, json.dumps({t: round(v, 4) for t, v in r.items()}), "| kb/rtn",
              {t: round(v, 2) for t, v in res[n]["kb_over_rtn"].items()},
              "| by decile", np.round(res[n]["kb_over_rtn_by_contrast_decile"], 2))
    Path("results/edge_split.json").write_text(json.dumps({"layers": res, "curves": curves}, indent=1))

    fig, ax = plt.subplots(2, 1, figsize=(3.3, 3.5))
    names = list(CONVS)
    x = np.arange(len(names))
    for i, (tag, lab) in enumerate((("rtn", "RTN"), ("kb", "kernel bal."))):
        pm = [res[n][tag]["patch_mean_term"] for n in names]
        dt = [res[n][tag]["detail_term"] for n in names]
        xx = x + (i - 0.5) * 0.36
        ax[0].bar(xx, pm, 0.34, color=COL[tag], label=f"{lab}: patch-mean term")
        ax[0].bar(xx, dt, 0.34, bottom=pm, color=COL[tag], alpha=0.35, label=f"{lab}: detail term")
    ax[0].set_xticks(x, names)
    ax[0].set_ylabel("sq. error / Var(z)")
    ax[0].set_title("(a) split of the output error per layer", fontsize=8, loc="left")
    ax[0].legend(fontsize=6, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    dec = np.arange(1, N_BINS + 1)
    shades = {"conv1": "#9ad0c8", "conv2": "#4fb3a5", "conv3": "#2a9d8f", "conv4": "#1b4f49"}
    for n in names:
        ax[1].plot(dec, res[n]["kb_over_rtn_by_contrast_decile"], "o-", ms=2.5, lw=1.2, color=shades[n], label=n)
    ax[1].axhline(1, color="0.5", lw=0.6, ls=":")
    ax[1].set_ylim(0, 1.75)
    ax[1].set_xticks(dec)
    ax[1].set_xlabel("patch contrast decile (1 = smooth, 10 = edges)")
    ax[1].set_ylabel("kernel bal. / RTN")
    ax[1].set_title("(b) error ratio vs. patch contrast", fontsize=8, loc="left")
    ax[1].legend(fontsize=6, frameon=False, loc="upper left", ncol=2, columnspacing=1.0)
    fig.tight_layout(pad=0.3, h_pad=1.0)
    fig.savefig(FIG / "edge_split.svg")


if __name__ == "__main__":
    main()
