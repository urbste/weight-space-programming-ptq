"""Architecture figure: CIFAR-10 example images -> the five layers with real channel activations of one image.
python analysis/fig_architecture.py -> paper/figs/architecture.svg
"""
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from data import CLASSES, load_raw, normalize  # noqa: E402
from ptq_eval import load_layers  # noqa: E402
from quant import conv3x3, maxpool2, relu  # noqa: E402

plt.rcParams.update({"font.size": 8, "svg.fonttype": "none", "figure.dpi": 120})
(_, _), (xt, yt) = load_raw()
fl = load_layers("exports/float32_folded.npz")

# one test image per class (first occurrence)
idx = [int(np.where(yt == c)[0][0]) for c in range(10)]
show = 7  # the cat-or-so example that is pushed through the net (index into idx)
img = xt[idx[show]]
x0 = normalize(img[None])
a1 = relu(conv3x3(x0, fl["conv1"]["weight"], fl["conv1"]["bias"]))
a2 = maxpool2(relu(conv3x3(a1, fl["conv2"]["weight"], fl["conv2"]["bias"])))
a3 = maxpool2(relu(conv3x3(a2, fl["conv3"]["weight"], fl["conv3"]["bias"])))
a4 = relu(conv3x3(a3, fl["conv4"]["weight"], fl["conv4"]["bias"]))
logits = a4.mean((2, 3)) @ fl["fc"]["weight"].T + fl["fc"]["bias"]
pred = int(logits.argmax())

fig = plt.figure(figsize=(7.4, 3.9))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 52.7); ax.axis("off")
GREY = "0.35"

# --- one test example per class, one row on top
ax.text(1, 51.3, "CIFAR-10 test images, one per class", fontsize=8, weight="bold", va="center")
for k, i in enumerate(idx):
    x = 1 + k * 9.85
    ax.imshow(np.clip(xt[i].transpose(1, 2, 0), 0, 1), extent=(x, x + 8.4, 38.5, 46.9), aspect="auto", zorder=2)
    ax.add_patch(plt.Rectangle((x, 38.5), 8.4, 8.4, fill=False, ec="0.25" if k == show else "0.8", lw=1.2 if k == show else 0.4, zorder=5))
    ax.text(x + 4.2, 37.5, CLASSES[yt[i]], ha="center", va="top", fontsize=5.6, color=GREY)

def stack(x0, y0, maps, w, h, label, sub, n=5, dx=0.8, dy=0.8):
    n = min(n, maps.shape[0])
    vmax = max(float(maps[:n].max()), 1e-6)
    for j in range(n - 1, -1, -1):
        ax.imshow(maps[j], extent=(x0 + j * dx, x0 + j * dx + w, y0 + (n - 1 - j) * dy, y0 + (n - 1 - j) * dy + h),
                  aspect="auto", cmap="magma", vmin=0, vmax=vmax, zorder=3 + (n - j))
        ax.add_patch(plt.Rectangle((x0 + j * dx, y0 + (n - 1 - j) * dy), w, h, fill=False, ec="0.25", lw=0.5, zorder=10 + (n - j)))
    cx = x0 + (w + (n - 1) * dx) / 2
    ax.text(cx, y0 - 1.0, label, ha="center", va="top", fontsize=7.5, weight="bold")
    ax.text(cx, y0 - 3.4, sub, ha="center", va="top", fontsize=6.2, color=GREY)

def arrow(xa, xb, y, txt):
    ax.annotate("", xy=(xb, y), xytext=(xa, y), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.3"), zorder=1)
    ax.text((xa + xb) / 2, y + 0.9, txt, ha="center", va="bottom", fontsize=5.6, color=GREY, linespacing=1.1)

yc = 19
xi = 1 + show * 9.85
ax.plot([xi + 4.2, xi + 4.2, 5.2], [34.0, 29.5, 29.5], color="0.3", lw=0.8, zorder=1)
ax.annotate("", xy=(5.2, yc + 4.9), xytext=(5.2, 29.5), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.3"), zorder=1)
ax.imshow(np.clip(img.transpose(1, 2, 0), 0, 1), extent=(1.0, 9.4, yc - 4.2, yc + 4.2), aspect="auto", zorder=3)
ax.add_patch(plt.Rectangle((1.0, yc - 4.2), 8.4, 8.4, fill=False, ec="0.25", lw=0.5, zorder=4))
ax.text(5.2, yc - 5.2, "input", ha="center", va="top", fontsize=7.5, weight="bold")
ax.text(5.2, yc - 7.6, "3 x 32 x 32", ha="center", va="top", fontsize=6.2, color=GREY)

arrow(10.2, 14.4, yc, "conv\nReLU")
stack(14.9, yc - 5.6, a1[0], 8.4, 8.4, "conv1", "8 ch, 32 x 32")
arrow(28.0, 33.2, yc, "conv\nReLU, pool")
stack(33.7, yc - 4.6, a2[0], 6.6, 6.6, "conv2", "16 ch, 16 x 16")
arrow(44.6, 49.8, yc, "conv\nReLU, pool")
stack(50.3, yc - 3.8, a3[0], 5.2, 5.2, "conv3", "32 ch, 8 x 8")
arrow(59.9, 64.1, yc, "conv\nReLU")
stack(64.6, yc - 3.2, a4[0], 4.2, 4.2, "conv4", "48 ch, 8 x 8")

# global average pool -> vector of 48 features
feat = a4[0].mean((1, 2))
arrow(72.8, 77.0, yc, "avg\npool")
fx = 77.6
ax.imshow(feat[:, None] / feat.max(), extent=(fx, fx + 2.4, yc - 7.2, yc + 7.2), aspect="auto", cmap="magma", vmin=0, vmax=1, zorder=3)
ax.add_patch(plt.Rectangle((fx, yc - 7.2), 2.4, 14.4, fill=False, ec="0.25", lw=0.5, zorder=4))
ax.text(fx + 1.2, yc - 8.2, "pooled", ha="center", va="top", fontsize=7.5, weight="bold")
ax.text(fx + 1.2, yc - 10.6, "48", ha="center", va="top", fontsize=6.2, color=GREY)

# fc -> 10 logits as bars
arrow(80.8, 85.6, yc, "fc\n48 to 10")
bx = 87.0
lg = logits[0]; lo, hi = lg.min(), lg.max()
for c in range(10):
    hgt = 0.35 + 11.5 * (lg[c] - lo) / (hi - lo)
    ax.add_patch(plt.Rectangle((bx + c * 1.0, yc - 6.5), 0.75, hgt, fc="#1f5fbf" if c == pred else "0.7", lw=0))
ax.text(bx + 5, yc - 7.5, "logits", ha="center", va="top", fontsize=7.5, weight="bold")
ax.text(bx + 5, yc - 9.9, f"argmax: {CLASSES[pred]}", ha="center", va="top", fontsize=6.2, color=GREY)

ax.text(50, 3.6, "3 x 3 kernels, stride 1, padding 1. BatchNorm is folded into weight and bias. Maps: the first 5 channels of each "
        "layer for the highlighted image.", ha="center", fontsize=5.8, color=GREY)
fig.savefig(ROOT / "paper/figs/architecture.svg")
print("pred", CLASSES[pred], "true", CLASSES[yt[idx[show]]])
