"""BN folding, standard per-channel symmetric round-to-nearest PTQ, and a numpy forward pass.

A quantized network is a dict  layer -> {"q": int codes, "scale": per-row float, "bias": per-row float}.
Effective weight w = q * scale[row]. Codes are in [-qmax, qmax] with qmax = 2**(bits-1) - 1
(8 bit: 127, 4 bit: 7, 2 bit: 1, i.e. ternary).
"""
import numpy as np

LAYERS = ("conv1", "conv2", "conv3", "conv4", "fc")


def qmax_for(bits: int) -> int:
    return 2 ** (bits - 1) - 1


def load_npz(path: str):
    z = np.load(path)
    sd = {k[3:]: z[k] for k in z.files if k.startswith("sd.")}
    meta = {k: z[k] for k in z.files if not k.startswith("sd.")}
    return sd, meta


def fold_bn(sd: dict, eps: float = 1e-5) -> dict:
    """layer -> {"weight", "bias"} (float32) with eval-mode BatchNorm folded into each conv."""
    out = {}
    for i in range(1, 5):
        w = sd[f"conv{i}.weight"].astype(np.float64)
        g, b = sd[f"bn{i}.weight"].astype(np.float64), sd[f"bn{i}.bias"].astype(np.float64)
        mu, var = sd[f"bn{i}.running_mean"].astype(np.float64), sd[f"bn{i}.running_var"].astype(np.float64)
        k = g / np.sqrt(var + eps)
        out[f"conv{i}"] = {"weight": (w * k[:, None, None, None]).astype(np.float32),
                           "bias": (b - mu * k).astype(np.float32)}
    out["fc"] = {"weight": sd["fc.weight"].astype(np.float32), "bias": sd["fc.bias"].astype(np.float32)}
    return out


def _bshape(w):
    return (-1,) + (1,) * (w.ndim - 1)


def rtn_quantize(float_layers: dict, bits: int) -> dict:
    """Standard PTQ: symmetric per output channel, scale = max|w_row| / qmax, q = round(w / scale)."""
    qmax = qmax_for(bits)
    out = {}
    for name, d in float_layers.items():
        w = d["weight"]
        scale = np.maximum(np.abs(w.reshape(w.shape[0], -1)).max(1) / qmax, 1e-12).astype(np.float32)
        q = np.clip(np.rint(w / scale.reshape(_bshape(w))), -qmax, qmax).astype(np.int32)
        out[name] = {"q": q, "scale": scale, "bias": d["bias"].copy()}
    return out


def dequantize(qlayers: dict) -> dict:
    return {n: {"weight": (d["q"] * d["scale"].reshape(_bshape(d["q"]))).astype(np.float32),
                "bias": d["bias"].astype(np.float32)} for n, d in qlayers.items()}


def conv3x3(x, w, b):
    """x: N,C,H,W ; w: O,C,3,3 ; padding 1, stride 1."""
    n, c, h, wd = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = np.empty((n, c, 3, 3, h, wd), dtype=np.float32)
    for kh in range(3):
        for kw in range(3):
            cols[:, :, kh, kw] = xp[:, :, kh:kh + h, kw:kw + wd]
    y = np.einsum("ok,nkp->nop", w.reshape(w.shape[0], -1), cols.reshape(n, c * 9, h * wd), optimize=True)
    return (y + b[None, :, None]).reshape(n, w.shape[0], h, wd)


def relu(x):
    return np.maximum(x, 0)


def maxpool2(x):
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).max((3, 5))


def forward_folded(L: dict, x: np.ndarray) -> np.ndarray:
    """Forward pass of the BN-folded network (layer -> {"weight", "bias"}); returns logits."""
    x = relu(conv3x3(x, L["conv1"]["weight"], L["conv1"]["bias"]))
    x = maxpool2(relu(conv3x3(x, L["conv2"]["weight"], L["conv2"]["bias"])))
    x = maxpool2(relu(conv3x3(x, L["conv3"]["weight"], L["conv3"]["bias"])))
    x = relu(conv3x3(x, L["conv4"]["weight"], L["conv4"]["bias"]))
    return x.mean((2, 3)) @ L["fc"]["weight"].T + L["fc"]["bias"]
