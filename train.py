"""Train the small FP32 CNN (8-16-32-48) on CIFAR-10 for 500 epochs.

python train.py            # -> checkpoints/small_e500.pt (best val), small_e500.npz (same weights + split, numpy)
python train.py --resume   # continue after an interruption from checkpoints/small_e500_state.pt

Augmentation: RandomCrop(32, padding=4, zero fill) + horizontal flip, done batch-wise on the device.
SGD + Nesterov, lr 0.1, wd 5e-4, OneCycleLR, batch 128, seed 0. Model selection on the 5k val split.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import MEAN, STD, load_raw
from model import CHANNELS, SmallCNN, count_params


def augment(x, gen):
    """x: raw [0,1] batch on device. Random crop with 4px zero padding + random hflip, then normalize."""
    n, dev = x.shape[0], x.device
    xp = F.pad(x, (4, 4, 4, 4))
    dx = torch.randint(0, 9, (n,), generator=gen).to(dev)
    dy = torch.randint(0, 9, (n,), generator=gen).to(dev)
    flip = (torch.rand(n, generator=gen) < 0.5).to(dev)
    ar = torch.arange(32, device=dev)
    rows = dy[:, None] + ar[None]
    cols = dx[:, None] + ar[None]
    cols = torch.where(flip[:, None], cols.flip(1), cols)
    out = xp.permute(0, 2, 3, 1)[torch.arange(n, device=dev)[:, None, None], rows[:, :, None], cols[:, None, :]]
    out = out.permute(0, 3, 1, 2)
    m = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    return (out - m) / s


@torch.no_grad()
def accuracy(model, x, y, bs=1000):
    model.eval()
    correct = sum((model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item() for i in range(0, len(x), bs))
    return correct / len(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default="checkpoints/small_e500.pt")
    p.add_argument("--channels", type=int, nargs=4, default=list(CHANNELS))
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, training on CPU (slow)")

    (x_all, y_all), (x_te, y_te) = load_raw(args.data)
    perm = torch.randperm(len(x_all), generator=torch.Generator().manual_seed(0)).numpy()  # same split as before
    val_idx, train_idx = perm[:5000], perm[5000:]
    x_tr = torch.from_numpy(x_all[train_idx]).to(device)
    y_tr = torch.from_numpy(y_all[train_idx]).to(device)
    norm = lambda a: torch.from_numpy((a - MEAN.reshape(1, 3, 1, 1)) / STD.reshape(1, 3, 1, 1)).float().to(device)
    x_val, y_val = norm(x_all[val_idx]), torch.from_numpy(y_all[val_idx]).to(device)
    steps = len(x_tr) // args.batch_size  # drop_last

    model = SmallCNN(channels=tuple(args.channels)).to(device)
    print(f"device {device}  params {count_params(model):,}  steps/epoch {steps}", flush=True)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps)
    crit = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state_path = out.with_name(out.stem + "_state.pt")
    log_path = out.with_name(out.stem + "_log.jsonl")
    start, best_val = 1, 0.0
    if args.resume and state_path.exists():
        s = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        gen.set_state(s["gen"]); torch.set_rng_state(s["torch_rng"])
        start, best_val = s["epoch"] + 1, s["best_val"]
        print(f"resumed at epoch {start}, best val {best_val:.4f}", flush=True)
    elif log_path.exists():
        log_path.unlink()

    for epoch in range(start, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = torch.zeros((), device=device)
        order = torch.randperm(len(x_tr), generator=gen).to(device)
        for i in range(steps):
            idx = order[i * args.batch_size:(i + 1) * args.batch_size]
            x, y = augment(x_tr[idx], gen), y_tr[idx]
            loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.detach()
        val_acc = accuracy(model, x_val, y_val)
        tag = ""
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_acc": val_acc, "args": vars(args)}, out)
            tag = "  *saved"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "gen": gen.get_state(), "torch_rng": torch.get_rng_state(), "epoch": epoch,
                    "best_val": best_val}, state_path)
        rec = {"epoch": epoch, "loss": loss_sum.item() / steps, "val_acc": val_acc,
               "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 2)}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"epoch {epoch:3d}/{args.epochs}  loss {rec['loss']:.4f}  val {val_acc:.4f}  "
              f"lr {rec['lr']:.4f}  {rec['sec']:.1f}s{tag}", flush=True)

    torch.save({"state_dict": model.state_dict(), "epoch": args.epochs, "args": vars(args)},
               out.with_name(out.stem + "_last.pt"))
    ckpt = torch.load(out, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_acc = accuracy(model, norm(x_te), torch.from_numpy(y_te).to(device))
    print(f"best epoch {ckpt['epoch']}  val {ckpt['val_acc']:.4f}  test {test_acc:.4f}  -> {out}", flush=True)

    # numpy export: everything downstream (PTQ, prompts, LLM-edit evaluation) runs without torch
    npz = out.with_suffix(".npz")
    np.savez(npz, val_idx=val_idx, epoch=ckpt["epoch"], val_acc_torch=ckpt["val_acc"], test_acc_torch=test_acc,
             **{f"sd.{k}": v.detach().cpu().numpy() for k, v in ckpt["state_dict"].items()})
    print(f"-> {npz}")


if __name__ == "__main__":
    main()
