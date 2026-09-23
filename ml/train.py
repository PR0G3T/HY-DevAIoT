"""Train a small 1D-CNN for HAR on the UCI HAR dataset, then quantize to int8.

Loads the inertial signals (total_acc xyz + body_gyro xyz, 6x128 @ 50 Hz),
trains FP32, then post-training quantization: symmetric int8 weights,
int32 folded bias, int8 activations, integer-only requantization.
"""
import os
import urllib.request
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "ml", "data", "UCI HAR Dataset")
FW = os.path.join(ROOT, "firmware")
URLS = [
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
]

LABELS = ["walking", "upstairs", "downstairs", "sitting", "standing", "laying"]
C, T, K, P = 6, 128, 9, 4  # channels, window, kernel, pool
SEED = 7

torch.manual_seed(SEED)
np.random.seed(SEED)


def fetch():
    if os.path.isdir(DATA):
        return
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    z = os.path.join(os.path.dirname(DATA), "har.zip")
    for u in URLS:
        try:
            urllib.request.urlretrieve(u, z)
            break
        except Exception as e:
            print("download failed:", u, e)
    with zipfile.ZipFile(z) as f:
        names = [n for n in f.namelist() if n.endswith(".zip")]
        if names:  # UCI wraps the dataset zip inside an outer archive
            f.extract(names[0], os.path.dirname(DATA))
            z = os.path.join(os.path.dirname(DATA), names[0])
        if zipfile.is_zipfile(z):
            with zipfile.ZipFile(z) as inner:
                inner.extractall(os.path.dirname(DATA))
    os.remove(z)


def load(split):
    sig = os.path.join(DATA, split, "Inertial Signals")
    X = np.stack(
        [
            np.loadtxt(os.path.join(sig, f"{s}_{ax}_{split}.txt"))
            for s in ("total_acc", "body_gyro")
            for ax in "xyz"
        ],
        axis=1,
    ).astype(np.float32)
    y = np.loadtxt(os.path.join(DATA, split, f"y_{split}.txt")).astype(int) - 1
    return X, y  # X: [N,6,128]  (g, rad/s)


class Net(nn.Module):
    def __init__(self, w1=16, w2=32):
        super().__init__()
        self.c1 = nn.Conv1d(C, w1, K, padding=K // 2)
        self.c2 = nn.Conv1d(w1, w2, K, padding=K // 2)
        self.fc = nn.Linear(w2 * (T // P // P), 6)

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.c1(x)), P)
        x = F.max_pool1d(F.relu(self.c2(x)), P)
        return self.fc(x.flatten(1))


def train(model, Xtr, ytr, epochs=30, bs=128):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            x = torch.from_numpy(Xtr[idx])
            # augmentation: amplitude jitter, gaussian noise, circular time shift
            x = x * torch.empty(len(idx), 1, 1).uniform_(0.9, 1.1)
            x = x + torch.randn_like(x) * 0.02
            x = torch.roll(x, shifts=int(np.random.randint(-8, 9)), dims=2)
            loss = F.cross_entropy(model(x), torch.from_numpy(ytr[idx]).long())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sch.step()
    return model


def acc(model, X, y):
    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(X)).argmax(1).numpy()
    return float((p == y).mean())


# ---------------- int8 PTQ ----------------


def frexp_mult(m):
    """m > 0 real multiplier -> (M0, e) with m ~= M0 * 2^(e-31), M0 int32."""
    frac, e = np.frexp(m)  # m = frac * 2^e, frac in [0.5, 1)
    M0 = np.clip(np.round(frac * 2**31), -(2**31), 2**31 - 1).astype(np.int64)
    return M0.astype(np.int32), np.int8(e)


def quant_w(w):
    """Symmetric int8 weights."""
    s = np.abs(w).amax() / 127.0
    s = max(float(s), 1e-12)
    return np.round(w / s).clip(-127, 127).astype(np.int8), np.full(
        w.shape[0], s, np.float32
    )


def requant(acc, m0, e):
    """Integer requantization, must be bit-exact with the on-device kernel."""
    out = np.empty(acc.shape, np.int8)
    for c in range(acc.shape[0]):
        r = acc[c].astype(np.int64) * np.int64(m0[c])
        s = int(31 - e[c])
        if s > 0:
            h = np.int64(1) << np.int64(s - 1)
            y = np.where(r >= 0, (r + h) >> s, -((-r + h) >> s))
        else:
            y = r << np.int64(-s)
        out[c] = np.clip(y, -128, 127)
    return out


def conv_i8(x, w, b32, m0, e, pad=K // 2):
    """x:[Ci,T] int8, w:[Co,Ci,K] int8, b32:[Co] -> [Co,T] int8."""
    co, ci, k = w.shape
    xp = np.pad(x.astype(np.int64), ((0, 0), (pad, pad)))
    acc = np.zeros((co, x.shape[1]), np.int64)
    for j in range(k):
        acc += w[:, :, j].astype(np.int64) @ xp[:, j : j + x.shape[1]]
    acc += b32[:, None]
    return requant(acc, m0, e)


def pool_i8(x, p=P):
    c, t = x.shape
    return x.reshape(c, t // p, p).max(axis=2)


def fc_i8(x, w, b32, m0, e):
    acc = w.astype(np.int64) @ x.astype(np.int64) + b32
    return requant(acc[:, None], m0, e)[:, 0]


def quantize(model, Xcal):
    """Calibrate activation scales, quantize weights/biases -> dict of int arrays."""
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(Xcal)
        a1 = F.max_pool1d(F.relu(model.c1(xb)), P)
        a2 = F.max_pool1d(F.relu(model.c2(a1)), P)
        lg = model.fc(a2.flatten(1))
    s0 = float(np.abs(Xcal).max() / 127)
    s1 = max(float(a1.abs().max() / 127), 1e-9)
    s2 = max(float(a2.abs().max() / 127), 1e-9)
    s3 = max(float(lg.abs().max() / 127), 1e-9)

    q = {"s0": s0, "s3": s3}
    for name, (w, b), sprev, snext in [
        ("1", (model.c1.weight.detach().numpy(), model.c1.bias.detach().numpy()), s0, s1),
        ("2", (model.c2.weight.detach().numpy(), model.c2.bias.detach().numpy()), s1, s2),
        ("3", (model.fc.weight.detach().numpy(), model.fc.bias.detach().numpy()), s2, s3),
    ]:
        wq, sw = quant_w(w)
        bq = np.round(b / (sprev * sw)).astype(np.int32)
        m0, e = np.vectorize(frexp_mult, otypes=[np.int32, np.int8])(sprev * sw / snext)
        q[f"w{name}"], q[f"b{name}"], q[f"m{name}"], q[f"e{name}"] = wq, bq, m0, e
    return q


def forward_i8(x, q):
    """x:[C,T] int8 -> logits int8 [6]."""
    a = pool_i8(np.maximum(conv_i8(x, q["w1"], q["b1"], q["m1"], q["e1"]), 0))
    a = pool_i8(np.maximum(conv_i8(a, q["w2"], q["b2"], q["m2"], q["e2"]), 0))
    return fc_i8(a.flatten(), q["w3"], q["b3"], q["m3"], q["e3"])


def quant_in(X, s0, mu, sd):
    return np.round((X - mu[:, None]) / sd[:, None] / s0).clip(-128, 127).astype(np.int8)


def acc_i8(q, Xq, y):
    p = np.stack([forward_i8(x, q) for x in Xq]).argmax(1)
    return float((p == y).mean()), p


def nparams(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    fetch()
    Xtr, ytr = load("train")
    Xte, yte = load("test")
    mu = Xtr.mean(axis=(0, 2))
    sd = Xtr.std(axis=(0, 2)) + 1e-8
    Xtr = (Xtr - mu[:, None]) / sd[:, None]
    Xte = (Xte - mu[:, None]) / sd[:, None]

    print(f"{'cfg':>10} {'params':>7} {'fp32':>6} {'int8':>6} {'fp32B':>7} {'int8B':>7}")
    for w1, w2 in [(16, 32), (8, 16)]:
        torch.manual_seed(SEED)
        model = train(Net(w1, w2), Xtr, ytr)
        a32 = acc(model, Xte, yte)
        q = quantize(model, Xtr[np.random.choice(len(Xtr), 512, replace=False)])
        Xteq = quant_in(Xte, q["s0"], mu * 0, sd * 0 + 1)  # Xte already normalized
        a8, pred = acc_i8(q, Xteq, yte)
        p = nparams(model)
        print(f"{w1:>4}/{w2:<5} {p:>7} {a32:6.4f} {a8:6.4f} {4*p:>7} {p:>7}")
