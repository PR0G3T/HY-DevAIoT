"""Train a small 1D-CNN for HAR on the UCI HAR dataset (FP32 for now).

Loads the inertial signals (total_acc xyz + body_gyro xyz, 6x128 @ 50 Hz),
standardizes per channel and trains. TODO: quantize to int8 for the ESP32.
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

    torch.manual_seed(SEED)
    model = train(Net(16, 32), Xtr, ytr)
    print(f"fp32 test acc: {acc(model, Xte, yte):.4f} ({nparams(model)} params)")
