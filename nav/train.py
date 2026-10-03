"""Behaviour cloning: numpy MLP (features -> v, w), Adam, MSE on standardised targets."""
from pathlib import Path

import numpy as np

from . import dataset, features


def init(sizes, rng):
    return [(rng.normal(0, np.sqrt(1 / a), (a, b)), np.zeros(b)) for a, b in zip(sizes[:-1], sizes[1:])]


def forward(params, x):
    acts = [x]
    for i, (W, b) in enumerate(params):
        x = x @ W + b
        if i < len(params) - 1:
            x = np.tanh(x)
        acts.append(x)
    return acts


def train(data_dirs, out, epochs=60, hidden=64, seed=0, lr=1e-3, batch=256):
    X, Y = dataset.load(data_dirs)
    rng = np.random.default_rng(seed)
    xm, xs = X.mean(0), X.std(0) + 1e-6
    ym, ys = Y.mean(0), Y.std(0) + 1e-6
    Xn, Yn = (X - xm) / xs, (Y - ym) / ys
    idx = rng.permutation(len(X))
    n_val = len(X) // 10
    val, tr = idx[:n_val], idx[n_val:]
    params = init([X.shape[1], hidden, hidden, Y.shape[1]], rng)
    m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
    v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
    t = 0
    for ep in range(epochs):
        rng.shuffle(tr)
        for k in range(0, len(tr), batch):
            bi = tr[k:k + batch]
            acts = forward(params, Xn[bi])
            g = 2 * (acts[-1] - Yn[bi]) / len(bi)
            t += 1
            for i in reversed(range(len(params))):
                W, b = params[i]
                gW, gb = acts[i].T @ g, g.sum(0)
                if i:
                    g = (g @ W.T) * (1 - acts[i] ** 2)
                for j, (p, gp) in enumerate(((W, gW), (b, gb))):  # Adam
                    m[i][j][...] = 0.9 * m[i][j] + 0.1 * gp
                    v[i][j][...] = 0.999 * v[i][j] + 0.001 * gp ** 2
                    p -= lr * (m[i][j] / (1 - 0.9 ** t)) / (np.sqrt(v[i][j] / (1 - 0.999 ** t)) + 1e-8)
        if ep % 10 == 0 or ep == epochs - 1:
            mse = lambda s: float(np.mean((forward(params, Xn[s])[-1] - Yn[s]) ** 2))
            print(f"epoch {ep:3d}  train {mse(tr):.4f}  val {mse(val):.4f}  (standardised MSE)")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    flat = {f"W{i}": W for i, (W, _) in enumerate(params)} | {f"b{i}": b for i, (_, b) in enumerate(params)}
    np.savez(out, xm=xm, xs=xs, ym=ym, ys=ys, n_layers=len(params), feature_names=features.names(), **flat)
    print(f"saved {out}  ({len(X)} samples from {len(data_dirs)} dir(s))")
