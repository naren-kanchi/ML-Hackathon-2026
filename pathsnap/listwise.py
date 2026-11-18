"""Compact listwise candidate scorer (shared by Model 1 and usable by teammates).

Each fix has up to K candidates. A shared MLP maps each candidate's feature
vector to one logit; a masked softmax over the fix's candidates gives
p(arc | fix). Loss = cross-entropy of the derived label (multiclass log loss per
fix). Fixes are excluded from the LOSS (not from evaluation) when their derived
label is not among the candidates or is flagged far from the road.

Learned:  MLP weights; feature standardisation (mean/std, fit on training rows).
Tuned:    nothing yet (hidden size, lr, epochs fixed below; BO happens later).
Fixed:    hidden=64, 2 layers, dropout=0.1, Adam lr=2e-3, wd=1e-4, 30 epochs,
          batch=1024 fixes, seed.
Calibration: one temperature fitted on out-of-fold logits (training trips only).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .candidates import MAX_K

K = MAX_K   # padding width; pad() widens automatically if a fix has more candidates


def pad(F: np.ndarray, ptr: np.ndarray, y_row: np.ndarray | None = None):
    """Flat candidate rows -> (n_fix, K, d) with mask; y_row = global row of label or -1."""
    T = len(ptr) - 1
    n = np.diff(ptr)
    width = max(K, int(n.max()) if len(n) else 0)
    X = np.zeros((T, width, F.shape[1]), np.float32)
    M = np.zeros((T, width), bool)
    fix = np.repeat(np.arange(T), n)
    slot = np.arange(len(F)) - ptr[fix]
    X[fix, slot] = F
    M[fix, slot] = True
    y = None
    if y_row is not None:
        y = np.where(y_row >= 0, y_row - ptr[:-1], -1)
    return X, M, y


def unpad(P: np.ndarray, ptr: np.ndarray):
    n = np.diff(ptr)
    fix = np.repeat(np.arange(len(n)), n)
    return P[fix, np.arange(len(fix)) - ptr[fix]]


class MLP(nn.Module):
    def __init__(self, d, h=64, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, X, M):
        z = self.net(X).squeeze(-1)
        return z.masked_fill(~M, -1e9)


class ListwiseScorer:
    def __init__(self, d, hidden=64, drop=0.1, lr=2e-3, wd=1e-4, epochs=30, batch=1024, seed=0):
        self.cfg = dict(d=d, hidden=hidden, drop=drop, lr=lr, wd=wd, epochs=epochs, batch=batch, seed=seed)
        self.mu = self.sd = None
        self.temperature = 1.0
        torch.manual_seed(seed)
        self.model = MLP(d, hidden, drop)
        self.history = []

    def _norm(self, X):
        return (X - self.mu) / self.sd

    def fit(self, X, M, y, Xv=None, Mv=None, yv=None, log=print):
        rows = X[M]
        self.mu = rows.mean(0)
        self.sd = rows.std(0) + 1e-6
        keep = y >= 0
        Xt = torch.tensor(self._norm(X[keep]) * M[keep][..., None])
        Mt, yt = torch.tensor(M[keep]), torch.tensor(y[keep])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["wd"])
        g = torch.Generator().manual_seed(self.cfg["seed"])
        lossf = nn.CrossEntropyLoss()
        t0 = time.time()
        for ep in range(self.cfg["epochs"]):
            self.model.train()
            perm = torch.randperm(len(yt), generator=g)
            tot = 0.0
            for i in range(0, len(perm), self.cfg["batch"]):
                b = perm[i:i + self.cfg["batch"]]
                loss = lossf(self.model(Xt[b], Mt[b]), yt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            rec = dict(epoch=ep + 1, train_loss=tot / len(yt))
            if Xv is not None:
                rec["val_loss"] = self.val_loss(Xv, Mv, yv)
            self.history.append(rec)
        self.train_time = time.time() - t0
        return self

    @torch.no_grad()
    def logits(self, X, M):
        self.model.eval()
        out = []
        for i in range(0, len(X), 8192):
            Xb = torch.tensor(self._norm(X[i:i + 8192]) * M[i:i + 8192][..., None])
            out.append(self.model(Xb, torch.tensor(M[i:i + 8192])).numpy())
        return np.concatenate(out) if out else np.zeros((0, X.shape[1]), np.float32)

    def proba_from_logits(self, Z, M, T=None):
        T = self.temperature if T is None else T
        Z = Z / T
        Z = np.where(M, Z, -np.inf)
        Z = Z - np.max(np.where(M, Z, -1e30), axis=1, keepdims=True)
        E = np.where(M, np.exp(Z), 0.0)
        return E / np.maximum(E.sum(1, keepdims=True), 1e-300)

    def predict_proba_padded(self, X, M):
        return self.proba_from_logits(self.logits(X, M), M)

    def val_loss(self, X, M, y):
        keep = y >= 0
        P = self.proba_from_logits(self.logits(X[keep], M[keep]), M[keep], T=1.0)
        return float(np.mean(-np.log(np.clip(P[np.arange(keep.sum()), y[keep]], 1e-12, 1))))

    def state(self):
        return dict(cfg=self.cfg, mu=self.mu, sd=self.sd, temperature=self.temperature,
                    weights={k: v.cpu().numpy() for k, v in self.model.state_dict().items()},
                    history=self.history)

    @classmethod
    def from_state(cls, s):
        c = s["cfg"]
        obj = cls(c["d"], c["hidden"], c["drop"], c["lr"], c["wd"], c["epochs"], c["batch"], c["seed"])
        obj.mu, obj.sd, obj.temperature = s["mu"], s["sd"], s["temperature"]
        obj.model.load_state_dict({k: torch.tensor(v) for k, v in s["weights"].items()})
        obj.history = s.get("history", [])
        return obj


def fit_temperature(Z, M, y):
    """Single temperature minimising NLL on held-out (OOF) logits; grid + refine."""
    keep = y >= 0
    Z, M, y = Z[keep], M[keep], y[keep]

    def nll(T):
        A = np.where(M, Z / T, -np.inf)
        mx = np.max(np.where(M, A, -1e30), 1, keepdims=True)
        lse = np.log(np.where(M, np.exp(A - mx), 0).sum(1)) + mx[:, 0]
        return float(np.mean(lse - A[np.arange(len(y)), y]))
    grid = np.exp(np.linspace(np.log(0.25), np.log(4), 41))
    vals = [nll(T) for T in grid]
    b = grid[int(np.argmin(vals))]
    fine = np.linspace(b * 0.9, b * 1.1, 21)
    vals2 = [nll(T) for T in fine]
    return float(fine[int(np.argmin(vals2))]), float(min(vals2)), nll(1.0)
