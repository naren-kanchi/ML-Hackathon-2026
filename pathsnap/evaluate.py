"""Evaluation harness shared by every model (EVALUATION code: reads labels/route).

evaluate_probs(partition, trip, probs) runs classifier-only and HMM decoding on
identical candidates and returns two metric dicts (see pathsnap/metrics.py).
"""
from __future__ import annotations

import os

import numpy as np
from joblib import Parallel, delayed

from .context import prepare
from .hmm import HMMParams, classifier_only, reconstruct_route, viterbi
from .io import load_trip
from .metrics import evaluate_trip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate_probs(part, tid, probs, hp: HMMParams = HMMParams(), ctx=None):
    ctx = ctx or prepare(load_trip(part, tid))
    L = np.load(os.path.join(ROOT, "outputs", "labels", part, f"{tid}.npz"))
    out = {}
    for mode in ("clf", "hmm"):
        dec = classifier_only(ctx, probs) if mode == "clf" else viterbi(ctx, probs, hp)
        route, breaks = reconstruct_route(ctx, dec, hp)
        m = evaluate_trip(ctx, probs, dec, L["label"], L["ambiguous"], L["route_canon"], route, hp)
        m.update(partition=part, trip=tid, mode=mode, route_breaks=breaks)
        out[mode] = m
    return out


def evaluate_many(items, hp: HMMParams = HMMParams(), n_jobs=12):
    """items: list of (part, tid, probs). Returns list of metric dicts (clf + hmm rows)."""
    res = Parallel(n_jobs=n_jobs)(delayed(evaluate_probs)(p, t, pr, hp) for p, t, pr in items)
    return [r[m] for r in res for m in ("clf", "hmm")]


def summarise(df):
    """Fix-weighted and trip-mean aggregates per mode."""
    import pandas as pd
    rows = []
    for mode, g in df.groupby("mode"):
        w = g.n_fix / g.n_fix.sum()
        r = dict(mode=mode, trips=len(g), fixes=int(g.n_fix.sum()))
        for c in ["top1", "top1_clean", "log_loss", "log_loss_covered", "brier", "miss_rate", "continuity",
                  "route_f1", "route_lcs"]:
            r[c] = float((g[c] * w).sum())
        r["macro_f1_tripmean"] = float(g.macro_f1.mean())
        r["macro_f1_tripstd"] = float(g.macro_f1.std())
        r["restarts_total"] = int(g.restarts.sum())
        rows.append(r)
    return pd.DataFrame(rows)
