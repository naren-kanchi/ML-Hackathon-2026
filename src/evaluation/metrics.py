"""
PathSnap - evaluation metrics for map matching.

Labels (true_arc, route_pos, reference route) are used ONLY here, never in decoding.

Pointwise   top-1 accuracy on labelled points (a point with no prediction counts as wrong),
            candidate miss rate (labelled points whose true arc is not a candidate at all).
Continuity  share of consecutive predicted arcs (same segment, both predicted) that are
            the same arc or directly connected (v of A == u of B).
Sequence    arc-level precision / recall / F1 and length-weighted precision / recall of the
            reconstructed route vs the reference route, restricted to the section covered by
            labelled points (route_pos min..max). The predicted route is built from the points
            between the first and last labelled point, so both sides cover the same stretch.
            Reported directed (exact arc) and undirected (arc or its reverse).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pointwise(pred: np.ndarray, true_arc: np.ndarray, emission_df: pd.DataFrame) -> dict:
    lab = true_arc >= 0
    n_lab = int(lab.sum())
    cand = emission_df[["pt", "arc_id"]]
    pts_lab = np.flatnonzero(lab)
    has_true = cand[cand.arc_id.values == true_arc[cand.pt.values]].pt.unique()
    return {
        "n_points": len(pred),
        "n_labelled": n_lab,
        "pct_predicted": float((pred >= 0).mean() * 100),
        "top1_acc": float((pred[lab] == true_arc[lab]).mean() * 100) if n_lab else np.nan,
        "candidate_miss_rate": float((~np.isin(pts_lab, has_true)).mean() * 100) if n_lab else np.nan,
    }


def continuity(pred: np.ndarray, segment: np.ndarray, arcs: pd.DataFrame) -> dict:
    ok = (pred[:-1] >= 0) & (pred[1:] >= 0) & (segment[:-1] == segment[1:])
    a, b = pred[:-1][ok], pred[1:][ok]
    if not len(a):
        return {"continuity_pct": np.nan, "n_pairs": 0}
    same = a == b
    connected = arcs.v.loc[a].values == arcs.u.loc[b].values
    return {"continuity_pct": float((same | connected).mean() * 100), "n_pairs": int(len(a))}


def _undirected(ids: np.ndarray, arcs: pd.DataFrame) -> np.ndarray:
    u, v = arcs.u.loc[ids].values, arcs.v.loc[ids].values
    return np.minimum(u, v) * (int(arcs.v.max()) + 1) + np.maximum(u, v)


def _prf(pred_keys, ref_keys, pred_len, ref_len) -> dict:
    """Set-based P/R/F1 plus length-weighted P/R. *_len: key -> length (m)."""
    P, R = set(pred_keys), set(ref_keys)
    tp = P & R
    prec = len(tp) / len(P) if P else np.nan
    rec = len(tp) / len(R) if R else np.nan
    if P and R:
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    else:
        f1 = np.nan
    lp = sum(pred_len[k] for k in tp) / sum(pred_len[k] for k in P) if P else np.nan
    lr = sum(ref_len[k] for k in tp) / sum(ref_len[k] for k in R) if R else np.nan
    return {"precision": prec * 100, "recall": rec * 100, "f1": f1 * 100,
            "len_precision": lp * 100, "len_recall": lr * 100}


def labelled_window(true_arc: np.ndarray, route_pos: np.ndarray):
    """(first labelled point, last labelled point, min route_pos, max route_pos) or None."""
    lab = np.flatnonzero(true_arc >= 0)
    if not len(lab):
        return None
    return int(lab[0]), int(lab[-1]), int(route_pos[lab].min()), int(route_pos[lab].max())


def sequence(pred_route: np.ndarray, ref_route: np.ndarray, arcs: pd.DataFrame) -> dict:
    """Compare a predicted arc sequence with the (already windowed) reference route."""
    out = {}
    length = arcs.length_m
    pr, rr = np.asarray(pred_route, np.int64), np.asarray(ref_route, np.int64)
    out.update({f"dir_{k}": v for k, v in _prf(
        pr, rr, dict(zip(pr, length.loc[pr].values)), dict(zip(rr, length.loc[rr].values))).items()})
    pu, ru = _undirected(pr, arcs) if len(pr) else pr, _undirected(rr, arcs) if len(rr) else rr
    out.update({f"undir_{k}": v for k, v in _prf(
        pu, ru, dict(zip(pu, length.loc[pr].values)), dict(zip(ru, length.loc[rr].values))).items()})
    out["pred_route_arcs"], out["ref_route_arcs"] = len(pr), len(rr)
    return out
