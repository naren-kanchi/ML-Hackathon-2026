"""Evaluation metrics (evaluation code: takes derived labels / route as input).

Definitions (all per trip, then aggregated as stated):

top1            Observation-level Top-1 accuracy. Denominator = ALL fixes of the
                trip. A fix whose label is not among its candidates (candidate
                miss), or that has no candidates, counts as an error.
top1_clean      Same, restricted to fixes whose alignment is not flagged ambiguous.
macro_f1        Per-trip arc-class macro F1: classes = trip-local canonical arcs
                that appear in the derived labels OR the predictions of that trip;
                F1 per class over the trip's fixes; unweighted mean over classes.
                Reported as the mean over trips (arc ids are NEVER pooled across
                trips because they are trip-local).
log_loss        Mean over fixes of -log p(label), with p clipped to [1e-6, 1].
                A candidate miss has p = 0 -> contributes -log(1e-6) = 13.82.
                log_loss_covered excludes misses (report both + miss rate).
brier           Multiclass Brier over each fix's candidate set
                sum_i (p_i - y_i)^2 ; a miss contributes sum p_i^2 + 1.
miss_rate       Fraction of fixes whose derived label is not a candidate.
continuity      Fraction of consecutive decoded fix pairs (both with predictions)
                whose predicted transition is physically feasible on the DIRECTED
                network: same arc moving forward (or within back_tol), or
                to(a1)->from(a2) reachable with route distance <= v_max*dt + slack.
                Connector insertion cannot raise this number: it is computed from
                the per-fix arcs, not from the reconstructed route.
restarts        Forced restarts of the Viterbi chain (gap / disconnected), excl. trip start.
route_p/r/f1    Length-weighted overlap between the reconstructed arc route and
                the reference .route (multiset of canonical arcs, so repeated
                traversals count twice).
route_lcs       LCS(predicted, reference arc sequences) / len(reference): an
                order-aware sequence metric.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .hmm import HMMParams, NetDist


def per_fix_label_prob(cands, probs, labels):
    T = len(cands.ptr) - 1
    p = np.zeros(T)
    is_lab = cands.arc == labels[cands.obs]
    p[cands.obs[is_lab]] = probs[is_lab]
    hit = np.zeros(T, bool)
    hit[cands.obs[is_lab]] = True
    return p, hit


def brier(cands, probs, labels):
    T = len(cands.ptr) - 1
    y = (cands.arc == labels[cands.obs]).astype(float)
    sq = np.bincount(cands.obs, weights=(probs - y) ** 2, minlength=T)
    hit = np.bincount(cands.obs, weights=y, minlength=T) > 0
    return sq + (~hit)  # miss: label prob 0 -> (0-1)^2 = 1 extra


def macro_f1_trip(pred, lab):
    classes = np.union1d(np.unique(lab), np.unique(pred[pred >= 0]))
    f1s = []
    for k in classes:
        tp = np.sum((pred == k) & (lab == k))
        fp = np.sum((pred == k) & (lab != k))
        fn = np.sum((pred != k) & (lab == k))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def continuity(ctx, arcs, p: HMMParams = HMMParams(), nd: NetDist | None = None):
    rn = ctx.rn
    nd = nd or NetDist(rn)
    idx = np.nonzero(arcs >= 0)[0]
    ok = 0
    n = 0
    xy = ctx.xy
    for i0, i1 in zip(idx[:-1], idx[1:]):
        a1, a2 = int(arcs[i0]), int(arcs[i1])
        dt = ctx.t[i1] - ctx.t[i0]
        n += 1
        lim = p.v_max * dt + p.slack
        if a1 == a2:
            _, s1, _, _ = rn.project_points(xy[i0:i0 + 1], np.array([a1]))
            _, s2, _, _ = rn.project_points(xy[i1:i1 + 1], np.array([a2]))
            ok += bool(s2[0] >= s1[0] - p.back_tol)
            continue
        _, s1, _, _ = rn.project_points(xy[i0:i0 + 1], np.array([a1]))
        _, s2, _, _ = rn.project_points(xy[i1:i1 + 1], np.array([a2]))
        tail = rn.length[a1] - s1[0]
        if tail > lim:
            continue
        dist, _ = nd.from_node(int(rn.to[a1]), lim - tail)
        dd = dist.get(int(rn.fr[a2]))
        ok += bool(dd is not None and tail + dd + s2[0] <= lim)
    return ok / max(n, 1)


def _lcs(a, b):
    # O(len(a)*len(b)) DP with numpy rows; routes are <= a few thousand arcs
    if len(a) == 0 or len(b) == 0:
        return 0
    b = np.asarray(b)
    prev = np.zeros(len(b) + 1, dtype=np.int32)
    for x in a:
        eq = (b == x)
        cur = np.zeros_like(prev)
        # cur[j+1] = prev[j]+1 if eq else max(prev[j+1], cur[j])
        diag = prev[:-1] + eq
        cur[1:] = np.maximum(prev[1:], diag)
        cur = np.maximum.accumulate(cur)
        prev = cur
    return int(prev[-1])


def route_metrics(rn, pred_route, ref_route_canon):
    L = rn.length
    cp, cr = Counter(pred_route.tolist()), Counter(ref_route_canon.tolist())
    inter = sum(min(cp[k], cr[k]) * L[k] for k in cp)
    lp = sum(L[k] * v for k, v in cp.items())
    lr = sum(L[k] * v for k, v in cr.items())
    P = inter / lp if lp > 0 else 0.0
    R = inter / lr if lr > 0 else 0.0
    dedup = lambda x: [v for i, v in enumerate(x) if i == 0 or x[i - 1] != v]  # noqa: E731
    return dict(route_p=P, route_r=R, route_f1=0 if P + R == 0 else 2 * P * R / (P + R),
                route_lcs=_lcs(dedup(pred_route.tolist()), dedup(ref_route_canon.tolist())) / max(len(dedup(ref_route_canon.tolist())), 1))


def evaluate_trip(ctx, probs, dec, labels, ambiguous, route_canon=None, pred_route=None, p=HMMParams()):
    arcs = dec["arc"]
    pl, hit = per_fix_label_prob(ctx.cands, probs, labels)
    correct = arcs == labels
    out = dict(
        n_fix=len(labels), top1=correct.mean(),
        top1_clean=correct[~ambiguous].mean() if (~ambiguous).any() else np.nan,
        macro_f1=macro_f1_trip(arcs, labels),
        log_loss=float(np.mean(-np.log(np.clip(pl, 1e-6, 1)))),
        log_loss_covered=float(np.mean(-np.log(np.clip(pl[hit], 1e-6, 1)))) if hit.any() else np.nan,
        brier=float(np.mean(brier(ctx.cands, probs, labels))),
        miss_rate=float(1 - hit.mean()),
        continuity=continuity(ctx, arcs, p, dec.get("netdist")),
        restarts=sum(1 for _, r in dec["restarts"] if r != "start"),
    )
    if route_canon is not None and pred_route is not None:
        out.update(route_metrics(ctx.rn, pred_route, route_canon))
    return out
