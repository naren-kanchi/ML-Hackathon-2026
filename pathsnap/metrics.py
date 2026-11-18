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
                whose predicted transition is feasible under EXACTLY the decoder's
                rules (hmm.pair_route_distance): same arc forward (or within
                back_tol), same arc further back via a network loop, or different
                arcs via the network; route distance must be <= v_max*dt + slack.
                Connector insertion cannot raise this number: it is computed from
                the per-fix arcs, not from the reconstructed route.
restarts        Forced restarts of the Viterbi chain (gap / disconnected), excl. trip start.
route_p/r/f1    Length-weighted overlap between the reconstructed arc route and
                the reference .route (multiset of canonical arcs, so repeated
                traversals count twice).
route_lcs       LCS(predicted, reference arc sequences) / len(reference): an
                order-aware sequence metric.
*_trim          Same, against only the stretch of the reference route between the
                first and last route positions that GPS fixes were aligned to
                (about 7-8% of route positions have no nearby fixes; they would cap
                recall below 1 even for a perfect decoder).
*_undir         Arc and its reverse twin counted as the same road.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .hmm import HMMParams, NetDist, pair_route_distance


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
    """Uses hmm.pair_route_distance, i.e. exactly the decoder's transition rules
    (same-arc forward incl. the speed limit, same-arc 'loop' via the network,
    different arcs via the network)."""
    rn = ctx.rn
    nd = nd or NetDist(rn)
    idx = np.nonzero(arcs >= 0)[0]
    if len(idx) < 2:
        return 1.0
    _, off, _, _ = rn.project_points(ctx.xy[idx], arcs[idx])
    ok = 0
    for k in range(len(idx) - 1):
        i0, i1 = idx[k], idx[k + 1]
        d, _ = pair_route_distance(rn, nd, int(arcs[i0]), float(off[k]), int(arcs[i1]), float(off[k + 1]),
                                   float(ctx.t[i1] - ctx.t[i0]), p)
        ok += bool(np.isfinite(d))
    return ok / (len(idx) - 1)


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


def _overlap(L, pred_keys, ref_keys, pred_len, ref_len):
    cp, cr = Counter(pred_keys), Counter(ref_keys)
    lp, lr = Counter(), Counter()
    for k, l in zip(pred_keys, pred_len):
        lp[k] = l
    for k, l in zip(ref_keys, ref_len):
        lr[k] = l
    inter = sum(min(cp[k], cr[k]) * lr[k] for k in cp if k in cr)
    P_den = sum(lp[k] * v for k, v in cp.items())
    R_den = sum(lr[k] * v for k, v in cr.items())
    P = inter / P_den if P_den > 0 else 0.0
    R = inter / R_den if R_den > 0 else 0.0
    return P, R, (0.0 if P + R == 0 else 2 * P * R / (P + R))


def route_metrics(rn, pred_route, ref_route_canon, covered=None):
    """Length-weighted route precision/recall/F1 and order-aware LCS.
    covered = (first, last) route positions that have GPS fixes aligned to them;
    the *_trim metrics compare against ref[first:last+1] only, so reference
    stretches with no GPS evidence do not cap recall. *_undir treats an arc and
    its reverse twin as the same road (direction-agnostic)."""
    L = rn.length
    dedup = lambda x: [v for i, v in enumerate(x) if i == 0 or x[i - 1] != v]  # noqa: E731
    und = lambda a: np.where(rn.reverse[a] >= 0, np.minimum(a, rn.reverse[a]), a)  # noqa: E731
    out = {}
    refs = {"": ref_route_canon}
    if covered is not None:
        refs["_trim"] = ref_route_canon[covered[0]:covered[1] + 1]
    for tag, ref in refs.items():
        P, R, F = _overlap(L, pred_route.tolist(), ref.tolist(), L[pred_route], L[ref])
        out.update({f"route_p{tag}": P, f"route_r{tag}": R, f"route_f1{tag}": F,
                    f"route_lcs{tag}": _lcs(dedup(pred_route.tolist()), dedup(ref.tolist())) / max(len(dedup(ref.tolist())), 1)})
        Pu, Ru, Fu = _overlap(L, und(pred_route).tolist(), und(ref).tolist(), L[pred_route], L[ref])
        out.update({f"route_p{tag}_undir": Pu, f"route_r{tag}_undir": Ru, f"route_f1{tag}_undir": Fu})
    return out


def evaluate_trip(ctx, probs, dec, labels, ambiguous, route_canon=None, pred_route=None, p=HMMParams(),
                  covered=None):
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
        out.update(route_metrics(ctx.rn, pred_route, route_canon, covered))
    return out
