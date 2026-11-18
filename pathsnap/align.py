"""Shared label alignment: ordered reference route -> one derived arc label per GPS fix.

TRAINING-LABEL / EVALUATION CODE ONLY (reads the route).

Method (monotone DTW-style dynamic programme)
--------------------------------------------
Route positions k = 0..K-1 (repeats kept as distinct positions, so loops and
repeated traversals keep their order). Each fix t is assigned a position k_t with
k_0 <= k_1 <= ... (the vehicle never goes back along the route list).
Cost of assigning fix t to position k:
    c(t,k) = min(d(t,k), D_CAP)^2 / (2 * SIGMA^2)
where d is point-to-segment distance in metres. We minimise sum_t c(t, k_t)
subject to monotonicity (prefix-minimum recursion, O(T*K)).
Skipping route arcs between fixes is free: short arcs are often passed between
two 1 Hz fixes.

The label is a DERIVED quantity. We flag fixes where it is uncertain:
  far        : d(t,k_t) > FAR_M  (GPS far from its own reference arc)
  boundary   : fix projects to an arc end (f<0.05 or >0.95) and the neighbouring
               route arc is within BOUNDARY_M of the assigned distance
  after_gap  : time since previous fix > GAP_S
  chain_break: assigned arc is adjacent (in route order) to a break where
               to(route[k]) != from(route[k+1])
"""
from __future__ import annotations

import numpy as np

from .roadnet import RoadNet

SIGMA = 10.0
D_CAP = 200.0
FAR_M = 30.0
BOUNDARY_M = 3.0
GAP_S = 5.0


def align_route(rn: RoadNet, xy: np.ndarray, t: np.ndarray, route: np.ndarray, chunk: int = 2048):
    route_c = rn.canon[route]
    T, K = len(xy), len(route_c)
    cost_prev = None
    back = np.empty((T, K), dtype=np.int32)
    dist_all = np.empty((T, K), dtype=np.float32)
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        P = xy[s:e, None, :]
        a = rn.p0[route_c][None]
        d = (rn.p1[route_c] - rn.p0[route_c])[None]
        L2 = np.maximum((d * d).sum(-1), 1e-9)
        f = np.clip(((P - a) * d).sum(-1) / L2, 0, 1)
        foot = a + f[..., None] * d
        dist_all[s:e] = np.hypot(*(P - foot).transpose(2, 0, 1))
    c = np.minimum(dist_all, D_CAP).astype(np.float64) ** 2 / (2 * SIGMA ** 2)
    cost_prev = c[0].copy()
    back[0] = np.arange(K)
    for i in range(1, T):
        # prefix min of previous row
        pm = np.minimum.accumulate(cost_prev)
        # argmin of prefix: index where prefix min last changed
        is_new = np.concatenate([[True], cost_prev[1:] < pm[:-1]])
        idx = np.maximum.accumulate(np.where(is_new, np.arange(K), 0))
        back[i] = idx
        cost_prev = pm + c[i]
    k = np.empty(T, dtype=np.int64)
    k[-1] = int(np.argmin(cost_prev))
    for i in range(T - 1, 0, -1):
        k[i - 1] = back[i, k[i]]
    lab = route_c[k]
    d_ass = dist_all[np.arange(T), k].astype(np.float64)
    _, _, frac, _ = rn.project_points(xy, lab)
    # flags
    far = d_ass > FAR_M
    kp = np.clip(k - 1, 0, K - 1)
    kn = np.clip(k + 1, 0, K - 1)
    d_prev = dist_all[np.arange(T), kp]
    d_next = dist_all[np.arange(T), kn]
    boundary = ((frac < 0.05) & (k > 0) & (d_prev - d_ass < BOUNDARY_M)) | \
               ((frac > 0.95) & (k < K - 1) & (d_next - d_ass < BOUNDARY_M))
    dt = np.diff(t, prepend=t[0])
    after_gap = dt > GAP_S
    breaks = np.zeros(K, bool)
    if K > 1:
        br = rn.to[route_c[:-1]] != rn.fr[route_c[1:]]
        breaks[:-1] |= br
        breaks[1:] |= br
    chain_break = breaks[k]
    ambiguous = far | boundary | after_gap | chain_break
    return dict(label=lab, route_pos=k, dist=d_ass, frac=frac, far=far, boundary=boundary,
                after_gap=after_gap, chain_break=chain_break, ambiguous=ambiguous,
                route_canon=route_c)
