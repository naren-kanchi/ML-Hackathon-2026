"""Mandatory HMM / Viterbi decoder over directed road candidates.

Hidden state at fix t: one candidate row = (directed canonical arc a, foot
offset s along a). Every state keeps its fix index, so the decoded output is one
arc per fix (observation-to-arc correspondence is preserved).

Emission:   log e_t(i) = w_e * log(max(p_i, EPS))
            p_i is the scorer's (calibrated) probability for candidate i among
            the candidates of fix t. Scorer posteriors are used directly as
            emission scores (hybrid HMM; the per-fix normaliser is constant
            across the states of that fix, so it does not change the argmax path).
Transition: route distance r between (a1,s1) at t-1 and (a2,s2) at t on the
            DIRECTED network:
              same arc and s2 >= s1 - BACK_TOL   -> r = max(s2 - s1, 0)
              otherwise                          -> r = (len a1 - s1) + D(to a1, from a2) + s2
            D = bounded Dijkstra over canonical arcs (never the reference route).
            log P = -|r - g| / beta  (Newson & Krumm 2009), g = straight-line
            distance between the two fixes.
            Infeasible if r > V_MAX * dt + SLACK (too fast) or no path within limit.
Stops:      a stationary vehicle has g ~ 0 and r ~ 0 on the same arc -> high
            probability of staying; small backward jitter is tolerated (BACK_TOL).
Gaps:       if dt > GAP_BREAK_S, or every transition into fix t is infeasible,
            the chain is cut and restarted at t from emissions only
            ("forced restart"; counted and reported, no connector inserted).
Missing:    a fix with zero candidates is skipped (prediction -1) and the next
            fix is connected to the last decoded fix with the accumulated dt.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .context import TripContext

EPS = 1e-6


@dataclass
class HMMParams:
    beta: float = 5.0
    w_emit: float = 1.0
    v_max: float = 55.0       # m/s, above max observed GPS step speed in train (see EDA)
    slack: float = 50.0       # m
    back_tol: float = 15.0    # m backward jitter allowed on same arc
    gap_break_s: float = 120.0


class NetDist:
    """Bounded single-source Dijkstra from nodes, with per-source cache."""

    MAX_CACHE = 20000   # source nodes kept; cleared when exceeded (bounded memory on long trips)

    def __init__(self, rn):
        self.rn = rn
        self.cache = {}

    def from_node(self, n: int, limit: float):
        c = self.cache.get(n)
        if c is not None and c[0] >= limit:
            return c[1], c[2]
        rn = self.rn
        dist = {n: 0.0}
        pred = {n: -1}   # predecessor arc reaching node
        h = [(0.0, n)]
        out_arcs, out_ptr, to, L = rn._out_arcs, rn._out_ptr, rn.to, rn.length
        while h:
            d, u = heapq.heappop(h)
            if d > dist.get(u, np.inf):
                continue
            if d > limit:
                break
            for a in out_arcs[out_ptr[u]:out_ptr[u + 1]]:
                v = to[a]
                nd = d + L[a]
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    pred[v] = a
                    heapq.heappush(h, (nd, v))
        if len(self.cache) >= self.MAX_CACHE:
            self.cache.clear()
        self.cache[n] = (limit, dist, pred)
        return dist, pred

    def path_arcs(self, n_from: int, n_to: int, limit: float):
        dist, pred = self.from_node(n_from, limit)
        if n_to not in dist:
            return None
        arcs = []
        v = n_to
        while v != n_from:
            a = pred[v]
            arcs.append(a)
            v = self.rn.fr[a]
        return arcs[::-1]


def pair_route_distance(rn, nd: NetDist, a1: int, s1: float, a2: int, s2: float, dt: float, p: HMMParams):
    """Route distance of ONE decoded transition (a1,s1) -> (a2,s2), using exactly the
    decoder's rules (same as _transition). Returns (distance or inf, kind) where kind is
      'same'  same arc, moving forward (or within back_tol)
      'loop'  same arc but further back: leave it and come round again via the network
      'net'   different arcs
      None    infeasible (no path within the speed limit)
    Used by reconstruct_route and metrics.continuity so all three agree."""
    limit = p.v_max * dt + p.slack
    if a1 == a2 and s2 >= s1 - p.back_tol:
        r = max(s2 - s1, 0.0)
        return (r, "same") if r <= limit else (np.inf, None)
    tail = rn.length[a1] - s1
    if tail > limit:
        return np.inf, None
    dist, _ = nd.from_node(int(rn.to[a1]), limit - tail)
    dd = dist.get(int(rn.fr[a2]))
    if dd is None or tail + dd + s2 > limit:
        return np.inf, None
    return tail + dd + s2, ("loop" if a1 == a2 else "net")


def _transition(ctx, nd: NetDist, prev_rows, cur_rows, dt, p: HMMParams):
    c, rn = ctx.cands, ctx.rn
    limit = p.v_max * dt + p.slack
    a1, s1 = c.arc[prev_rows], c.offset[prev_rows]
    a2, s2 = c.arc[cur_rows], c.offset[cur_rows]
    R = np.full((len(prev_rows), len(cur_rows)), np.inf)
    for i in range(len(prev_rows)):
        tail = rn.length[a1[i]] - s1[i]
        same = a2 == a1[i]
        fwd_same = same & (s2 >= s1[i] - p.back_tol)
        R[i, fwd_same] = np.maximum(s2[fwd_same] - s1[i], 0.0)
        rest = ~fwd_same
        if not rest.any() or tail > limit:
            continue
        dist, _ = nd.from_node(int(rn.to[a1[i]]), limit - tail)
        for j in np.nonzero(rest)[0]:
            dd = dist.get(int(rn.fr[a2[j]]))
            if dd is not None:
                R[i, j] = tail + dd + s2[j]
    R[R > limit] = np.inf
    return R


def viterbi(ctx: TripContext, probs: np.ndarray, p: HMMParams = HMMParams(), nd: NetDist | None = None):
    c = ctx.cands
    T = ctx.T
    nd = nd or NetDist(ctx.rn)
    logp = p.w_emit * np.log(np.maximum(probs, EPS))
    choice = np.full(T, -1)          # chosen candidate row per fix
    back = np.full(len(c.arc), -1)   # best previous row for each row (-1 = segment start)
    final = np.full(len(c.arc), -np.inf)
    seg_end = []                     # last fix of each chain segment
    restarts = []
    score_prev, rows_prev, t_prev = None, None, None
    for t in range(T):
        rows = np.arange(c.ptr[t], c.ptr[t + 1])
        if len(rows) == 0:
            continue
        em = logp[rows]
        reason = None
        if rows_prev is None:
            reason = "start"
        else:
            dt = ctx.t[t] - ctx.t[t_prev]
            if dt > p.gap_break_s:
                reason = "gap"
            else:
                R = _transition(ctx, nd, rows_prev, rows, dt, p)
                g = float(np.hypot(*(ctx.xy[t] - ctx.xy[t_prev])))
                tot = score_prev[:, None] - np.abs(R - g) / p.beta
                best = np.argmax(tot, axis=0)
                bs = tot[best, np.arange(len(rows))]
                if not np.isfinite(bs).any():
                    reason = "disconnected"
                else:
                    score = bs + em
                    back[rows] = np.where(np.isfinite(bs), rows_prev[best], -1)
        if reason is not None:
            if t_prev is not None:
                seg_end.append(t_prev)
            score = em.copy()
            restarts.append((t, reason))
        final[rows] = score
        score_prev, rows_prev, t_prev = score, rows, t
    if t_prev is not None:
        seg_end.append(t_prev)
    for te in seg_end:
        rows = np.arange(c.ptr[te], c.ptr[te + 1])
        cur = int(rows[np.argmax(final[rows])])
        while cur >= 0:
            choice[c.obs[cur]] = cur
            cur = back[cur]
    arc = np.where(choice >= 0, c.arc[np.maximum(choice, 0)], -1)
    return dict(row=choice, arc=arc, restarts=restarts, netdist=nd)


def reconstruct_route(ctx: TripContext, dec: dict, p: HMMParams = HMMParams()):
    """Arc sequence incl. connecting arcs between consecutive decoded fixes.
    Connectors are inserted only along the network path the decoder scored; at a
    forced restart no connector is inserted and a break marker is recorded.
    A backward move on the same arc beyond back_tol is the decoder's 'loop'
    transition (leave the arc and come round again): its connector arcs and the
    repeated arc are appended, so the loop appears in the route."""
    rn, c, nd = ctx.rn, ctx.cands, dec["netdist"]
    restart_t = {t for t, _ in dec["restarts"]}
    route, breaks = [], 0
    prev_t = prev_r = None
    for t in range(ctx.T):
        r = dec["row"][t]
        if r < 0:
            continue
        a = int(c.arc[r])
        if prev_t is None or t in restart_t:
            if route and t in restart_t:
                breaks += 1
            if not route or route[-1] != a:
                route.append(a)
        else:
            pa = int(c.arc[prev_r])
            dt = ctx.t[t] - ctx.t[prev_t]
            _, kind = pair_route_distance(rn, nd, pa, float(c.offset[prev_r]), a, float(c.offset[r]), dt, p)
            if kind != "same":
                lim = p.v_max * dt + p.slack
                conn = nd.path_arcs(int(rn.to[pa]), int(rn.fr[a]), lim) if kind is not None else None
                if conn is None:
                    breaks += 1
                else:
                    route.extend(int(x) for x in conn)
                route.append(a)
        prev_t, prev_r = t, r
    return np.array(route, dtype=np.int64), breaks


def classifier_only(ctx: TripContext, probs: np.ndarray):
    c = ctx.cands
    T = ctx.T
    arc = np.full(T, -1)
    row = np.full(T, -1)
    for t in range(T):
        s, e = c.ptr[t], c.ptr[t + 1]
        if e > s:
            r = s + int(np.argmax(probs[s:e]))
            row[t], arc[t] = r, c.arc[r]
    return dict(row=row, arc=arc, restarts=[], netdist=NetDist(ctx.rn))
